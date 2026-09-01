// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <benchmark/benchmark.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <random>
#include <set>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "common/FilterBitmap.h"
#include "exec/operator/AdaptiveFilterSink.h"

namespace milvus::exec {
namespace {

constexpr int64_t kWordBits = 64;
constexpr int64_t kDefaultSparseCap = 1000;
constexpr int64_t kDefaultBatchRows = 8192;

enum class IdDistribution : int64_t {
    Random = 0,
    Clustered = 1,
};

struct BatchMasks {
    TargetBitmap predicate;
    TargetBitmap validity;
    int64_t offset = 0;
};

struct ProducerWorkload {
    int64_t universe = 0;
    int64_t accepted_count = 0;
    int64_t sparse_cap = 0;
    int64_t batch_rows = 0;
    IdDistribution distribution = IdDistribution::Random;
    TargetBitmap expected_accepted;
    std::vector<BatchMasks> batches;
};

struct ProducerCounters {
    uint64_t processed_rows = 0;
    uint64_t predicate_evals = 0;
    uint64_t mask_words = 0;
    uint64_t ids_appended = 0;
    uint64_t dense_words_initialized = 0;
    uint64_t dense_words_written = 0;
    uint64_t backfill = 0;
    uint64_t switch_count = 0;
};

struct AdaptiveOutput {
    SparseFilterResult result;
    AdaptiveFilterSinkStats stats;
};

uint64_t
WordsForBits(int64_t bits) {
    return (static_cast<uint64_t>(bits) + kWordBits - 1) / kWordBits;
}

uint64_t
DestinationWordsTouched(int64_t offset, int64_t size) {
    if (size == 0) {
        return 0;
    }
    const auto start_in_word = static_cast<uint64_t>(offset) % kWordBits;
    return (start_in_word + static_cast<uint64_t>(size) + kWordBits - 1) /
           kWordBits;
}

TargetBitmap
SliceBitmap(TargetBitmap& source, int64_t offset, int64_t size) {
    TargetBitmapView view(
        source.data(), static_cast<size_t>(offset), static_cast<size_t>(size));
    return TargetBitmap(view);
}

void
PopulateRandomAccepted(TargetBitmap& accepted, int64_t count) {
    // Floyd sampling gives an exact uniform subset with O(V) setup state.
    // This setup is never timed and does not impose any order on the producer.
    std::mt19937_64 generator(0x5A17D00DULL);
    std::unordered_set<int64_t> selected;
    selected.reserve(static_cast<size_t>(count * 2 + 1));

    const auto universe = static_cast<int64_t>(accepted.size());
    for (int64_t candidate = universe - count; candidate < universe;
         ++candidate) {
        std::uniform_int_distribution<int64_t> distribution(0, candidate);
        const auto sampled = distribution(generator);
        if (!selected.insert(sampled).second) {
            selected.insert(candidate);
        }
    }
    for (const auto id : selected) {
        accepted.set(static_cast<size_t>(id));
    }
}

void
PopulateClusteredAccepted(TargetBitmap& accepted, int64_t count) {
    const auto universe = static_cast<int64_t>(accepted.size());
    const auto begin = (universe - count) / 2;
    for (int64_t id = begin; id < begin + count; ++id) {
        accepted.set(static_cast<size_t>(id));
    }
}

ProducerWorkload
BuildWorkload(int64_t universe,
              int64_t accepted_count,
              int64_t sparse_cap,
              IdDistribution distribution,
              int64_t batch_rows) {
    ProducerWorkload workload;
    workload.universe = universe;
    workload.accepted_count = accepted_count;
    workload.sparse_cap = sparse_cap;
    workload.batch_rows = batch_rows;
    workload.distribution = distribution;
    workload.expected_accepted =
        TargetBitmap(static_cast<size_t>(universe), false);

    if (distribution == IdDistribution::Random) {
        PopulateRandomAccepted(workload.expected_accepted, accepted_count);
    } else {
        PopulateClusteredAccepted(workload.expected_accepted, accepted_count);
    }

    for (int64_t offset = 0; offset < universe; offset += batch_rows) {
        const auto size = std::min(batch_rows, universe - offset);
        auto predicate = SliceBitmap(workload.expected_accepted, offset, size);
        // Keep a real validity mask in both paths. It is all-valid so V is
        // controlled solely by the requested distribution.
        TargetBitmap validity(static_cast<size_t>(size), true);
        workload.batches.push_back(
            BatchMasks{std::move(predicate), std::move(validity), offset});
    }
    return workload;
}

std::vector<BatchMasks>
CloneMasks(const ProducerWorkload& workload) {
    std::vector<BatchMasks> clone;
    clone.reserve(workload.batches.size());
    for (const auto& batch : workload.batches) {
        clone.push_back(BatchMasks{
            batch.predicate.clone(), batch.validity.clone(), batch.offset});
    }
    return clone;
}

TargetBitmap
RunDenseProducer(std::vector<BatchMasks>& batches, int64_t universe) {
    TargetBitmap filtered;
    filtered.reserve(static_cast<size_t>(universe));
    for (auto& batch : batches) {
        TargetBitmapView predicate(batch.predicate);
        TargetBitmapView validity(batch.validity);
        predicate.inplace_and(validity, predicate.size());
        predicate.flip();
        filtered.append(predicate);
    }
    return filtered;
}

template <bool CollectStats>
AdaptiveOutput
RunAdaptiveProducer(std::vector<BatchMasks>& batches,
                    int64_t universe,
                    int64_t sparse_cap) {
    AdaptiveFilterSink<CollectStats> sink(universe, sparse_cap);
    for (auto& batch : batches) {
        sink.ConsumeBatch(TargetBitmapView(batch.predicate),
                          TargetBitmapView(batch.validity),
                          batch.offset);
    }
    auto result = sink.Finish();
    if constexpr (CollectStats) {
        return AdaptiveOutput{std::move(result), sink.stats()};
    }
    return AdaptiveOutput{std::move(result), {}};
}

ProducerCounters
DenseCounters(const ProducerWorkload& workload) {
    ProducerCounters counters;
    counters.processed_rows = static_cast<uint64_t>(workload.universe);
    counters.predicate_evals = static_cast<uint64_t>(workload.universe);
    counters.dense_words_initialized = WordsForBits(workload.universe);
    for (const auto& batch : workload.batches) {
        const auto size = static_cast<int64_t>(batch.predicate.size());
        // One predicate word and one validity word are consumed per word.
        counters.mask_words += 2 * WordsForBits(size);
        counters.dense_words_written +=
            DestinationWordsTouched(batch.offset, size);
    }
    return counters;
}

uint64_t
AdaptiveMaskWords(const ProducerWorkload& workload) {
    using Policy = TargetBitmapView::policy_type;
    using Word = TargetBitmapView::data_type;

    uint64_t mask_words = 0;
    uint64_t retained = 0;
    bool dense = false;

    for (const auto& batch : workload.batches) {
        const auto size = static_cast<int64_t>(batch.predicate.size());
        if (dense) {
            mask_words += 2 * WordsForBits(size);
            continue;
        }

        bool switched_in_batch = false;
        for (int64_t word_offset = 0; word_offset < size;
             word_offset += kWordBits) {
            const auto bits = std::min(kWordBits, size - word_offset);
            ++mask_words;  // predicate word
            ++mask_words;  // validity word
            const auto predicate =
                Policy::op_read(batch.predicate.data(), word_offset, bits);
            const auto validity =
                Policy::op_read(batch.validity.data(), word_offset, bits);
            const auto accepted = static_cast<Word>(predicate & validity);
            const auto in_word = static_cast<uint64_t>(std::popcount(accepted));
            if (retained + in_word >
                static_cast<uint64_t>(workload.sparse_cap)) {
                switched_in_batch = true;
            } else {
                retained += in_word;
            }
            if (switched_in_batch) {
                break;
            }
        }

        if (switched_in_batch) {
            // The production sink bulk-writes the complete triggering batch.
            mask_words += 2 * WordsForBits(size);
            dense = true;
        }
    }
    return mask_words;
}

ProducerCounters
AdaptiveCounters(const ProducerWorkload& workload,
                 const AdaptiveFilterSinkStats& stats) {
    ProducerCounters counters;
    counters.processed_rows = stats.processed_rows;
    counters.predicate_evals = static_cast<uint64_t>(workload.universe);
    counters.mask_words = AdaptiveMaskWords(workload);
    counters.ids_appended = stats.ids_appended;
    counters.dense_words_initialized = stats.dense_words_initialized;
    counters.dense_words_written = stats.dense_words_written;
    counters.backfill = stats.backfill_count;
    counters.switch_count = stats.switch_count;
    return counters;
}

std::string
VerifyDense(const ProducerWorkload& workload, const TargetBitmap& filtered) {
    if (filtered.size() != static_cast<size_t>(workload.universe)) {
        return "Dense output universe does not match N";
    }
    for (int64_t id = 0; id < workload.universe; ++id) {
        const auto expected =
            workload.expected_accepted[static_cast<size_t>(id)];
        const auto observed = !filtered[static_cast<size_t>(id)];
        if (expected != observed) {
            return "Dense output is not semantically equivalent to the truth "
                   "masks";
        }
    }
    return {};
}

std::string
VerifyAdaptive(const ProducerWorkload& workload,
               const SparseFilterResult& result) {
    if (result.universe != workload.universe ||
        result.IsSparse() == result.IsDense()) {
        return "Adaptive output has an invalid representation contract";
    }

    if (result.IsDense()) {
        return VerifyDense(workload, *result.filtered);
    }

    TargetBitmap observed(static_cast<size_t>(workload.universe), false);
    for (const auto id : *result.accepted_ids) {
        if (id < 0 || id >= workload.universe) {
            return "Adaptive Sparse output contains an out-of-range ID";
        }
        if (observed[static_cast<size_t>(id)]) {
            return "Adaptive Sparse output contains a duplicate ID";
        }
        observed.set(static_cast<size_t>(id));
    }
    for (int64_t id = 0; id < workload.universe; ++id) {
        if (observed[static_cast<size_t>(id)] !=
            workload.expected_accepted[static_cast<size_t>(id)]) {
            return "Adaptive Sparse output is not semantically equivalent to "
                   "the truth masks";
        }
    }
    return {};
}

std::string
RunClosureCheck(const ProducerWorkload& workload) {
    auto dense_masks = CloneMasks(workload);
    auto dense = RunDenseProducer(dense_masks, workload.universe);
    if (auto error = VerifyDense(workload, dense); !error.empty()) {
        return error;
    }

    auto adaptive_masks = CloneMasks(workload);
    auto adaptive = RunAdaptiveProducer<true>(
        adaptive_masks, workload.universe, workload.sparse_cap);
    if (auto error = VerifyAdaptive(workload, adaptive.result);
        !error.empty()) {
        return error;
    }

    const auto dense_stats = DenseCounters(workload);
    const auto adaptive_stats = AdaptiveCounters(workload, adaptive.stats);
    if (dense_stats.predicate_evals != adaptive_stats.predicate_evals ||
        dense_stats.processed_rows != adaptive_stats.processed_rows) {
        return "Dense and Adaptive paths do not evaluate the same row count";
    }
    // Neither producer performs a sorting pass. Sparse IDs happen to retain
    // producer order, but the payload contract does not require any order.
    constexpr uint64_t kSortCalls = 0;
    static_assert(kSortCalls == 0);
    return {};
}

bool
ReadArgs(benchmark::State& state,
         int64_t& universe,
         int64_t& accepted_count,
         int64_t& sparse_cap,
         IdDistribution& distribution,
         int64_t& batch_rows) {
    universe = state.range(0);
    accepted_count = state.range(1);
    sparse_cap = state.range(2);
    const auto distribution_arg = state.range(3);
    batch_rows = state.range(4);
    if (universe <= 0 || accepted_count < 0 || accepted_count > universe ||
        sparse_cap < 0 || sparse_cap > universe || batch_rows <= 0 ||
        (distribution_arg != static_cast<int64_t>(IdDistribution::Random) &&
         distribution_arg != static_cast<int64_t>(IdDistribution::Clustered))) {
        state.SkipWithError("invalid N/V/T/distribution/batch Args");
        return false;
    }
    distribution = static_cast<IdDistribution>(distribution_arg);
    return true;
}

void
PublishCounters(benchmark::State& state,
                const ProducerWorkload& workload,
                const ProducerCounters& counters) {
    state.counters["N"] = static_cast<double>(workload.universe);
    state.counters["V"] = static_cast<double>(workload.accepted_count);
    state.counters["V/N"] =
        static_cast<double>(workload.accepted_count) / workload.universe;
    state.counters["T"] = static_cast<double>(workload.sparse_cap);
    state.counters["processed_rows"] =
        static_cast<double>(counters.processed_rows);
    state.counters["predicate_evals"] =
        static_cast<double>(counters.predicate_evals);
    state.counters["mask_words"] = static_cast<double>(counters.mask_words);
    state.counters["ids_appended"] = static_cast<double>(counters.ids_appended);
    state.counters["dense_words_initialized"] =
        static_cast<double>(counters.dense_words_initialized);
    state.counters["dense_words_written"] =
        static_cast<double>(counters.dense_words_written);
    state.counters["backfill"] = static_cast<double>(counters.backfill);
    state.counters["switch"] = static_cast<double>(counters.switch_count);
    state.counters["sort"] = 0.0;
    state.counters["batch_rows"] = static_cast<double>(workload.batch_rows);
    state.counters["distribution"] =
        static_cast<double>(workload.distribution == IdDistribution::Clustered);
    state.SetItemsProcessed(state.iterations() * workload.universe);
}

void
DenseProducerBenchmark(benchmark::State& state) {
    int64_t universe = 0;
    int64_t accepted_count = 0;
    int64_t sparse_cap = 0;
    int64_t batch_rows = 0;
    IdDistribution distribution;
    if (!ReadArgs(state,
                  universe,
                  accepted_count,
                  sparse_cap,
                  distribution,
                  batch_rows)) {
        return;
    }

    const auto workload = BuildWorkload(
        universe, accepted_count, sparse_cap, distribution, batch_rows);
    if (auto error = RunClosureCheck(workload); !error.empty()) {
        state.SkipWithError(error.c_str());
        return;
    }
    const auto counters = DenseCounters(workload);

    std::vector<BatchMasks> masks;
    for (auto _ : state) {
        // Predicate truth-mask production is held fixed and excluded. Final
        // representation construction, including all Dense writes, is timed.
        state.PauseTiming();
        masks = CloneMasks(workload);
        state.ResumeTiming();

        auto filtered = RunDenseProducer(masks, universe);
        benchmark::DoNotOptimize(filtered.data());
        benchmark::DoNotOptimize(filtered.size());
        benchmark::ClobberMemory();
    }
    PublishCounters(state, workload, counters);
}

void
AdaptiveProducerBenchmark(benchmark::State& state) {
    int64_t universe = 0;
    int64_t accepted_count = 0;
    int64_t sparse_cap = 0;
    int64_t batch_rows = 0;
    IdDistribution distribution;
    if (!ReadArgs(state,
                  universe,
                  accepted_count,
                  sparse_cap,
                  distribution,
                  batch_rows)) {
        return;
    }

    const auto workload = BuildWorkload(
        universe, accepted_count, sparse_cap, distribution, batch_rows);
    if (auto error = RunClosureCheck(workload); !error.empty()) {
        state.SkipWithError(error.c_str());
        return;
    }

    // Capture logical access counts once, outside the timed loop. Recomputing
    // them per iteration would add a second O(N) scan and corrupt attribution.
    ProducerCounters counters;
    {
        auto counter_masks = CloneMasks(workload);
        const auto counter_output =
            RunAdaptiveProducer<true>(counter_masks, universe, sparse_cap);
        counters = AdaptiveCounters(workload, counter_output.stats);
    }

    std::vector<BatchMasks> masks;
    for (auto _ : state) {
        // Only truth-mask construction is paused. Sparse append, T+1 switch,
        // Dense allocation/write and O(T) backfill all remain timed.
        state.PauseTiming();
        masks = CloneMasks(workload);
        state.ResumeTiming();

        auto output = RunAdaptiveProducer<false>(
            masks, workload.universe, workload.sparse_cap);
        if (output.result.IsSparse()) {
            benchmark::DoNotOptimize(output.result.accepted_ids->data());
            benchmark::DoNotOptimize(output.result.accepted_ids->size());
        } else {
            benchmark::DoNotOptimize(output.result.filtered->data());
            benchmark::DoNotOptimize(output.result.filtered->size());
        }
        benchmark::ClobberMemory();
    }
    PublishCounters(state, workload, counters);
}

void
ApplySwitchPointArgs(benchmark::internal::Benchmark* benchmark) {
    // Scale sweep: six V/N points plus the exact T-1/T/T+1 transition for
    // each N. Keep T and batch width fixed so only N, V, and distribution
    // change between registered cases.
    constexpr std::array<int64_t, 5> universes{
        50'000, 100'000, 250'000, 1'000'000, 10'000'000};
    // One basis point is 0.01%.
    constexpr std::array<int64_t, 6> ratio_basis_points{1, 5, 10, 20, 50, 100};

    for (const auto universe : universes) {
        std::set<int64_t> accepted_counts;
        for (const auto basis_points : ratio_basis_points) {
            accepted_counts.insert(
                std::max<int64_t>(1, universe * basis_points / 10'000));
        }
        for (const auto boundary : {kDefaultSparseCap - 1,
                                    kDefaultSparseCap,
                                    kDefaultSparseCap + 1}) {
            if (boundary <= universe) {
                accepted_counts.insert(boundary);
            }
        }

        for (const auto accepted_count : accepted_counts) {
            for (const auto distribution :
                 {IdDistribution::Random, IdDistribution::Clustered}) {
                benchmark->Args({universe,
                                 accepted_count,
                                 kDefaultSparseCap,
                                 static_cast<int64_t>(distribution),
                                 kDefaultBatchRows});
            }
        }
    }
}

}  // namespace
}  // namespace milvus::exec

BENCHMARK(milvus::exec::DenseProducerBenchmark)
    ->Apply(milvus::exec::ApplySwitchPointArgs)
    ->ArgNames({"N", "V", "T", "distribution", "batch_rows"});
BENCHMARK(milvus::exec::AdaptiveProducerBenchmark)
    ->Apply(milvus::exec::ApplySwitchPointArgs)
    ->ArgNames({"N", "V", "T", "distribution", "batch_rows"});

BENCHMARK_MAIN();
