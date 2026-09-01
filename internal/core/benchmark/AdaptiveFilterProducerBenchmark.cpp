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
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fmt/core.h>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
#include <unistd.h>

#include "cachinglayer/Manager.h"
#include "common/Common.h"
#include "common/Schema.h"
#include "common/common_type_c.h"
#include "exec/QueryContext.h"
#include "exec/expression/Expr.h"
#include "exec/expression/ExprCache.h"
#include "exec/operator/AdaptiveFilterSink.h"
#include "expr/ITypeExpr.h"
#include "folly/executors/CPUThreadPoolExecutor.h"
#include "folly/init/Init.h"
#include "index/BitmapIndex.h"
#include "index/ScalarIndexSort.h"
#include "milvus-storage/filesystem/fs.h"
#include "pb/schema.pb.h"
#include "segcore/arrow_fs_c.h"
#include "storage/LocalChunkManagerSingleton.h"
#include "storage/MmapManager.h"
#include "storage/RemoteChunkManagerSingleton.h"
#include "test_utils/DataGen.h"
#include "test_utils/storage_test_utils.h"

// storage_test_utils.h deliberately shares these paths with the unit-test
// runner.  This benchmark has its own main, so it owns an isolated set.
std::string TestLocalPath;
std::string TestRemotePath;
std::string TestMmapPath;

namespace milvus::exec {
namespace {

// Layer-2 producer benchmark, not a FilterBits/MVCC/search E2E benchmark.
// RawData runs the physical expression producer. STL_SORT and Bitmap run the
// exact native index primitives used by that producer, while deliberately
// excluding QueryContext payload handoff, visibility merge, and consumers.
constexpr int64_t kBatchRows = 8192;
constexpr int64_t kRatioScale = 1'000'000;
constexpr int64_t kCandidateRatioPpm = 9000;  // 0.9%
constexpr int64_t kAbsoluteSafeCap = 9000;

enum class IdDistribution : int64_t {
    Random = 0,
    Clustered = 1,
};

enum class OutputMode : int64_t {
    Dense = 0,
    Adaptive = 1,
};

struct BenchmarkArgs {
    int64_t universe = 0;
    int64_t accepted = 0;
    int64_t cap = 0;
    IdDistribution distribution = IdDistribution::Random;
    OutputMode mode = OutputMode::Dense;
};

struct ProducerOutput {
    std::shared_ptr<const std::vector<int32_t>> accepted_ids;
    std::shared_ptr<TargetBitmap> filtered;
    int64_t universe = 0;
    int64_t producer_batches = 0;

    bool
    IsSparse() const {
        return accepted_ids != nullptr;
    }

    bool
    IsDense() const {
        return filtered != nullptr;
    }
};

std::vector<int64_t>
BuildValues(int64_t universe, IdDistribution distribution) {
    std::vector<int64_t> values(static_cast<size_t>(universe));
    std::iota(values.begin(), values.end(), int64_t{0});
    if (distribution == IdDistribution::Random) {
        std::mt19937_64 generator(0xD15EA5E5ULL);
        std::shuffle(values.begin(), values.end(), generator);
    }
    return values;
}

void
SetInt64FieldData(segcore::GeneratedData& dataset,
                  FieldId field_id,
                  const std::vector<int64_t>& values) {
    AssertInfo(dataset.raw_->num_rows() == values.size(),
               "benchmark values size must match row count");
    for (int i = 0; i < dataset.raw_->fields_data_size(); ++i) {
        auto* field_data = dataset.raw_->mutable_fields_data(i);
        if (field_data->field_id() != field_id.get()) {
            continue;
        }
        auto* data =
            field_data->mutable_scalars()->mutable_long_data()->mutable_data();
        data->Clear();
        data->Add(values.data(), values.data() + values.size());
        return;
    }
    ThrowInfo(FieldIDInvalid,
              "benchmark field id {} was not generated",
              field_id.get());
}

struct RawFixture {
    explicit RawFixture(int64_t row_count, IdDistribution id_distribution)
        : universe(row_count), distribution(id_distribution) {
        schema = std::make_shared<Schema>();
        const auto pk = schema->AddDebugField("pk", DataType::INT64);
        value_field = schema->AddDebugField("value", DataType::INT64);
        schema->set_primary_field_id(pk);

        auto dataset = segcore::DataGen(schema, universe, /*seed=*/731);
        SetInt64FieldData(
            dataset, value_field, BuildValues(universe, distribution));
        segment = CreateSealedWithFieldDataLoaded(schema, dataset);
        AssertInfo(segment->num_chunk_data(value_field) == 1,
                   "raw producer benchmark requires one fixed data chunk, "
                   "got {}",
                   segment->num_chunk_data(value_field));
        AssertInfo(!segment->HasIndex(value_field),
                   "raw producer benchmark unexpectedly loaded an index");
    }

    int64_t universe;
    IdDistribution distribution;
    SchemaPtr schema;
    FieldId value_field;
    std::unique_ptr<segcore::SegmentSealed> segment;
};

struct RawFixtureCache {
    std::tuple<int64_t, int64_t> key{-1, -1};
    std::shared_ptr<RawFixture> value;
};

RawFixtureCache&
GetRawFixtureCache() {
    static RawFixtureCache cache;
    return cache;
}

std::shared_ptr<RawFixture>
GetRawFixture(int64_t universe, IdDistribution distribution) {
    auto& cache = GetRawFixtureCache();
    const auto key =
        std::make_tuple(universe, static_cast<int64_t>(distribution));
    if (cache.value == nullptr || cache.key != key) {
        cache.value = std::make_shared<RawFixture>(universe, distribution);
        cache.key = key;
    }
    return cache.value;
}

std::shared_ptr<folly::CPUThreadPoolExecutor>&
GetPrefetchPool() {
    static auto pool = std::make_shared<folly::CPUThreadPoolExecutor>(1);
    return pool;
}

struct PreparedRawExpression {
    PreparedRawExpression(const RawFixture& fixture, int64_t upper_bound) {
        proto::plan::GenericValue bound;
        bound.set_int64_val(upper_bound);
        logical = std::make_shared<expr::UnaryRangeFilterExpr>(
            expr::ColumnInfo(fixture.value_field, DataType::INT64),
            proto::plan::OpType::LessThan,
            bound,
            std::vector<proto::plan::GenericValue>{});

        auto config = std::make_shared<QueryConfig>(
            std::unordered_map<std::string, std::string>{
                {QueryConfig::kExprEvalBatchSize, std::to_string(kBatchRows)}});
        query_context =
            std::make_shared<QueryContext>("adaptive_filter_producer_benchmark",
                                           fixture.segment.get(),
                                           fixture.universe,
                                           MAX_TIMESTAMP,
                                           /*collection_ttl=*/0,
                                           /*consistency_level=*/0,
                                           query::PlanOptions{},
                                           std::move(config));
        query_context->set_enable_expr_cache(false);
        exec_context = std::make_unique<ExecContext>(query_context.get());
        expr_set =
            std::make_unique<ExprSet>(std::vector<expr::TypedExprPtr>{logical},
                                      exec_context.get(),
                                      /*null_rejecting=*/true);
        eval_context = std::make_unique<EvalCtx>(exec_context.get());

        // Data loading and asynchronous prefetch are fixed setup terms.  Both
        // modes complete the same prefetch before their timed producer call.
        expr_set->PrefetchAsync(GetPrefetchPool());
        expr_set->WaitPrefetch();
    }

    expr::TypedExprPtr logical;
    std::shared_ptr<QueryContext> query_context;
    std::unique_ptr<ExecContext> exec_context;
    std::unique_ptr<ExprSet> expr_set;
    std::unique_ptr<EvalCtx> eval_context;
};

ProducerOutput
RunRawDense(PreparedRawExpression& prepared, int64_t universe) {
    std::vector<VectorPtr> results;
    TargetBitmap accepted;
    TargetBitmap valid;
    int64_t batches = 0;
    while (static_cast<int64_t>(accepted.size()) < universe) {
        prepared.expr_set->Eval(0, 1, true, *prepared.eval_context, results);
        AssertInfo(results.size() == 1 && results[0] != nullptr,
                   "raw Dense benchmark expression returned no result");
        auto column = std::dynamic_pointer_cast<ColumnVector>(results[0]);
        AssertInfo(column != nullptr && column->IsBitmap(),
                   "raw Dense benchmark expression returned a non-bitmap");
        const auto rows = column->size();
        AssertInfo(
            rows > 0 && accepted.size() + rows <= static_cast<size_t>(universe),
            "raw Dense benchmark returned invalid batch size {}",
            rows);
        accepted.append(TargetBitmapView(column->GetRawData(), rows));
        valid.append(TargetBitmapView(column->GetValidRawData(), rows));
        ++batches;
    }
    accepted.inplace_and(valid, accepted.size());
    accepted.flip();
    return ProducerOutput{nullptr,
                          std::make_shared<TargetBitmap>(std::move(accepted)),
                          universe,
                          batches};
}

ProducerOutput
RunRawAdaptive(PreparedRawExpression& prepared, int64_t universe, int64_t cap) {
    auto physical = prepared.expr_set->expr(0);
    AssertInfo(physical->CanApplySparseFilter(
                   *prepared.eval_context, /*has_sparse_input=*/false, cap),
               "raw INT64 producer failed capability preflight");
    auto result = physical->TryApplySparseFilter(
        *prepared.eval_context, std::nullopt, cap);
    AssertInfo(result.has_value(),
               "raw INT64 producer declined after successful preflight");
    return ProducerOutput{std::move(result->accepted_ids),
                          std::move(result->filtered),
                          result->universe,
                          (universe + kBatchRows - 1) / kBatchRows};
}

template <typename DenseFn, typename AdaptiveFn>
std::string
CheckEquivalent(int64_t universe,
                int64_t expected_accepted,
                int64_t cap,
                DenseFn&& dense_fn,
                AdaptiveFn&& adaptive_fn) {
    const auto dense = dense_fn();
    const auto adaptive = adaptive_fn();
    if (!dense.IsDense() || dense.IsSparse() || dense.universe != universe ||
        dense.filtered->size() != universe) {
        return "Dense producer returned an invalid representation";
    }
    if (adaptive.IsDense() == adaptive.IsSparse() ||
        adaptive.universe != universe) {
        return "Adaptive producer returned an invalid representation";
    }
    if (dense.producer_batches <= 0 ||
        dense.producer_batches != adaptive.producer_batches) {
        return "Dense and Adaptive producer batch counts differ";
    }
    if (expected_accepted <= cap && !adaptive.IsSparse()) {
        return "Adaptive producer did not keep a within-cap result Sparse";
    }
    if (expected_accepted > cap && !adaptive.IsDense()) {
        return "Adaptive producer did not switch an over-cap result to Dense";
    }

    TargetBitmap adaptive_accepted(static_cast<size_t>(universe), false);
    if (adaptive.IsSparse()) {
        if (adaptive.accepted_ids->size() !=
            static_cast<size_t>(expected_accepted)) {
            return "Adaptive Sparse cardinality differs from requested V";
        }
        for (const auto id : *adaptive.accepted_ids) {
            if (id < 0 || id >= universe || adaptive_accepted[id]) {
                return "Adaptive Sparse IDs violate range/unique contract";
            }
            adaptive_accepted.set(static_cast<size_t>(id));
        }
    } else {
        if (adaptive.filtered->size() != static_cast<size_t>(universe)) {
            return "Adaptive Dense universe differs from N";
        }
        for (int64_t id = 0; id < universe; ++id) {
            adaptive_accepted[id] = !(*adaptive.filtered)[id];
        }
    }

    int64_t observed_accepted = 0;
    for (int64_t id = 0; id < universe; ++id) {
        const bool expected = !(*dense.filtered)[id];
        if (expected != adaptive_accepted[id]) {
            return "Dense and Adaptive producer outputs differ";
        }
        observed_accepted += expected;
    }
    if (observed_accepted != expected_accepted) {
        return "predicate did not produce the requested V";
    }
    return {};
}

bool
ReadArgs(benchmark::State& state, BenchmarkArgs& args) {
    args.universe = state.range(0);
    args.accepted = state.range(1);
    args.cap = state.range(2);
    const auto distribution = state.range(3);
    const auto mode = state.range(4);
    if (args.universe <= 0 || args.accepted < 0 ||
        args.accepted > args.universe || args.cap < 0 ||
        args.cap > args.universe ||
        (distribution != static_cast<int64_t>(IdDistribution::Random) &&
         distribution != static_cast<int64_t>(IdDistribution::Clustered)) ||
        (mode != static_cast<int64_t>(OutputMode::Dense) &&
         mode != static_cast<int64_t>(OutputMode::Adaptive))) {
        state.SkipWithError("invalid N/V/T/distribution/mode Args");
        return false;
    }
    args.distribution = static_cast<IdDistribution>(distribution);
    args.mode = static_cast<OutputMode>(mode);
    return true;
}

void
PublishCounters(benchmark::State& state,
                const BenchmarkArgs& args,
                bool final_sparse,
                int64_t producer_batches,
                int64_t predicate_rows,
                int64_t row_value_loads,
                int64_t index_lookups) {
    const auto dense_words =
        final_sparse ? 0 : (args.universe + int64_t{63}) / int64_t{64};
    state.counters["N"] = static_cast<double>(args.universe);
    state.counters["V"] = static_cast<double>(args.accepted);
    state.counters["V/N"] = static_cast<double>(args.accepted) / args.universe;
    state.counters["T"] = static_cast<double>(args.cap);
    state.counters["adaptive"] =
        static_cast<double>(args.mode == OutputMode::Adaptive);
    state.counters["random"] =
        static_cast<double>(args.distribution == IdDistribution::Random);
    state.counters["predicate_passes"] = 1.0;
    state.counters["predicate_rows"] = static_cast<double>(predicate_rows);
    state.counters["producer_batches"] = static_cast<double>(producer_batches);
    state.counters["row_value_loads"] = static_cast<double>(row_value_loads);
    state.counters["index_lookups"] = static_cast<double>(index_lookups);
    state.counters["ids_emitted"] =
        final_sparse ? static_cast<double>(args.accepted) : 0.0;
    state.counters["dense_words"] = static_cast<double>(dense_words);
    state.counters["dense_to_sparse"] = 0.0;
    state.counters["sort_calls"] = 0.0;
    state.counters["final_sparse"] = static_cast<double>(final_sparse);
    state.counters["threshold_fallback"] = static_cast<double>(
        args.mode == OutputMode::Adaptive && args.accepted > args.cap);
    state.SetItemsProcessed(state.iterations() * args.universe);
    state.SetLabel(fmt::format(
        "{} / {}",
        args.mode == OutputMode::Dense ? "dense" : "adaptive",
        args.distribution == IdDistribution::Random ? "random" : "clustered"));
}

void
RawProducerBenchmark(benchmark::State& state) {
    BenchmarkArgs args;
    if (!ReadArgs(state, args)) {
        return;
    }
    const auto fixture = GetRawFixture(args.universe, args.distribution);

    if (auto error = CheckEquivalent(
            args.universe,
            args.accepted,
            args.cap,
            [&]() {
                PreparedRawExpression prepared(*fixture, args.accepted);
                return RunRawDense(prepared, args.universe);
            },
            [&]() {
                PreparedRawExpression prepared(*fixture, args.accepted);
                return RunRawAdaptive(prepared, args.universe, args.cap);
            });
        !error.empty()) {
        state.SkipWithError(error.c_str());
        return;
    }

    const bool final_sparse =
        args.mode == OutputMode::Adaptive && args.accepted <= args.cap;
    // Closure always runs Dense then Adaptive. Repeat the selected mode once
    // outside the timed loop so neither side wins merely by running last in
    // closure and retaining hotter instructions/output-path metadata.
    {
        PreparedRawExpression prepared(*fixture, args.accepted);
        auto warm = args.mode == OutputMode::Dense
                        ? RunRawDense(prepared, args.universe)
                        : RunRawAdaptive(prepared, args.universe, args.cap);
        if (warm.IsSparse()) {
            benchmark::DoNotOptimize(warm.accepted_ids->data());
        } else {
            benchmark::DoNotOptimize(warm.filtered->data());
        }
    }

    int64_t observed_batches = 0;
    for (auto _ : state) {
        state.PauseTiming();
        auto prepared =
            std::make_unique<PreparedRawExpression>(*fixture, args.accepted);
        state.ResumeTiming();

        {
            ProducerOutput output;
            if (args.mode == OutputMode::Dense) {
                output = RunRawDense(*prepared, args.universe);
            } else {
                output = RunRawAdaptive(*prepared, args.universe, args.cap);
            }
            observed_batches = output.producer_batches;
            if (output.IsSparse()) {
                benchmark::DoNotOptimize(output.accepted_ids->data());
                benchmark::DoNotOptimize(output.accepted_ids->size());
            } else {
                benchmark::DoNotOptimize(output.filtered->data());
                benchmark::DoNotOptimize(output.filtered->size());
            }
            benchmark::ClobberMemory();
            // Both list and bitmap outputs are released while timing remains
            // active. Only expression/query scaffolding destruction is
            // excluded below, symmetrically for both modes.
        }

        state.PauseTiming();
        prepared.reset();
        state.ResumeTiming();
    }
    PublishCounters(state,
                    args,
                    final_sparse,
                    observed_batches,
                    /*predicate_rows=*/args.universe,
                    /*row_value_loads=*/args.universe,
                    /*index_lookups=*/0);
}

struct SortFixture {
    SortFixture(int64_t universe, IdDistribution distribution)
        : values(BuildValues(universe, distribution)) {
        index.Build(values.size(), values.data());
    }

    std::vector<int64_t> values;
    index::ScalarIndexSort<int64_t> index;
};

struct SortFixtureCache {
    std::tuple<int64_t, int64_t> key{-1, -1};
    std::shared_ptr<SortFixture> value;
};

SortFixtureCache&
GetSortFixtureCache() {
    static SortFixtureCache cache;
    return cache;
}

std::shared_ptr<SortFixture>
GetSortFixture(int64_t universe, IdDistribution distribution) {
    auto& cache = GetSortFixtureCache();
    const auto key =
        std::make_tuple(universe, static_cast<int64_t>(distribution));
    if (cache.value == nullptr || cache.key != key) {
        cache.value = std::make_shared<SortFixture>(universe, distribution);
        cache.key = key;
    }
    return cache.value;
}

ProducerOutput
RunSortDense(SortFixture& fixture, int64_t universe, int64_t upper_bound) {
    auto accepted =
        fixture.index.Range(upper_bound, proto::plan::OpType::LessThan);
    accepted.flip();
    return ProducerOutput{nullptr,
                          std::make_shared<TargetBitmap>(std::move(accepted)),
                          universe,
                          1};
}

ProducerOutput
RunSortAdaptive(SortFixture& fixture,
                int64_t universe,
                int64_t upper_bound,
                int64_t cap) {
    if (fixture.index.CanGetValidIdRange(upper_bound,
                                         proto::plan::OpType::LessThan,
                                         static_cast<size_t>(cap))) {
        auto ids =
            fixture.index.TryGetValidIdRange(upper_bound,
                                             proto::plan::OpType::LessThan,
                                             static_cast<size_t>(cap));
        AssertInfo(ids != nullptr,
                   "STL_SORT declined after successful capability check");
        return ProducerOutput{std::move(ids), nullptr, universe, 1};
    }

    // Native preflight has proved that the range exists but exceeds the cap.
    // Production now evaluates the ordinary Dense index result directly;
    // feeding it through AdaptiveFilterSink would scan the same accepted
    // bitmap a second time only to rediscover T+1.
    return RunSortDense(fixture, universe, upper_bound);
}

void
SortProducerBenchmark(benchmark::State& state) {
    BenchmarkArgs args;
    if (!ReadArgs(state, args)) {
        return;
    }
    auto fixture = GetSortFixture(args.universe, args.distribution);
    if (auto error = CheckEquivalent(
            args.universe,
            args.accepted,
            args.cap,
            [&]() {
                return RunSortDense(*fixture, args.universe, args.accepted);
            },
            [&]() {
                return RunSortAdaptive(
                    *fixture, args.universe, args.accepted, args.cap);
            });
        !error.empty()) {
        state.SkipWithError(error.c_str());
        return;
    }

    const bool final_sparse =
        args.mode == OutputMode::Adaptive && args.accepted <= args.cap;
    {
        auto warm = args.mode == OutputMode::Dense
                        ? RunSortDense(*fixture, args.universe, args.accepted)
                        : RunSortAdaptive(
                              *fixture, args.universe, args.accepted, args.cap);
        if (warm.IsSparse()) {
            benchmark::DoNotOptimize(warm.accepted_ids->data());
        } else {
            benchmark::DoNotOptimize(warm.filtered->data());
        }
    }
    for (auto _ : state) {
        ProducerOutput output;
        if (args.mode == OutputMode::Dense) {
            output = RunSortDense(*fixture, args.universe, args.accepted);
        } else {
            output = RunSortAdaptive(
                *fixture, args.universe, args.accepted, args.cap);
        }
        if (output.IsSparse()) {
            benchmark::DoNotOptimize(output.accepted_ids->data());
            benchmark::DoNotOptimize(output.accepted_ids->size());
        } else {
            benchmark::DoNotOptimize(output.filtered->data());
            benchmark::DoNotOptimize(output.filtered->size());
        }
        benchmark::ClobberMemory();
    }
    PublishCounters(state,
                    args,
                    final_sparse,
                    /*producer_batches=*/1,
                    /*predicate_rows=*/0,
                    /*row_value_loads=*/0,
                    /*index_lookups=*/
                    args.mode == OutputMode::Dense ? 1 : 2);
}

struct BitmapFixture {
    BitmapFixture(int64_t universe,
                  int64_t accepted,
                  IdDistribution distribution) {
        const auto ordering = BuildValues(universe, distribution);
        values.assign(static_cast<size_t>(universe), int64_t{0});
        for (int64_t row = 0; row < universe; ++row) {
            if (ordering[static_cast<size_t>(row)] < accepted) {
                values[static_cast<size_t>(row)] = 1;
            }
        }
        index.Build(values.size(), values.data());
    }

    std::vector<int64_t> values;
    index::BitmapIndex<int64_t> index;
};

struct BitmapFixtureCache {
    std::tuple<int64_t, int64_t, int64_t> key{-1, -1, -1};
    std::shared_ptr<BitmapFixture> value;
};

BitmapFixtureCache&
GetBitmapFixtureCache() {
    static BitmapFixtureCache cache;
    return cache;
}

std::shared_ptr<BitmapFixture>
GetBitmapFixture(int64_t universe,
                 int64_t accepted,
                 IdDistribution distribution) {
    auto& cache = GetBitmapFixtureCache();
    const auto key =
        std::make_tuple(universe, accepted, static_cast<int64_t>(distribution));
    if (cache.value == nullptr || cache.key != key) {
        cache.value =
            std::make_shared<BitmapFixture>(universe, accepted, distribution);
        cache.key = key;
    }
    return cache.value;
}

ProducerOutput
RunBitmapDense(BitmapFixture& fixture, int64_t universe) {
    constexpr int64_t kSelectedValue = 1;
    auto accepted = fixture.index.In(1, &kSelectedValue);
    accepted.flip();
    return ProducerOutput{nullptr,
                          std::make_shared<TargetBitmap>(std::move(accepted)),
                          universe,
                          1};
}

ProducerOutput
RunBitmapAdaptive(BitmapFixture& fixture, int64_t universe, int64_t cap) {
    constexpr int64_t kSelectedValue = 1;
    if (fixture.index.CanGetValidIdEqual(kSelectedValue,
                                         static_cast<size_t>(cap))) {
        auto ids = fixture.index.TryGetValidIdEqual(kSelectedValue,
                                                    static_cast<size_t>(cap));
        AssertInfo(ids != nullptr,
                   "BitmapIndex declined after successful capability check");
        return ProducerOutput{std::move(ids), nullptr, universe, 1};
    }

    return RunBitmapDense(fixture, universe);
}

void
BitmapProducerBenchmark(benchmark::State& state) {
    BenchmarkArgs args;
    if (!ReadArgs(state, args)) {
        return;
    }
    auto fixture =
        GetBitmapFixture(args.universe, args.accepted, args.distribution);
    if (auto error = CheckEquivalent(
            args.universe,
            args.accepted,
            args.cap,
            [&]() { return RunBitmapDense(*fixture, args.universe); },
            [&]() {
                return RunBitmapAdaptive(*fixture, args.universe, args.cap);
            });
        !error.empty()) {
        state.SkipWithError(error.c_str());
        return;
    }

    const bool final_sparse =
        args.mode == OutputMode::Adaptive && args.accepted <= args.cap;
    {
        auto warm = args.mode == OutputMode::Dense
                        ? RunBitmapDense(*fixture, args.universe)
                        : RunBitmapAdaptive(*fixture, args.universe, args.cap);
        if (warm.IsSparse()) {
            benchmark::DoNotOptimize(warm.accepted_ids->data());
        } else {
            benchmark::DoNotOptimize(warm.filtered->data());
        }
    }
    for (auto _ : state) {
        ProducerOutput output;
        if (args.mode == OutputMode::Dense) {
            output = RunBitmapDense(*fixture, args.universe);
        } else {
            output = RunBitmapAdaptive(*fixture, args.universe, args.cap);
        }
        if (output.IsSparse()) {
            benchmark::DoNotOptimize(output.accepted_ids->data());
            benchmark::DoNotOptimize(output.accepted_ids->size());
        } else {
            benchmark::DoNotOptimize(output.filtered->data());
            benchmark::DoNotOptimize(output.filtered->size());
        }
        benchmark::ClobberMemory();
    }
    PublishCounters(state,
                    args,
                    final_sparse,
                    /*producer_batches=*/1,
                    /*predicate_rows=*/0,
                    /*row_value_loads=*/0,
                    /*index_lookups=*/
                    args.mode == OutputMode::Dense ? 1 : 2);
}

void
ApplySwitchPointArgs(benchmark::internal::Benchmark* benchmark) {
    constexpr std::array<int64_t, 5> universes{
        50'000, 100'000, 250'000, 1'000'000, 10'000'000};
    // Parts per million: 0.01%, 0.05%, 0.1%, 0.2%, 0.5%, 1%.
    constexpr std::array<int64_t, 6> ratios_ppm{
        100, 500, 1000, 2000, 5000, 10'000};

    for (const auto universe : universes) {
        std::set<std::pair<int64_t, int64_t>> accepted_and_cap;

        // Lower-bound discovery sweep: cap=V is the smallest threshold that
        // admits each point and therefore keeps it Sparse.  T (and, for the
        // raw sink, its reserved capacity) changes with V, so this family is
        // not a fixed-policy causal sweep.  Use the fixed effective-cap
        // boundary below, plus Milvus E2E, for an actual policy decision.
        for (const auto ratio : ratios_ppm) {
            const auto accepted =
                std::max<int64_t>(1, universe * ratio / kRatioScale);
            accepted_and_cap.emplace(accepted, accepted);
        }

        // Provisional post-batch-fix policy boundary discovered by Milvus E2E:
        // T=min(0.9% * N, absolute safety cap).  The production selector is
        // intentionally not implied by this benchmark-only constant.
        const auto effective_cap = std::max<int64_t>(
            1,
            std::min<int64_t>(universe * kCandidateRatioPpm / kRatioScale,
                              kAbsoluteSafeCap));
        for (const auto accepted :
             {effective_cap - 1, effective_cap, effective_cap + 1}) {
            if (accepted >= 0 && accepted <= universe) {
                accepted_and_cap.emplace(accepted, effective_cap);
            }
        }

        for (const auto distribution :
             {IdDistribution::Random, IdDistribution::Clustered}) {
            for (const auto [accepted, cap] : accepted_and_cap) {
                for (const auto mode :
                     {OutputMode::Dense, OutputMode::Adaptive}) {
                    benchmark->Args({universe,
                                     accepted,
                                     cap,
                                     static_cast<int64_t>(distribution),
                                     static_cast<int64_t>(mode)});
                }
            }
        }
    }
}

void
ResetFixtures() {
    GetRawFixtureCache().value.reset();
    GetSortFixtureCache().value.reset();
    GetBitmapFixtureCache().value.reset();
    GetPrefetchPool().reset();
}

}  // namespace
}  // namespace milvus::exec

BENCHMARK(milvus::exec::RawProducerBenchmark)
    ->Apply(milvus::exec::ApplySwitchPointArgs)
    ->ArgNames({"N", "V", "T", "distribution", "mode"})
    ->UseRealTime();
BENCHMARK(milvus::exec::SortProducerBenchmark)
    ->Apply(milvus::exec::ApplySwitchPointArgs)
    ->ArgNames({"N", "V", "T", "distribution", "mode"})
    ->UseRealTime();
BENCHMARK(milvus::exec::BitmapProducerBenchmark)
    ->Apply(milvus::exec::ApplySwitchPointArgs)
    ->ArgNames({"N", "V", "T", "distribution", "mode"})
    ->UseRealTime();

int
main(int argc, char** argv) {
    // Consume Google Benchmark's flags before folly/gflags sees argv.  This
    // mirrors the unit-test runner's InitGoogleTest-before-folly ordering and
    // keeps flags such as --benchmark_filter and --benchmark_repetitions from
    // being rejected as unknown gflags.
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }
    folly::Init folly_init(&argc, &argv, false);

    const auto root = std::filesystem::temp_directory_path() /
                      fmt::format("milvus-adaptive-producer-{}", ::getpid());
    TestLocalPath = (root / "local").string() + "/";
    TestRemotePath = (root / "remote").string() + "/";
    TestMmapPath = (root / "mmap").string() + "/";
    std::filesystem::create_directories(TestLocalPath);
    std::filesystem::create_directories(TestRemotePath);
    std::filesystem::create_directories(TestMmapPath);

    milvus::storage::LocalChunkManagerSingleton::GetInstance().Init(
        TestLocalPath);
    milvus::storage::RemoteChunkManagerSingleton::GetInstance().Init(
        get_default_local_storage_config());
    milvus::storage::MmapManager::GetInstance().Init(get_default_mmap_config());

    CStorageConfig arrow_fs_config = {};
    arrow_fs_config.root_path = TestLocalPath.c_str();
    arrow_fs_config.storage_type = "local";
    const auto arrow_status = InitArrowFileSystem(arrow_fs_config);
    if (arrow_status.error_code != 0) {
        throw std::runtime_error("failed to initialize benchmark Arrow FS");
    }

    constexpr int64_t kMiB = 1024 * 1024;
    milvus::cachinglayer::Manager::ConfigureTieredStorage(
        {CacheWarmupPolicy::CacheWarmupPolicy_Disable,
         CacheWarmupPolicy::CacheWarmupPolicy_Disable,
         CacheWarmupPolicy::CacheWarmupPolicy_Disable,
         CacheWarmupPolicy::CacheWarmupPolicy_Disable},
        {1024 * kMiB,
         1024 * kMiB,
         1024 * kMiB,
         1024 * kMiB,
         1024 * kMiB,
         1024 * kMiB},
        true,
        true,
        {10, true, 30},
        std::chrono::milliseconds(0),
        std::chrono::milliseconds(-1));
    milvus::exec::ExprResCacheManager::SetEnabled(false);

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();

    milvus::exec::ResetFixtures();
    std::error_code cleanup_error;
    std::filesystem::remove_all(root, cleanup_error);
    return 0;
}
