package tasks

import "C"

import (
	"bytes"
	"context"
	"fmt"
	"strconv"

	"github.com/samber/lo"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/querynodev2/segments"
	"github.com/milvus-io/milvus/internal/util/searchutil/scheduler"
	"github.com/milvus-io/milvus/internal/util/segcore"
	"github.com/milvus-io/milvus/pkg/v3/metrics"
	"github.com/milvus-io/milvus/pkg/v3/mlog"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/querypb"
	"github.com/milvus-io/milvus/pkg/v3/util/funcutil"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
	"github.com/milvus-io/milvus/pkg/v3/util/resource"
	"github.com/milvus-io/milvus/pkg/v3/util/timerecord"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

var (
	_ scheduler.Task      = &SearchTask{}
	_ scheduler.MergeTask = &SearchTask{}
)

type SearchTask struct {
	ctx              context.Context
	collection       *segments.Collection
	segmentManager   *segments.Manager
	req              *querypb.SearchRequest
	result           *internalpb.SearchResults
	merged           bool
	groupSize        int64
	topk             int64
	nq               int64
	placeholderGroup []byte
	originTopks      []int64
	originNqs        []int64
	others           []*SearchTask
	notifier         chan error
	serverID         int64

	tr           *timerecord.TimeRecorder
	scheduleSpan trace.Span
}

func NewSearchTask(ctx context.Context,
	collection *segments.Collection,
	manager *segments.Manager,
	req *querypb.SearchRequest,
	serverID int64,
) *SearchTask {
	ctx, span := otel.Tracer(typeutil.QueryNodeRole).Start(ctx, "schedule")
	return &SearchTask{
		ctx:              ctx,
		collection:       collection,
		segmentManager:   manager,
		req:              req,
		merged:           false,
		groupSize:        1,
		topk:             req.GetReq().GetTopk(),
		nq:               req.GetReq().GetNq(),
		placeholderGroup: req.GetReq().GetPlaceholderGroup(),
		originTopks:      []int64{req.GetReq().GetTopk()},
		originNqs:        []int64{req.GetReq().GetNq()},
		notifier:         make(chan error, 1),
		tr:               timerecord.NewTimeRecorderWithTrace(ctx, "searchTask"),
		scheduleSpan:     span,
		serverID:         serverID,
	}
}

// Return the username which task is belong to.
// Return "" if the task do not contain any user info.
func (t *SearchTask) Username() string {
	return t.req.Req.GetUsername()
}

func (t *SearchTask) GetNodeID() int64 {
	return t.serverID
}

// subTaskAt returns the i-th sub-task: the receiver itself for i==0,
// otherwise t.others[i-1]. Sub-tasks index in lock-step with originNqs.
func (t *SearchTask) subTaskAt(i int) *SearchTask {
	if i == 0 {
		return t
	}
	return t.others[i-1]
}

func (t *SearchTask) IsGpuIndex() bool {
	return t.collection.IsGpuIndex()
}

func (t *SearchTask) PreExecute() error {
	// Update task wait time metric before execute
	nodeID := strconv.FormatInt(t.GetNodeID(), 10)
	inQueueDuration := t.tr.ElapseSpan()
	inQueueDurationMS := inQueueDuration.Seconds() * 1000

	// Update in queue metric for prometheus.
	metrics.QueryNodeSQLatencyInQueue.WithLabelValues(
		nodeID,
		metrics.SearchLabel,
		t.collection.GetDBName(),
		t.collection.GetResourceGroup(),
		// TODO: resource group and db name may be removed at runtime,
		// should be refactor into metricsutil.observer in the future.
	).Observe(inQueueDurationMS)

	username := t.Username()
	metrics.QueryNodeSQPerUserLatencyInQueue.WithLabelValues(
		nodeID,
		metrics.SearchLabel,
		username).
		Observe(inQueueDurationMS)

	// Execute merged task's PreExecute.
	for _, subTask := range t.others {
		err := subTask.PreExecute()
		if err != nil {
			return err
		}
	}

	return nil
}

func (t *SearchTask) Execute() error {
	if t.scheduleSpan != nil {
		t.scheduleSpan.End()
	}
	tr := timerecord.NewTimeRecorderWithTrace(t.ctx, "SearchTask")

	req := t.req
	err := t.combinePlaceHolderGroups()
	if err != nil {
		return err
	}
	searchReq, err := segcore.NewSearchRequest(t.collection.GetCCollection(), req, t.placeholderGroup)
	if err != nil {
		return err
	}
	defer searchReq.Delete()

	var (
		results          []*segments.SearchResult
		searchedSegments []segments.Segment
	)
	if req.GetScope() == querypb.DataScope_Historical {
		results, searchedSegments, err = segments.SearchHistorical(
			t.ctx,
			t.segmentManager,
			searchReq,
			req.GetReq().GetCollectionID(),
			req.GetReq().GetPartitionIDs(),
			req.GetSegmentIDs(),
		)
	} else if req.GetScope() == querypb.DataScope_Streaming {
		results, searchedSegments, err = segments.SearchStreaming(
			t.ctx,
			t.segmentManager,
			searchReq,
			req.GetReq().GetCollectionID(),
			req.GetReq().GetPartitionIDs(),
			req.GetSegmentIDs(),
		)
	}
	defer t.segmentManager.Segment.Unpin(searchedSegments)
	if err != nil {
		return err
	}
	defer segments.DeleteSearchResults(results)

	// In filter-only mode, extract filter statistics and return early.
	// This supports two-stage search: stage-1 collects per-segment valid
	// counts so the delegator can optimize search params for stage-2.
	if searchReq.FilterOnly() {
		if len(results) != len(searchedSegments) {
			return merr.WrapErrServiceInternalMsg("filter-only search: result count %d != segment count %d", len(results), len(searchedSegments))
		}
		segmentIDs := make([]int64, 0, len(searchedSegments))
		validCounts := make([]int64, 0, len(searchedSegments))
		for i, result := range results {
			segmentIDs = append(segmentIDs, searchedSegments[i].ID())
			validCounts = append(validCounts, result.ValidCount())
		}
		relatedDataSize := lo.Reduce(searchedSegments, func(acc int64, seg segments.Segment, _ int) int64 {
			return acc + segments.GetSegmentRelatedDataSize(seg)
		}, 0)
		for i := range t.originNqs {
			task := t.subTaskAt(i)
			task.result = &internalpb.SearchResults{
				Status:                   merr.Success(),
				SealedSegmentIDsSearched: segmentIDs,
				FilterValidCounts:        validCounts,
				CostAggregation: &internalpb.CostAggregation{
					ServiceTime:          tr.ElapseSpan().Milliseconds(),
					TotalRelatedDataSize: relatedDataSize,
				},
			}
		}
		mlog.Debug(t.ctx, "filter-only search completed", mlog.Int("segments", len(segmentIDs)))
		return nil
	}

	// plan.MetricType is accurate, though req.MetricType may be empty
	metricType := searchReq.Plan().GetMetricType()

	if len(results) == 0 {
		for i := range t.originNqs {
			task := t.subTaskAt(i)

			searchResults, err := segments.EncodeSearchResultData(
				t.ctx,
				emptySearchResultData(t.originNqs[i], t.originTopks[i]),
				t.originNqs[i],
				t.originTopks[i],
				metricType,
			)
			if err != nil {
				return err
			}
			searchResults.Base = &commonpb.MsgBase{
				SourceID: t.GetNodeID(),
			}
			searchResults.SlicedOffset = 1
			searchResults.SlicedNumCount = 1
			searchResults.CostAggregation = &internalpb.CostAggregation{
				ServiceTime: tr.ElapseSpan().Milliseconds(),
			}
			task.result = searchResults
		}
		return nil
	}

	relatedDataSize := lo.Reduce(searchedSegments, func(acc int64, seg segments.Segment, _ int) int64 {
		return acc + segments.GetSegmentRelatedDataSize(seg)
	}, 0)

	tr.RecordSpan()
	blobs, err := segcore.ReduceSearchResultsAndFillData(
		t.ctx,
		searchReq.Plan(),
		searchReq.PlaceholderGroup(),
		results,
		int64(len(results)),
		t.originNqs,
		t.originTopks,
	)
	if err != nil {
		mlog.Warn(t.ctx, "failed to reduce search results", mlog.Err(err))
		return err
	}
	allTasks := append([]*SearchTask{t}, t.others...)
	refs, err := resource.NewSharedPinnedRefs(
		blobs,
		len(allTasks),
		segcore.DeleteSearchResultDataBlobs,
		"SearchResultDataBlobs",
	)
	if err != nil {
		segcore.DeleteSearchResultDataBlobs(blobs)
		return err
	}

	metrics.QueryNodeReduceLatency.WithLabelValues(
		fmt.Sprint(t.GetNodeID()),
		metrics.SearchLabel,
		metrics.ReduceSegments,
		metrics.BatchReduce).
		Observe(float64(tr.RecordSpan().Microseconds()) / 1000.0)

	zeroCopy := paramtable.Get().QueryNodeCfg.EnableResultZeroCopy.GetAsBool()

	// Phase 1: build all results.
	var phaseErr error

	for i := range t.originNqs {
		blob, cost, err := segcore.GetSearchResultDataBlob(t.ctx, blobs, i)
		if err != nil {
			phaseErr = err
			break
		}
		slicedBlob := blob
		if !zeroCopy && len(blob) > 0 {
			slicedBlob = make([]byte, len(blob))
			copy(slicedBlob, blob)
		}
		allTasks[i].result = &internalpb.SearchResults{
			Base: &commonpb.MsgBase{
				SourceID: t.GetNodeID(),
			},
			Status:         merr.Success(),
			MetricType:     metricType,
			NumQueries:     t.originNqs[i],
			TopK:           t.originTopks[i],
			SlicedBlob:     slicedBlob,
			SlicedOffset:   1,
			SlicedNumCount: 1,
			CostAggregation: &internalpb.CostAggregation{
				ServiceTime:          tr.ElapseSpan().Milliseconds(),
				TotalRelatedDataSize: relatedDataSize,
			},
			ScannedRemoteBytes: cost.ScannedRemoteBytes,
			ScannedTotalBytes:  cost.ScannedTotalBytes,
		}
	}

	if phaseErr != nil {
		for _, task := range allTasks {
			task.result = nil
		}
		for _, ref := range refs {
			ref.Release()
		}
		return phaseErr
	}
	if zeroCopy {
		for i, task := range allTasks {
			resource.MsgPins.Pin(task.result, refs[i].Release)
		}
	} else {
		for _, ref := range refs {
			ref.Release()
		}
	}
	return nil
}

func emptySearchResultData(nq, topK int64) *schemapb.SearchResultData {
	return &schemapb.SearchResultData{
		NumQueries: nq,
		TopK:       topK,
		Ids:        &schemapb.IDs{},
		Scores:     []float32{},
		Topks:      make([]int64, int(nq)),
		FieldsData: []*schemapb.FieldData{},
	}
}

func (t *SearchTask) Merge(other *SearchTask) bool {
	var (
		nq        = t.nq
		topk      = t.topk
		otherNq   = other.nq
		otherTopk = other.topk
	)

	diffTopk := topk != otherTopk
	pre := funcutil.Min(nq*topk, otherNq*otherTopk)
	maxTopk := funcutil.Max(topk, otherTopk)
	after := (nq + otherNq) * maxTopk
	ratio := float64(after) / float64(pre)

	// Check mergeable
	if t.req.GetFilterOnly() != other.req.GetFilterOnly() ||
		t.req.GetEnableExprCache() != other.req.GetEnableExprCache() ||
		t.req.GetReq().GetDbID() != other.req.GetReq().GetDbID() ||
		t.req.GetReq().GetCollectionID() != other.req.GetReq().GetCollectionID() ||
		t.req.GetReq().GetMvccTimestamp() != other.req.GetReq().GetMvccTimestamp() ||
		t.req.GetReq().GetDslType() != other.req.GetReq().GetDslType() ||
		t.req.GetDmlChannels()[0] != other.req.GetDmlChannels()[0] ||
		nq+otherNq > paramtable.Get().QueryNodeCfg.MaxGroupNQ.GetAsInt64() ||
		diffTopk && ratio > paramtable.Get().QueryNodeCfg.TopKMergeRatio.GetAsFloat() ||
		!funcutil.SliceSetEqual(t.req.GetReq().GetPartitionIDs(), other.req.GetReq().GetPartitionIDs()) ||
		!funcutil.SliceSetEqual(t.req.GetSegmentIDs(), other.req.GetSegmentIDs()) ||
		!bytes.Equal(t.req.GetReq().GetSerializedExprPlan(), other.req.GetReq().GetSerializedExprPlan()) {
		return false
	}

	// Merge
	t.groupSize += other.groupSize
	t.topk = maxTopk
	t.nq += otherNq
	t.originTopks = append(t.originTopks, other.originTopks...)
	t.originNqs = append(t.originNqs, other.originNqs...)
	t.others = append(t.others, other)
	other.merged = true

	return true
}

func (t *SearchTask) Done(err error) {
	if !t.merged {
		metrics.QueryNodeSearchGroupSize.WithLabelValues(fmt.Sprint(t.GetNodeID())).Observe(float64(t.groupSize))
		metrics.QueryNodeSearchGroupNQ.WithLabelValues(fmt.Sprint(t.GetNodeID())).Observe(float64(t.nq))
		metrics.QueryNodeSearchGroupTopK.WithLabelValues(fmt.Sprint(t.GetNodeID())).Observe(float64(t.topk))
	}
	t.notifier <- err
	for _, other := range t.others {
		other.Done(err)
	}
}

func (t *SearchTask) Canceled() error {
	return t.ctx.Err()
}

func (t *SearchTask) Wait() error {
	return <-t.notifier
}

func (t *SearchTask) SearchResult() *internalpb.SearchResults {
	if t.result != nil {
		channelsMvcc := make(map[string]uint64)
		for _, ch := range t.req.GetDmlChannels() {
			channelsMvcc[ch] = t.req.GetReq().GetMvccTimestamp()
		}
		t.result.ChannelsMvcc = channelsMvcc
	}
	return t.result
}

func (t *SearchTask) NQ() int64 {
	return t.nq
}

func (t *SearchTask) MergeWith(other scheduler.Task) bool {
	switch other := other.(type) {
	case *SearchTask:
		return t.Merge(other)
	}
	return false
}

// combinePlaceHolderGroups combine all the placeholder groups.
func (t *SearchTask) combinePlaceHolderGroups() error {
	if len(t.others) == 0 {
		return nil
	}

	ret := &commonpb.PlaceholderGroup{}
	if err := proto.Unmarshal(t.placeholderGroup, ret); err != nil {
		return merr.WrapErrParameterInvalidMsg("invalid search vector placeholder: %v", err)
	}

	if len(ret.GetPlaceholders()) == 0 {
		return merr.WrapErrParameterInvalidMsg("empty search vector is not allowed")
	}
	for _, t := range t.others {
		x := &commonpb.PlaceholderGroup{}
		if err := proto.Unmarshal(t.placeholderGroup, x); err != nil {
			return merr.WrapErrParameterInvalidMsg("invalid search vector placeholder: %v", err)
		}
		if len(x.GetPlaceholders()) == 0 {
			return merr.WrapErrParameterInvalidMsg("empty search vector is not allowed")
		}
		ret.Placeholders[0].Values = append(ret.Placeholders[0].Values, x.Placeholders[0].Values...)
	}
	t.placeholderGroup, _ = proto.Marshal(ret)
	return nil
}
