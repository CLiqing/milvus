package optimizers

import (
	"context"
	"fmt"
	"sync/atomic"

	"go.uber.org/zap"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus/pkg/v3/common"
	"github.com/milvus-io/milvus/pkg/v3/log"
	"github.com/milvus-io/milvus/pkg/v3/metrics"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/querypb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
)

var codexQueryHookDiagCounter atomic.Uint64

func codexQueryHookShouldLogDiag(n uint64) bool {
	return n <= 20 || n%1000 == 0
}

func codexDiagInt64Param(params map[string]any, key string) (int64, bool) {
	v, ok := params[key]
	if !ok {
		return 0, false
	}
	switch value := v.(type) {
	case int64:
		return value, true
	case int:
		return int64(value), true
	case int32:
		return int64(value), true
	case float64:
		return int64(value), true
	case float32:
		return int64(value), true
	default:
		return 0, false
	}
}

func codexDiagBoolParam(params map[string]any, key string) (bool, bool) {
	v, ok := params[key]
	if !ok {
		return false, false
	}
	value, ok := v.(bool)
	return value, ok
}

func codexDiagStringParam(params map[string]any, key string) (string, bool) {
	v, ok := params[key]
	if !ok {
		return "", false
	}
	value, ok := v.(string)
	return codexDiagTruncate(value), ok
}

func codexDiagTruncate(value string) string {
	const maxLen = 512
	if len(value) <= maxLen {
		return value
	}
	return value[:maxLen] + "...<truncated>"
}

// QueryHook is the interface for search/query parameter optimizer.
type QueryHook interface {
	Run(map[string]any) error
	Init(string) error
	InitTuningConfig(map[string]string) error
	DeleteTuningConfig(string) error
	CalculateEffectiveSegmentNum(rowCounts []int64, topk int64) int
}

// OptimizeSearchParams optimizes search parameters using the query hook.
// numSegments is the effective segment number, pre-computed by the caller via CalculateEffectiveSegmentNum.
// isSecondStageSearch is true for the vector search stage of two-stage search, refer to delegator_twostage.go.
// At this time, we need to set WithFilterKey to false to allow some aggressive optimizations.
func OptimizeSearchParams(ctx context.Context, req *querypb.SearchRequest, queryHook QueryHook, numSegments int, isSecondStageSearch bool, dimFunc func(fieldID int64) int64) (*querypb.SearchRequest, error) {
	// no hook applied or disabled, just return
	if queryHook == nil || !paramtable.Get().AutoIndexConfig.Enable.GetAsBool() {
		req.Req.IsTopkReduce = false
		req.Req.IsRecallEvaluation = false
		return req, nil
	}

	collectionId := req.GetReq().GetCollectionID()
	log := log.Ctx(ctx).With(zap.Int64("collection", collectionId))

	serializedPlan := req.GetReq().GetSerializedExprPlan()
	// plan not found
	if serializedPlan == nil {
		log.Warn("serialized plan not found")
		return req, merr.WrapErrParameterInvalid("serialized search plan", "nil")
	}

	channelNum := req.GetTotalChannelNum()
	// not set, change to conservative channel num 1
	if channelNum <= 0 {
		channelNum = 1
	}

	plan := planpb.PlanNode{}
	err := proto.Unmarshal(serializedPlan, &plan)
	if err != nil {
		log.Warn("failed to unmarshal plan", zap.Error(err))
		return nil, merr.WrapErrParameterInvalid("valid serialized search plan", "no unmarshalable one", err.Error())
	}

	switch plan.GetNode().(type) {
	case *planpb.PlanNode_VectorAnns:
		// use shardNum * segments num in shard to estimate total segment number
		estSegmentNum := numSegments * int(channelNum)
		metrics.QueryNodeSearchHitSegmentNum.WithLabelValues(paramtable.GetStringNodeID(), fmt.Sprint(collectionId), metrics.SearchLabel).Observe(float64(estSegmentNum))

		withFilter := (plan.GetVectorAnns().GetPredicates() != nil)
		queryInfo := plan.GetVectorAnns().GetQueryInfo()
		planTopKBeforeHook := queryInfo.GetTopk()
		searchParamsBeforeHook := queryInfo.GetSearchParams()
		reqTopKBeforeHook := req.GetReq().GetTopk()
		reqIsTopkReduceBeforeHook := req.GetReq().GetIsTopkReduce()
		reqIsRecallEvaluationBeforeHook := req.GetReq().GetIsRecallEvaluation()
		params := map[string]any{
			common.TopKKey:         queryInfo.GetTopk(),
			common.SearchParamKey:  queryInfo.GetSearchParams(),
			common.SegmentNumKey:   estSegmentNum,
			common.WithFilterKey:   withFilter && !isSecondStageSearch,
			common.DataTypeKey:     int32(plan.GetVectorAnns().GetVectorType()),
			common.WithOptimizeKey: paramtable.Get().AutoIndexConfig.EnableOptimize.GetAsBool() && req.GetReq().GetIsTopkReduce() && queryInfo.GetGroupByFieldId() < 0,
			common.CollectionKey:   req.GetReq().GetCollectionID(),
			common.RecallEvalKey:   req.GetReq().GetIsRecallEvaluation(),
		}
		if withFilter && channelNum > 1 {
			params[common.ChannelNumKey] = channelNum
		}
		globalRefineEnable := paramtable.Get().AutoIndexConfig.GlobalRefineEnable.GetAsBool()
		// Only check dim threshold and other conditions when global refine is enabled to reduce overhead
		if globalRefineEnable && (req.GetReq().GetSearchType() == internalpb.SearchType_PURE_ANN_SEARCH_NO_FILTER || req.GetReq().GetSearchType() == internalpb.SearchType_PURE_ANN_SEARCH_WITH_FILTER) {
			isFloatVector := plan.GetVectorAnns().GetVectorType() <= planpb.VectorType_BFloat16Vector && plan.GetVectorAnns().GetVectorType() >= planpb.VectorType_FloatVector
			minDimThreshold := paramtable.Get().AutoIndexConfig.GlobalRefineMinDimThreshold.GetAsInt64()
			// Disable global refine for group_by, non-float vector queries, and low-dimension vectors
			if queryInfo.GetGroupByFieldId() < 0 && isFloatVector && dimFunc(plan.GetVectorAnns().GetFieldId()) >= minDimThreshold {
				params[common.SearchTopkRatioKey] = float32(paramtable.Get().AutoIndexConfig.GlobalRefineSearchTopkRatio.GetAsFloat())
				params[common.RefineTopkRatioKey] = float32(paramtable.Get().AutoIndexConfig.GlobalRefineRefineTopkRatio.GetAsFloat())
			}
		}
		paramsTopKBeforeHook, paramsTopKBeforeHookOK := codexDiagInt64Param(params, common.TopKKey)
		paramsSearchBeforeHook, paramsSearchBeforeHookOK := codexDiagStringParam(params, common.SearchParamKey)
		paramsSegmentNumBeforeHook, paramsSegmentNumBeforeHookOK := codexDiagInt64Param(params, common.SegmentNumKey)
		paramsWithOptimizeBeforeHook, paramsWithOptimizeBeforeHookOK := codexDiagBoolParam(params, common.WithOptimizeKey)
		paramsWithFilterBeforeHook, paramsWithFilterBeforeHookOK := codexDiagBoolParam(params, common.WithFilterKey)
		err := queryHook.Run(params)
		if err != nil {
			log.Warn("failed to execute queryHook", zap.Error(err))
			return nil, merr.WrapErrServiceUnavailable(err.Error(), "queryHook execution failed")
		}
		finalTopk := params[common.TopKKey].(int64)
		isTopkReduce := req.GetReq().GetIsTopkReduce() && (finalTopk < queryInfo.GetTopk()) && !isSecondStageSearch
		paramsSearchAfterHook := params[common.SearchParamKey].(string)
		paramsWithOptimizeAfterHook, paramsWithOptimizeAfterHookOK := codexDiagBoolParam(params, common.WithOptimizeKey)
		paramsWithFilterAfterHook, paramsWithFilterAfterHookOK := codexDiagBoolParam(params, common.WithFilterKey)
		paramsSegmentNumAfterHook, paramsSegmentNumAfterHookOK := codexDiagInt64Param(params, common.SegmentNumKey)
		paramsRecallEvalAfterHook, paramsRecallEvalAfterHookOK := codexDiagBoolParam(params, common.RecallEvalKey)
		if reqIsTopkReduceBeforeHook || isTopkReduce || finalTopk < planTopKBeforeHook {
			diagSeq := codexQueryHookDiagCounter.Add(1)
			if codexQueryHookShouldLogDiag(diagSeq) {
				log.Warn("CODEX retry diagnosis: queryHook optimized search params",
					zap.Uint64("diagSeq", diagSeq),
					zap.Int64("collectionID", collectionId),
					zap.Int64("reqNq", req.GetReq().GetNq()),
					zap.Int64("reqTopKBeforeHook", reqTopKBeforeHook),
					zap.Int64("planTopKBeforeHook", planTopKBeforeHook),
					zap.Int64("paramsTopKBeforeHook", paramsTopKBeforeHook),
					zap.Bool("paramsTopKBeforeHookOK", paramsTopKBeforeHookOK),
					zap.Int64("paramsTopKAfterHook", finalTopk),
					zap.String("planSearchParamsBeforeHook", codexDiagTruncate(searchParamsBeforeHook)),
					zap.String("paramsSearchBeforeHook", paramsSearchBeforeHook),
					zap.Bool("paramsSearchBeforeHookOK", paramsSearchBeforeHookOK),
					zap.String("paramsSearchAfterHook", codexDiagTruncate(paramsSearchAfterHook)),
					zap.Int("numSegmentsInput", numSegments),
					zap.Int("estSegmentNum", estSegmentNum),
					zap.Int("channelNum", int(channelNum)),
					zap.Int64("paramsSegmentNumBeforeHook", paramsSegmentNumBeforeHook),
					zap.Bool("paramsSegmentNumBeforeHookOK", paramsSegmentNumBeforeHookOK),
					zap.Int64("paramsSegmentNumAfterHook", paramsSegmentNumAfterHook),
					zap.Bool("paramsSegmentNumAfterHookOK", paramsSegmentNumAfterHookOK),
					zap.Bool("withFilter", withFilter),
					zap.Bool("isSecondStageSearch", isSecondStageSearch),
					zap.Bool("reqIsTopkReduceBeforeHook", reqIsTopkReduceBeforeHook),
					zap.Bool("reqIsTopkReduceAfterHook", isTopkReduce),
					zap.Bool("reqIsRecallEvaluationBeforeHook", reqIsRecallEvaluationBeforeHook),
					zap.Bool("paramsRecallEvalAfterHook", paramsRecallEvalAfterHook),
					zap.Bool("paramsRecallEvalAfterHookOK", paramsRecallEvalAfterHookOK),
					zap.Bool("paramsWithOptimizeBeforeHook", paramsWithOptimizeBeforeHook),
					zap.Bool("paramsWithOptimizeBeforeHookOK", paramsWithOptimizeBeforeHookOK),
					zap.Bool("paramsWithOptimizeAfterHook", paramsWithOptimizeAfterHook),
					zap.Bool("paramsWithOptimizeAfterHookOK", paramsWithOptimizeAfterHookOK),
					zap.Bool("paramsWithFilterBeforeHook", paramsWithFilterBeforeHook),
					zap.Bool("paramsWithFilterBeforeHookOK", paramsWithFilterBeforeHookOK),
					zap.Bool("paramsWithFilterAfterHook", paramsWithFilterAfterHook),
					zap.Bool("paramsWithFilterAfterHookOK", paramsWithFilterAfterHookOK),
					zap.Int64("groupByFieldID", queryInfo.GetGroupByFieldId()),
					zap.Int64("fieldID", plan.GetVectorAnns().GetFieldId()),
					zap.Int32("vectorType", int32(plan.GetVectorAnns().GetVectorType())),
					zap.String("searchType", req.GetReq().GetSearchType().String()),
					zap.Bool("globalRefineEnabledConfig", globalRefineEnable))
			}
		}
		queryInfo.Topk = finalTopk
		queryInfo.SearchParams = paramsSearchAfterHook
		// Pass global refine decision to C++ via proto after hook validation
		if globalRefineVal, ok := params[common.GlobalRefineKey]; ok && globalRefineVal.(bool) {
			queryInfo.SearchTopkRatio = params[common.SearchTopkRatioKey].(float32)
			queryInfo.RefineTopkRatio = params[common.RefineTopkRatioKey].(float32)
			metrics.QueryNodeGlobalRefineCount.WithLabelValues(paramtable.GetStringNodeID(), fmt.Sprint(collectionId)).Inc()
		} else {
			queryInfo.SearchTopkRatio = 0
			queryInfo.RefineTopkRatio = 0
		}
		serializedExprPlan, err := proto.Marshal(&plan)
		if err != nil {
			log.Warn("failed to marshal optimized plan", zap.Error(err))
			return nil, merr.WrapErrParameterInvalid("marshalable search plan", "plan with marshal error", err.Error())
		}
		req.Req.SerializedExprPlan = serializedExprPlan
		req.Req.IsTopkReduce = isTopkReduce
		if isRecallEvaluation, ok := params[common.RecallEvalKey]; ok {
			req.Req.IsRecallEvaluation = isRecallEvaluation.(bool) && queryInfo.GetGroupByFieldId() < 0
		} else {
			req.Req.IsRecallEvaluation = false
		}

		log.Debug("optimized search params done", zap.Any("queryInfo", queryInfo))
	default:
		log.Warn("not supported node type", zap.String("nodeType", fmt.Sprintf("%T", plan.GetNode())))
	}
	return req, nil
}

// CalculateEffectiveSegmentNum delegates to queryHook.CalculateEffectiveSegmentNum when
// a hook is available; otherwise returns len(rowCounts) (the raw sealed segment count).
func CalculateEffectiveSegmentNum(queryHook QueryHook, rowCounts []int64, topk int64) int {
	if queryHook != nil && paramtable.Get().AutoIndexConfig.Enable.GetAsBool() {
		return queryHook.CalculateEffectiveSegmentNum(rowCounts, topk)
	}
	return len(rowCounts)
}

// ShouldUseTwoStageSearch determines if two-stage search should be used for this request
// based on paramtable config, segment count, topk, and search type.
func ShouldUseTwoStageSearch(req *querypb.SearchRequest, effectiveSegmentNum int) bool {
	if !paramtable.Get().AutoIndexConfig.TwoStageSearchEnabled.GetAsBool() {
		return false
	}
	if effectiveSegmentNum < paramtable.Get().AutoIndexConfig.TwoStageSearchMinNumSegments.GetAsInt() || req.GetReq().GetTopk() < paramtable.Get().AutoIndexConfig.TwoStageSearchMinTopk.GetAsInt64() {
		return false
	}
	return req.GetReq().GetSearchType() == internalpb.SearchType_PURE_ANN_SEARCH_WITH_FILTER
}
