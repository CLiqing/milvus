// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file distributed
// with this work for additional information regarding copyright ownership.
// The ASF licenses this file to you under the Apache License, Version 2.0.

package paramtable

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestFilterMapStartupParameters(t *testing.T) {
	params := &ComponentParam{}
	base := NewBaseTable(SkipRemote(true), SkipEnv(true))
	require.NoError(t, base.Save("localStorage.path", t.TempDir()))
	params.Init(base)
	require.False(t, params.QueryNodeCfg.FilterMapEnabled.GetAsBool())
	require.EqualValues(t, 50000, params.QueryNodeCfg.FilterMapMinRows.GetAsInt64())
	require.Equal(t, 0.004, params.QueryNodeCfg.FilterMapMaxRatio.GetAsFloat())
	require.NoError(t, params.Save(params.QueryNodeCfg.FilterMapEnabled.Key, "true"))
	require.NoError(t, params.Save(params.QueryNodeCfg.FilterMapMinRows.Key, "100000"))
	require.NoError(t, params.Save(params.QueryNodeCfg.FilterMapMaxRatio.Key, "0.001"))
	require.True(t, params.QueryNodeCfg.FilterMapEnabled.GetAsBool())
	require.EqualValues(t, 100000, params.QueryNodeCfg.FilterMapMinRows.GetAsInt64())
	require.Equal(t, 0.001, params.QueryNodeCfg.FilterMapMaxRatio.GetAsFloat())
}
