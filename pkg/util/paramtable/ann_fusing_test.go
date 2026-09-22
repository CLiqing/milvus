// Licensed to the LF AI & Data foundation under one or more contributor
// license agreements. Licensed under the Apache License, Version 2.0.
package paramtable

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestAnnFusingStartupConfig(t *testing.T) {
	t.Run("unconfigured", func(t *testing.T) {
		var params ComponentParam
		params.Init(NewBaseTable(SkipRemote(true)))
		require.Empty(t, params.QueryNodeCfg.AnnFusingPluginPath.GetValue())
		require.Empty(t, params.QueryNodeCfg.AnnFusingConfigPath.GetValue())
	})
	t.Run("standard environment source", func(t *testing.T) {
		t.Setenv("MILVUS_CONF_QUERYNODE_ANNFUSING_PLUGINPATH", "/plugin/policy.so")
		t.Setenv("MILVUS_CONF_QUERYNODE_ANNFUSING_CONFIGPATH", "/plugin/policy.yaml")
		var params ComponentParam
		params.Init(NewBaseTable(SkipRemote(true)))
		require.Equal(t, "/plugin/policy.so", params.QueryNodeCfg.AnnFusingPluginPath.GetValue())
		require.Equal(t, "/plugin/policy.yaml", params.QueryNodeCfg.AnnFusingConfigPath.GetValue())
	})
}
