package milvus

import (
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"

	"go.uber.org/zap"

	"github.com/milvus-io/milvus/pkg/v2/common"
	"github.com/milvus-io/milvus/pkg/v2/log"
	"github.com/milvus-io/milvus/pkg/v2/metrics"
	"github.com/milvus-io/milvus/pkg/v2/util/hardware"
	"github.com/milvus-io/milvus/pkg/v2/util/metricsinfo"
	"github.com/milvus-io/milvus/pkg/v2/util/paramtable"
)

type run struct{}

func (c *run) execute(args []string, flags *flag.FlagSet) {
	if len(args) < 3 {
		fmt.Fprintln(os.Stderr, getHelp())
		return
	}
	flags.Usage = func() {
		fmt.Fprintln(os.Stderr, getHelp())
	}
	// make go ignore SIGPIPE when all cgo thread set mask SIGPIPE
	signal.Ignore(syscall.SIGPIPE)

	serverType := args[2]
	roles := GetMilvusRoles(args, flags)
	roles.PostParamtableInit = func() {
		c.configureS3ReadPath(flags.Output())
	}
	// setup config for embedded milvus

	runtimeDir := createRuntimeDir(serverType)
	filename := getPidFileName(serverType, roles.Alias)

	maybeEnableOpenSSLFIPS()
	c.printBanner(flags.Output())
	c.injectVariablesToEnv()
	c.printHardwareInfo(flags.Output())
	lock, err := createPidFile(flags.Output(), filename, runtimeDir)
	if err != nil {
		panic(err)
	}
	defer removePidFile(lock)
	roles.Run()
}

func (c *run) printBanner(w io.Writer) {
	fmt.Fprintln(w)
	fmt.Fprintln(w, "    __  _________ _   ____  ______    ")
	fmt.Fprintln(w, "   /  |/  /  _/ /| | / / / / / __/    ")
	fmt.Fprintln(w, "  / /|_/ // // /_| |/ / /_/ /\\ \\    ")
	fmt.Fprintln(w, " /_/  /_/___/____/___/\\____/___/     ")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Welcome to Milvus!")
	fmt.Fprintln(w, "Version:   "+getEffectiveVersion())
	fmt.Fprintln(w, "Built:     "+BuildTime)
	fmt.Fprintln(w, "GitCommit: "+GitCommit)
	fmt.Fprintln(w, "GoVersion: "+GoVersion)
	fmt.Fprintf(w, "Milvus FIPS in Go: BoringCrypto %v\n", boringEnabled())
	fmt.Fprintln(w)
	metrics.BuildInfo.WithLabelValues(getEffectiveVersion(), BuildTime, GitCommit).Set(1)
}

func (c *run) printHardwareInfo(w io.Writer) {
	totalMem := hardware.GetMemoryCount()
	usedMem := hardware.GetUsedMemoryCount()
	fmt.Fprintf(w, "TotalMem: %d\n", totalMem)
	fmt.Fprintf(w, "UsedMem: %d\n", usedMem)
	fmt.Fprintln(w)
}

func getEffectiveVersion() string {
	if MilvusVersion != "" && MilvusVersion != "unknown" {
		return MilvusVersion
	}
	return common.Version.String()
}

func (c *run) injectVariablesToEnv() {
	// inject in need

	var err error

	err = os.Setenv(metricsinfo.GitCommitEnvKey, GitCommit)
	if err != nil {
		log.Warn(fmt.Sprintf("failed to inject %s to environment variable", metricsinfo.GitCommitEnvKey),
			zap.Error(err))
	}

	err = os.Setenv(metricsinfo.GitBuildTagsEnvKey, getEffectiveVersion())
	if err != nil {
		log.Warn(fmt.Sprintf("failed to inject %s to environment variable", metricsinfo.GitBuildTagsEnvKey),
			zap.Error(err))
	}

	err = os.Setenv(metricsinfo.MilvusBuildTimeEnvKey, BuildTime)
	if err != nil {
		log.Warn(fmt.Sprintf("failed to inject %s to environment variable", metricsinfo.MilvusBuildTimeEnvKey),
			zap.Error(err))
	}

	err = os.Setenv(metricsinfo.MilvusUsedGoVersion, GoVersion)
	if err != nil {
		log.Warn(fmt.Sprintf("failed to inject %s to environment variable", metricsinfo.MilvusUsedGoVersion),
			zap.Error(err))
	}
}

func (c *run) configureS3ReadPath(w io.Writer) {
	cfg := c.effectiveS3ReadPathConfig()
	path := strings.ToLower(strings.TrimSpace(cfg.path))
	if path == "" {
		return
	}

	switch path {
	case "baseline", "sync":
		unsetS3ReadPathEnv("MILVUS_S3_GETOBJECT_ASYNC")
		unsetS3ReadPathEnv("MILVUS_S3_CLIENT_COROUTINE")
		unsetS3ReadPathEnv("MILVUS_S3_CLIENT_CRT")
		unsetS3ReadPathEnv("MILVUS_S3_ASYNC_MAX_INFLIGHT")
		unsetS3ReadPathEnv("MILVUS_S3_CLIENT_COROUTINE_EVENTLOOPS")
		unsetS3ReadPathEnv("MILVUS_S3_CLIENT_CRT_EVENTLOOPS")
		unsetS3ReadPathEnv("MILVUS_S3_CLIENT_CRT_MAX_CONNECTIONS")
		unsetS3ReadPathEnv("MILVUS_S3_CLIENT_CRT_THROUGHPUT_GBPS")
		fmt.Fprintln(w, "S3ReadPath: baseline")
	case "curl_multi", "curl-multi", "curl":
		setS3ReadPathEnv("MILVUS_S3_GETOBJECT_ASYNC", "1")
		setS3ReadPathEnv("MILVUS_S3_CLIENT_COROUTINE", "1")
		unsetS3ReadPathEnv("MILVUS_S3_CLIENT_CRT")
		setS3ReadPathEnv("MILVUS_S3_ASYNC_MAX_INFLIGHT", strconv.FormatUint(cfg.maxInflight, 10))
		setS3ReadPathEnv("MILVUS_S3_CLIENT_COROUTINE_EVENTLOOPS", strconv.FormatUint(cfg.eventLoops, 10))
		fmt.Fprintf(w, "S3ReadPath: curl_multi max_inflight=%d eventloops=%d\n",
			cfg.maxInflight, cfg.eventLoops)
	case "crt", "awscrt", "aws-crt":
		setS3ReadPathEnv("MILVUS_S3_GETOBJECT_ASYNC", "1")
		unsetS3ReadPathEnv("MILVUS_S3_CLIENT_COROUTINE")
		setS3ReadPathEnv("MILVUS_S3_CLIENT_CRT", "1")
		setS3ReadPathEnv("MILVUS_S3_ASYNC_MAX_INFLIGHT", strconv.FormatUint(cfg.maxInflight, 10))
		setS3ReadPathEnv("MILVUS_S3_CLIENT_CRT_EVENTLOOPS", strconv.FormatUint(cfg.eventLoops, 10))
		setS3ReadPathEnv("MILVUS_S3_CLIENT_CRT_MAX_CONNECTIONS", strconv.FormatUint(cfg.crtMaxConnections, 10))
		if cfg.crtThroughputGbps != "" {
			setS3ReadPathEnv("MILVUS_S3_CLIENT_CRT_THROUGHPUT_GBPS", cfg.crtThroughputGbps)
		}
		fmt.Fprintf(w, "S3ReadPath: crt max_inflight=%d eventloops=%d crt_max_connections=%d",
			cfg.maxInflight, cfg.eventLoops, cfg.crtMaxConnections)
		if cfg.crtThroughputGbps != "" {
			fmt.Fprintf(w, " crt_throughput_gbps=%s", cfg.crtThroughputGbps)
		}
		fmt.Fprintln(w)
	default:
		fmt.Fprintf(os.Stderr, "Unknown S3 read path=%s, expected baseline, curl_multi, or crt\n", cfg.path)
		os.Exit(-1)
	}
}

func (c *run) effectiveS3ReadPathConfig() struct {
	path              string
	maxInflight       uint64
	eventLoops        uint64
	crtMaxConnections uint64
	crtThroughputGbps string
} {
	cfg := struct {
		path              string
		maxInflight       uint64
		eventLoops        uint64
		crtMaxConnections uint64
		crtThroughputGbps string
	}{
		path:              s3ReadPathConfig.path,
		maxInflight:       s3ReadPathConfig.maxInflight,
		eventLoops:        s3ReadPathConfig.eventLoops,
		crtMaxConnections: s3ReadPathConfig.crtMaxConnections,
		crtThroughputGbps: s3ReadPathConfig.crtThroughputGbps,
	}
	if s3ReadPathConfig.pathSet {
		return cfg
	}

	params := paramtable.Get()
	cfg.path = params.QueryNodeCfg.S3ReadPathMode.GetValue()
	if !s3ReadPathConfig.maxInflightSet {
		cfg.maxInflight = params.QueryNodeCfg.S3ReadPathMaxInflight.GetAsUint64()
	}
	if !s3ReadPathConfig.eventLoopsSet {
		cfg.eventLoops = params.QueryNodeCfg.S3ReadPathEventLoops.GetAsUint64()
	}
	if !s3ReadPathConfig.crtConnectionsSet {
		cfg.crtMaxConnections = params.QueryNodeCfg.S3ReadPathCrtMaxConnections.GetAsUint64()
	}
	if !s3ReadPathConfig.crtThroughputSet {
		cfg.crtThroughputGbps = params.QueryNodeCfg.S3ReadPathCrtThroughputGbps.GetValue()
	}
	return cfg
}

func setS3ReadPathEnv(name string, value string) {
	if err := os.Setenv(name, value); err != nil {
		log.Warn(fmt.Sprintf("failed to set %s for S3 read path", name), zap.Error(err))
	}
}

func unsetS3ReadPathEnv(name string) {
	if err := os.Unsetenv(name); err != nil {
		log.Warn(fmt.Sprintf("failed to unset %s for S3 read path", name), zap.Error(err))
	}
}
