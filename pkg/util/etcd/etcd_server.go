package etcd

import (
	"errors"
	"sync"
	"time"

	clientv3 "go.etcd.io/etcd/client/v3"
	"go.etcd.io/etcd/server/v3/embed"
	"go.etcd.io/etcd/server/v3/etcdserver/api/v3client"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus/pkg/v2/log"
)

// EtcdServer is the singleton of embedded etcd server
var (
	initOnce   sync.Once
	closeOnce  sync.Once
	etcdServer *embed.Etcd
)

// GetEmbedEtcdClient returns client of embed etcd server
func GetEmbedEtcdClient() (*clientv3.Client, error) {
	client := v3client.New(etcdServer.Server)
	return client, nil
}

// InitEtcdServer initializes embedded etcd server singleton.
func InitEtcdServer(
	useEmbedEtcd bool,
	configPath string,
	dataDir string,
	logPath string,
	logLevel string,
) error {
	if useEmbedEtcd {
		var initError error
		initOnce.Do(func() {
			path := configPath
			var cfg *embed.Config
			if len(path) > 0 {
				cfgFromFile, err := embed.ConfigFromFile(path)
				if err != nil {
					initError = err
				}
				cfg = cfgFromFile
			} else {
				cfg = embed.NewConfig()
			}
			cfg.Dir = dataDir
			cfg.LogOutputs = []string{logPath}
			cfg.LogLevel = logLevel
			e, err := embed.StartEtcd(cfg)
			if err != nil {
				log.Error("failed to init embedded Etcd server", zap.Error(err))
				initError = err
				return
			}
			select {
			case <-e.Server.ReadyNotify():
				log.Info("embedded Etcd server is ready")
			case <-time.After(60 * time.Second):
				e.Server.Stop()
				initError = errors.New("embedded Etcd took too long to start")
				return
			}
			etcdServer = e
			log.Info("finish init Etcd config", zap.String("path", path), zap.String("data", dataDir))
		})
		return initError
	}
	return nil
}

func HasServer() bool {
	return etcdServer != nil
}

// StopEtcdServer stops embedded etcd server singleton.
func StopEtcdServer() {
	if etcdServer != nil {
		closeOnce.Do(func() {
			etcdServer.Close()
		})
	}
}
