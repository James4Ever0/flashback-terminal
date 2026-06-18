package cmd

import (
	"context"
	"crypto/md5"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"flashback-shell-pty/pkg/capture"
	"flashback-shell-pty/pkg/config"
	"flashback-shell-pty/pkg/log"
	"flashback-shell-pty/pkg/server"
	"flashback-shell-pty/pkg/session"
)

// ServerCmd is the internal command used to run a session server process.
// It is not exposed in user-facing help.
func ServerCmd(cfg *config.Config, logger *log.Logger, args []string) int {
	fs := flag.NewFlagSet("__server", flag.ContinueOnError)
	sessionID := fs.String("session-id", "", "session identifier")
	socketPath := fs.String("socket", "", "Unix socket path")
	cwd := fs.String("cwd", "", "working directory")

	if err := fs.Parse(args); err != nil {
		logger.Errorf("server parse flags: %v", err)
		return 1
	}

	shellBin := cfg.Shell
	if shellBin == "" {
		shellBin = "/bin/bash"
	}

	if *sessionID == "" || *socketPath == "" {
		logger.Errorf("server requires --session-id and --socket")
		return 1
	}

	serverObj, err := session.NewServer(cfg, logger, *sessionID, *socketPath, *cwd, shellBin, fs.Args())
	if err != nil {
		logger.Errorf("create server: %v", err)
		return 1
	}

	var captureCancel context.CancelFunc
	var captureWg sync.WaitGroup
	if !cfg.DisableCapture && cfg.CaptureInterval > 0 {
		var captureCtx context.Context
		captureCtx, captureCancel = context.WithCancel(context.Background())
		captureWg.Add(1)
		go func() {
			<-serverObj.Ready()
			backgroundCapture(captureCtx, &captureWg, cfg, serverObj, *sessionID, logger)
		}()
	}

	logger.Infof("starting session server %s on %s", *sessionID, *socketPath)
	if err := serverObj.Run(); err != nil {
		if fmt.Sprintf("%v", err) != "" {
			logger.Debugf("server exited: %v", err)
		}
	}

	if captureCancel != nil {
		captureCancel()
		captureWg.Wait()
	}
	return 0
}

func backgroundCapture(ctx context.Context, wg *sync.WaitGroup, cfg *config.Config, srv *session.Server, sessionID string, logger *log.Logger) {
	defer wg.Done()

	home, _ := os.UserHomeDir()
	stateDir := filepath.Join(home, ".flashback-shell-pty", "state")
	bufferDir := filepath.Join(home, ".flashback-shell-pty", "buffer")

	engine := capture.NewEngine(stateDir)
	engine.DiffOnly = cfg.DiffOnly
	engine.DiffMode = cfg.DiffMode
	engine.TextOnly = cfg.TextOnly
	buff := capture.NewBuffer(bufferDir, cfg.BufferSize)
	client := server.NewClient(cfg, buff, logger)

	if buff.Count() > 0 {
		logger.Infof("flushing %d buffered capture batch(es)", buff.Count())
		if err := client.FlushRetries(); err != nil {
			logger.Warnf("flush retries failed: %v", err)
		}
	}

	interval := time.Duration(cfg.CaptureInterval) * time.Second
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	logger.Infof("background capture started for session %s (first_capture_delay=%s, interval=%s)", sessionID, time.Duration(cfg.FirstCaptureDelay)*time.Second, interval)

	select {
	case <-ctx.Done():
		logger.Infof("background capture stopped for session %s", sessionID)
		return
	case <-time.After(time.Duration(cfg.FirstCaptureDelay) * time.Second):
	}

	doCapture(engine, client, cfg, srv, sessionID, logger)

	for {
		select {
		case <-ctx.Done():
			logger.Infof("background capture stopped for session %s", sessionID)
			return
		case <-ticker.C:
			doCapture(engine, client, cfg, srv, sessionID, logger)
		}
	}
}

func doCapture(engine *capture.Engine, client *server.Client, cfg *config.Config, srv *session.Server, sessionID string, logger *log.Logger) {
	ansi, text, cols, rows, _, _ := srv.CaptureScreen(false)
	if cfg.TextOnly {
		ansi = ""
	}

	hashInput := ansi
	if cfg.TextOnly {
		hashInput = text
	}
	if hashInput == "" {
		return
	}

	hash := fmt.Sprintf("%x", md5.Sum([]byte(hashInput)))
	if engine.IsDuplicate(sessionID, "main", hash) {
		logger.Debugf("background capture: session %s no changes", sessionID)
		return
	}

	cap := capture.Capture{
		SessionID: sessionID,
		PaneID:    "main",
		Target:    "main",
		ANSI:      ansi,
		Text:      text,
		Hash:      hash,
		Cols:      cols,
		Rows:      rows,
		Timestamp: time.Now().UTC(),
	}
	if cfg.TextOnly {
		cap.Metadata = map[string]string{"ansi": "false"}
	}

	logger.Infof("background capture: session %s changed", sessionID)
	if err := engine.SaveHashes([]capture.Capture{cap}); err != nil {
		logger.Warnf("failed to save hash: %v", err)
	}

	if cfg.ServerURL == "" {
		logger.Infof("no server URL configured; declining upload, keeping locally")
		return
	}

	if err := client.Upload([]capture.Capture{cap}); err != nil {
		logger.Warnf("background upload failed: %v", err)
	} else {
		logger.Debugf("background upload: 1 capture uploaded")
	}
}
