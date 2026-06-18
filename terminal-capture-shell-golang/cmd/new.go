package cmd

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"

	"flashback-shell/pkg/capture"
	"flashback-shell/pkg/config"
	"flashback-shell/pkg/log"
	"flashback-shell/pkg/server"
	"flashback-shell/pkg/shell"
)

// NewCmd handles the "new" subcommand.
// All remaining args are passed directly to the spawned shell. Since "new" has
// no command-specific flags, users do not need a "--" separator to pass shell
// arguments such as -c.
func NewCmd(cfg *config.Config, logger *log.Logger, args []string) int {
	if len(args) > 0 && strings.TrimLeft(args[0], "-") == "no-capture" {
		logger.Errorf("--no-capture is a global flag and must appear before the subcommand (e.g. flashback-shell --no-capture new)")
		return 1
	}

	// Allow the common "--" separator for users who still write
	// `flashback-shell new -- -c 'echo hello'`.
	if len(args) > 0 && args[0] == "--" {
		args = args[1:]
	}

	if cfg.AllowNestedTmux && !shell.HasEnv() {
		logger.Warnf("env binary not found on PATH; allow_nested_tmux ignored. Nested tmux requires env to unset TMUX variables.")
		cfg.AllowNestedTmux = false
	}

	logger.Infof("effective config: server_url=%s socket_dir=%s shell=%s buffer_size=%d device_id=%s capture_interval=%d disable_capture=%t capture_scrollback=%t allow_nested_tmux=%t diff_only=%t diff_mode=%s text_only=%t",
		cfg.ServerURL, cfg.SocketDir, cfg.Shell, cfg.BufferSize, cfg.DeviceID, cfg.CaptureInterval, cfg.DisableCapture, cfg.CaptureScrollback, cfg.AllowNestedTmux, cfg.DiffOnly, cfg.DiffMode, cfg.TextOnly)

	shellBin := cfg.Shell
	var shellArgs []string
	if shellBin == "" {
		shellBin = os.Getenv("SHELL")
		if shellBin == "" {
			shellBin = "/bin/bash"
		}
	}

	// Remaining args are passed directly to the spawned shell.
	shellArgs = args

	if shell.HasTmux() {
		return runTmux(cfg, shellBin, shellArgs, logger)
	}
	shell.PrintTmuxInstallHint()
	return runFallbackShell(shellBin, shellArgs, logger)
}

func runTmux(cfg *config.Config, shellBin string, shellArgs []string, logger *log.Logger) int {
	cwd, err := os.Getwd()
	if err != nil {
		logger.Errorf("getwd: %v", err)
		return 1
	}

	// Generate a simple session ID
	sessionID := fmt.Sprintf("%d", os.Getpid())

	logger.Infof("starting tmux session %s in %s", sessionID, cwd)

	sess := shell.NewTmuxSession(sessionID, cfg.SocketDir, cwd, shellBin, shellArgs)
	sess.CaptureScrollback = cfg.CaptureScrollback
	sess.AllowNestedTmux = cfg.AllowNestedTmux
	if err := sess.Start(); err != nil {
		logger.Errorf("failed to start tmux session: %v", err)
		return 1
	}

	// Start background capture goroutine if enabled
	var captureCtx context.Context
	var captureCancel context.CancelFunc
	var captureWg sync.WaitGroup

	if !cfg.DisableCapture && cfg.CaptureInterval > 0 {
		captureCtx, captureCancel = context.WithCancel(context.Background())
		captureWg.Add(1)
		go backgroundCapture(captureCtx, &captureWg, cfg, sess, logger)
	}

	// Ensure cleanup runs exactly once, even on signals or panics.
	var cleanupOnce sync.Once
	cleanup := func() {
		cleanupOnce.Do(func() {
			if captureCancel != nil {
				logger.Debugf("stopping background capture")
				captureCancel()
				captureWg.Wait()
			}
			logger.Infof("cleaning up tmux session %s", sessionID)
			sess.Kill()
		})
	}
	defer cleanup()

	// Handle SIGINT/SIGTERM so the tmux session is killed even if the parent
	// process is terminated externally.
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, os.Interrupt, syscall.SIGTERM)
	defer signal.Stop(sigCh)

	go func() {
		sig := <-sigCh
		logger.Infof("received signal %s, cleaning up", sig)
		cleanup()
		os.Exit(1)
	}()

	logger.Infof("attaching to tmux session %s", sessionID)
	// Attach and block
	attachErr := sess.Attach()

	if attachErr != nil {
		logger.Errorf("tmux attach exited: %v", attachErr)
		return 1
	}
	return 0
}

func backgroundCapture(ctx context.Context, wg *sync.WaitGroup, cfg *config.Config, sess *shell.TmuxSession, logger *log.Logger) {
	defer wg.Done()

	home, _ := os.UserHomeDir()
	stateDir := filepath.Join(home, ".flashback-shell", "state")
	bufferDir := filepath.Join(home, ".flashback-shell", "buffer")

	engine := capture.NewEngine(stateDir)
	engine.CaptureScrollback = cfg.CaptureScrollback
	engine.AllowNestedTmux = cfg.AllowNestedTmux
	engine.DiffOnly = cfg.DiffOnly
	engine.DiffMode = cfg.DiffMode
	engine.TextOnly = cfg.TextOnly
	buff := capture.NewBuffer(bufferDir, cfg.BufferSize)
	client := server.NewClient(cfg, buff, logger)

	// Flush any buffered captures from previous runs
	if buff.Count() > 0 {
		logger.Infof("flushing %d buffered capture batch(es)", buff.Count())
		if err := client.FlushRetries(); err != nil {
			logger.Warnf("flush retries failed: %v", err)
		}
	}

	interval := time.Duration(cfg.CaptureInterval) * time.Second
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	if cfg.ServerURL == "" {
		logger.Infof("background capture started for session %s (interval=%s, server=none, local buffering only)", sess.Name, interval)
	} else {
		logger.Infof("background capture started for session %s (interval=%s, server=%s)", sess.Name, interval, cfg.ServerURL)
	}

	// Initial capture immediately
	doBackgroundCapture(ctx, engine, client, cfg, sess, logger)

	for {
		select {
		case <-ctx.Done():
			logger.Infof("background capture stopped for session %s", sess.Name)
			return
		case <-ticker.C:
			doBackgroundCapture(ctx, engine, client, cfg, sess, logger)
		}
	}
}

func doBackgroundCapture(ctx context.Context, engine *capture.Engine, client *server.Client, cfg *config.Config, sess *shell.TmuxSession, logger *log.Logger) {
	if !sess.IsRunning() {
		logger.Debugf("background capture: session %s not running, skipping", sess.Name)
		return
	}

	panes, err := sess.ListPanes()
	if err != nil {
		logger.Warnf("background capture: cannot list panes for session %s: %v", sess.Name, err)
		return
	}
	logger.Debugf("background capture: session %s has %d pane(s)", sess.Name, len(panes))

	captures, err := engine.CaptureSession(sess)
	if err != nil {
		logger.Warnf("background capture failed for session %s: %v", sess.Name, err)
		return
	}

	if len(captures) == 0 {
		logger.Debugf("background capture: session %s no changes", sess.Name)
		return
	}

	logger.Infof("background capture: session %s %d pane(s) changed", sess.Name, len(captures))

	if err := engine.SaveHashes(captures); err != nil {
		logger.Warnf("failed to save hashes: %v", err)
	}

	if cfg.ServerURL == "" {
		logger.Infof("no server URL configured; declining upload of %d capture(s), keeping locally", len(captures))
		return
	}

	if client != nil {
		if err := client.Upload(captures); err != nil {
			logger.Warnf("background upload failed: %v", err)
		} else {
			logger.Debugf("background upload: %d capture(s) uploaded", len(captures))
		}
	}
}

func runFallbackShell(shellBin string, shellArgs []string, logger *log.Logger) int {
	if shellBin == "" {
		var err error
		shellBin, err = shell.FindFallbackShell()
		if err != nil {
			logger.Errorf("no shell found: %v", err)
			return 1
		}
	}

	fmt.Fprintln(os.Stderr, "warning: tmux is not available; running a plain shell without flashback-shell session management and background capture.")

	logger.Infof("running fallback shell: %s %v", shellBin, shellArgs)
	cmd := exec.Command(shellBin, shellArgs...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Dir, _ = os.Getwd()

	if err := cmd.Run(); err != nil {
		logger.Errorf("shell exited: %v", err)
		return 1
	}
	return 0
}
