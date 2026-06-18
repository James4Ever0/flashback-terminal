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

	"flashback-shell-tmux-renderer/internal/capture"
	"flashback-shell-tmux-renderer/internal/config"
	"flashback-shell-tmux-renderer/internal/forwarder"
	"flashback-shell-tmux-renderer/internal/log"
	"flashback-shell-tmux-renderer/internal/rawmode"
	"flashback-shell-tmux-renderer/internal/server"
	"flashback-shell-tmux-renderer/internal/tmux"
)

// NewCmd handles the "new" subcommand.
// All remaining args are passed directly to the spawned shell.
func NewCmd(cfg *config.Config, logger *log.Logger, args []string) int {
	if len(args) > 0 && strings.TrimLeft(args[0], "-") == "no-capture" {
		logger.Errorf("--no-capture is a global flag and must appear before the subcommand (e.g. flashback-shell --no-capture new)")
		return 1
	}

	if len(args) > 0 && args[0] == "--" {
		args = args[1:]
	}

	logger.Infof("effective config: server_url=%s socket_dir=%s shell=%s buffer_size=%d device_id=%s capture_interval=%d disable_capture=%t capture_scrollback=%t diff_only=%t diff_mode=%s text_only=%t",
		cfg.ServerURL, cfg.SocketDir, cfg.Shell, cfg.BufferSize, cfg.DeviceID, cfg.CaptureInterval, cfg.DisableCapture, cfg.CaptureScrollback, cfg.DiffOnly, cfg.DiffMode, cfg.TextOnly)

	shellBin := cfg.Shell
	if shellBin == "" {
		shellBin = os.Getenv("SHELL")
		if shellBin == "" {
			shellBin = "/bin/bash"
		}
	}
	shellArgs := args

	// We need tmux for the renderer. If tmux is missing, fall back to a plain
	// shell without forwarding/renderer/capture.
	if !tmux.HasTmux() {
		tmux.PrintInstallHint()
		return runFallbackShell(shellBin, shellArgs, logger)
	}

	return runForwarderWithRenderer(cfg, shellBin, shellArgs, logger)
}

func runForwarderWithRenderer(cfg *config.Config, shellBin string, shellArgs []string, logger *log.Logger) int {
	cwd, err := os.Getwd()
	if err != nil {
		logger.Errorf("getwd: %v", err)
		return 1
	}

	sessionID := fmt.Sprintf("%d", os.Getpid())

	// Ensure socket directories exist.
	if err := os.MkdirAll(cfg.SocketDir, 0755); err != nil {
		logger.Errorf("create socket dir: %v", err)
		return 1
	}
	forwarderSocket := filepath.Join(cfg.SocketDir, fmt.Sprintf("flashback-%s.fwd.sock", sessionID))

	// Normalize shell arguments. When the user passes `-c echo a && sleep 1`
	// (args split by an intermediate shell), bash -c would only execute `echo`
	// and treat the rest as positional parameters. Join everything after `-c`
	// into a single command string so the intended pipeline is preserved.
	if len(shellArgs) >= 2 && shellArgs[0] == "-c" {
		shellArgs = []string{"-c", strings.Join(shellArgs[1:], " ")}
	}

	// Determine terminal size to size the renderer tmux session initially.
	cols, rows := 80, 24
	if rawmode.IsTerminal(int(os.Stdin.Fd())) {
		cols, rows, _ = rawmode.Size(int(os.Stdin.Fd()))
	}
	if cols <= 0 {
		cols = 80
	}
	if rows <= 0 {
		rows = 24
	}

	// Find our own binary path so tmux can invoke the renderer subcommand.
	selfPath, err := os.Executable()
	if err != nil {
		logger.Errorf("cannot determine executable path: %v", err)
		return 1
	}
	selfPath, err = filepath.Abs(selfPath)
	if err != nil {
		logger.Errorf("cannot resolve executable path: %v", err)
		return 1
	}

	renderer := tmux.NewRendererSession(sessionID, cfg.SocketDir, forwarderSocket, selfPath, cols, rows)

	logger.Infof("starting renderer tmux session %s", renderer.Name)
	if err := renderer.Start(); err != nil {
		logger.Errorf("failed to start renderer tmux session: %v", err)
		return 1
	}

	// Start background capture goroutine if enabled.
	var captureCtx context.Context
	var captureCancel context.CancelFunc
	var captureWg sync.WaitGroup

	if !cfg.DisableCapture && cfg.CaptureInterval > 0 {
		captureCtx, captureCancel = context.WithCancel(context.Background())
		captureWg.Add(1)
		go backgroundCapture(captureCtx, &captureWg, cfg, renderer, logger)
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
			logger.Infof("cleaning up renderer tmux session %s", renderer.Name)
			renderer.Kill()
			_ = os.Remove(forwarderSocket)
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

	logger.Infof("starting PTY forwarder for session %s", sessionID)
	exitCode := forwarder.Run(forwarder.Options{
		Command:    shellBin,
		Args:       shellArgs,
		Dir:        cwd,
		SocketPath: forwarderSocket,
		Logger:     logger,
	})

	return exitCode
}

func runFallbackShell(shellBin string, shellArgs []string, logger *log.Logger) int {
	if shellBin == "" {
		var err error
		shellBin, err = findFallbackShell()
		if err != nil {
			logger.Errorf("no shell found: %v", err)
			return 1
		}
	}

	fmt.Fprintln(os.Stderr, "warning: tmux is not available; running a plain shell without renderer and capture.")

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

func backgroundCapture(ctx context.Context, wg *sync.WaitGroup, cfg *config.Config, sess *tmux.RendererSession, logger *log.Logger) {
	defer wg.Done()

	home, _ := os.UserHomeDir()
	stateDir := filepath.Join(home, ".flashback-shell", "state")
	bufferDir := filepath.Join(home, ".flashback-shell", "buffer")

	engine := capture.NewEngine(stateDir)
	engine.CaptureScrollback = cfg.CaptureScrollback
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

	if cfg.ServerURL == "" {
		logger.Infof("background capture started for session %s (interval=%s, server=none, local buffering only)", sess.Name, interval)
	} else {
		logger.Infof("background capture started for session %s (interval=%s, server=%s)", sess.Name, interval, cfg.ServerURL)
	}

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

func doBackgroundCapture(ctx context.Context, engine *capture.Engine, client *server.Client, cfg *config.Config, sess *tmux.RendererSession, logger *log.Logger) {
	if !sess.IsRunning() {
		logger.Debugf("background capture: session %s not running, skipping", sess.Name)
		return
	}

	captures, err := engine.CaptureSession(sess, sess.SessionID)
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

func findFallbackShell() (string, error) {
	if cfg := os.Getenv("SHELL"); cfg != "" {
		if _, err := exec.LookPath(cfg); err == nil {
			return cfg, nil
		}
	}
	if p, err := exec.LookPath("bash"); err == nil {
		return p, nil
	}
	if p, err := exec.LookPath("sh"); err == nil {
		return p, nil
	}
	return "", fmt.Errorf("no shell found in PATH (tried $SHELL, bash, sh)")
}
