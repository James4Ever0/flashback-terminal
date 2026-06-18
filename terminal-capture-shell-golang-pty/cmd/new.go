package cmd

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"flashback-shell-pty/pkg/config"
	"flashback-shell-pty/pkg/log"
	"flashback-shell-pty/pkg/rawmode"
	"flashback-shell-pty/pkg/session"
)

// NewCmd handles the "new" subcommand.
func NewCmd(cfg *config.Config, logger *log.Logger, args []string) int {
	if len(args) > 0 && strings.TrimLeft(args[0], "-") == "no-capture" {
		logger.Errorf("--no-capture is a global flag and must appear before the subcommand (e.g. flashback-shell-pty --no-capture new)")
		return 1
	}
	if len(args) > 0 && args[0] == "--" {
		args = args[1:]
	}

	if os.Getenv("FLASHBACK_SHELL") != "" && !cfg.AllowNested {
		logger.Errorf("refusing to start a nested flashback-shell session (FLASHBACK_SHELL is set); use --allow-nested or set allow_nested: true to override")
		return 1
	}

	logger.Infof("effective config: server_url=%s socket_dir=%s shell=%s buffer_size=%d device_id=%s capture_interval=%d first_capture_delay=%d disable_capture=%t diff_only=%t diff_mode=%s text_only=%t scrollback_lines=%d allow_nested=%t",
		cfg.ServerURL, cfg.SocketDir, cfg.Shell, cfg.BufferSize, cfg.DeviceID, cfg.CaptureInterval, cfg.FirstCaptureDelay, cfg.DisableCapture, cfg.DiffOnly, cfg.DiffMode, cfg.TextOnly, cfg.ScrollbackLines, cfg.AllowNested)

	shellBin := cfg.Shell
	if shellBin == "" {
		shellBin = os.Getenv("SHELL")
		if shellBin == "" {
			shellBin = "/bin/bash"
		}
	}

	cwd, err := os.Getwd()
	if err != nil {
		logger.Errorf("getwd: %v", err)
		return 1
	}

	sessionID := fmt.Sprintf("%d", os.Getpid())
	_ = os.MkdirAll(cfg.SocketDir, 0755)
	socketPath := session.SocketPath(cfg.SocketDir, sessionID)

	exe, err := os.Executable()
	if err != nil {
		logger.Errorf("cannot locate executable: %v", err)
		return 1
	}
	exe, _ = filepath.Abs(exe)

	logger.Infof("starting session server %s", sessionID)

	serverArgs := []string{
		"__server",
		"--session-id", sessionID,
		"--socket", socketPath,
		"--cwd", cwd,
		"--",
	}
	serverArgs = append(serverArgs, args...)
	serverCmd := exec.Command(exe, serverArgs...)
	serverCmd.Stdin = nil

	var stderrBuf bytes.Buffer
	serverCmd.Stderr = &stderrBuf

	logger.Debugf("server command: %s %v", exe, serverArgs)

	if err := serverCmd.Start(); err != nil {
		logger.Errorf("failed to start session server: %v", err)
		return 1
	}

	logger.Debugf("server process started: pid=%d", serverCmd.Process.Pid)

	// Wait for the socket to appear.
	if err := waitForSocket(socketPath, 5*time.Second); err != nil {
		logger.Errorf("session server did not start: %v", err)
		_ = serverCmd.Process.Kill()
		serverCmd.Wait()
		if stderrBuf.Len() > 0 {
			logger.Debugf("server stderr:\n%s", stderrBuf.String())
		}
		logger.Debugf("server exit code: %d", serverCmd.ProcessState.ExitCode())
		return 1
	}

	// Attach.
	client := session.NewClient(socketPath)
	var cols, rows int
	if rawmode.IsTerminal(int(os.Stdin.Fd())) {
		cols, rows, _ = rawmode.Size(int(os.Stdin.Fd()))
	}
	conn, status, err := client.Attach(cols, rows)
	if err != nil {
		logger.Errorf("attach failed: %v", err)
		_ = session.NewClient(socketPath).Kill()
		return 1
	}

	// Print the current screen snapshot so the user sees output that was
	// produced before the attach completed.
	if status.Screen != "" {
		_, _ = os.Stdout.WriteString(status.Screen)
	}

	logger.Infof("attaching to session %s", sessionID)

	// Raw mode.
	var restore func() error
	if rawmode.IsTerminal(int(os.Stdin.Fd())) {
		var err error
		restore, _, err = rawmode.MakeRaw(int(os.Stdin.Fd()))
		if err != nil {
			logger.Warnf("make raw: %v", err)
		}
	}
	if restore != nil {
		defer restore()
	}

	// Resize monitor.
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, os.Interrupt, syscall.SIGTERM)
	defer signal.Stop(sigCh)

	resizeCh := make(chan os.Signal, 1)
	signal.Notify(resizeCh, syscall.SIGWINCH)
	defer signal.Stop(resizeCh)

	// Forward stdin -> connection.
	stdinDone := make(chan struct{})
	go func() {
		defer close(stdinDone)
		buf := make([]byte, 4096)
		for {
			n, err := os.Stdin.Read(buf)
			if n > 0 {
				_, _ = conn.Write(buf[:n])
			}
			if err != nil {
				return
			}
		}
	}()

	// Forward connection -> stdout.
	stdoutDone := make(chan struct{})
	go func() {
		defer close(stdoutDone)
		buf := make([]byte, 4096)
		for {
			n, err := conn.Read(buf)
			if n > 0 {
				_, _ = os.Stdout.Write(buf[:n])
			}
			if err != nil {
				return
			}
		}
	}()

	// Wait for disconnect, signal, or shell exit.
	for {
		select {
		case sig := <-sigCh:
			logger.Infof("received signal %s, detaching", sig)
			_ = conn.Close()
			<-stdoutDone
			return 0
		case <-stdoutDone:
			logger.Infof("session disconnected")
			return 0
		case <-resizeCh:
			if rawmode.IsTerminal(int(os.Stdin.Fd())) {
				c, r, err := rawmode.Size(int(os.Stdin.Fd()))
				if err == nil && c > 0 && r > 0 {
					if _, err := session.NewClient(socketPath).Resize(c, r); err != nil {
						logger.Debugf("resize failed: %v", err)
					}
				}
			}
		}
	}
}

func waitForSocket(path string, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if _, err := os.Stat(path); err == nil {
			return nil
		}
		time.Sleep(50 * time.Millisecond)
	}
	return fmt.Errorf("timeout waiting for socket %s", path)
}
