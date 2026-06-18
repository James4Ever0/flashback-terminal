package cmd

import (
	"flag"
	"os"
	"path/filepath"

	"flashback-shell-pty/pkg/capture"
	"flashback-shell-pty/pkg/config"
	"flashback-shell-pty/pkg/log"
	"flashback-shell-pty/pkg/server"
)

// CaptureCmd handles the "capture" subcommand.
func CaptureCmd(cfg *config.Config, logger *log.Logger, args []string) int {
	fs := flag.NewFlagSet("capture", flag.ExitOnError)
	scrollback := fs.Bool("scrollback", cfg.ScrollbackLines > 0, "include scrollback in capture")
	_, err := config.ParseSubcommandFlags(fs, args)
	if err != nil {
		logger.Errorf("failed to parse flags: %v", err)
		return 1
	}

	logger.Infof("effective config: server_url=%s socket_dir=%s shell=%s buffer_size=%d device_id=%s capture_interval=%d disable_capture=%t diff_only=%t diff_mode=%s text_only=%t scrollback_lines=%d",
		cfg.ServerURL, cfg.SocketDir, cfg.Shell, cfg.BufferSize, cfg.DeviceID, cfg.CaptureInterval, cfg.DisableCapture, cfg.DiffOnly, cfg.DiffMode, cfg.TextOnly, cfg.ScrollbackLines)

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

	logger.Infof("capturing all sessions")
	captures, err := engine.CaptureAll(cfg.SocketDir, *scrollback)
	if err != nil {
		logger.Errorf("capture failed: %v", err)
		return 1
	}

	if len(captures) == 0 {
		logger.Infof("no changes detected")
		return 0
	}

	logger.Infof("captured %d session(s)", len(captures))

	if err := engine.SaveHashes(captures); err != nil {
		logger.Warnf("failed to save hashes: %v", err)
	}

	if cfg.ServerURL == "" {
		logger.Warnf("no server configured (set FLASHBACK_SHELL_PTY_SERVER_URL or server_url in config); captures kept locally")
		return 0
	}

	logger.Infof("uploading %d capture(s) to %s", len(captures), cfg.ServerURL)
	if err := client.Upload(captures); err != nil {
		logger.Errorf("upload failed: %v", err)
		return 1
	}

	logger.Infof("uploaded successfully")
	return 0
}
