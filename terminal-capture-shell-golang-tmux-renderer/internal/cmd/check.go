package cmd

import (
	"flag"
	"fmt"
	"os"
	"text/tabwriter"

	"flashback-shell-tmux-renderer/internal/config"
	"flashback-shell-tmux-renderer/internal/log"
	"flashback-shell-tmux-renderer/internal/tmux"
)

// CheckCmd handles the "check" subcommand: it validates required external
// binaries and prints the effective configuration with provenance.
func CheckCmd(cfg *config.Config, src *config.ConfigWithSource, logger *log.Logger, args []string) int {
	fs := flag.NewFlagSet("check", flag.ExitOnError)
	if _, err := config.ParseSubcommandFlags(fs, args); err != nil {
		logger.Errorf("failed to parse flags: %v", err)
		return 1
	}

	var warnings []string

	tmuxPath, tmuxErr := tmux.Path()
	if tmuxErr != nil {
		warnings = append(warnings, "tmux is not installed or not on PATH. flashback-shell will fall back to a plain shell; the renderer and capture features will be unavailable.")
	}

	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)

	fmt.Fprintln(w, "BINARIES")
	fmt.Fprintln(w, "NAME\tPATH")
	if tmuxErr != nil {
		fmt.Fprintln(w, "tmux\tnot found")
	} else {
		fmt.Fprintf(w, "tmux\t%s\n", tmuxPath)
	}
	w.Flush()
	fmt.Println()

	fmt.Fprintln(w, "CONFIG")
	fmt.Fprintln(w, "FIELD\tVALUE\tSOURCE")
	fmt.Fprintf(w, "server_url\t%s\t%s\n", cfg.ServerURL, src.ServerURL.Origin)
	fmt.Fprintf(w, "socket_dir\t%s\t%s\n", cfg.SocketDir, src.SocketDir.Origin)
	fmt.Fprintf(w, "shell\t%s\t%s\n", cfg.Shell, src.Shell.Origin)
	fmt.Fprintf(w, "buffer_size\t%d\t%s\n", cfg.BufferSize, src.BufferSize.Origin)
	fmt.Fprintf(w, "device_id\t%s\t%s\n", cfg.DeviceID, src.DeviceID.Origin)
	fmt.Fprintf(w, "capture_interval\t%d\t%s\n", cfg.CaptureInterval, src.CaptureInterval.Origin)
	fmt.Fprintf(w, "disable_capture\t%t\t%s\n", cfg.DisableCapture, src.DisableCapture.Origin)
	fmt.Fprintf(w, "capture_scrollback\t%t\t%s\n", cfg.CaptureScrollback, src.CaptureScrollback.Origin)
	fmt.Fprintf(w, "diff_only\t%t\t%s\n", cfg.DiffOnly, src.DiffOnly.Origin)
	fmt.Fprintf(w, "diff_mode\t%s\t%s\n", cfg.DiffMode, src.DiffMode.Origin)
	fmt.Fprintf(w, "text_only\t%t\t%s\n", cfg.TextOnly, src.TextOnly.Origin)
	w.Flush()
	fmt.Println()

	fmt.Fprintln(w, "WARNINGS")
	if len(warnings) == 0 {
		fmt.Fprintln(w, "No warnings")
	} else {
		for _, warning := range warnings {
			fmt.Fprintln(w, warning)
		}
	}
	w.Flush()

	return 0
}
