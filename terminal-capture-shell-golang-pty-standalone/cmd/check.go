package cmd

import (
	"flag"
	"fmt"
	"os"
	"os/exec"

	"flashback-shell-pty/pkg/config"
	"flashback-shell-pty/pkg/log"
	"flashback-shell-pty/pkg/ptywrap"
)

// CheckCmd handles the "check" subcommand.
func CheckCmd(cfg *config.Config, src *config.ConfigWithSource, logger *log.Logger, args []string) int {
	fs := flag.NewFlagSet("check", flag.ExitOnError)
	_, err := config.ParseSubcommandFlags(fs, args)
	if err != nil {
		logger.Errorf("failed to parse flags: %v", err)
		return 1
	}

	fmt.Println("flashback-shell-pty configuration:")
	fmt.Printf("  server_url:          %s (%s)\n", src.ServerURL.Value, src.ServerURL.Origin)
	fmt.Printf("  shell:               %s (%s)\n", src.Shell.Value, src.Shell.Origin)
	fmt.Printf("  buffer_size:         %s (%s)\n", src.BufferSize.Value, src.BufferSize.Origin)
	fmt.Printf("  buffer_mode:         %s (%s)\n", src.BufferMode.Value, src.BufferMode.Origin)
	fmt.Printf("  buffer_dir:          %s (%s)\n", src.BufferDir.Value, src.BufferDir.Origin)
	fmt.Printf("  device_id:           %s (%s)\n", src.DeviceID.Value, src.DeviceID.Origin)
	fmt.Printf("  capture_interval:    %s (%s)\n", src.CaptureInterval.Value, src.CaptureInterval.Origin)
	fmt.Printf("  first_capture_delay: %s (%s)\n", src.FirstCaptureDelay.Value, src.FirstCaptureDelay.Origin)
	fmt.Printf("  disable_capture:     %s (%s)\n", src.DisableCapture.Value, src.DisableCapture.Origin)
	fmt.Printf("  diff_only:           %s (%s)\n", src.DiffOnly.Value, src.DiffOnly.Origin)
	fmt.Printf("  diff_mode:           %s (%s)\n", src.DiffMode.Value, src.DiffMode.Origin)
	fmt.Printf("  text_only:           %s (%s)\n", src.TextOnly.Value, src.TextOnly.Origin)
	fmt.Printf("  scrollback_lines:    %s (%s)\n", src.ScrollbackLines.Value, src.ScrollbackLines.Origin)
	fmt.Printf("  allow_nested:        %s (%s)\n", src.AllowNested.Value, src.AllowNested.Origin)

	fmt.Println()
	fmt.Println("Validation:")

	shellBin := cfg.Shell
	if shellBin == "" {
		shellBin = os.Getenv("SHELL")
		if shellBin == "" {
			shellBin = "/bin/bash"
		}
	}
	if _, err := os.Stat(shellBin); err != nil {
		fmt.Printf("  shell:             FAIL (%s not found: %v)\n", shellBin, err)
		return 1
	}
	if _, err := exec.LookPath(shellBin); err != nil {
		fmt.Printf("  shell:             FAIL (%s not executable: %v)\n", shellBin, err)
		return 1
	}
	fmt.Printf("  shell:             OK (%s)\n", shellBin)

	// Try to open a PTY with a trivial command.
	testCmd := exec.Command(shellBin, "-c", "exit 0")
	ptmx, err := ptywrap.Start(testCmd)
	if err != nil {
		fmt.Printf("  pty:               FAIL (%v)\n", err)
		return 1
	}
	_ = ptmx.Close()
	_ = testCmd.Wait()
	fmt.Println("  pty:               OK")

	return 0
}
