package main

import (
	"testing"
)

func TestParseArgsNoCaptureBeforeCommand(t *testing.T) {
	verbose, configPath, logFile, noCapture, command, subArgs, err := parseArgs([]string{"--no-capture", "new", "-c", "echo hello"})
	if err != nil {
		t.Fatalf("parseArgs error: %v", err)
	}
	if verbose != 0 {
		t.Fatalf("verbose = %d, want 0", verbose)
	}
	if configPath != "" {
		t.Fatalf("configPath = %q, want empty", configPath)
	}
	if logFile != "" {
		t.Fatalf("logFile = %q, want empty", logFile)
	}
	if !noCapture {
		t.Fatal("noCapture = false, want true")
	}
	if command != "new" {
		t.Fatalf("command = %q, want new", command)
	}
	if len(subArgs) != 2 || subArgs[0] != "-c" || subArgs[1] != "echo hello" {
		t.Fatalf("subArgs = %v, want [-c echo hello]", subArgs)
	}
}

func TestParseArgsNoCaptureAfterCommandBecomesShellArg(t *testing.T) {
	verbose, _, _, noCapture, command, subArgs, err := parseArgs([]string{"new", "--no-capture"})
	if err != nil {
		t.Fatalf("parseArgs error: %v", err)
	}
	if verbose != 0 {
		t.Fatalf("verbose = %d, want 0", verbose)
	}
	if noCapture {
		t.Fatal("noCapture = true, want false when placed after subcommand")
	}
	if command != "new" {
		t.Fatalf("command = %q, want new", command)
	}
	if len(subArgs) != 1 || subArgs[0] != "--no-capture" {
		t.Fatalf("subArgs = %v, want [--no-capture]", subArgs)
	}
}

func TestParseArgsVerboseAndConfig(t *testing.T) {
	verbose, configPath, logFile, noCapture, command, subArgs, err := parseArgs([]string{"-vv", "-c", "/tmp/cfg.yaml", "-l", "-", "capture"})
	if err != nil {
		t.Fatalf("parseArgs error: %v", err)
	}
	if verbose != 2 {
		t.Fatalf("verbose = %d, want 2", verbose)
	}
	if configPath != "/tmp/cfg.yaml" {
		t.Fatalf("configPath = %q, want /tmp/cfg.yaml", configPath)
	}
	if logFile != "-" {
		t.Fatalf("logFile = %q, want -", logFile)
	}
	if noCapture {
		t.Fatal("noCapture = true, want false")
	}
	if command != "capture" {
		t.Fatalf("command = %q, want capture", command)
	}
	if len(subArgs) != 0 {
		t.Fatalf("subArgs = %v, want empty", subArgs)
	}
}

func TestParseArgsNoCommand(t *testing.T) {
	_, _, _, _, _, _, err := parseArgs([]string{"-c", "/tmp/cfg.yaml"})
	if err == nil {
		t.Fatal("expected error for missing command")
	}
}
