package cmd

import (
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"flashback-shell/pkg/config"
	"flashback-shell/pkg/log"
)

func TestCheckCmdPrintsConfig(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.yaml")
	if err := os.WriteFile(path, []byte("server_url: http://example.com\n"), 0644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	cfg, src, err := config.Load(path)
	if err != nil {
		t.Fatalf("load config: %v", err)
	}

	logger := log.New(0, io.Discard)

	old := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("pipe: %v", err)
	}
	os.Stdout = w

	code := CheckCmd(cfg, src, logger, nil)

	w.Close()
	os.Stdout = old

	if code != 0 {
		t.Fatalf("CheckCmd returned %d, want 0", code)
	}

	out, err := io.ReadAll(r)
	if err != nil {
		t.Fatalf("read stdout: %v", err)
	}

	if !strings.Contains(string(out), "BINARIES") {
		t.Fatalf("expected BINARIES section, got:\n%s", string(out))
	}
	if !strings.Contains(string(out), "CONFIG") {
		t.Fatalf("expected CONFIG section, got:\n%s", string(out))
	}
	if !strings.Contains(string(out), "WARNINGS") {
		t.Fatalf("expected WARNINGS section, got:\n%s", string(out))
	}
	if !strings.Contains(string(out), "server_url") {
		t.Fatalf("expected config fields, got:\n%s", string(out))
	}
}

func TestCheckCmdReportsMissingBinaries(t *testing.T) {
	// Isolate PATH so tmux/env cannot be found.
	t.Setenv("PATH", t.TempDir())

	dir := t.TempDir()
	path := filepath.Join(dir, "config.yaml")
	if err := os.WriteFile(path, []byte("server_url: http://example.com\n"), 0644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	cfg, src, err := config.Load(path)
	if err != nil {
		t.Fatalf("load config: %v", err)
	}

	logger := log.New(0, io.Discard)

	old := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("pipe: %v", err)
	}
	os.Stdout = w

	code := CheckCmd(cfg, src, logger, nil)

	w.Close()
	os.Stdout = old

	if code != 0 {
		t.Fatalf("CheckCmd returned %d, want 0", code)
	}

	out, err := io.ReadAll(r)
	if err != nil {
		t.Fatalf("read stdout: %v", err)
	}

	if !strings.Contains(string(out), "not found") {
		t.Fatalf("expected 'not found' for missing binaries, got:\n%s", string(out))
	}
	if !strings.Contains(string(out), "tmux is not installed") {
		t.Fatalf("expected tmux warning, got:\n%s", string(out))
	}
	if !strings.Contains(string(out), "env(1) is not installed") {
		t.Fatalf("expected env warning, got:\n%s", string(out))
	}
}
