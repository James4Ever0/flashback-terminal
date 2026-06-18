package config

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

// TestEnvOverridesConfigFile verifies that every config option can be overridden
// by its matching environment variable, including overriding a config-file true
// back to false.
func TestEnvOverridesConfigFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.yaml")
	content := `
server_url: "http://from-config"
shell: "/bin/from-config"
buffer_size: 5
buffer_mode: "disk"
buffer_dir: "/from/config"
device_id: "config-device"
capture_interval: 60
first_capture_delay: 5
disable_capture: true
diff_only: true
diff_mode: "index"
text_only: true
scrollback_lines: 500
allow_nested: true
`
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	t.Setenv("FLASHBACK_SHELL_PTY_SERVER_URL", "http://from-env")
	t.Setenv("FLASHBACK_SHELL_PTY_SHELL", "/bin/from-env")
	t.Setenv("FLASHBACK_SHELL_PTY_BUFFER_SIZE", "42")
	t.Setenv("FLASHBACK_SHELL_PTY_BUFFER_MODE", "memory")
	t.Setenv("FLASHBACK_SHELL_PTY_BUFFER_DIR", "/from/env")
	t.Setenv("FLASHBACK_SHELL_PTY_DEVICE_ID", "env-device")
	t.Setenv("FLASHBACK_SHELL_PTY_CAPTURE_INTERVAL", "10")
	t.Setenv("FLASHBACK_SHELL_PTY_FIRST_CAPTURE_DELAY", "2")
	t.Setenv("FLASHBACK_SHELL_PTY_DISABLE_CAPTURE", "false")
	t.Setenv("FLASHBACK_SHELL_PTY_DIFF_ONLY", "false")
	t.Setenv("FLASHBACK_SHELL_PTY_DIFF_MODE", "suffix")
	t.Setenv("FLASHBACK_SHELL_PTY_TEXT_ONLY", "false")
	t.Setenv("FLASHBACK_SHELL_PTY_SCROLLBACK_LINES", "100")
	t.Setenv("FLASHBACK_SHELL_PTY_ALLOW_NESTED", "false")

	cfg, src, err := Load(path)
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	checks := []struct {
		name   string
		got    string
		origin string
		want   string
	}{
		{"server_url", cfg.ServerURL, src.ServerURL.Origin, "http://from-env"},
		{"shell", cfg.Shell, src.Shell.Origin, "/bin/from-env"},
		{"buffer_size", fmt.Sprintf("%d", cfg.BufferSize), src.BufferSize.Origin, "42"},
		{"buffer_mode", cfg.BufferMode, src.BufferMode.Origin, "memory"},
		{"buffer_dir", cfg.BufferDir, src.BufferDir.Origin, "/from/env"},
		{"device_id", cfg.DeviceID, src.DeviceID.Origin, "env-device"},
		{"capture_interval", fmt.Sprintf("%d", cfg.CaptureInterval), src.CaptureInterval.Origin, "10"},
		{"first_capture_delay", fmt.Sprintf("%d", cfg.FirstCaptureDelay), src.FirstCaptureDelay.Origin, "2"},
		{"disable_capture", fmt.Sprintf("%t", cfg.DisableCapture), src.DisableCapture.Origin, "false"},
		{"diff_only", fmt.Sprintf("%t", cfg.DiffOnly), src.DiffOnly.Origin, "false"},
		{"diff_mode", cfg.DiffMode, src.DiffMode.Origin, "suffix"},
		{"text_only", fmt.Sprintf("%t", cfg.TextOnly), src.TextOnly.Origin, "false"},
		{"scrollback_lines", fmt.Sprintf("%d", cfg.ScrollbackLines), src.ScrollbackLines.Origin, "100"},
		{"allow_nested", fmt.Sprintf("%t", cfg.AllowNested), src.AllowNested.Origin, "false"},
	}

	for _, tc := range checks {
		t.Run(tc.name, func(t *testing.T) {
			if tc.got != tc.want {
				t.Fatalf("value: got %q, want %q", tc.got, tc.want)
			}
			if tc.origin != "env" {
				t.Fatalf("source: got %q, want env", tc.origin)
			}
		})
	}
}
