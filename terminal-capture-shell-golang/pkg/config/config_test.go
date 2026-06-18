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
	// Write a config file that sets values different from the env vars we will set.
	dir := t.TempDir()
	path := filepath.Join(dir, "config.yaml")
	content := `
server_url: "http://from-config"
socket_dir: "~/.flashback-shell/tmux-from-config"
shell: "/bin/from-config"
buffer_size: 5
device_id: "config-device"
capture_interval: 60
disable_capture: true
capture_scrollback: true
allow_nested_tmux: true
diff_only: true
diff_mode: "index"
text_only: true
`
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	t.Setenv("FLASHBACK_SHELL_SERVER_URL", "http://from-env")
	t.Setenv("FLASHBACK_SHELL_SOCKET_DIR", "~/.flashback-shell/tmux-from-env")
	t.Setenv("FLASHBACK_SHELL_SHELL", "/bin/from-env")
	t.Setenv("FLASHBACK_SHELL_BUFFER_SIZE", "42")
	t.Setenv("FLASHBACK_SHELL_DEVICE_ID", "env-device")
	t.Setenv("FLASHBACK_SHELL_CAPTURE_INTERVAL", "10")
	t.Setenv("FLASHBACK_SHELL_DISABLE_CAPTURE", "false")
	t.Setenv("FLASHBACK_SHELL_CAPTURE_SCROLLBACK", "false")
	t.Setenv("FLASHBACK_SHELL_ALLOW_NESTED_TMUX", "false")
	t.Setenv("FLASHBACK_SHELL_DIFF_ONLY", "false")
	t.Setenv("FLASHBACK_SHELL_DIFF_MODE", "suffix")
	t.Setenv("FLASHBACK_SHELL_TEXT_ONLY", "false")

	cfg, src, err := Load(path)
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatalf("home dir: %v", err)
	}

	checks := []struct {
		name   string
		got    string
		origin string
		want   string
	}{
		{"server_url", cfg.ServerURL, src.ServerURL.Origin, "http://from-env"},
		{"socket_dir", cfg.SocketDir, src.SocketDir.Origin, filepath.Join(home, ".flashback-shell", "tmux-from-env")},
		{"shell", cfg.Shell, src.Shell.Origin, "/bin/from-env"},
		{"buffer_size", fmt.Sprintf("%d", cfg.BufferSize), src.BufferSize.Origin, "42"},
		{"device_id", cfg.DeviceID, src.DeviceID.Origin, "env-device"},
		{"capture_interval", fmt.Sprintf("%d", cfg.CaptureInterval), src.CaptureInterval.Origin, "10"},
		{"disable_capture", fmt.Sprintf("%t", cfg.DisableCapture), src.DisableCapture.Origin, "false"},
		{"capture_scrollback", fmt.Sprintf("%t", cfg.CaptureScrollback), src.CaptureScrollback.Origin, "false"},
		{"allow_nested_tmux", fmt.Sprintf("%t", cfg.AllowNestedTmux), src.AllowNestedTmux.Origin, "false"},
		{"diff_only", fmt.Sprintf("%t", cfg.DiffOnly), src.DiffOnly.Origin, "false"},
		{"diff_mode", cfg.DiffMode, src.DiffMode.Origin, "suffix"},
		{"text_only", fmt.Sprintf("%t", cfg.TextOnly), src.TextOnly.Origin, "false"},
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
