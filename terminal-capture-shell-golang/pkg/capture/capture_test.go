package capture

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"flashback-shell/pkg/shell"
)

func TestCaptureAllDetectsChanges(t *testing.T) {
	stateDir := t.TempDir()
	engine := NewEngine(stateDir)

	socketDir := t.TempDir()
	sid := fmt.Sprintf("test%d", time.Now().UnixNano())
	name := "flashback-" + sid
	sock := filepath.Join(socketDir, name)

	// Create detached tmux session
	cmd := exec.Command("tmux", "-S", sock, "new-session", "-d", "-s", name)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("new-session: %v\n%s", err, out)
	}
	defer func() {
		_ = exec.Command("tmux", "-S", sock, "kill-session", "-t", name).Run()
		_ = os.Remove(sock)
	}()

	time.Sleep(200 * time.Millisecond)

	// Send output
	if out, err := exec.Command("tmux", "-S", sock, "send-keys", "-t", name, "echo capturetest", "Enter").CombinedOutput(); err != nil {
		t.Fatalf("send-keys: %v\n%s", err, out)
	}
	time.Sleep(500 * time.Millisecond)

	captures, err := engine.CaptureAll(socketDir)
	if err != nil {
		t.Fatalf("CaptureAll error: %v", err)
	}
	t.Logf("captures: %d", len(captures))
	for _, c := range captures {
		t.Logf("session=%s pane=%s text=%q", c.SessionID, c.PaneID, strings.TrimSpace(c.Text))
	}
	if len(captures) == 0 {
		t.Fatal("expected at least one capture")
	}
}

func TestDiffLines(t *testing.T) {
	cases := []struct {
		name     string
		prev     []string
		curr     []string
		expected []string
	}{
		{
			name:     "no overlap returns full current",
			prev:     []string{"a", "b"},
			curr:     []string{"c", "d"},
			expected: []string{"c", "d"},
		},
		{
			name:     "identical returns empty",
			prev:     []string{"a", "b", "c"},
			curr:     []string{"a", "b", "c"},
			expected: []string{},
		},
		{
			name:     "append at end",
			prev:     []string{"a", "b"},
			curr:     []string{"a", "b", "c", "d"},
			expected: []string{"c", "d"},
		},
		{
			name:     "ring scroll: old lines drop, new lines append",
			prev:     []string{"a", "b", "c", "d"},
			curr:     []string{"c", "d", "e", "f"},
			expected: []string{"e", "f"},
		},
		{
			name:     "partial overlap",
			prev:     []string{"a", "b", "c"},
			curr:     []string{"b", "c", "d"},
			expected: []string{"d"},
		},
		{
			name:     "current shorter than previous",
			prev:     []string{"a", "b", "c", "d"},
			curr:     []string{"c", "d"},
			expected: []string{},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, _ := diffLines(tc.prev, tc.curr)
			if len(got) != len(tc.expected) {
				t.Fatalf("expected %v, got %v", tc.expected, got)
			}
			for i := range got {
				if got[i] != tc.expected[i] {
					t.Fatalf("expected %v, got %v", tc.expected, got)
				}
			}
		})
	}
}

func TestDiffLinesIndex(t *testing.T) {
	cases := []struct {
		name            string
		prev            []string
		curr            []string
		expectedLines   []string
		expectedIndices []int
	}{
		{
			name:            "identical returns empty",
			prev:            []string{"a", "b", "c"},
			curr:            []string{"a", "b", "c"},
			expectedLines:   []string{},
			expectedIndices: nil,
		},
		{
			name:            "append: pad previous and mark all current as new",
			prev:            []string{"a", "b"},
			curr:            []string{"a", "b", "c", "d"},
			expectedLines:   []string{"a", "b", "c", "d"},
			expectedIndices: []int{0, 1, 2, 3},
		},
		{
			name:            "equal length ring scroll reports every line changed",
			prev:            []string{"a", "b", "c", "d"},
			curr:            []string{"c", "d", "e", "f"},
			expectedLines:   []string{"c", "d", "e", "f"},
			expectedIndices: []int{0, 1, 2, 3},
		},
		{
			name:            "in-place middle change",
			prev:            []string{"a", "b", "c", "d"},
			curr:            []string{"a", "b", "X", "d"},
			expectedLines:   []string{"X"},
			expectedIndices: []int{2},
		},
		{
			name:            "current shorter than previous",
			prev:            []string{"a", "b", "c", "d"},
			curr:            []string{"c", "d"},
			expectedLines:   []string{},
			expectedIndices: nil,
		},
		{
			name:            "equal length shift reports every line changed",
			prev:            []string{"a", "b", "c"},
			curr:            []string{"b", "c", "d"},
			expectedLines:   []string{"b", "c", "d"},
			expectedIndices: []int{0, 1, 2},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			gotLines, gotIndices := diffLinesIndex(tc.prev, tc.curr)
			if len(gotLines) != len(tc.expectedLines) {
				t.Fatalf("expected lines %v, got %v", tc.expectedLines, gotLines)
			}
			for i := range gotLines {
				if gotLines[i] != tc.expectedLines[i] {
					t.Fatalf("expected lines %v, got %v", tc.expectedLines, gotLines)
				}
			}
			if len(gotIndices) != len(tc.expectedIndices) {
				t.Fatalf("expected indices %v, got %v", tc.expectedIndices, gotIndices)
			}
			for i := range gotIndices {
				if gotIndices[i] != tc.expectedIndices[i] {
					t.Fatalf("expected indices %v, got %v", tc.expectedIndices, gotIndices)
				}
			}
		})
	}
}

func TestSessionIsRunning(t *testing.T) {
	socketDir := t.TempDir()
	sid := fmt.Sprintf("run%d", time.Now().UnixNano())
	name := "flashback-" + sid
	sock := filepath.Join(socketDir, name)

	cmd := exec.Command("tmux", "-S", sock, "new-session", "-d", "-s", name)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("new-session: %v\n%s", err, out)
	}
	defer func() {
		_ = exec.Command("tmux", "-S", sock, "kill-session", "-t", name).Run()
		_ = os.Remove(sock)
	}()

	sess := &shell.TmuxSession{Name: name, SocketPath: sock}
	if !sess.IsRunning() {
		t.Fatal("expected session to be running")
	}
}

func TestTextOnlyCapture(t *testing.T) {
	stateDir := t.TempDir()
	engine := NewEngine(stateDir)
	engine.TextOnly = true

	socketDir := t.TempDir()
	sid := fmt.Sprintf("textonly%d", time.Now().UnixNano())
	name := "flashback-" + sid
	sock := filepath.Join(socketDir, name)

	cmd := exec.Command("tmux", "-S", sock, "new-session", "-d", "-s", name)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("new-session: %v\n%s", err, out)
	}
	defer func() {
		_ = exec.Command("tmux", "-S", sock, "kill-session", "-t", name).Run()
		_ = os.Remove(sock)
	}()

	time.Sleep(200 * time.Millisecond)

	if out, err := exec.Command("tmux", "-S", sock, "send-keys", "-t", name, "echo textonlytest", "Enter").CombinedOutput(); err != nil {
		t.Fatalf("send-keys: %v\n%s", err, out)
	}
	time.Sleep(500 * time.Millisecond)

	captures, err := engine.CaptureAll(socketDir)
	if err != nil {
		t.Fatalf("CaptureAll error: %v", err)
	}
	if len(captures) == 0 {
		t.Fatal("expected at least one capture")
	}

	c := captures[0]
	if c.ANSI != "" {
		t.Fatalf("expected empty ANSI in text-only mode, got %q", c.ANSI)
	}
	if strings.TrimSpace(c.Text) == "" {
		t.Fatal("expected non-empty text capture")
	}
	if c.Metadata == nil || c.Metadata["ansi"] != "false" {
		t.Fatalf("expected metadata ansi=false, got %v", c.Metadata)
	}
}
