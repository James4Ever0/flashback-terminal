package shell

import (
	"os/exec"
	"testing"
)

func TestHasTmux(t *testing.T) {
	want := false
	if _, err := exec.LookPath("tmux"); err == nil {
		want = true
	}
	if got := HasTmux(); got != want {
		t.Fatalf("HasTmux() = %v, want %v", got, want)
	}
}

func TestHasEnv(t *testing.T) {
	want := false
	if _, err := exec.LookPath("env"); err == nil {
		want = true
	}
	if got := HasEnv(); got != want {
		t.Fatalf("HasEnv() = %v, want %v", got, want)
	}
}
