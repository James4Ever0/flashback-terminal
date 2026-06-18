package capture

import (
	"crypto/md5"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"flashback-shell-pty/pkg/config"
	"flashback-shell-pty/pkg/vtcapture"
)

// Capture holds the captured content of a single session screen.
type Capture struct {
	SessionID string            `json:"session_id"`
	PaneID    string            `json:"pane_id"`
	Target    string            `json:"target"`
	ANSI      string            `json:"ansi"`
	Text      string            `json:"text"`
	Hash      string            `json:"hash"`
	Cols      int               `json:"cols"`
	Rows      int               `json:"rows"`
	Timestamp time.Time         `json:"timestamp"`
	Metadata  map[string]string `json:"metadata,omitempty"`
}

// Engine captures terminal content from a single VT emulator.
type Engine struct {
	stateDir string
	// Options configured from the outside.
	DiffOnly bool
	DiffMode string // "suffix" (default) or "index"
	TextOnly bool   // when true, send plain text only and omit ANSI

	// In-memory state for the lifetime of the engine.
	lastHash string
	lastText string
}

// NewEngine creates a capture engine.
// stateDir is optional; when empty, previous-snapshot deduplication/diff state
// is kept only in memory for the lifetime of the process.
func NewEngine(stateDir string) *Engine {
	return &Engine{stateDir: stateDir}
}

// CaptureScreen captures the current VT screen. It returns nil when the screen
// is unchanged (according to the last hash) or empty.
func (e *Engine) CaptureScreen(vtTerm *vtcapture.Terminal, cfg *config.Config) *Capture {
	ansi := string(vtTerm.CaptureANSI())
	text := string(vtTerm.CaptureText())

	if e.TextOnly {
		ansi = ""
	}

	hashInput := ansi
	if e.TextOnly {
		hashInput = text
	}
	if hashInput == "" {
		return nil
	}

	hash := fmt.Sprintf("%x", md5.Sum([]byte(hashInput)))
	if e.isDuplicate(hash) {
		return nil
	}
	e.lastHash = hash
	_ = e.saveHash(hash)

	captureANSI := ansi
	captureText := text

	if e.DiffOnly {
		prevText := e.lastText
		if prevText == "" && e.stateDir != "" {
			prevText, _ = e.loadPrev()
		}
		if len(prevText) > 0 {
			prevLines := strings.Split(prevText, "\n")
			currLines := strings.Split(text, "\n")

			var diff []string
			var indices []int
			switch strings.ToLower(e.DiffMode) {
			case "index":
				diff, indices = diffLinesIndex(prevLines, currLines)
			default:
				diff, indices = diffLines(prevLines, currLines)
			}

			if len(diff) == 0 {
				e.lastText = text
				_ = e.savePrev(text)
				return nil
			}

			captureText = strings.Join(diff, "\n")
			ansiLines := strings.Split(ansi, "\n")
			if len(ansiLines) == len(currLines) {
				selected := make([]string, 0, len(indices))
				for _, idx := range indices {
					selected = append(selected, ansiLines[idx])
				}
				captureANSI = strings.Join(selected, "\n")
			}
		}
		e.lastText = text
		_ = e.savePrev(text)
	}

	var metadata map[string]string
	if e.TextOnly {
		metadata = map[string]string{"ansi": "false"}
	}

	cols, rows := vtTerm.Size()
	return &Capture{
		SessionID: "main",
		PaneID:    "main",
		Target:    "main",
		ANSI:      captureANSI,
		Text:      captureText,
		Hash:      hash,
		Cols:      cols,
		Rows:      rows,
		Timestamp: time.Now().UTC(),
		Metadata:  metadata,
	}
}

// diffLines returns the suffix of curr that did not exist as a contiguous
// block at the end of prev.
func diffLines(prev, curr []string) ([]string, []int) {
	max := len(prev)
	if len(curr) < max {
		max = len(curr)
	}
	for i := max; i > 0; i-- {
		if slices.Equal(prev[len(prev)-i:], curr[:i]) {
			return curr[i:], makeRange(i, len(curr))
		}
	}
	return curr, makeRange(0, len(curr))
}

// diffLinesIndex aligns prev to the length of curr and returns every line that
// differs at the same vertical position.
func diffLinesIndex(prev, curr []string) ([]string, []int) {
	aligned := make([]string, len(curr))
	switch {
	case len(prev) > len(curr):
		copy(aligned, prev[len(prev)-len(curr):])
	case len(prev) < len(curr):
		copy(aligned[len(curr)-len(prev):], prev)
	default:
		copy(aligned, prev)
	}

	var diff []string
	var indices []int
	for i, line := range curr {
		if line != aligned[i] {
			diff = append(diff, line)
			indices = append(indices, i)
		}
	}
	return diff, indices
}

func makeRange(start, end int) []int {
	if start >= end {
		return nil
	}
	r := make([]int, end-start)
	for i := range r {
		r[i] = start + i
	}
	return r
}

func (e *Engine) prevPath() string {
	return filepath.Join(e.stateDir, "main.prev")
}

func (e *Engine) hashPath() string {
	return filepath.Join(e.stateDir, "main.hash")
}

func (e *Engine) loadPrev() (string, error) {
	if e.stateDir == "" {
		return "", os.ErrNotExist
	}
	data, err := os.ReadFile(e.prevPath())
	if err != nil {
		return "", err
	}
	return string(data), nil
}

func (e *Engine) savePrev(text string) error {
	if e.stateDir == "" {
		return nil
	}
	if err := os.MkdirAll(e.stateDir, 0755); err != nil {
		return err
	}
	return os.WriteFile(e.prevPath(), []byte(text), 0644)
}

func (e *Engine) saveHash(hash string) error {
	if e.stateDir == "" {
		return nil
	}
	if err := os.MkdirAll(e.stateDir, 0755); err != nil {
		return err
	}
	return os.WriteFile(e.hashPath(), []byte(hash), 0644)
}

func (e *Engine) isDuplicate(hash string) bool {
	if hash != "" && hash == e.lastHash {
		return true
	}
	if e.stateDir == "" {
		return false
	}
	data, err := os.ReadFile(e.hashPath())
	if err != nil {
		return false
	}
	return string(data) == hash
}
