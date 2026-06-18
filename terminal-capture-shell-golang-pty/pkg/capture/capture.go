package capture

import (
	"crypto/md5"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"flashback-shell-pty/pkg/session"
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

// Engine captures terminal content from all managed sessions.
type Engine struct {
	stateDir string
	// Options configured from the outside.
	DiffOnly   bool
	DiffMode   string // "suffix" (default) or "index"
	TextOnly   bool   // when true, send plain text only and omit ANSI
}

// NewEngine creates a capture engine.
func NewEngine(stateDir string) *Engine {
	return &Engine{stateDir: stateDir}
}

// CaptureAll discovers all sessions and captures changed screens.
func (e *Engine) CaptureAll(socketDir string, scrollback bool) ([]Capture, error) {
	sessions, err := session.DiscoverSessions(socketDir)
	if err != nil {
		return nil, err
	}

	var results []Capture
	for _, si := range sessions {
		c, err := e.CaptureSession(si.SocketPath, si.SessionID, scrollback)
		if err != nil {
			continue // skip dead sessions
		}
		if c != nil {
			results = append(results, *c)
		}
	}
	return results, nil
}

// CaptureSession captures a single session screen.
func (e *Engine) CaptureSession(socketPath, sessionID string, scrollback bool) (*Capture, error) {
	client := session.NewClient(socketPath)
	payload, err := client.Capture(scrollback)
	if err != nil {
		return nil, err
	}

	ansi := payload.ANSI
	text := payload.Text

	if e.TextOnly {
		ansi = ""
	}

	hashInput := ansi
	if e.TextOnly {
		hashInput = text
	}
	hash := fmt.Sprintf("%x", md5.Sum([]byte(hashInput)))
	if e.isDuplicate(sessionID, "main", hash) {
		return nil, nil
	}

	captureANSI := ansi
	captureText := text

	if e.DiffOnly {
		prevText, err := e.loadPrev(sessionID, "main")
		if err == nil && len(prevText) > 0 {
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
				_ = e.savePrev(sessionID, "main", text)
				return nil, nil
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
		_ = e.savePrev(sessionID, "main", text)
	}

	var metadata map[string]string
	if e.TextOnly {
		metadata = map[string]string{"ansi": "false"}
	}

	return &Capture{
		SessionID: sessionID,
		PaneID:    "main",
		Target:    "main",
		ANSI:      captureANSI,
		Text:      captureText,
		Hash:      hash,
		Cols:      payload.Cols,
		Rows:      payload.Rows,
		Timestamp: time.Now().UTC(),
		Metadata:  metadata,
	}, nil
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

func (e *Engine) prevPath(sessionID, paneID string) string {
	key := hashKey(sessionID, paneID)
	return filepath.Join(e.stateDir, key+".prev")
}

func (e *Engine) loadPrev(sessionID, paneID string) (string, error) {
	data, err := os.ReadFile(e.prevPath(sessionID, paneID))
	if err != nil {
		return "", err
	}
	return string(data), nil
}

func (e *Engine) savePrev(sessionID, paneID, text string) error {
	if err := os.MkdirAll(e.stateDir, 0755); err != nil {
		return err
	}
	return os.WriteFile(e.prevPath(sessionID, paneID), []byte(text), 0644)
}

// SaveHashes writes the last-seen hashes for captured sessions to disk.
func (e *Engine) SaveHashes(captures []Capture) error {
	if err := os.MkdirAll(e.stateDir, 0755); err != nil {
		return err
	}
	for _, c := range captures {
		key := hashKey(c.SessionID, c.PaneID)
		path := filepath.Join(e.stateDir, key+".hash")
		if err := os.WriteFile(path, []byte(c.Hash), 0644); err != nil {
			return err
		}
	}
	return nil
}

// IsDuplicate reports whether the given hash matches the last known hash.
func (e *Engine) IsDuplicate(sessionID, paneID, hash string) bool {
	return e.isDuplicate(sessionID, paneID, hash)
}

func (e *Engine) isDuplicate(sessionID, paneID, hash string) bool {
	key := hashKey(sessionID, paneID)
	path := filepath.Join(e.stateDir, key+".hash")
	data, err := os.ReadFile(path)
	if err != nil {
		return false
	}
	return string(data) == hash
}

func hashKey(sessionID, paneID string) string {
	return fmt.Sprintf("%s_%s", sessionID, paneID)
}
