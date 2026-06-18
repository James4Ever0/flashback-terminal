package capture

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"time"
)

// Buffer stores failed uploads for retry.
type Buffer struct {
	dir     string
	maxSize int
}

// NewBuffer creates a local retry buffer.
func NewBuffer(dir string, maxSize int) *Buffer {
	return &Buffer{dir: dir, maxSize: maxSize}
}

// Add stores captures that failed to upload.
func (b *Buffer) Add(captures []Capture) error {
	if err := os.MkdirAll(b.dir, 0755); err != nil {
		return err
	}

	entry := bufferEntry{
		Timestamp: time.Now().UTC(),
		Captures:  captures,
	}
	data, err := json.Marshal(entry)
	if err != nil {
		return err
	}

	name := fmt.Sprintf("retry_%d.json", time.Now().UnixNano())
	path := filepath.Join(b.dir, name)
	if err := os.WriteFile(path, data, 0644); err != nil {
		return err
	}

	return b.trim()
}

// PopAll retrieves and removes all buffered entries.
func (b *Buffer) PopAll() ([][]Capture, error) {
	entries, err := os.ReadDir(b.dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}

	sort.Slice(entries, func(i, j int) bool {
		return entries[i].Name() < entries[j].Name()
	})

	var result [][]Capture
	for _, entry := range entries {
		if entry.IsDir() || filepath.Ext(entry.Name()) != ".json" {
			continue
		}
		path := filepath.Join(b.dir, entry.Name())
		data, err := os.ReadFile(path)
		if err != nil {
			continue
		}
		var be bufferEntry
		if err := json.Unmarshal(data, &be); err != nil {
			os.Remove(path)
			continue
		}
		result = append(result, be.Captures)
		os.Remove(path)
	}
	return result, nil
}

// Count returns the number of buffered entries.
func (b *Buffer) Count() int {
	entries, err := os.ReadDir(b.dir)
	if err != nil {
		return 0
	}
	count := 0
	for _, e := range entries {
		if !e.IsDir() && filepath.Ext(e.Name()) == ".json" {
			count++
		}
	}
	return count
}

func (b *Buffer) trim() error {
	entries, err := os.ReadDir(b.dir)
	if err != nil {
		return err
	}
	sort.Slice(entries, func(i, j int) bool {
		return entries[i].Name() > entries[j].Name() // oldest first
	})
	for len(entries) > b.maxSize {
		os.Remove(filepath.Join(b.dir, entries[0].Name()))
		entries = entries[1:]
	}
	return nil
}

type bufferEntry struct {
	Timestamp time.Time `json:"timestamp"`
	Captures  []Capture `json:"captures"`
}
