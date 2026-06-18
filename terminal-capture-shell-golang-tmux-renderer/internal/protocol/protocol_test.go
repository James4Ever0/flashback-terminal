package protocol

import (
	"bytes"
	"testing"
	"time"

	"flashback-shell-tmux-renderer/internal/event"
)

func TestEncodeDecode(t *testing.T) {
	cases := []struct {
		name string
		ev   interface{}
	}{
		{
			name: "output",
			ev:   &event.OutputEvent{Event: event.Event{ID: 1, Time: time.Microsecond}, Data: EncodeBytes([]byte("hello"))},
		},
		{
			name: "input",
			ev:   &event.InputEvent{Event: event.Event{ID: 2, Time: 2 * time.Microsecond}, Data: EncodeBytes([]byte("x"))},
		},
		{
			name: "resize",
			ev:   &event.ResizeEvent{Event: event.Event{ID: 3, Time: 3 * time.Microsecond}, Cols: 80, Rows: 24},
		},
		{
			name: "exit",
			ev:   &event.ExitEvent{Event: event.Event{ID: 4, Time: 4 * time.Microsecond}, Status: 42},
		},
		{
			name: "eof",
			ev:   &event.EofEvent{Event: event.Event{ID: 5, Time: 5 * time.Microsecond}},
		},
		{
			name: "init",
			ev:   &event.InitEvent{Event: event.Event{ID: 0, Time: 0}, Cols: 80, Rows: 24, Screen: ""},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			data, err := Encode(tc.ev)
			if err != nil {
				t.Fatalf("encode: %v", err)
			}
			if !bytes.HasSuffix(data, []byte("\n")) {
				t.Fatalf("encoded data missing trailing newline: %q", data)
			}
			decoded, err := Decode(data)
			if err != nil {
				t.Fatalf("decode: %v", err)
			}
			if decoded == nil {
				t.Fatalf("decoded nil event")
			}
		})
	}
}

func TestDecodeBytes(t *testing.T) {
	original := []byte("hello\x00world")
	encoded := EncodeBytes(original)
	decoded, err := DecodeBytes(encoded)
	if err != nil {
		t.Fatalf("decode bytes: %v", err)
	}
	if !bytes.Equal(decoded, original) {
		t.Fatalf("decoded bytes mismatch: got %q, want %q", decoded, original)
	}
}
