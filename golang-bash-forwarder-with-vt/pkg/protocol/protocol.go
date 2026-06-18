// Package protocol implements the line-delimited JSON-RPC-like event encoding.
package protocol

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"time"

	"bash-forwarder-vt/pkg/event"
)

// Envelope is the JSON wire format. The Method field discriminates the event.
type Envelope struct {
	Method string          `json:"method"`
	ID     uint64          `json:"id"`
	Time   time.Duration   `json:"time"`
	Params json.RawMessage `json:"params"`
}

// Encode serializes an event into one JSON line including a trailing newline.
func Encode(ev interface{}) ([]byte, error) {
	var env Envelope
	switch e := ev.(type) {
	case *event.InitEvent:
		env = Envelope{Method: "init", ID: e.ID, Time: e.Time}
		env.Params, _ = json.Marshal(struct {
			Cols   uint16 `json:"cols"`
			Rows   uint16 `json:"rows"`
			Screen string `json:"screen"`
		}{Cols: e.Cols, Rows: e.Rows, Screen: e.Screen})
	case *event.OutputEvent:
		env = Envelope{Method: "output", ID: e.ID, Time: e.Time}
		env.Params, _ = json.Marshal(struct {
			Data string `json:"data"`
		}{Data: e.Data})
	case *event.InputEvent:
		env = Envelope{Method: "input", ID: e.ID, Time: e.Time}
		env.Params, _ = json.Marshal(struct {
			Data string `json:"data"`
		}{Data: e.Data})
	case *event.ResizeEvent:
		env = Envelope{Method: "resize", ID: e.ID, Time: e.Time}
		env.Params, _ = json.Marshal(struct {
			Cols uint16 `json:"cols"`
			Rows uint16 `json:"rows"`
		}{Cols: e.Cols, Rows: e.Rows})
	case *event.MarkerEvent:
		env = Envelope{Method: "marker", ID: e.ID, Time: e.Time}
		env.Params, _ = json.Marshal(struct {
			Label string `json:"label"`
		}{Label: e.Label})
	case *event.ExitEvent:
		env = Envelope{Method: "exit", ID: e.ID, Time: e.Time}
		env.Params, _ = json.Marshal(struct {
			Status int `json:"status"`
		}{Status: e.Status})
	case *event.EofEvent:
		env = Envelope{Method: "eof", ID: e.ID, Time: e.Time}
		env.Params = json.RawMessage("{}")
	default:
		return nil, fmt.Errorf("unknown event type %T", ev)
	}
	buf, err := json.Marshal(env)
	if err != nil {
		return nil, err
	}
	return append(buf, '\n'), nil
}

// Decode parses one JSON line into an event value.
func Decode(line []byte) (interface{}, error) {
	line = bytes.TrimRight(line, "\r\n")
	if len(line) == 0 {
		return nil, nil
	}
	var env Envelope
	if err := json.Unmarshal(line, &env); err != nil {
		return nil, err
	}
	switch env.Method {
	case "init":
		var p struct {
			Cols   uint16 `json:"cols"`
			Rows   uint16 `json:"rows"`
			Screen string `json:"screen"`
		}
		if err := json.Unmarshal(env.Params, &p); err != nil {
			return nil, err
		}
		return &event.InitEvent{Event: event.Event{ID: env.ID, Time: env.Time}, Cols: p.Cols, Rows: p.Rows, Screen: p.Screen}, nil
	case "output":
		var p struct{ Data string `json:"data"` }
		if err := json.Unmarshal(env.Params, &p); err != nil {
			return nil, err
		}
		return &event.OutputEvent{Event: event.Event{ID: env.ID, Time: env.Time}, Data: p.Data}, nil
	case "input":
		var p struct{ Data string `json:"data"` }
		if err := json.Unmarshal(env.Params, &p); err != nil {
			return nil, err
		}
		return &event.InputEvent{Event: event.Event{ID: env.ID, Time: env.Time}, Data: p.Data}, nil
	case "resize":
		var p struct {
			Cols uint16 `json:"cols"`
			Rows uint16 `json:"rows"`
		}
		if err := json.Unmarshal(env.Params, &p); err != nil {
			return nil, err
		}
		return &event.ResizeEvent{Event: event.Event{ID: env.ID, Time: env.Time}, Cols: p.Cols, Rows: p.Rows}, nil
	case "marker":
		var p struct{ Label string `json:"label"` }
		if err := json.Unmarshal(env.Params, &p); err != nil {
			return nil, err
		}
		return &event.MarkerEvent{Event: event.Event{ID: env.ID, Time: env.Time}, Label: p.Label}, nil
	case "exit":
		var p struct{ Status int `json:"status"` }
		if err := json.Unmarshal(env.Params, &p); err != nil {
			return nil, err
		}
		return &event.ExitEvent{Event: event.Event{ID: env.ID, Time: env.Time}, Status: p.Status}, nil
	case "eof":
		return &event.EofEvent{Event: event.Event{ID: env.ID, Time: env.Time}}, nil
	}
	return nil, fmt.Errorf("unknown method %q", env.Method)
}

// EncodeBytes is a helper that base64-encodes raw bytes.
func EncodeBytes(b []byte) string {
	return base64.StdEncoding.EncodeToString(b)
}

// DecodeBytes is a helper that base64-decodes a string.
func DecodeBytes(s string) ([]byte, error) {
	return base64.StdEncoding.DecodeString(s)
}
