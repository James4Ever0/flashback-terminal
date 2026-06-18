// Package event defines the typed event bus used by the forwarder.
package event

import (
	"sync"
	"time"
)

// Event is the base fields shared by all events.
type Event struct {
	ID   uint64        `json:"id"`
	Time time.Duration `json:"time"` // elapsed microseconds since session start
}

// OutputEvent carries terminal output bytes as a base64 string.
type OutputEvent struct {
	Event
	Data string `json:"data"`
}

// InputEvent carries terminal input bytes as a base64 string.
type InputEvent struct {
	Event
	Data string `json:"data"`
}

// ResizeEvent carries a terminal size change.
type ResizeEvent struct {
	Event
	Cols uint16 `json:"cols"`
	Rows uint16 `json:"rows"`
}

// MarkerEvent carries a user-defined marker label.
type MarkerEvent struct {
	Event
	Label string `json:"label"`
}

// ExitEvent carries the shell exit status.
type ExitEvent struct {
	Event
	Status int `json:"status"`
}

// EofEvent signals end of stream.
type EofEvent struct {
	Event
}

// InitEvent is sent to new subscribers so they can start from current state.
type InitEvent struct {
	Event
	Cols   uint16 `json:"cols"`
	Rows   uint16 `json:"rows"`
	Screen string `json:"screen"` // base64 snapshot (optional, may be empty)
}

// Bus broadcasts typed events to multiple subscribers.
type Bus struct {
	mu          sync.RWMutex
	subs        map[uint64]chan interface{}
	nextSubID   uint64
	lastID      uint64
	start       time.Time
	screenCols  uint16
	screenRows  uint16
	screen      []byte
}

// NewBus creates a new event bus.
func NewBus() *Bus {
	return &Bus{
		subs:  make(map[uint64]chan interface{}),
		start: time.Now(),
	}
}

// SetScreen updates the current screen snapshot stored for InitEvent.
func (b *Bus) SetScreen(cols, rows uint16, screen []byte) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.screenCols = cols
	b.screenRows = rows
	if screen != nil {
		b.screen = append([]byte(nil), screen...)
	}
}

// NextID returns the next unique event id and elapsed time.
func (b *Bus) NextID() (uint64, time.Duration) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.lastID++
	return b.lastID, time.Since(b.start)
}

// Publish sends an event to all current subscribers.
func (b *Bus) Publish(ev interface{}) {
	b.mu.RLock()
	defer b.mu.RUnlock()
	for _, ch := range b.subs {
		// Non-blocking send; slow consumers drop events.
		select {
		case ch <- ev:
		default:
		}
	}
}

// Subscribe registers a new subscriber. On return the caller also receives the
// current init event so late joiners can catch up.
func (b *Bus) Subscribe() (subID uint64, init *InitEvent, ch chan interface{}) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.nextSubID++
	subID = b.nextSubID
	ch = make(chan interface{}, 256)
	b.subs[subID] = ch

	id, t := b.lastID, time.Since(b.start)
	init = &InitEvent{
		Event:  Event{ID: id, Time: t},
		Cols:   b.screenCols,
		Rows:   b.screenRows,
		Screen: string(b.screen),
	}
	return subID, init, ch
}

// Unsubscribe removes a subscriber.
func (b *Bus) Unsubscribe(subID uint64) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if ch, ok := b.subs[subID]; ok {
		close(ch)
		delete(b.subs, subID)
	}
}

// SubscriberCount returns the number of active subscribers.
func (b *Bus) SubscriberCount() int {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return len(b.subs)
}
