package capture

import (
	"reflect"
	"testing"
)

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
			name:     "scroll off top",
			prev:     []string{"a", "b", "c"},
			curr:     []string{"b", "c", "d"},
			expected: []string{"d"},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, _ := diffLines(tc.prev, tc.curr)
			if !reflect.DeepEqual(got, tc.expected) {
				t.Fatalf("got %v, want %v", got, tc.expected)
			}
		})
	}
}

func TestDiffLinesIndex(t *testing.T) {
	cases := []struct {
		name         string
		prev         []string
		curr         []string
		expectedLen  int
	}{
		{
			name:        "in-place update",
			prev:        []string{"a", "b", "c"},
			curr:        []string{"a", "B", "c"},
			expectedLen: 1,
		},
		{
			name:        "shorter current pads top",
			prev:        []string{"a", "b", "c", "d"},
			curr:        []string{"c", "d"},
			expectedLen: 0,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, _ := diffLinesIndex(tc.prev, tc.curr)
			if len(got) != tc.expectedLen {
				t.Fatalf("got %v (len %d), want len %d", got, len(got), tc.expectedLen)
			}
		})
	}
}
