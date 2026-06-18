module bash-forwarder-vt

go 1.26.4

replace github.com/charmbracelet/x/vt => /home/jamesbrown/Desktop/works/terminal-searcher/reference/x-main/vt

replace github.com/charmbracelet/x/ansi => /home/jamesbrown/Desktop/works/terminal-searcher/reference/x-main/ansi

replace github.com/charmbracelet/x/term => /home/jamesbrown/Desktop/works/terminal-searcher/reference/x-main/term

replace github.com/charmbracelet/x/termios => /home/jamesbrown/Desktop/works/terminal-searcher/reference/x-main/termios

replace github.com/charmbracelet/x/exp/ordered => /home/jamesbrown/Desktop/works/terminal-searcher/reference/x-main/exp/ordered

replace github.com/charmbracelet/ultraviolet => /home/jamesbrown/go/pkg/mod/github.com/charmbracelet/ultraviolet@v0.0.0-20260303162955-0b88c25f3fff

replace github.com/charmbracelet/colorprofile => /home/jamesbrown/go/pkg/mod/github.com/charmbracelet/colorprofile@v0.4.2

replace github.com/charmbracelet/x/cellbuf => /home/jamesbrown/Desktop/works/terminal-searcher/reference/x-main/cellbuf

replace github.com/charmbracelet/x/windows => /home/jamesbrown/Desktop/works/terminal-searcher/reference/x-main/windows

require (
	github.com/charmbracelet/ultraviolet v0.0.0-20260615092913-2399af76d5b1
	github.com/charmbracelet/x/vt v0.0.0-20260615092313-b57e5e6d29bb
	github.com/creack/pty v1.1.24
	golang.org/x/sys v0.46.0
	golang.org/x/term v0.44.0
)

require (
	github.com/charmbracelet/colorprofile v0.4.2 // indirect
	github.com/charmbracelet/x/ansi v0.11.7 // indirect
	github.com/charmbracelet/x/exp/ordered v0.1.0 // indirect
	github.com/charmbracelet/x/term v0.2.2 // indirect
	github.com/charmbracelet/x/termios v0.1.1 // indirect
	github.com/charmbracelet/x/windows v0.2.2 // indirect
	github.com/clipperhouse/displaywidth v0.11.0 // indirect
	github.com/clipperhouse/uax29/v2 v2.7.0 // indirect
	github.com/lucasb-eyer/go-colorful v1.4.0 // indirect
	github.com/mattn/go-runewidth v0.0.24 // indirect
	github.com/muesli/cancelreader v0.2.2 // indirect
	github.com/rivo/uniseg v0.4.7 // indirect
	github.com/xo/terminfo v0.0.0-20220910002029-abceb7e1c41e // indirect
	golang.org/x/sync v0.19.0 // indirect
)
