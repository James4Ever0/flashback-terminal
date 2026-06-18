#!/bin/bash

export FLASHBACK_SHELL_ALLOW_NESTED_TMUX=true
export FLASHBACK_SHELL_SERVER_URL=http://127.0.0.1:8080

./dist/flashback-shell-linux-amd64 $@ 
