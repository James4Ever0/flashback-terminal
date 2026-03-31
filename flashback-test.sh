#!/bin/bash

conda deactivate

test -f .venv/bin/activate
if [ $? -ne 0 ]; then
    echo "Virtual env not found at .venv"
    echo "Create one with uv venv"
    exit 1
fi

source .venv/bin/activate
uv pip install --reinstall .[dev,embedding,screenshot,search]

# if developing terminal reattach, then we should not remove them.

# DEV_REATTACH=0
DEV_REATTACH=1

if [ $DEV_REATTACH -eq 0 ]; then
    echo "Not in dev reattach mode, removing cache files"
    echo "Remove default data storage"
    rm -rf /home/jamesbrown/.local/share/flashback-terminal
    echo "Remove tmux/screen sockets"
    rm -rf /home/jamesbrown/.flashback-terminal
else
    echo "In reattach dev mode, so cache files kept"
fi

flashback-terminal $@