cd flashback_terminal/static/js/vendor

rm *.js
rm *.map

export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897

# Main libraries
curl -OL https://cdn.jsdelivr.net/npm/xterm@5.3.0/lib/xterm.js
curl -OL https://cdn.jsdelivr.net/npm/xterm@5.3.0/lib/xterm.js.map

curl -OL https://cdn.jsdelivr.net/npm/xterm-addon-fit@0.8.0/lib/xterm-addon-fit.js
curl -OL https://cdn.jsdelivr.net/npm/xterm-addon-fit@0.8.0/lib/xterm-addon-fit.js.map