#!/bin/bash
# One-time setup for the dashboard browser smoke test. Fetches (never commits):
#   smoke/vendor/{react,react-dom,three} UMD bundles + jsdom in smoke/node_modules.
# The pytest skips the smoke test when any of these are absent.
set -e
cd "$(dirname "$0")"
mkdir -p vendor
curl -sL --max-time 90 -o vendor/react.min.js https://unpkg.com/react@18/umd/react.production.min.js
curl -sL --max-time 90 -o vendor/react-dom.min.js https://unpkg.com/react-dom@18/umd/react-dom.production.min.js
curl -sL --max-time 90 -o vendor/three.min.js https://unpkg.com/three@0.149.0/build/three.min.js
npm init -y >/dev/null 2>&1 || true
npm install jsdom
node harness.js
