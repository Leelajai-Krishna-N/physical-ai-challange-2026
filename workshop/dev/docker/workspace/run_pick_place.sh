#!/usr/bin/env bash
set -eo pipefail

cd /home/hacker/workspace
exec /usr/bin/python3 auto_pick_place.py "$@"
