#!/usr/bin/env bash
set -euo pipefail
exec /usr/sbin/runc --rootless=true "$@"
