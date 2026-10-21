#!/usr/bin/env bash

# Check resolved names, not manifest text: dependency aliases, optional features,
# target-specific dependencies and indirect edges must not hide a Bevy crate.
set -euo pipefail

particle_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
particle_report="$(mktemp "${TMPDIR:-/tmp}/prism-particle-dependencies.XXXXXX")"
trap 'rm -f "$particle_report"' EXIT

cargo tree --manifest-path "$particle_root/Cargo.toml" --locked \
  --all-features --target all --edges normal,build,dev \
  --prefix none --format '{p}' \
  -p prism_particle_app -p prism_particle_render \
  -p prism_render_gpudriven -p prism_render_app >"$particle_report"

if rg -n '^bevy([_ -]|$)' "$particle_report"; then
  printf 'particle dependency boundary failed: resolved Bevy dependency\n' >&2
  exit 1
else
  particle_search_result=$?
  if [[ "$particle_search_result" -ne 1 ]]; then
    exit "$particle_search_result"
  fi
fi

printf 'particle dependency boundary passed: all features, all targets, normal/build/dev edges\n'
