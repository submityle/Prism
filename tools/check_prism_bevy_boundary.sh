#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="strict"

case "${1:-}" in
  "") ;;
  --report) MODE="report" ;;
  --strict) MODE="strict" ;;
  -h|--help)
    cat <<'USAGE'
Usage: tools/check_prism_bevy_boundary.sh [--report|--strict]

--report  Print the current Cargo/source boundary and always succeed.
--strict  Enforce the post-migration boundary and fail on any violation.
USAGE
    exit 0
    ;;
  *)
    printf 'unknown argument: %s\n' "$1" >&2
    exit 2
    ;;
esac

for command in cargo jq rg; do
  command -v "$command" >/dev/null 2>&1 || {
    printf 'required command not found: %s\n' "$command" >&2
    exit 2
  }
done

metadata="$(mktemp "${TMPDIR:-/tmp}/prism-bevy-boundary.XXXXXX.json")"
trap 'rm -f "$metadata"' EXIT

cargo metadata --manifest-path "$ROOT/Cargo.toml" --format-version 1 --no-deps >"$metadata"

workspace_members="$(jq '.workspace_members | length' "$metadata")"
default_members="$(jq '.workspace_default_members | length' "$metadata")"
default_names="$(jq -r '[.workspace_default_members[] as $id | .packages[] | select(.id == $id) | .name] | join(", ")' "$metadata")"
bevy_packages="$(jq '[.packages[] | select(.name | startswith("bevy"))] | length' "$metadata")"

printf 'workspace_members=%s\n' "$workspace_members"
printf 'workspace_default_members=%s\n' "$default_members"
printf 'workspace_default_member_names=%s\n' "$default_names"
printf 'bevy_workspace_packages=%s\n' "$bevy_packages"

printf '\n[direct Prism-package Bevy dependencies]\n'
direct_manifest_violations=0
while IFS= read -r manifest; do
  package="$(jq -r --arg path "$ROOT/$manifest" '.packages[] | select(.manifest_path == $path) | .name' "$metadata")"
  deps="$(jq -r --arg path "$ROOT/$manifest" '[.packages[] | select(.manifest_path == $path) | .dependencies[] | select(.name | startswith("bevy")) | .name] | join(", ")' "$metadata")"
  if [[ -n "$deps" ]]; then
    printf '%s: %s\n' "$package" "$deps"
    direct_manifest_violations=$((direct_manifest_violations + 1))
  fi
done < <(
  find "$ROOT/pkg" "$ROOT/os-backends" -name Cargo.toml -print \
    | sed "s#^$ROOT/##" \
    | sort
)

printf '\n[production source imports]\n'
source_import_violations=0
for source_root in \
  "$ROOT/pkg/prism_render_scene/src" \
  "$ROOT/pkg/prism_render_app/src" \
  "$ROOT/pkg/prism_app/src" \
  "$ROOT/pkg/prism_scene/src" \
  "$ROOT/pkg/prism_render_architecture/src" \
  "$ROOT/pkg/prism_render_runtime/src" \
  "$ROOT/pkg/prism_render_visibility/src"; do
  [[ -d "$source_root" ]] || continue
  matches="$(rg -n -P --no-heading '^\s*(?:pub\s+)?use\s+bevy|^\s*extern\s+crate\s+bevy|^\s*bevy_[A-Za-z0-9_]+::|^\s*bevy::' "$source_root" --glob '*.rs' || true)"
  if [[ -n "$matches" ]]; then
    printf '%s\n' "$matches"
    source_import_violations=$((source_import_violations + 1))
  fi
done

printf '\n[render-scene source reference summary]\n'
scene_rs_files="$(rg --files "$ROOT/pkg/prism_render_scene/src" | rg '\.rs$' | wc -l | tr -d ' ')"
scene_ref_files="$(rg -l '(^|[^A-Za-z0-9_])bevy_[A-Za-z0-9_]+|::bevy::|\bbevy::' "$ROOT/pkg/prism_render_scene/src" --glob '*.rs' || true)"
scene_ref_files="$(printf '%s\n' "$scene_ref_files" | sed '/^$/d' | wc -l | tr -d ' ')"
scene_ref_matches="$(rg -o 'bevy_[A-Za-z0-9_]+|::bevy::|\bbevy::' "$ROOT/pkg/prism_render_scene/src" --glob '*.rs' || true)"
scene_ref_matches="$(printf '%s\n' "$scene_ref_matches" | sed '/^$/d' | wc -l | tr -d ' ')"
printf 'rust_files=%s\n' "$scene_rs_files"
printf 'files_with_bevy_tokens=%s\n' "$scene_ref_files"
printf 'bevy_token_matches=%s\n' "$scene_ref_matches"

printf '\n[default production boundary]\n'
if [[ "$default_names" == *bevy* ]]; then
  printf 'violation: the workspace default member is a Bevy package (%s)\n' "$default_names"
  default_boundary_violation=1
else
  printf 'ok: no workspace default member name contains bevy\n'
  default_boundary_violation=0
fi

if [[ "$MODE" == "report" ]]; then
  exit 0
fi

violations=0
if (( direct_manifest_violations > 0 )); then
  printf '\nstrict failure: Prism production manifests still declare Bevy dependencies\n' >&2
  violations=$((violations + direct_manifest_violations))
fi
if (( source_import_violations > 0 )); then
  printf '\nstrict failure: the production source boundary still imports Bevy\n' >&2
  violations=$((violations + source_import_violations))
fi
if (( default_boundary_violation > 0 )); then
  printf '\nstrict failure: the workspace default production member is still Bevy\n' >&2
  violations=$((violations + default_boundary_violation))
fi

if (( violations > 0 )); then
  exit 1
fi

printf '\nstrict result: Prism production boundary is free of direct Bevy dependencies/imports\n'
