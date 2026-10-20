# prism_editor_content

`prism_editor_content` is the deterministic data layer for Prism's content
browser. It has no renderer, filesystem, OS, or legacy-engine dependency. Adapters
own I/O and translate their observations into the contracts in this crate.

## Data model

- `ContentCatalog` stores canonical project-relative asset records and keeps a
  path index, an immediate-child tree index, descendant counts, a reverse
  dependency index, VCS metadata, and a mutation generation.
- `AssetQuery` and `query_page` provide case-insensitive search over paths,
  kinds, and tags, with kind/tag/favorite/pinned/VCS filters and an optional
  subtree root.
- `TreeQuery` and `tree_page` return only immediate children of a folder. A
  folder row includes its descendant asset count, so a UI can virtualize both
  rows and deep folders without materializing the whole tree.
- `AssetCursor` and `TreeCursor` include the catalog generation and query
  fingerprint. Any catalog mutation invalidates old cursors and clears cached
  pages instead of returning an inconsistent view.

## Bounds and performance

`ContentLimits` controls page size, cache entries/items, task history, export
selection size, diagnostics, and dependency traversal. The default page cache
is an LRU-style deterministic cache: the least recently used page is evicted,
with `CacheKey` ordering as the tie-breaker. Catalog queries walk the ordered
path index; tree queries walk only one folder's child index and use the path
index for filtered descendant checks.

## Dependencies and diagnostics

Dependencies are identity edges and may point to an asset that has not arrived
yet. This makes incremental filesystem/import updates safe: missing edges are
retained and reported by `validate`, while reverse references remain queryable.
The report deterministically covers missing dependencies, one discovered cycle,
depth-limit warnings, and VCS conflicts. `dependency_closure` and
`dependent_closure` enforce both node and depth bounds.

## Import/export and version control

`TaskStore` models queued, running, succeeded, failed, and cancelled import or
export operations. Progress is monotonic, terminal tasks are immutable, and
completed history is evicted only when the configured bound is reached.
`VcsRecord` stores provider state, staging, revision, and lock-owner metadata;
VCS adapters can publish status before or after an asset is imported.

## Verification

The crate's tests cover deterministic ordering, cursor invalidation, tree
virtualization, filtering, cache bounds/eviction, dependency cycles and
missing references, index updates/removal, VCS diagnostics, and task
transitions. Run:

```text
cargo fmt --package prism_editor_content
cargo clippy -p prism_editor_content --all-targets -- -D warnings
cargo test -p prism_editor_content
```
