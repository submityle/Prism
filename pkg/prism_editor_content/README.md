# prism_editor_content

`prism_editor_content` is the deterministic data layer for Prism's content
browser. Its catalog has no renderer or engine dependency. The filesystem
adapter uses standard-library workers to discover actual project sources;
hosts supply stable asset identities and importer-specific metadata.

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
  fingerprint plus the last returned asset or child name. Effective catalog
  mutations invalidate old cursors and clear cached pages. Repeated identical
  records, dependency edges, and VCS observations retain cursors and cached pages.
- Paths are exact UTF-8 project-relative keys. Spaces, Unicode and newlines are
  preserved. Absolute paths, backslashes, NUL, empty components and dot/parent
  components are rejected rather than normalized into a different identity.

## Bounds and performance

`ContentLimits` controls page size, cache entries/items, task history, export
selection size, diagnostics, and dependency traversal. The default page cache
is an LRU-style deterministic cache: the least recently used page is evicted,
with `CacheKey` ordering as the tie-breaker. Catalog queries walk the ordered
path index starting after the cursor's last path, or at the subtree prefix for
the first page. Tree queries borrow one folder's child index and continue after
the last child name, without cloning the folder or rescanning prior pages.
Filtered queries still inspect nonmatching candidates and use descendant path
scans to decide whether a folder matches. No constant-cost claim is made for
those scans. Reverse edges are maintained from each edit's dependency difference;
metadata-only edits do not rebuild path/tree indexes.

## Dependencies and diagnostics

Dependencies are identity edges and may point to an asset that has not arrived
yet. This makes incremental filesystem/import updates safe: missing edges are
retained and reported by `validate`, while reverse references remain queryable.
The report deterministically covers missing dependencies, one discovered cycle,
depth-limit warnings, and VCS conflicts. Validation retains only the configured
number of diagnostics while collecting them, preserving the sorted prefix and
reporting truncation. `dependency_closure` and `dependent_closure` enforce both
node and depth bounds while borrowing adjacency sets, so an oversized fan-out
is not copied before the node limit can reject it. Cycle validation keeps
borrowed edge iterators per DFS frame instead of copying each node's outgoing
edges. Roots have depth zero; complete leaves and back-edges at the configured
boundary succeed, while unseen deeper dependencies report the depth limit.

## Import/export and version control

`TaskStore` models queued, running, succeeded, failed, and cancelled import or
export operations. Progress is monotonic, terminal tasks are immutable, and
completed history is evicted only when the configured bound is reached.
`VcsRecord` stores provider state, staging, revision, and lock-owner metadata;
VCS adapters can publish status before or after an asset is imported.
`replace_vcs_statuses` atomically publishes a complete snapshot and removes stale
statuses. Invalid or duplicate paths reject the entire snapshot; an identical
snapshot preserves the generation and cached pages.

## Source discovery and publication

The public filesystem module enumerates real files and directories into a sorted
complete snapshot on a bounded worker. The service limits concurrency, entries,
path bytes, scanned entries, file sizes and depth; it prunes configurable build
and repository metadata, skips symlinks, and reports errors without partial
snapshots. Polling is nonblocking. Cancellation and deadlines are cooperative;
dropping a request joins its worker and can wait for a blocking OS call. See
the filesystem design note in docs for platform and confinement boundaries.

Capture the catalog generation before starting a scan, then pass the complete
snapshot to ContentCatalog::replace_from_filesystem. The host resolves each file
to an optional AssetRecord using its manifest or sidecars. The record must retain
the exact discovered path. Stable IDs are not derived from paths; the host can
preserve identity and authored metadata across renames. A resolver error, invalid
record, duplicate identity/path, file/folder collision or stale generation leaves
the current catalog, indexes, cursors and caches intact.

ContentCatalog::replace_assets also accepts complete records from other source
adapters. CatalogSnapshotLimits bounds records, path bytes/depth, dependencies,
tags and logical payload bytes while consuming the candidate. Byte accounting
does not measure allocator or index overhead. Replacement supports path swaps,
removes omitted assets, retains unresolved dependency references, and preserves
VCS observations by exact path. An effective refresh changes generation once;
an identical refresh retains cached pages. Validation requires inspecting the
complete input, but unchanged snapshots do not rebuild derived indexes, metadata
edits retain the tree, and reverse references apply only edge differences.

## Verification

The crate's tests cover deterministic ordering, cursor invalidation, tree
virtualization, filtering, cache bounds/eviction, dependency cycles and
missing references, index updates/removal, VCS diagnostics, and task
transitions. Additional regressions cover 257-folder filtered pagination with
changing page sizes and cache eviction, exact path identity, file/folder prefix
collisions, atomic VCS publication, bounded diagnostics, and 400 mixed edits
checked against an independent reverse-reference reconstruction. On 2026-10-20,
18 original catalog tests, 28 real filesystem tests and 4 source-publication
tests passed. Publication tests cover path swaps/removals, every snapshot bound,
bounded iterator consumption, stale refresh refusal, resolver failure and real
file rename with host-owned stable identity. This does not verify importer,
thumbnail production or the content UI. Run:

```text
cargo fmt --package prism_editor_content
cargo clippy -p prism_editor_content --all-targets -- -D warnings
cargo test -p prism_editor_content
```
