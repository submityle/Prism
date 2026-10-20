//! Atomic complete-source refreshes against real discovery and catalog indexes.

extern crate alloc;

use alloc::collections::BTreeMap;
use prism_editor_content::{
    filesystem::{FilesystemDiscovery, FilesystemExcludes, FilesystemLimits, FilesystemSnapshot},
    AssetId, AssetKind, AssetQuery, AssetRecord, CacheStatus, CatalogSnapshotLimits,
    ContentCatalog, ContentError, PageRequest, TreeQuery, VcsRecord, VcsState,
};
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
    task::Poll,
    time::{Duration, Instant},
};

fn record(id: u128, path: &str) -> AssetRecord {
    AssetRecord::new(AssetId::from_raw(id), path, AssetKind::new("data").unwrap()).unwrap()
}

#[test]
fn full_refresh_accepts_path_swaps_and_removals_with_one_generation_change() {
    let mut catalog = ContentCatalog::new();
    catalog.insert(record(1, "old/a")).unwrap();
    catalog.insert(record(2, "old/b")).unwrap();
    catalog.insert(record(3, "removed")).unwrap();
    let vcs = VcsRecord {
        state: VcsState::Modified,
        ..VcsRecord::default()
    };
    catalog.set_vcs_status("old/a", vcs.clone()).unwrap();
    let mut first = record(1, "old/b");
    first.dependencies.insert(AssetId::from_raw(3));
    first.add_tag("authored").unwrap();
    first.favorite = true;
    let second = record(2, "old/a");
    let generation = catalog.generation();
    let page = catalog
        .query_page(&AssetQuery::default(), PageRequest::new(1), None)
        .unwrap();
    assert!(catalog
        .replace_assets(
            generation,
            [first.clone(), second.clone()],
            CatalogSnapshotLimits::default()
        )
        .unwrap());
    assert_eq!(catalog.generation(), generation + 1);
    assert_eq!(catalog.get(first.id), Some(&first));
    assert!(catalog.get(AssetId::from_raw(3)).is_none());
    assert_eq!(
        catalog.reverse_dependencies(AssetId::from_raw(3)),
        [first.id]
    );
    assert!(catalog.validate().has_errors());
    // VCS describes exact paths, so swapping identities does not swap status.
    assert_eq!(catalog.vcs_status(second.id), Some(vcs));
    assert_eq!(catalog.vcs_status(first.id), Some(VcsRecord::default()));
    assert_eq!(
        catalog.query_page(&AssetQuery::default(), PageRequest::new(1), page.next),
        Err(ContentError::InvalidCursor)
    );
    let tree = catalog
        .tree_page(&TreeQuery::default(), PageRequest::new(16), None)
        .unwrap();
    assert_eq!(tree.items.len(), 1);
    assert_eq!(tree.items[0].path, "old");
    assert_eq!(tree.items[0].descendant_assets, 2);
    let generation = catalog.generation();
    let _ = catalog
        .query_page(&AssetQuery::default(), PageRequest::new(1), None)
        .unwrap();
    assert!(!catalog
        .replace_assets(
            generation,
            [second, first],
            CatalogSnapshotLimits::default()
        )
        .unwrap());
    assert_eq!(catalog.generation(), generation);
    assert_eq!(
        catalog
            .query_page(&AssetQuery::default(), PageRequest::new(1), None)
            .unwrap()
            .cache,
        CacheStatus::Hit
    );
    assert!(catalog
        .replace_assets(generation, [], CatalogSnapshotLimits::default())
        .unwrap());
    assert!(catalog.is_empty());
    assert!(catalog
        .reverse_dependencies(AssetId::from_raw(3))
        .is_empty());
    assert!(catalog
        .tree_page(&TreeQuery::default(), PageRequest::new(16), None)
        .unwrap()
        .items
        .is_empty());
}

#[test]
fn invalid_candidates_and_all_snapshot_bounds_leave_live_indexes_untouched() {
    let mut catalog = ContentCatalog::new();
    catalog.insert(record(7, "live/a")).unwrap();
    catalog.insert(record(8, "live/b")).unwrap();
    let generation = catalog.generation();
    let page = catalog
        .query_page(&AssetQuery::default(), PageRequest::new(1), None)
        .unwrap();
    let mut invalid_path = record(5, "valid");
    invalid_path.path = "../escape".into();
    let mut invalid_tag = record(5, "valid");
    invalid_tag.tags.insert(" ".into());
    for candidate in [
        vec![record(1, "a"), record(1, "b")],
        vec![record(1, "a"), record(2, "a")],
        vec![record(1, "a.other"), record(2, "a/child"), record(3, "a")],
        vec![record(1, "a"), record(2, "a/child")],
        vec![record(1, "a"), invalid_path],
        vec![record(1, "a"), invalid_tag],
    ] {
        assert!(catalog
            .replace_assets(generation, candidate, CatalogSnapshotLimits::default())
            .is_err());
    }
    let mut rich = record(1, "a/b");
    rich.dependencies.insert(AssetId::from_raw(2));
    rich.add_tag("tag").unwrap();
    let bounds = CatalogSnapshotLimits {
        max_assets: 1,
        max_path_bytes: 3,
        max_path_depth: 2,
        max_dependencies: 1,
        max_tags: 1,
        max_payload_bytes: 42,
    };
    for (limits, name) in [
        (
            CatalogSnapshotLimits {
                max_assets: 0,
                ..bounds
            },
            "asset count",
        ),
        (
            CatalogSnapshotLimits {
                max_path_bytes: 2,
                ..bounds
            },
            "path bytes",
        ),
        (
            CatalogSnapshotLimits {
                max_path_depth: 1,
                ..bounds
            },
            "path depth",
        ),
        (
            CatalogSnapshotLimits {
                max_dependencies: 0,
                ..bounds
            },
            "dependency count",
        ),
        (
            CatalogSnapshotLimits {
                max_tags: 0,
                ..bounds
            },
            "tag count",
        ),
        (
            CatalogSnapshotLimits {
                max_payload_bytes: 41,
                ..bounds
            },
            "payload bytes",
        ),
    ] {
        assert_eq!(
            catalog.replace_assets(generation, [rich.clone()], limits),
            Err(ContentError::SnapshotLimitExceeded(name))
        );
    }
    assert_eq!(catalog.generation(), generation);
    assert_eq!(catalog.len(), 2);
    assert_eq!(
        catalog
            .query_page(&AssetQuery::default(), PageRequest::new(1), page.next)
            .unwrap()
            .items,
        [AssetId::from_raw(8)]
    );
    assert_eq!(catalog.reverse_dependencies(AssetId::from_raw(2)), []);
    assert!(catalog.replace_assets(generation, [rich], bounds).unwrap());
    assert_eq!(catalog.generation(), generation + 1);
}

static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

#[test]
fn refresh_consumption_is_bounded_and_dependency_edits_keep_external_references() {
    let mut catalog = ContentCatalog::new();
    let mut consumed = 0u128;
    let generation = catalog.generation();
    let records = std::iter::from_fn(|| {
        consumed += 1;
        Some(record(consumed, &format!("asset/{consumed}")))
    });
    assert_eq!(
        catalog.replace_assets(
            generation,
            records,
            CatalogSnapshotLimits {
                max_assets: 2,
                ..CatalogSnapshotLimits::default()
            }
        ),
        Err(ContentError::SnapshotLimitExceeded("asset count"))
    );
    assert_eq!(consumed, 3);
    assert!(catalog.is_empty());
    assert_eq!(catalog.generation(), generation);
    let mut a = record(1, "a");
    let mut b = record(2, "b");
    a.dependencies
        .extend([AssetId::from_raw(2), AssetId::from_raw(99)]);
    b.dependencies.insert(AssetId::from_raw(99));
    catalog
        .replace_assets(generation, [a.clone(), b], CatalogSnapshotLimits::default())
        .unwrap();
    let generation = catalog.generation();
    a.dependencies = [AssetId::from_raw(77)].into();
    a.pinned = true;
    catalog
        .replace_assets(generation, [a.clone()], CatalogSnapshotLimits::default())
        .unwrap();
    assert!(catalog
        .reverse_dependencies(AssetId::from_raw(99))
        .is_empty());
    assert!(catalog
        .reverse_dependencies(AssetId::from_raw(2))
        .is_empty());
    assert_eq!(catalog.reverse_dependencies(AssetId::from_raw(77)), [a.id]);
    assert_eq!(
        catalog
            .tree_page(&TreeQuery::default(), PageRequest::new(16), None)
            .unwrap()
            .items[0]
            .asset,
        Some(a.id)
    );
    let generation = catalog.generation();
    a.dependencies = [AssetId::from_raw(88)].into();
    catalog
        .replace_assets(generation, [a.clone()], CatalogSnapshotLimits::default())
        .unwrap();
    assert!(catalog
        .reverse_dependencies(AssetId::from_raw(77))
        .is_empty());
    assert_eq!(catalog.reverse_dependencies(AssetId::from_raw(88)), [a.id]);
    assert_eq!(
        catalog
            .tree_page(&TreeQuery::default(), PageRequest::new(16), None)
            .unwrap()
            .items[0]
            .asset,
        Some(a.id)
    );
}

struct SourceDirectory(PathBuf);
impl SourceDirectory {
    fn new() -> Self {
        loop {
            let path = std::env::temp_dir().join(format!(
                "prism-source-publication-{}-{}",
                std::process::id(),
                NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Self(fs::canonicalize(path).unwrap()),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("create source test directory: {error}"),
            }
        }
    }
    fn snapshot(&self) -> FilesystemSnapshot {
        let discovery =
            FilesystemDiscovery::new(FilesystemLimits::default(), FilesystemExcludes::default())
                .unwrap();
        let mut request = discovery.start(&self.0).unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            match request.poll() {
                Poll::Ready(result) => return result.unwrap(),
                Poll::Pending => {
                    assert!(Instant::now() < deadline);
                    std::thread::sleep(Duration::from_millis(1));
                }
            }
        }
    }
}
impl Drop for SourceDirectory {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

#[test]
fn real_scan_publication_keeps_host_identity_through_rename_and_rejects_stale_edits() {
    let directory = SourceDirectory::new();
    fs::create_dir(directory.0.join(" 资源 ")).unwrap();
    let old_path = " 资源 /line\nname.asset ";
    fs::write(directory.0.join(old_path), b"source").unwrap();
    fs::write(directory.0.join("source.manifest"), b"host metadata").unwrap();
    let mut manifest = BTreeMap::from([(old_path.to_string(), record(50, old_path))]);
    let mut catalog = ContentCatalog::new();
    let generation = catalog.generation();
    let snapshot = directory.snapshot();
    assert!(catalog
        .replace_from_filesystem(
            generation,
            &snapshot,
            CatalogSnapshotLimits::default(),
            |entry| Ok(manifest.get(&entry.path).cloned())
        )
        .unwrap());
    let old_generation = catalog.generation();
    let mut authored = catalog.get(AssetId::from_raw(50)).unwrap().clone();
    authored.favorite = true;
    authored.add_tag("retain").unwrap();
    catalog.update(authored.clone()).unwrap();
    let mut called = false;
    assert_eq!(
        catalog.replace_from_filesystem(
            old_generation,
            &snapshot,
            CatalogSnapshotLimits::default(),
            |_| {
                called = true;
                Ok(None)
            }
        ),
        Err(ContentError::StaleSnapshot {
            expected: old_generation,
            actual: catalog.generation()
        })
    );
    assert!(!called);
    let generation = catalog.generation();
    let new_path = "renamed.asset";
    fs::rename(directory.0.join(old_path), directory.0.join(new_path)).unwrap();
    authored.path = new_path.into();
    manifest = BTreeMap::from([(new_path.into(), authored.clone())]);
    let snapshot = directory.snapshot();
    assert!(catalog
        .replace_from_filesystem(
            generation,
            &snapshot,
            CatalogSnapshotLimits::default(),
            |entry| Ok(manifest.get(&entry.path).cloned())
        )
        .unwrap());
    assert_eq!(catalog.len(), 1);
    assert_eq!(catalog.get(authored.id), Some(&authored));
    let generation = catalog.generation();
    assert!(catalog
        .replace_from_filesystem(
            generation,
            &snapshot,
            CatalogSnapshotLimits::default(),
            |_| Ok(Some(record(1, "different-path")))
        )
        .is_err());
    assert_eq!(catalog.generation(), generation);
    assert_eq!(catalog.get(authored.id), Some(&authored));
    assert!(catalog
        .replace_from_filesystem(
            generation,
            &snapshot,
            CatalogSnapshotLimits::default(),
            |_| Err(ContentError::UnknownAsset(AssetId::from_raw(99)))
        )
        .is_err());
    assert_eq!(catalog.generation(), generation);
}
