//! Catalog mutation, exact-path identity, and provider-publication regressions.

extern crate alloc;

use alloc::collections::{BTreeMap, BTreeSet};

use prism_editor_content::{
    AssetId, AssetKind, AssetQuery, AssetRecord, CacheStatus, ContentCatalog, ContentError,
    ContentLimits, PageRequest, TreeQuery, VcsRecord, VcsState,
};

fn record(id: u128, path: &str) -> AssetRecord {
    AssetRecord::new(AssetId::from_raw(id), path, AssetKind::new("data").unwrap()).unwrap()
}

#[test]
fn file_folder_collisions_reject_without_changing_catalog_or_cursors() {
    let mut catalog = ContentCatalog::new();
    catalog.insert(record(1, "folder.other")).unwrap();
    catalog.insert(record(2, "folder/child")).unwrap();
    let page = catalog
        .query_page(&AssetQuery::default(), PageRequest::new(1), None)
        .unwrap();
    let generation = catalog.generation();
    assert_eq!(
        catalog.insert(record(3, "folder")),
        Err(ContentError::PathPrefixConflict("folder".into()))
    );
    assert_eq!(
        catalog.update(record(1, "folder")),
        Err(ContentError::PathPrefixConflict("folder".into()))
    );
    assert_eq!(catalog.generation(), generation);
    let next = catalog
        .query_page(&AssetQuery::default(), PageRequest::new(1), page.next)
        .unwrap();
    assert_eq!(next.items, [AssetId::from_raw(2)]);
    let root = catalog
        .tree_page(&TreeQuery::default(), PageRequest::new(8), None)
        .unwrap();
    assert_eq!(root.items.len(), 2);
    assert_eq!(root.items[0].name, "folder");
    assert_eq!(root.items[0].descendant_assets, 1);
}

#[test]
fn exact_paths_preserve_whitespace_unicode_and_newlines() {
    let path = " 资源 /line\nname.asset ";
    let mut catalog = ContentCatalog::new();
    catalog.insert(record(1, path)).unwrap();
    catalog.insert(record(2, "资源/line\nname.asset")).unwrap();
    assert_eq!(catalog.get(AssetId::from_raw(1)).unwrap().path, path);
    for path in [
        "/absolute",
        "trailing/",
        "double//slash",
        "a/./b",
        "../b",
        "a\\b",
        "bad\0path",
    ] {
        assert!(
            AssetRecord::new(AssetId::from_raw(3), path, AssetKind::new("data").unwrap()).is_err()
        );
    }
}

#[test]
fn unchanged_observations_preserve_cached_pages_and_generation() {
    let mut catalog = ContentCatalog::new();
    let mut first = record(1, "a");
    first.dependencies.insert(AssetId::from_raw(2));
    catalog.insert(first.clone()).unwrap();
    catalog.insert(record(2, "b")).unwrap();
    let status = VcsRecord {
        state: VcsState::Modified,
        ..VcsRecord::default()
    };
    catalog.set_vcs_status("a", status.clone()).unwrap();
    let page = catalog
        .query_page(&AssetQuery::default(), PageRequest::new(1), None)
        .unwrap();
    let generation = catalog.generation();
    catalog.update(first).unwrap();
    catalog
        .add_dependency(AssetId::from_raw(1), AssetId::from_raw(2))
        .unwrap();
    assert!(!catalog
        .remove_dependency(AssetId::from_raw(1), AssetId::from_raw(99))
        .unwrap());
    catalog.set_vcs_status("a", status.clone()).unwrap();
    assert!(!catalog
        .replace_vcs_statuses([("a".into(), status)])
        .unwrap());
    assert_eq!(catalog.generation(), generation);
    assert_eq!(
        catalog
            .query_page(&AssetQuery::default(), PageRequest::new(1), None)
            .unwrap()
            .cache,
        CacheStatus::Hit
    );
    assert_eq!(
        catalog
            .query_page(&AssetQuery::default(), PageRequest::new(1), page.next)
            .unwrap()
            .items,
        [AssetId::from_raw(2)]
    );
}

#[test]
fn provider_snapshot_is_atomic_and_clears_stale_states() {
    let mut catalog = ContentCatalog::new();
    catalog.insert(record(1, "a")).unwrap();
    catalog.insert(record(2, "b")).unwrap();
    let modified = VcsRecord {
        state: VcsState::Modified,
        ..VcsRecord::default()
    };
    catalog.set_vcs_status("a", modified.clone()).unwrap();
    let generation = catalog.generation();
    assert!(catalog
        .replace_vcs_statuses([
            ("b".into(), modified.clone()),
            ("../escape".into(), modified.clone()),
        ])
        .is_err());
    assert!(catalog
        .replace_vcs_statuses([
            ("b".into(), modified.clone()),
            ("b".into(), modified.clone()),
        ])
        .is_err());
    assert_eq!(catalog.generation(), generation);
    assert_eq!(
        catalog.vcs_status(AssetId::from_raw(1)),
        Some(modified.clone())
    );
    assert_eq!(
        catalog.vcs_status(AssetId::from_raw(2)),
        Some(VcsRecord::default())
    );
    assert!(catalog
        .replace_vcs_statuses([("b".into(), modified.clone())])
        .unwrap());
    assert_eq!(catalog.generation(), generation + 1);
    assert_eq!(
        catalog.vcs_status(AssetId::from_raw(1)),
        Some(VcsRecord::default())
    );
    assert_eq!(catalog.vcs_status(AssetId::from_raw(2)), Some(modified));
}

#[test]
fn incremental_dependency_updates_match_full_rebuild_across_mixed_edits() {
    let mut catalog = ContentCatalog::new();
    let mut authored = BTreeMap::<AssetId, AssetRecord>::new();
    let mut random = 0x9e37_79b9_u64;
    for step in 0..400 {
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        let raw = u128::from(random % 32);
        let id = AssetId::from_raw(raw);
        if step % 5 == 0 && authored.contains_key(&id) {
            catalog.remove(id).unwrap();
            authored.remove(&id);
        } else {
            let mut next = record(raw, &format!("dir{}/asset{raw}", step % 3));
            next.dependencies = BTreeSet::from([
                AssetId::from_raw(u128::from((random >> 9) % 37)),
                AssetId::from_raw(u128::from((random >> 21) % 37)),
            ]);
            if authored.contains_key(&id) {
                catalog.update(next.clone()).unwrap();
            } else {
                catalog.insert(next.clone()).unwrap();
            }
            authored.insert(id, next);
        }
        for target in 0..37 {
            let target = AssetId::from_raw(target);
            let expected: Vec<_> = authored
                .values()
                .filter(|record| record.dependencies.contains(&target))
                .map(|record| record.id)
                .collect();
            assert_eq!(
                catalog.reverse_dependencies(target),
                expected,
                "step {step}"
            );
        }
        assert_eq!(catalog.len(), authored.len());
    }
}

#[test]
fn deep_pages_continue_across_page_sizes_cache_hits_and_eviction() {
    let mut catalog = ContentCatalog::with_limits(ContentLimits {
        max_cache_entries: 2,
        max_cache_items: 32,
        ..ContentLimits::default()
    })
    .unwrap();
    catalog.insert(record(1000, "assets.other")).unwrap();
    catalog.insert(record(1001, "assets0/outside")).unwrap();
    let mut expected = Vec::new();
    let mut expected_tree = Vec::new();
    for index in 0..257 {
        let mut asset = record(257 - index, &format!("assets/branch{index:03}/item"));
        if index % 2 == 0 {
            asset.add_tag("included").unwrap();
            expected.push(asset.id);
            expected_tree.push(format!("branch{index:03}"));
        }
        catalog.insert(asset).unwrap();
    }
    let query = AssetQuery {
        root: Some("assets".into()),
        tag: Some("included".into()),
        ..AssetQuery::default()
    };
    let tree_query = TreeQuery {
        root: "assets".into(),
        tag: Some("included".into()),
        ..TreeQuery::default()
    };
    let mut assets = Vec::new();
    let mut children = Vec::new();
    let mut asset_cursor = None;
    let mut tree_cursor = None;
    for step in 0..100 {
        let request = PageRequest::new([1, 7, 4, 13, 2][step % 5]);
        let page = catalog.query_page(&query, request, asset_cursor).unwrap();
        let cached = catalog.query_page(&query, request, asset_cursor).unwrap();
        assert_eq!(cached.cache, CacheStatus::Hit);
        assert_eq!(cached.items, page.items);
        assert_eq!(cached.next, page.next);
        assets.extend(page.items);
        if let Some(cursor) = page.next {
            assert_eq!(cursor.offset(), assets.len());
        }
        let tree = catalog
            .tree_page(&tree_query, request, tree_cursor.clone())
            .unwrap();
        let cached = catalog
            .tree_page(&tree_query, request, tree_cursor.clone())
            .unwrap();
        assert_eq!(cached.cache, CacheStatus::Hit);
        assert_eq!(cached.items, tree.items);
        assert_eq!(cached.next, tree.next);
        children.extend(tree.items.into_iter().map(|entry| entry.name));
        if let Some(cursor) = &tree.next {
            assert_eq!(cursor.offset(), children.len());
        }
        assert_eq!(page.next.is_none(), tree.next.is_none());
        asset_cursor = page.next;
        tree_cursor = tree.next;
        // Evict both current pages before continuing from their retained cursors.
        for limit in [2, 3] {
            catalog
                .query_page(&AssetQuery::default(), PageRequest::new(limit), None)
                .unwrap();
        }
        assert!(catalog.cache_stats().entries <= 2);
        assert!(catalog.cache_stats().items <= 32);
        if asset_cursor.is_none() {
            break;
        }
    }
    assert!(asset_cursor.is_none());
    assert_eq!(assets, expected);
    assert_eq!(children, expected_tree);
    for (root, expected) in [("assets/branch000/item", 1), ("absent", 0)] {
        let page = catalog
            .query_page(
                &AssetQuery {
                    root: Some(root.into()),
                    ..AssetQuery::default()
                },
                PageRequest::new(4),
                None,
            )
            .unwrap();
        assert_eq!(page.items.len(), expected);
        assert!(page.next.is_none());
    }
}

#[test]
fn bounded_diagnostics_keep_the_same_deterministic_prefix_as_full_validation() {
    let mut full = ContentCatalog::with_limits(ContentLimits {
        max_diagnostics: 1000,
        ..ContentLimits::default()
    })
    .unwrap();
    let mut bounded = ContentCatalog::with_limits(ContentLimits {
        max_diagnostics: 7,
        ..ContentLimits::default()
    })
    .unwrap();
    for id in (0..40).rev() {
        let mut asset = record(id, &format!("asset{id:03}"));
        asset
            .dependencies
            .extend([AssetId::from_raw(1000 + id), AssetId::from_raw(2000 + id)]);
        full.insert(asset.clone()).unwrap();
        bounded.insert(asset).unwrap();
        let status = VcsRecord {
            state: VcsState::Conflicted,
            ..VcsRecord::default()
        };
        full.set_vcs_status(format!("asset{id:03}"), status.clone())
            .unwrap();
        bounded
            .set_vcs_status(format!("asset{id:03}"), status)
            .unwrap();
    }
    let full_report = full.validate();
    assert_eq!(full_report.diagnostics().len(), 120);
    assert!(!full_report.truncated);
    let report = bounded.validate();
    assert!(report.truncated);
    assert_eq!(report.diagnostics(), &full_report.diagnostics()[..7]);
    assert!(report.has_errors());
}
