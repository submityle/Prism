//! Exact identity and bounded metadata publication across asynchronous refreshes.
extern crate alloc;

use alloc::{string::String, vec::Vec};
use core::cell::Cell;
use prism_editor_content::{
    filesystem::FilesystemSnapshot, AssetId, AssetKind, AssetRecord, CatalogSnapshotLimits,
    ContentCatalog, ContentError, VcsRecord, VcsSnapshotLimits, VcsState,
};

fn record() -> AssetRecord {
    AssetRecord::new(
        AssetId::from_raw(1),
        "a.png",
        AssetKind::new("texture").unwrap(),
    )
    .unwrap()
}

fn catalog() -> ContentCatalog {
    let mut catalog = ContentCatalog::new();
    catalog.insert(record()).unwrap();
    catalog
}

#[test]
fn exact_fences_reject_unrelated_equal_generations_before_consuming_input() {
    let original = catalog();
    let token = original.snapshot_token();
    let mut unrelated = catalog();
    assert_eq!(unrelated.generation(), original.generation());
    let calls = Cell::new(0);
    let input = core::iter::from_fn(|| {
        calls.set(calls.get() + 1);
        Some(record())
    });
    assert_eq!(
        unrelated.publish_assets(&token, input, CatalogSnapshotLimits::default()),
        Err(ContentError::StaleCatalogSnapshot)
    );
    assert_eq!(calls.get(), 0);
    let input = core::iter::from_fn(|| {
        calls.set(calls.get() + 1);
        Some((String::from("a.png"), VcsRecord::default()))
    });
    assert_eq!(
        unrelated.publish_vcs(&token, input, VcsSnapshotLimits::default()),
        Err(ContentError::StaleCatalogSnapshot)
    );
    assert_eq!(calls.get(), 0);
    let snapshot = FilesystemSnapshot {
        root: std::path::PathBuf::from("/unused"),
        entries: Vec::new(),
        scanned_entries: 0,
        total_path_bytes: 0,
    };
    assert_eq!(
        unrelated.publish_filesystem(&token, &snapshot, CatalogSnapshotLimits::default(), |_| {
            calls.set(calls.get() + 1);
            Ok(None)
        }),
        Err(ContentError::StaleCatalogSnapshot)
    );
    assert_eq!(calls.get(), 0);
}

#[test]
fn shared_unchanged_snapshots_retain_fences_but_divergent_clones_revoke_them() {
    let mut original = catalog();
    let token = original.snapshot_token();
    let mut branch = original.clone();
    assert!(branch.matches_snapshot(&token));
    assert!(!branch
        .publish_assets(&token, [record()], CatalogSnapshotLimits::default())
        .unwrap());
    assert!(branch.matches_snapshot(&token));
    let mut next = record();
    next.favorite = true;
    original.update(next).unwrap();
    let mut next = record();
    next.pinned = true;
    branch.update(next).unwrap();
    assert_eq!(original.generation(), branch.generation());
    assert!(!original.matches_snapshot(&token));
    assert!(!branch.matches_snapshot(&token));
    let other = original.snapshot_token();
    assert!(!branch.matches_snapshot(&other));
}

#[test]
fn vcs_budget_and_duplicate_errors_are_atomic_and_identical_refresh_retains_receipts() {
    let mut catalog = catalog();
    let status = VcsRecord {
        state: VcsState::Modified,
        revision: Some("rev".into()),
        lock_owner: Some("u".into()),
        staged: true,
    };
    let rows = [(String::from("a.png"), status.clone())];
    let token = catalog.snapshot_token();
    assert!(catalog
        .publish_vcs(&token, rows.clone(), VcsSnapshotLimits::default())
        .unwrap());
    assert!(!catalog.matches_snapshot(&token));
    let token = catalog.snapshot_token();
    assert!(!catalog
        .publish_vcs(&token, rows.clone(), VcsSnapshotLimits::default())
        .unwrap());
    assert!(catalog.matches_snapshot(&token));
    for limits in [
        VcsSnapshotLimits {
            max_paths: 0,
            ..VcsSnapshotLimits::default()
        },
        VcsSnapshotLimits {
            max_path_bytes: 4,
            ..VcsSnapshotLimits::default()
        },
        VcsSnapshotLimits {
            max_payload_bytes: 8,
            ..VcsSnapshotLimits::default()
        },
    ] {
        assert!(matches!(
            catalog.publish_vcs(&token, rows.clone(), limits),
            Err(ContentError::SnapshotLimitExceeded(_))
        ));
        assert!(catalog.matches_snapshot(&token));
        assert_eq!(
            catalog.vcs_status(AssetId::from_raw(1)),
            Some(status.clone())
        );
    }
    assert_eq!(
        catalog.publish_vcs(
            &token,
            [rows[0].clone(), rows[0].clone()],
            VcsSnapshotLimits::default()
        ),
        Err(ContentError::DuplicatePath("a.png".into()))
    );
    assert!(catalog.matches_snapshot(&token));
    assert!(catalog
        .publish_vcs(
            &token,
            rows,
            VcsSnapshotLimits {
                max_paths: 1,
                max_path_bytes: 5,
                max_payload_bytes: 9
            }
        )
        .is_ok());
    assert!(catalog.matches_snapshot(&token));
    let calls = Cell::new(0);
    let input = core::iter::from_fn(|| {
        calls.set(calls.get() + 1);
        Some((format!("{}.png", calls.get()), VcsRecord::default()))
    });
    assert!(catalog
        .publish_vcs(
            &token,
            input,
            VcsSnapshotLimits {
                max_paths: 1,
                ..VcsSnapshotLimits::default()
            }
        )
        .is_err());
    assert_eq!(calls.get(), 2);
    assert!(catalog.matches_snapshot(&token));
    assert!(catalog
        .publish_vcs(&token, [], VcsSnapshotLimits::default())
        .unwrap());
    assert_eq!(
        catalog.vcs_status(AssetId::from_raw(1)),
        Some(VcsRecord::default())
    );
}
