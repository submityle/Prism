use prism_asset_build_core::{
    AssetId, ContentHash, EditableAsset, EditableAssetError, EditableDependencyKind,
    EditableSubAsset, EditableSubAssetReference, EditableWorkspace, EditableWorkspaceEdit,
    SourceRecord, MAX_EDITABLE_WORKSPACE_ENCODED_BYTES,
};
use std::fs;
use std::sync::{Arc, Barrier};

fn asset(id: u128) -> EditableAsset {
    EditableAsset::new(
        AssetId(id),
        7,
        0,
        vec![],
        SourceRecord::embedded("source.raw", "test/raw", b"source".to_vec()).unwrap(),
        vec![EditableSubAsset {
            id: 1,
            name: "part".into(),
            kind: 9,
        }],
        vec![],
        vec![],
        b"payload".to_vec(),
        None,
    )
    .unwrap()
}

fn edit(previous: &EditableAsset, replacement: EditableAsset) -> EditableWorkspaceEdit {
    EditableWorkspaceEdit {
        expected_hash: ContentHash::digest(&previous.encode().unwrap()),
        replacement,
    }
}

fn write(root: &std::path::Path, value: &EditableAsset) {
    value
        .write_to_path(root.join(format!("{}.prism", value.asset.0)))
        .unwrap();
}

#[test]
fn save_updates_references_in_one_batch_and_receipt_supports_explicit_undo() {
    let root = tempfile::tempdir().unwrap();
    let target = asset(10);
    let mut consumer = asset(20);
    consumer
        .subasset_references
        .push(EditableSubAssetReference {
            asset: target.asset,
            subasset: 1,
            kind: EditableDependencyKind::Hard,
        });
    write(root.path(), &target);
    write(root.path(), &consumer);
    let workspace = EditableWorkspace::open(root.path()).unwrap();
    let mut next_target = target.clone();
    next_target.subassets[0].id = 2;
    let mut next_consumer = consumer.clone();
    next_consumer.subasset_references[0].subasset = 2;
    next_consumer.authoring_payload = b"ref:2".to_vec();
    // Input ordering does not influence publication or the receipt.
    let receipt = workspace
        .save(
            vec![
                edit(&consumer, next_consumer.clone()),
                edit(&target, next_target.clone()),
            ],
            MAX_EDITABLE_WORKSPACE_ENCODED_BYTES,
        )
        .unwrap();
    assert_eq!(receipt.updated_assets, [target.asset, consumer.asset]);
    let snapshot = workspace
        .snapshot(MAX_EDITABLE_WORKSPACE_ENCODED_BYTES)
        .unwrap();
    assert_eq!(snapshot[&target.asset], next_target);
    assert_eq!(snapshot[&consumer.asset], next_consumer);
    let reopened = EditableWorkspace::open(root.path()).unwrap();
    reopened
        .save(
            vec![
                EditableWorkspaceEdit {
                    expected_hash: receipt.content_hashes[&target.asset],
                    replacement: target.clone(),
                },
                EditableWorkspaceEdit {
                    expected_hash: receipt.content_hashes[&consumer.asset],
                    replacement: consumer.clone(),
                },
            ],
            MAX_EDITABLE_WORKSPACE_ENCODED_BYTES,
        )
        .unwrap();
    let snapshot = reopened
        .snapshot(MAX_EDITABLE_WORKSPACE_ENCODED_BYTES)
        .unwrap();
    assert_eq!(snapshot[&target.asset], target);
    assert_eq!(snapshot[&consumer.asset], consumer);
}

#[test]
fn stale_preview_only_edit_rejects_the_entire_batch_without_staging() {
    let root = tempfile::tempdir().unwrap();
    let first = asset(1);
    let second = asset(2);
    write(root.path(), &first);
    write(root.path(), &second);
    let workspace = EditableWorkspace::open(root.path()).unwrap();
    let mut changed = second.clone();
    changed.preview = Some(b"new preview".to_vec());
    workspace
        .save(
            vec![edit(&second, changed.clone())],
            MAX_EDITABLE_WORKSPACE_ENCODED_BYTES,
        )
        .unwrap();
    let mut next_first = first.clone();
    next_first.authoring_payload = b"edited".to_vec();
    let result = workspace.save(
        vec![edit(&first, next_first), edit(&second, second.clone())],
        MAX_EDITABLE_WORKSPACE_ENCODED_BYTES,
    );
    assert_eq!(
        result,
        Err(EditableAssetError::WorkspaceEditConflict {
            asset: second.asset,
            expected: ContentHash::digest(&second.encode().unwrap()),
            actual: ContentHash::digest(&changed.encode().unwrap()),
        })
    );
    assert_eq!(
        EditableAsset::read_from_path(root.path().join("1.prism")).unwrap(),
        first
    );
    assert_eq!(
        EditableAsset::read_from_path(root.path().join("2.prism")).unwrap(),
        changed
    );
    assert_eq!(
        fs::read_dir(root.path().join(".prism-editable-transactions"))
            .unwrap()
            .count(),
        0
    );
}

#[test]
fn dangling_subasset_rejects_the_save_before_any_package_changes() {
    let root = tempfile::tempdir().unwrap();
    let first = asset(1);
    let mut second = asset(2);
    second.subasset_references.push(EditableSubAssetReference {
        asset: first.asset,
        subasset: 1,
        kind: EditableDependencyKind::Hard,
    });
    write(root.path(), &first);
    write(root.path(), &second);
    let mut next = first.clone();
    next.subassets.clear();
    let workspace = EditableWorkspace::open(root.path()).unwrap();
    assert_eq!(
        workspace.save(
            vec![edit(&first, next)],
            MAX_EDITABLE_WORKSPACE_ENCODED_BYTES
        ),
        Err(EditableAssetError::MissingReferencedSubAsset {
            asset: first.asset,
            subasset: 1
        })
    );
    assert_eq!(
        EditableAsset::read_from_path(root.path().join("1.prism")).unwrap(),
        first
    );
    assert!(!root.path().join(".prism-editable-transactions").exists());
}

#[test]
fn aggregate_budget_counts_unchanged_packages_before_encoding_outputs() {
    let root = tempfile::tempdir().unwrap();
    let first = asset(1);
    let second = asset(2);
    write(root.path(), &first);
    write(root.path(), &second);
    let old_bytes = (first.encode().unwrap().len() + second.encode().unwrap().len()) as u64;
    let mut next = first.clone();
    next.authoring_payload.extend_from_slice(&[42; 100]);
    let workspace = EditableWorkspace::open(root.path()).unwrap();
    assert_eq!(
        workspace.save(vec![edit(&first, next)], old_bytes),
        Err(EditableAssetError::WorkspaceEncodedBytesLimitExceeded {
            actual: old_bytes + 100,
            limit: old_bytes
        })
    );
    assert_eq!(
        EditableAsset::read_from_path(root.path().join("1.prism")).unwrap(),
        first
    );
    assert!(!root.path().join(".prism-editable-transactions").exists());
}

#[test]
fn noops_return_hashes_without_creating_a_transaction() {
    let root = tempfile::tempdir().unwrap();
    let first = asset(1);
    write(root.path(), &first);
    let workspace = EditableWorkspace::open(root.path()).unwrap();
    let receipt = workspace
        .save(
            vec![edit(&first, first.clone())],
            MAX_EDITABLE_WORKSPACE_ENCODED_BYTES,
        )
        .unwrap();
    assert!(receipt.updated_assets.is_empty());
    assert_eq!(
        receipt.content_hashes[&first.asset],
        ContentHash::digest(&first.encode().unwrap())
    );
    assert!(!root.path().join(".prism-editable-transactions").exists());
    assert!(workspace
        .save(vec![], MAX_EDITABLE_WORKSPACE_ENCODED_BYTES)
        .unwrap()
        .content_hashes
        .is_empty());
}

#[test]
fn malformed_duplicate_and_unknown_edits_do_not_publish() {
    let root = tempfile::tempdir().unwrap();
    let first = asset(1);
    write(root.path(), &first);
    let workspace = EditableWorkspace::open(root.path()).unwrap();
    let request = edit(&first, first.clone());
    assert_eq!(
        workspace.save(
            vec![request.clone(), request],
            MAX_EDITABLE_WORKSPACE_ENCODED_BYTES
        ),
        Err(EditableAssetError::DuplicateWorkspaceAsset { asset: first.asset })
    );
    assert_eq!(
        workspace.save(
            vec![edit(&first, asset(99))],
            MAX_EDITABLE_WORKSPACE_ENCODED_BYTES
        ),
        Err(EditableAssetError::WorkspaceAssetNotFound { asset: AssetId(99) })
    );
    let mut invalid = first.clone();
    invalid.source.embedded_source = Some(b"wrong".to_vec());
    assert_eq!(
        workspace.save(
            vec![edit(&first, invalid)],
            MAX_EDITABLE_WORKSPACE_ENCODED_BYTES
        ),
        Err(EditableAssetError::EmbeddedSourceHashMismatch)
    );
    for limit in [0, MAX_EDITABLE_WORKSPACE_ENCODED_BYTES + 1] {
        assert!(matches!(
            workspace.save(vec![], limit),
            Err(EditableAssetError::InvalidWorkspaceSnapshotBudget { .. })
        ));
    }
    assert!(!root.path().join(".prism-editable-transactions").exists());
}

#[test]
fn concurrent_editors_with_the_same_baseline_have_exactly_one_winner() {
    let root = tempfile::tempdir().unwrap();
    let first = asset(1);
    write(root.path(), &first);
    let barrier = Arc::new(Barrier::new(2));
    let workspaces = [
        EditableWorkspace::open(root.path()).unwrap(),
        EditableWorkspace::open(root.path()).unwrap(),
    ];
    let results = std::thread::scope(|scope| {
        let handles = workspaces
            .into_iter()
            .enumerate()
            .map(|(i, workspace)| {
                let barrier = barrier.clone();
                let mut next = first.clone();
                next.authoring_payload = vec![i as u8];
                let request = edit(&first, next);
                scope.spawn(move || {
                    barrier.wait();
                    workspace.save(vec![request], MAX_EDITABLE_WORKSPACE_ENCODED_BYTES)
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect::<Vec<_>>()
    });
    assert_eq!(results.iter().filter(|r| r.is_ok()).count(), 1);
    assert_eq!(
        results
            .iter()
            .filter(|r| matches!(r, Err(EditableAssetError::WorkspaceEditConflict { .. })))
            .count(),
        1
    );
    let saved = EditableAsset::read_from_path(root.path().join("1.prism")).unwrap();
    assert!(saved.authoring_payload == [0] || saved.authoring_payload == [1]);
}
