use prism_asset_build_core::{
    AssetId, ContentHash, EditableAsset, EditableAssetError, EditableDependency,
    EditableDependencyKind as Kind, EditableSubAsset, EditableSubAssetReference, EditableWorkspace,
    EditableWorkspaceMutation as Mutation, SourceRecord,
    MAX_EDITABLE_WORKSPACE_ENCODED_BYTES as BUDGET,
};
use std::fs;
use std::path::Path;

fn asset(id: u128) -> EditableAsset {
    EditableAsset::new(
        AssetId(id),
        7,
        0,
        vec![],
        SourceRecord::embedded("source.raw", "test/raw", vec![1, 2]).unwrap(),
        vec![EditableSubAsset {
            id: 1,
            name: "part".into(),
            kind: 7,
        }],
        vec![],
        vec![],
        vec![id as u8; 3],
        None,
    )
    .unwrap()
}

fn hash(asset: &EditableAsset) -> ContentHash {
    ContentHash::digest(&asset.encode().unwrap())
}

fn put(path: &str, previous: Option<&EditableAsset>, value: EditableAsset) -> Mutation {
    Mutation::Put {
        expected_hash: previous.map(hash),
        relative_path: path.into(),
        replacement: value,
    }
}

fn remove(value: &EditableAsset) -> Mutation {
    Mutation::Remove {
        asset: value.asset,
        expected_hash: hash(value),
    }
}

fn read(root: &Path, path: &str) -> EditableAsset {
    EditableAsset::read_from_path(root.join(path)).unwrap()
}

#[test]
fn create_replace_move_remove_and_receipts_preserve_identity() {
    let root = tempfile::tempdir().unwrap();
    fs::create_dir(root.path().join("nested")).unwrap();
    let workspace = EditableWorkspace::open(root.path()).unwrap();
    let original = asset(10);
    let created = workspace
        .mutate(vec![put("a.prism", None, original.clone())], BUDGET)
        .unwrap();
    assert_eq!(created.changes.len(), 1);
    assert_eq!(created.changes[0].old_hash, None);
    assert_eq!(created.changes[0].new_hash, Some(hash(&original)));
    let mut replacement = original.clone();
    replacement.authoring_payload = b"edited".to_vec();
    let moved = workspace
        .mutate(
            vec![put("nested/b.prism", Some(&original), replacement.clone())],
            BUDGET,
        )
        .unwrap();
    assert!(!root.path().join("a.prism").exists());
    assert_eq!(read(root.path(), "nested/b.prism"), replacement);
    assert_eq!(
        moved.changes[0].old_path.as_deref(),
        Some(Path::new("a.prism"))
    );
    assert_eq!(
        moved.changes[0].new_path.as_deref(),
        Some(Path::new("nested/b.prism"))
    );
    let reopened = EditableWorkspace::open(root.path()).unwrap();
    let removed = reopened.mutate(vec![remove(&replacement)], BUDGET).unwrap();
    assert_eq!(removed.changes[0].asset, original.asset);
    assert_eq!(removed.changes[0].new_hash, None);
    assert!(reopened.snapshot(BUDGET).unwrap().is_empty());
    // Caller-retained data makes undo explicit and guards identity reuse.
    reopened
        .mutate(vec![put("a.prism", None, original.clone())], BUDGET)
        .unwrap();
    assert_eq!(read(root.path(), "a.prism"), original);
}

#[test]
fn path_swaps_and_remove_then_reuse_are_one_batch() {
    let root = tempfile::tempdir().unwrap();
    let workspace = EditableWorkspace::open(root.path()).unwrap();
    let a = asset(1);
    let b = asset(2);
    workspace
        .mutate(
            vec![
                put("a.prism", None, a.clone()),
                put("b.prism", None, b.clone()),
            ],
            BUDGET,
        )
        .unwrap();
    let swapped = workspace
        .mutate(
            vec![
                put("a.prism", Some(&b), b.clone()),
                put("b.prism", Some(&a), a.clone()),
            ],
            BUDGET,
        )
        .unwrap();
    assert_eq!(
        swapped
            .changes
            .iter()
            .map(|change| change.asset)
            .collect::<Vec<_>>(),
        [a.asset, b.asset]
    );
    assert_eq!(read(root.path(), "a.prism"), b);
    assert_eq!(read(root.path(), "b.prism"), a);
    let c = asset(3);
    workspace
        .mutate(vec![remove(&b), put("a.prism", None, c.clone())], BUDGET)
        .unwrap();
    assert_eq!(read(root.path(), "a.prism"), c);
    assert_eq!(workspace.snapshot(BUDGET).unwrap().len(), 2);
}

#[test]
fn deletion_checks_hard_editor_and_explicit_references_but_allows_missing_soft() {
    for kind in [Kind::Hard, Kind::EditorOnly, Kind::Soft] {
        for explicit in [false, true] {
            let root = tempfile::tempdir().unwrap();
            let target = asset(1);
            let mut consumer = asset(2);
            if explicit {
                consumer
                    .subasset_references
                    .push(EditableSubAssetReference {
                        asset: target.asset,
                        subasset: 1,
                        kind,
                    });
            } else {
                consumer.dependencies.push(EditableDependency {
                    asset: target.asset,
                    kind,
                });
            }
            let workspace = EditableWorkspace::open(root.path()).unwrap();
            workspace
                .mutate(
                    vec![
                        put("a.prism", None, target.clone()),
                        put("b.prism", None, consumer.clone()),
                    ],
                    BUDGET,
                )
                .unwrap();
            let result = workspace.mutate(vec![remove(&target)], BUDGET);
            if kind == Kind::Soft {
                result.unwrap();
                assert!(!root.path().join("a.prism").exists());
                assert_eq!(read(root.path(), "b.prism"), consumer);
            } else {
                assert_eq!(
                    result,
                    Err(EditableAssetError::MissingReferencedAsset {
                        asset: target.asset
                    })
                );
                assert_eq!(read(root.path(), "a.prism"), target);
                consumer.dependencies.clear();
                consumer.subasset_references.clear();
                let previous = read(root.path(), "b.prism");
                workspace
                    .mutate(
                        vec![
                            remove(&target),
                            put("b.prism", Some(&previous), consumer.clone()),
                        ],
                        BUDGET,
                    )
                    .unwrap();
                assert_eq!(read(root.path(), "b.prism"), consumer);
            }
        }
    }
}

#[test]
fn stale_and_duplicate_mutations_never_publish_another_candidate() {
    let root = tempfile::tempdir().unwrap();
    let workspace = EditableWorkspace::open(root.path()).unwrap();
    let a = asset(1);
    workspace
        .mutate(vec![put("a.prism", None, a.clone())], BUDGET)
        .unwrap();
    let mut stale = a.clone();
    stale.preview = Some(vec![5]);
    assert!(matches!(
        workspace.mutate(
            vec![
                put("new.prism", None, asset(2)),
                put("move.prism", Some(&stale), a.clone())
            ],
            BUDGET
        ),
        Err(EditableAssetError::WorkspaceEditConflict { .. })
    ));
    assert!(!root.path().join("new.prism").exists());
    assert_eq!(read(root.path(), "a.prism"), a);
    assert_eq!(
        workspace.mutate(
            vec![remove(&a), put("b.prism", Some(&a), a.clone())],
            BUDGET
        ),
        Err(EditableAssetError::DuplicateWorkspaceAsset { asset: a.asset })
    );
    assert_eq!(
        workspace.mutate(vec![put("b.prism", None, a.clone())], BUDGET),
        Err(EditableAssetError::DuplicateWorkspaceAsset { asset: a.asset })
    );
    assert!(matches!(
        workspace.mutate(vec![remove(&asset(99))], BUDGET),
        Err(EditableAssetError::WorkspaceAssetNotFound { .. })
    ));
}

#[test]
fn paths_budgets_and_noops_are_checked_before_staging() {
    let root = tempfile::tempdir().unwrap();
    let workspace = EditableWorkspace::open(root.path()).unwrap();
    let a = asset(1);
    let b = asset(2);
    a.write_to_path(root.path().join("a.prism")).unwrap();
    // A valid no-op must not even create the transaction subtree.
    assert!(workspace
        .mutate(vec![put("a.prism", Some(&a), a.clone())], BUDGET)
        .unwrap()
        .changes
        .is_empty());
    assert!(!root.path().join(".prism-editable-transactions").exists());
    assert!(workspace.mutate(vec![], BUDGET).unwrap().changes.is_empty());
    for bad in [
        "../escape.prism",
        "/absolute.prism",
        ".prism-editable-transactions/hidden.prism",
        "missing/parent/b.prism",
        "bad.txt",
    ] {
        assert!(workspace
            .mutate(vec![put(bad, None, b.clone())], BUDGET)
            .is_err());
    }
    assert!(matches!(
        workspace.mutate(vec![put("a.prism", None, b.clone())], BUDGET),
        Err(EditableAssetError::WorkspacePathConflict { .. })
    ));
    assert!(matches!(
        workspace.mutate(
            vec![put("b.prism", None, b.clone())],
            a.encode().unwrap().len() as u64
        ),
        Err(EditableAssetError::WorkspaceEncodedBytesLimitExceeded { .. })
    ));
    assert!(workspace
        .mutate(
            vec![
                put("same.prism", Some(&a), a.clone()),
                put("same.prism", None, b)
            ],
            BUDGET
        )
        .is_err());
    assert_eq!(read(root.path(), "a.prism"), a);
    assert!(!root.path().join(".prism-editable-transactions").exists());
}

#[test]
fn candidate_hard_cycles_fail_without_files_and_soft_cycles_are_allowed() {
    for kind in [Kind::Hard, Kind::Soft] {
        let root = tempfile::tempdir().unwrap();
        let workspace = EditableWorkspace::open(root.path()).unwrap();
        let mut a = asset(1);
        let mut b = asset(2);
        a.dependencies.push(EditableDependency {
            asset: b.asset,
            kind,
        });
        b.dependencies.push(EditableDependency {
            asset: a.asset,
            kind,
        });
        let result = workspace.mutate(
            vec![
                put("a.prism", None, a.clone()),
                put("b.prism", None, b.clone()),
            ],
            BUDGET,
        );
        if kind == Kind::Hard {
            assert_eq!(
                result,
                Err(EditableAssetError::WorkspaceDependencyCycle {
                    assets: vec![a.asset, b.asset]
                })
            );
            assert!(!root.path().join("a.prism").exists());
            assert!(!root.path().join(".prism-editable-transactions").exists());
        } else {
            result.unwrap();
            assert_eq!(workspace.snapshot(BUDGET).unwrap().len(), 2);
        }
    }
}

#[cfg(unix)]
#[test]
fn symlinked_destinations_and_ancestors_cannot_modify_outside_workspace() {
    let root = tempfile::tempdir().unwrap();
    let external = tempfile::tempdir().unwrap();
    let workspace = EditableWorkspace::open(root.path()).unwrap();
    let original = asset(99);
    original
        .write_to_path(external.path().join("external.prism"))
        .unwrap();
    std::os::unix::fs::symlink(external.path(), root.path().join("link")).unwrap();
    std::os::unix::fs::symlink(
        external.path().join("external.prism"),
        root.path().join("link.prism"),
    )
    .unwrap();
    for path in ["link/created.prism", "link/external.prism", "link.prism"] {
        assert!(workspace
            .mutate(vec![put(path, None, asset(1))], BUDGET)
            .is_err());
    }
    assert_eq!(read(external.path(), "external.prism"), original);
    assert!(!external.path().join("created.prism").exists());
}
