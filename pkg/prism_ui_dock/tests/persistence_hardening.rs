#![allow(missing_docs, reason = "integration tests are not public API")]

extern crate alloc;

use alloc::sync::Arc;
use prism_ui_dock::{
    encode_workspace, rewrite_workspace_version, DockLayout, DockNode, MigrationError,
    MigrationStep, PanelRegistry, WorkspaceDecodeLimits, WorkspaceFileStore, WorkspaceLoadSource,
    WorkspaceManager, WorkspaceMigrationRegistry, WorkspacePersistError, WorkspaceStoreError,
    CURRENT_WORKSPACE_VERSION,
};
use std::{
    fs,
    path::PathBuf,
    sync::{
        atomic::{AtomicU64, Ordering},
        Barrier,
    },
    thread,
    time::{SystemTime, UNIX_EPOCH},
};

static TEST_SEQUENCE: AtomicU64 = AtomicU64::new(0);

fn temp_store() -> (WorkspaceFileStore, PathBuf) {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let sequence = TEST_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let directory = std::env::temp_dir().join(format!(
        "prism-dock-persistence-hardening-{}-{sequence}",
        nonce
    ));
    (
        WorkspaceFileStore::new(directory.join("workspace.prism")),
        directory,
    )
}

fn manager(panel: &str) -> WorkspaceManager {
    let mut manager = WorkspaceManager::new();
    manager.add_workspace("coding").unwrap();
    manager
        .add_perspective(
            "default",
            &DockLayout {
                root: DockNode::panel(panel),
                ..DockLayout::default()
            },
        )
        .unwrap();
    manager
}

fn legacy_workspace(bytes: &[u8]) -> Vec<u8> {
    rewrite_workspace_version(bytes, 0).unwrap()
}

fn oversized_migration(bytes: &[u8]) -> Result<Vec<u8>, MigrationError> {
    let mut output = rewrite_workspace_version(bytes, CURRENT_WORKSPACE_VERSION)?;
    output.extend_from_slice(&[0; 32]);
    Ok(output)
}

fn migration_registry(
    step: fn(&[u8]) -> Result<Vec<u8>, MigrationError>,
) -> WorkspaceMigrationRegistry {
    let mut registry = WorkspaceMigrationRegistry::new();
    registry
        .register(MigrationStep {
            from: 0,
            to: CURRENT_WORKSPACE_VERSION,
            migrate: step,
        })
        .unwrap();
    registry
}

#[test]
fn repeated_saves_keep_the_previous_commit_as_last_good() {
    let (store, directory) = temp_store();
    for index in 0..32 {
        store.save(&manager(&format!("panel-{index}"))).unwrap();
    }

    let loaded = store.load(Default::default(), WorkspaceManager::new());
    assert_eq!(loaded.source, WorkspaceLoadSource::Primary);
    assert_eq!(
        loaded.manager.active_layout().unwrap().root,
        DockNode::panel("panel-31")
    );

    let previous = WorkspaceFileStore::new(store.last_good_path())
        .load(Default::default(), WorkspaceManager::new());
    assert_eq!(previous.source, WorkspaceLoadSource::Primary);
    assert_eq!(
        previous.manager.active_layout().unwrap().root,
        DockNode::panel("panel-30")
    );

    let entries = fs::read_dir(&directory)
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .collect::<Vec<_>>();
    assert_eq!(entries.len(), 3);
    let _ = fs::remove_dir_all(directory);
}

#[test]
fn concurrent_saves_leave_primary_and_last_good_decodable() {
    let (store, directory) = temp_store();
    let workers = 8;
    let barrier = Arc::new(Barrier::new(workers));
    let handles = (0..workers)
        .map(|index| {
            let barrier = Arc::clone(&barrier);
            let store = store.clone();
            thread::spawn(move || {
                barrier.wait();
                store.save(&manager(&format!("panel-{index}")))
            })
        })
        .collect::<Vec<_>>();

    for handle in handles {
        handle.join().unwrap().unwrap();
    }

    let primary = store.load(Default::default(), WorkspaceManager::new());
    assert_eq!(primary.source, WorkspaceLoadSource::Primary);
    assert!(matches!(
        primary.manager.active_layout().unwrap().root,
        DockNode::TabStack { tabs, active: 0 } if tabs.len() == 1
    ));

    let last_good = WorkspaceFileStore::new(store.last_good_path())
        .load(Default::default(), WorkspaceManager::new());
    assert_eq!(last_good.source, WorkspaceLoadSource::Primary);
    assert!(matches!(
        last_good.manager.active_layout().unwrap().root,
        DockNode::TabStack { tabs, active: 0 } if tabs.len() == 1
    ));
    let _ = fs::remove_dir_all(directory);
}

#[test]
fn registry_migration_rejects_oversized_input_before_decode() {
    let (store, directory) = temp_store();
    fs::create_dir_all(&directory).unwrap();
    fs::write(store.primary_path(), vec![0; 4096]).unwrap();

    let limits = WorkspaceDecodeLimits {
        max_bytes: 64,
        ..WorkspaceDecodeLimits::default()
    };
    let loaded = store.load_with_registry_and_migration(
        &WorkspaceMigrationRegistry::new(),
        &PanelRegistry::new(),
        None,
        limits,
        WorkspaceManager::new(),
    );

    assert_eq!(loaded.source, WorkspaceLoadSource::Default);
    assert!(matches!(
        loaded.primary_error,
        Some(WorkspaceStoreError::Codec(
            WorkspacePersistError::ResourceLimit("bytes")
        ))
    ));
    let _ = fs::remove_dir_all(directory);
}

#[test]
fn registry_migration_applies_limits_to_each_migration_result() {
    let (store, directory) = temp_store();
    let canonical = encode_workspace(&manager("panel")).unwrap();
    let legacy = legacy_workspace(&canonical);
    fs::create_dir_all(&directory).unwrap();
    fs::write(store.primary_path(), &legacy).unwrap();

    let limits = WorkspaceDecodeLimits {
        max_bytes: legacy.len(),
        ..WorkspaceDecodeLimits::default()
    };
    let loaded = store.load_with_registry_and_migration(
        &migration_registry(oversized_migration),
        &PanelRegistry::new(),
        None,
        limits,
        WorkspaceManager::new(),
    );

    assert_eq!(loaded.source, WorkspaceLoadSource::Default);
    assert!(matches!(
        loaded.primary_error,
        Some(WorkspaceStoreError::Migration(
            MigrationError::InvalidWorkspaceOutput(WorkspacePersistError::ResourceLimit("bytes"))
        ))
    ));
    let _ = fs::remove_dir_all(directory);
}
