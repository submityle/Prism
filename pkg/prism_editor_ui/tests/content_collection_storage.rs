//! Real project-file persistence and restored Loom/controller behavior.
use prism_editor_content::{AssetId, AssetKind, AssetRecord, ContentCatalog};
use prism_editor_ui::{
    CollectionStorageError as Error, CollectionStore, ContentBrowserAction as Action,
    ContentBrowserController as Controller, ContentBrowserError, SelectionMode, UiEvent,
    MAX_COLLECTION_FILE_BYTES,
};
use prism_ui::{Element, Key, RecordingBackend, Ui};
use std::{
    collections::BTreeSet,
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

static SERIAL: AtomicU64 = AtomicU64::new(1);
struct Project(PathBuf);
impl Project {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "prism-collection-test-{}-{}",
            std::process::id(),
            SERIAL.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn store(&self) -> CollectionStore {
        CollectionStore::for_project(&self.0).unwrap()
    }
}
impl Drop for Project {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
fn id(raw: u128) -> AssetId {
    AssetId::from_raw(raw)
}
fn catalog() -> ContentCatalog {
    let mut catalog = ContentCatalog::new();
    for raw in [0, 7, u128::MAX] {
        catalog
            .insert(
                AssetRecord::new(
                    id(raw),
                    format!("{raw}.png"),
                    AssetKind::new("texture").unwrap(),
                )
                .unwrap(),
            )
            .unwrap();
    }
    catalog
}
fn populated(catalog: &mut ContentCatalog) -> Controller {
    let mut controller = Controller::default();
    controller.build(catalog).unwrap();
    for raw in [0, 7, u128::MAX] {
        controller
            .apply(
                catalog,
                Action::Select {
                    asset: id(raw),
                    mode: SelectionMode::Add,
                },
            )
            .unwrap();
    }
    controller
        .apply(catalog, Action::CreateCollection("项目贴图 🖼".into()))
        .unwrap();
    controller
        .apply(catalog, Action::AddSelectionToCollection(1))
        .unwrap();
    controller
        .apply(catalog, Action::CreateCollection("deleted".into()))
        .unwrap();
    controller
        .apply(catalog, Action::DeleteCollection(2))
        .unwrap();
    controller
}
fn find<'a>(element: &'a Element, key: &str) -> Option<&'a Element> {
    if matches!(element.explicit_key(), Some(Key::Str(value)) if value==key) {
        return Some(element);
    }
    element
        .child_elements()
        .iter()
        .find_map(|child| find(child, key))
}
fn checksum(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}
fn seal(mut payload: Vec<u8>) -> Vec<u8> {
    payload.extend(checksum(&payload).to_le_bytes());
    payload
}
fn raw_file(next: u64, entries: &[(u64, &[u8], &[u128])]) -> Vec<u8> {
    let mut bytes = b"PRISMCOL".to_vec();
    bytes.extend(1u32.to_le_bytes());
    bytes.extend(next.to_le_bytes());
    bytes.extend((entries.len() as u32).to_le_bytes());
    for (id, name, members) in entries {
        bytes.extend(id.to_le_bytes());
        bytes.extend((name.len() as u32).to_le_bytes());
        bytes.extend(*name);
        bytes.extend((members.len() as u32).to_le_bytes());
        for asset in *members {
            bytes.extend(asset.to_le_bytes());
        }
    }
    seal(bytes)
}

#[test]
fn project_roundtrip_preserves_identity_high_water_mark_and_restores_real_loom_actions() {
    let project = Project::new();
    let store = project.store();
    let mut catalog = catalog();
    let original = populated(&mut catalog);
    let saved = store.save(&original).unwrap();
    assert!(saved.durability_warning.is_none());
    let bytes = fs::read(store.path()).unwrap();
    assert_eq!(saved.bytes, bytes.len());
    assert_eq!(store.save(&original).unwrap().bytes, saved.bytes);
    assert_eq!(bytes, fs::read(store.path()).unwrap());
    let mut restored = Controller::default();
    let old = restored.build(&mut catalog).unwrap();
    let mut ui = Ui::new(RecordingBackend::new());
    ui.mount(old.element());
    let report = project.store().load_into(&mut restored, &catalog).unwrap();
    assert!(!report.missing_file);
    assert_eq!(report.members, 3);
    assert!(report.missing_members.is_empty());
    assert_eq!(original.collections(), restored.collections());
    assert_eq!(
        restored.handle_event(&mut catalog, &old, "content:home", &UiEvent::Activate),
        Err(ContentBrowserError::StaleView)
    );
    ui.backend_mut().clear();
    let view = restored.build(&mut catalog).unwrap();
    ui.update(view.element());
    assert!(
        !ui.backend().is_empty(),
        "loaded collections reconcile actual Loom nodes"
    );
    assert!(find(view.element(), "content:collection:open:1").is_some());
    let Some(Key::Str(target)) = find(view.element(), "content:collection:open:1")
        .unwrap()
        .explicit_key()
    else {
        panic!("real collection target");
    };
    restored
        .handle_event(&mut catalog, &view, target, &UiEvent::Activate)
        .unwrap();
    let view = restored.build(&mut catalog).unwrap();
    ui.update(view.element());
    for raw in [0, 7, u128::MAX] {
        assert!(find(view.element(), &format!("content:asset:{raw}")).is_some());
    }
    ui.backend_mut().clear();
    let unchanged = restored.build(&mut catalog).unwrap();
    ui.update(unchanged.element());
    assert!(
        ui.backend().is_empty(),
        "unchanged restored tree has zero backend mutations"
    );
    restored
        .apply(
            &mut catalog,
            Action::CreateCollection("after restart".into()),
        )
        .unwrap();
    assert_eq!(restored.collections()[1].id(), 3);
    let metadata: Vec<_> = fs::read_dir(store.path().parent().unwrap())
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .collect();
    assert_eq!(
        metadata.len(),
        2,
        "only final file and permanent lock, no temp"
    );
}

#[test]
fn process_restart_loads_project_collections_and_preserves_next_identity() {
    const CHILD_PROJECT: &str = "PRISM_COLLECTION_STORAGE_CHILD_PROJECT";
    if let Some(root) = std::env::var_os(CHILD_PROJECT) {
        let store = CollectionStore::for_project(PathBuf::from(root)).unwrap();
        let mut controller = Controller::default();
        let mut catalog = catalog();
        let report = store.load_into(&mut controller, &catalog).unwrap();
        assert_eq!(report.collections, 1);
        assert_eq!(report.members, 3);
        controller
            .apply(
                &mut catalog,
                Action::CreateCollection("child process".into()),
            )
            .unwrap();
        assert_eq!(controller.collections()[1].id(), 3);
        store.save(&controller).unwrap();
        return;
    }
    let project = Project::new();
    let mut catalog = catalog();
    project.store().save(&populated(&mut catalog)).unwrap();
    let child = std::process::Command::new(std::env::current_exe().unwrap())
        .arg("--exact")
        .arg("process_restart_loads_project_collections_and_preserves_next_identity")
        .env(CHILD_PROJECT, &project.0)
        .output()
        .unwrap();
    assert!(
        child.status.success(),
        "child stdout={} stderr={}",
        String::from_utf8_lossy(&child.stdout),
        String::from_utf8_lossy(&child.stderr)
    );
    let mut controller = Controller::default();
    project
        .store()
        .load_into(&mut controller, &catalog)
        .unwrap();
    assert_eq!(controller.collections()[1].name(), "child process");
    controller
        .apply(
            &mut catalog,
            Action::CreateCollection("parent reload".into()),
        )
        .unwrap();
    assert_eq!(controller.collections()[2].id(), 4);
}

#[test]
fn maximum_valid_capacity_roundtrips_with_bounded_missing_reference_report() {
    let project = Project::new();
    let store = project.store();
    fs::create_dir(project.0.join(".prism")).unwrap();
    let members: Vec<u128> = (0..4096).collect();
    let name = "n".repeat(4096);
    let entries: Vec<_> = (1..=64)
        .map(|collection| (collection, name.as_bytes(), members.as_slice()))
        .collect();
    let bytes = raw_file(u64::MAX, &entries);
    assert!(bytes.len() < MAX_COLLECTION_FILE_BYTES);
    fs::write(store.path(), &bytes).unwrap();
    let mut controller = Controller::default();
    let mut catalog = ContentCatalog::new();
    let report = store.load_into(&mut controller, &catalog).unwrap();
    assert_eq!(report.collections, 64);
    assert_eq!(report.members, 64 * 4096);
    assert_eq!(report.missing_members.len(), 64 * 4096);
    store.save(&controller).unwrap();
    assert_eq!(fs::read(store.path()).unwrap(), bytes);
    controller
        .apply(&mut catalog, Action::DeleteCollection(64))
        .unwrap();
    assert!(
        controller
            .apply(&mut catalog, Action::CreateCollection("exhausted".into()))
            .is_err(),
        "exhausted monotonic identity survives restart"
    );
}

#[test]
fn readers_observe_only_complete_old_or_new_files_during_atomic_publication() {
    use std::sync::{Arc, Barrier};
    let project = Project::new();
    let store = project.store();
    let mut catalog = catalog();
    let mut controller = populated(&mut catalog);
    store.save(&controller).unwrap();
    let old = fs::read(store.path()).unwrap();
    controller
        .apply(
            &mut catalog,
            Action::RenameCollection {
                id: 1,
                name: "replacement".into(),
            },
        )
        .unwrap();
    store.save(&controller).unwrap();
    let new = fs::read(store.path()).unwrap();
    controller
        .apply(
            &mut catalog,
            Action::RenameCollection {
                id: 1,
                name: "项目贴图 🖼".into(),
            },
        )
        .unwrap();
    store.save(&controller).unwrap();
    let barrier = Arc::new(Barrier::new(2));
    let reader_barrier = barrier.clone();
    let path = store.path();
    let reader = std::thread::spawn(move || {
        reader_barrier.wait();
        for _ in 0..4096 {
            let actual = fs::read(&path).unwrap();
            assert!(
                actual == old || actual == new,
                "no truncated or mixed generation"
            );
        }
    });
    barrier.wait();
    for index in 0..32 {
        let name = if index % 2 == 0 {
            "replacement"
        } else {
            "项目贴图 🖼"
        };
        controller
            .apply(
                &mut catalog,
                Action::RenameCollection {
                    id: 1,
                    name: name.into(),
                },
            )
            .unwrap();
        store.save(&controller).unwrap();
    }
    reader.join().unwrap();
    assert_eq!(
        fs::read_dir(store.path().parent().unwrap())
            .unwrap()
            .count(),
        2
    );
}

#[test]
fn missing_deleted_and_unscanned_references_are_reported_retained_and_never_fake_rows() {
    let project = Project::new();
    let store = project.store();
    let mut catalog = catalog();
    let original = populated(&mut catalog);
    store.save(&original).unwrap();
    let mut empty = ContentCatalog::new();
    let mut restored = Controller::default();
    let report = store.load_into(&mut restored, &empty).unwrap();
    assert_eq!(
        report
            .missing_members
            .iter()
            .map(|member| (member.collection, member.asset))
            .collect::<Vec<_>>(),
        vec![(1, id(0)), (1, id(7)), (1, id(u128::MAX))]
    );
    restored
        .apply(&mut empty, Action::OpenCollection(Some(1)))
        .unwrap();
    let view = restored.build(&mut empty).unwrap();
    assert!(find(view.element(), "content:asset:7").is_none());
    empty.insert(catalog.get(id(7)).unwrap().clone()).unwrap();
    let view = restored.build(&mut empty).unwrap();
    assert!(find(view.element(), "content:asset:7").is_some());
    restored
        .apply(
            &mut empty,
            Action::Select {
                asset: id(7),
                mode: SelectionMode::Replace,
            },
        )
        .unwrap();
    restored
        .apply(&mut empty, Action::RemoveSelectionFromCollection(1))
        .unwrap();
    store.save(&restored).unwrap();
    let mut again = Controller::default();
    store.load_into(&mut again, &catalog).unwrap();
    assert_eq!(
        again.collections()[0].members(),
        &BTreeSet::from([id(0), id(u128::MAX)])
    );
}

#[test]
fn missing_file_resets_other_project_and_failed_corrupt_load_preserves_state_and_disk() {
    let project = Project::new();
    let store = project.store();
    let mut catalog = catalog();
    let mut controller = populated(&mut catalog);
    assert!(
        store
            .load_into(&mut controller, &catalog)
            .unwrap()
            .missing_file
    );
    assert!(controller.collections().is_empty());
    controller = populated(&mut catalog);
    store.save(&controller).unwrap();
    let before = controller.collections().to_vec();
    let good = fs::read(store.path()).unwrap();
    let mut cases = vec![Vec::new(), good[..20].to_vec()];
    let mut magic = good.clone();
    magic[0] ^= 1;
    cases.push(magic);
    let mut altered = good.clone();
    altered[35] ^= 1;
    cases.push(altered);
    let mut version = good.clone();
    version[8..12].copy_from_slice(&99u32.to_le_bytes());
    cases.push(version);
    cases.extend([
        raw_file(2, &[(0, b"x", &[])]),
        raw_file(u64::MAX, &[(u64::MAX, b"x", &[])]),
        raw_file(4, &[(2, b"x", &[]), (1, b"y", &[])]),
        raw_file(3, &[(1, b"x", &[]), (1, b"x", &[])]),
        raw_file(1, &[(1, b"x", &[])]),
        raw_file(0, &[]),
        raw_file(2, &[(1, b" ", &[])]),
        raw_file(2, &[(1, b" x", &[])]),
        raw_file(2, &[(1, b"", &[])]),
        raw_file(2, &[(1, &[0xff], &[])]),
        raw_file(2, &[(1, b"x", &[7, 7])]),
        raw_file(2, &[(1, b"x", &[8, 7])]),
    ]);
    let mut trailing = good[..good.len() - 8].to_vec();
    trailing.push(0);
    cases.push(seal(trailing));
    for bytes in cases {
        fs::write(store.path(), &bytes).unwrap();
        assert!(store.load_into(&mut controller, &catalog).is_err());
        assert!(
            store.save(&controller).is_err(),
            "save must not destroy corrupt or unsupported project data"
        );
        assert_eq!(controller.collections(), before);
        assert_eq!(fs::read(store.path()).unwrap(), bytes);
    }
    fs::write(store.path(), &good).unwrap();
    store.load_into(&mut controller, &catalog).unwrap();
    assert_eq!(controller.collections(), before);
}

#[test]
fn untrusted_counts_and_sparse_oversized_files_are_bounded_before_allocation() {
    let project = Project::new();
    let store = project.store();
    let mut catalog = catalog();
    let mut controller = populated(&mut catalog);
    store.save(&controller).unwrap();
    let good = fs::read(store.path()).unwrap();
    for (offset, limit, field) in [
        (20, 65, "collections"),
        (32, 4097, "name bytes"),
        (36 + "项目贴图 🖼".len(), 4097, "members"),
    ] {
        let mut bytes = good[..good.len() - 8].to_vec();
        bytes[offset..offset + 4].copy_from_slice(&(limit as u32).to_le_bytes());
        fs::write(store.path(), seal(bytes)).unwrap();
        assert_eq!(
            store.load_into(&mut controller, &catalog),
            Err(Error::Limit(field))
        );
    }
    fs::File::create(store.path())
        .unwrap()
        .set_len(MAX_COLLECTION_FILE_BYTES as u64 + 1)
        .unwrap();
    assert_eq!(
        store.load_into(&mut controller, &catalog),
        Err(Error::Limit("file bytes"))
    );
    assert_eq!(controller.collections().len(), 1);
}

#[test]
fn project_locks_report_busy_and_never_overwrite_previous_generation() {
    let project = Project::new();
    let store = project.store();
    let mut catalog = catalog();
    let mut controller = populated(&mut catalog);
    store.save(&controller).unwrap();
    let before = fs::read(store.path()).unwrap();
    let lock = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(store.path().with_file_name("content-collections.lock"))
        .unwrap();
    lock.try_lock().unwrap();
    controller
        .apply(
            &mut catalog,
            Action::RenameCollection {
                id: 1,
                name: "new".into(),
            },
        )
        .unwrap();
    assert_eq!(store.save(&controller), Err(Error::Busy));
    assert_eq!(store.load_into(&mut controller, &catalog), Err(Error::Busy));
    assert_eq!(before, fs::read(store.path()).unwrap());
    drop(lock);
    store.save(&controller).unwrap();
    assert_ne!(before, fs::read(store.path()).unwrap());
}

#[cfg(unix)]
#[test]
fn symlink_targets_and_directories_are_rejected_without_touching_external_files() {
    use std::os::unix::fs::symlink;
    let project = Project::new();
    let outside = Project::new();
    let store = project.store();
    let catalog = catalog();
    let mut controller = Controller::default();
    let sentinel = outside.0.join("sentinel");
    fs::write(&sentinel, b"untouched").unwrap();
    symlink(&outside.0, project.0.join(".prism")).unwrap();
    assert_eq!(store.save(&controller), Err(Error::UnsafePath));
    fs::remove_file(project.0.join(".prism")).unwrap();
    fs::create_dir(project.0.join(".prism")).unwrap();
    symlink(&sentinel, store.path()).unwrap();
    assert_eq!(
        store.load_into(&mut controller, &catalog),
        Err(Error::UnsafePath)
    );
    assert_eq!(store.save(&controller), Err(Error::UnsafePath));
    assert_eq!(fs::read(&sentinel).unwrap(), b"untouched");
    fs::remove_file(store.path()).unwrap();
    fs::create_dir(store.path()).unwrap();
    assert_eq!(store.save(&controller), Err(Error::UnsafePath));
    assert!(matches!(
        CollectionStore::for_project("relative"),
        Err(Error::InvalidProjectRoot)
    ));
}
