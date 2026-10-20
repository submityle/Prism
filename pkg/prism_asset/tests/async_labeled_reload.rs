//! End-to-end regressions for atomic source and direct labeled async reloads.

#![cfg(feature = "std")]

extern crate alloc;

use alloc::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use prism_asset::{
    Asset, AssetLoader, AssetMeta, AssetPath, AssetReader, AssetServer, AssetTypeId, DepRequest,
    LoadContext, LoadError, LoadFailure, LoadScheduler, LoadState, MemSource, ReadError,
    ScheduleError,
};
use prism_tasks::{Priority, TaskPool};

#[derive(Clone, Debug, PartialEq, Eq)]
struct Scene(u8);

impl Asset for Scene {
    const TYPE_NAME: &'static str = "async_labeled_reload::Scene";
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Label(u8);

impl Asset for Label {
    const TYPE_NAME: &'static str = "async_labeled_reload::Label";
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Leaf;

impl Asset for Leaf {
    const TYPE_NAME: &'static str = "async_labeled_reload::Leaf";
}

struct SceneLoader;

impl AssetLoader for SceneLoader {
    type Asset = Scene;

    fn extensions(&self) -> &'static [&'static str] {
        &["async-revision"]
    }

    fn load(&self, bytes: &[u8], ctx: &mut LoadContext<'_>) -> Result<Scene, LoadError> {
        let revision = *bytes
            .first()
            .ok_or_else(|| LoadError::Malformed("empty revision".into()))?;
        match revision {
            1 => {
                ctx.add_labeled_asset("keep", Label(10));
                ctx.add_labeled_asset("gone", Label(11));
                ctx.add_labeled_asset_with_deps(
                    "dependent",
                    Label(12),
                    vec![DepRequest::typed(
                        AssetPath::parse("resident.async-leaf"),
                        AssetTypeId::of::<Leaf>(),
                    )],
                );
            }
            2 => {
                ctx.add_labeled_asset("keep", Label(20));
                ctx.add_labeled_asset("dependent", Label(22));
            }
            3 => {
                // Stage a fresh output before emitting the duplicate. The
                // complete revision must be rejected and the fresh slot must
                // not escape the transaction.
                ctx.add_labeled_asset("provisional", Label(30));
                ctx.add_labeled_asset("keep", Label(31));
                ctx.add_labeled_asset("keep", Label(32));
            }
            4 => {
                // The dependency is missing. The async path must reject
                // before publishing any decoded output.
                ctx.add_typed_dependency(
                    AssetPath::parse("missing.async-leaf"),
                    AssetTypeId::of::<Leaf>(),
                );
                ctx.add_labeled_asset("keep", Label(40));
            }
            5 => {
                ctx.add_labeled_asset_with_deps(
                    "new-dependent",
                    Label(50),
                    vec![DepRequest::typed(
                        AssetPath::parse("new.async-leaf"),
                        AssetTypeId::of::<Leaf>(),
                    )],
                );
            }
            6 => {
                ctx.add_typed_dependency(
                    AssetPath::parse("child.async-revision"),
                    AssetTypeId::of::<Scene>(),
                );
                ctx.add_labeled_asset_with_deps(
                    "uses-child",
                    Label(60),
                    vec![DepRequest::typed(
                        AssetPath::parse("child.async-revision#keep"),
                        AssetTypeId::of::<Label>(),
                    )],
                );
            }
            7 => {
                ctx.add_typed_dependency(
                    AssetPath::parse("missing.async-revision"),
                    AssetTypeId::of::<Scene>(),
                );
                ctx.add_labeled_asset("keep", Label(70));
            }
            8 => {
                ctx.add_typed_dependency(
                    AssetPath::parse("a.async-cycle"),
                    AssetTypeId::of::<Scene>(),
                );
                ctx.add_labeled_asset("keep", Label(80));
            }
            9 => {
                ctx.add_typed_dependency(
                    AssetPath::parse("shared.async-leaf"),
                    AssetTypeId::of::<Leaf>(),
                );
                ctx.add_typed_dependency(
                    AssetPath::parse("shared.async-leaf"),
                    AssetTypeId::of::<Leaf>(),
                );
            }
            10 => {
                ctx.add_typed_dependency(
                    AssetPath::parse("child.async-revision#gone"),
                    AssetTypeId::of::<Label>(),
                );
                ctx.add_labeled_asset("keep", Label(100));
            }
            11 => {
                ctx.add_labeled_asset("keep", Scene(110));
            }
            12 => {
                ctx.add_labeled_asset_with_deps(
                    "keep",
                    Label(120),
                    vec![DepRequest::typed(
                        AssetPath::parse("new.async-leaf"),
                        AssetTypeId::of::<Leaf>(),
                    )],
                );
                ctx.add_labeled_asset("fresh", Label(121));
                ctx.add_labeled_asset("dependent", Label(122));
            }
            13 => panic!("revision decoder panicked"),
            _ => return Err(LoadError::Malformed("unknown revision".into())),
        }
        Ok(Scene(revision))
    }
}

struct CycleLoader;

impl AssetLoader for CycleLoader {
    type Asset = Scene;

    fn extensions(&self) -> &'static [&'static str] {
        &["async-cycle"]
    }

    fn load(&self, _bytes: &[u8], ctx: &mut LoadContext<'_>) -> Result<Scene, LoadError> {
        let dependency = if ctx.path().path().starts_with("a.") {
            "b.async-cycle"
        } else {
            "a.async-cycle"
        };
        ctx.add_typed_dependency(AssetPath::parse(dependency), AssetTypeId::of::<Scene>());
        Ok(Scene(99))
    }
}

struct LeafLoader;

impl AssetLoader for LeafLoader {
    type Asset = Leaf;

    fn extensions(&self) -> &'static [&'static str] {
        &["async-leaf"]
    }

    fn load(&self, _bytes: &[u8], _ctx: &mut LoadContext<'_>) -> Result<Leaf, LoadError> {
        Ok(Leaf)
    }
}

fn mount_revision(server: &AssetServer, revision: u8) {
    mount_revision_with_child(server, revision, 1);
}

fn mount_revision_with_child(server: &AssetServer, revision: u8, child_revision: u8) {
    let mut source = MemSource::new();
    source.insert("scene.async-revision", [revision]);
    source.insert("child.async-revision", [child_revision]);
    source.insert("resident.async-leaf", [0]);
    source.insert("new.async-leaf", [0]);
    source.insert("shared.async-leaf", [0]);
    source.insert("a.async-cycle", [0]);
    source.insert("b.async-cycle", [0]);
    server.mount("mem", Arc::new(source), 0);
}

fn server(revision: u8) -> Arc<AssetServer> {
    let server = Arc::new(AssetServer::new());
    server.register_loader(SceneLoader);
    server.register_loader(LeafLoader);
    server.register_loader(CycleLoader);
    mount_revision(&server, revision);
    server
}

struct CountingSource {
    source: MemSource,
    shared_reads: Arc<AtomicUsize>,
    release: Arc<AtomicBool>,
}

impl AssetReader for CountingSource {
    fn read(&self, path: &str) -> Result<Vec<u8>, ReadError> {
        if path == "shared.async-leaf" {
            self.shared_reads.fetch_add(1, Ordering::AcqRel);
            while !self.release.load(Ordering::Acquire) {
                std::thread::yield_now();
            }
        }
        self.source.read(path)
    }

    fn metadata(&self, path: &str) -> Result<AssetMeta, ReadError> {
        self.source.metadata(path)
    }
}

struct GateSource {
    started: Arc<AtomicBool>,
    release: Arc<AtomicBool>,
}

impl AssetReader for GateSource {
    fn read(&self, path: &str) -> Result<Vec<u8>, ReadError> {
        if path == "scene.async-revision" {
            self.started.store(true, Ordering::Release);
            while !self.release.load(Ordering::Acquire) {
                std::thread::yield_now();
            }
            return Ok(vec![2]);
        }
        Err(ReadError::NotFound)
    }

    fn metadata(&self, _path: &str) -> Result<AssetMeta, ReadError> {
        Ok(AssetMeta {
            size: 1,
            modified_ms: None,
            is_dir: false,
        })
    }
}

fn async_reload(server: &Arc<AssetServer>) -> Result<prism_asset::UntypedHandle, LoadFailure> {
    let pool = TaskPool::with_threads(1);
    let scheduler = LoadScheduler::new(Arc::clone(server), pool.clone(), 1);
    let result = pool.block_on(
        scheduler
            .schedule_reload("scene.async-revision", Priority::Normal)
            .expect("resident root should admit an async reload"),
    );
    assert_eq!(scheduler.in_flight(), 0);
    result
}

#[test]
fn async_initial_load_commits_labeled_outputs_and_their_dependencies() {
    let server = server(1);
    let pool = TaskPool::with_threads(2);
    let scheduler = LoadScheduler::new(Arc::clone(&server), pool.clone(), 2);
    let root = pool.block_on(
        scheduler
            .schedule("scene.async-revision", Priority::Normal)
            .expect("initial async load should be admitted"),
    );
    let root = root.expect("initial async load should commit");

    let source = server.load::<Scene>("scene.async-revision");
    let keep = server.load::<Label>("scene.async-revision#keep");
    let dependent = server.load::<Label>("scene.async-revision#dependent");
    let leaf = server.load::<Leaf>("resident.async-leaf");

    assert_eq!(root.id(), source.untyped_id());
    assert_eq!(server.with_asset(source.id(), |value| value.0), Some(1));
    assert_eq!(server.with_asset(keep.id(), |value| value.0), Some(10));
    assert_eq!(server.with_asset(dependent.id(), |value| value.0), Some(12));
    assert!(server.with_asset(leaf.id(), |_| ()).is_some());
    assert_eq!(server.load_state(source.untyped_id()), LoadState::Loaded);
    assert_eq!(server.load_state(keep.untyped_id()), LoadState::Loaded);
    assert_eq!(server.load_state(dependent.untyped_id()), LoadState::Loaded);
    assert_eq!(scheduler.in_flight(), 0);
}

#[test]
fn async_reload_commits_labeled_revision_and_preserves_identity() {
    let server = server(1);
    let source = server.load::<Scene>("scene.async-revision");
    let keep = server.load::<Label>("scene.async-revision#keep");
    let gone = server.load::<Label>("scene.async-revision#gone");
    let dependent = server.load::<Label>("scene.async-revision#dependent");
    let source_id = source.untyped_id();
    let keep_id = keep.untyped_id();
    let dependent_id = dependent.untyped_id();

    mount_revision(&server, 2);
    let reloaded = async_reload(&server).expect("resident labeled revision should commit");

    assert_eq!(reloaded.id(), source_id);
    assert_eq!(server.with_asset(source.id(), |value| value.0), Some(2));
    assert_eq!(server.with_asset(keep.id(), |value| value.0), Some(20));
    assert_eq!(server.with_asset(dependent.id(), |value| value.0), Some(22));
    assert_eq!(keep.untyped_id(), keep_id);
    assert_eq!(dependent.untyped_id(), dependent_id);
    assert_eq!(server.load_state(source_id), LoadState::Loaded);
    assert_eq!(server.load_state(keep_id), LoadState::Loaded);
    assert_eq!(server.load_state(dependent_id), LoadState::Loaded);

    assert!(server.with_asset(gone.id(), |value| value.0).is_none());
    assert!(server.load_state(gone.untyped_id()).is_failed());
    assert!(server.load_untyped("scene.async-revision#gone").is_none());
}

#[test]
fn duplicate_async_label_rejects_the_complete_revision() {
    let server = server(1);
    let source = server.load::<Scene>("scene.async-revision");
    let keep = server.load::<Label>("scene.async-revision#keep");
    let gone = server.load::<Label>("scene.async-revision#gone");

    mount_revision(&server, 3);
    let result = async_reload(&server);
    assert!(matches!(
        result,
        Err(LoadFailure::AssemblyFailed(error)) if error.contains("duplicate labeled asset")
    ));
    assert_eq!(server.with_asset(source.id(), |value| value.0), Some(1));
    assert_eq!(server.with_asset(keep.id(), |value| value.0), Some(10));
    assert_eq!(server.with_asset(gone.id(), |value| value.0), Some(11));
    assert!(server
        .load_untyped("scene.async-revision#provisional")
        .is_none());
    assert_eq!(server.load_state(source.untyped_id()), LoadState::Loaded);
    assert_eq!(server.load_state(keep.untyped_id()), LoadState::Loaded);
    assert_eq!(server.load_state(gone.untyped_id()), LoadState::Loaded);
}

#[test]
fn unresident_async_dependency_rejects_without_publishing_labels() {
    let server = server(1);
    let source = server.load::<Scene>("scene.async-revision");
    let keep = server.load::<Label>("scene.async-revision#keep");
    let gone = server.load::<Label>("scene.async-revision#gone");
    let assets_before = server.total_assets();

    mount_revision(&server, 4);
    let result = async_reload(&server);
    assert!(matches!(
        result,
        Err(LoadFailure::AssemblyFailed(error)) if error.contains("asset not found")
    ));
    assert_eq!(server.with_asset(source.id(), |value| value.0), Some(1));
    assert_eq!(server.with_asset(keep.id(), |value| value.0), Some(10));
    assert_eq!(server.with_asset(gone.id(), |value| value.0), Some(11));
    assert_eq!(server.load_state(source.untyped_id()), LoadState::Loaded);
    assert_eq!(server.load_state(keep.untyped_id()), LoadState::Loaded);
    assert_eq!(server.load_state(gone.untyped_id()), LoadState::Loaded);
    assert_eq!(server.total_assets(), assets_before);
}

#[test]
fn async_reload_commits_a_new_labeled_dependency_closure() {
    let server = server(1);
    let source = server.load::<Scene>("scene.async-revision");
    let keep = server.load::<Label>("scene.async-revision#keep");

    mount_revision(&server, 5);
    let reloaded = async_reload(&server).expect("new labeled dependency should commit");
    assert_eq!(reloaded.id(), source.untyped_id());
    assert_eq!(server.with_asset(source.id(), |value| value.0), Some(5));
    assert_eq!(server.with_asset(keep.id(), |value| value.0), None);
    let label = server.load::<Label>("scene.async-revision#new-dependent");
    assert_eq!(server.with_asset(label.id(), |value| value.0), Some(50));
    let leaf = server.load::<Leaf>("new.async-leaf");
    assert!(server.with_asset(leaf.id(), |_| ()).is_some());
    assert!(server.is_loaded(source.untyped_id()));
    assert!(server.is_loaded(label.untyped_id()));
    assert!(server.is_loaded(leaf.untyped_id()));
}

#[test]
fn async_reload_loads_new_ordinary_dependency_and_new_sibling_label() {
    let server = server(1);
    let root = server.load::<Scene>("scene.async-revision");

    mount_revision_with_child(&server, 6, 1);
    let reloaded = async_reload(&server).expect("new dependency closure should commit");
    assert_eq!(reloaded.id(), root.untyped_id());
    assert_eq!(server.with_asset(root.id(), |value| value.0), Some(6));

    let child = server.load::<Scene>("child.async-revision");
    let child_keep = server.load::<Label>("child.async-revision#keep");
    let sibling = server.load::<Label>("scene.async-revision#uses-child");
    assert_eq!(server.with_asset(child.id(), |value| value.0), Some(1));
    assert_eq!(
        server.with_asset(child_keep.id(), |value| value.0),
        Some(10)
    );
    assert_eq!(server.with_asset(sibling.id(), |value| value.0), Some(60));
    assert!(server.is_loaded(child.untyped_id()));
    assert!(server.is_loaded(child_keep.untyped_id()));
    assert!(server.is_loaded(sibling.untyped_id()));
}

#[test]
fn async_reload_retires_labels_removed_from_a_reloaded_ordinary_dependency() {
    let server = server(1);
    let root = server.load::<Scene>("scene.async-revision");
    mount_revision_with_child(&server, 6, 1);
    async_reload(&server).expect("first dependency closure should commit");
    let child_gone = server.load::<Label>("child.async-revision#gone");
    let child_keep = server.load::<Label>("child.async-revision#keep");

    mount_revision_with_child(&server, 6, 2);
    async_reload(&server).expect("second dependency closure should commit");
    assert_eq!(server.with_asset(root.id(), |value| value.0), Some(6));
    assert_eq!(
        server.with_asset(child_keep.id(), |value| value.0),
        Some(20)
    );
    assert!(server
        .with_asset(child_gone.id(), |value| value.0)
        .is_none());
    assert!(server.load_state(child_gone.untyped_id()).is_failed());
    assert!(server.load_untyped("child.async-revision#gone").is_none());
}

#[test]
fn async_reload_failure_keeps_the_old_revision_when_a_new_dependency_is_missing() {
    let server = server(1);
    let root = server.load::<Scene>("scene.async-revision");
    let keep = server.load::<Label>("scene.async-revision#keep");
    let assets_before = server.total_assets();

    mount_revision(&server, 7);
    let result = async_reload(&server);
    assert!(
        matches!(result, Err(LoadFailure::AssemblyFailed(error)) if error.contains("asset not found"))
    );
    assert_eq!(server.with_asset(root.id(), |value| value.0), Some(1));
    assert_eq!(server.with_asset(keep.id(), |value| value.0), Some(10));
    assert_eq!(server.total_assets(), assets_before);
}

#[test]
fn async_reload_cycle_in_new_dependency_closure_is_rejected_atomically() {
    let server = server(1);
    let root = server.load::<Scene>("scene.async-revision");
    let assets_before = server.total_assets();

    mount_revision(&server, 8);
    let result = async_reload(&server);
    assert!(matches!(result, Err(LoadFailure::AssemblyFailed(error)) if error.contains("cycle")));
    assert_eq!(server.with_asset(root.id(), |value| value.0), Some(1));
    // The candidate is decoded off-server, so a rejected cycle must not
    // publish either newly discovered source into the resident stores.
    for _ in 0..3 {
        server.collect_releases();
    }
    assert_eq!(server.total_assets(), assets_before);
}

#[test]
fn async_reload_deduplicates_concurrent_reads_for_the_same_new_dependency() {
    let server = server(1);
    let root = server.load::<Scene>("scene.async-revision");
    let reads = Arc::new(AtomicUsize::new(0));
    let release = Arc::new(AtomicBool::new(false));
    let _release = ReleaseGate(Arc::clone(&release));
    let mut source = MemSource::new();
    source.insert("scene.async-revision", [9]);
    source.insert("shared.async-leaf", [0]);
    server.mount(
        "counted",
        Arc::new(CountingSource {
            source,
            shared_reads: Arc::clone(&reads),
            release: Arc::clone(&release),
        }),
        10,
    );
    let pool = TaskPool::with_threads(2);
    let scheduler = LoadScheduler::new(Arc::clone(&server), pool.clone(), 2);
    let first = scheduler
        .schedule_reload("scene.async-revision", Priority::Normal)
        .unwrap();
    let second = scheduler
        .schedule_reload("scene.async-revision", Priority::Critical)
        .unwrap();
    wait_until(|| reads.load(Ordering::Acquire) == 1);
    release.store(true, Ordering::Release);
    let first_result = pool.block_on(first);
    let second_result = pool.block_on(second);
    assert!(first_result.is_ok() || second_result.is_ok());
    assert!(!matches!(first_result, Err(LoadFailure::AssemblyFailed(_))));
    assert!(!matches!(
        second_result,
        Err(LoadFailure::AssemblyFailed(_))
    ));
    assert_eq!(reads.load(Ordering::Acquire), 1);
    assert!(server.is_loaded(root.untyped_id()));
}

#[test]
fn cancelling_reload_ticket_does_not_publish_a_partial_candidate() {
    let server = server(1);
    let root = server.load::<Scene>("scene.async-revision");
    let started = Arc::new(AtomicBool::new(false));
    let release = Arc::new(AtomicBool::new(false));
    server.mount(
        "gate",
        Arc::new(GateSource {
            started: Arc::clone(&started),
            release: Arc::clone(&release),
        }),
        20,
    );
    let pool = TaskPool::with_threads(2);
    let scheduler = LoadScheduler::new(Arc::clone(&server), pool.clone(), 1);
    let ticket = scheduler
        .schedule_reload("scene.async-revision", Priority::Normal)
        .unwrap();
    while !started.load(Ordering::Acquire) {
        std::thread::yield_now();
    }
    ticket.cancel();
    assert_eq!(pool.block_on(ticket), Err(LoadFailure::Cancelled));
    assert_eq!(server.with_asset(root.id(), |value| value.0), Some(1));

    release.store(true, Ordering::Release);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
    while std::time::Instant::now() < deadline && scheduler.in_flight() != 0 {
        std::thread::yield_now();
    }
    assert_eq!(scheduler.in_flight(), 0);
    assert_eq!(server.with_asset(root.id(), |value| value.0), Some(1));
}

#[test]
fn direct_typed_and_untyped_label_reload_returns_the_exact_resident_handle() {
    for typed in [false, true] {
        let server = server(1);
        let root = server.load::<Scene>("scene.async-revision");
        let keep = server.load::<Label>("scene.async-revision#keep");
        let dependent = server.load::<Label>("scene.async-revision#dependent");
        let gone = server.load::<Label>("scene.async-revision#gone");
        mount_revision(&server, 2);
        let pool = TaskPool::with_threads(1);
        let scheduler = LoadScheduler::new(Arc::clone(&server), pool.clone(), 1);
        let ticket = if typed {
            scheduler.schedule_reload_typed::<Label>("scene.async-revision#keep", Priority::Normal)
        } else {
            scheduler.schedule_reload("scene.async-revision#keep", Priority::Normal)
        }
        .unwrap();
        let handle = pool.block_on(ticket).expect("direct label reload commits");
        assert_eq!(handle.id(), keep.untyped_id());
        assert_ne!(handle.id(), root.untyped_id());
        assert_eq!(handle.id().type_id(), AssetTypeId::of::<Label>());
        assert_eq!(server.with_asset(root.id(), |value| value.0), Some(2));
        assert_eq!(server.with_asset(keep.id(), |value| value.0), Some(20));
        assert_eq!(server.with_asset(dependent.id(), |value| value.0), Some(22));
        assert!(server.load_state(gone.untyped_id()).is_failed());
        assert!(server.is_loaded(keep.untyped_id()));
        assert_eq!(scheduler.in_flight(), 0);
    }
}

#[test]
fn direct_label_reload_rejects_wrong_requested_type_without_publishing_source() {
    let server = server(1);
    let root = server.load::<Scene>("scene.async-revision");
    let keep = server.load::<Label>("scene.async-revision#keep");
    mount_revision(&server, 2);
    let pool = TaskPool::with_threads(1);
    let scheduler = LoadScheduler::new(Arc::clone(&server), pool.clone(), 1);
    let result = pool.block_on(
        scheduler
            .schedule_reload_typed::<Scene>("scene.async-revision#keep", Priority::Normal)
            .unwrap(),
    );
    assert!(
        matches!(result, Err(LoadFailure::AssemblyFailed(error)) if error.contains("another type"))
    );
    assert_eq!(server.with_asset(root.id(), |value| value.0), Some(1));
    assert_eq!(server.with_asset(keep.id(), |value| value.0), Some(10));
    assert!(server.is_loaded(keep.untyped_id()));
}

#[test]
fn direct_removed_label_rejects_the_revision_but_source_reload_can_retire_it() {
    for typed in [false, true] {
        let server = server(1);
        let root = server.load::<Scene>("scene.async-revision");
        let gone = server.load::<Label>("scene.async-revision#gone");
        let keep = server.load::<Label>("scene.async-revision#keep");
        let before = server.total_assets();
        mount_revision(&server, 2);
        let pool = TaskPool::with_threads(1);
        let scheduler = LoadScheduler::new(Arc::clone(&server), pool.clone(), 1);
        let ticket = if typed {
            scheduler.schedule_reload_typed::<Label>("scene.async-revision#gone", Priority::Normal)
        } else {
            scheduler.schedule_reload("scene.async-revision#gone", Priority::Normal)
        }
        .unwrap();
        let result = pool.block_on(ticket);
        assert!(
            matches!(result, Err(LoadFailure::AssemblyFailed(error)) if error.contains("was not emitted"))
        );
        assert_eq!(server.with_asset(root.id(), |value| value.0), Some(1));
        assert_eq!(server.with_asset(gone.id(), |value| value.0), Some(11));
        assert_eq!(server.with_asset(keep.id(), |value| value.0), Some(10));
        assert!(server.is_loaded(gone.untyped_id()));
        assert_eq!(server.total_assets(), before);

        pool.block_on(
            scheduler
                .schedule_reload("scene.async-revision", Priority::Normal)
                .unwrap(),
        )
        .unwrap();
        assert_eq!(server.with_asset(root.id(), |value| value.0), Some(2));
        assert!(server.load_state(gone.untyped_id()).is_failed());
        let result = pool.block_on(
            scheduler
                .schedule_reload("scene.async-revision#gone", Priority::Normal)
                .unwrap(),
        );
        assert!(
            matches!(result, Err(LoadFailure::AssemblyFailed(error)) if error.contains("not resident"))
        );
        assert_eq!(server.with_asset(keep.id(), |value| value.0), Some(20));
    }
}

#[test]
fn direct_unknown_label_does_not_load_or_reload_its_source() {
    for resident in [false, true] {
        let server = server(1);
        let root = resident.then(|| server.load::<Scene>("scene.async-revision"));
        let before = server.total_assets();
        mount_revision(&server, 2);
        let pool = TaskPool::with_threads(1);
        let scheduler = LoadScheduler::new(Arc::clone(&server), pool.clone(), 1);
        for typed in [false, true] {
            let ticket = if typed {
                scheduler.schedule_reload_typed::<Label>(
                    "scene.async-revision#unknown",
                    Priority::Normal,
                )
            } else {
                scheduler.schedule_reload("scene.async-revision#unknown", Priority::Normal)
            }
            .unwrap();
            let result = pool.block_on(ticket);
            assert!(
                matches!(result, Err(LoadFailure::AssemblyFailed(error)) if error.contains("not resident"))
            );
            assert_eq!(server.total_assets(), before);
        }
        if let Some(root) = root {
            assert_eq!(server.with_asset(root.id(), |value| value.0), Some(1));
            assert!(server.is_loaded(root.untyped_id()));
        }
    }
}

#[test]
fn direct_label_reload_failures_preserve_the_complete_last_good_revision() {
    for (revision, diagnostic) in [
        (3, "duplicate labeled asset"),
        (4, "asset not found"),
        (8, "cycle"),
        (11, "type changed"),
        (13, "panicked"),
        (0, "unknown revision"),
    ] {
        let server = server(1);
        let root = server.load::<Scene>("scene.async-revision");
        let keep = server.load::<Label>("scene.async-revision#keep");
        let gone = server.load::<Label>("scene.async-revision#gone");
        let dependent = server.load::<Label>("scene.async-revision#dependent");
        let before = server.total_assets();
        mount_revision(&server, revision);
        let pool = TaskPool::with_threads(1);
        let scheduler = LoadScheduler::new(Arc::clone(&server), pool.clone(), 1);
        let result = pool.block_on(
            scheduler
                .schedule_reload_typed::<Label>("scene.async-revision#keep", Priority::Normal)
                .unwrap(),
        );
        assert!(
            matches!(result, Err(LoadFailure::AssemblyFailed(ref error)) if error.contains(diagnostic)),
            "revision {revision}: {result:?}"
        );
        assert_eq!(server.with_asset(root.id(), |value| value.0), Some(1));
        assert_eq!(server.with_asset(keep.id(), |value| value.0), Some(10));
        assert_eq!(server.with_asset(gone.id(), |value| value.0), Some(11));
        assert_eq!(server.with_asset(dependent.id(), |value| value.0), Some(12));
        for handle in [
            root.untyped_id(),
            keep.untyped_id(),
            gone.untyped_id(),
            dependent.untyped_id(),
        ] {
            assert!(server.is_loaded(handle));
            assert!(server.recursive_state(handle).is_loaded());
        }
        for _ in 0..3 {
            server.collect_releases();
        }
        assert_eq!(server.total_assets(), before);
        assert!(server
            .load_untyped("scene.async-revision#provisional")
            .is_none());
        assert_eq!(scheduler.in_flight(), 0);
    }
}

#[test]
fn direct_label_reload_publishes_new_dependencies_and_retains_the_source() {
    let server = server(1);
    let keep = server.load::<Label>("scene.async-revision#keep");
    mount_revision(&server, 12);
    let pool = TaskPool::with_threads(1);
    let scheduler = LoadScheduler::new(Arc::clone(&server), pool.clone(), 1);
    let result = pool
        .block_on(
            scheduler
                .schedule_reload_typed::<Label>("scene.async-revision#keep", Priority::Normal)
                .unwrap(),
        )
        .unwrap();
    assert_eq!(result.id(), keep.untyped_id());
    for _ in 0..3 {
        server.collect_releases();
    }
    assert_eq!(server.with_asset(keep.id(), |value| value.0), Some(120));
    let root = server.load::<Scene>("scene.async-revision");
    let fresh = server.load::<Label>("scene.async-revision#fresh");
    assert_eq!(server.with_asset(root.id(), |value| value.0), Some(12));
    assert_eq!(server.with_asset(fresh.id(), |value| value.0), Some(121));
    assert!(server.is_loaded(keep.untyped_id()));
    let leaf = server.load::<Leaf>("new.async-leaf");
    assert!(server.is_loaded(leaf.untyped_id()));
    drop((keep, result, root, fresh, leaf));
    for _ in 0..8 {
        server.collect_releases();
    }
    assert_eq!(server.total_assets(), 0);
}

#[test]
fn reload_dependency_cannot_resolve_a_label_removed_by_the_same_candidate() {
    let server = server(1);
    let root = server.load::<Scene>("scene.async-revision");
    let keep = server.load::<Label>("scene.async-revision#keep");
    let child = server.load::<Scene>("child.async-revision");
    let child_gone = server.load::<Label>("child.async-revision#gone");
    mount_revision_with_child(&server, 10, 2);
    let pool = TaskPool::with_threads(1);
    let scheduler = LoadScheduler::new(Arc::clone(&server), pool.clone(), 1);
    let result = pool.block_on(
        scheduler
            .schedule_reload("scene.async-revision#keep", Priority::Normal)
            .unwrap(),
    );
    assert!(
        matches!(result, Err(LoadFailure::AssemblyFailed(error)) if error.contains("dependency is not resident"))
    );
    assert_eq!(server.with_asset(root.id(), |value| value.0), Some(1));
    assert_eq!(server.with_asset(keep.id(), |value| value.0), Some(10));
    assert_eq!(server.with_asset(child.id(), |value| value.0), Some(1));
    assert_eq!(
        server.with_asset(child_gone.id(), |value| value.0),
        Some(11)
    );
    assert!(server.is_loaded(root.untyped_id()));
    assert!(server.is_loaded(child_gone.untyped_id()));
}

// Gates are released on panic too, so a failed regression cannot strand a pool
// worker and hide the assertion that actually failed.
struct ReleaseGate(Arc<AtomicBool>);

impl Drop for ReleaseGate {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

fn wait_until(mut condition: impl FnMut() -> bool) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while !condition() {
        assert!(std::time::Instant::now() < deadline, "gate timed out");
        std::thread::yield_now();
    }
}

fn mount_latest(server: &AssetServer, revision: u8) {
    let mut source = MemSource::new();
    source.insert("scene.async-revision", [revision]);
    source.insert("new.async-leaf", [0]);
    server.mount("latest", Arc::new(source), 30);
}

#[test]
fn direct_label_source_and_sibling_reload_share_the_same_source_epoch() {
    for (older_path, newer_path) in [
        ("scene.async-revision", "scene.async-revision#keep"),
        ("scene.async-revision#keep", "scene.async-revision"),
        (
            "scene.async-revision#dependent",
            "scene.async-revision#keep",
        ),
    ] {
        let server = server(1);
        let root = server.load::<Scene>("scene.async-revision");
        let keep = server.load::<Label>("scene.async-revision#keep");
        let dependent = server.load::<Label>("scene.async-revision#dependent");
        let started = Arc::new(AtomicBool::new(false));
        let release = Arc::new(AtomicBool::new(false));
        let _release = ReleaseGate(Arc::clone(&release));
        server.mount(
            "gate",
            Arc::new(GateSource {
                started: Arc::clone(&started),
                release: Arc::clone(&release),
            }),
            20,
        );
        let pool = TaskPool::with_threads(2);
        let scheduler = LoadScheduler::new(Arc::clone(&server), pool.clone(), 2);
        let older = scheduler
            .schedule_reload(older_path, Priority::Normal)
            .unwrap();
        wait_until(|| started.load(Ordering::Acquire));
        mount_latest(&server, 12);
        let newer = scheduler
            .schedule_reload(newer_path, Priority::Critical)
            .unwrap();
        let result = pool.block_on(newer).unwrap();
        assert_eq!(
            result.id(),
            if newer_path.contains('#') {
                keep.untyped_id()
            } else {
                root.untyped_id()
            }
        );
        release.store(true, Ordering::Release);
        assert_eq!(pool.block_on(older), Err(LoadFailure::Superseded));
        assert_eq!(server.with_asset(root.id(), |value| value.0), Some(12));
        assert_eq!(server.with_asset(keep.id(), |value| value.0), Some(120));
        assert_eq!(
            server.with_asset(dependent.id(), |value| value.0),
            Some(122)
        );
        assert!(server.is_loaded(keep.untyped_id()));
        assert_eq!(scheduler.in_flight(), 0);
    }
}

#[test]
fn failed_newer_label_reload_supersedes_the_older_source_without_publishing_either() {
    let server = server(1);
    let root = server.load::<Scene>("scene.async-revision");
    let keep = server.load::<Label>("scene.async-revision#keep");
    let started = Arc::new(AtomicBool::new(false));
    let release = Arc::new(AtomicBool::new(false));
    let _release = ReleaseGate(Arc::clone(&release));
    server.mount(
        "gate",
        Arc::new(GateSource {
            started: Arc::clone(&started),
            release: Arc::clone(&release),
        }),
        20,
    );
    let pool = TaskPool::with_threads(2);
    let scheduler = LoadScheduler::new(Arc::clone(&server), pool.clone(), 2);
    let older = scheduler
        .schedule_reload("scene.async-revision", Priority::Normal)
        .unwrap();
    wait_until(|| started.load(Ordering::Acquire));
    mount_latest(&server, 0);
    let newer = scheduler
        .schedule_reload_typed::<Label>("scene.async-revision#keep", Priority::Critical)
        .unwrap();
    assert!(
        matches!(pool.block_on(newer), Err(LoadFailure::AssemblyFailed(error)) if error.contains("unknown revision"))
    );
    release.store(true, Ordering::Release);
    assert_eq!(pool.block_on(older), Err(LoadFailure::Superseded));
    assert_eq!(server.with_asset(root.id(), |value| value.0), Some(1));
    assert_eq!(server.with_asset(keep.id(), |value| value.0), Some(10));
    assert!(server.is_loaded(keep.untyped_id()));
}

#[test]
fn reload_epochs_follow_admission_order_even_when_the_newer_worker_runs_first() {
    let server = server(1);
    let root = server.load::<Scene>("scene.async-revision");
    let keep = server.load::<Label>("scene.async-revision#keep");
    mount_revision(&server, 2);
    let pool = TaskPool::with_threads(1);
    let started = Arc::new(AtomicBool::new(false));
    let release = Arc::new(AtomicBool::new(false));
    let _release = ReleaseGate(Arc::clone(&release));
    let blocked = {
        let started = Arc::clone(&started);
        let release = Arc::clone(&release);
        pool.spawn_future(async move {
            started.store(true, Ordering::Release);
            while !release.load(Ordering::Acquire) {
                std::thread::yield_now();
            }
        })
    };
    wait_until(|| started.load(Ordering::Acquire));
    let scheduler = LoadScheduler::new(Arc::clone(&server), pool.clone(), 2);
    let older = scheduler
        .schedule_reload("scene.async-revision", Priority::Background)
        .unwrap();
    let newer = scheduler
        .schedule_reload_typed::<Label>("scene.async-revision#keep", Priority::Critical)
        .unwrap();
    release.store(true, Ordering::Release);
    pool.block_on(blocked);
    assert_eq!(pool.block_on(newer).unwrap().id(), keep.untyped_id());
    assert_eq!(pool.block_on(older), Err(LoadFailure::Superseded));
    assert_eq!(server.with_asset(root.id(), |value| value.0), Some(2));
    assert_eq!(server.with_asset(keep.id(), |value| value.0), Some(20));
}

#[test]
fn cancellation_of_a_newer_admitted_label_reload_does_not_revive_older_work() {
    let server = server(1);
    let root = server.load::<Scene>("scene.async-revision");
    let keep = server.load::<Label>("scene.async-revision#keep");
    let started = Arc::new(AtomicBool::new(false));
    let release = Arc::new(AtomicBool::new(false));
    let _release = ReleaseGate(Arc::clone(&release));
    server.mount(
        "gate",
        Arc::new(GateSource {
            started: Arc::clone(&started),
            release: Arc::clone(&release),
        }),
        20,
    );
    let pool = TaskPool::with_threads(2);
    let scheduler = LoadScheduler::new(Arc::clone(&server), pool.clone(), 2);
    let older = scheduler
        .schedule_reload("scene.async-revision", Priority::Normal)
        .unwrap();
    wait_until(|| started.load(Ordering::Acquire));
    let newer = scheduler
        .schedule_reload_typed::<Label>("scene.async-revision#keep", Priority::Critical)
        .unwrap();
    newer.cancel();
    assert_eq!(pool.block_on(newer), Err(LoadFailure::Cancelled));
    release.store(true, Ordering::Release);
    assert_eq!(pool.block_on(older), Err(LoadFailure::Superseded));
    wait_until(|| scheduler.in_flight() == 0);
    assert_eq!(server.with_asset(root.id(), |value| value.0), Some(1));
    assert_eq!(server.with_asset(keep.id(), |value| value.0), Some(10));
    assert!(server.is_loaded(keep.untyped_id()));
}

#[test]
fn rejected_label_reload_at_capacity_does_not_supersede_admitted_source_reload() {
    let server = server(1);
    let root = server.load::<Scene>("scene.async-revision");
    let keep = server.load::<Label>("scene.async-revision#keep");
    let started = Arc::new(AtomicBool::new(false));
    let release = Arc::new(AtomicBool::new(false));
    let _release = ReleaseGate(Arc::clone(&release));
    server.mount(
        "gate",
        Arc::new(GateSource {
            started: Arc::clone(&started),
            release: Arc::clone(&release),
        }),
        20,
    );
    let pool = TaskPool::with_threads(2);
    let scheduler = LoadScheduler::new(Arc::clone(&server), pool.clone(), 1);
    let admitted = scheduler
        .schedule_reload("scene.async-revision", Priority::Normal)
        .unwrap();
    wait_until(|| started.load(Ordering::Acquire));
    for _ in 0..32 {
        assert!(matches!(
            scheduler
                .schedule_reload_typed::<Label>("scene.async-revision#keep", Priority::Critical),
            Err(ScheduleError::AtCapacity { limit: 1 })
        ));
        assert_eq!(scheduler.in_flight(), 1);
    }
    release.store(true, Ordering::Release);
    assert_eq!(pool.block_on(admitted).unwrap().id(), root.untyped_id());
    assert_eq!(server.with_asset(keep.id(), |value| value.0), Some(20));
    assert_eq!(scheduler.in_flight(), 0);
}

#[test]
fn cancelling_direct_label_reload_preserves_old_siblings_and_dependency_graph() {
    for typed in [false, true] {
        let server = server(1);
        let root = server.load::<Scene>("scene.async-revision");
        let keep = server.load::<Label>("scene.async-revision#keep");
        let gone = server.load::<Label>("scene.async-revision#gone");
        let dependent = server.load::<Label>("scene.async-revision#dependent");
        let before = server.total_assets();
        let started = Arc::new(AtomicBool::new(false));
        let release = Arc::new(AtomicBool::new(false));
        let _release = ReleaseGate(Arc::clone(&release));
        server.mount(
            "gate",
            Arc::new(GateSource {
                started: Arc::clone(&started),
                release: Arc::clone(&release),
            }),
            20,
        );
        let pool = TaskPool::with_threads(2);
        let scheduler = LoadScheduler::new(Arc::clone(&server), pool.clone(), 1);
        let ticket = if typed {
            scheduler.schedule_reload_typed::<Label>("scene.async-revision#keep", Priority::Normal)
        } else {
            scheduler.schedule_reload("scene.async-revision#keep", Priority::Normal)
        }
        .unwrap();
        wait_until(|| started.load(Ordering::Acquire));
        ticket.cancel();
        assert_eq!(pool.block_on(ticket), Err(LoadFailure::Cancelled));
        release.store(true, Ordering::Release);
        wait_until(|| scheduler.in_flight() == 0);
        assert_eq!(server.with_asset(root.id(), |value| value.0), Some(1));
        assert_eq!(server.with_asset(keep.id(), |value| value.0), Some(10));
        assert_eq!(server.with_asset(gone.id(), |value| value.0), Some(11));
        assert_eq!(server.with_asset(dependent.id(), |value| value.0), Some(12));
        for _ in 0..3 {
            server.collect_releases();
        }
        assert_eq!(server.total_assets(), before);
        assert!(server.is_loaded(dependent.untyped_id()));
        assert!(server.recursive_state(dependent.untyped_id()).is_loaded());
    }
}

struct HigherPriorityMainLoader;

impl AssetLoader for HigherPriorityMainLoader {
    type Asset = Leaf;

    fn extensions(&self) -> &'static [&'static str] {
        &["async-revision"]
    }

    fn priority(&self) -> i32 {
        100
    }

    fn load(&self, _: &[u8], _: &mut LoadContext<'_>) -> Result<Leaf, LoadError> {
        Ok(Leaf)
    }
}

#[test]
fn direct_label_reload_uses_resident_source_type_when_default_loader_differs() {
    let server = server(1);
    let root = server.load::<Scene>("scene.async-revision");
    let keep = server.load::<Label>("scene.async-revision#keep");
    server.register_loader(HigherPriorityMainLoader);
    mount_revision(&server, 2);
    let pool = TaskPool::with_threads(1);
    let scheduler = LoadScheduler::new(Arc::clone(&server), pool.clone(), 1);
    for typed in [false, true] {
        let ticket = if typed {
            scheduler.schedule_reload_typed::<Label>("scene.async-revision#keep", Priority::Normal)
        } else {
            scheduler.schedule_reload("scene.async-revision#keep", Priority::Normal)
        }
        .unwrap();
        assert_eq!(pool.block_on(ticket).unwrap().id(), keep.untyped_id());
        assert_eq!(server.with_asset(root.id(), |value| value.0), Some(2));
        assert_eq!(server.with_asset(keep.id(), |value| value.0), Some(20));
    }
    assert_eq!(
        pool.block_on(
            scheduler
                .schedule_reload("scene.async-revision", Priority::Normal)
                .unwrap()
        )
        .unwrap()
        .id(),
        root.untyped_id()
    );
}

struct MutableRevisionSource {
    revision: Arc<AtomicUsize>,
    reads: Arc<AtomicUsize>,
    first_read_gate: Option<(Arc<AtomicBool>, Arc<AtomicBool>)>,
}

impl AssetReader for MutableRevisionSource {
    fn read(&self, path: &str) -> Result<Vec<u8>, ReadError> {
        if path != "scene.async-revision" {
            return Err(ReadError::NotFound);
        }
        let revision = u8::try_from(self.revision.load(Ordering::Acquire)).unwrap();
        let previous_reads = self.reads.fetch_add(1, Ordering::AcqRel);
        if previous_reads == 0 {
            if let Some((started, release)) = &self.first_read_gate {
                started.store(true, Ordering::Release);
                while !release.load(Ordering::Acquire) {
                    std::thread::yield_now();
                }
            }
        }
        Ok(vec![revision])
    }

    fn metadata(&self, path: &str) -> Result<AssetMeta, ReadError> {
        if path != "scene.async-revision" {
            return Err(ReadError::NotFound);
        }
        Ok(AssetMeta {
            size: 1,
            modified_ms: None,
            is_dir: false,
        })
    }
}

#[test]
fn sequential_label_reloads_read_changed_bytes_without_remounting_the_source() {
    let server = server(1);
    let root = server.load::<Scene>("scene.async-revision");
    let keep = server.load::<Label>("scene.async-revision#keep");
    let revision = Arc::new(AtomicUsize::new(2));
    let reads = Arc::new(AtomicUsize::new(0));
    server.mount(
        "mutable",
        Arc::new(MutableRevisionSource {
            revision: Arc::clone(&revision),
            reads: Arc::clone(&reads),
            first_read_gate: None,
        }),
        10,
    );
    let pool = TaskPool::with_threads(1);
    let scheduler = LoadScheduler::new(Arc::clone(&server), pool.clone(), 1);
    for current in [2, 1, 2] {
        revision.store(current, Ordering::Release);
        let result = pool
            .block_on(
                scheduler
                    .schedule_reload("scene.async-revision#keep", Priority::Normal)
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(result.id(), keep.untyped_id());
        assert_eq!(
            server.with_asset(root.id(), |value| usize::from(value.0)),
            Some(current)
        );
        assert_eq!(
            server.with_asset(keep.id(), |value| usize::from(value.0)),
            Some(current * 10)
        );
    }
    assert_eq!(reads.load(Ordering::Acquire), 3);
}

#[test]
fn newer_same_mount_label_reload_reads_fresh_root_instead_of_sharing_old_read() {
    let server = server(1);
    let root = server.load::<Scene>("scene.async-revision");
    let keep = server.load::<Label>("scene.async-revision#keep");
    let revision = Arc::new(AtomicUsize::new(2));
    let reads = Arc::new(AtomicUsize::new(0));
    let started = Arc::new(AtomicBool::new(false));
    let release = Arc::new(AtomicBool::new(false));
    let _release = ReleaseGate(Arc::clone(&release));
    server.mount(
        "mutable",
        Arc::new(MutableRevisionSource {
            revision: Arc::clone(&revision),
            reads: Arc::clone(&reads),
            first_read_gate: Some((Arc::clone(&started), Arc::clone(&release))),
        }),
        10,
    );
    let pool = TaskPool::with_threads(2);
    let scheduler = LoadScheduler::new(Arc::clone(&server), pool.clone(), 2);
    let older = scheduler
        .schedule_reload("scene.async-revision", Priority::Normal)
        .unwrap();
    wait_until(|| started.load(Ordering::Acquire));
    revision.store(1, Ordering::Release);
    let newer = scheduler
        .schedule_reload_typed::<Label>("scene.async-revision#keep", Priority::Critical)
        .unwrap();
    // The newer source read must finish while the older one is still gated.
    wait_until(|| reads.load(Ordering::Acquire) == 2 && scheduler.in_flight() == 1);
    assert_eq!(pool.block_on(newer).unwrap().id(), keep.untyped_id());
    release.store(true, Ordering::Release);
    assert_eq!(pool.block_on(older), Err(LoadFailure::Superseded));
    assert_eq!(server.with_asset(root.id(), |value| value.0), Some(1));
    assert_eq!(server.with_asset(keep.id(), |value| value.0), Some(10));
}
