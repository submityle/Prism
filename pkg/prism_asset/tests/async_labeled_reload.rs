//! End-to-end regressions for the narrow resident-dependency labeled reload path.

#![cfg(feature = "std")]

extern crate alloc;

use alloc::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use prism_asset::{
    Asset, AssetLoader, AssetMeta, AssetPath, AssetReader, AssetServer, AssetTypeId, DepRequest,
    LoadContext, LoadError, LoadFailure, LoadScheduler, LoadState, MemSource, ReadError,
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
}

impl AssetReader for CountingSource {
    fn read(&self, path: &str) -> Result<Vec<u8>, ReadError> {
        if path == "shared.async-leaf" {
            self.shared_reads.fetch_add(1, Ordering::AcqRel);
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
    let mut source = MemSource::new();
    source.insert("scene.async-revision", [9]);
    source.insert("shared.async-leaf", [0]);
    server.mount(
        "counted",
        Arc::new(CountingSource {
            source,
            shared_reads: Arc::clone(&reads),
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
    assert_eq!(server.with_asset(root.id(), |value| value.0), Some(2));
}
