//! End-to-end regressions for the narrow resident-dependency labeled reload path.

#![cfg(feature = "std")]

use std::sync::Arc;

use prism_asset::{
    Asset, AssetLoader, AssetPath, AssetServer, AssetTypeId, DepRequest, LoadContext, LoadError,
    LoadFailure, LoadScheduler, LoadState, MemSource,
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
                // The dependency is intentionally not resident. The async
                // path must reject before publishing any decoded output.
                ctx.add_dependency(DepRequest::typed(
                    AssetPath::parse("missing.async-leaf"),
                    AssetTypeId::of::<Leaf>(),
                ));
                ctx.add_labeled_asset("keep", Label(40));
            }
            _ => return Err(LoadError::Malformed("unknown revision".into())),
        }
        Ok(Scene(revision))
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
    let mut source = MemSource::new();
    source.insert("scene.async-revision", [revision]);
    source.insert("resident.async-leaf", [0]);
    server.mount("mem", Arc::new(source), 0);
}

fn server(revision: u8) -> Arc<AssetServer> {
    let server = Arc::new(AssetServer::new());
    server.register_loader(SceneLoader);
    server.register_loader(LeafLoader);
    mount_revision(&server, revision);
    server
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
    assert!(server.load_untyped("scene.async-revision#provisional").is_none());
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

    mount_revision(&server, 4);
    let result = async_reload(&server);
    assert!(matches!(
        result,
        Err(LoadFailure::AssemblyFailed(error)) if error.contains("dependency is not resident")
    ));
    assert_eq!(server.with_asset(source.id(), |value| value.0), Some(1));
    assert_eq!(server.with_asset(keep.id(), |value| value.0), Some(10));
    assert_eq!(server.with_asset(gone.id(), |value| value.0), Some(11));
    assert_eq!(server.load_state(source.untyped_id()), LoadState::Loaded);
    assert_eq!(server.load_state(keep.untyped_id()), LoadState::Loaded);
    assert_eq!(server.load_state(gone.untyped_id()), LoadState::Loaded);
    assert!(server.load_untyped("missing.async-leaf").is_none());
}
