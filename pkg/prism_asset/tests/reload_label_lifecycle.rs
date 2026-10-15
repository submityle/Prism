//! Real `AssetServer` revision/ownership regressions, without a scheduler or API mocks.

#![cfg(feature = "std")]

extern crate alloc;

use alloc::sync::Arc;

use prism_asset::{
    Asset, AssetEvent, AssetLoader, AssetPath, AssetServer, AssetTypeId, DepRequest, LoadContext,
    LoadError, LoadState, MemSource, RecursiveDependencyLoadState, SoftHandle,
};

#[derive(Clone, Debug, PartialEq, Eq)]
struct Scene(u8);

impl Asset for Scene {
    const TYPE_NAME: &'static str = "reload_label_lifecycle::Scene";
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Label(u8);

impl Asset for Label {
    const TYPE_NAME: &'static str = "reload_label_lifecycle::Label";
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Leaf;

impl Asset for Leaf {
    const TYPE_NAME: &'static str = "reload_label_lifecycle::Leaf";
}

struct SceneLoader;

impl AssetLoader for SceneLoader {
    type Asset = Scene;

    fn extensions(&self) -> &'static [&'static str] {
        &["revision"]
    }

    fn load(&self, bytes: &[u8], ctx: &mut LoadContext<'_>) -> Result<Scene, LoadError> {
        let revision = *bytes
            .first()
            .ok_or_else(|| LoadError::Malformed("empty revision".into()))?;
        ctx.add_labeled_asset("keep", Label(revision));
        if revision == 1 {
            ctx.add_labeled_asset_with_deps(
                "gone",
                Label(revision),
                vec![DepRequest::typed(
                    AssetPath::parse("gone.leaf"),
                    AssetTypeId::of::<Leaf>(),
                )],
            );
        }
        if revision == 3 {
            // Fail after staging an existing label and a provisional new one.
            // Omitting `gone` must not retire it when this revision is rejected.
            ctx.add_labeled_asset("provisional", Label(revision));
            ctx.add_labeled_asset("keep", Label(revision));
        }
        Ok(Scene(revision))
    }
}

struct LeafLoader;

impl AssetLoader for LeafLoader {
    type Asset = Leaf;

    fn extensions(&self) -> &'static [&'static str] {
        &["leaf"]
    }

    fn load(&self, _bytes: &[u8], _ctx: &mut LoadContext<'_>) -> Result<Leaf, LoadError> {
        Ok(Leaf)
    }
}

fn mount_revision(server: &AssetServer, revision: u8) {
    let mut source = MemSource::new();
    source.insert("scene.revision", [revision]);
    source.insert("gone.leaf", [0]);
    server.mount("mem", Arc::new(source), 0);
}

fn server() -> AssetServer {
    let server = AssetServer::new();
    server.register_loader(SceneLoader);
    server.register_loader(LeafLoader);
    mount_revision(&server, 1);
    server
}

fn collect(server: &AssetServer) {
    // Multiple frames cover the arena grace window as well as ownership cascades.
    for _ in 0..8 {
        server.collect_releases();
    }
}

#[test]
fn successful_revision_retires_label_without_pinning_source_or_old_dependencies() {
    let server = server();
    let source = server.load::<Scene>("scene.revision");
    let keep = server.load::<Label>("scene.revision#keep");
    let gone = server.load::<Label>("scene.revision#gone");
    let gone_clone = gone.clone();
    let leaf = server.load::<Leaf>("gone.leaf");
    let leaf_weak = leaf.downgrade();
    let source_weak = source.downgrade();
    let keep_soft = SoftHandle::<Label>::from_path("scene.revision#keep");
    let gone_soft = SoftHandle::<Label>::from_path("scene.revision#gone");
    drop(leaf);
    server.drain_events_with_snapshots::<Scene>().unwrap();
    server.drain_events_with_snapshots::<Label>().unwrap();

    mount_revision(&server, 2);
    assert_eq!(
        server.reload("scene.revision").unwrap().id(),
        source.untyped_id()
    );
    assert_eq!(server.with_asset(source.id(), |value| value.0), Some(2));
    assert_eq!(server.with_asset(keep.id(), |value| value.0), Some(2));
    let resolved_keep = server.load_soft(&keep_soft);
    assert_eq!(resolved_keep.id(), keep.id());
    assert_eq!(resolved_keep.handle_id(), keep.handle_id());
    drop(resolved_keep);

    let error = server.load_state(gone.untyped_id()).error().unwrap();
    assert_eq!(gone_clone.id(), gone.id());
    assert_eq!(gone_clone.handle_id(), gone.handle_id());
    assert_eq!(server.with_asset(gone.id(), |value| value.0), None);
    assert_eq!(
        server.recursive_state(gone.untyped_id()),
        RecursiveDependencyLoadState::Failed(error)
    );
    let (events, snapshots) = server.drain_events_with_snapshots::<Label>().unwrap();
    assert_eq!(
        events,
        vec![
            AssetEvent::Modified { id: keep.id() },
            AssetEvent::Failed {
                id: gone.id(),
                error,
            },
        ]
    );
    assert_eq!(snapshots, vec![(keep.id(), Label(2))]);
    assert!(server.load_untyped("scene.revision#gone").is_none());
    assert!(server.reload("scene.revision#gone").is_none());
    let missing = server.load_soft(&gone_soft);
    assert!(server.load_state(missing.untyped_id()).is_failed());
    assert_ne!(missing.id(), gone.id());
    drop(missing);

    collect(&server);
    assert!(
        leaf_weak.upgrade().is_none(),
        "retired revision pins its leaf"
    );
    drop(keep);
    drop(source);
    collect(&server);
    assert!(
        source_weak.upgrade().is_none(),
        "externally held retired label pins its former source"
    );
    assert_eq!(
        server.total_assets(),
        1,
        "only the failed external slot remains"
    );
    assert_eq!(
        server.load_state(gone.untyped_id()),
        LoadState::Failed(error)
    );
    assert_eq!(
        server.recursive_state(gone.untyped_id()),
        RecursiveDependencyLoadState::Failed(error),
        "retired handle must keep its failure after its source is reclaimed"
    );
    drop(gone_clone);
    drop(gone);
    collect(&server);
    assert_eq!(server.total_assets(), 0);
}

#[test]
fn rejected_revision_preserves_label_values_identity_and_ownership() {
    let server = server();
    let source = server.load::<Scene>("scene.revision");
    let keep = server.load::<Label>("scene.revision#keep");
    let gone = server.load::<Label>("scene.revision#gone");
    let source_weak = source.downgrade();
    let keep_soft = SoftHandle::<Label>::from_path("scene.revision#keep");
    let gone_soft = SoftHandle::<Label>::from_path("scene.revision#gone");
    server.drain_events_with_snapshots::<Scene>().unwrap();
    server.drain_events_with_snapshots::<Label>().unwrap();

    mount_revision(&server, 3);
    assert_eq!(
        server.reload("scene.revision").unwrap().id(),
        source.untyped_id()
    );
    assert_eq!(server.with_asset(source.id(), |value| value.0), Some(1));
    for (handle, soft) in [(&keep, &keep_soft), (&gone, &gone_soft)] {
        assert_eq!(server.load_state(handle.untyped_id()), LoadState::Loaded);
        assert_eq!(server.with_asset(handle.id(), |value| value.0), Some(1));
        assert_eq!(server.load_soft(soft).handle_id(), handle.handle_id());
    }
    let (events, snapshots) = server.drain_events_with_snapshots::<Label>().unwrap();
    assert_eq!(events.len(), 1, "only the provisional slot is failed");
    assert!(events[0].is_failed());
    assert_ne!(events[0].id(), keep.id());
    assert_ne!(events[0].id(), gone.id());
    let provisional_id = events[0].id().untyped();
    assert!(server.load_state(provisional_id).is_failed());
    assert!(snapshots.is_empty());
    assert!(server
        .drain_events_with_snapshots::<Scene>()
        .unwrap()
        .0
        .is_empty());
    assert!(server.load_untyped("scene.revision#provisional").is_none());
    let provisional_soft = SoftHandle::<Label>::from_path("scene.revision#provisional");
    let unresolved = server.load_soft(&provisional_soft);
    assert!(server.load_state(unresolved.untyped_id()).is_failed());
    drop(unresolved);

    drop(source);
    collect(&server);
    assert_eq!(server.load_state(provisional_id), LoadState::NotLoaded);
    assert_eq!(
        server.recursive_state(provisional_id),
        RecursiveDependencyLoadState::NotLoaded
    );
    assert!(
        source_weak.upgrade().is_some(),
        "live old labels must retain their source"
    );
    drop(keep);
    drop(gone);
    collect(&server);
    assert_eq!(server.total_assets(), 0);
}

#[test]
fn retired_external_label_graph_state_is_reclaimed_with_its_slot() {
    let server = server();
    let _source = server.load::<Scene>("scene.revision");
    let gone = server.load::<Label>("scene.revision#gone");
    let gone_id = gone.untyped_id();
    mount_revision(&server, 2);
    server.reload("scene.revision").unwrap();
    assert!(server.load_state(gone_id).is_failed());
    assert!(server.recursive_state(gone_id).is_failed());
    drop(gone);
    collect(&server);

    assert_eq!(server.load_state(gone_id), LoadState::NotLoaded);
    assert_eq!(
        server.recursive_state(gone_id),
        RecursiveDependencyLoadState::NotLoaded,
        "retired label left a dangling graph/ownership record after slot reclamation"
    );
}

#[test]
fn retired_unresolved_label_graph_state_is_reclaimed_with_its_slot() {
    let server = server();
    let _source = server.load::<Scene>("scene.revision");
    // Observe the id without resolving/promoting the source-owned label.
    let (events, _) = server.drain_events_with_snapshots::<Label>().unwrap();
    assert_eq!(events.len(), 2);
    // The fixture emits `keep` first and `gone` second.
    let gone_id = events[1].id();
    mount_revision(&server, 2);
    server.reload("scene.revision").unwrap();
    assert!(server.load_state(gone_id.untyped()).is_failed());
    collect(&server);

    assert_eq!(server.load_state(gone_id.untyped()), LoadState::NotLoaded);
    assert_eq!(
        server.recursive_state(gone_id.untyped()),
        RecursiveDependencyLoadState::NotLoaded,
        "unresolved retired label escaped graph cleanup"
    );
}

#[test]
fn unresolved_failed_handle_keeps_graph_state_until_its_slot_is_reclaimed() {
    let server = server();
    let missing = server.load_soft(&SoftHandle::<Label>::from_path("unknown.revision#gone"));
    let id = missing.untyped_id();
    let error = server.load_state(id).error().unwrap();
    let held = missing.clone();
    drop(missing);
    collect(&server);
    assert_eq!(server.load_state(id), LoadState::Failed(error));
    assert_eq!(
        server.recursive_state(id),
        RecursiveDependencyLoadState::Failed(error)
    );
    drop(held);
    collect(&server);
    assert_eq!(server.load_state(id), LoadState::NotLoaded);
    assert_eq!(
        server.recursive_state(id),
        RecursiveDependencyLoadState::NotLoaded
    );
}

#[test]
fn failed_initial_load_graph_state_is_reclaimed_with_its_slot() {
    let server = server();
    let missing = server.load::<Scene>("missing.revision");
    let id = missing.untyped_id();
    let error = server.load_state(id).error().unwrap();
    collect(&server);
    assert_eq!(
        server.recursive_state(id),
        RecursiveDependencyLoadState::Failed(error)
    );
    drop(missing);
    collect(&server);
    assert_eq!(server.load_state(id), LoadState::NotLoaded);
    assert_eq!(
        server.recursive_state(id),
        RecursiveDependencyLoadState::NotLoaded
    );
}

#[test]
fn source_owned_label_graph_states_are_reclaimed_with_their_slots() {
    let server = server();
    let source = server.load::<Scene>("scene.revision");
    let (events, _) = server.drain_events_with_snapshots::<Label>().unwrap();
    assert_eq!(events.len(), 2);
    drop(source);
    collect(&server);
    for event in events {
        let id = event.id().untyped();
        assert_eq!(server.load_state(id), LoadState::NotLoaded);
        assert_eq!(
            server.recursive_state(id),
            RecursiveDependencyLoadState::NotLoaded
        );
    }
    assert_eq!(server.total_assets(), 0);
}
