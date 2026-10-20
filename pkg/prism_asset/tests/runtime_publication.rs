//! Runtime publication uses the server's normal identity, events and lifetime.

#![cfg(feature = "std")]

use std::sync::{mpsc, Arc, Mutex, Weak};
use std::time::Duration;

use prism_asset::{
    Asset, AssetDependency, AssetEvent, AssetLoader, AssetMeta, AssetReader, AssetServer,
    AssetTypeId, Assets, Handle, LoadContext, LoadError, LoadScheduler, LoadState, MemSource,
    ReadError, RecursiveDependencyLoadState, RuntimePublicationError, SoftHandle, UntypedAssetId,
};
use prism_tasks::{Priority, TaskPool};

#[derive(Clone, Debug, PartialEq, Eq)]
struct Leaf(u8);
impl Asset for Leaf {
    const TYPE_NAME: &'static str = "runtime_publication::Leaf";
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Node {
    revision: u8,
    references: Vec<UntypedAssetId>,
}
impl Node {
    fn new(revision: u8, references: &[UntypedAssetId]) -> Self {
        Self {
            revision,
            references: references.to_vec(),
        }
    }
}
impl Asset for Node {
    const TYPE_NAME: &'static str = "runtime_publication::Node";
    fn visit_dependencies(&self, visit: &mut dyn FnMut(UntypedAssetId)) {
        for &id in &self.references {
            visit(id);
        }
    }
}

struct LeafLoader;
impl AssetLoader for LeafLoader {
    type Asset = Leaf;
    fn extensions(&self) -> &'static [&'static str] {
        &["runtime-leaf"]
    }
    fn load(&self, bytes: &[u8], ctx: &mut LoadContext<'_>) -> Result<Leaf, LoadError> {
        let value = *bytes
            .first()
            .ok_or_else(|| LoadError::Malformed("empty".into()))?;
        ctx.add_labeled_asset("child", Leaf(value + 1));
        Ok(Leaf(value))
    }
}

fn mount(server: &AssetServer, value: u8) {
    let mut source = MemSource::new();
    source.insert("input.runtime-leaf", [value]);
    server.mount("mem", Arc::new(source), 0);
}

#[test]
fn publication_and_replacement_preserve_identity_and_emit_normal_snapshots() {
    let server = AssetServer::new();
    let handle = server.insert_runtime(Leaf(1), []).unwrap();
    assert_eq!(handle.strong_count(), 1);
    assert_eq!(server.loader_count(), 0);
    assert_eq!(server.load_state(handle.untyped_id()), LoadState::Loaded);
    assert_eq!(
        server.recursive_state(handle.untyped_id()),
        RecursiveDependencyLoadState::Loaded
    );
    let (events, snapshots) = server.drain_events_with_snapshots::<Leaf>().unwrap();
    assert_eq!(events, vec![AssetEvent::Added { id: handle.id() }]);
    assert_eq!(snapshots, vec![(handle.id(), Leaf(1))]);
    let alias = handle.clone();
    server.replace_runtime(&alias, Leaf(2), []).unwrap();
    assert_eq!(handle.handle_id(), alias.handle_id());
    assert_eq!(server.with_asset(handle.id(), |value| value.0), Some(2));
    let (events, snapshots) = server.drain_events_with_snapshots::<Leaf>().unwrap();
    assert_eq!(events, vec![AssetEvent::Modified { id: handle.id() }]);
    assert_eq!(snapshots, vec![(handle.id(), Leaf(2))]);
    assert_eq!(server.total_assets(), 1);
}

#[test]
fn path_and_labeled_assets_cannot_be_overwritten_by_runtime_publication() {
    let server = AssetServer::new();
    server.register_loader(LeafLoader);
    mount(&server, 10);
    let path = server.load::<Leaf>("input.runtime-leaf");
    let label = server.load::<Leaf>("input.runtime-leaf#child");
    let runtime = server.insert_runtime(Leaf(30), []).unwrap();
    drop(server.drain_events_with_snapshots::<Leaf>());
    for handle in [&path, &label] {
        assert_eq!(
            server.replace_runtime(handle, Leaf(99), []),
            Err(RuntimePublicationError::NotRuntimeAsset {
                id: handle.untyped_id()
            })
        );
    }
    assert!(server
        .drain_events_with_snapshots::<Leaf>()
        .unwrap()
        .0
        .is_empty());
    assert_eq!(server.with_asset(path.id(), |value| value.0), Some(10));
    assert_eq!(server.with_asset(label.id(), |value| value.0), Some(11));
    mount(&server, 20);
    drop(server.reload("input.runtime-leaf").unwrap());
    assert_eq!(server.with_asset(path.id(), |value| value.0), Some(20));
    assert_eq!(server.with_asset(label.id(), |value| value.0), Some(21));
    assert_eq!(server.with_asset(runtime.id(), |value| value.0), Some(30));
}

#[test]
fn foreign_equal_slot_ids_and_duplicate_dependencies_are_rejected() {
    let server = AssetServer::new();
    let other = AssetServer::new();
    let local = server.insert_runtime(Leaf(1), []).unwrap();
    let foreign = other.insert_runtime(Leaf(2), []).unwrap();
    let mut standalone = Assets::new();
    let detached = standalone.insert(Leaf(3));
    assert_eq!(local.id(), foreign.id());
    assert_eq!(local.id(), detached.id());
    assert_ne!(local.handle_id(), foreign.handle_id());
    let owner = server
        .insert_runtime(Node::new(1, &[local.untyped_id()]), [local.untyped()])
        .unwrap();
    drop(server.drain_events_with_snapshots::<Leaf>());
    drop(server.drain_events_with_snapshots::<Node>());
    for invalid in [&foreign, &detached] {
        let error = RuntimePublicationError::ForeignHandle {
            id: invalid.untyped_id(),
        };
        assert_eq!(
            server.replace_runtime(invalid, Leaf(9), []),
            Err(error.clone())
        );
        assert_eq!(
            server.insert_runtime(Node::new(2, &[]), [local.untyped(), invalid.untyped()]),
            Err(error.clone())
        );
        assert_eq!(
            server.replace_runtime(
                &owner,
                Node::new(3, &[]),
                [local.untyped(), invalid.untyped()]
            ),
            Err(error)
        );
    }
    assert_eq!(server.with_asset(local.id(), |value| value.0), Some(1));
    assert_eq!(other.with_asset(foreign.id(), |value| value.0), Some(2));
    assert_eq!(
        server.with_asset(owner.id(), |value| value.revision),
        Some(1)
    );
    assert_eq!(server.total_assets(), 2);
    assert!(server
        .drain_events_with_snapshots::<Leaf>()
        .unwrap()
        .0
        .is_empty());
    assert!(server
        .drain_events_with_snapshots::<Node>()
        .unwrap()
        .0
        .is_empty());
}

#[test]
fn missing_disclosed_dependency_is_transactional_for_insert_and_replace() {
    let server = AssetServer::new();
    let leaf = server.insert_runtime(Leaf(1), []).unwrap();
    let id = leaf.untyped_id();
    assert_eq!(
        server.insert_runtime(Node::new(1, &[id]), []),
        Err(RuntimePublicationError::MissingDependency { id })
    );
    assert!(server.drain_events_with_snapshots::<Node>().is_none());
    assert_eq!(server.total_assets(), 1);
    let owner = server
        .insert_runtime(Node::new(2, &[id]), [leaf.untyped()])
        .unwrap();
    drop(server.drain_events_with_snapshots::<Node>());
    assert_eq!(
        server.replace_runtime(&owner, Node::new(3, &[id]), []),
        Err(RuntimePublicationError::MissingDependency { id })
    );
    assert_eq!(
        server.with_asset(owner.id(), Clone::clone),
        Some(Node::new(2, &[id]))
    );
    assert_eq!(
        server.recursive_state(owner.untyped_id()),
        RecursiveDependencyLoadState::Loaded
    );
    assert!(server
        .drain_events_with_snapshots::<Node>()
        .unwrap()
        .0
        .is_empty());
    let weak = leaf.downgrade();
    drop(leaf);
    assert_eq!(server.collect_releases(), 0);
    assert!(
        weak.upgrade().is_some(),
        "failed replace removed the old ownership edge"
    );
    drop(owner);
    assert_eq!(server.collect_releases(), 2);
    assert!(weak.upgrade().is_none());
}

#[test]
fn dependency_replacement_releases_old_closure_and_deduplicates_new_handles() {
    let server = AssetServer::new();
    let old_leaf = server.insert_runtime(Leaf(1), []).unwrap();
    let old = server
        .insert_runtime(Node::new(1, &[old_leaf.untyped_id()]), [old_leaf.untyped()])
        .unwrap();
    let new = server.insert_runtime(Leaf(2), []).unwrap();
    let owner = server
        .insert_runtime(Node::new(2, &[old.untyped_id()]), [old.untyped()])
        .unwrap();
    let old_weak = old.downgrade();
    let old_leaf_weak = old_leaf.downgrade();
    let new_weak = new.downgrade();
    drop(old);
    drop(old_leaf);
    assert_eq!(server.collect_releases(), 0);
    server
        .replace_runtime(
            &owner,
            Node::new(3, &[new.untyped_id()]),
            [new.untyped(), new.untyped()],
        )
        .unwrap();
    assert_eq!(
        new.strong_count(),
        2,
        "duplicate input handles must not be retained twice"
    );
    assert_eq!(server.collect_releases(), 2);
    assert!(old_weak.upgrade().is_none());
    assert!(old_leaf_weak.upgrade().is_none());
    drop(new);
    assert_eq!(server.collect_releases(), 0);
    assert!(new_weak.upgrade().is_some());
    drop(owner);
    assert_eq!(server.collect_releases(), 2);
    assert!(new_weak.upgrade().is_none());
    assert_eq!(server.total_assets(), 0);
}

#[test]
fn cycle_refusal_preserves_previous_graph_value_state_and_event_queue() {
    let server = AssetServer::new();
    server.register_loader(LeafLoader);
    let failed = server.load::<Leaf>("missing.runtime-leaf");
    let a = server
        .insert_runtime(Node::new(1, &[]), [failed.untyped()])
        .unwrap();
    let b = server
        .insert_runtime(Node::new(2, &[]), [a.untyped()])
        .unwrap();
    let c = server
        .insert_runtime(Node::new(3, &[]), [b.untyped()])
        .unwrap();
    let prior = server.recursive_state(a.untyped_id());
    assert!(prior.is_failed());
    drop(server.drain_events_with_snapshots::<Node>());
    assert!(matches!(
        server.replace_runtime(&a, Node::new(9, &[]), [c.untyped()]),
        Err(RuntimePublicationError::DependencyCycle(_))
    ));
    assert_eq!(server.recursive_state(a.untyped_id()), prior);
    assert_eq!(server.recursive_state(c.untyped_id()), prior);
    assert_eq!(server.with_asset(a.id(), |value| value.revision), Some(1));
    assert!(server
        .drain_events_with_snapshots::<Node>()
        .unwrap()
        .0
        .is_empty());
    let failed_weak = failed.downgrade();
    drop(failed);
    drop(b);
    drop(c);
    assert_eq!(
        server.collect_releases(),
        2,
        "rejected edge must not pin b or c"
    );
    assert!(
        failed_weak.upgrade().is_some(),
        "previous dependency must survive refusal"
    );
    drop(a);
    assert_eq!(server.collect_releases(), 2);
    assert_eq!(server.total_assets(), 0);
}

#[test]
fn self_dependencies_are_rejected_even_when_not_disclosed_or_not_supplied() {
    let server = AssetServer::new();
    let owner = server.insert_runtime(Node::new(1, &[]), []).unwrap();
    drop(server.drain_events_with_snapshots::<Node>());
    let error = RuntimePublicationError::SelfDependency {
        id: owner.untyped_id(),
    };
    assert_eq!(
        server.replace_runtime(&owner, Node::new(2, &[]), [owner.untyped()]),
        Err(error.clone())
    );
    assert_eq!(
        server.replace_runtime(&owner, Node::new(3, &[owner.untyped_id()]), []),
        Err(error)
    );
    assert_eq!(
        server.with_asset(owner.id(), |value| value.revision),
        Some(1)
    );
    assert!(server
        .drain_events_with_snapshots::<Node>()
        .unwrap()
        .0
        .is_empty());
    drop(owner);
    assert_eq!(server.collect_releases(), 1);
}

#[test]
fn failed_dependencies_affect_recursive_state_and_updates_propagate_to_dependents() {
    let server = AssetServer::new();
    server.register_loader(LeafLoader);
    let failed = server.load::<Leaf>("missing.runtime-leaf");
    let owner = server
        .insert_runtime(Node::new(1, &[]), [failed.untyped()])
        .unwrap();
    let parent = server
        .insert_runtime(Node::new(2, &[]), [owner.untyped()])
        .unwrap();
    assert_eq!(server.load_state(owner.untyped_id()), LoadState::Loaded);
    assert_eq!(
        server.recursive_state(parent.untyped_id()).error(),
        server.load_state(failed.untyped_id()).error()
    );
    server
        .replace_runtime(&owner, Node::new(3, &[]), [])
        .unwrap();
    assert_eq!(
        server.recursive_state(parent.untyped_id()),
        RecursiveDependencyLoadState::Loaded
    );
    // Reintroducing a failure must propagate too, without discarding the
    // already resident values of either runtime publication.
    server
        .replace_runtime(&owner, Node::new(4, &[]), [failed.untyped()])
        .unwrap();
    assert!(server.recursive_state(parent.untyped_id()).is_failed());
    assert_eq!(
        server.with_asset(parent.id(), |value| value.revision),
        Some(2)
    );
    assert_eq!(
        server.with_asset(owner.id(), |value| value.revision),
        Some(4)
    );
}

struct GatedSource {
    entered: mpsc::Sender<()>,
    release: Mutex<mpsc::Receiver<()>>,
}
impl AssetReader for GatedSource {
    fn read(&self, _: &str) -> Result<Vec<u8>, ReadError> {
        self.entered.send(()).unwrap();
        self.release
            .lock()
            .unwrap()
            .recv_timeout(Duration::from_secs(10))
            .map_err(|_| ReadError::Io("gate was not released".into()))?;
        Ok(vec![5])
    }
    fn metadata(&self, _: &str) -> Result<AssetMeta, ReadError> {
        Ok(AssetMeta {
            size: 1,
            modified_ms: None,
            is_dir: false,
        })
    }
}

#[test]
fn pending_dependency_completion_repairs_runtime_readiness_without_republishing() {
    let server = Arc::new(AssetServer::new());
    server.register_loader(LeafLoader);
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    server.mount(
        "mem",
        Arc::new(GatedSource {
            entered: entered_tx,
            release: Mutex::new(release_rx),
        }),
        0,
    );
    let pool = TaskPool::with_threads(2);
    let scheduler = LoadScheduler::new(Arc::clone(&server), pool.clone(), 1);
    let ticket = scheduler
        .schedule("pending.runtime-leaf", Priority::Normal)
        .unwrap();
    entered_rx.recv_timeout(Duration::from_secs(10)).unwrap();
    // The async source read runs outside the lock; load returns the already
    // reserved exact handle without running another read.
    let pending = server.load::<Leaf>("pending.runtime-leaf");
    assert_eq!(server.load_state(pending.untyped_id()), LoadState::Loading);
    let owner = server
        .insert_runtime(Node::new(1, &[]), [pending.untyped()])
        .unwrap();
    assert_eq!(server.load_state(owner.untyped_id()), LoadState::Loaded);
    assert_eq!(
        server.recursive_state(owner.untyped_id()),
        RecursiveDependencyLoadState::Loading
    );
    drop(server.drain_events_with_snapshots::<Node>());
    release_tx.send(()).unwrap();
    let completed = pool.block_on(ticket).unwrap();
    assert_eq!(completed.handle_id(), pending.handle_id());
    assert_eq!(
        server.recursive_state(owner.untyped_id()),
        RecursiveDependencyLoadState::Loaded
    );
    assert_eq!(
        server.with_asset(owner.id(), |value| value.revision),
        Some(1)
    );
    assert!(!server
        .drain_events_with_snapshots::<Node>()
        .unwrap()
        .0
        .iter()
        .any(AssetEvent::is_modified));
}

#[derive(Clone, Debug, prism_asset::Asset)]
struct Embedded {
    strong: Handle<Leaf>,
    soft: SoftHandle<Leaf>,
}

#[test]
fn derived_strong_references_require_witnesses_but_soft_references_do_not_load() {
    let server = AssetServer::new();
    let leaf = server.insert_runtime(Leaf(1), []).unwrap();
    let value = Embedded {
        strong: leaf.clone(),
        soft: SoftHandle::from_path("never-loaded.runtime-leaf"),
    };
    assert_eq!(
        server.insert_runtime(value.clone(), []),
        Err(RuntimePublicationError::MissingDependency {
            id: leaf.untyped_id()
        })
    );
    let owner = server.insert_runtime(value, [leaf.untyped()]).unwrap();
    assert_eq!(server.total_assets(), 2);
    assert_eq!(
        server.recursive_state(owner.untyped_id()),
        RecursiveDependencyLoadState::Loaded
    );
    assert_eq!(
        server.with_asset(owner.id(), |value| value.soft.guid()),
        Some(SoftHandle::<Leaf>::from_path("never-loaded.runtime-leaf").guid())
    );
    let weak = leaf.downgrade();
    drop(leaf);
    assert_eq!(server.collect_releases(), 0);
    drop(owner);
    assert_eq!(server.collect_releases(), 2);
    assert!(weak.upgrade().is_none());
}

#[derive(Clone, Debug)]
struct Collision;
impl Asset for Collision {
    const TYPE_NAME: &'static str = Leaf::TYPE_NAME;
}

#[test]
fn type_collisions_and_retyped_handles_fail_without_mutation() {
    let server = AssetServer::new();
    let leaf = server.insert_runtime(Leaf(1), []).unwrap();
    let node = server.insert_runtime(Node::new(1, &[]), []).unwrap();
    drop(server.drain_events_with_snapshots::<Leaf>());
    assert_eq!(
        server.insert_runtime(Collision, []),
        Err(RuntimePublicationError::TypeMismatch {
            type_id: AssetTypeId::of::<Collision>()
        })
    );
    let collision = leaf.untyped().typed_unchecked::<Collision>();
    assert_eq!(
        server.replace_runtime(&collision, Collision, []),
        Err(RuntimePublicationError::TypeMismatch {
            type_id: AssetTypeId::of::<Collision>()
        })
    );
    let retyped = leaf.untyped().typed_unchecked::<Node>();
    assert_eq!(
        server.replace_runtime(&retyped, Node::new(9, &[]), []),
        Err(RuntimePublicationError::TypeMismatch {
            type_id: AssetTypeId::of::<Node>()
        })
    );
    assert_eq!(server.with_asset(leaf.id(), |value| value.0), Some(1));
    assert_eq!(
        server.with_asset(node.id(), |value| value.revision),
        Some(1)
    );
    assert_eq!(server.total_assets(), 2);
    assert!(server
        .drain_events_with_snapshots::<Leaf>()
        .unwrap()
        .0
        .is_empty());
}

#[derive(Clone, Debug)]
struct Visitor {
    panic: bool,
}
impl Asset for Visitor {
    const TYPE_NAME: &'static str = "runtime_publication::Visitor";
    fn visit_references(&self, _: &mut dyn FnMut(AssetDependency)) {
        assert!(!self.panic, "controlled visitor failure");
    }
}

#[test]
fn dependency_visitor_panic_preserves_value_and_leaves_server_usable() {
    let server = AssetServer::new();
    assert_eq!(
        server.insert_runtime(Visitor { panic: true }, []),
        Err(RuntimePublicationError::DependencyVisitorPanicked)
    );
    assert_eq!(server.total_assets(), 0);
    assert!(server.drain_events_with_snapshots::<Visitor>().is_none());
    let owner = server.insert_runtime(Visitor { panic: false }, []).unwrap();
    drop(server.drain_events_with_snapshots::<Visitor>());
    assert_eq!(
        server.replace_runtime(&owner, Visitor { panic: true }, []),
        Err(RuntimePublicationError::DependencyVisitorPanicked)
    );
    assert_eq!(
        server.with_asset(owner.id(), |value| value.panic),
        Some(false)
    );
    assert!(server
        .drain_events_with_snapshots::<Visitor>()
        .unwrap()
        .0
        .is_empty());
    server
        .replace_runtime(&owner, Visitor { panic: false }, [])
        .unwrap();
}

struct ReentrantDrop {
    server: Weak<AssetServer>,
    observed: Arc<Mutex<Vec<usize>>>,
}
impl Asset for ReentrantDrop {
    const TYPE_NAME: &'static str = "runtime_publication::ReentrantDrop";
}
impl Drop for ReentrantDrop {
    fn drop(&mut self) {
        if let Some(server) = self.server.upgrade() {
            self.observed.lock().unwrap().push(server.total_assets());
        }
    }
}

#[test]
fn replacement_drops_old_value_after_unlocking_server() {
    let server = Arc::new(AssetServer::new());
    let observed = Arc::new(Mutex::new(Vec::new()));
    let value = || ReentrantDrop {
        server: Arc::downgrade(&server),
        observed: Arc::clone(&observed),
    };
    let owner = server.insert_runtime(value(), []).unwrap();
    server.replace_runtime(&owner, value(), []).unwrap();
    assert_eq!(*observed.lock().unwrap(), vec![1]);
    // Reclaiming arbitrary asset destructors is the existing server contract;
    // only replacement's new explicit unlock guarantee is exercised here.
    drop(server);
    drop(owner);
}

#[test]
fn reclamation_emits_removed_and_slot_reuse_does_not_inherit_runtime_provenance() {
    let server = AssetServer::new();
    let runtime = server.insert_runtime(Leaf(1), []).unwrap();
    let old_id = runtime.id();
    let old_allocation = runtime.handle_id();
    let weak = runtime.downgrade();
    drop(server.drain_events_with_snapshots::<Leaf>());
    drop(runtime);
    assert_eq!(server.collect_releases(), 1);
    assert!(weak.upgrade().is_none());
    assert_eq!(server.load_state(old_id.untyped()), LoadState::NotLoaded);
    assert_eq!(
        server.recursive_state(old_id.untyped()),
        RecursiveDependencyLoadState::NotLoaded
    );
    let (events, snapshots) = server.drain_events_with_snapshots::<Leaf>().unwrap();
    assert_eq!(
        events,
        vec![
            AssetEvent::Unused { id: old_id },
            AssetEvent::Removed { id: old_id }
        ]
    );
    assert!(snapshots.is_empty());
    server.register_loader(LeafLoader);
    mount(&server, 5);
    let path = server.load::<Leaf>("input.runtime-leaf");
    assert_eq!(path.id().index().index(), old_id.index().index());
    assert_ne!(path.id(), old_id);
    assert_ne!(path.handle_id(), old_allocation);
    assert_eq!(
        server.replace_runtime(&path, Leaf(9), []),
        Err(RuntimePublicationError::NotRuntimeAsset {
            id: path.untyped_id()
        })
    );
    assert_eq!(server.with_asset(path.id(), |value| value.0), Some(5));
}
