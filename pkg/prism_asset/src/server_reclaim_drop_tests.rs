//! Reclamation retires slots under the lock, but destroys payloads outside it.

use super::*;
use crate::{AssetId, RetainBudget, RetentionPolicy};
use std::sync::Weak;

#[derive(Debug)]
struct Observation {
    number: u8,
    unlocked: bool,
    pruned: bool,
    reentered: bool,
}

struct DropLog {
    server: Weak<AssetServer>,
    ids: Mutex<Vec<(u8, UntypedAssetId)>>,
    drops: Mutex<Vec<Observation>>,
    panic_at: Option<u8>,
}

struct Probe {
    number: u8,
    log: Arc<DropLog>,
    dependencies: Vec<Handle<Probe>>,
}

impl Asset for Probe {
    const TYPE_NAME: &'static str = "server_reclaim_drop_tests::Probe";

    fn visit_dependencies(&self, visit: &mut dyn FnMut(UntypedAssetId)) {
        for dependency in &self.dependencies {
            visit(dependency.untyped_id());
        }
    }
}

impl Drop for Probe {
    fn drop(&mut self) {
        // Weak ownership also makes ordinary server teardown unobservable.
        let Some(server) = self.log.server.upgrade() else {
            return;
        };
        let id = self
            .log
            .ids
            .lock()
            .unwrap()
            .iter()
            .find_map(|(number, id)| (*number == self.number).then_some(*id));
        let (unlocked, pruned, remaining) = match server.inner.try_lock() {
            Ok(inner) => (
                true,
                id.is_some_and(|id| {
                    !inner.stores.contains(id)
                        && !inner.retained.contains_key(&id)
                        && !inner.owner_weaks.contains_key(&id)
                        && !inner.runtime_assets.contains(&id)
                        && inner.graph.self_state(id) == LoadState::NotLoaded
                        && inner.graph.recursive_state(id)
                            == RecursiveDependencyLoadState::NotLoaded
                }),
                Some(inner.stores.total_assets()),
            ),
            Err(_) => (false, false, None),
        };
        // The try_lock guard above is gone before any public API is called.
        // On the regression, record failure instead of blocking on reentry.
        let reentered = unlocked
            && id.is_some_and(|id| {
                server.total_assets() == remaining.unwrap()
                    && server.load_state(id) == LoadState::NotLoaded
                    && server.recursive_state(id) == RecursiveDependencyLoadState::NotLoaded
                    && !server.is_loaded(id)
                    && server
                        .with_asset(id.typed::<Probe>().unwrap(), |asset| asset.number)
                        .is_none()
            });
        self.log.drops.lock().unwrap().push(Observation {
            number: self.number,
            unlocked,
            pruned,
            reentered,
        });
        // Both server and log guards must be gone before the intentional panic.
        if self.log.panic_at == Some(self.number) && !std::thread::panicking() {
            panic!("expected reclaim-payload destructor panic: {}", self.number);
        }
        // Embedded dependency handles are dropped after this method, including
        // on unwind. Only a later, unlocked fixpoint pass can reclaim them.
    }
}

struct Timed(u8);

impl Asset for Timed {
    const TYPE_NAME: &'static str = "server_reclaim_drop_tests::Timed";
}

fn scenario(panic_at: Option<u8>) -> (Arc<AssetServer>, Arc<DropLog>) {
    let server = Arc::new(AssetServer::new());
    let log = Arc::new(DropLog {
        server: Arc::downgrade(&server),
        ids: Mutex::new(Vec::new()),
        drops: Mutex::new(Vec::new()),
        panic_at,
    });
    (server, log)
}

fn insert(
    server: &AssetServer,
    log: &Arc<DropLog>,
    number: u8,
    dependencies: Vec<Handle<Probe>>,
) -> Handle<Probe> {
    let retained: Vec<_> = dependencies
        .iter()
        .map(|handle| handle.clone().untyped())
        .collect();
    let handle = server
        .insert_runtime(
            Probe {
                number,
                log: Arc::clone(log),
                dependencies,
            },
            retained,
        )
        .unwrap();
    log.ids.lock().unwrap().push((number, handle.untyped_id()));
    handle
}

fn assert_unlocked(server: &AssetServer) {
    assert!(
        server.inner.try_lock().is_ok(),
        "server mutex is still held or poisoned"
    );
    assert!(!server.inner.is_poisoned());
}

fn assert_drops(log: &DropLog, expected: &[u8]) {
    let drops = log.drops.lock().unwrap();
    let mut numbers: Vec<_> = drops.iter().map(|drop| drop.number).collect();
    let mut expected = expected.to_vec();
    numbers.sort_unstable();
    expected.sort_unstable();
    assert_eq!(numbers, expected, "missing or repeated payload destructor");
    for observation in drops.iter() {
        assert!(observation.unlocked, "{observation:?}");
        assert!(observation.pruned, "{observation:?}");
        assert!(observation.reentered, "{observation:?}");
    }
}

fn assert_gone(server: &AssetServer, ids: &[AssetId<Probe>]) {
    assert_unlocked(server);
    for id in ids {
        assert_eq!(server.load_state(id.untyped()), LoadState::NotLoaded);
        assert!(server.with_asset(*id, |asset| asset.number).is_none());
    }
}

/// Leaves are inserted first, and both actual payload fields and the server's
/// disclosed dependency closure hold them. Pruning retained alone is not enough.
fn abandon_chain(server: &AssetServer, log: &Arc<DropLog>, first: u8) -> [AssetId<Probe>; 3] {
    let leaf = insert(server, log, first, vec![]);
    let leaf_id = leaf.id();
    let middle = insert(server, log, first + 1, vec![leaf]);
    let middle_id = middle.id();
    let root = insert(server, log, first + 2, vec![middle]);
    let root_id = root.id();
    assert_eq!(
        server.recursive_state(root.untyped_id()),
        RecursiveDependencyLoadState::Loaded
    );
    drop(root);
    [leaf_id, middle_id, root_id]
}

#[test]
fn collect_releases_drops_after_unlock_and_reenters_public_reads() {
    let (server, log) = scenario(None);
    let handle = insert(&server, &log, 0, vec![]);
    let id = handle.id();
    let weak = handle.downgrade();
    drop(handle);

    assert_eq!(server.collect_releases(), 1);
    assert_drops(&log, &[0]);
    assert_gone(&server, &[id]);
    assert!(weak.upgrade().is_none());
    assert_eq!(server.total_assets(), 0);
    assert_eq!(server.collect_releases(), 0);
    assert_drops(&log, &[0]);
}

#[test]
fn collect_releases_reaches_payload_handle_cascade_fixpoint_in_one_call() {
    let (server, log) = scenario(None);
    let ids = abandon_chain(&server, &log, 0);
    assert_eq!(server.total_assets(), 3);

    assert_eq!(server.collect_releases(), 3);
    assert_drops(&log, &[0, 1, 2]);
    assert_eq!(
        log.drops
            .lock()
            .unwrap()
            .iter()
            .map(|drop| drop.number)
            .collect::<Vec<_>>(),
        vec![2, 1, 0],
        "each dependency remains owned until its parent's payload is destroyed"
    );
    assert_gone(&server, &ids);
    assert_eq!(server.total_assets(), 0);
    assert_eq!(server.collect_releases(), 0);
}

fn panic_does_not_poison_or_leave_dead_slots(collect: fn(&AssetServer) -> usize) {
    let (server, log) = scenario(Some(0));
    let first = insert(&server, &log, 0, vec![]);
    let second = insert(&server, &log, 1, vec![]);
    let ids = [first.id(), second.id()];
    let weaks = [first.downgrade(), second.downgrade()];
    drop((first, second));

    assert!(catch_unwind(AssertUnwindSafe(|| collect(&server))).is_err());
    // Vec unwinding must destroy the rest of the already retired batch too.
    assert_unlocked(&server);
    assert_drops(&log, &[0, 1]);
    assert_gone(&server, &ids);
    assert!(weaks.iter().all(|weak| weak.upgrade().is_none()));
    assert_eq!(server.total_assets(), 0);
    assert_eq!(server.collect_releases(), 0);

    let replacements = [
        insert(&server, &log, 2, vec![]),
        insert(&server, &log, 3, vec![]),
    ];
    for handle in &replacements {
        let old = ids
            .iter()
            .find(|id| id.index().index() == handle.id().index().index())
            .expect("retired slots must be reusable after payload panic");
        assert_ne!(old.index().generation(), handle.id().index().generation());
        assert!(server
            .with_asset(handle.id(), |asset| asset.number)
            .is_some());
    }
    assert_gone(&server, &ids);
    drop(replacements);
    assert_eq!(collect(&server), 2);
    assert_drops(&log, &[0, 1, 2, 3]);
    assert_eq!(server.total_assets(), 0);
}

#[test]
fn collect_releases_payload_panic_does_not_poison_or_leave_dead_slots() {
    panic_does_not_poison_or_leave_dead_slots(AssetServer::collect_releases);
}

#[test]
fn collect_releases_resumes_payload_handle_cascade_after_drop_panic() {
    let (server, log) = scenario(Some(2));
    let ids = abandon_chain(&server, &log, 0);

    assert!(catch_unwind(AssertUnwindSafe(|| server.collect_releases())).is_err());
    assert_unlocked(&server);
    assert_drops(&log, &[2]);
    assert_gone(&server, &ids[2..]);
    assert_eq!(server.total_assets(), 2);
    assert_eq!(server.collect_releases(), 2);
    assert_drops(&log, &[0, 1, 2]);
    assert_gone(&server, &ids);
    assert_eq!(server.total_assets(), 0);
}

#[test]
fn remove_unused_drops_after_unlock_and_ignores_grace_without_advancing_it() {
    let (server, log) = scenario(None);
    let handle = insert(&server, &log, 0, vec![]);
    let id = handle.id();
    {
        let mut inner = server.lock();
        inner
            .stores
            .arena_mut::<Probe>()
            .unwrap()
            .set_grace_frames(100);
    }
    drop(handle);

    assert_eq!(server.remove_unused(), 1);
    assert_drops(&log, &[0]);
    assert_gone(&server, &[id]);
    assert_eq!(server.total_assets(), 0);
    assert_eq!(server.lock().stores.arena::<Probe>().unwrap().frame(), 0);
    assert_eq!(server.remove_unused(), 0);
}

#[test]
fn remove_unused_payload_panic_does_not_poison_or_leave_dead_slots() {
    panic_does_not_poison_or_leave_dead_slots(AssetServer::remove_unused);
}

/// Arena policy has no server-level setter. Configure through its public arena
/// APIs, but exercise every reclamation through the public server entry point.
fn timed_scenario(
    grace: u64,
    keepalive: Option<u32>,
) -> (Arc<AssetServer>, Arc<DropLog>, AssetId<Timed>) {
    let (server, log) = scenario(None);
    let handle = server.insert_runtime(Timed(42), vec![]).unwrap();
    let id = handle.id();
    {
        let mut inner = server.lock();
        let arena = inner.stores.arena_mut::<Timed>().unwrap();
        arena.set_grace_frames(grace);
        if let Some(frames) = keepalive {
            arena
                .set_retain_budget(RetainBudget {
                    max_bytes: 4,
                    max_entries: 1,
                })
                .unwrap();
            arena
                .retain(&handle, 4, RetentionPolicy::KeepAlive { frames })
                .unwrap();
        }
        arena.drain_events();
    }
    drop(handle);
    (server, log, id)
}

fn policy_advances_once_per_public_collect(grace: u64, keepalive: Option<u32>, expires_on: u8) {
    let (server, log, id) = timed_scenario(grace, keepalive);
    let mut expected_drops = Vec::new();
    let mut unused = 0;
    let mut removed = 0;
    for call in 1..=expires_on {
        // Force several outside-lock passes on EVERY call, not just on the
        // first idle observation or the final policy-expiry boundary.
        let first = (call - 1) * 3;
        let ids = abandon_chain(&server, &log, first);
        let expired = call == expires_on;
        assert_eq!(server.collect_releases(), 3 + usize::from(expired));
        expected_drops.extend(first..first + 3);
        assert_drops(&log, &expected_drops);
        assert_gone(&server, &ids);
        assert_eq!(
            server.with_asset(id, |value| value.0),
            (!expired).then_some(42)
        );
        assert_eq!(server.total_assets(), usize::from(!expired));
        {
            let mut inner = server.lock();
            assert_eq!(
                inner.stores.arena::<Probe>().unwrap().frame(),
                u64::from(call)
            );
            let arena = inner.stores.arena_mut::<Timed>().unwrap();
            assert_eq!(arena.frame(), u64::from(call));
            if let Some(frames) = keepalive {
                let retained = arena.retain_cache().get(id);
                if u32::from(call) <= frames {
                    assert_eq!(retained.unwrap().idle_frames, Some(u32::from(call - 1)));
                } else {
                    assert!(retained.is_none());
                }
            }
            let events = arena.drain_events();
            unused += events.iter().filter(|event| event.is_unused()).count();
            removed += events.iter().filter(|event| event.is_removed()).count();
        }
        assert_eq!(unused, 1, "Unused must be emitted once per idle episode");
        assert_eq!(removed, usize::from(expired));
    }
    assert_eq!(server.load_state(id.untyped()), LoadState::NotLoaded);
    assert_eq!(server.collect_releases(), 0);
}

#[test]
fn grace_advances_once_per_public_collect_despite_payload_fixpoint_rounds() {
    policy_advances_once_per_public_collect(2, None, 3);
}

#[test]
fn keepalive_advances_once_per_public_collect_despite_payload_fixpoint_rounds() {
    policy_advances_once_per_public_collect(0, Some(2), 3);
}

#[test]
fn keepalive_expiry_and_grace_share_one_public_collect_clock() {
    policy_advances_once_per_public_collect(3, Some(1), 4);
}
