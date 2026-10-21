//! Synchronous publication and rejection retire payloads after releasing the server lock.

use super::*;
use crate::{DepRequest, LoadContext, LoadError, MemSource};
use std::sync::Weak;

const PATH: &str = "asset.sync_drop";
const DEPENDENCY: &str = "dependency.sync_drop";
const MISSING: &str = "missing.sync_drop";
const UNKNOWN: &str = "dependency.no_sync_drop_loader";
const MIDDLE: &str = "middle.sync_drop";
const LEAF: &str = "leaf.sync_drop";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Good,
    Replace,
    RemoveLabel,
    PrimaryUnresolved,
    PrimaryFailed,
    DuplicateLabel,
    LabelTypeChanged,
    LabelUnresolved,
    LabelFailed,
    LabelCycle,
    DeepPrimaryFailed,
    DeepLabelFailed,
    LoaderError,
    #[cfg(panic = "unwind")]
    LoaderPanic,
}

impl Mode {
    fn reason(self) -> &'static str {
        match self {
            Self::Good | Self::Replace | Self::RemoveLabel => {
                panic!("a successful revision has no rejection reason")
            }
            Self::PrimaryUnresolved => "dependency could not be resolved",
            Self::PrimaryFailed => "dependency failed to load",
            Self::DuplicateLabel => "duplicate labeled asset",
            Self::LabelTypeChanged => "type changed",
            Self::LabelUnresolved => "labeled dependency could not be resolved",
            Self::LabelFailed => "labeled dependency failed to load",
            Self::LabelCycle => "labeled asset dependency cycle",
            Self::DeepPrimaryFailed => "dependency failed to load",
            Self::DeepLabelFailed => "labeled dependency failed to load",
            Self::LoaderError => "expected sync loader error",
            #[cfg(panic = "unwind")]
            Self::LoaderPanic => "asset source or loader panicked",
        }
    }

    fn drops(self) -> Vec<u8> {
        match self {
            Self::DeepPrimaryFailed | Self::DeepLabelFailed => {
                vec![10, 11, 12, 13, 14, 20, 21, 22, 30, 31, 32]
            }
            Self::LoaderError => vec![11, 12, 13, 14],
            #[cfg(panic = "unwind")]
            Self::LoaderPanic => vec![11, 12, 13, 14],
            _ => vec![10, 11, 12, 13, 14],
        }
    }
}

struct Probe {
    number: u8,
    log: Arc<DropLog>,
}

impl Asset for Probe {
    const TYPE_NAME: &'static str = "server_sync_drop_tests::Probe";
}

struct Other {
    _probe: Probe,
}

impl Asset for Other {
    const TYPE_NAME: &'static str = "server_sync_drop_tests::Other";
}

#[derive(Debug)]
struct Observation {
    number: u8,
    unlocked: bool,
    settled: bool,
    reentered: bool,
}

struct DropLog {
    server: Weak<AssetServer>,
    expected: Mutex<Publication>,
    drops: Mutex<Vec<Observation>>,
}

impl Drop for Probe {
    fn drop(&mut self) {
        // Do not observe server teardown or create a server -> payload -> server cycle.
        let Some(server) = self.log.server.upgrade() else {
            return;
        };
        // Never block on the server from Drop: a lock-held regression must fail,
        // not hang the package test runner. The guard is gone before logging.
        let (unlocked, settled, readbacks) = match server.inner.try_lock() {
            Ok(inner) => {
                let expected = self.log.expected.lock().unwrap();
                (true, expected.matches(&inner), expected.readbacks(&inner))
            }
            Err(_) => (false, false, Vec::new()),
        };
        // Release both guards before actually reentering public APIs. The
        // nonblocking check above prevents a regression from deadlocking.
        let reentered = unlocked
            && readbacks.into_iter().all(|(id, number, failed)| {
                let state = server.load_state(id);
                (if failed {
                    state.is_failed()
                } else {
                    state == LoadState::Loaded
                }) && server.with_asset(id.typed::<Probe>().unwrap(), |value| value.number)
                    == number
            });
        self.log.drops.lock().unwrap().push(Observation {
            number: self.number,
            unlocked,
            settled,
            reentered,
        });
    }
}

struct Ready {
    path: AssetPath,
    id: UntypedAssetId,
    number: u8,
    dependencies: Vec<UntypedAssetId>,
    retained: Vec<UntypedAssetId>,
    promoted: Option<UntypedAssetId>,
}

struct Publication {
    ready: Vec<Ready>,
    absent: Vec<AssetPath>,
    failed: Vec<UntypedAssetId>,
    failed_attempts: Vec<(AssetPath, &'static str)>,
}

fn retained_ids(inner: &AssetServerInner, id: UntypedAssetId) -> Vec<UntypedAssetId> {
    let mut ids: Vec<_> = inner
        .retained
        .get(&id)
        .into_iter()
        .flatten()
        .map(UntypedHandle::id)
        .collect();
    ids.sort();
    ids
}

impl Publication {
    fn initial_failure() -> Self {
        Self {
            ready: Vec::new(),
            absent: ["fresh", "part", "keep", "tail"]
                .into_iter()
                .map(label_path)
                .collect(),
            failed: Vec::new(),
            failed_attempts: Vec::new(),
        }
    }

    fn failed_attempt_id(
        inner: &AssetServerInner,
        path: &AssetPath,
        reason: &str,
    ) -> Option<UntypedAssetId> {
        // Nested source handles may already be dead. Resolve their failed
        // slots through recorded errors, not by resurrecting/loading paths.
        inner.errors.iter().find_map(|(error_id, error)| {
            if error.path != path.path() || !error.reason.contains(reason) {
                return None;
            }
            inner.retained.keys().copied().find(|id| {
                inner.stores.load_state(*id) == LoadState::Failed(error_id)
                    && inner.graph.dependencies(*id).is_empty()
            })
        })
    }

    fn readbacks(&self, inner: &AssetServerInner) -> Vec<(UntypedAssetId, Option<u8>, bool)> {
        let mut reads: Vec<_> = self
            .ready
            .iter()
            .map(|ready| (ready.id, Some(ready.number), false))
            .chain(self.failed.iter().map(|id| (*id, None, true)))
            .chain(self.failed_attempts.iter().filter_map(|(path, reason)| {
                Self::failed_attempt_id(inner, path, reason).map(|id| (id, None, true))
            }))
            .collect();
        if self.ready.is_empty()
            && let Some(root) = inner
                .interned
                .get(&AssetPath::from(PATH))
                .and_then(UntypedWeakHandle::upgrade)
        {
            reads.push((root.id(), None, true));
        }
        reads
    }

    fn matches(&self, inner: &AssetServerInner) -> bool {
        if self.absent.iter().any(|path| {
            inner.interned.contains_key(path)
                || inner
                    .catalog
                    .contains_key(&StableGuid::from_path(&path.to_string()))
        }) {
            return false;
        }
        let failed_attempts: Option<Vec<_>> = self
            .failed_attempts
            .iter()
            .map(|(path, reason)| Self::failed_attempt_id(inner, path, reason))
            .collect();
        let Some(failed_attempts) = failed_attempts else {
            return false;
        };
        if self.failed.iter().chain(&failed_attempts).any(|id| {
            let state = inner.stores.load_state(*id);
            !state.is_failed()
                || inner.graph.self_state(*id) != state
                || !inner.graph.recursive_state(*id).is_failed()
                || inner.stores.get(id.typed::<Probe>().unwrap()).is_some()
                || !inner.retained.get(id).is_some_and(Vec::is_empty)
                || inner.promoted_labels.contains_key(id)
        }) {
            return false;
        }
        if self.ready.is_empty() {
            let Some(root) = inner
                .interned
                .get(&AssetPath::from(PATH))
                .and_then(UntypedWeakHandle::upgrade)
            else {
                return false;
            };
            let state = inner.stores.load_state(root.id());
            return state.is_failed()
                && inner.graph.self_state(root.id()) == state
                && inner.graph.recursive_state(root.id()).is_failed()
                && inner
                    .stores
                    .get(root.id().typed::<Probe>().unwrap())
                    .is_none()
                && inner.graph.dependencies(root.id()).is_empty()
                && inner.retained.get(&root.id()).is_some_and(Vec::is_empty);
        }
        self.ready.iter().all(|ready| {
            inner
                .stores
                .get(ready.id.typed::<Probe>().unwrap())
                .map(|value| value.number)
                == Some(ready.number)
                && inner.stores.load_state(ready.id) == LoadState::Loaded
                && inner.graph.self_state(ready.id) == LoadState::Loaded
                && inner.graph.recursive_state(ready.id) == RecursiveDependencyLoadState::Loaded
                && !inner.graph.is_stale(ready.id)
                && inner.graph.dependencies(ready.id) == ready.dependencies
                && retained_ids(inner, ready.id) == ready.retained
                && inner.promoted_labels.get(&ready.id).copied() == ready.promoted
                && inner
                    .interned
                    .get(&ready.path)
                    .and_then(UntypedWeakHandle::upgrade)
                    .map(|handle| handle.id())
                    == Some(ready.id)
                && inner
                    .catalog
                    .get(&StableGuid::from_path(&ready.path.to_string()))
                    == Some(&ready.path)
        })
    }
}

fn label_path(label: &str) -> AssetPath {
    AssetPath::from(PATH).with_label(label)
}

fn probe(log: &Arc<DropLog>, number: u8) -> Probe {
    Probe {
        number,
        log: Arc::clone(log),
    }
}

fn dependency(path: AssetPath) -> DepRequest {
    DepRequest::typed(path, AssetTypeId::of::<Probe>())
}

struct Loader {
    log: Arc<DropLog>,
    mode: Arc<Mutex<Mode>>,
}

impl AssetLoader for Loader {
    type Asset = Probe;

    fn extensions(&self) -> &'static [&'static str] {
        &["sync_drop"]
    }

    fn load(&self, _bytes: &[u8], ctx: &mut LoadContext) -> Result<Probe, LoadError> {
        if ctx.path() == &AssetPath::from(DEPENDENCY) {
            return Ok(probe(&self.log, 4));
        }
        let mode = *self.mode.lock().unwrap();
        if ctx.path() == &AssetPath::from(MIDDLE) {
            ctx.add_typed_dependency(LEAF.into(), AssetTypeId::of::<Probe>());
            ctx.add_labeled_asset("middle_part", probe(&self.log, 21));
            ctx.add_labeled_asset("middle_tail", probe(&self.log, 22));
            return Ok(probe(&self.log, 20));
        }
        if ctx.path() == &AssetPath::from(LEAF) {
            if mode == Mode::DeepPrimaryFailed {
                ctx.add_typed_dependency(MISSING.into(), AssetTypeId::of::<Probe>());
                ctx.add_labeled_asset("leaf_part", probe(&self.log, 31));
            } else {
                ctx.add_labeled_asset_with_deps(
                    "leaf_part",
                    probe(&self.log, 31),
                    vec![DepRequest::new(UNKNOWN.into())],
                );
            }
            ctx.add_labeled_asset("leaf_tail", probe(&self.log, 32));
            return Ok(probe(&self.log, 30));
        }
        if matches!(mode, Mode::Good | Mode::Replace | Mode::RemoveLabel) {
            let replacing = mode != Mode::Good;
            ctx.add_typed_dependency(DEPENDENCY.into(), AssetTypeId::of::<Probe>());
            ctx.add_labeled_asset_with_deps(
                "part",
                probe(&self.log, if replacing { 41 } else { 2 }),
                vec![dependency(DEPENDENCY.into())],
            );
            if mode != Mode::RemoveLabel {
                ctx.add_labeled_asset("keep", probe(&self.log, if replacing { 42 } else { 3 }));
            }
            return Ok(probe(&self.log, if replacing { 40 } else { 1 }));
        }

        match mode {
            Mode::PrimaryUnresolved => ctx.add_dependency(UNKNOWN.into()),
            Mode::PrimaryFailed => {
                ctx.add_typed_dependency(MISSING.into(), AssetTypeId::of::<Probe>());
            }
            _ => ctx.add_typed_dependency(DEPENDENCY.into(), AssetTypeId::of::<Probe>()),
        }
        if mode == Mode::DeepPrimaryFailed {
            ctx.add_typed_dependency(MIDDLE.into(), AssetTypeId::of::<Probe>());
        }
        // Stage a new path before rejecting an existing path. This exercises
        // rollback without letting it erase the last-good label revision.
        ctx.add_labeled_asset("fresh", probe(&self.log, 11));
        match mode {
            Mode::LabelTypeChanged => ctx.add_labeled_asset(
                "part",
                Other {
                    _probe: probe(&self.log, 12),
                },
            ),
            Mode::LabelUnresolved
            | Mode::LabelFailed
            | Mode::LabelCycle
            | Mode::DeepLabelFailed => {
                let dep = match mode {
                    Mode::LabelUnresolved => DepRequest::new(UNKNOWN.into()),
                    Mode::LabelFailed => dependency(MISSING.into()),
                    // Both old labels are live, so this resolves through the
                    // real server and temporarily promotes label ownership.
                    Mode::LabelCycle => dependency(label_path("keep")),
                    Mode::DeepLabelFailed => dependency(MIDDLE.into()),
                    _ => unreachable!(),
                };
                ctx.add_labeled_asset_with_deps("part", probe(&self.log, 12), vec![dep]);
            }
            _ => ctx.add_labeled_asset("part", probe(&self.log, 12)),
        }
        match mode {
            Mode::DuplicateLabel => ctx.add_labeled_asset("fresh", probe(&self.log, 13)),
            Mode::LabelCycle => ctx.add_labeled_asset_with_deps(
                "keep",
                probe(&self.log, 13),
                vec![dependency(label_path("part"))],
            ),
            _ => ctx.add_labeled_asset("keep", probe(&self.log, 13)),
        }
        // For early label rejection this value remains in the unvisited
        // iterator tail. It must be retired too, not implicitly dropped there.
        ctx.add_labeled_asset("tail", probe(&self.log, 14));
        match mode {
            Mode::LoaderError => Err(LoadError::Other("expected sync loader error".into())),
            #[cfg(panic = "unwind")]
            Mode::LoaderPanic => panic!("expected sync loader panic after adding labels"),
            _ => Ok(probe(&self.log, 10)),
        }
    }
}

struct Scenario {
    server: Arc<AssetServer>,
    log: Arc<DropLog>,
    mode: Arc<Mutex<Mode>>,
}

impl Scenario {
    fn new() -> Self {
        let server = Arc::new(AssetServer::new());
        let log = Arc::new(DropLog {
            server: Arc::downgrade(&server),
            expected: Mutex::new(Publication::initial_failure()),
            drops: Mutex::new(Vec::new()),
        });
        let mode = Arc::new(Mutex::new(Mode::Good));
        server.register_loader(Loader {
            log: Arc::clone(&log),
            mode: Arc::clone(&mode),
        });
        let mut source = MemSource::new();
        source.insert(PATH, vec![1]);
        source.insert(DEPENDENCY, vec![2]);
        source.insert(MIDDLE, vec![3]);
        source.insert(LEAF, vec![4]);
        // MISSING has a known suffix but no bytes; UNKNOWN has no loader.
        server.mount("sync_drop", Arc::new(source), 0);
        Self { server, log, mode }
    }

    fn seed_last_good(&self) -> Handle<Probe> {
        let root = self.server.load::<Probe>(PATH);
        assert_eq!(
            self.server.load_state(root.id().untyped()),
            LoadState::Loaded
        );
        let mut inner = self.server.lock();
        let ready = [
            (AssetPath::from(PATH), 1),
            (label_path("part"), 2),
            (label_path("keep"), 3),
            (AssetPath::from(DEPENDENCY), 4),
        ]
        .into_iter()
        .map(|(path, number)| {
            // Inspect without loading the label, which would deliberately
            // promote ownership and change the baseline being tested.
            let id = inner
                .interned
                .get(&path)
                .and_then(UntypedWeakHandle::upgrade)
                .unwrap()
                .id();
            Ready {
                path,
                id,
                number,
                dependencies: inner.graph.dependencies(id),
                retained: retained_ids(&inner, id),
                promoted: inner.promoted_labels.get(&id).copied(),
            }
        })
        .collect();
        let publication = Publication {
            ready,
            absent: vec![label_path("fresh"), label_path("tail")],
            failed: Vec::new(),
            failed_attempts: Vec::new(),
        };
        assert!(
            publication.matches(&inner),
            "last-good fixture must be ready"
        );
        *self.log.expected.lock().unwrap() = publication;
        let _ = inner.stores.arena_mut::<Probe>().unwrap().drain_events();
        assert!(self.log.drops.lock().unwrap().is_empty());
        root
    }

    fn expect_deep_failure(&self, mode: Mode) {
        if !matches!(mode, Mode::DeepPrimaryFailed | Mode::DeepLabelFailed) {
            return;
        }
        let mut expected = self.log.expected.lock().unwrap();
        expected.absent.extend([
            AssetPath::from(MIDDLE).with_label("middle_part"),
            AssetPath::from(MIDDLE).with_label("middle_tail"),
            AssetPath::from(LEAF).with_label("leaf_part"),
            AssetPath::from(LEAF).with_label("leaf_tail"),
        ]);
        expected
            .failed_attempts
            .push((MIDDLE.into(), "dependency failed to load"));
        expected.failed_attempts.push((
            LEAF.into(),
            if mode == Mode::DeepPrimaryFailed {
                "dependency failed to load"
            } else {
                "labeled dependency could not be resolved"
            },
        ));
        if mode == Mode::DeepPrimaryFailed {
            expected.failed_attempts.push((MISSING.into(), ""));
        }
    }

    fn assert_drops(&self, expected_numbers: &[u8]) {
        let drops = self.log.drops.lock().unwrap();
        let mut numbers: Vec<_> = drops.iter().map(|drop| drop.number).collect();
        numbers.sort();
        // Exact membership/count also proves neither the accepted old values
        // nor the untouched dependency was destroyed, and no tail was leaked.
        assert_eq!(numbers, expected_numbers, "{drops:?}");
        assert!(drops.iter().all(|drop| drop.unlocked), "{drops:?}");
        assert!(drops.iter().all(|drop| drop.settled), "{drops:?}");
        assert!(drops.iter().all(|drop| drop.reentered), "{drops:?}");
    }

    fn assert_rejected(&self, mode: Mode) {
        self.assert_drops(&mode.drops());

        assert!(!self.server.inner.is_poisoned(), "{mode:?}");
        let mut inner = self
            .server
            .inner
            .try_lock()
            .expect("server must be unlocked");
        let expected = self.log.expected.lock().unwrap();
        assert!(expected.matches(&inner), "{mode:?}: publication changed");
        assert!(
            inner
                .errors
                .iter()
                .any(|(_, error)| error.path == PATH && error.reason.contains(mode.reason())),
            "{mode:?}: the intended rejection must actually be reached"
        );
        let events = inner.stores.arena_mut::<Probe>().unwrap().drain_events();
        for (path, reason) in &expected.failed_attempts {
            let id = Publication::failed_attempt_id(&inner, path, reason).unwrap();
            assert_eq!(
                events
                    .iter()
                    .filter(|event| event.id().untyped() == id && event.is_failed())
                    .count(),
                1,
                "{mode:?}: nested failure event for {path}"
            );
        }
        assert!(
            events.iter().all(|event| expected
                .ready
                .iter()
                .all(|ready| event.id().untyped() != ready.id)),
            "{mode:?}: rejected revision emitted events for last-good values"
        );
    }
}

fn rejected_load(mode: Mode) {
    // Exercise both public synchronous entry points, with independent arenas
    // so an earlier failed load cannot short-circuit a later decode.
    for untyped in [false, true] {
        let scenario = Scenario::new();
        scenario.expect_deep_failure(mode);
        *scenario.mode.lock().unwrap() = mode;
        let handle = if untyped {
            scenario.server.load_untyped(PATH).unwrap()
        } else {
            scenario.server.load::<Probe>(PATH).untyped()
        };
        assert!(scenario.server.load_state(handle.id()).is_failed());
        scenario.assert_rejected(mode);
    }
}

fn rejected_reload(mode: Mode) {
    let scenario = Scenario::new();
    let last_good = scenario.seed_last_good();
    scenario.expect_deep_failure(mode);
    *scenario.mode.lock().unwrap() = mode;
    let reloaded = scenario.server.reload(PATH).unwrap();
    assert_eq!(reloaded.id(), last_good.id().untyped());
    assert_eq!(scenario.server.load_state(reloaded.id()), LoadState::Loaded);
    scenario.assert_rejected(mode);
    assert_eq!(
        scenario
            .server
            .with_asset(last_good.id(), |value| value.number),
        Some(1)
    );
}

fn successful_reload(remove_label: bool) {
    let scenario = Scenario::new();
    let root = scenario.seed_last_good();
    // Keep the old label allocation externally observable without promoting
    // its ownership. Removing a revision output must fail even a live handle.
    let keep = scenario
        .server
        .lock()
        .interned
        .get(&label_path("keep"))
        .and_then(UntypedWeakHandle::upgrade)
        .unwrap();
    let part_id = {
        let mut expected = scenario.log.expected.lock().unwrap();
        let part_id = expected
            .ready
            .iter()
            .find(|ready| ready.path == label_path("part"))
            .unwrap()
            .id;
        // Install the complete future publication before reload can retire
        // anything. Every old payload's Drop must observe it, not a partially
        // replaced root/label pair or the original last-good values.
        for ready in &mut expected.ready {
            ready.number = match ready.number {
                1 => 40,
                2 => 41,
                3 => 42,
                other => other,
            };
            if remove_label && ready.id == root.id().untyped() {
                ready.retained.retain(|id| *id != keep.id());
            }
        }
        if remove_label {
            expected.ready.retain(|ready| ready.id != keep.id());
            expected.absent.push(label_path("keep"));
            expected.failed.push(keep.id());
        }
        part_id
    };
    *scenario.mode.lock().unwrap() = if remove_label {
        Mode::RemoveLabel
    } else {
        Mode::Replace
    };

    let reloaded = scenario.server.reload(PATH).unwrap();
    assert_eq!(reloaded.id(), root.id().untyped());
    scenario.assert_drops(&[1, 2, 3]);
    assert!(!scenario.server.inner.is_poisoned());
    assert_eq!(
        scenario.server.with_asset(root.id(), |value| value.number),
        Some(40)
    );
    assert_eq!(
        scenario
            .server
            .with_asset(part_id.typed::<Probe>().unwrap(), |value| value.number),
        Some(41)
    );
    assert_eq!(
        scenario
            .server
            .with_asset(keep.id().typed::<Probe>().unwrap(), |value| value.number),
        if remove_label { None } else { Some(42) }
    );

    let mut inner = scenario.server.inner.try_lock().unwrap();
    assert!(scenario.log.expected.lock().unwrap().matches(&inner));
    assert!(inner.errors.iter().all(|(_, error)| {
        remove_label && error.reason.contains("removed from the source revision")
    }));
    let events = inner.stores.arena_mut::<Probe>().unwrap().drain_events();
    for id in [root.id().untyped(), part_id] {
        assert_eq!(
            events
                .iter()
                .filter(|event| event.id().untyped() == id && event.is_modified())
                .count(),
            1
        );
    }
    let keep_events: Vec<_> = events
        .iter()
        .filter(|event| event.id().untyped() == keep.id())
        .collect();
    assert_eq!(keep_events.len(), 1);
    if remove_label {
        assert!(keep_events[0].is_failed());
        assert!(inner.stores.load_state(keep.id()).is_failed());
    } else {
        assert!(keep_events[0].is_modified());
    }
    assert!(events.iter().all(|event| {
        [root.id().untyped(), part_id, keep.id()].contains(&event.id().untyped())
            && !event.is_added()
            && !event.is_removed()
    }));
}

#[test]
fn sync_reload_old_root_and_labels_drop_after_new_publication_is_reentrantly_readable() {
    successful_reload(false);
}

#[test]
fn sync_reload_removed_label_drops_after_new_publication_and_failed_handle_are_readable() {
    successful_reload(true);
}

#[test]
fn sync_load_deep_primary_dependency_failure_releases_all_staged_payloads_unlocked() {
    rejected_load(Mode::DeepPrimaryFailed);
}

#[test]
fn sync_load_deep_label_dependency_failure_releases_all_staged_payloads_unlocked() {
    rejected_load(Mode::DeepLabelFailed);
}

#[test]
fn sync_reload_deep_primary_dependency_failure_releases_staging_and_reenters_last_good() {
    rejected_reload(Mode::DeepPrimaryFailed);
}

#[test]
fn sync_reload_deep_label_dependency_failure_releases_staging_and_reenters_last_good() {
    rejected_reload(Mode::DeepLabelFailed);
}

#[test]
fn sync_load_unresolved_primary_dependency_drops_complete_candidate_unlocked() {
    rejected_load(Mode::PrimaryUnresolved);
}

#[test]
fn sync_load_failed_primary_dependency_drops_complete_candidate_unlocked() {
    rejected_load(Mode::PrimaryFailed);
}

#[test]
fn sync_load_duplicate_label_drops_staged_current_and_remaining_outputs_unlocked() {
    rejected_load(Mode::DuplicateLabel);
}

#[test]
fn sync_load_unresolved_label_dependency_rolls_back_before_unlocked_drop() {
    rejected_load(Mode::LabelUnresolved);
}

#[test]
fn sync_load_failed_label_dependency_rolls_back_before_unlocked_drop() {
    rejected_load(Mode::LabelFailed);
}

#[test]
fn sync_load_loader_error_drops_context_labels_unlocked() {
    rejected_load(Mode::LoaderError);
}

#[test]
#[cfg(panic = "unwind")]
fn sync_load_loader_panic_drops_context_labels_unlocked_without_poisoning() {
    rejected_load(Mode::LoaderPanic);
}

#[test]
fn sync_reload_unresolved_primary_dependency_keeps_last_good_before_unlocked_drop() {
    rejected_reload(Mode::PrimaryUnresolved);
}

#[test]
fn sync_reload_failed_primary_dependency_keeps_last_good_before_unlocked_drop() {
    rejected_reload(Mode::PrimaryFailed);
}

#[test]
fn sync_reload_duplicate_label_keeps_last_good_and_drops_iterator_tail_unlocked() {
    rejected_reload(Mode::DuplicateLabel);
}

#[test]
fn sync_reload_label_type_change_keeps_typed_last_good_and_drops_tail_unlocked() {
    rejected_reload(Mode::LabelTypeChanged);
}

#[test]
fn sync_reload_unresolved_label_dependency_keeps_last_good_before_unlocked_drop() {
    rejected_reload(Mode::LabelUnresolved);
}

#[test]
fn sync_reload_failed_label_dependency_keeps_last_good_before_unlocked_drop() {
    rejected_reload(Mode::LabelFailed);
}

#[test]
fn sync_reload_label_cycle_restores_last_good_ownership_before_unlocked_drop() {
    rejected_reload(Mode::LabelCycle);
}

#[test]
fn sync_reload_loader_error_keeps_last_good_and_drops_context_labels_unlocked() {
    rejected_reload(Mode::LoaderError);
}

#[test]
#[cfg(panic = "unwind")]
fn sync_reload_loader_panic_keeps_last_good_and_drops_context_labels_unlocked() {
    rejected_reload(Mode::LoaderPanic);
}
