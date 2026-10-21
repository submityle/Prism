//! Publication and candidate destruction must happen outside the server lock.

use super::*;
use std::sync::Weak;

const PATH: &str = "asset.commit_drop";

#[derive(Clone)]
struct Publication {
    root: UntypedAssetId,
    ready: Vec<(AssetPath, UntypedAssetId, u8)>,
    failed: Vec<(AssetPath, UntypedAssetId, bool)>,
    absent: Vec<AssetPath>,
    check_ownership: bool,
    epoch: Option<u64>,
}

impl Publication {
    fn matches(&self, inner: &AssetServerInner) -> bool {
        for (path, id, number) in &self.ready {
            if inner
                .stores
                .get(id.typed::<Probe>().unwrap())
                .map(|v| v.number)
                != Some(*number)
                || inner.stores.load_state(*id) != LoadState::Loaded
                || inner.graph.self_state(*id) != LoadState::Loaded
                || inner.graph.recursive_state(*id) != RecursiveDependencyLoadState::Loaded
                || inner
                    .interned
                    .get(path)
                    .and_then(UntypedWeakHandle::upgrade)
                    .map(|h| h.id())
                    != Some(*id)
                || inner.catalog.get(&StableGuid::from_path(&path.to_string())) != Some(path)
            {
                return false;
            }
            if self.check_ownership && path.has_label() {
                if !inner
                    .retained
                    .get(&self.root)
                    .is_some_and(|handles| handles.iter().any(|handle| handle.id() == *id))
                    || !inner.graph.dependencies(*id).contains(&self.root)
                {
                    return false;
                }
            }
        }
        for (path, id, uninterned) in &self.failed {
            let state = inner.stores.load_state(*id);
            if !state.is_failed()
                || inner.graph.self_state(*id) != state
                || inner.stores.get(id.typed::<Probe>().unwrap()).is_some()
                || !inner.retained.get(id).is_some_and(Vec::is_empty)
                || (*uninterned
                    && (inner.interned.contains_key(path)
                        || inner
                            .catalog
                            .contains_key(&StableGuid::from_path(&path.to_string()))))
                || (*id != self.root
                    && inner
                        .retained
                        .get(&self.root)
                        .is_some_and(|handles| handles.iter().any(|handle| handle.id() == *id)))
            {
                return false;
            }
        }
        if self.absent.iter().any(|path| {
            inner.interned.contains_key(path)
                || inner
                    .catalog
                    .contains_key(&StableGuid::from_path(&path.to_string()))
        }) {
            return false;
        }
        // current() is nonblocking. It observes publication's epoch, but does
        // not prove the private epoch gate is unlocked. Calling begin/commit
        // here could spin forever on precisely the regression being tested.
        self.epoch.is_none_or(|epoch| {
            inner
                .reload_epochs
                .get(&AssetPath::from(PATH))
                .map(ReloadEpoch::current)
                == Some(epoch)
        })
    }
}

#[derive(Clone, Debug)]
struct Observation {
    number: u8,
    unlocked: bool,
    published: bool,
}

struct DropLog {
    server: Weak<AssetServer>,
    expected: Mutex<Option<Publication>>,
    drops: Mutex<Vec<Observation>>,
    panic_at: Option<u8>,
}

struct Probe {
    number: u8,
    log: Arc<DropLog>,
}

impl Asset for Probe {
    const TYPE_NAME: &'static str = "server_commit_drop_tests::Probe";
}

impl Drop for Probe {
    fn drop(&mut self) {
        // Server teardown is not part of the observed commit. Weak ownership
        // also avoids retaining the server through its own stored payloads.
        let Some(server) = self.log.server.upgrade() else {
            return;
        };
        let (unlocked, published) = match server.inner.try_lock() {
            Ok(inner) => {
                let expected = self.log.expected.lock().unwrap();
                (
                    true,
                    expected.as_ref().is_some_and(|state| state.matches(&inner)),
                )
            }
            Err(_) => (false, false),
        };
        self.log.drops.lock().unwrap().push(Observation {
            number: self.number,
            unlocked,
            published,
        });
        // Panic only after both guards above are gone, so the test probe does
        // not itself poison the server or the observation mutex.
        if self.log.panic_at == Some(self.number) {
            panic!("expected commit-payload destructor panic: {}", self.number);
        }
    }
}

struct Other {
    _probe: Probe,
}

impl Asset for Other {
    const TYPE_NAME: &'static str = "server_commit_drop_tests::Other";
}

fn scenario(panic_at: Option<u8>) -> (Arc<AssetServer>, Arc<DropLog>) {
    let server = Arc::new(AssetServer::new());
    let log = Arc::new(DropLog {
        server: Arc::downgrade(&server),
        expected: Mutex::new(None),
        drops: Mutex::new(Vec::new()),
        panic_at,
    });
    (server, log)
}

fn probe(log: &Arc<DropLog>, number: u8) -> Probe {
    Probe {
        number,
        log: Arc::clone(log),
    }
}

fn erased(log: &Arc<DropLog>, number: u8, label: &str) -> ErasedLoadedAsset {
    ErasedLoadedAsset {
        type_id: AssetTypeId::of::<Probe>(),
        value: Box::new(probe(log, number)),
        dependencies: Vec::new(),
        label: label.into(),
        make_store: || Box::new(Assets::<Probe>::new()),
    }
}

fn malformed(log: &Arc<DropLog>, number: u8, label: &str) -> ErasedLoadedAsset {
    // The tag and factory agree with the destination; only actual Any type
    // validation can reject this value. Other's field logs its destruction.
    ErasedLoadedAsset {
        type_id: AssetTypeId::of::<Probe>(),
        value: Box::new(Other {
            _probe: probe(log, number),
        }),
        dependencies: Vec::new(),
        label: label.into(),
        make_store: || Box::new(Assets::<Probe>::new()),
    }
}

fn output(log: &Arc<DropLog>, root: u8, labels: &[(&str, u8)]) -> FullLoadOutput {
    FullLoadOutput {
        primary: erased(log, root, ""),
        labeled: labels
            .iter()
            .map(|(label, number)| erased(log, *number, label))
            .collect(),
    }
}

fn prepare_load(server: &AssetServer) -> AsyncLoadPlan {
    server.lock().stores.register::<Probe>();
    match server
        .prepare_async_load(PATH.into(), Some(AssetTypeId::of::<Probe>()))
        .unwrap()
    {
        AsyncLoadPreparation::New(plan) => *plan,
        AsyncLoadPreparation::Existing(_) => panic!("expected a fresh load"),
    }
}

fn label_path(label: &str) -> AssetPath {
    AssetPath::from(PATH).with_label(label)
}

fn label_handle(server: &AssetServer, label: &str) -> UntypedHandle {
    server
        .lock()
        .interned
        .get(&label_path(label))
        .and_then(UntypedWeakHandle::upgrade)
        .unwrap()
}

fn seed_ready_label(
    server: &AssetServer,
    log: &Arc<DropLog>,
    label: &str,
    number: u8,
) -> UntypedHandle {
    let mut inner = server.lock();
    let handle = inner.stores.insert(probe(log, number));
    let path = label_path(label);
    inner.graph.add_asset(handle.id());
    let _ = inner.graph.set_self_state(handle.id(), LoadState::Loaded);
    inner.retained.entry(handle.id()).or_default();
    inner.owner_weaks.insert(handle.id(), handle.downgrade());
    inner.interned.insert(path.clone(), handle.downgrade());
    inner
        .catalog
        .insert(StableGuid::from_path(&path.to_string()), path);
    handle
}

fn seed_source(
    server: &AssetServer,
    log: &Arc<DropLog>,
) -> (UntypedHandle, UntypedHandle, UntypedHandle) {
    let plan = prepare_load(server);
    let root = server
        .commit_async_load(
            plan,
            output(log, 0, &[("part", 1), ("retired", 2)]),
            vec![],
            vec![vec![], vec![]],
        )
        .unwrap();
    (
        root,
        label_handle(server, "part"),
        label_handle(server, "retired"),
    )
}

fn publication(
    root: &UntypedHandle,
    number: u8,
    labels: &[(&str, &UntypedHandle, u8)],
) -> Publication {
    let mut ready = vec![(AssetPath::from(PATH), root.id(), number)];
    ready.extend(
        labels
            .iter()
            .map(|(label, handle, value)| (label_path(label), handle.id(), *value)),
    );
    Publication {
        root: root.id(),
        ready,
        failed: vec![],
        absent: vec![],
        check_ownership: true,
        epoch: None,
    }
}

fn expect(log: &DropLog, state: Publication) {
    *log.expected.lock().unwrap() = Some(state);
}

fn assert_drops(server: &AssetServer, log: &DropLog, numbers: &[u8]) {
    let mut drops = log.drops.lock().unwrap().clone();
    drops.sort_unstable_by_key(|drop| drop.number);
    let mut expected = numbers.to_vec();
    expected.sort_unstable();
    assert_eq!(
        drops.iter().map(|drop| drop.number).collect::<Vec<_>>(),
        expected
    );
    assert!(
        drops.iter().all(|drop| drop.unlocked),
        "Drop ran with server locked: {drops:?}"
    );
    assert!(
        drops.iter().all(|drop| drop.published),
        "Drop saw incomplete publication: {drops:?}"
    );
    assert!(
        !server.inner.is_poisoned(),
        "payload Drop poisoned the server"
    );
    let inner = server
        .inner
        .try_lock()
        .expect("commit must release the server mutex");
    assert!(log
        .expected
        .lock()
        .unwrap()
        .as_ref()
        .unwrap()
        .matches(&inner));
}

fn candidate(log: &Arc<DropLog>, root: u8, labels: &[(&str, u8)]) -> AsyncReloadCandidate {
    AsyncReloadCandidate {
        assets: vec![AsyncReloadCandidateAsset {
            path: PATH.into(),
            type_id: AssetTypeId::of::<Probe>(),
            output: output(log, root, labels),
        }],
    }
}

fn reload_ok(
    server: &AssetServer,
    plan: AsyncReloadPlan,
    candidate: AsyncReloadCandidate,
) -> UntypedHandle {
    match server.commit_async_reload(plan, candidate) {
        Ok(handle) => handle,
        Err(AsyncReloadCommitError::Rejected(reason)) => panic!("unexpected rejection: {reason}"),
        Err(AsyncReloadCommitError::Superseded) => panic!("unexpected supersession"),
    }
}

fn load_reuses_ready_labels(panic_at: Option<u8>) {
    let (server, log) = scenario(panic_at);
    let plan = prepare_load(&server);
    let root = plan.handle.clone();
    let part = seed_ready_label(&server, &log, "part", 1);
    let next = seed_ready_label(&server, &log, "next", 2);
    expect(
        &log,
        publication(&root, 10, &[("part", &part, 11), ("next", &next, 12)]),
    );
    let result = catch_unwind(AssertUnwindSafe(|| {
        server.commit_async_load(
            plan,
            output(&log, 10, &[("part", 11), ("next", 12)]),
            vec![],
            vec![vec![], vec![]],
        )
    }));
    if panic_at.is_some() {
        assert!(result.is_err(), "old Ready Drop should panic");
    } else {
        assert_eq!(result.unwrap().unwrap().id(), root.id());
    }
    assert_drops(&server, &log, &[1, 2]);
    // Worker unwind cleanup must not undo an already published revision.
    server.fail_async_load_if_loading(root.id(), &PATH.into(), "late Drop unwind");
    assert_eq!(server.load_state(root.id()), LoadState::Loaded);
}

#[test]
fn successful_async_load_reuses_ready_labels_and_drops_old_values_after_publication() {
    load_reuses_ready_labels(None);
}

#[test]
fn async_load_ready_label_drop_panic_leaves_publication_complete_and_mutex_unpoisoned() {
    for number in [1, 2] {
        load_reuses_ready_labels(Some(number));
    }
}

fn reload_replaces_and_retires(panic_at: Option<u8>) {
    let (server, log) = scenario(panic_at);
    let (root, part, retired) = seed_source(&server, &log);
    let plan = server.prepare_async_reload(PATH.into(), None).unwrap();
    let mut state = publication(&root, 10, &[("part", &part, 11)]);
    state
        .failed
        .push((label_path("retired"), retired.id(), true));
    state.epoch = Some(plan.epoch.epoch());
    expect(&log, state);
    let result = catch_unwind(AssertUnwindSafe(|| {
        reload_ok(&server, plan, candidate(&log, 10, &[("part", 11)]))
    }));
    if panic_at.is_some() {
        assert!(result.is_err(), "old Ready Drop should panic");
    } else {
        assert_eq!(result.unwrap().id(), root.id());
    }
    assert_drops(&server, &log, &[0, 1, 2]);
    assert_eq!(label_handle(&server, "part").id(), part.id());
    assert!(server.load_state(retired.id()).is_failed());
}

#[test]
fn successful_async_reload_replaces_root_and_label_and_retires_removed_label_unlocked() {
    reload_replaces_and_retires(None);
}

#[test]
fn async_reload_old_ready_drop_panic_preserves_full_publication_and_does_not_poison_mutex() {
    // Root replacement, label replacement, and removed-label failure all
    // defer their old Ready payloads; one panic must not skip the other Drops.
    for number in [0, 1, 2] {
        reload_replaces_and_retires(Some(number));
    }
}

#[derive(Clone, Copy, Debug)]
enum Rejection {
    SourceChanged,
    MissingRoot,
    DuplicateSource,
    DuplicateLabel,
    MissingRequestedLabel,
}

fn reload_rejects_unlocked(rejection: Rejection, panic_at: Option<u8>) {
    let (server, log) = scenario(panic_at);
    let (root, part, retired) = seed_source(&server, &log);
    let requested = if matches!(rejection, Rejection::MissingRequestedLabel) {
        label_path("part")
    } else {
        PATH.into()
    };
    let plan = server.prepare_async_reload(requested, None).unwrap();
    let mut state = publication(&root, 0, &[("part", &part, 1), ("retired", &retired, 2)]);
    state.epoch = Some(plan.epoch.epoch());
    state.absent = ["next", "third", "tail"].map(label_path).to_vec();
    expect(&log, state);
    let mut candidate = candidate(
        &log,
        10,
        &[("part", 11), ("next", 12), ("third", 13), ("tail", 14)],
    );
    let expected = match rejection {
        Rejection::SourceChanged => {
            server.mount("changed", Arc::new(crate::MemSource::new()), 0);
            "source overlay changed"
        }
        Rejection::MissingRoot => {
            candidate.assets[0].path = "different.commit_drop".into();
            "missing its root"
        }
        Rejection::DuplicateSource => {
            candidate.assets.push(AsyncReloadCandidateAsset {
                path: PATH.into(),
                type_id: AssetTypeId::of::<Probe>(),
                output: output(&log, 15, &[]),
            });
            "duplicate source"
        }
        Rejection::DuplicateLabel => {
            candidate.assets[0].output.labeled[2].label = "next".into();
            "duplicate labeled asset"
        }
        Rejection::MissingRequestedLabel => {
            candidate.assets[0].output.labeled[0].label = "different".into();
            "labeled asset was not emitted"
        }
    };
    let result = catch_unwind(AssertUnwindSafe(|| {
        server.commit_async_reload(plan, candidate)
    }));
    if panic_at.is_some() {
        assert!(
            result.is_err(),
            "expected candidate Drop panic: {rejection:?}"
        );
    } else {
        assert!(
            matches!(result.unwrap(), Err(AsyncReloadCommitError::Rejected(reason)) if reason.contains(expected)),
            "wrong rejection for {rejection:?}"
        );
    }
    let mut numbers = vec![10, 11, 12, 13, 14];
    if matches!(rejection, Rejection::DuplicateSource) {
        numbers.push(15);
    }
    assert_drops(&server, &log, &numbers);
}

#[test]
fn rejected_reload_candidate_drops_all_outputs_unlocked_without_changing_ready_revision() {
    for rejection in [
        Rejection::SourceChanged,
        Rejection::MissingRoot,
        Rejection::DuplicateSource,
        Rejection::DuplicateLabel,
        Rejection::MissingRequestedLabel,
    ] {
        reload_rejects_unlocked(rejection, None);
    }
}

#[test]
fn rejected_reload_candidate_drop_panic_does_not_poison_or_skip_remaining_payloads() {
    for number in 10..=14 {
        reload_rejects_unlocked(Rejection::DuplicateLabel, Some(number));
    }
}

fn superseded_candidate_drops_unlocked(panic_at: Option<u8>) {
    let (server, log) = scenario(panic_at);
    let (root, part, retired) = seed_source(&server, &log);
    let old = server.prepare_async_reload(PATH.into(), None).unwrap();
    let latest = server.prepare_async_reload(PATH.into(), None).unwrap();
    assert!(latest.epoch.epoch() > old.epoch.epoch());
    let mut state = publication(&root, 0, &[("part", &part, 1), ("retired", &retired, 2)]);
    state.epoch = Some(latest.epoch.epoch());
    expect(&log, state);
    let result = catch_unwind(AssertUnwindSafe(|| {
        server.commit_async_reload(old, candidate(&log, 10, &[("part", 11), ("retired", 12)]))
    }));
    if panic_at.is_some() {
        assert!(result.is_err(), "expected superseded candidate Drop panic");
    } else {
        assert!(matches!(
            result.unwrap(),
            Err(AsyncReloadCommitError::Superseded)
        ));
    }
    assert_drops(&server, &log, &[10, 11, 12]);
    // A stale candidate (including a panicking one) must not consume the
    // latest plan's commit capability or corrupt the surviving Ready slots.
    log.drops.lock().unwrap().clear();
    let mut state = publication(&root, 20, &[("part", &part, 21), ("retired", &retired, 22)]);
    state.epoch = Some(latest.epoch.epoch());
    expect(&log, state);
    assert_eq!(
        reload_ok(
            &server,
            latest,
            candidate(&log, 20, &[("part", 21), ("retired", 22)])
        )
        .id(),
        root.id()
    );
    assert_drops(&server, &log, &[0, 1, 2]);
}

#[test]
fn superseded_reload_candidate_drops_unlocked_and_latest_plan_can_still_commit() {
    superseded_candidate_drops_unlocked(None);
}

#[test]
fn superseded_reload_candidate_drop_panic_preserves_ready_revision_and_mutex() {
    for number in [10, 11, 12] {
        superseded_candidate_drops_unlocked(Some(number));
    }
}

#[test]
fn malformed_async_load_any_type_is_rejected_before_overwriting_any_ready_label() {
    // Exercise a bad primary and each early/late label. A bad primary must
    // not overwrite labels first; a bad late label must not overwrite earlier
    // labels. All rejected boxes, including the actual Other, are observed.
    for malformed_index in 0..4 {
        let (server, log) = scenario(None);
        let plan = prepare_load(&server);
        let root = plan.handle.clone();
        let part = seed_ready_label(&server, &log, "part", 1);
        let next = seed_ready_label(&server, &log, "next", 2);
        let mut state = publication(&root, 0, &[("part", &part, 1), ("next", &next, 2)]);
        state.ready.remove(0);
        state.failed.push((PATH.into(), root.id(), false));
        state.check_ownership = false;
        state.absent.push(label_path("tail"));
        expect(&log, state);
        let mut output = FullLoadOutput {
            primary: if malformed_index == 0 {
                malformed(&log, 10, "")
            } else {
                erased(&log, 10, "")
            },
            labeled: ["part", "next", "tail"]
                .iter()
                .enumerate()
                .map(|(index, label)| {
                    if malformed_index == index + 1 {
                        malformed(&log, 11 + index as u8, label)
                    } else {
                        erased(&log, 11 + index as u8, label)
                    }
                })
                .collect(),
        };
        // Keep the destination tag explicitly valid in every malformed case.
        output.primary.type_id = plan.type_id;
        let reason = server
            .commit_async_load(plan, output, vec![], vec![vec![], vec![], vec![]])
            .unwrap_err();
        assert!(reason.contains("does not match store type"), "{reason}");
        assert_drops(&server, &log, &[10, 11, 12, 13]);
    }
}

#[test]
fn malformed_async_reload_any_type_is_rejected_before_any_root_or_label_is_overwritten() {
    for malformed_index in 0..4 {
        let (server, log) = scenario(None);
        let (root, part, retired) = seed_source(&server, &log);
        let plan = server.prepare_async_reload(PATH.into(), None).unwrap();
        let mut state = publication(&root, 0, &[("part", &part, 1), ("retired", &retired, 2)]);
        state.epoch = Some(plan.epoch.epoch());
        state.absent.push(label_path("tail"));
        expect(&log, state);
        let output = FullLoadOutput {
            primary: if malformed_index == 0 {
                malformed(&log, 10, "")
            } else {
                erased(&log, 10, "")
            },
            labeled: ["part", "retired", "tail"]
                .iter()
                .enumerate()
                .map(|(index, label)| {
                    if malformed_index == index + 1 {
                        malformed(&log, 11 + index as u8, label)
                    } else {
                        erased(&log, 11 + index as u8, label)
                    }
                })
                .collect(),
        };
        let candidate = AsyncReloadCandidate {
            assets: vec![AsyncReloadCandidateAsset {
                path: PATH.into(),
                type_id: plan.type_id,
                output,
            }],
        };
        assert!(matches!(server.commit_async_reload(plan, candidate),
            Err(AsyncReloadCommitError::Rejected(reason)) if reason.contains("does not match store type")));
        assert_drops(&server, &log, &[10, 11, 12, 13]);
    }
}

#[test]
fn malformed_later_reload_source_does_not_partially_publish_earlier_source_outputs() {
    let (server, log) = scenario(None);
    let (root, part, retired) = seed_source(&server, &log);
    let other_path = AssetPath::from("other.commit_drop");
    let other_plan = match server
        .prepare_async_load(other_path.clone(), Some(AssetTypeId::of::<Probe>()))
        .unwrap()
    {
        AsyncLoadPreparation::New(plan) => *plan,
        AsyncLoadPreparation::Existing(_) => panic!("expected a fresh dependency source"),
    };
    let other = server
        .commit_async_load(other_plan, output(&log, 3, &[]), vec![], vec![])
        .unwrap();
    let plan = server.prepare_async_reload(PATH.into(), None).unwrap();
    let mut state = publication(&root, 0, &[("part", &part, 1), ("retired", &retired, 2)]);
    state.ready.push((other_path.clone(), other.id(), 3));
    state.absent.push(other_path.clone().with_label("tail"));
    state.epoch = Some(plan.epoch.epoch());
    expect(&log, state);
    // All outputs of the first source are valid. The next source's primary
    // has the right tag but wrong Any type, after every earlier output would
    // already have been replaced without complete-batch preflight.
    let mut candidate = candidate(&log, 10, &[("part", 11), ("retired", 12)]);
    candidate.assets.push(AsyncReloadCandidateAsset {
        path: other_path,
        type_id: AssetTypeId::of::<Probe>(),
        output: FullLoadOutput {
            primary: malformed(&log, 13, ""),
            labeled: vec![erased(&log, 14, "tail")],
        },
    });
    assert!(matches!(server.commit_async_reload(plan, candidate),
        Err(AsyncReloadCommitError::Rejected(reason)) if reason.contains("does not match store type")));
    assert_drops(&server, &log, &[10, 11, 12, 13, 14]);
}
