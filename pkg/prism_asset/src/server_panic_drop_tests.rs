//! Caught panic boxes and rejected erased outputs are user-owned Drop code too.

use super::*;
use crate::loader_exec::{AssetReadPlan, ErasedAssetLoader};
use crate::source::AssetMeta;
use crate::{LoadContext, LoadError, MemSource};
use core::any::Any;
use std::cell::RefCell;
use std::panic::panic_any;
use std::sync::Weak;

const PATH: &str = "asset.panic_drop";
const PANIC_REASON: &str = "asset source or loader panicked";

struct Value {
    number: u8,
    _probe: Option<Probe>,
}

impl Asset for Value {
    const TYPE_NAME: &'static str = "server_panic_drop_tests::Value";
}

struct Other;

impl Asset for Other {
    const TYPE_NAME: &'static str = "server_panic_drop_tests::Other";
}

#[derive(Clone, Debug)]
struct Observation {
    name: &'static str,
    unlocked: bool,
    reentered: bool,
    epoch_unlocked: Option<bool>,
    state: Option<LoadState>,
    number: Option<u8>,
    diagnostic: Option<String>,
    provisional_absent: bool,
    rollback_terminal: bool,
}

struct DropLog {
    server: Weak<AssetServer>,
    observations: Mutex<Vec<Observation>>,
    panic_at: Option<&'static str>,
    check_epoch_at: Option<&'static str>,
}

struct Probe {
    name: &'static str,
    log: Arc<DropLog>,
}

impl Drop for Probe {
    fn drop(&mut self) {
        let Some(server) = self.log.server.upgrade() else {
            return;
        };
        let mut observation = Observation {
            name: self.name,
            unlocked: false,
            reentered: false,
            epoch_unlocked: None,
            state: None,
            number: None,
            diagnostic: None,
            provisional_absent: false,
            rollback_terminal: false,
        };
        // A regression must produce a failed assertion, never a blocked test.
        // Reenter only after try_lock succeeded and that guard has gone away.
        let (root_id, reload_epoch) = match server.inner.try_lock() {
            Ok(inner) => {
                observation.unlocked = true;
                let root_id = inner
                    .interned
                    .get(&AssetPath::from(PATH))
                    .and_then(UntypedWeakHandle::upgrade)
                    .map(|handle| handle.id());
                if let Some(id) = root_id {
                    observation.state = Some(inner.stores.load_state(id));
                    observation.number = inner
                        .stores
                        .get(id.typed::<Value>().unwrap())
                        .map(|value| value.number);
                }
                observation.diagnostic = inner
                    .errors
                    .iter()
                    .filter(|(_, error)| error.path == PATH)
                    .last()
                    .map(|(_, error)| error.reason.clone());
                observation.provisional_absent = [
                    AssetPath::from(PATH).with_label("fresh"),
                    AssetPath::from(PATH).with_label("fresh2"),
                    AssetPath::from(PATH).with_label("broken"),
                    AssetPath::from(PATH).with_label("tail"),
                    AssetPath::from(PATH).with_label("emitted"),
                    AssetPath::from("dependency.panic_drop"),
                    AssetPath::from("dependency.panic_drop").with_label("part"),
                    AssetPath::from("dependency.panic_drop").with_label("broken"),
                    AssetPath::from("dependency.panic_drop").with_label("tail"),
                ]
                .into_iter()
                .all(|path| {
                    !inner.interned.contains_key(&path)
                        && !inner
                            .catalog
                            .contains_key(&StableGuid::from_path(&path.to_string()))
                });
                observation.rollback_terminal = inner.retained.iter().all(|(&id, handles)| {
                    // Rolled-back slots have no interned path. They must be
                    // failed in storage before any user Drop. Async reload
                    // rollback removes their graph nodes entirely, while load
                    // rollback leaves a matching Failed graph state.
                    let interned = inner
                        .interned
                        .values()
                        .any(|weak| weak.upgrade().is_some_and(|handle| handle.id() == id));
                    interned
                        || (handles.is_empty()
                            && inner.stores.load_state(id).is_failed()
                            && matches!(
                                inner.graph.self_state(id),
                                LoadState::NotLoaded | LoadState::Failed(_)
                            ))
                });
                let reload_epoch = (self.log.check_epoch_at == Some(self.name))
                    .then(|| inner.reload_epochs.get(&AssetPath::from(PATH)).cloned())
                    .flatten();
                (root_id, reload_epoch)
            }
            Err(_) => (None, None),
        };
        if observation.unlocked {
            observation.reentered = server.loader_count() == 1
                && root_id.is_some_and(|id| {
                    Some(server.load_state(id)) == observation.state
                        && server.with_asset(id.typed::<Value>().unwrap(), |value| value.number)
                            == observation.number
                });
        }
        if let Some(epoch) = reload_epoch {
            let (sender, receiver) = std::sync::mpsc::sync_channel(1);
            std::thread::spawn(move || {
                let _ = sender.send(epoch.try_begin().is_some());
            });
            observation.epoch_unlocked = Some(
                receiver
                    .recv_timeout(std::time::Duration::from_secs(2))
                    .unwrap_or(false),
            );
        }
        self.log.observations.lock().unwrap().push(observation);
        // All guards, including the observation guard, are gone before panic.
        if self.log.panic_at == Some(self.name) {
            panic!("expected destructor panic: {}", self.name);
        }
    }
}

fn value(number: u8) -> Value {
    Value {
        number,
        _probe: None,
    }
}

fn probe(log: &Arc<DropLog>, name: &'static str) -> Probe {
    Probe {
        name,
        log: Arc::clone(log),
    }
}

#[derive(Clone, Copy)]
enum Mode {
    Good,
    Error,
    Panic,
}

struct Loader {
    log: Arc<DropLog>,
    mode: Arc<Mutex<Mode>>,
}

impl AssetLoader for Loader {
    type Asset = Value;

    fn extensions(&self) -> &'static [&'static str] {
        &["panic_drop"]
    }

    fn load(&self, _bytes: &[u8], ctx: &mut LoadContext) -> Result<Value, LoadError> {
        // Never hold the control mutex while deliberately panicking.
        let mode = *self.mode.lock().unwrap();
        match mode {
            Mode::Good => {
                ctx.add_labeled_asset("stable", value(2));
                Ok(value(1))
            }
            Mode::Error | Mode::Panic => {
                ctx.add_labeled_asset(
                    "emitted",
                    Value {
                        number: 11,
                        _probe: Some(probe(&self.log, "context")),
                    },
                );
                match mode {
                    Mode::Error => Err(LoadError::Malformed("expected decoder error".into())),
                    Mode::Panic => panic_any(probe(&self.log, "panic")),
                    Mode::Good => unreachable!(),
                }
            }
        }
    }
}

struct Scenario {
    server: Arc<AssetServer>,
    log: Arc<DropLog>,
    mode: Arc<Mutex<Mode>>,
    loader: LoaderId,
}

impl Scenario {
    fn new(panic_on_drop: bool) -> Self {
        Self::with_panicking_drop(panic_on_drop.then_some("panic"))
    }

    fn with_panicking_drop(panic_at: Option<&'static str>) -> Self {
        Self::with_drop_checks(panic_at, None)
    }

    fn with_drop_checks(
        panic_at: Option<&'static str>,
        check_epoch_at: Option<&'static str>,
    ) -> Self {
        let server = Arc::new(AssetServer::new());
        let log = Arc::new(DropLog {
            server: Arc::downgrade(&server),
            observations: Mutex::new(Vec::new()),
            panic_at,
            check_epoch_at,
        });
        let mode = Arc::new(Mutex::new(Mode::Good));
        let (loader, _) = server.register_loader(Loader {
            log: Arc::clone(&log),
            mode: Arc::clone(&mode),
        });
        let mut source = MemSource::new();
        source.insert(PATH, vec![1]);
        server.mount("panic_drop", Arc::new(source), 0);
        Self {
            server,
            log,
            mode,
            loader,
        }
    }

    fn assert_drops(&self, names: &[&str], old_ready: bool, reason: &str) {
        let observations = self.log.observations.lock().unwrap().clone();
        let mut actual: Vec<_> = observations.iter().map(|drop| drop.name).collect();
        actual.sort();
        let mut expected = names.to_vec();
        expected.sort();
        assert_eq!(actual, expected, "{observations:?}");
        for drop in observations {
            assert!(drop.unlocked && drop.reentered, "{drop:?}");
            let expected_epoch = (self.log.check_epoch_at == Some(drop.name)).then_some(true);
            assert_eq!(drop.epoch_unlocked, expected_epoch, "{drop:?}");
            assert!(drop.provisional_absent, "{drop:?}");
            assert!(drop.rollback_terminal, "{drop:?}");
            assert_eq!(drop.diagnostic.as_deref(), Some(reason), "{drop:?}");
            if old_ready {
                assert_eq!(drop.state, Some(LoadState::Loaded), "{drop:?}");
                assert_eq!(drop.number, Some(1), "{drop:?}");
            } else {
                assert!(drop.state.is_some_and(LoadState::is_failed), "{drop:?}");
                assert_eq!(drop.number, None, "{drop:?}");
            }
        }
        assert!(!self.server.inner.is_poisoned());
        let inner = self.server.inner.try_lock().unwrap();
        assert!(inner.retired_payloads.is_empty());
        assert!(inner.staged_payloads.is_empty());
        assert!(inner.promotion_journal.is_empty());
        assert_eq!(inner.promotion_depth, 0);
    }

    fn replace_erased(&self, fault: ErasedFault) {
        self.server.lock().loaders.replace_erased_for_test(
            self.loader,
            Arc::new(ErasedLoader {
                log: Arc::clone(&self.log),
                fault,
            }),
        );
    }
}

#[test]
fn custom_panic_payload_and_context_labels_drop_after_failed_load_unlocks() {
    let scenario = Scenario::new(false);
    *scenario.mode.lock().unwrap() = Mode::Panic;
    let root = scenario.server.load::<Value>(PATH);
    scenario.assert_drops(&["context", "panic"], false, PANIC_REASON);
    let inner = scenario.server.inner.try_lock().unwrap();
    let error = inner
        .stores
        .load_state(root.id().untyped())
        .error()
        .unwrap();
    assert_eq!(inner.errors.get(error).unwrap().reason, PANIC_REASON);
    assert_eq!(
        inner.graph.self_state(root.id().untyped()),
        LoadState::Failed(error)
    );
}

#[test]
fn custom_panic_payload_reenters_after_reload_preserves_last_good_revision() {
    let scenario = Scenario::new(false);
    let root = scenario.server.load::<Value>(PATH);
    let label = scenario
        .server
        .load::<Value>(AssetPath::from(PATH).with_label("stable"));
    *scenario.mode.lock().unwrap() = Mode::Panic;
    assert_eq!(
        scenario.server.reload(PATH).unwrap().id(),
        root.id().untyped()
    );
    scenario.assert_drops(&["context", "panic"], true, PANIC_REASON);
    assert_eq!(
        scenario.server.with_asset(label.id(), |value| value.number),
        Some(2)
    );
    // The failed revision must not leave the epoch gate held either.
    *scenario.mode.lock().unwrap() = Mode::Good;
    assert_eq!(
        scenario.server.reload(PATH).unwrap().id(),
        root.id().untyped()
    );
    assert!(scenario.server.is_loaded(root.id().untyped()));
}

#[test]
fn context_labels_drop_after_loader_error_with_diagnostic_already_recorded() {
    let scenario = Scenario::new(false);
    *scenario.mode.lock().unwrap() = Mode::Error;
    let root = scenario.server.load::<Value>(PATH);
    scenario.assert_drops(
        &["context"],
        false,
        "malformed asset: expected decoder error",
    );
    assert!(scenario.server.load_state(root.id().untyped()).is_failed());
}

#[test]
fn panic_payload_destructor_panic_does_not_poison_or_leave_loading_slot() {
    let scenario = Scenario::new(true);
    // Hold a last-good handle so the slot stays observable if returning the
    // reloaded handle is interrupted by the deferred destructor's own panic.
    let root = scenario.server.load::<Value>(PATH);
    *scenario.mode.lock().unwrap() = Mode::Panic;
    let result = catch_unwind(AssertUnwindSafe(|| scenario.server.reload(PATH)));
    assert!(result.is_err());
    scenario.assert_drops(&["context", "panic"], true, PANIC_REASON);
    assert_eq!(
        scenario.server.load_state(root.id().untyped()),
        LoadState::Loaded
    );
    *scenario.mode.lock().unwrap() = Mode::Good;
    assert!(scenario.server.reload(PATH).is_some());
}

#[derive(Clone, Copy)]
enum ErasedFault {
    #[cfg(debug_assertions)]
    WrongTag,
    #[cfg(debug_assertions)]
    ProductTypePanic,
    StoreFactoryPanic,
}

struct ErasedLoader {
    log: Arc<DropLog>,
    fault: ErasedFault,
}

fn erased(log: &Arc<DropLog>, name: &'static str, label: &str) -> ErasedLoadedAsset {
    ErasedLoadedAsset {
        type_id: AssetTypeId::of::<Value>(),
        value: Box::new(Value {
            number: 10,
            _probe: Some(probe(log, name)),
        }),
        dependencies: Vec::new(),
        label: label.into(),
        make_store: || Box::new(Assets::<Value>::new()),
    }
}

impl ErasedAssetLoader for ErasedLoader {
    fn produced_type(&self) -> AssetTypeId {
        #[cfg(debug_assertions)]
        if matches!(self.fault, ErasedFault::ProductTypePanic) {
            panic_any(probe(&self.log, "panic"));
        }
        AssetTypeId::of::<Value>()
    }

    fn read_plan(
        &self,
        _path: &AssetPath,
        _metadata: AssetMeta,
    ) -> Result<AssetReadPlan, LoadError> {
        Ok(AssetReadPlan::Whole)
    }

    fn load_erased(
        &self,
        _path: &AssetPath,
        _bytes: &[u8],
        _retired: Option<&mut Vec<Box<dyn Any + Send>>>,
    ) -> Result<FullLoadOutput, LoadError> {
        let primary = erased(&self.log, "root", "");
        match self.fault {
            #[cfg(debug_assertions)]
            ErasedFault::WrongTag => {
                let mut primary = primary;
                primary.type_id = AssetTypeId::of::<Other>();
                Ok(FullLoadOutput {
                    primary,
                    labeled: vec![
                        erased(&self.log, "label", "fresh"),
                        erased(&self.log, "tail", "tail"),
                    ],
                })
            }
            #[cfg(debug_assertions)]
            ErasedFault::ProductTypePanic => Ok(FullLoadOutput {
                primary,
                labeled: vec![
                    erased(&self.log, "label", "fresh"),
                    erased(&self.log, "tail", "tail"),
                ],
            }),
            ErasedFault::StoreFactoryPanic => {
                let mut broken = erased(&self.log, "broken", "broken");
                broken.type_id = AssetTypeId::of::<Other>();
                broken.make_store = || panic!("expected labeled store factory panic");
                Ok(FullLoadOutput {
                    primary,
                    labeled: vec![
                        erased(&self.log, "label", "fresh"),
                        broken,
                        erased(&self.log, "tail", "tail"),
                    ],
                })
            }
        }
    }
}

#[cfg(debug_assertions)]
#[test]
fn output_type_assertion_retires_whole_output_before_unwinding() {
    let scenario = Scenario::new(false);
    scenario.replace_erased(ErasedFault::WrongTag);
    let root = scenario.server.load::<Value>(PATH);
    scenario.assert_drops(&["root", "label", "tail"], false, PANIC_REASON);
    assert!(scenario.server.load_state(root.id().untyped()).is_failed());
}

#[cfg(debug_assertions)]
#[test]
fn output_type_callback_custom_panic_retires_output_and_panic_box() {
    let scenario = Scenario::new(false);
    scenario.replace_erased(ErasedFault::ProductTypePanic);
    let root = scenario.server.load::<Value>(PATH);
    scenario.assert_drops(&["root", "label", "tail", "panic"], false, PANIC_REASON);
    assert!(scenario.server.load_state(root.id().untyped()).is_failed());
}

#[test]
fn store_factory_unwind_rolls_back_prior_provisional_labels_before_drop() {
    let scenario = Scenario::new(false);
    scenario.replace_erased(ErasedFault::StoreFactoryPanic);
    let root = scenario.server.load::<Value>(PATH);
    scenario.assert_drops(&["root", "label", "broken", "tail"], false, PANIC_REASON);
    assert!(scenario.server.load_state(root.id().untyped()).is_failed());
    // The provisional slot may await collection, but it is never left Loading
    // or reachable through the intern/catalog tables when deferred Drop runs.
    {
        let inner = scenario.server.inner.try_lock().unwrap();
        assert_eq!(inner.retained.len(), 2);
        for &id in inner.retained.keys() {
            assert!(inner.stores.load_state(id).is_failed());
            assert_eq!(inner.graph.self_state(id), inner.stores.load_state(id));
        }
    }
    assert_eq!(scenario.server.collect_releases(), 1);
    assert_eq!(scenario.server.total_assets(), 1);
    drop(root);
    assert_eq!(scenario.server.collect_releases(), 1);
    assert_eq!(scenario.server.total_assets(), 0);
}

thread_local! {
    // make_store is a fn pointer, not a capturing closure. Thread-local state
    // keeps parallel tests isolated without a global lock across the panic.
    static FACTORY_PANIC: RefCell<Option<Probe>> = const { RefCell::new(None) };
}

fn panic_store_factory() -> Box<dyn crate::stores::ErasedAssetStore> {
    let payload = FACTORY_PANIC.with(|slot| slot.borrow_mut().take());
    panic_any(payload.expect("factory panic probe is armed"));
}

fn async_factory_unwind(custom_payload: bool, panic_at: Option<&'static str>) {
    let scenario = Scenario::with_panicking_drop(panic_at);
    let plan = match scenario
        .server
        .prepare_async_load(PATH.into(), Some(AssetTypeId::of::<Value>()))
        .unwrap()
    {
        AsyncLoadPreparation::New(plan) => *plan,
        AsyncLoadPreparation::Existing(_) => panic!("expected a fresh async load"),
    };
    // Keep the root observable even if a deferred destructor interrupts the
    // method's return. The commit must itself publish failure before any Drop.
    let root = plan.handle.clone();
    let stable_path = AssetPath::from(PATH).with_label("stable");
    let stable = {
        let mut inner = scenario.server.lock();
        let handle = inner.stores.insert(value(2));
        inner.graph.add_asset(handle.id());
        let _ = inner.graph.set_self_state(handle.id(), LoadState::Loaded);
        inner.retained.entry(handle.id()).or_default();
        inner.owner_weaks.insert(handle.id(), handle.downgrade());
        inner
            .interned
            .insert(stable_path.clone(), handle.downgrade());
        inner.catalog.insert(
            StableGuid::from_path(&stable_path.to_string()),
            stable_path.clone(),
        );
        handle
    };

    let mut broken = erased(&scenario.log, "broken", "broken");
    broken.type_id = AssetTypeId::of::<Other>();
    if custom_payload {
        FACTORY_PANIC.with(|slot| {
            assert!(slot.borrow().is_none());
            *slot.borrow_mut() = Some(probe(&scenario.log, "panic"));
        });
        broken.make_store = panic_store_factory;
    } else {
        broken.make_store = || panic!("expected async labeled store factory panic");
    }
    let output = FullLoadOutput {
        primary: erased(&scenario.log, "root", ""),
        labeled: vec![
            erased(&scenario.log, "existing", "stable"),
            erased(&scenario.log, "label", "fresh"),
            erased(&scenario.log, "second", "fresh2"),
            broken,
            erased(&scenario.log, "tail", "tail"),
        ],
    };
    let result = catch_unwind(AssertUnwindSafe(|| {
        scenario
            .server
            .commit_async_load(plan, output, Vec::new(), vec![Vec::new(); 5])
    }));
    if panic_at.is_some() {
        assert!(result.is_err(), "expected deferred destructor panic");
    } else {
        // The erased callback panic is converted into a commit rejection,
        // without relying on the scheduler's outer unwind cleanup.
        assert_eq!(result.unwrap().unwrap_err(), PANIC_REASON);
    }
    FACTORY_PANIC.with(|slot| assert!(slot.borrow().is_none()));
    let mut names = vec!["root", "existing", "label", "second", "broken", "tail"];
    if custom_payload {
        names.push("panic");
    }
    scenario.assert_drops(&names, false, PANIC_REASON);
    {
        let inner = scenario.server.inner.try_lock().unwrap();
        assert_eq!(inner.retained.len(), 4);
        assert_eq!(inner.interned.len(), 2);
        assert_eq!(inner.catalog.len(), 2);
        assert!(!inner.stores.contains_type(AssetTypeId::of::<Other>()));
        for &id in inner.retained.keys() {
            assert!(inner.retained[&id].is_empty());
            assert_eq!(inner.graph.self_state(id), inner.stores.load_state(id));
            assert!(inner.graph.dependencies(id).is_empty());
            if id != stable.id() {
                assert!(inner.stores.load_state(id).is_failed());
            }
        }
        // A preexisting label was only staged, not published or rolled back.
        assert_eq!(inner.stores.load_state(stable.id()), LoadState::Loaded);
        assert_eq!(
            inner
                .stores
                .get(stable.id().typed::<Value>().unwrap())
                .unwrap()
                .number,
            2
        );
        assert_eq!(
            inner.interned[&stable_path].upgrade().unwrap().id(),
            stable.id()
        );
        assert_eq!(
            inner.catalog[&StableGuid::from_path(&stable_path.to_string())],
            stable_path
        );
    }
    assert_eq!(scenario.server.collect_releases(), 2);
    assert_eq!(scenario.server.total_assets(), 2);

    // Failure does not pin the path to its terminal allocation: a subsequent
    // async plan can commit while the old diagnostic root is still held.
    let retry = match scenario
        .server
        .prepare_async_load(PATH.into(), Some(AssetTypeId::of::<Value>()))
        .unwrap()
    {
        AsyncLoadPreparation::New(plan) => *plan,
        AsyncLoadPreparation::Existing(_) => panic!("expected a fresh retry allocation"),
    };
    assert_ne!(retry.id, root.id());
    let retry_output = FullLoadOutput {
        primary: ErasedLoadedAsset {
            type_id: AssetTypeId::of::<Value>(),
            value: Box::new(value(3)),
            dependencies: Vec::new(),
            label: String::new(),
            make_store: || Box::new(Assets::<Value>::new()),
        },
        labeled: Vec::new(),
    };
    let retry = scenario
        .server
        .commit_async_load(retry, retry_output, Vec::new(), Vec::new())
        .unwrap();
    assert!(scenario.server.is_loaded(retry.id()));
    assert!(scenario.server.load_state(root.id()).is_failed());
    drop(root);
    drop(stable);
    drop(retry);
    assert_eq!(scenario.server.collect_releases(), 3);
    assert_eq!(scenario.server.total_assets(), 0);
}

fn async_reload_factory_unwind(custom_payload: bool, panic_at: Option<&'static str>) {
    let epoch_probe = if custom_payload { "panic" } else { "broken" };
    let scenario = Scenario::with_drop_checks(panic_at, Some(epoch_probe));
    let root = scenario.server.load::<Value>(PATH);
    let stable_path = AssetPath::from(PATH).with_label("stable");
    let stable = scenario.server.load::<Value>(stable_path.clone());
    assert_eq!(
        scenario.server.with_asset(root.id(), |value| value.number),
        Some(1)
    );
    assert_eq!(
        scenario
            .server
            .with_asset(stable.id(), |value| value.number),
        Some(2)
    );

    let plan = scenario
        .server
        .prepare_async_reload(PATH.into(), None)
        .unwrap();
    let dependency_path = AssetPath::from("dependency.panic_drop");
    let mut root_primary = erased(&scenario.log, "root", "");
    root_primary.dependencies = vec![DepRequest::new(dependency_path.clone())];
    let mut broken = erased(&scenario.log, "broken", "broken");
    broken.type_id = AssetTypeId::of::<Other>();
    if custom_payload {
        FACTORY_PANIC.with(|slot| {
            assert!(slot.borrow().is_none());
            *slot.borrow_mut() = Some(probe(&scenario.log, "panic"));
        });
        broken.make_store = panic_store_factory;
    } else {
        broken.make_store = || panic!("expected async reload label store factory panic");
    }
    let candidate = AsyncReloadCandidate {
        assets: vec![
            AsyncReloadCandidateAsset {
                path: PATH.into(),
                type_id: AssetTypeId::of::<Value>(),
                output: FullLoadOutput {
                    primary: root_primary,
                    labeled: vec![
                        erased(&scenario.log, "existing", "stable"),
                        erased(&scenario.log, "label", "fresh"),
                        erased(&scenario.log, "second", "fresh2"),
                    ],
                },
            },
            AsyncReloadCandidateAsset {
                path: dependency_path.clone(),
                type_id: AssetTypeId::of::<Value>(),
                output: FullLoadOutput {
                    primary: erased(&scenario.log, "dependency", ""),
                    labeled: vec![
                        erased(&scenario.log, "dependency_label", "part"),
                        broken,
                        erased(&scenario.log, "tail", "tail"),
                    ],
                },
            },
        ],
    };
    let result = catch_unwind(AssertUnwindSafe(|| {
        scenario.server.commit_async_reload(plan, candidate)
    }));
    if panic_at.is_some() {
        assert!(result.is_err(), "expected deferred destructor panic");
    } else {
        assert!(matches!(
            result.unwrap(),
            Err(AsyncReloadCommitError::Rejected(reason)) if reason == PANIC_REASON
        ));
    }
    FACTORY_PANIC.with(|slot| assert!(slot.borrow().is_none()));
    let mut names = vec![
        "root",
        "existing",
        "label",
        "second",
        "dependency",
        "dependency_label",
        "broken",
        "tail",
    ];
    if custom_payload {
        names.push("panic");
    }
    scenario.assert_drops(&names, true, PANIC_REASON);

    {
        let inner = scenario.server.inner.try_lock().unwrap();
        assert_eq!(inner.interned.len(), 2);
        assert_eq!(inner.catalog.len(), 2);
        assert!(!inner.stores.contains_type(AssetTypeId::of::<Other>()));
        assert_eq!(
            inner.stores.load_state(root.id().untyped()),
            LoadState::Loaded
        );
        assert_eq!(
            inner.stores.load_state(stable.id().untyped()),
            LoadState::Loaded
        );
        assert_eq!(
            inner
                .stores
                .get(root.id().untyped().typed::<Value>().unwrap())
                .unwrap()
                .number,
            1
        );
        assert_eq!(
            inner
                .stores
                .get(stable.id().untyped().typed::<Value>().unwrap())
                .unwrap()
                .number,
            2
        );
        assert_eq!(
            inner.interned[&AssetPath::from(PATH)]
                .upgrade()
                .unwrap()
                .id(),
            root.id().untyped()
        );
        assert_eq!(
            inner.interned[&stable_path].upgrade().unwrap().id(),
            stable.id().untyped()
        );
        let rolled_back: Vec<_> = inner
            .retained
            .keys()
            .copied()
            .filter(|id| *id != root.id().untyped() && *id != stable.id().untyped())
            .collect();
        assert_eq!(rolled_back.len(), 4);
        for id in rolled_back {
            assert!(inner.retained[&id].is_empty());
            assert!(inner.stores.load_state(id).is_failed());
            assert_eq!(inner.graph.self_state(id), LoadState::NotLoaded);
            assert!(inner.graph.dependencies(id).is_empty());
        }
    }
    assert_eq!(scenario.server.collect_releases(), 4);
    assert_eq!(scenario.server.total_assets(), 2);

    // The bounded Drop probe advanced the epoch only after both publication
    // locks were released. A later reload can still publish normally.
    let retry = scenario
        .server
        .prepare_async_reload(PATH.into(), None)
        .unwrap();
    let retry_candidate = AsyncReloadCandidate {
        assets: vec![AsyncReloadCandidateAsset {
            path: PATH.into(),
            type_id: AssetTypeId::of::<Value>(),
            output: FullLoadOutput {
                primary: ErasedLoadedAsset {
                    type_id: AssetTypeId::of::<Value>(),
                    value: Box::new(value(3)),
                    dependencies: Vec::new(),
                    label: String::new(),
                    make_store: || Box::new(Assets::<Value>::new()),
                },
                labeled: vec![ErasedLoadedAsset {
                    type_id: AssetTypeId::of::<Value>(),
                    value: Box::new(value(4)),
                    dependencies: Vec::new(),
                    label: "stable".into(),
                    make_store: || Box::new(Assets::<Value>::new()),
                }],
            },
        }],
    };
    let retry_handle = match scenario.server.commit_async_reload(retry, retry_candidate) {
        Ok(handle) => handle,
        Err(AsyncReloadCommitError::Rejected(reason)) => {
            panic!("unexpected async reload rejection: {reason}")
        }
        Err(AsyncReloadCommitError::Superseded) => {
            panic!("unexpected async reload supersession")
        }
    };
    assert_eq!(retry_handle.id(), root.id().untyped());
    assert_eq!(
        scenario.server.with_asset(root.id(), |value| value.number),
        Some(3)
    );
    assert_eq!(
        scenario
            .server
            .with_asset(stable.id(), |value| value.number),
        Some(4)
    );
}

#[test]
fn async_store_factory_panic_rolls_back_multiple_labels_before_candidate_drops() {
    async_factory_unwind(false, None);
}

#[test]
fn async_store_factory_custom_panic_payload_reenters_after_rollback() {
    async_factory_unwind(true, None);
}

#[test]
fn async_store_factory_panic_payload_destructor_panic_keeps_failure_unpoisoned() {
    async_factory_unwind(true, Some("panic"));
}

#[test]
fn async_store_factory_candidate_destructor_panics_do_not_skip_unlocked_drops() {
    for name in ["root", "existing", "label", "second", "broken", "tail"] {
        async_factory_unwind(false, Some(name));
    }
}

#[test]
fn async_reload_store_factory_panic_rolls_back_dependency_and_multiple_labels() {
    async_reload_factory_unwind(false, None);
}

#[test]
fn async_reload_store_factory_custom_panic_payload_drops_outside_both_locks() {
    async_reload_factory_unwind(true, None);
}

#[test]
fn async_reload_store_factory_panic_payload_destructor_panic_preserves_old_revision() {
    async_reload_factory_unwind(true, Some("panic"));
}

#[test]
fn async_reload_store_factory_candidate_destructor_panics_after_rollback() {
    for name in [
        "root",
        "existing",
        "label",
        "second",
        "dependency",
        "dependency_label",
        "broken",
        "tail",
    ] {
        async_reload_factory_unwind(false, Some(name));
    }
}
