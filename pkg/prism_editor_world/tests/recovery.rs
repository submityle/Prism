//! Regression tests for storage execution, policy gates, and failure recovery.

use std::{
    collections::BTreeMap,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
    task::{Context, Poll, Wake, Waker},
};

use prism_asset::StableGuid;
use prism_ecs::partition::{cell::CellCoord, data_layer::DataLayerId};
use prism_editor_world::{
    CellIoProvider, CellLoadFuture, CellLoadRequest, CellLockError, CellLockRegistry,
    CellPayloadError, CellResidency, CellSaveFuture, CellSaveRequest, DataLayer, EditorWorldData,
    EditorWorldDataError, EditorWorldLimits, HlodCluster, HlodLayer, HlodPlanner, HlodView,
    IoBudget, IoCompletion, IoSubmitError, LoadSubmitError, OfpaCatalog, OfpaError, PlanBlock,
    ResidencyDemand, WorldError, WorldIoDriver,
};

#[derive(Clone, Default)]
struct MemoryProvider {
    files: Arc<Mutex<BTreeMap<String, Vec<u8>>>>,
    reject_load: bool,
}

impl CellIoProvider for MemoryProvider {
    type Error = &'static str;

    fn load_cell(
        &self,
        request: CellLoadRequest,
    ) -> Result<CellLoadFuture<Self::Error>, Self::Error> {
        if self.reject_load {
            return Err("load startup rejected");
        }
        let files = self.files.clone();
        let mut yielded = false;
        Ok(Box::pin(std::future::poll_fn(move |context| {
            if !yielded {
                yielded = true;
                context.waker().wake_by_ref();
                return Poll::Pending;
            }
            let files = files.lock().expect("storage mutex");
            Poll::Ready(match files.get(request.file().relative_path()) {
                Some(bytes) if bytes.len() <= request.max_bytes() => Ok(bytes.clone()),
                Some(_) => Err("payload exceeds read budget"),
                None => Err("cell not found"),
            })
        })))
    }

    fn save_cell(
        &self,
        request: CellSaveRequest,
        payload: Vec<u8>,
    ) -> Result<CellSaveFuture<Self::Error>, Self::Error> {
        let files = self.files.clone();
        Ok(Box::pin(async move {
            files
                .lock()
                .expect("storage mutex")
                .insert(request.file().relative_path().to_owned(), payload);
            Ok(())
        }))
    }
}

#[derive(Default)]
struct WakeCounter(AtomicUsize);

impl Wake for WakeCounter {
    fn wake(self: Arc<Self>) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

fn world() -> EditorWorldData {
    EditorWorldData::new(
        StableGuid::from_path("world/recovery"),
        EditorWorldLimits::new(8, 32, 8, 4, 8).expect("limits"),
    )
    .expect("world")
}

fn coord() -> CellCoord {
    CellCoord::new(0, 0, 0)
}

fn load_direct(world: &mut EditorWorldData, cell: CellCoord) {
    let mut demand = ResidencyDemand::new();
    demand.require(cell);
    let (_, tokens) = world.begin_residency(&demand, 0).expect("load plan");
    world
        .complete_load(tokens[0], b"initial".to_vec())
        .expect("load payload");
}

fn next(driver: &mut WorldIoDriver<MemoryProvider>) -> IoCompletion<&'static str> {
    let mut context = Context::from_waker(Waker::noop());
    for _ in 0..2 {
        if let Poll::Ready(Some(completion)) = driver.poll(&mut context) {
            return completion;
        }
    }
    panic!("memory provider should complete within two polls");
}

#[test]
fn asynchronous_load_save_evict_reload_round_trips_opaque_bytes() {
    let mut world = world();
    world.add_cell(coord(), []).expect("register cell");
    let provider = MemoryProvider::default();
    provider.files.lock().expect("storage").insert(
        world
            .cell_file(coord())
            .expect("file")
            .relative_path()
            .to_owned(),
        b"initial".to_vec(),
    );
    let mut driver = world.make_io_driver(provider, IoBudget::new(1, 32).expect("budget"));
    let mut demand = ResidencyDemand::new();
    demand.require(coord());
    let (_, tokens) = world.begin_residency(&demand, 0).expect("plan");
    world.submit_load(&mut driver, tokens[0]).expect("queue");
    assert!(matches!(
        world.submit_load(&mut driver, tokens[0]),
        Err(LoadSubmitError::Io(IoSubmitError::CellBusy(_)))
    ));
    let counter = Arc::new(WakeCounter::default());
    let waker = Waker::from(counter.clone());
    assert!(driver.poll(&mut Context::from_waker(&waker)).is_pending());
    assert_eq!(counter.0.load(Ordering::SeqCst), 1);
    assert_eq!(driver.reserved_bytes(), 32);
    world
        .apply_load_completion(next(&mut driver))
        .expect("apply read");
    assert_eq!(driver.in_flight(), 0);

    let lock = world
        .acquire_lock(coord(), StableGuid::from_path("editor/a"), 1, 20)
        .expect("lease");
    world
        .replace_payload(coord(), &lock, 1, b"edited".to_vec())
        .expect("edit");
    let pending = world.prepare_save(coord(), &lock, 1).expect("snapshot");
    driver
        .submit_save(pending.request().clone(), pending.payload().to_vec())
        .expect("queue write");
    assert!(world
        .apply_save_completion(&pending, next(&mut driver), 2)
        .expect("commit write"));
    assert!(!world.model().cell(coord()).expect("cell").is_dirty());
    world.release_lock(&lock).expect("release");
    let (_, tokens) = world
        .begin_residency(&ResidencyDemand::new(), 2)
        .expect("evict");
    world
        .complete_unload(tokens[0])
        .expect("drop resident bytes");
    assert!(world.payload(coord()).is_none());
    let (_, tokens) = world.begin_residency(&demand, 3).expect("reload");
    world
        .submit_load(&mut driver, tokens[0])
        .expect("queue reload");
    world
        .apply_load_completion(next(&mut driver))
        .expect("restore saved bytes");
    assert_eq!(world.payload(coord()), Some(&b"edited"[..]));
}

#[test]
fn old_or_unrelated_save_acknowledgement_cannot_clear_dirty_state() {
    let mut world = world();
    world.add_cell(coord(), []).expect("cell");
    load_direct(&mut world, coord());
    let lock = world
        .acquire_lock(coord(), StableGuid::from_path("editor/a"), 1, 20)
        .expect("lease");
    world.mark_dirty(coord(), &lock, 1).expect("edit one");
    let old = world.prepare_save(coord(), &lock, 1).expect("snapshot one");
    world
        .mark_dirty(coord(), &lock, 2)
        .expect("byte-identical second edit");
    assert!(matches!(
        world.commit_save(&old, 2),
        Err(EditorWorldDataError::Model(WorldError::StalePlan { .. }))
    ));

    let current = world
        .prepare_save(coord(), &lock, 2)
        .expect("current snapshot");
    let mut driver = world.make_io_driver(
        MemoryProvider::default(),
        IoBudget::new(1, 32).expect("budget"),
    );
    driver
        .submit_save(old.request().clone(), old.payload().to_vec())
        .expect("old write");
    assert_eq!(
        world.apply_save_completion(&current, next(&mut driver), 2),
        Err(EditorWorldDataError::UnexpectedCompletion)
    );
    assert!(world.model().cell(coord()).expect("cell").is_dirty());
}

#[test]
fn live_lock_and_dirty_state_block_eviction_and_unload_failure_keeps_bytes() {
    let mut world = world();
    world.add_cell(coord(), []).expect("cell");
    load_direct(&mut world, coord());
    let lock = world
        .acquire_lock(coord(), StableGuid::from_path("editor/a"), 1, 3)
        .expect("lease");
    let empty = ResidencyDemand::new();
    let plan = world.plan_residency(&empty, 2).expect("plan");
    assert_eq!(plan.blocked(), &[PlanBlock::LockedCell(coord())]);
    assert!(plan.operations().is_empty());
    world.mark_dirty(coord(), &lock, 2).expect("edit");
    assert_eq!(
        world
            .plan_residency(&empty, 4)
            .expect("dirty plan")
            .blocked(),
        &[PlanBlock::DirtyCell(coord())]
    );
    world.renew_lock(&lock, 3, 10).expect("renew before expiry");
    let pending = world.prepare_save(coord(), &lock, 3).expect("snapshot");
    world
        .commit_save(&pending, 3)
        .expect("acknowledge persisted snapshot");
    world.release_lock(&lock).expect("release");
    let (_, tokens) = world.begin_residency(&empty, 4).expect("unload");
    assert!(world
        .acquire_lock(coord(), StableGuid::from_path("editor/b"), 4, 2)
        .is_err());
    world.fail_unload(tokens[0]).expect("rollback eviction");
    assert_eq!(world.payload(coord()), Some(&b"initial"[..]));
    let (_, retried) = world.begin_residency(&empty, 4).expect("retry");
    assert!(world.complete_unload(tokens[0]).is_err());
    world.complete_unload(retried[0]).expect("unload succeeds");
    assert!(world.payload(coord()).is_none());
}

#[test]
fn rejection_and_oversized_payloads_restore_retryable_unloaded_state() {
    let mut world = world();
    world.add_cell(coord(), []).expect("cell");
    let mut demand = ResidencyDemand::new();
    demand.require(coord());
    let (_, tokens) = world.begin_residency(&demand, 0).expect("begin");
    let mut driver = world.make_io_driver(
        MemoryProvider::default(),
        IoBudget::new(1, 8).expect("budget"),
    );
    assert!(matches!(
        world.submit_load(&mut driver, tokens[0]),
        Err(LoadSubmitError::Io(
            IoSubmitError::ByteBudgetExceeded { .. }
        ))
    ));
    assert_eq!(
        world.model().cell(coord()).expect("cell").residency(),
        CellResidency::Unloaded
    );
    let (_, tokens) = world.begin_residency(&demand, 0).expect("retry");
    let mut driver = world.make_io_driver(
        MemoryProvider {
            reject_load: true,
            ..MemoryProvider::default()
        },
        IoBudget::new(1, 32).expect("budget"),
    );
    assert!(matches!(
        world.submit_load(&mut driver, tokens[0]),
        Err(LoadSubmitError::Io(IoSubmitError::Provider(
            "load startup rejected"
        )))
    ));
    assert_eq!(driver.in_flight(), 0);
    assert_eq!(driver.reserved_bytes(), 0);
    let (_, tokens) = world.begin_residency(&demand, 0).expect("retry");
    assert_eq!(
        world.complete_load(tokens[0], vec![0; 33]),
        Err(EditorWorldDataError::Payload(CellPayloadError::TooLarge {
            actual: 33,
            limit: 32
        }))
    );
    assert_eq!(
        world.model().cell(coord()).expect("cell").residency(),
        CellResidency::Unloaded
    );
    assert!(world.payload(coord()).is_none());
    load_direct(&mut world, coord());
    assert_eq!(world.payload(coord()), Some(&b"initial"[..]));
}

#[test]
fn hierarchical_edit_policy_and_actor_commits_preserve_cell_dirtiness() {
    let mut world = world();
    let parent = DataLayerId::new(1);
    let child = DataLayerId::new(2);
    world
        .add_layer(DataLayer::new(parent, "parent").expect("parent"))
        .expect("register parent");
    world
        .add_layer(DataLayer::with_parent(child, "child", parent).expect("child"))
        .expect("register child");
    world.add_cell(coord(), [child]).expect("cell");
    load_direct(&mut world, coord());
    let actor = StableGuid::from_path("actor/a");
    world.register_actor(actor, coord()).expect("actor");
    let lock = world
        .acquire_lock(coord(), StableGuid::from_path("editor/a"), 1, 20)
        .expect("lease");
    world.set_layer_visible(parent, false).expect("hide parent");
    assert_eq!(world.visibility_plan().hidden(), &[coord()]);
    world
        .set_layer_editable(parent, false)
        .expect("disable parent edits");
    assert!(world.mark_actor_dirty(actor, &lock, 1).is_err());
    assert!(world.plan_ofpa().operations().is_empty());
    assert!(!world.model().cell(coord()).expect("cell").is_dirty());
    world
        .set_layer_editable(parent, true)
        .expect("enable edits");
    world.mark_actor_dirty(actor, &lock, 1).expect("edit actor");
    let plan = world.plan_ofpa();
    let pending = world.prepare_save(coord(), &lock, 1).expect("snapshot");
    assert_eq!(
        world.commit_save(&pending, 1),
        Err(EditorWorldDataError::DirtyActors(coord()))
    );
    world
        .mark_actor_dirty(actor, &lock, 2)
        .expect("new actor edit");
    assert!(matches!(
        world.commit_ofpa(&plan.operations()[0], &lock, 2),
        Err(EditorWorldDataError::Ofpa(OfpaError::StaleRevision { .. }))
    ));
    let plan = world.plan_ofpa();
    world
        .commit_ofpa(&plan.operations()[0], &lock, 2)
        .expect("persist current actor");
    assert!(world.plan_ofpa().operations().is_empty());
    assert!(world.model().cell(coord()).expect("cell").is_dirty());
    let pending = world
        .prepare_save(coord(), &lock, 2)
        .expect("current cell snapshot");
    world
        .commit_save(&pending, 2)
        .expect("persist cell manifest");
    assert!(!world.model().cell(coord()).expect("cell").is_dirty());
}

#[test]
fn ofpa_order_and_identity_are_independent_of_registration_order() {
    let a = StableGuid::from_path("actor/a");
    let b = StableGuid::from_path("actor/b");
    let mut first = OfpaCatalog::new(2);
    let mut second = OfpaCatalog::new(2);
    for id in [a, b] {
        first.register(id, coord()).expect("register");
        first.mark_dirty(id).expect("dirty");
    }
    for id in [b, a] {
        second.register(id, coord()).expect("register");
        second.mark_dirty(id).expect("dirty");
    }
    assert_eq!(first.plan(), second.plan());
    let old = first.plan();
    first.mark_dirty(a).expect("new edit");
    let operation = old
        .operations()
        .iter()
        .find(|op| op.identity().actor() == a)
        .expect("actor operation");
    assert!(matches!(
        first.commit(operation),
        Err(OfpaError::StaleRevision { .. })
    ));
    assert!(first.actor(a).expect("actor").is_dirty());
}

#[test]
fn hlod_bands_switch_hide_before_show_and_handle_extreme_coordinates() {
    let mut planner = HlodPlanner::new(2, 1);
    planner
        .add_layer(HlodLayer::new(0, 0, 8).expect("near"))
        .expect("near");
    planner
        .add_layer(HlodLayer::new(1, 8, u32::MAX).expect("far"))
        .expect("far");
    planner
        .add_cluster(
            HlodCluster::new(
                StableGuid::from_path("cluster/a"),
                coord(),
                [coord()],
                [(0, "near".to_owned()), (1, "far".to_owned())],
            )
            .expect("cluster"),
        )
        .expect("register cluster");
    assert_eq!(
        planner
            .plan(&HlodView::new(CellCoord::new(8, 0, 0), []))
            .operations()[0]
            .level(),
        1
    );
    let switch = planner.plan(&HlodView::new(CellCoord::new(7, 0, 0), []));
    assert_eq!(switch.operations().len(), 2);
    assert!(!switch.operations()[0].is_show());
    assert!(switch.operations()[1].is_show());
    assert!(planner
        .plan(&HlodView::new(CellCoord::new(7, 0, 0), []))
        .operations()
        .is_empty());
    let extreme = planner.plan(&HlodView::new(CellCoord::new(i32::MIN, i32::MAX, 0), []));
    assert_eq!(extreme.operations().len(), 2);
}

#[test]
fn leases_expire_exactly_and_old_tokens_cannot_release_reacquired_locks() {
    let mut locks = CellLockRegistry::new();
    let owner = StableGuid::from_path("editor/a");
    let old = locks.acquire(coord(), owner, 2, 2).expect("lease");
    assert!(!locks.is_current(&old, 1));
    assert!(locks.is_current(&old, 3));
    assert!(!locks.is_current(&old, 4));
    let new = locks.acquire(coord(), owner, 4, 3).expect("reacquire");
    assert_eq!(locks.release(&old), Err(CellLockError::InvalidToken));
    assert!(locks.is_current(&new, 4));
    assert_eq!(
        locks.acquire(CellCoord::new(1, 0, 0), owner, u64::MAX, 1),
        Err(CellLockError::ExpiryOverflow)
    );
}
