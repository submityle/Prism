//! Executable provider, lifecycle, lock, OFPA, and HLOD contract tests.

use std::task::{Context, Poll, Waker};

use prism_asset::StableGuid;
use prism_ecs::partition::{cell::CellCoord, data_layer::DataLayerId};
use prism_editor_world::{
    CellIoProvider, CellLoadFuture, CellLoadRequest, CellResidency, CellSaveFuture,
    CellSaveRequest, DataLayer, EditorWorldData, EditorWorldDataError, EditorWorldLimits,
    HlodCluster, HlodLayer, HlodView, IoBudget, IoCompletion, IoSubmitError, OfpaError,
    ResidencyDemand, WorldIoDriver,
};

fn limits() -> EditorWorldLimits {
    EditorWorldLimits::new(32, 32, 32, 4, 32).expect("limits")
}

fn cell(x: i32, y: i32, z: i32) -> CellCoord {
    CellCoord::new(x, y, z)
}

#[derive(Clone, Copy)]
struct Provider {
    fail_load: bool,
    fail_save: bool,
    oversized_load: bool,
}

impl CellIoProvider for Provider {
    type Error = &'static str;

    fn load_cell(
        &self,
        _request: CellLoadRequest,
    ) -> Result<CellLoadFuture<Self::Error>, Self::Error> {
        if self.fail_load {
            return Ok(Box::pin(async { Err("read failed") }));
        }
        let oversized = self.oversized_load;
        Ok(Box::pin(async move {
            if oversized {
                Ok(vec![7; 128])
            } else {
                Ok(b"cell-payload".to_vec())
            }
        }))
    }

    fn save_cell(
        &self,
        _request: CellSaveRequest,
        _payload: Vec<u8>,
    ) -> Result<CellSaveFuture<Self::Error>, Self::Error> {
        if self.fail_save {
            Ok(Box::pin(async { Err("write failed") }))
        } else {
            Ok(Box::pin(async { Ok(()) }))
        }
    }
}

fn context() -> Context<'static> {
    Context::from_waker(Waker::noop())
}

fn setup_world() -> (EditorWorldData, CellCoord) {
    let mut world =
        EditorWorldData::new(StableGuid::from_path("world/editor"), limits()).expect("world");
    world
        .add_layer(DataLayer::new(DataLayerId::new(1), "base").expect("layer"))
        .expect("layer registration");
    let coord = cell(3, -2, 1);
    world.add_cell(coord, [DataLayerId::new(1)]).expect("cell");
    (world, coord)
}

fn load_with_driver(
    world: &mut EditorWorldData,
    coord: CellCoord,
    provider: Provider,
) -> (WorldIoDriver<Provider>, prism_editor_world::IoRequestId) {
    let mut demand = ResidencyDemand::new();
    demand.require(coord);
    let (_, tokens) = world.begin_residency(&demand, 0).expect("begin load");
    let token = tokens[0];
    let request = world.load_request(token).expect("load request");
    let budget = IoBudget::new(2, 32).expect("budget");
    let mut driver = WorldIoDriver::new(provider, budget);
    let request_id = driver.submit_load(request).expect("submit load");
    (driver, request_id)
}

#[test]
fn provider_completion_installs_payload_and_marks_cell_loaded() {
    let (mut world, coord) = setup_world();
    let (mut driver, _) = load_with_driver(
        &mut world,
        coord,
        Provider {
            fail_load: false,
            fail_save: false,
            oversized_load: false,
        },
    );
    let completion = match driver.poll(&mut context()) {
        Poll::Ready(Some(completion)) => completion,
        other => panic!("unexpected completion: {other:?}"),
    };
    assert!(matches!(completion, IoCompletion::Loaded { .. }));
    world.apply_load_completion(completion).expect("apply load");
    assert_eq!(
        world.model().cell(coord).expect("cell").residency(),
        CellResidency::Loaded
    );
    assert_eq!(world.payload(coord), Some(&b"cell-payload"[..]));
}

#[test]
fn provider_failure_and_cancellation_return_to_unloaded_without_payload() {
    let (mut failed, coord) = setup_world();
    let (mut driver, _) = load_with_driver(
        &mut failed,
        coord,
        Provider {
            fail_load: true,
            fail_save: false,
            oversized_load: false,
        },
    );
    let completion = match driver.poll(&mut context()) {
        Poll::Ready(Some(completion)) => completion,
        other => panic!("unexpected completion: {other:?}"),
    };
    assert!(matches!(completion, IoCompletion::Failed { .. }));
    failed
        .apply_load_completion(completion)
        .expect("apply failure");
    assert_eq!(
        failed.model().cell(coord).expect("cell").residency(),
        CellResidency::Unloaded
    );
    assert!(failed.payload(coord).is_none());

    let (mut cancelled, coord) = setup_world();
    let (mut driver, request_id) = load_with_driver(
        &mut cancelled,
        coord,
        Provider {
            fail_load: false,
            fail_save: false,
            oversized_load: false,
        },
    );
    assert!(driver.cancel(request_id));
    let completion = match driver.poll(&mut context()) {
        Poll::Ready(Some(completion)) => completion,
        other => panic!("unexpected cancellation: {other:?}"),
    };
    assert!(matches!(completion, IoCompletion::Cancelled { .. }));
    cancelled
        .apply_load_completion(completion)
        .expect("apply cancellation");
    assert_eq!(
        cancelled.model().cell(coord).expect("cell").residency(),
        CellResidency::Unloaded
    );
}

#[test]
fn oversized_provider_result_is_failed_and_restores_retryable_state() {
    let (mut world, coord) = setup_world();
    let (mut driver, _) = load_with_driver(
        &mut world,
        coord,
        Provider {
            fail_load: false,
            fail_save: false,
            oversized_load: true,
        },
    );
    let completion = match driver.poll(&mut context()) {
        Poll::Ready(Some(completion)) => completion,
        other => panic!("unexpected completion: {other:?}"),
    };
    assert!(matches!(
        completion,
        IoCompletion::Failed {
            error: prism_editor_world::IoFailure::PayloadTooLarge {
                limit: 32,
                actual: 128
            },
            ..
        }
    ));
    assert!(!world
        .apply_load_completion(completion)
        .expect("apply size failure"));
    assert_eq!(
        world.model().cell(coord).expect("cell").residency(),
        CellResidency::Unloaded
    );
    assert_eq!(driver.reserved_bytes(), 0);
    assert!(world.payload(coord).is_none());
}

#[test]
fn driver_enforces_request_and_byte_bounds_and_releases_reservations() {
    let (mut world, coord) = setup_world();
    let mut demand = ResidencyDemand::new();
    demand.require(coord);
    let (_, tokens) = world.begin_residency(&demand, 0).expect("begin");
    let request = world.load_request(tokens[0]).expect("request");
    let mut driver = WorldIoDriver::new(
        Provider {
            fail_load: false,
            fail_save: false,
            oversized_load: false,
        },
        IoBudget::new(1, 32).expect("budget"),
    );
    driver.submit_load(request.clone()).expect("first request");
    assert!(matches!(
        driver.submit_load(request),
        Err(IoSubmitError::Backpressure { max_in_flight: 1 })
    ));
    let completion = match driver.poll(&mut context()) {
        Poll::Ready(Some(completion)) => completion,
        other => panic!("unexpected completion: {other:?}"),
    };
    assert_eq!(driver.reserved_bytes(), 0);
    world.apply_load_completion(completion).expect("apply load");

    let (mut bounded_world, bounded_coord) = setup_world();
    let mut bounded_demand = ResidencyDemand::new();
    bounded_demand.require(bounded_coord);
    let (_, bounded_tokens) = bounded_world
        .begin_residency(&bounded_demand, 0)
        .expect("bounded begin");
    let bounded_request = bounded_world
        .load_request(bounded_tokens[0])
        .expect("bounded request");
    let mut bounded_driver = WorldIoDriver::new(
        Provider {
            fail_load: false,
            fail_save: false,
            oversized_load: false,
        },
        IoBudget::new(1, 8).expect("small budget"),
    );
    assert!(matches!(
        bounded_driver.submit_load(bounded_request),
        Err(IoSubmitError::ByteBudgetExceeded {
            requested: 32,
            max_bytes: 8
        })
    ));
    assert!(matches!(
        bounded_world
            .model()
            .cell(bounded_coord)
            .expect("cell")
            .residency(),
        CellResidency::Loading(_)
    ));
}

#[test]
fn lock_protects_dirty_payload_and_save_commit_is_revision_safe() {
    let (mut world, coord) = setup_world();
    let owner = StableGuid::from_path("actor/editor");
    let lock = world.acquire_lock(coord, owner, 10, 5).expect("lock");
    let mut demand = ResidencyDemand::new();
    demand.require(coord);
    let (_, tokens) = world.begin_residency(&demand, 10).expect("begin");
    world
        .complete_load(tokens[0], b"old".to_vec())
        .expect("load");
    world
        .replace_payload(coord, &lock, 11, b"new".to_vec())
        .expect("replace");
    let pending = world.prepare_save(coord, &lock, 12).expect("save snapshot");
    let mut driver = WorldIoDriver::new(
        Provider {
            fail_load: false,
            fail_save: true,
            oversized_load: false,
        },
        IoBudget::new(1, 32).expect("budget"),
    );
    driver
        .submit_save(pending.request().clone(), pending.payload().to_vec())
        .expect("save submission");
    let completion = match driver.poll(&mut context()) {
        Poll::Ready(Some(completion)) => completion,
        other => panic!("unexpected save completion: {other:?}"),
    };
    assert!(!world
        .apply_save_completion(&pending, completion, 12)
        .expect("failed save"));
    assert!(world.model().cell(coord).expect("cell").is_dirty());
    assert!(world.commit_save(&pending, 15).is_err());
}

#[test]
fn hlod_plan_is_deterministic_and_resident_detail_suppresses_proxy() {
    let id = StableGuid::from_path("hlod/cluster");
    let mut planner = prism_editor_world::HlodPlanner::new(2, 4);
    planner
        .add_layer(HlodLayer::new(0, 0, 8).expect("layer"))
        .expect("layer");
    planner
        .add_layer(HlodLayer::new(1, 8, 32).expect("layer"))
        .expect("layer");
    planner
        .add_cluster(
            HlodCluster::new(
                id,
                cell(0, 0, 0),
                [cell(0, 0, 0)],
                [(0, "proxy-near".to_owned()), (1, "proxy-far".to_owned())],
            )
            .expect("cluster"),
        )
        .expect("cluster");
    let shown = planner.plan(&HlodView::new(cell(0, 0, 0), []));
    assert_eq!(shown.operations().len(), 1);
    assert!(shown.operations()[0].is_show());
    let hidden = planner.plan(&HlodView::new(cell(0, 0, 0), [cell(0, 0, 0)]));
    assert_eq!(hidden.operations().len(), 1);
    assert!(!hidden.operations()[0].is_show());
}

#[test]
fn ofpa_capacity_and_revision_fail_closed() {
    let (mut world, coord) = setup_world();
    let actor = StableGuid::from_path("actor/one");
    world.register_actor(actor, coord).expect("actor");
    assert_eq!(
        world.register_actor(actor, coord),
        Err(EditorWorldDataError::Ofpa(OfpaError::DuplicateActor(actor)))
    );
    let other = StableGuid::from_path("actor/two");
    let mut small = EditorWorldData::new(
        StableGuid::from_path("world/small"),
        EditorWorldLimits::new(4, 16, 1, 1, 1).expect("limits"),
    )
    .expect("world");
    small.add_cell(coord, []).expect("cell");
    small.register_actor(actor, coord).expect("actor");
    assert_eq!(
        small.register_actor(other, coord),
        Err(EditorWorldDataError::Ofpa(OfpaError::CapacityExceeded {
            limit: 1
        }))
    );
    assert!(!world
        .plan_ofpa()
        .operations()
        .iter()
        .any(|op| op.identity().actor() == other));
}
