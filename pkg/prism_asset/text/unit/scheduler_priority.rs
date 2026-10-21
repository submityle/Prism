//! Priority dispatch, inheritance, cancellation, and atomic soft-batch contracts.

use super::*;
use crate::{AssetLoader, AssetMeta, AssetReader, LoadContext, LoadError, ReadError, StableGuid};
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::Duration;

#[derive(Debug)]
struct Value;

impl Asset for Value {
    const TYPE_NAME: &'static str = "priority_tests::Value";
}

#[derive(Debug)]
struct Other;

impl Asset for Other {
    const TYPE_NAME: &'static str = "priority_tests::Other";
}

struct RecordingLoader(Arc<Mutex<Vec<u8>>>);

impl AssetLoader for RecordingLoader {
    type Asset = Value;

    fn extensions(&self) -> &'static [&'static str] {
        &["value"]
    }

    fn load(&self, bytes: &[u8], _: &mut LoadContext) -> Result<Value, LoadError> {
        self.0.lock().unwrap().push(bytes[0]);
        Ok(Value)
    }
}

struct RecordingSource(Arc<Mutex<Vec<String>>>);

impl AssetReader for RecordingSource {
    fn read(&self, path: &str) -> Result<Vec<u8>, ReadError> {
        self.0.lock().unwrap().push(path.to_string());
        Ok(vec![path.as_bytes()[0]])
    }

    fn metadata(&self, _: &str) -> Result<AssetMeta, ReadError> {
        Ok(AssetMeta {
            size: 1,
            modified_ms: None,
            is_dir: false,
        })
    }
}

type Fixture = (LoadScheduler, Arc<Mutex<Vec<String>>>, Arc<Mutex<Vec<u8>>>);

fn setup(limit: usize, io: TaskPool, cpu: TaskPool) -> Fixture {
    let reads = Arc::new(Mutex::new(Vec::new()));
    let decoded = Arc::new(Mutex::new(Vec::new()));
    let server = Arc::new(AssetServer::new());
    server.register_loader(RecordingLoader(Arc::clone(&decoded)));
    server.mount("mem", Arc::new(RecordingSource(Arc::clone(&reads))), 0);
    (
        LoadScheduler::with_io_cpu_pools(server, io, cpu, limit),
        reads,
        decoded,
    )
}

fn flight(priority: Priority) -> Arc<Flight> {
    Arc::new(Flight {
        notification: Mutex::new(FlightNotification {
            result: None,
            waiters: Vec::new(),
        }),
        priority: FlightPriority::new(priority),
        requested_type: None,
        cancellation: RangeAssemblyCancellation::new(),
        lifecycle: Mutex::new(FlightLifecycle::default()),
    })
}

fn receive<T>(receiver: &Receiver<T>) -> T {
    receiver
        .recv_timeout(Duration::from_secs(10))
        .expect("bounded gate timed out")
}

/// Parks the sole I/O worker while callers enqueue work; dropping the sender
/// releases it even if a test assertion unwinds.
fn hold_worker(pool: &TaskPool) -> Sender<()> {
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    pool.spawn_future_with_priority(
        Arc::new(PriorityCell::new(Priority::Critical)),
        async move {
            entered_tx.send(()).unwrap();
            // A disconnected gate also releases the worker during test unwinding.
            let _release = release_rx.recv_timeout(Duration::from_secs(10));
        },
    )
    .detach();
    receive(&entered_rx);
    release_tx
}

/// Every semantic level maps to a distinct ordered task lane.
#[test]
fn semantic_levels_preserve_urgency_order() {
    let levels = [
        LoadPriority::Background,
        LoadPriority::Prefetch,
        LoadPriority::Streaming,
        LoadPriority::Immediate,
    ];
    let mapped = levels.map(Priority::from);
    assert_eq!(
        mapped,
        [
            Priority::Background,
            Priority::Low,
            Priority::High,
            Priority::Critical
        ]
    );
    assert!(mapped.windows(2).all(|pair| pair[0] < pair[1]));
}

/// Zero-worker dispatch is deterministic: promoting an already-enqueued
/// background flight must change decode order before its first poll.
#[test]
fn cpu_initial_dispatch_obeys_live_promotion_and_fifo_ties() {
    let cpu = TaskPool::with_threads(0);
    let (scheduler, reads, decoded) = setup(4, TaskPool::with_threads(0), cpu.clone());
    let first = scheduler
        .schedule("a.value", LoadPriority::Background)
        .unwrap();
    let second = scheduler
        .schedule("b.value", LoadPriority::Streaming)
        .unwrap();
    let third = scheduler
        .schedule("c.value", LoadPriority::Immediate)
        .unwrap();
    let fourth = scheduler
        .schedule("d.value", LoadPriority::Prefetch)
        .unwrap();
    first.reprioritize(LoadPriority::Immediate);
    first.reprioritize(LoadPriority::Background);
    assert_eq!(first.effective_priority(), Priority::Critical);
    cpu.block_on(async {
        first.await.unwrap();
        second.await.unwrap();
        third.await.unwrap();
        fourth.await.unwrap();
    });
    assert_eq!(*decoded.lock().unwrap(), b"acbd");
    assert_eq!(
        *reads.lock().unwrap(),
        ["a.value", "c.value", "b.value", "d.value"]
    );
    assert_eq!(scheduler.in_flight(), 0);
}

/// A join at full capacity shares the live cell and read; cancelling one
/// consumer neither cancels nor later promotes another consumer's work.
#[test]
fn duplicate_priority_and_cancellation_are_consumer_safe() {
    let cpu = TaskPool::with_threads(0);
    let (scheduler, reads, _) = setup(1, TaskPool::with_threads(0), cpu.clone());
    let first = scheduler
        .schedule("a.value", LoadPriority::Background)
        .unwrap();
    let duplicate = scheduler
        .schedule("a.value", LoadPriority::Streaming)
        .unwrap();
    assert_eq!(first.effective_priority(), Priority::High);
    first.cancel();
    first.reprioritize(LoadPriority::Immediate);
    assert_eq!(duplicate.effective_priority(), Priority::High);
    assert_eq!(cpu.block_on(first), Err(LoadFailure::Cancelled));
    let handle = cpu.block_on(duplicate).unwrap();
    assert!(scheduler.server.is_loaded(handle.id()));
    assert_eq!(reads.lock().unwrap().len(), 1);
}

/// A held I/O worker distinguishes queue ordering from reading priority after
/// start. Joining a queued low-priority read promotes it ahead of streaming.
#[test]
fn io_dispatch_uses_live_priority_not_spawn_order() {
    let io = TaskPool::with_threads(1);
    if io.is_single_threaded() {
        return;
    }
    let gate = hold_worker(&io);
    let cpu = TaskPool::with_threads(0);
    let (scheduler, reads, _) = setup(3, io, cpu.clone());
    let low = scheduler
        .schedule("a.value", LoadPriority::Background)
        .unwrap();
    let middle = scheduler
        .schedule("b.value", LoadPriority::Streaming)
        .unwrap();
    let high = scheduler
        .schedule("c.value", LoadPriority::Immediate)
        .unwrap();
    // All CPU flights suspend at I/O; the empty target does not await them.
    cpu.block_on(async {});
    low.reprioritize(LoadPriority::Immediate);
    drop(gate);
    cpu.block_on(async {
        low.await.unwrap();
        middle.await.unwrap();
        high.await.unwrap();
    });
    // c's I/O entered the queue first at Critical. a was submitted last but
    // promoted before dispatch, so it still precedes b's Streaming read.
    assert_eq!(*reads.lock().unwrap(), ["c.value", "a.value", "b.value"]);
}

/// A completed read queues a CPU continuation at Background. Promoting its
/// ticket must reorder that already-woken continuation before High CPU work.
#[test]
fn cpu_wakeup_continuation_keeps_live_flight_priority() {
    let io = TaskPool::with_threads(1);
    if io.is_single_threaded() {
        return;
    }
    let gate = hold_worker(&io);
    let cpu = TaskPool::with_threads(0);
    let (scheduler, _, decoded) = setup(1, io.clone(), cpu.clone());
    let ticket = scheduler
        .schedule("a.value", LoadPriority::Background)
        .unwrap();
    cpu.block_on(async {});
    let (finished_tx, finished_rx) = mpsc::channel();
    // Same-priority FIFO on the sole worker places this after the read, whose
    // completion wakes and queues the scheduler's CPU continuation first.
    io.spawn_future_with_priority(
        Arc::new(PriorityCell::new(Priority::Background)),
        async move {
            finished_tx.send(()).unwrap();
        },
    )
    .detach();
    drop(gate);
    receive(&finished_rx);
    let other_decoded = decoded.clone();
    let unrelated =
        cpu.spawn_future_with_priority(Arc::new(PriorityCell::new(Priority::High)), async move {
            other_decoded.lock().unwrap().push(b'x');
        });
    ticket.reprioritize(LoadPriority::Immediate);
    cpu.block_on(ticket).unwrap();
    cpu.block_on(unrelated);
    assert_eq!(*decoded.lock().unwrap(), b"ax");
}

/// Dependency inheritance is transitive, works after edge insertion, never
/// demotes, and traverses cycles without leaking or recursing unboundedly.
#[test]
fn inheritance_reaches_existing_dependency_cells_and_weak_links_expire() {
    let root = FlightPriority::new(Priority::Background);
    let child = FlightPriority::new(Priority::Low);
    let leaf = FlightPriority::new(Priority::Background);
    root.inherit(&child);
    child.inherit(&leaf);
    leaf.inherit(&root);
    root.boost_to(Priority::Critical);
    assert_eq!(child.effective_priority(), Priority::Critical);
    assert_eq!(leaf.effective_priority(), Priority::Critical);
    let weak = Arc::downgrade(&child.0);
    drop(child);
    assert!(weak.upgrade().is_none());
    root.boost_to(Priority::Low);
    assert_eq!(root.effective_priority(), Priority::Critical);
}

/// Attaching an already critical parent must immediately promote a new child.
#[test]
fn inheritance_at_edge_insertion_promotes_new_dependency() {
    let parent = FlightPriority::new(Priority::Critical);
    let child = FlightPriority::new(Priority::Background);
    parent.inherit(&child);
    assert_eq!(child.effective_priority(), Priority::Critical);
}

struct ParentLoader(Arc<Mutex<Vec<u8>>>);

impl AssetLoader for ParentLoader {
    type Asset = Value;

    fn extensions(&self) -> &'static [&'static str] {
        &["parent"]
    }

    fn load(&self, _: &[u8], context: &mut LoadContext) -> Result<Value, LoadError> {
        self.0.lock().unwrap().push(b'p');
        context.add_dependency("c.value".into());
        Ok(Value)
    }
}

/// Drive the production parent assembly once before pumping the CPU pool, so
/// its child is genuinely queued when a later consumer promotes the parent.
#[test]
fn parent_promotion_reorders_its_already_queued_dependency() {
    let cpu = TaskPool::with_threads(0);
    let (scheduler, _, decoded) = setup(2, TaskPool::with_threads(0), cpu.clone());
    scheduler
        .server
        .register_loader(ParentLoader(decoded.clone()));
    let parent = flight(Priority::Background);
    let key = FlightKey::Load("p.parent".into());
    {
        let mut state = scheduler.state.lock().unwrap();
        state.in_flight.insert(key.clone(), parent.clone());
        state.running.insert(key.clone(), parent.clone());
    }
    let mut assembly = Box::pin(execute_async_load(
        scheduler.clone(),
        scheduler.server.clone(),
        "p.parent".into(),
        parent.clone(),
        None,
        Priority::Background,
        BTreeSet::new(),
    ));
    let waker = Waker::from(Arc::new(NoopWake));
    assert!(assembly
        .as_mut()
        .poll(&mut Context::from_waker(&waker))
        .is_pending());
    let unrelated = scheduler
        .schedule("u.value", LoadPriority::Streaming)
        .unwrap();
    let parent_ticket = LoadTicket::new(parent.clone());
    parent_ticket.reprioritize(LoadPriority::Immediate);
    {
        let state = scheduler.state.lock().unwrap();
        let child = state
            .in_flight
            .get(&FlightKey::Load("c.value".into()))
            .unwrap();
        assert_eq!(child.priority.effective_priority(), Priority::Critical);
    }
    let result = cpu.block_on(assembly);
    assert!(result.is_ok());
    scheduler.finish_flight(key, parent, result);
    drop(parent_ticket);
    cpu.block_on(unrelated).unwrap();
    assert_eq!(*decoded.lock().unwrap(), b"pcu");
}

struct LabelLoader;

impl AssetLoader for LabelLoader {
    type Asset = Value;

    fn extensions(&self) -> &'static [&'static str] {
        &["scene"]
    }

    fn load(&self, _: &[u8], context: &mut LoadContext) -> Result<Value, LoadError> {
        context.add_labeled_asset("label", Other);
        Ok(Value)
    }
}

/// Catalog labels may be registered before emission, but the runtime cannot
/// return a different emitted type as a successful typed batch result.
#[test]
fn soft_batch_checks_emitted_label_type() {
    let cpu = TaskPool::with_threads(0);
    let (scheduler, _, _) = setup(1, TaskPool::with_threads(0), cpu.clone());
    scheduler.server.register_loader(LabelLoader);
    let soft = scheduler
        .server
        .register_soft_path::<Value>("a.scene#label")
        .unwrap();
    let mut tickets = scheduler
        .schedule_soft_batch(
            &[soft],
            LoadPriority::Immediate,
            SoftLoadBudget {
                max_requests: 1,
                max_unique_assets: 1,
            },
        )
        .unwrap();
    assert!(matches!(cpu.block_on(tickets.remove(0)),
        Err(LoadFailure::AssemblyFailed(message)) if message.contains("type")));
}

struct NoopWake;
impl std::task::Wake for NoopWake {
    fn wake(self: Arc<Self>) {}
}

/// Admission must compare live cells rather than captured priorities; a slot
/// handed to a waiter remains reserved against queued dependency admissions.
#[test]
fn admission_live_priority_and_reserved_slots_are_respected() {
    let (scheduler, _, _) = setup(1, TaskPool::with_threads(0), TaskPool::with_threads(0));
    let blocker = flight(Priority::Normal);
    let blocker_key = FlightKey::Load("block.value".into());
    scheduler
        .state
        .lock()
        .unwrap()
        .running
        .insert(blocker_key.clone(), blocker.clone());
    let background = flight(Priority::Background);
    let streaming = flight(Priority::High);
    let mut low_wait = Box::pin(scheduler.acquire_execution_slot(
        FlightKey::Load("a.value".into()),
        background.clone(),
        Priority::Background,
    ));
    let mut high_wait = Box::pin(scheduler.acquire_execution_slot(
        FlightKey::Load("b.value".into()),
        streaming,
        Priority::High,
    ));
    let waker = Waker::from(Arc::new(NoopWake));
    let mut cx = Context::from_waker(&waker);
    assert!(low_wait.as_mut().poll(&mut cx).is_pending());
    assert!(high_wait.as_mut().poll(&mut cx).is_pending());
    background.priority.boost_to(Priority::Critical);
    scheduler.release_execution_slot(&blocker_key, &blocker);
    {
        let mut state = scheduler.state.lock().unwrap();
        state.queued.push_back(QueuedFlight {
            key: FlightKey::Load("queued.value".into()),
            flight: flight(Priority::Critical),
            priority: Priority::Critical,
            ancestors: BTreeSet::new(),
            reload: false,
        });
        assert!(scheduler.admit_queued_locked(&mut state).is_empty());
        state.queued.clear();
    }
    assert!(high_wait.as_mut().poll(&mut cx).is_pending());
    assert!(low_wait.as_mut().poll(&mut cx).is_ready());
}

/// Registering catalog paths and resolving soft identities must be zero-I/O;
/// explicit null/type/unknown/loader errors replace silent batch rejection.
#[test]
fn soft_catalog_resolution_reports_identity_and_type_errors_without_io() {
    let (scheduler, reads, _) = setup(1, TaskPool::with_threads(0), TaskPool::with_threads(0));
    let server = &scheduler.server;
    let known = server.register_soft_path::<Value>("a.value").unwrap();
    assert_eq!(
        server.resolve_soft(&known).unwrap(),
        AssetPath::from("a.value")
    );
    assert_eq!(
        server.resolve_soft(&SoftHandle::<Value>::null(AssetTypeId::of::<Value>())),
        Err(SoftLoadError::Null)
    );
    let mistyped = SoftHandle::<Value>::new(known.guid(), AssetTypeId::of::<Other>());
    assert!(matches!(
        server.resolve_soft(&mistyped),
        Err(SoftLoadError::TypeMismatch { .. })
    ));
    let unknown = SoftHandle::<Value>::from_path("unknown.value");
    assert_eq!(
        server.resolve_soft(&unknown),
        Err(SoftLoadError::UnknownGuid {
            guid: unknown.guid()
        })
    );
    assert!(matches!(
        server.register_soft_path::<Value>("a.unsupported"),
        Err(SoftLoadError::NoLoader { .. })
    ));
    assert!(reads.lock().unwrap().is_empty());
    assert_eq!(server.total_assets(), 0);
}

/// Invalid input in the tail must not admit/promote the valid prefix.
#[test]
fn soft_batch_validation_is_atomic_and_includes_input_index() {
    let cpu = TaskPool::with_threads(0);
    let (scheduler, reads, _) = setup(2, TaskPool::with_threads(0), cpu.clone());
    let known = scheduler
        .server
        .register_soft_path::<Value>("a.value")
        .unwrap();
    let existing = scheduler
        .schedule("a.value", LoadPriority::Background)
        .unwrap();
    let invalid = SoftHandle::<Value>::null(AssetTypeId::of::<Value>());
    let error = scheduler
        .schedule_soft_batch(
            &[known, invalid],
            LoadPriority::Immediate,
            SoftLoadBudget {
                max_requests: 2,
                max_unique_assets: 2,
            },
        )
        .err()
        .unwrap();
    assert_eq!(
        error,
        SoftBatchError::InvalidHandle {
            index: 1,
            reason: SoftLoadError::Null
        }
    );
    assert!(error.to_string().contains("reference 1"));
    assert_eq!(existing.effective_priority(), Priority::Background);
    assert_eq!(scheduler.in_flight(), 1);
    assert!(reads.lock().unwrap().is_empty());
    cpu.block_on(existing).unwrap();
}

/// A typed batch cannot silently join work whose emitted type is unresolved;
/// it must refuse before admitting an earlier valid root or promoting work.
#[test]
fn soft_batch_rejects_unresolved_shared_type_without_partial_admission() {
    let (scheduler, reads, _) = setup(2, TaskPool::with_threads(0), TaskPool::with_threads(0));
    let a = scheduler
        .server
        .register_soft_path::<Value>("a.value")
        .unwrap();
    let b = scheduler
        .server
        .register_soft_path::<Value>("b.value")
        .unwrap();
    let unresolved = flight(Priority::Background);
    let key = FlightKey::Load("b.value".into());
    {
        let mut state = scheduler.state.lock().unwrap();
        state.in_flight.insert(key.clone(), unresolved.clone());
        state.running.insert(key, unresolved.clone());
    }
    let error = scheduler
        .schedule_soft_batch(
            &[a, b],
            LoadPriority::Immediate,
            SoftLoadBudget {
                max_requests: 2,
                max_unique_assets: 2,
            },
        )
        .err()
        .unwrap();
    assert_eq!(
        error,
        SoftBatchError::ConflictingFlight {
            index: 1,
            path: "b.value".into()
        }
    );
    assert_eq!(
        unresolved.priority.effective_priority(),
        Priority::Background
    );
    assert_eq!(scheduler.in_flight(), 1);
    assert!(reads.lock().unwrap().is_empty());
}

/// Request budgets charge duplicates, distinct-root budgets do not; capacity
/// refusal admits no prefix and performs no source reads.
#[test]
fn soft_batch_budget_and_capacity_refusals_are_exact_and_side_effect_free() {
    let cpu = TaskPool::with_threads(0);
    let (scheduler, reads, _) = setup(1, TaskPool::with_threads(0), cpu.clone());
    let a = scheduler
        .server
        .register_soft_path::<Value>("a.value")
        .unwrap();
    let b = scheduler
        .server
        .register_soft_path::<Value>("b.value")
        .unwrap();
    let error = scheduler
        .schedule_soft_batch(
            &[a, a],
            LoadPriority::Streaming,
            SoftLoadBudget {
                max_requests: 1,
                max_unique_assets: 2,
            },
        )
        .err()
        .unwrap();
    assert_eq!(
        error,
        SoftBatchError::RequestBudgetExceeded {
            requested: 2,
            limit: 1
        }
    );
    let error = scheduler
        .schedule_soft_batch(
            &[a, b],
            LoadPriority::Streaming,
            SoftLoadBudget {
                max_requests: 2,
                max_unique_assets: 1,
            },
        )
        .err()
        .unwrap();
    assert_eq!(
        error,
        SoftBatchError::UniqueAssetBudgetExceeded {
            requested: 2,
            limit: 1
        }
    );
    let error = scheduler
        .schedule_soft_batch(
            &[a, b],
            LoadPriority::Streaming,
            SoftLoadBudget {
                max_requests: 2,
                max_unique_assets: 2,
            },
        )
        .err()
        .unwrap();
    assert_eq!(
        error,
        SoftBatchError::AtCapacity {
            required: 2,
            available: 1
        }
    );
    assert!(reads.lock().unwrap().is_empty());
    assert_eq!(scheduler.in_flight(), 0);
    let tickets = scheduler
        .schedule_soft_batch(
            &[a, a],
            LoadPriority::Streaming,
            SoftLoadBudget {
                max_requests: 2,
                max_unique_assets: 1,
            },
        )
        .unwrap();
    assert_eq!(scheduler.in_flight(), 1);
    let mut tickets = tickets.into_iter();
    let cancelled = tickets.next().unwrap();
    let survivor = tickets.next().unwrap();
    cancelled.cancel();
    assert_eq!(cpu.block_on(cancelled), Err(LoadFailure::Cancelled));
    cpu.block_on(survivor).unwrap();
    assert_eq!(*reads.lock().unwrap(), ["a.value"]);
}

/// An empty batch is valid with zero budgets and cannot allocate or read.
#[test]
fn empty_soft_batch_is_a_noop_and_existing_duplicates_fit_full_scheduler() {
    let cpu = TaskPool::with_threads(0);
    let (scheduler, reads, _) = setup(1, TaskPool::with_threads(0), cpu.clone());
    assert!(scheduler
        .schedule_soft_batch::<Value>(
            &[],
            LoadPriority::Background,
            SoftLoadBudget {
                max_requests: 0,
                max_unique_assets: 0
            }
        )
        .unwrap()
        .is_empty());
    let a = scheduler
        .server
        .register_soft_path::<Value>("a.value")
        .unwrap();
    let first = scheduler
        .schedule("a.value", LoadPriority::Background)
        .unwrap();
    let mut joined = scheduler
        .schedule_soft_batch(
            &[a],
            LoadPriority::Immediate,
            SoftLoadBudget {
                max_requests: 1,
                max_unique_assets: 1,
            },
        )
        .unwrap();
    assert_eq!(first.effective_priority(), Priority::Critical);
    let first_handle = cpu.block_on(first).unwrap();
    assert_eq!(
        cpu.block_on(joined.remove(0)).unwrap().id(),
        first_handle.id()
    );
    assert_eq!(reads.lock().unwrap().len(), 1);
}

/// Normalized GUIDs cannot silently redirect an existing catalog registration.
#[test]
fn soft_catalog_collision_does_not_replace_existing_mapping() {
    let (scheduler, _, _) = setup(1, TaskPool::with_threads(0), TaskPool::with_threads(0));
    let first = scheduler
        .server
        .register_soft_path::<Value>("a.value")
        .unwrap();
    assert_eq!(
        StableGuid::from_path("a.value"),
        StableGuid::from_path("./a.value")
    );
    assert!(matches!(
        scheduler.server.register_soft_path::<Value>("./a.value"),
        Err(SoftLoadError::CatalogCollision { .. })
    ));
    assert_eq!(
        scheduler.server.resolve_soft(&first).unwrap(),
        AssetPath::from("a.value")
    );
}

/// Shared reload reads must use live inheritance too; promoting either reload
/// promotes the queued read without issuing a second source operation.
#[test]
fn reload_read_dedup_and_io_promotion_share_the_same_live_cell() {
    let io = TaskPool::with_threads(1);
    if io.is_single_threaded() {
        return;
    }
    let gate = hold_worker(&io);
    let (scheduler, reads, _) = setup(1, io, TaskPool::with_threads(0));
    let mut sources = AssetSources::new();
    sources.mount("mem", Arc::new(RecordingSource(Arc::clone(&reads))), 0);
    let low = FlightPriority::new(Priority::Background);
    let high = FlightPriority::new(Priority::High);
    let mut loaders = AssetLoaders::new();
    loaders.register(RecordingLoader(Arc::new(Mutex::new(Vec::new()))));
    let path = AssetPath::from("a.value");
    let first = scheduler.reload_read(
        7,
        &sources,
        &loaders,
        &path,
        AssetTypeId::of::<Value>(),
        &low,
        false,
    );
    let unrelated = scheduler.reload_read(
        7,
        &sources,
        &loaders,
        &AssetPath::from("b.value"),
        AssetTypeId::of::<Value>(),
        &high,
        false,
    );
    let joining = FlightPriority::new(Priority::Background);
    let duplicate = scheduler.reload_read(
        7,
        &sources,
        &loaders,
        &path,
        AssetTypeId::of::<Value>(),
        &joining,
        false,
    );
    joining.boost_to(Priority::Critical);
    assert!(Arc::ptr_eq(&first.flight, &duplicate.flight));
    assert_eq!(
        first.flight.priority.effective_priority(),
        Priority::Critical
    );
    drop(gate);
    let drive = TaskPool::with_threads(0);
    let bytes = drive.block_on(first).unwrap();
    assert_eq!(&*drive.block_on(duplicate).unwrap(), &*bytes);
    drive.block_on(unrelated).unwrap();
    assert_eq!(*reads.lock().unwrap(), ["a.value", "b.value"]);
}
