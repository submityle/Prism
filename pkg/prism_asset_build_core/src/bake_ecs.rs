//! Entity-owned runtime baking. The host supplies the service and semantic
//! revisions; the bridge never chooses game types, kernels, or cache policy.
//! Lifecycle hooks enqueue requests in a bounded queue. Admission failures are
//! terminal for that request instance; insert a new component to retry.

use crate::{BakeError, BakeHandle, BakeRequest, BakeResult, BakeService};
use prism_app::{App, Plugin, PreUpdate};
use prism_ecs::prelude::{Component, ComponentHooks, Entity, HookContext, Resource, World};
use std::collections::VecDeque;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// Caller-selected semantic revision. It need not be numeric or monotonic.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BakeRevision(pub u128);

struct Instance {
    cancelled: AtomicBool,
}

/// One immutable bake submission on an entity. Replacing this request creates
/// a new attempt, even with the same revision and identical request bytes.
/// Removing it or despawning cancels its subscription. Outputs survive request
/// replacement/removal until explicitly removed or the entity is despawned.
/// Install with [`BakeRuntime::insert_request`]. The actual ECS component is
/// private: this value deliberately does not implement Component, preventing
/// mutable queries or get_mut assignments from bypassing lifecycle hooks.
pub struct BakeRequestComponent {
    revision: BakeRevision,
    request: BakeRequest,
    instance: Arc<Instance>,
}

struct RequestSlot(BakeRequestComponent);
impl Component for RequestSlot {}
impl std::ops::Deref for RequestSlot {
    type Target = BakeRequestComponent;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl BakeRequestComponent {
    /// Creates an independently identified attempt with explicit revision.
    pub fn new(revision: BakeRevision, request: BakeRequest) -> Self {
        Self {
            revision,
            request,
            instance: Arc::new(Instance {
                cancelled: AtomicBool::new(false),
            }),
        }
    }

    /// Semantic revision supplied at construction.
    pub fn revision(&self) -> BakeRevision {
        self.revision
    }

    /// Immutable input contract. Replace the component to change it.
    pub fn request(&self) -> &BakeRequest {
        &self.request
    }

    /// Prevents ECS publication, including a result already delivered by the
    /// service but not yet drained. Worker cancellation follows next update.
    pub fn cancel(&self) {
        self.instance.cancelled.store(true, Ordering::Release);
    }
}

impl Drop for BakeRequestComponent {
    fn drop(&mut self) {
        self.cancel();
    }
}

/// One last-good, owned result. Failed, cancelled, or stale attempts leave it
/// untouched. There is no per-entity result history in the bridge.
#[derive(Debug)]
pub struct BakeOutput {
    /// Revision whose successful result was applied.
    pub revision: BakeRevision,
    /// Owned bytes and descriptor delivered by BakeService.
    pub result: BakeResult,
}
impl Component for BakeOutput {}

/// Observable admission or execution failure for one request.
#[derive(Clone, Debug)]
pub enum BakeEcsError {
    /// The bridge's bounded ticket slots are occupied, including cancelled
    /// kernels that have not finished. No hidden retry queue is created.
    AtCapacity {
        /// Configured maximum retained tickets.
        max_pending: usize,
    },
    /// The bounded admission queue was full when the component was inserted.
    QueueFull {
        /// Configured maximum requests waiting for frame admission.
        max_queued: usize,
    },
    /// Service admission, dispatch, validation, or kernel failure.
    Bake(BakeError),
}

impl fmt::Display for BakeEcsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AtCapacity { max_pending } => write!(f, "ECS bake tickets full ({max_pending})"),
            Self::QueueFull { max_queued } => {
                write!(f, "ECS bake admission queue full ({max_queued})")
            }
            Self::Bake(error) => error.fmt(f),
        }
    }
}
impl std::error::Error for BakeEcsError {}

/// Latest state for the request instance recorded in BakeStatus.
#[derive(Clone, Debug)]
pub enum BakeState {
    /// Waiting in the bounded frame admission queue.
    Queued,
    /// Accepted by the service; a result awaits nonblocking frame drain.
    Running,
    /// Result published to BakeOutput.
    Ready,
    /// Attempt failed; any previous BakeOutput remains valid.
    Failed(BakeEcsError),
    /// Attempt explicitly cancelled or its request component removed.
    Cancelled,
}

/// Latest attempt only. Queued requests have not consumed a service ticket yet.
pub struct BakeStatus {
    /// Semantic revision of this attempt.
    pub revision: BakeRevision,
    /// Current observable state.
    pub state: BakeState,
    instance: Arc<Instance>,
}
impl Component for BakeStatus {}

/// Explicit nonzero work bounds. These count attempts/results, not kernel time
/// or bytes; kernels run on the injected service's pool, never in App.update.
#[derive(Clone, Copy, Debug)]
pub struct BakeEcsConfig {
    /// Maximum requests waiting for frame admission. Overflow fails immediately.
    pub max_queued: usize,
    /// Maximum retained handles, including undrained and cancelled work.
    pub max_pending: usize,
    /// Maximum request admission attempts per update.
    pub max_admissions_per_frame: usize,
    /// Maximum results removed from tickets per update, including stale ones.
    pub max_completions_per_frame: usize,
}

/// Configuration cannot provide bounded, asynchronous frame consumption.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BakeEcsConfigError {
    /// Every admission, retained-ticket, and completion bound must be nonzero.
    ZeroBudget,
    /// The task pool was compiled to execute kernels inline on submission.
    InlineService,
}
impl fmt::Display for BakeEcsConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::ZeroBudget => "ECS bake bounds must be nonzero",
            Self::InlineService => "ECS baking requires worker-backed BakeService execution",
        })
    }
}
impl std::error::Error for BakeEcsConfigError {}

/// Counters for the latest frame only; no growing diagnostic history.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BakeEcsFrame {
    /// Requests examined for admission, including immediate rejections.
    pub admissions: usize,
    /// Completion messages drained, including obsolete attempts.
    pub completions: usize,
    /// Successful outputs applied to live matching requests.
    pub applied: usize,
    /// Drained messages discarded after cancellation or replacement.
    pub discarded: usize,
}

struct Pending {
    entity: Entity,
    instance: Arc<Instance>,
    ticket: BakeHandle,
}
struct Queued {
    entity: Entity,
    instance: Arc<Instance>,
}
impl Drop for Pending {
    fn drop(&mut self) {
        self.ticket.cancel();
    }
}

/// World-local bounded bridge state. The mutex makes the receive-only service
/// handles safe as a resource; only the exclusive frame system drains them.
pub struct BakeRuntime {
    service: BakeService,
    config: BakeEcsConfig,
    pending: Mutex<VecDeque<Pending>>,
    queued: VecDeque<Queued>,
    frame: BakeEcsFrame,
    stopped: bool,
}
impl Resource for BakeRuntime {}
impl Drop for BakeRuntime {
    fn drop(&mut self) {
        // Cancel before dropping the service, which may own the final pool
        // reference and join workers during application teardown.
        for entry in self
            .pending
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
        {
            entry.instance.cancelled.store(true, Ordering::Release);
            entry.ticket.cancel();
        }
        for entry in &self.queued {
            entry.instance.cancelled.store(true, Ordering::Release);
        }
    }
}

impl BakeRuntime {
    /// Installs or supersedes an entity's request through ECS lifecycle hooks.
    /// Returns false for a dead entity or when the plugin is not installed.
    pub fn insert_request(
        world: &mut World,
        entity: Entity,
        request: BakeRequestComponent,
    ) -> bool {
        if !world.contains_resource::<Self>() {
            return false;
        }
        world.insert(entity, (RequestSlot(request),))
    }
    /// Read-only request access. Cancellation is available on the returned value.
    pub fn request(world: &World, entity: Entity) -> Option<&BakeRequestComponent> {
        world.get::<RequestSlot>(entity).map(|slot| &slot.0)
    }
    /// Removes and cancels the request, retaining its last-good BakeOutput.
    pub fn remove_request(world: &mut World, entity: Entity) -> bool {
        world.remove::<(RequestSlot,)>(entity)
    }
    /// Retained tickets, including cancelled work until its worker completes.
    pub fn pending(&self) -> usize {
        self.pending
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }
    /// Requests in the bounded admission queue.
    pub fn queued(&self) -> usize {
        self.queued.len()
    }
    /// Counters for the latest App update.
    pub fn frame(&self) -> BakeEcsFrame {
        self.frame
    }
}

/// Installs the bridge into PreUpdate with a caller-injected BakeService.
pub struct BakeEcsPlugin {
    service: BakeService,
    config: BakeEcsConfig,
}
impl BakeEcsPlugin {
    /// Validates all bounds before registering any systems/resources.
    pub fn new(service: BakeService, config: BakeEcsConfig) -> Result<Self, BakeEcsConfigError> {
        if config.max_queued == 0
            || config.max_pending == 0
            || config.max_admissions_per_frame == 0
            || config.max_completions_per_frame == 0
        {
            return Err(BakeEcsConfigError::ZeroBudget);
        }
        if service.runs_inline() {
            return Err(BakeEcsConfigError::InlineService);
        }
        Ok(Self { service, config })
    }
}
impl Plugin for BakeEcsPlugin {
    fn build(&self, app: &mut App) {
        let world = app.world_mut();
        assert!(
            !world.contains_resource::<BakeRuntime>(),
            "bake runtime already installed"
        );
        world.insert_resource(BakeRuntime {
            service: self.service.clone(),
            config: self.config,
            pending: Mutex::new(VecDeque::new()),
            queued: VecDeque::new(),
            frame: BakeEcsFrame::default(),
            stopped: false,
        });
        world.register_component_hooks::<RequestSlot>(
            ComponentHooks::new()
                .with_on_replace(cancel_replaced_request)
                .with_on_insert(enqueue_inserted_request),
        );
        app.add_systems(PreUpdate, update_bakes);
    }
    fn shutdown(&self, app: &mut App) {
        let world = app.world_mut();
        if let Some(runtime) = world.get_resource_mut::<BakeRuntime>() {
            runtime.stopped = true;
            let mut cancelled: Vec<_> = runtime
                .queued
                .drain(..)
                .map(|entry| (entry.entity, entry.instance))
                .collect();
            for pending in runtime
                .pending
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .iter()
            {
                pending.ticket.cancel();
                cancelled.push((pending.entity, Arc::clone(&pending.instance)));
            }
            for (entity, instance) in cancelled {
                instance.cancelled.store(true, Ordering::Release);
                if let Some(mut status) = world.get_mut::<BakeStatus>(entity) {
                    if Arc::ptr_eq(&status.instance, &instance) {
                        status.state = BakeState::Cancelled;
                    }
                }
            }
        }
    }
}

fn cancel_replaced_request(ctx: HookContext<'_>) {
    let Some(request) = ctx.world.get::<RequestSlot>(ctx.entity) else {
        return;
    };
    request.cancel();
    let instance = Arc::clone(&request.instance);
    if let Some(runtime) = ctx.world.get_resource_mut::<BakeRuntime>() {
        runtime
            .queued
            .retain(|entry| !Arc::ptr_eq(&entry.instance, &instance));
        for pending in runtime
            .pending
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
        {
            if Arc::ptr_eq(&pending.instance, &instance) {
                pending.ticket.cancel();
            }
        }
    }
    if let Some(mut status) = ctx.world.get_mut::<BakeStatus>(ctx.entity) {
        if Arc::ptr_eq(&status.instance, &instance) {
            status.state = BakeState::Cancelled;
        }
    }
}

fn enqueue_inserted_request(ctx: HookContext<'_>) {
    enqueue_request(ctx.world, ctx.entity);
}

fn enqueue_request(world: &mut World, entity: Entity) {
    let Some(request) = world.get::<RequestSlot>(entity) else {
        return;
    };
    let revision = request.revision;
    let instance = Arc::clone(&request.instance);
    let Some(runtime) = world.get_resource_mut::<BakeRuntime>() else {
        return;
    };
    let state = if runtime.stopped || instance.cancelled.load(Ordering::Acquire) {
        instance.cancelled.store(true, Ordering::Release);
        BakeState::Cancelled
    } else if runtime.queued.len() >= runtime.config.max_queued {
        BakeState::Failed(BakeEcsError::QueueFull {
            max_queued: runtime.config.max_queued,
        })
    } else {
        runtime.queued.push_back(Queued {
            entity,
            instance: Arc::clone(&instance),
        });
        BakeState::Queued
    };
    world.insert(
        entity,
        (BakeStatus {
            revision,
            state,
            instance,
        },),
    );
}

fn matches_request(world: &World, pending: &Pending) -> bool {
    world
        .get::<RequestSlot>(pending.entity)
        .is_some_and(|request| {
            Arc::ptr_eq(&request.instance, &pending.instance)
                && !request.instance.cancelled.load(Ordering::Acquire)
        })
}

fn update_bakes(world: &mut World) {
    let Some(runtime) = world.get_resource_mut::<BakeRuntime>() else {
        return;
    };
    runtime.frame = BakeEcsFrame::default();
    let config = runtime.config;
    let count = runtime
        .pending
        .get_mut()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .len();
    // Always inspect the bounded live set for cleanup, even if the completion
    // budget is exhausted. Keep cancelled handles until drained so repeatedly
    // replacing a shared long-running job cannot grow subscriber/result queues.
    for _ in 0..count {
        let entry = world
            .resource_mut::<BakeRuntime>()
            .pending
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .pop_front()
            .expect("bounded pending set");
        if !matches_request(world, &entry) {
            entry.ticket.cancel();
            if let Some(mut status) = world.get_mut::<BakeStatus>(entry.entity) {
                if Arc::ptr_eq(&status.instance, &entry.instance) {
                    status.state = BakeState::Cancelled;
                }
            }
        }
        let result = if world.resource::<BakeRuntime>().frame.completions
            < config.max_completions_per_frame
        {
            entry.ticket.try_recv()
        } else {
            None
        };
        let Some(result) = result else {
            world
                .resource_mut::<BakeRuntime>()
                .pending
                .get_mut()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push_back(entry);
            continue;
        };
        world.resource_mut::<BakeRuntime>().frame.completions += 1;
        if !matches_request(world, &entry) {
            world.resource_mut::<BakeRuntime>().frame.discarded += 1;
            continue;
        }
        let revision = world
            .get::<RequestSlot>(entry.entity)
            .expect("matching request")
            .revision;
        let state = match result {
            Ok(result) => {
                if world.insert(entry.entity, (BakeOutput { revision, result },)) {
                    world.resource_mut::<BakeRuntime>().frame.applied += 1;
                }
                BakeState::Ready
            }
            Err(BakeError::Cancelled) => BakeState::Cancelled,
            Err(error) => BakeState::Failed(BakeEcsError::Bake(error)),
        };
        if matches_request(world, &entry) {
            world.insert(
                entry.entity,
                (BakeStatus {
                    revision,
                    state,
                    instance: Arc::clone(&entry.instance),
                },),
            );
        }
    }
    for _ in 0..config.max_admissions_per_frame {
        if world.resource::<BakeRuntime>().stopped {
            break;
        }
        let Some(queued) = world.resource_mut::<BakeRuntime>().queued.pop_front() else {
            break;
        };
        world.resource_mut::<BakeRuntime>().frame.admissions += 1;
        let entity = queued.entity;
        let Some(request) = world
            .get::<RequestSlot>(entity)
            .filter(|request| Arc::ptr_eq(&request.instance, &queued.instance))
        else {
            continue;
        };
        let revision = request.revision;
        let instance = Arc::clone(&request.instance);
        let runtime = world.resource::<BakeRuntime>();
        let state = if instance.cancelled.load(Ordering::Acquire) {
            BakeState::Cancelled
        } else if runtime.pending() >= config.max_pending {
            BakeState::Failed(BakeEcsError::AtCapacity {
                max_pending: config.max_pending,
            })
        } else {
            match runtime.service.submit(request.request.clone()) {
                Ok(ticket) => {
                    world
                        .resource_mut::<BakeRuntime>()
                        .pending
                        .get_mut()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .push_back(Pending {
                            entity,
                            instance: Arc::clone(&instance),
                            ticket,
                        });
                    BakeState::Running
                }
                Err(error) => BakeState::Failed(BakeEcsError::Bake(error)),
            }
        };
        world.insert(
            entity,
            (BakeStatus {
                revision,
                state,
                instance,
            },),
        );
    }
}
