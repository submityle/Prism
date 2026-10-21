//! Bounded native cell-file IO, deliberately without runtime/OFPA consumers.
//!
//! Construct outside the frame loop. The host must have no consumers of cell
//! payloads: completing an unload here does not detach ECS entities or renderers.
//! Project replacement/retirement is never acknowledged by this adapter.

use std::{
    path::Path,
    task::{Context, Poll, Waker},
    time::{Duration, Instant},
};

use prism_editor_ui::{
    WorldPanelController, WorldPanelError, WorldPanelIntent, WorldPanelIntentKind, WorldPanelSave,
    WorldPanelTransition,
};
use prism_editor_world::{
    project::WorldProjectToken, CellLoadRequest, FileCellIoLimits, FileCellIoProvider, IoBudget,
    IoCompletion, IoFailure, IoRequestId, WorldIoDriver,
};

use crate::DesktopPanelUpdate;

/// Hard limit on queue operations and terminal deliveries in one host poll.
pub const MAX_WORLD_SESSION_WORK_PER_POLL: usize = 32;

/// One request at a time; limits bound its bytes, poll work and host clock.
#[derive(Clone, Copy, Debug)]
pub struct WorldProjectSessionLimits {
    /// Maximum opaque cell payload, not including the filesystem envelope.
    pub max_cell_bytes: usize,
    /// Admission ceiling for model-wide residency planning (1..=65536 cells).
    pub max_cells: usize,
    /// Combined dequeues and terminal deliveries per poll, in 1..=32.
    pub work_per_poll: usize,
    /// Polling wake interval while a worker or queued command remains.
    pub poll_interval: Duration,
    /// Observation deadline, not permission to abandon/cancel a worker.
    pub operation_timeout: Duration,
    /// Duration of one controller lease tick, relative to construction time.
    pub clock_tick: Duration,
}

impl Default for WorldProjectSessionLimits {
    fn default() -> Self {
        Self {
            max_cell_bytes: 8 * 1024 * 1024,
            max_cells: 4096,
            work_per_poll: 4,
            poll_interval: Duration::from_millis(10),
            operation_timeout: Duration::from_secs(10),
            clock_tick: Duration::from_secs(1),
        }
    }
}

/// Observed host work, not runtime entity/OFPA integration statistics.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct WorldProjectSessionWork {
    /// Commands removed from the controller queue, including retained rejects.
    pub intents_taken: u64,
    /// Load/save requests successfully admitted by the real IO driver.
    pub requests_submitted: u64,
    /// Actual terminal receipts received from the driver, not inferred timeouts.
    pub completions_received: u64,
    /// Detached authoring unloads acknowledged by the original controller.
    pub unloads_completed: u64,
    /// Counts each still-retained request at most once.
    pub deadlines_exceeded: u64,
}

/// An explicitly rejected command or real terminal failure retained for the host.
/// Taking this record transfers responsibility for any unreconciled transition;
/// it is not an acknowledgement of durable success or runtime detachment.
pub struct WorldProjectSessionFailure {
    /// Visible diagnostic for the retained failure.
    pub message: String,
    /// Exact original command and controller generation ticket.
    pub intent: WorldPanelIntent,
    /// Residency transition, if one was begun before the failure.
    pub transition: Option<WorldPanelTransition>,
    /// Original save snapshot, retained for explicit reconciliation.
    pub save: Option<WorldPanelSave>,
    /// Actual terminal completion, with the provider error rendered as text.
    pub completion: Option<IoCompletion<String>>,
    /// True if no transition needs repair, or the controller accepted the
    /// terminal result/rolled back startup. This does not imply save success.
    pub reconciled: bool,
}

enum Operation {
    Load(WorldPanelTransition),
    Save(WorldPanelSave),
}

struct InFlight {
    intent: WorldPanelIntent,
    operation: Operation,
    id: IoRequestId,
    deadline: Instant,
    deadline_exceeded: bool,
    completion: Option<IoCompletion<String>>,
}

/// Real FileCellIoProvider + WorldIoDriver host for a detached world panel.
///
/// Serial admission deliberately avoids multiple model-wide residency plans
/// and duplicate byte snapshots in a frame. The queue remains in the controller
/// until admission. Accepted tickets survive deadlines and all error returns.
pub struct WorldProjectSession {
    driver: WorldIoDriver<FileCellIoProvider>,
    limits: WorldProjectSessionLimits,
    binding: WorldProjectToken,
    origin: Instant,
    last_now: Instant,
    last_tick: u64,
    active: Option<InFlight>,
    failure: Option<WorldProjectSessionFailure>,
    work: WorldProjectSessionWork,
}

impl WorldProjectSession {
    /// Binds an existing root and exact controller project snapshot.
    ///
    /// This is an explicit no-runtime-consumers contract: callers must not use
    /// it while ECS/render/physics consumers refer to current or retired cells.
    /// No entities are attached on load; no consumers are detached on unload.
    /// The session owns the controller's monotonic clock, starting at tick zero.
    pub fn new_detached(
        root: impl AsRef<Path>,
        controller: &WorldPanelController,
        limits: WorldProjectSessionLimits,
        now: Instant,
    ) -> Result<Self, String> {
        if !(1..=MAX_WORLD_SESSION_WORK_PER_POLL).contains(&limits.work_per_poll)
            || !(1..=65536).contains(&limits.max_cells)
            || controller.project().cells().len() > limits.max_cells
            || limits.poll_interval.is_zero()
            || limits.poll_interval > Duration::from_secs(1)
            || limits.operation_timeout.is_zero()
            || limits.operation_timeout > Duration::from_secs(3600)
            || limits.clock_tick.is_zero()
            || limits.clock_tick > Duration::from_secs(60)
            || now.checked_add(limits.operation_timeout).is_none()
            || now.checked_add(limits.poll_interval).is_none()
        {
            return Err("invalid world session work/clock/deadline limits".into());
        }
        Self::check_world(controller)?;
        let provider_limits =
            FileCellIoLimits::new(limits.max_cell_bytes, 1).map_err(|e| e.to_string())?;
        let budget = IoBudget::new(1, limits.max_cell_bytes)
            .map_err(|_| "invalid world session IO budget".to_owned())?;
        let provider = FileCellIoProvider::new(root, provider_limits).map_err(|e| e.to_string())?;
        Ok(Self {
            driver: WorldIoDriver::new(provider, budget),
            limits,
            binding: controller.project().token(),
            origin: now,
            last_now: now,
            last_tick: 0,
            active: None,
            failure: None,
            work: WorldProjectSessionWork::default(),
        })
    }

    fn check_world(controller: &WorldPanelController) -> Result<(), String> {
        if controller.has_retired_world() {
            return Err(
                "world session refuses unacknowledged retired_world; detach real consumers first"
                    .into(),
            );
        }
        Ok(())
    }

    /// Explicitly authorize a new project snapshot of the same stable world.
    /// This does not take retired_world or rewrite any accepted generation
    /// ticket. Even after rebind, a foreign controller cannot accept old IO.
    /// The caller must continue to guarantee no runtime consumers exist.
    pub fn rebind_detached(&mut self, controller: &WorldPanelController) -> Result<(), String> {
        Self::check_world(controller)?;
        if controller.project().identity() != self.binding.identity()
            || self.failure.is_some()
            || controller.project().cells().len() > self.limits.max_cells
        {
            return Err(
                "world session identity differs or a retained failure needs explicit handling"
                    .into(),
            );
        }
        self.binding = controller.project().token();
        Ok(())
    }

    /// Canonical provider root used for all cell file operations.
    pub fn root(&self) -> &Path {
        self.driver.provider().root()
    }

    /// Monotonic counters for actual work performed by this session.
    pub const fn work(&self) -> WorldProjectSessionWork {
        self.work
    }

    /// Accepted request identity, including a retained terminal receipt.
    pub fn pending_request(&self) -> Option<IoRequestId> {
        self.active.as_ref().map(|a| a.id)
    }

    /// Includes a terminal receipt waiting for its original controller.
    pub fn is_busy(&self) -> bool {
        self.active.is_some()
    }

    /// Whether the retained request exceeded its observation deadline.
    pub fn deadline_exceeded(&self) -> bool {
        self.active.as_ref().is_some_and(|a| a.deadline_exceeded)
    }

    /// Failure still owned by this session; no ownership is transferred.
    pub fn failure(&self) -> Option<&WorldProjectSessionFailure> {
        self.failure.as_ref()
    }

    /// Explicit ownership handoff, including CrossCell's unexecuted ticket.
    /// An unreconciled record must be repaired with its original controller;
    /// simply discarding it can leave that model fenced in Loading/Unloading.
    pub fn take_failure(&mut self) -> Option<WorldProjectSessionFailure> {
        self.failure.take()
    }

    /// Polls bounded work without sleeping, joining workers or cancelling IO.
    ///
    /// Deadline expiry remains fenced and schedules another wake until the real
    /// completion. A late genuine save may clean only via controller receipt
    /// validation. Errors retain every ticket/receipt; DesktopWorldPanel hosts
    /// must retry the panel after handling the error, not drop this session.
    pub fn poll(
        &mut self,
        controller: &mut WorldPanelController,
        now: Instant,
    ) -> Result<DesktopPanelUpdate, String> {
        // Fence before touching clocks, queues, workers or any authoring data.
        Self::check_world(controller)?;
        if !controller.project().matches_token(&self.binding) {
            return Err("world session project/controller snapshot changed; explicit rebind_detached required".into());
        }
        let elapsed = now
            .checked_duration_since(self.origin)
            .filter(|_| now >= self.last_now)
            .ok_or("world session clock moved backwards")?;
        let ticks = u64::try_from(elapsed.as_nanos() / self.limits.clock_tick.as_nanos())
            .map_err(|_| "world session lease clock overflow")?;
        // Validate both deadlines before dequeuing an intent or changing time.
        let deadline = now
            .checked_add(self.limits.operation_timeout)
            .ok_or("world session operation deadline overflow")?;
        let wake = now
            .checked_add(self.limits.poll_interval)
            .ok_or("world session wake deadline overflow")?;
        controller.set_time(ticks).map_err(|e| e.to_string())?;
        self.last_now = now;
        let mut changed = ticks != self.last_tick;
        self.last_tick = ticks;
        if let Some(failure) = &self.failure {
            return Err(failure.message.clone());
        }

        for _ in 0..self.limits.work_per_poll {
            if let Some(active) = self.active.as_mut() {
                if now >= active.deadline && !active.deadline_exceeded {
                    active.deadline_exceeded = true;
                    self.work.deadlines_exceeded = self.work.deadlines_exceeded.saturating_add(1);
                    changed = true;
                }
                if active.completion.is_none() {
                    let mut cx = Context::from_waker(Waker::noop());
                    match self.driver.poll(&mut cx) {
                        Poll::Pending => break,
                        Poll::Ready(None) => {
                            return Err(
                                "world IO driver lost an accepted request; ticket retained".into()
                            )
                        }
                        Poll::Ready(Some(completion)) => {
                            active.completion = Some(text_completion(completion));
                            self.work.completions_received =
                                self.work.completions_received.saturating_add(1);
                        }
                    }
                }
                self.reconcile(controller)?;
                changed = true;
                continue;
            }

            // Do not make a residency transition if its bounded worker slot is
            // still held by filesystem cleanup from a completed operation.
            if self.driver.provider().outstanding_operations() != 0 {
                break;
            }
            let Some(intent) = controller.take_intent() else {
                break;
            };
            self.work.intents_taken = self.work.intents_taken.saturating_add(1);
            changed = true;
            self.start(controller, intent, deadline)?;
            // A newly accepted request is polled on the next host tick. This
            // avoids completing arbitrarily fast worker chains in one frame.
            if self.active.is_some() {
                break;
            }
        }
        let queued = controller.pending_intents() != 0;
        let fenced = self.active.is_some()
            || self.failure.is_some()
            || queued
            || controller.project_is_dirty()
            || controller.project_io_pending()
            || controller.project_candidate().is_some()
            || controller.has_retired_world();
        Ok(DesktopPanelUpdate {
            changed,
            fenced,
            wake_at: (self.active.is_some() || queued).then_some(wake),
        })
    }

    fn reject(
        &mut self,
        intent: WorldPanelIntent,
        message: String,
        transition: Option<WorldPanelTransition>,
        reconciled: bool,
    ) -> String {
        self.failure = Some(WorldProjectSessionFailure {
            message: message.clone(),
            intent,
            transition,
            save: None,
            completion: None,
            reconciled,
        });
        message
    }

    fn start(
        &mut self,
        controller: &mut WorldPanelController,
        intent: WorldPanelIntent,
        deadline: Instant,
    ) -> Result<(), String> {
        match intent.kind() {
            WorldPanelIntentKind::CrossCell { .. } => {
                Err(self.reject(intent, "CrossCell rejected: opaque cell bytes cannot supply authoritative actor/cell transaction edits; ticket retained for an explicit transaction host".into(), None, true))
            }
            WorldPanelIntentKind::Load(_) | WorldPanelIntentKind::Unload(_) => {
                let transition = match controller.begin_residency(&intent) {
                    Ok(t) => t,
                    Err(e) => return Err(self.reject(intent, e.to_string(), None, true)),
                };
                if matches!(intent.kind(), WorldPanelIntentKind::Unload(_)) {
                    // The constructor/rebind's detached-consumer contract is
                    // essential: this acknowledges authoring bytes only.
                    return match controller.complete_unload(transition.clone()) {
                        Ok(()) => {
                            self.work.unloads_completed = self.work.unloads_completed.saturating_add(1);
                            Ok(())
                        }
                        Err(e) => Err(self.reject(intent, e.to_string(), Some(transition), false)),
                    };
                }
                let token = transition.token();
                let Some(file) = controller.world().cell_file(token.cell()).cloned() else {
                    let rollback = controller.fail_residency(transition.clone());
                    let reconciled = rollback.is_ok();
                    let message = match rollback {
                        Ok(()) => "begun load has no cell file identity".into(),
                        Err(e) => format!("begun load has no cell file identity; rollback failed: {e}"),
                    };
                    return Err(self.reject(intent, message, Some(transition), reconciled));
                };
                let max_bytes = self.limits.max_cell_bytes.min(controller.world().limits().max_cell_bytes());
                let request = CellLoadRequest::new(token, file, max_bytes);
                match self.driver.submit_load(request) {
                    Ok(id) => {
                        self.active = Some(InFlight { intent, operation: Operation::Load(transition), id,
                            deadline, deadline_exceeded: false, completion: None });
                        self.work.requests_submitted = self.work.requests_submitted.saturating_add(1);
                        Ok(())
                    }
                    Err(e) => {
                        let rollback = controller.fail_residency(transition.clone());
                        let reconciled = rollback.is_ok();
                        let message = match rollback {
                            Ok(()) => e.to_string(),
                            Err(rollback) => format!("{e}; load rollback failed: {rollback}"),
                        };
                        Err(self.reject(intent, message, Some(transition), reconciled))
                    }
                }
            }
            WorldPanelIntentKind::Save(cell) => {
                // Bound before prepare_save allocates its real byte snapshot.
                if controller.world().payload(cell).is_some_and(|b| b.len() > self.limits.max_cell_bytes) {
                    return Err(self.reject(intent, "cell save exceeds world session payload budget".into(), None, true));
                }
                let save = match controller.prepare_save(&intent) {
                    Ok(save) => save,
                    Err(e) => return Err(self.reject(intent, e.to_string(), None, true)),
                };
                match self.driver.submit_save(save.request().clone(), save.payload().to_vec()) {
                    Ok(id) => {
                        self.active = Some(InFlight { intent, operation: Operation::Save(save), id,
                            deadline, deadline_exceeded: false, completion: None });
                        self.work.requests_submitted = self.work.requests_submitted.saturating_add(1);
                        Ok(())
                    }
                    Err(e) => {
                        let message = e.to_string();
                        self.failure = Some(WorldProjectSessionFailure {
                            message: message.clone(), intent, transition: None, save: Some(save),
                            completion: None, reconciled: true,
                        });
                        Err(message)
                    }
                }
            }
        }
    }

    fn reconcile(&mut self, controller: &mut WorldPanelController) -> Result<(), String> {
        let active = self.active.as_ref().expect("accepted request retained");
        let completion = active.completion.as_ref().expect("real terminal received");
        if completion.request_id() != active.id {
            return Err(
                "world IO completion identity mismatch; ticket and receipt retained".into(),
            );
        }
        let failure = match completion {
            IoCompletion::Failed { error, .. } => Some(error.to_string()),
            IoCompletion::Cancelled { .. } => Some("actual cell IO cancellation".into()),
            _ => None,
        };
        // Preserve the original terminal bytes/receipt if controller admission
        // fails. This costs at most one additional bounded load payload copy.
        let receipt = copy_completion(completion);
        let result = match &active.operation {
            Operation::Load(t) => controller.apply_load_completion(t, receipt),
            Operation::Save(s) => controller.apply_save_completion(s, receipt),
        };
        if matches!(&result, Err(WorldPanelError::StaleIntent)) {
            return Err(
                "stale controller generation; original IO ticket and terminal receipt retained"
                    .into(),
            );
        }
        let reconciled = result.is_ok();
        let message = match result {
            Ok(true) => None,
            Ok(false) => {
                Some(failure.unwrap_or_else(|| "cell IO completion was not published".into()))
            }
            Err(e) => Some(format!("cell IO terminal could not reconcile: {e}")),
        };
        let active = self.active.take().expect("reconciled request retained");
        if let Some(message) = message {
            let (transition, save) = match active.operation {
                Operation::Load(t) => (Some(t), None),
                Operation::Save(s) => (None, Some(s)),
            };
            self.failure = Some(WorldProjectSessionFailure {
                message: message.clone(),
                intent: active.intent,
                transition,
                save,
                completion: active.completion,
                reconciled,
            });
            return Err(message);
        }
        Ok(())
    }
}

fn text_completion<E: std::fmt::Display>(c: IoCompletion<E>) -> IoCompletion<String> {
    match c {
        IoCompletion::Loaded {
            request_id,
            request,
            bytes,
            revision,
        } => IoCompletion::Loaded {
            request_id,
            request,
            bytes,
            revision,
        },
        IoCompletion::Saved {
            request_id,
            request,
        } => IoCompletion::Saved {
            request_id,
            request,
        },
        IoCompletion::Failed {
            request_id,
            operation,
            error,
        } => {
            let error = match error {
                IoFailure::Provider(e) => IoFailure::Provider(e.to_string()),
                IoFailure::PayloadTooLarge { limit, actual } => {
                    IoFailure::PayloadTooLarge { limit, actual }
                }
            };
            IoCompletion::Failed {
                request_id,
                operation,
                error,
            }
        }
        IoCompletion::Cancelled {
            request_id,
            operation,
        } => IoCompletion::Cancelled {
            request_id,
            operation,
        },
    }
}

fn copy_completion(c: &IoCompletion<String>) -> IoCompletion<String> {
    match c {
        IoCompletion::Loaded {
            request_id,
            request,
            bytes,
            revision,
        } => IoCompletion::Loaded {
            request_id: *request_id,
            request: request.clone(),
            bytes: bytes.clone(),
            revision: *revision,
        },
        IoCompletion::Saved {
            request_id,
            request,
        } => IoCompletion::Saved {
            request_id: *request_id,
            request: request.clone(),
        },
        IoCompletion::Failed {
            request_id,
            operation,
            error,
        } => IoCompletion::Failed {
            request_id: *request_id,
            operation: operation.clone(),
            error: error.clone(),
        },
        IoCompletion::Cancelled {
            request_id,
            operation,
        } => IoCompletion::Cancelled {
            request_id: *request_id,
            operation: operation.clone(),
        },
    }
}
