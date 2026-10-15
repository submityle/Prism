//! Externally driven, isolated play-in-editor (PIE) applications.
//!
//! Each instance owns an independent [`App`], world, clocks and plugin lifecycle.
//! Creation installs [`RunMode::EditorEmbedded`] before calling the builder and
//! starts paused. No task pool, render lane, runner or shared resource is installed.
//! Plugins may explicitly opt into those facilities, including sharing handles.
//!
//! Frames use a host-supplied delta for the **main** time domain. Paused calls
//! consume no delta; resuming never catches up wall time. Virtual time's own
//! pause, scale and max-delta rules still apply. Secondary sub-app time domains
//! retain their own configured strategies: hosts using them should explicitly
//! select manual deltas too if wall-clock catch-up is undesirable.
//!
//! System debugging is an explicit, independent **schedule pass**, not a partial
//! App frame. Opening a session first polls plugin readiness and runs Startup
//! once. Stepping then reuses ECS [`Stepping`] without advancing clocks, changing
//! the active time source, driving the fixed accumulator, extracting sub-apps or
//! running adjacent phases. In particular, debugging a fixed schedule does not
//! manufacture a fixed tick: systems observe the world's existing clock context.
//! An ordinary frame after the session starts at `First` and runs all its phases,
//! including any schedule already executed by the debugger. Debug effects are
//! retained; this API promises no rollback or frame continuation.
//!
//! ```
//! use prism_app::{PieInstances, Update};
//! use prism_time::Duration;
//!
//! let mut editor = PieInstances::new();
//! let preview = editor.create(|app| { app.add_systems(Update, || {}); })?;
//! let instance = editor.get_mut(preview)?;
//! instance.step_frame(Duration::from_millis(16))?;
//! instance.debug_schedule(Update)?;
//! instance.step_system()?;
//! instance.finish_schedule()?;
//! instance.end_debug()?;
//! editor.remove(preview)?;
//! # Ok::<(), prism_app::PieError>(())
//! ```

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use prism_ecs::schedule::{
    BoxedScheduleLabel, ContinueReport, Schedule, ScheduleLabel, Schedules, StepReport, Stepping,
};
use prism_ecs::world::World;
use prism_time::Duration;

use crate::fixed::{
    AfterFixedMainLoop, BeforeFixedMainLoop, FixedFirst, FixedLast, FixedPostUpdate,
    FixedPreUpdate, FixedUpdate,
};
use crate::frame::{FrameContinueReport, FrameSession, FrameStepReport, FrameSystem};
use crate::schedule::{
    ExitConfirmation, First, Last, PostStartup, PostUpdate, PreStartup, PreUpdate, Shutdown,
    Startup, StateTransition, Update,
};
use crate::{
    App, AppExit, AppExitRequest, Capabilities, EngineClocks, QualityTier, RunMode,
    TimeUpdateStrategy,
};

// Identity allocation alone is process-wide; no application state is shared.
// Checked allocation prevents stale or foreign-manager IDs from aliasing.
static NEXT_INSTANCE_ID: AtomicU64 = AtomicU64::new(1);

/// An opaque, process-local instance identity, never reused by another manager.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PieInstanceId(u64);

impl PieInstanceId {
    /// Numeric identity for editor inspection; not a persistent save-file ID.
    #[must_use]
    pub fn get(self) -> u64 {
        self.0
    }
}

/// The host-controlled instance lifecycle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PieState {
    /// Whole-App updates are suspended, including all secondary sub-apps.
    Paused,
    /// The host may drive frames with [`PieInstance::update`].
    Running,
    /// Shutdown has been entered; the instance cannot restart.
    Stopped(AppExit),
}

/// Outcome of a host frame request. `Updated` and an exit after an update
/// consume their supplied delta; readiness polling and pre-frame exits do not.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PieFrameOutcome {
    /// Paused: no readiness polling, Startup or frame execution occurred.
    Paused,
    /// Plugins are not ready; no frame or time advancement occurred.
    WaitingForPlugins,
    /// One complete App update ran, including its configured phase order.
    Updated,
    /// A confirmed exit stopped the instance (possibly after this frame ran).
    Stopped(AppExit),
}

/// A rejected operation; state errors do not execute systems.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PieError {
    /// No instance with this ID belongs to the manager.
    UnknownInstance(PieInstanceId),
    /// All process-local identities have been consumed.
    IdsExhausted,
    /// This operation requires a paused instance.
    MustBePaused,
    /// The instance has entered shutdown.
    Stopped(AppExit),
    /// A debug session needs Startup, but plugins are not ready yet.
    PluginsNotReady,
    /// Close the existing session before editing the App or running frames.
    DebugSessionActive,
    /// A complete-frame stepping session is already open.
    FrameSessionActive,
    /// No complete-frame stepping session is open.
    NoFrameSession,
    /// The complete-frame session still has systems to execute.
    IncompleteFrame,
    /// No schedule is currently being debugged.
    NoDebugSession,
    /// Finish the schedule pass or explicitly discard its remaining work.
    IncompleteSchedule,
    /// The requested label is not registered in the main world.
    MissingSchedule(BoxedScheduleLabel),
    /// Startup and exit schedules belong to the App lifecycle, not the debugger.
    LifecycleSchedule(BoxedScheduleLabel),
    /// The breakpoint is outside the initialized schedule's node range.
    InvalidBreakpoint(usize),
}

impl std::fmt::Display for PieError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownInstance(id) => write!(f, "unknown PIE instance {}", id.get()),
            Self::IdsExhausted => f.write_str("PIE instance identities exhausted"),
            Self::MustBePaused => f.write_str("pause the PIE instance first"),
            Self::Stopped(exit) => write!(f, "PIE instance stopped: {exit:?}"),
            Self::PluginsNotReady => f.write_str("PIE plugins are not ready"),
            Self::DebugSessionActive => f.write_str("close the schedule debug session first"),
            Self::FrameSessionActive => f.write_str("close the complete-frame session first"),
            Self::NoFrameSession => f.write_str("no complete-frame stepping session"),
            Self::IncompleteFrame => {
                f.write_str("complete-frame session still has uncompleted work")
            }
            Self::NoDebugSession => f.write_str("no schedule debug session"),
            Self::IncompleteSchedule => f.write_str("schedule pass still has uncompleted work"),
            Self::MissingSchedule(label) => write!(f, "missing schedule {label:?}"),
            Self::LifecycleSchedule(label) => {
                write!(f, "lifecycle schedule {label:?} cannot be stepped")
            }
            Self::InvalidBreakpoint(node) => write!(f, "invalid breakpoint node {node}"),
        }
    }
}

impl std::error::Error for PieError {}

/// Owns independent PIE instances. All operations run on the calling thread
/// unless an instance's builder explicitly enables App concurrency.
#[derive(Default)]
pub struct PieInstances {
    instances: BTreeMap<PieInstanceId, PieInstance>,
}

impl PieInstances {
    /// An empty editor instance collection.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Build an independent, initially paused App. The builder sees embedded
    /// mode and private clocks before installing plugins. Startup is deferred
    /// until preparation, a frame step, or a running update.
    ///
    /// The builder is an assembly hook: do not replace the App or call its runner.
    pub fn create(&mut self, build: impl FnOnce(&mut App)) -> Result<PieInstanceId, PieError> {
        let id = NEXT_INSTANCE_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
                next.checked_add(1)
            })
            .map(PieInstanceId)
            .map_err(|_| PieError::IdsExhausted)?;
        let mut instance = PieInstance {
            id,
            app: embedded_app(),
            state: PieState::Paused,
            debug: None,
            frame: None,
        };
        build(&mut instance.app);
        instance.app.set_run_mode(RunMode::EditorEmbedded);
        self.instances.insert(id, instance);
        Ok(id)
    }

    /// Inspect a specific instance. Removed and foreign-manager IDs are errors.
    pub fn get(&self, id: PieInstanceId) -> Result<&PieInstance, PieError> {
        self.instances.get(&id).ok_or(PieError::UnknownInstance(id))
    }

    /// Control or edit a specific instance.
    pub fn get_mut(&mut self, id: PieInstanceId) -> Result<&mut PieInstance, PieError> {
        self.instances
            .get_mut(&id)
            .ok_or(PieError::UnknownInstance(id))
    }

    /// The live collection's IDs, in creation order (including stopped Apps).
    pub fn ids(&self) -> impl Iterator<Item = PieInstanceId> + '_ {
        self.instances.keys().copied()
    }

    /// Number of retained instances, including stopped Apps.
    #[must_use]
    pub fn len(&self) -> usize {
        self.instances.len()
    }

    /// Whether no instances are retained.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }

    /// Stop gracefully before removing the instance. The returned stopped App
    /// remains available for inspection. If shutdown unwinds, it stays in this
    /// collection marked stopped, following App's idempotent shutdown contract.
    pub fn remove(&mut self, id: PieInstanceId) -> Result<PieInstance, PieError> {
        self.get_mut(id)?.stop();
        Ok(self.instances.remove(&id).expect("instance was checked"))
    }
}

/// An independently owned App with host frame and schedule-debug controls.
///
/// Normal removal/drop runs graceful shutdown. During an existing panic unwind,
/// Drop does not call user shutdown code; explicitly stop before fallible editor
/// teardown if graceful shutdown must be observed on that path too. This module
/// does not add a panic isolation policy to App or its systems.
pub struct PieInstance {
    id: PieInstanceId,
    app: App,
    state: PieState,
    debug: Option<PieScheduleDebug>,
    frame: Option<FrameSession>,
}

impl PieInstance {
    /// This instance's stable identity.
    #[must_use]
    pub fn id(&self) -> PieInstanceId {
        self.id
    }

    /// Current host-controlled lifecycle state.
    #[must_use]
    pub fn state(&self) -> PieState {
        self.state
    }

    /// Read the owned App, including after shutdown or during debugging.
    #[must_use]
    pub fn app(&self) -> &App {
        &self.app
    }

    /// Edit a paused App outside a debug session. This is an assembly/world-edit
    /// escape hatch, not an alternate runner: direct lifecycle calls, replacing
    /// the App or manually executing schedules bypass PIE's control contract.
    pub fn app_mut(&mut self) -> Result<&mut App, PieError> {
        self.require_paused()?;
        self.require_no_debug()?;
        self.require_no_frame()?;
        self.app.sync_sub_apps();
        Ok(&mut self.app)
    }

    /// Poll plugin readiness and run Startup exactly once, without a frame or
    /// clock advancement. A confirmed exit is shut down and returned as an
    /// error. Returns `false` while plugins are still preparing.
    pub fn prepare(&mut self) -> Result<bool, PieError> {
        self.require_live()?;
        self.require_no_debug()?;
        if !self.app.try_finish() {
            return Ok(false);
        }
        if let Some(exit) = self.app.poll_exit() {
            self.stop_with(exit);
            return Err(PieError::Stopped(exit));
        }
        Ok(true)
    }

    /// Suspend whole-App updates and join any explicitly enabled render lane.
    /// Does not change any virtual clock's own pause flag.
    pub fn pause(&mut self) -> Result<(), PieError> {
        self.require_live()?;
        self.require_no_frame()?;
        self.app.sync_sub_apps();
        self.state = PieState::Paused;
        Ok(())
    }

    /// Permit subsequent host updates. A schedule debug session must be closed
    /// explicitly first, even when parked at a pass boundary.
    pub fn resume(&mut self) -> Result<(), PieError> {
        self.require_live()?;
        self.require_no_debug()?;
        self.require_no_frame()?;
        self.state = PieState::Running;
        Ok(())
    }

    /// Run one normal frame if running. Paused calls discard `delta` entirely.
    /// Exit confirmation is polled before and after the frame. A frame includes
    /// the existing fixed driver, phase insertions, extraction and sub-apps.
    pub fn update(&mut self, delta: Duration) -> Result<PieFrameOutcome, PieError> {
        self.require_live()?;
        self.require_no_debug()?;
        self.require_no_frame()?;
        if self.state == PieState::Paused {
            return Ok(PieFrameOutcome::Paused);
        }
        self.advance_frame(delta)
    }

    /// Run exactly one normal App frame while remaining paused. The supplied
    /// main-world delta still obeys virtual-time pause, dilation and clamp rules.
    /// A not-ready plugin consumes neither a frame nor the supplied delta.
    pub fn step_frame(&mut self, delta: Duration) -> Result<PieFrameOutcome, PieError> {
        self.require_paused()?;
        self.require_no_debug()?;
        self.require_no_frame()?;
        self.advance_frame(delta)
    }

    /// Open a persistent complete-frame session without executing its first
    /// system. The instance remains paused while the host advances it through
    /// [`step_frame_system`](Self::step_frame_system) or
    /// [`continue_frame`](Self::continue_frame).
    pub fn begin_frame(&mut self, delta: Duration) -> Result<(), PieError> {
        self.require_paused()?;
        self.require_no_debug()?;
        if self.frame.is_some() {
            return Err(PieError::FrameSessionActive);
        }
        match self.prepare() {
            Ok(true) => {}
            Ok(false) => return Err(PieError::PluginsNotReady),
            Err(error) => return Err(error),
        }
        self.frame = Some(FrameSession::start(&mut self.app, delta));
        Ok(())
    }

    /// Execute one system, extract event, or frame-boundary event in the open
    /// complete-frame session.
    pub fn step_frame_system(&mut self) -> Result<FrameStepReport, PieError> {
        self.require_paused()?;
        self.require_no_debug()?;
        let session = self.frame.as_mut().ok_or(PieError::NoFrameSession)?;
        Ok(session.step(&mut self.app))
    }

    /// Continue the open frame to its next break-before breakpoint or complete
    /// frame boundary.
    pub fn continue_frame(&mut self) -> Result<FrameContinueReport, PieError> {
        self.require_paused()?;
        self.require_no_debug()?;
        let session = self.frame.as_mut().ok_or(PieError::NoFrameSession)?;
        Ok(session.continue_frame(&mut self.app))
    }

    /// Add a break-before breakpoint to the current complete-frame session.
    pub fn add_frame_breakpoint(&mut self, breakpoint: FrameSystem) -> Result<bool, PieError> {
        let session = self.frame.as_mut().ok_or(PieError::NoFrameSession)?;
        Ok(session.add_breakpoint(breakpoint))
    }

    /// Remove a complete-frame breakpoint.
    pub fn remove_frame_breakpoint(&mut self, breakpoint: FrameSystem) -> Result<bool, PieError> {
        let session = self.frame.as_mut().ok_or(PieError::NoFrameSession)?;
        Ok(session.remove_breakpoint(breakpoint))
    }

    /// Abandon the current frame session. Executed systems remain applied and
    /// no implicit rollback or replay occurs.
    pub fn abort_frame(&mut self) -> Result<(), PieError> {
        self.require_paused()?;
        self.require_no_debug()?;
        let mut session = self.frame.take().ok_or(PieError::NoFrameSession)?;
        session.abort();
        self.app.sync_sub_apps();
        Ok(())
    }

    /// Close a frame session after it reached its complete-frame boundary.
    pub fn end_frame(&mut self) -> Result<(), PieError> {
        let session = self.frame.as_ref().ok_or(PieError::NoFrameSession)?;
        if !session.is_complete() {
            return Err(PieError::IncompleteFrame);
        }
        self.frame = None;
        Ok(())
    }

    /// Force an editor stop: discard any debug remainder, synchronize an
    /// opted-in render lane and call [`App::run_shutdown`] exactly once.
    /// This explicit host stop bypasses exit vetoes; system exit requests use
    /// `poll_exit` through frame/preparation operations instead. Never runs
    /// Startup merely to stop an unprepared instance.
    pub fn stop(&mut self) -> AppExit {
        self.stop_with(AppExit::Success)
    }

    /// Open a persistent debug session for a concrete label in the main world.
    /// Other instances can continue to run while this instance remains paused.
    /// Built-in startup/exit schedules are rejected to protect their lifecycle
    /// gates. A custom lifecycle phase remains the caller's responsibility.
    pub fn debug_schedule<L: ScheduleLabel + Clone>(&mut self, label: L) -> Result<(), PieError> {
        self.require_paused()?;
        self.require_no_debug()?;
        let key = BoxedScheduleLabel::new(label.clone());
        if [
            BoxedScheduleLabel::new(PreStartup),
            BoxedScheduleLabel::new(Startup),
            BoxedScheduleLabel::new(PostStartup),
            BoxedScheduleLabel::new(Shutdown),
            BoxedScheduleLabel::new(ExitConfirmation),
        ]
        .contains(&key)
        {
            return Err(PieError::LifecycleSchedule(key));
        }
        if !self.prepare()? {
            return Err(PieError::PluginsNotReady);
        }
        let mut debug = PieScheduleDebug {
            label: key,
            target: Box::new(ConcreteSchedule(label)),
            stepping: Stepping::new(),
            node_count: 0,
        };
        debug.node_count = debug.with_schedule(self.app.world_mut(), |_, schedule, world| {
            schedule.initialize(world);
            schedule.len()
        })?;
        self.debug = Some(debug);
        Ok(())
    }

    /// Inspect the session cursor, label and breakpoints.
    #[must_use]
    pub fn debug(&self) -> Option<&PieScheduleDebug> {
        self.debug.as_ref()
    }

    /// Add a validated break-before breakpoint to this instance's session.
    /// Mutable session ownership never escapes, preventing cursor/cache swaps
    /// between instances that happen to use the same schedule label.
    pub fn add_breakpoint(&mut self, node: usize) -> Result<bool, PieError> {
        self.debug
            .as_mut()
            .ok_or(PieError::NoDebugSession)?
            .add_breakpoint(node)
    }

    /// Remove a breakpoint, returning whether it existed.
    pub fn remove_breakpoint(&mut self, node: usize) -> Result<bool, PieError> {
        Ok(self
            .debug
            .as_mut()
            .ok_or(PieError::NoDebugSession)?
            .stepping
            .remove_breakpoint(node))
    }

    /// Clear breakpoints without resetting the cursor or condition cache.
    pub fn clear_breakpoints(&mut self) -> Result<(), PieError> {
        self.debug
            .as_mut()
            .ok_or(PieError::NoDebugSession)?
            .stepping
            .clear_breakpoints();
        Ok(())
    }

    /// Resolve a system's full ECS name to a node index for a breakpoint.
    /// This inspection does not advance the pass or evaluate conditions.
    pub fn find_debug_system(&mut self, name: &str) -> Result<Option<usize>, PieError> {
        self.debug
            .as_mut()
            .ok_or(PieError::NoDebugSession)?
            .with_schedule(self.app.world_mut(), |stepping, schedule, _| {
                stepping.find_system(schedule, name)
            })
    }

    /// Evaluate and process one system using ECS conditions/deferred semantics.
    /// `FrameBoundary` here means **schedule-pass boundary**, not App frame.
    /// System-generated exit requests wait until preparation/a frame or stop;
    /// stepping never implicitly executes exit-confirmation/shutdown systems.
    pub fn step_system(&mut self) -> Result<StepReport, PieError> {
        self.debug
            .as_mut()
            .ok_or(PieError::NoDebugSession)?
            .with_schedule(self.app.world_mut(), |stepping, schedule, world| {
                stepping.step_system(schedule, world)
            })
    }

    /// Continue the selected schedule to its next breakpoint or pass boundary.
    pub fn continue_schedule(&mut self) -> Result<ContinueReport, PieError> {
        self.debug
            .as_mut()
            .ok_or(PieError::NoDebugSession)?
            .with_schedule(self.app.world_mut(), |stepping, schedule, world| {
                stepping.continue_frame(schedule, world)
            })
    }

    /// Complete the current schedule pass, ignoring breakpoints. At a fresh
    /// pass boundary this runs a whole schedule pass. Does not close the session.
    pub fn finish_schedule(&mut self) -> Result<Vec<usize>, PieError> {
        self.debug
            .as_mut()
            .ok_or(PieError::NoDebugSession)?
            .with_schedule(self.app.world_mut(), |stepping, schedule, world| {
                stepping.run_remaining_frame(schedule, world)
            })
    }

    /// Close only at a schedule-pass boundary. After the last system, use
    /// `step_system` once more to observe that boundary, or `finish_schedule`.
    pub fn end_debug(&mut self) -> Result<(), PieError> {
        if !self
            .debug
            .as_ref()
            .ok_or(PieError::NoDebugSession)?
            .at_boundary()
        {
            return Err(PieError::IncompleteSchedule);
        }
        self.debug = None;
        Ok(())
    }

    /// Explicitly abandon the remainder of a pass. Already applied effects
    /// remain; there is no rollback and no deferred execution in session Drop.
    pub fn discard_debug(&mut self) -> Result<(), PieError> {
        self.debug.take().ok_or(PieError::NoDebugSession)?;
        Ok(())
    }

    fn require_live(&self) -> Result<(), PieError> {
        if let PieState::Stopped(exit) = self.state {
            Err(PieError::Stopped(exit))
        } else {
            Ok(())
        }
    }

    fn require_paused(&self) -> Result<(), PieError> {
        self.require_live()?;
        if self.state == PieState::Paused {
            Ok(())
        } else {
            Err(PieError::MustBePaused)
        }
    }

    fn require_no_debug(&self) -> Result<(), PieError> {
        if self.debug.is_some() {
            Err(PieError::DebugSessionActive)
        } else {
            Ok(())
        }
    }

    fn require_no_frame(&self) -> Result<(), PieError> {
        if self.frame.is_some() {
            Err(PieError::FrameSessionActive)
        } else {
            Ok(())
        }
    }

    fn advance_frame(&mut self, delta: Duration) -> Result<PieFrameOutcome, PieError> {
        match self.prepare() {
            Ok(false) => return Ok(PieFrameOutcome::WaitingForPlugins),
            Err(PieError::Stopped(exit)) => return Ok(PieFrameOutcome::Stopped(exit)),
            Err(error) => return Err(error),
            Ok(true) => {}
        }
        self.app
            .set_time_update_strategy(TimeUpdateStrategy::ManualDelta(delta));
        self.app.update();
        // A single-frame step is complete, including any explicitly opted-in
        // pipeline, before the editor inspects or pauses this instance again.
        self.app.sync_sub_apps();
        if let Some(exit) = self.app.poll_exit() {
            self.stop_with(exit);
            Ok(PieFrameOutcome::Stopped(exit))
        } else {
            Ok(PieFrameOutcome::Updated)
        }
    }

    fn stop_with(&mut self, exit: AppExit) -> AppExit {
        if let PieState::Stopped(previous) = self.state {
            return previous;
        }
        // Mark stopped before invoking user teardown; App marks its own
        // shutdown path entered before running those systems too.
        self.state = PieState::Stopped(exit);
        self.debug = None;
        self.frame = None;
        self.app.sync_sub_apps();
        self.app.run_shutdown();
        exit
    }
}

impl Drop for PieInstance {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            self.stop();
        }
    }
}

/// Persistent schedule debugger metadata. Reports use ECS node indices and
/// ordered cursor positions, exactly as [`Stepping`] does.
pub struct PieScheduleDebug {
    label: BoxedScheduleLabel,
    target: Box<dyn ScheduleTarget>,
    stepping: Stepping,
    node_count: usize,
}

impl PieScheduleDebug {
    /// Label selected when the session opened.
    #[must_use]
    pub fn label(&self) -> &BoxedScheduleLabel {
        &self.label
    }

    /// Next position in the initialized schedule's execution order.
    #[must_use]
    pub fn position(&self) -> usize {
        self.stepping.position()
    }

    /// Whether the session is between schedule passes.
    #[must_use]
    pub fn at_boundary(&self) -> bool {
        self.stepping.is_at_frame_boundary()
    }

    fn add_breakpoint(&mut self, node: usize) -> Result<bool, PieError> {
        if node >= self.node_count {
            return Err(PieError::InvalidBreakpoint(node));
        }
        Ok(self.stepping.add_breakpoint(node))
    }

    /// Breakpoint node indices, in ascending order.
    pub fn breakpoints(&self) -> impl Iterator<Item = usize> + '_ {
        self.stepping.breakpoints()
    }

    fn with_schedule<R>(
        &mut self,
        world: &mut World,
        run: impl FnOnce(&mut Stepping, &mut Schedule, &mut World) -> R,
    ) -> Result<R, PieError> {
        let schedule = self
            .target
            .remove(world)
            .ok_or_else(|| PieError::MissingSchedule(self.label.clone()))?;
        let mut guard = ScheduleGuard {
            world,
            target: &*self.target,
            schedule: Some(schedule),
        };
        Ok(run(
            &mut self.stepping,
            guard.schedule.as_mut().expect("leased schedule"),
            guard.world,
        ))
    }
}

// Retain the concrete label type. Passing a BoxedScheduleLabel to Schedules'
// generic methods would look up a different, double-boxed label instead.
trait ScheduleTarget: Send + Sync {
    fn remove(&self, world: &mut World) -> Option<Schedule>;
    fn restore(&self, world: &mut World, schedule: Schedule);
}

struct ConcreteSchedule<L>(L);

impl<L: ScheduleLabel + Clone> ScheduleTarget for ConcreteSchedule<L> {
    fn remove(&self, world: &mut World) -> Option<Schedule> {
        world
            .get_resource_mut::<Schedules>()?
            .remove(self.0.clone())
    }

    fn restore(&self, world: &mut World, schedule: Schedule) {
        world.init_resource::<Schedules>();
        world
            .resource_mut::<Schedules>()
            .insert(self.0.clone(), schedule);
    }
}

// A schedule is absent only during its execution, preventing self-recursion.
// Restore on unwind too; do not lose systems when ECS propagates a panic.
struct ScheduleGuard<'a> {
    world: &'a mut World,
    target: &'a dyn ScheduleTarget,
    schedule: Option<Schedule>,
}

impl Drop for ScheduleGuard<'_> {
    fn drop(&mut self) {
        if let Some(schedule) = self.schedule.take() {
            self.target.restore(self.world, schedule);
        }
    }
}

fn embedded_app() -> App {
    // App::new installs a worker pool under multi_thread. Build its core
    // resources/schedules explicitly so PIE has no implicit thread creation.
    let mut app = App::empty();
    let schedules = app.world_mut().resource_mut::<Schedules>();
    macro_rules! install {
        ($($label:expr),+ $(,)?) => { $(schedules.insert($label, Schedule::new());)+ };
    }
    install!(
        PreStartup,
        Startup,
        PostStartup,
        First,
        PreUpdate,
        StateTransition,
        Update,
        PostUpdate,
        Last,
        Shutdown,
        ExitConfirmation,
        FixedFirst,
        FixedPreUpdate,
        FixedUpdate,
        FixedPostUpdate,
        FixedLast,
        BeforeFixedMainLoop,
        AfterFixedMainLoop
    );
    let capabilities = Capabilities::default();
    app.insert_resource(QualityTier::from_capabilities(&capabilities))
        .insert_resource(capabilities)
        .insert_resource(AppExitRequest::default())
        .insert_resource(EngineClocks::new())
        .set_time_update_strategy(TimeUpdateStrategy::ManualDelta(Duration::ZERO))
        .set_run_mode(RunMode::EditorEmbedded);
    app
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;
    use std::sync::{Arc, Mutex};

    use prism_ecs::command::Commands;
    use prism_ecs::resource::Resource;
    use prism_ecs::schedule::{ContinueStop, IntoSystemConfigs};
    use prism_ecs::system::{Res, ResMut};

    use crate::frame::{FrameApp, FrameContinueStop, FrameStepReport};
    use crate::{Plugin, SubApp};

    #[derive(Default)]
    struct Counts {
        startup: usize,
        updates: usize,
        deferred: usize,
        gates: usize,
        shutdown: usize,
    }
    impl Resource for Counts {}

    #[derive(Default)]
    struct Trace(Vec<&'static str>);
    impl Resource for Trace {}

    #[derive(Default)]
    struct CompleteFrameCounts {
        update: usize,
        fixed: usize,
        first: usize,
        second: usize,
    }
    impl Resource for CompleteFrameCounts {}

    #[derive(Clone)]
    struct FrameEventLog(Arc<Mutex<Vec<&'static str>>>);
    impl Resource for FrameEventLog {}

    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    struct DebugPass(u8);

    fn startup(mut counts: ResMut<Counts>) {
        counts.startup += 1;
    }
    fn update(mut counts: ResMut<Counts>) {
        counts.updates += 1;
    }

    fn complete_frame_update(mut counts: ResMut<CompleteFrameCounts>) {
        counts.update += 1;
    }

    fn complete_frame_fixed(mut counts: ResMut<CompleteFrameCounts>, clocks: Res<EngineClocks>) {
        assert_eq!(clocks.source(), prism_time::DefaultSource::Fixed);
        assert_eq!(clocks.default_time().delta(), Duration::from_millis(10));
        counts.fixed += 1;
    }

    fn complete_frame_first(mut counts: ResMut<CompleteFrameCounts>) {
        counts.first += 1;
    }

    fn complete_frame_second(mut counts: ResMut<CompleteFrameCounts>) {
        counts.second += 1;
    }
    fn shutdown(mut counts: ResMut<Counts>) {
        counts.shutdown += 1;
    }
    fn configured(app: &mut App) {
        app.init_resource::<Counts>()
            .add_systems(Startup, startup)
            .add_systems(Update, update)
            .add_systems(Shutdown, shutdown);
    }

    fn elapsed(instance: &PieInstance) -> Duration {
        instance
            .app()
            .world()
            .resource::<EngineClocks>()
            .real()
            .elapsed()
    }

    #[test]
    fn isolated_worlds_ids_modes_and_no_implicit_pool() {
        let mut instances = PieInstances::new();
        let first = instances
            .create(|app| {
                assert_eq!(*app.world().resource::<RunMode>(), RunMode::EditorEmbedded);
                #[cfg(feature = "multi_thread")]
                assert!(app.task_pool().is_none());
                configured(app);
            })
            .unwrap();
        let second = instances.create(configured).unwrap();
        let mut foreign = PieInstances::new();
        let foreign_id = foreign.create(configured).unwrap();
        assert_ne!(first, second);
        assert_ne!(first, foreign_id);
        assert!(
            matches!(instances.get(foreign_id), Err(PieError::UnknownInstance(id)) if id == foreign_id)
        );
        assert_eq!(instances.ids().collect::<Vec<_>>(), [first, second]);
        instances
            .get_mut(first)
            .unwrap()
            .step_frame(Duration::from_millis(10))
            .unwrap();
        assert_eq!(
            instances
                .get(first)
                .unwrap()
                .app()
                .world()
                .resource::<Counts>()
                .updates,
            1
        );
        assert_eq!(
            instances
                .get(second)
                .unwrap()
                .app()
                .world()
                .resource::<Counts>()
                .updates,
            0
        );
        assert_eq!(elapsed(instances.get(second).unwrap()), Duration::ZERO);
        let removed = instances.remove(first).unwrap();
        assert_eq!(removed.id(), first);
        assert_eq!(removed.state(), PieState::Stopped(AppExit::Success));
        assert!(matches!(
            instances.get(first),
            Err(PieError::UnknownInstance(_))
        ));
        let replacement = instances.create(configured).unwrap();
        assert_ne!(replacement, first);
        assert_eq!(instances.len(), 2);
    }

    #[test]
    fn paused_frames_discard_delta_and_startup_runs_once() {
        let mut instances = PieInstances::new();
        let id = instances.create(configured).unwrap();
        let instance = instances.get_mut(id).unwrap();
        assert_eq!(
            instance.update(Duration::from_secs(100)).unwrap(),
            PieFrameOutcome::Paused
        );
        assert_eq!(instance.app().world().resource::<Counts>().startup, 0);
        assert_eq!(elapsed(instance), Duration::ZERO);
        assert_eq!(
            instance.step_frame(Duration::from_millis(10)).unwrap(),
            PieFrameOutcome::Updated
        );
        assert_eq!(instance.state(), PieState::Paused);
        instance.resume().unwrap();
        assert_eq!(
            instance.step_frame(Duration::ZERO),
            Err(PieError::MustBePaused)
        );
        assert!(matches!(instance.app_mut(), Err(PieError::MustBePaused)));
        instance.update(Duration::from_millis(20)).unwrap();
        instance.pause().unwrap();
        instance.update(Duration::from_secs(100)).unwrap();
        instance.resume().unwrap();
        instance.update(Duration::from_millis(30)).unwrap();
        instance.pause().unwrap();
        assert_eq!(elapsed(instance), Duration::from_millis(60));
        let counts = instance.app().world().resource::<Counts>();
        assert_eq!((counts.startup, counts.updates), (1, 3));
    }

    #[test]
    fn frame_step_preserves_virtual_pause_policy() {
        let mut instances = PieInstances::new();
        let id = instances.create(configured).unwrap();
        let instance = instances.get_mut(id).unwrap();
        instance
            .app_mut()
            .unwrap()
            .world_mut()
            .resource_mut::<EngineClocks>()
            .virtual_time_mut()
            .pause();
        instance.step_frame(Duration::from_millis(20)).unwrap();
        let clocks = instance.app().world().resource::<EngineClocks>();
        assert!(clocks.virtual_time().is_paused());
        assert_eq!(clocks.real().elapsed(), Duration::from_millis(20));
        assert_eq!(clocks.virtual_time().elapsed(), Duration::ZERO);
        assert_eq!(instance.app().world().resource::<Counts>().updates, 1);
    }

    struct ReadyGate(Arc<AtomicBool>);
    impl Plugin for ReadyGate {
        fn build(&self, app: &mut App) {
            assert_eq!(*app.world().resource::<RunMode>(), RunMode::EditorEmbedded);
        }
        fn ready(&self, _app: &App) -> bool {
            self.0.load(Ordering::Relaxed)
        }
    }

    #[test]
    fn not_ready_neither_starts_nor_consumes_a_frame() {
        let ready = Arc::new(AtomicBool::new(false));
        let mut instances = PieInstances::new();
        let id = instances
            .create(|app| {
                configured(app);
                app.add_plugins(ReadyGate(ready.clone()));
            })
            .unwrap();
        let instance = instances.get_mut(id).unwrap();
        assert_eq!(
            instance.step_frame(Duration::from_secs(20)).unwrap(),
            PieFrameOutcome::WaitingForPlugins
        );
        assert_eq!(
            instance.debug_schedule(Update),
            Err(PieError::PluginsNotReady)
        );
        assert_eq!(instance.app().world().resource::<Counts>().startup, 0);
        assert_eq!(elapsed(instance), Duration::ZERO);
        ready.store(true, Ordering::Relaxed);
        instance.step_frame(Duration::from_millis(5)).unwrap();
        assert_eq!(instance.app().world().resource::<Counts>().startup, 1);
        assert_eq!(elapsed(instance), Duration::from_millis(5));
    }

    #[test]
    fn complete_frame_keeps_fixed_phases_and_secondary_extract_order() {
        let mut instances = PieInstances::new();
        let id = instances
            .create(|app| {
                app.init_resource::<Trace>()
                    .set_fixed_timestep(Duration::from_millis(10));
                macro_rules! phase {
                    ($label:expr, $name:literal) => {
                        app.add_systems($label, |mut trace: ResMut<Trace>| trace.0.push($name));
                    };
                }
                phase!(PreStartup, "pre-start");
                phase!(Startup, "start");
                phase!(DebugPass(1), "custom-start");
                app.insert_phase_after(Startup, DebugPass(1)).unwrap();
                phase!(PostStartup, "post-start");
                phase!(First, "first");
                phase!(BeforeFixedMainLoop, "before-fixed");
                phase!(FixedFirst, "fixed-first");
                phase!(FixedPreUpdate, "fixed-pre");
                phase!(FixedUpdate, "fixed");
                phase!(FixedPostUpdate, "fixed-post");
                phase!(FixedLast, "fixed-last");
                phase!(AfterFixedMainLoop, "after-fixed");
                app.add_systems(FixedUpdate, |clocks: Res<EngineClocks>| {
                    assert_eq!(clocks.source(), prism_time::DefaultSource::Fixed);
                    assert_eq!(clocks.default_time().delta(), Duration::from_millis(10));
                });
                phase!(PreUpdate, "pre");
                phase!(StateTransition, "state");
                phase!(Update, "update");
                phase!(DebugPass(2), "custom-frame");
                app.insert_phase_after(Update, DebugPass(2)).unwrap();
                phase!(PostUpdate, "post");
                phase!(Last, "last");
                let mut secondary = SubApp::new();
                secondary.world.insert_resource(Trace::default());
                let mut startup = Schedule::new();
                startup.add_systems(|mut trace: ResMut<Trace>| trace.0.push("secondary-start"));
                secondary
                    .world
                    .resource_mut::<Schedules>()
                    .insert(Startup, startup);
                secondary.set_extract(|main, world| {
                    assert_eq!(main.resource::<Trace>().0.last(), Some(&"last"));
                    world.resource_mut::<Trace>().0.push("extract");
                });
                let mut update = Schedule::new();
                update.add_systems(|mut trace: ResMut<Trace>| trace.0.push("secondary-update"));
                secondary
                    .world
                    .resource_mut::<Schedules>()
                    .insert(Update, update);
                app.insert_sub_app("preview", secondary);
            })
            .unwrap();
        let instance = instances.get_mut(id).unwrap();
        instance.step_frame(Duration::from_millis(20)).unwrap();
        let clocks = instance.app().world().resource::<EngineClocks>();
        assert_eq!(clocks.source(), prism_time::DefaultSource::Virtual);
        assert_eq!(clocks.default_time().delta(), Duration::from_millis(20));
        assert_eq!(
            instance.app().world().resource::<Trace>().0,
            [
                "pre-start",
                "start",
                "custom-start",
                "post-start",
                "first",
                "before-fixed",
                "fixed-first",
                "fixed-pre",
                "fixed",
                "fixed-post",
                "fixed-last",
                "fixed-first",
                "fixed-pre",
                "fixed",
                "fixed-post",
                "fixed-last",
                "after-fixed",
                "pre",
                "state",
                "update",
                "custom-frame",
                "post",
                "last",
            ]
        );
        assert_eq!(
            instance
                .app()
                .get_sub_app("preview")
                .unwrap()
                .world
                .resource::<Trace>()
                .0,
            ["secondary-start", "extract", "secondary-update"]
        );
    }

    #[test]
    fn complete_frame_step_advances_clock_once_and_runs_each_fixed_tick_once() {
        let mut instances = PieInstances::new();
        let id = instances
            .create(|app| {
                app.init_resource::<CompleteFrameCounts>()
                    .set_fixed_timestep(Duration::from_millis(10))
                    .add_systems(Update, complete_frame_update)
                    .add_systems(FixedUpdate, complete_frame_fixed);
            })
            .unwrap();
        let instance = instances.get_mut(id).unwrap();

        instance.begin_frame(Duration::from_millis(20)).unwrap();
        assert_eq!(elapsed(instance), Duration::from_millis(20));

        let mut reports = Vec::new();
        loop {
            let report = instance.step_frame_system().unwrap();
            let complete = matches!(report, FrameStepReport::Complete);
            reports.push(report);
            assert_eq!(elapsed(instance), Duration::from_millis(20));
            if complete {
                break;
            }
        }

        let counts = instance.app().world().resource::<CompleteFrameCounts>();
        assert_eq!(counts.update, 1);
        assert_eq!(counts.fixed, 2);
        assert_eq!(
            reports
                .iter()
                .filter(|report| matches!(report, FrameStepReport::Ran(_)))
                .count(),
            3
        );
        instance.end_frame().unwrap();
        assert_eq!(elapsed(instance), Duration::from_millis(20));
    }

    #[test]
    fn complete_frame_continue_does_not_replay_prefix_and_breakpoint_is_before_execution() {
        let mut instances = PieInstances::new();
        let id = instances
            .create(|app| {
                app.init_resource::<CompleteFrameCounts>().add_systems(
                    Update,
                    (complete_frame_first, complete_frame_second).chain(),
                );
            })
            .unwrap();
        let instance = instances.get_mut(id).unwrap();

        instance.begin_frame(Duration::from_millis(1)).unwrap();
        let first = match instance.step_frame_system().unwrap() {
            FrameStepReport::Ran(system) => system,
            report => panic!("expected first system to run, got {report:?}"),
        };
        assert_eq!(first.app, FrameApp::Main);

        let second = FrameSystem {
            app: first.app,
            phase: first.phase,
            node: first.node + 1,
        };
        assert!(instance.add_frame_breakpoint(second).unwrap());

        let stopped = instance.continue_frame().unwrap();
        assert_eq!(stopped.ran, Vec::<FrameSystem>::new());
        assert_eq!(stopped.stop, FrameContinueStop::Breakpoint(second));
        let counts = instance.app().world().resource::<CompleteFrameCounts>();
        assert_eq!((counts.first, counts.second), (1, 0));

        assert!(instance.remove_frame_breakpoint(second).unwrap());
        let completed = instance.continue_frame().unwrap();
        assert_eq!(completed.ran, [second]);
        assert_eq!(completed.stop, FrameContinueStop::Complete);
        let counts = instance.app().world().resource::<CompleteFrameCounts>();
        assert_eq!((counts.first, counts.second), (1, 1));
        instance.end_frame().unwrap();
    }

    #[test]
    fn complete_frame_reports_secondary_extracts_in_insertion_order() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut instances = PieInstances::new();
        let id = instances
            .create({
                let log = log.clone();
                move |app| {
                    app.insert_resource(FrameEventLog(log.clone()));
                    app.add_systems(Last, |events: ResMut<FrameEventLog>| {
                        events.0.lock().unwrap().push("main-last");
                    });

                    let mut first = SubApp::new();
                    first.world.insert_resource(FrameEventLog(log.clone()));
                    let mut startup = Schedule::new();
                    startup.add_systems(|events: ResMut<FrameEventLog>| {
                        events.0.lock().unwrap().push("secondary-0-start");
                    });
                    first
                        .world
                        .resource_mut::<Schedules>()
                        .insert(Startup, startup);
                    first.set_extract(|_main, world| {
                        world
                            .resource_mut::<FrameEventLog>()
                            .0
                            .lock()
                            .unwrap()
                            .push("extract-0");
                    });
                    let mut update = Schedule::new();
                    update.add_systems(|events: ResMut<FrameEventLog>| {
                        events.0.lock().unwrap().push("secondary-0-update");
                    });
                    first
                        .world
                        .resource_mut::<Schedules>()
                        .insert(Update, update);
                    app.insert_sub_app("preview-0", first);

                    let mut second = SubApp::new();
                    second.world.insert_resource(FrameEventLog(log.clone()));
                    let mut startup = Schedule::new();
                    startup.add_systems(|events: ResMut<FrameEventLog>| {
                        events.0.lock().unwrap().push("secondary-1-start");
                    });
                    second
                        .world
                        .resource_mut::<Schedules>()
                        .insert(Startup, startup);
                    second.set_extract(|_main, world| {
                        world
                            .resource_mut::<FrameEventLog>()
                            .0
                            .lock()
                            .unwrap()
                            .push("extract-1");
                    });
                    let mut update = Schedule::new();
                    update.add_systems(|events: ResMut<FrameEventLog>| {
                        events.0.lock().unwrap().push("secondary-1-update");
                    });
                    second
                        .world
                        .resource_mut::<Schedules>()
                        .insert(Update, update);
                    app.insert_sub_app("preview-1", second);
                }
            })
            .unwrap();
        let instance = instances.get_mut(id).unwrap();

        instance.begin_frame(Duration::from_millis(1)).unwrap();
        log.lock().unwrap().clear();

        let mut reports = Vec::new();
        loop {
            let report = instance.step_frame_system().unwrap();
            let complete = matches!(report, FrameStepReport::Complete);
            reports.push(report);
            if complete {
                break;
            }
        }

        assert_eq!(
            *log.lock().unwrap(),
            [
                "main-last",
                "extract-0",
                "secondary-0-update",
                "extract-1",
                "secondary-1-update",
            ]
        );
        let extract_0 = reports
            .iter()
            .position(|report| *report == FrameStepReport::Extracted(0))
            .unwrap();
        let secondary_0 = reports
            .iter()
            .position(|report| {
                matches!(
                    report,
                    FrameStepReport::Ran(FrameSystem {
                        app: FrameApp::Secondary(0),
                        ..
                    })
                )
            })
            .unwrap();
        let extract_1 = reports
            .iter()
            .position(|report| *report == FrameStepReport::Extracted(1))
            .unwrap();
        let secondary_1 = reports
            .iter()
            .position(|report| {
                matches!(
                    report,
                    FrameStepReport::Ran(FrameSystem {
                        app: FrameApp::Secondary(1),
                        ..
                    })
                )
            })
            .unwrap();
        assert!(extract_0 < secondary_0 && secondary_0 < extract_1 && extract_1 < secondary_1);
        instance.end_frame().unwrap();
    }

    #[test]
    fn complete_frame_abort_preserves_prefix_and_end_frame_requires_completion() {
        let mut instances = PieInstances::new();
        let id = instances
            .create(|app| {
                app.init_resource::<CompleteFrameCounts>().add_systems(
                    Update,
                    (complete_frame_first, complete_frame_second).chain(),
                );
            })
            .unwrap();
        let instance = instances.get_mut(id).unwrap();

        instance.begin_frame(Duration::from_millis(5)).unwrap();
        assert!(matches!(
            instance.step_frame_system().unwrap(),
            FrameStepReport::Ran(_)
        ));
        assert_eq!(instance.end_frame(), Err(PieError::IncompleteFrame));
        assert_eq!(instance.abort_frame(), Ok(()));
        let counts = instance.app().world().resource::<CompleteFrameCounts>();
        assert_eq!((counts.first, counts.second), (1, 0));
        assert_eq!(instance.end_frame(), Err(PieError::NoFrameSession));
        assert_eq!(instance.step_frame_system(), Err(PieError::NoFrameSession));

        instance.begin_frame(Duration::from_millis(5)).unwrap();
        loop {
            if matches!(
                instance.step_frame_system().unwrap(),
                FrameStepReport::Complete
            ) {
                break;
            }
        }
        let counts = instance.app().world().resource::<CompleteFrameCounts>();
        assert_eq!((counts.first, counts.second), (2, 1));
        assert_eq!(elapsed(instance), Duration::from_millis(10));
        instance.end_frame().unwrap();
        assert_eq!(instance.end_frame(), Err(PieError::NoFrameSession));
    }

    fn gate(mut counts: ResMut<Counts>) -> bool {
        counts.gates += 1;
        true
    }
    fn deferred(mut commands: Commands) {
        commands.queue(|world| {
            world.resource_mut::<Counts>().deferred += 1;
        });
    }
    fn after_deferred(mut counts: ResMut<Counts>) {
        assert_eq!(counts.deferred, counts.updates + 1);
        counts.updates += 1;
    }

    #[test]
    fn persistent_debug_pass_uses_conditions_deferred_and_breakpoints() {
        let mut instances = PieInstances::new();
        let id = instances
            .create(|app| {
                configured(app);
                app.add_systems(
                    DebugPass(7),
                    (deferred, after_deferred).chain().run_if(gate),
                );
            })
            .unwrap();
        let other = instances.create(configured).unwrap();
        instances
            .get_mut(id)
            .unwrap()
            .debug_schedule(DebugPass(7))
            .unwrap();
        instances
            .get_mut(other)
            .unwrap()
            .step_frame(Duration::from_millis(10))
            .unwrap();
        let instance = instances.get_mut(id).unwrap();
        assert_eq!(instance.app().world().resource::<Counts>().startup, 1);
        assert_eq!(
            instance.debug().unwrap().label(),
            &BoxedScheduleLabel::new(DebugPass(7))
        );
        let breakpoint = instance
            .find_debug_system(std::any::type_name_of_val(&after_deferred))
            .unwrap()
            .unwrap();
        assert_eq!(breakpoint, 1);
        instance.add_breakpoint(breakpoint).unwrap();
        let first = instance.continue_schedule().unwrap();
        assert_eq!(first.ran, [0]);
        assert_eq!(first.stop, ContinueStop::Breakpoint(1));
        assert_eq!(instance.app().world().resource::<Counts>().deferred, 1);
        assert_eq!(instance.end_debug(), Err(PieError::IncompleteSchedule));
        assert_eq!(instance.resume(), Err(PieError::DebugSessionActive));
        assert!(matches!(
            instance.app_mut(),
            Err(PieError::DebugSessionActive)
        ));
        assert_eq!(
            instance.step_frame(Duration::ZERO),
            Err(PieError::DebugSessionActive)
        );
        let second = instance.continue_schedule().unwrap();
        assert_eq!(second.ran, [1]);
        assert_eq!(second.stop, ContinueStop::FrameBoundary);
        assert_eq!(instance.app().world().resource::<Counts>().gates, 1);
        assert_eq!(elapsed(instance), Duration::ZERO);
        assert_eq!(instance.finish_schedule().unwrap(), [0, 1]);
        assert_eq!(instance.app().world().resource::<Counts>().gates, 2);
        instance.end_debug().unwrap();
        // A debug pass is independent; the next normal frame does not consume
        // or skip any phase on the strength of earlier debug execution.
        instance.step_frame(Duration::from_millis(5)).unwrap();
        assert_eq!(instance.app().world().resource::<Counts>().updates, 3);
        assert_eq!(instance.app().world().resource::<Counts>().startup, 1);
    }

    #[test]
    fn step_boundary_discard_and_skipped_systems_are_explicit() {
        let mut instances = PieInstances::new();
        let id = instances
            .create(|app| {
                configured(app);
                app.add_systems(DebugPass(1), (update, update.run_if(|| false)).chain());
                app.world_mut()
                    .resource_mut::<Schedules>()
                    .insert(DebugPass(0), Schedule::new());
            })
            .unwrap();
        let instance = instances.get_mut(id).unwrap();
        instance.debug_schedule(DebugPass(1)).unwrap();
        assert_eq!(instance.step_system().unwrap(), StepReport::Ran(0));
        assert_eq!(instance.step_system().unwrap(), StepReport::Skipped(1));
        assert_eq!(instance.end_debug(), Err(PieError::IncompleteSchedule));
        assert_eq!(instance.step_system().unwrap(), StepReport::FrameBoundary);
        instance.end_debug().unwrap();
        instance.debug_schedule(Update).unwrap();
        instance.step_system().unwrap();
        instance.discard_debug().unwrap();
        assert_eq!(instance.app().world().resource::<Counts>().updates, 2);
        instance.resume().unwrap();
        instance.update(Duration::ZERO).unwrap();
        instance.pause().unwrap();
        assert_eq!(instance.app().world().resource::<Counts>().updates, 3);
        instance.debug_schedule(DebugPass(0)).unwrap();
        assert_eq!(instance.step_system().unwrap(), StepReport::FrameBoundary);
        instance.end_debug().unwrap();
    }

    #[test]
    fn invalid_debug_operations_preserve_lifecycle_and_registry() {
        let mut instances = PieInstances::new();
        let id = instances.create(configured).unwrap();
        let instance = instances.get_mut(id).unwrap();
        assert_eq!(instance.step_system(), Err(PieError::NoDebugSession));
        assert_eq!(instance.add_breakpoint(0), Err(PieError::NoDebugSession));
        assert_eq!(instance.remove_breakpoint(0), Err(PieError::NoDebugSession));
        assert_eq!(instance.clear_breakpoints(), Err(PieError::NoDebugSession));
        assert_eq!(
            instance.debug_schedule(Startup),
            Err(PieError::LifecycleSchedule(BoxedScheduleLabel::new(
                Startup
            )))
        );
        assert_eq!(
            instance.debug_schedule(Shutdown),
            Err(PieError::LifecycleSchedule(BoxedScheduleLabel::new(
                Shutdown
            )))
        );
        assert_eq!(instance.app().world().resource::<Counts>().startup, 0);
        assert_eq!(
            instance.debug_schedule(DebugPass(99)),
            Err(PieError::MissingSchedule(BoxedScheduleLabel::new(
                DebugPass(99)
            )))
        );
        instance.debug_schedule(Update).unwrap();
        assert!(instance.add_breakpoint(0).unwrap());
        assert!(!instance.add_breakpoint(0).unwrap());
        assert_eq!(
            instance.debug().unwrap().breakpoints().collect::<Vec<_>>(),
            [0]
        );
        assert!(instance.remove_breakpoint(0).unwrap());
        assert!(!instance.remove_breakpoint(0).unwrap());
        instance.add_breakpoint(0).unwrap();
        instance.clear_breakpoints().unwrap();
        assert!(instance.debug().unwrap().breakpoints().next().is_none());
        assert_eq!(
            instance.debug_schedule(Update),
            Err(PieError::DebugSessionActive)
        );
        assert_eq!(
            instance.add_breakpoint(1),
            Err(PieError::InvalidBreakpoint(1))
        );
        instance.discard_debug().unwrap();
        assert!(instance
            .app()
            .world()
            .resource::<Schedules>()
            .contains(Update));
        instance.stop();
        assert_eq!(instance.resume(), Err(PieError::Stopped(AppExit::Success)));
        assert_eq!(
            instance.step_frame(Duration::ZERO),
            Err(PieError::Stopped(AppExit::Success))
        );
    }

    #[test]
    fn schedule_lease_restores_on_unwind_without_running_teardown() {
        let mut instances = PieInstances::new();
        let id = instances.create(configured).unwrap();
        let instance = instances.get_mut(id).unwrap();
        instance.debug_schedule(Update).unwrap();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let debug = instance.debug.as_mut().unwrap();
            let _ = debug.with_schedule(instance.app.world_mut(), |_, _, _| {
                panic!("test schedule lease restoration");
            });
        }));
        assert!(result.is_err());
        assert!(instance
            .app()
            .world()
            .resource::<Schedules>()
            .contains(Update));
        assert_eq!(instance.app().world().resource::<Counts>().shutdown, 0);
        assert_eq!(instance.step_system().unwrap(), StepReport::Ran(0));
    }

    struct Teardown {
        name: &'static str,
        log: Arc<Mutex<Vec<&'static str>>>,
    }
    impl Plugin for Teardown {
        fn build(&self, _app: &mut App) {}
        fn name(&self) -> &str {
            self.name
        }
        fn shutdown(&self, app: &mut App) {
            assert_eq!(app.world().resource::<Counts>().shutdown, 1);
            self.log.lock().unwrap().push(self.name);
        }
    }

    #[test]
    fn stop_remove_and_drop_shutdown_once_in_reverse_plugin_order() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut instances = PieInstances::new();
        let id = instances
            .create(|app| {
                configured(app);
                app.add_plugins((
                    Teardown {
                        name: "first",
                        log: log.clone(),
                    },
                    Teardown {
                        name: "second",
                        log: log.clone(),
                    },
                ));
            })
            .unwrap();
        let instance = instances.get_mut(id).unwrap();
        // Stop before readiness or Startup; host stop still tears down build
        // resources, and does not let an exit-confirmation veto prevent removal.
        instance.app_mut().unwrap().add_systems(
            ExitConfirmation,
            |mut exit: ResMut<AppExitRequest>| {
                exit.cancel();
            },
        );
        assert_eq!(instance.stop(), AppExit::Success);
        assert_eq!(instance.app().world().resource::<Counts>().startup, 0);
        assert_eq!(instance.app().world().resource::<Counts>().shutdown, 1);
        instance.stop();
        drop(instances.remove(id).unwrap());
        assert!(instances.is_empty());
        assert_eq!(*log.lock().unwrap(), ["second", "first"]);
        let id = instances.create(configured).unwrap();
        // Exercise normal manager Drop separately, with a fresh instance.
        let seen = Arc::new(AtomicBool::new(false));
        let flag = seen.clone();
        instances
            .get_mut(id)
            .unwrap()
            .app_mut()
            .unwrap()
            .add_systems(Shutdown, move || {
                flag.store(true, Ordering::Relaxed);
            });
        drop(instances);
        assert!(seen.load(Ordering::Relaxed));
    }

    #[test]
    fn exit_requests_are_vetoable_and_preserve_exit_code_after_frame() {
        let mut instances = PieInstances::new();
        let id = instances
            .create(|app| {
                configured(app);
                app.add_systems(Update, |mut exit: ResMut<AppExitRequest>| {
                    exit.send(AppExit::error());
                });
                app.add_systems(
                    ExitConfirmation,
                    |counts: Res<Counts>, mut exit: ResMut<AppExitRequest>| {
                        if counts.updates == 1 {
                            exit.cancel();
                        }
                    },
                );
            })
            .unwrap();
        let instance = instances.get_mut(id).unwrap();
        assert_eq!(
            instance.step_frame(Duration::from_millis(5)).unwrap(),
            PieFrameOutcome::Updated
        );
        assert_eq!(
            instance.step_frame(Duration::from_millis(5)).unwrap(),
            PieFrameOutcome::Stopped(AppExit::error())
        );
        assert_eq!(elapsed(instance), Duration::from_millis(10));
        assert_eq!(instance.stop(), AppExit::error());
        assert_eq!(instance.app().world().resource::<Counts>().shutdown, 1);
    }

    #[test]
    fn startup_exit_skips_frame_and_stop_abandons_debug_remainder() {
        let mut instances = PieInstances::new();
        let id = instances
            .create(|app| {
                configured(app);
                app.add_systems(Startup, |mut exit: ResMut<AppExitRequest>| {
                    exit.send_success();
                });
            })
            .unwrap();
        let instance = instances.get_mut(id).unwrap();
        assert_eq!(
            instance.step_frame(Duration::from_millis(20)).unwrap(),
            PieFrameOutcome::Stopped(AppExit::Success)
        );
        assert_eq!(instance.app().world().resource::<Counts>().updates, 0);
        assert_eq!(elapsed(instance), Duration::ZERO);
        let id = instances
            .create(|app| {
                configured(app);
                app.add_systems(DebugPass(1), (update, update).chain());
            })
            .unwrap();
        let instance = instances.get_mut(id).unwrap();
        instance.debug_schedule(DebugPass(1)).unwrap();
        instance.step_system().unwrap();
        instance.stop();
        assert!(instance.debug().is_none());
        assert_eq!(instance.app().world().resource::<Counts>().updates, 1);
        assert_eq!(instance.app().world().resource::<Counts>().shutdown, 1);
    }
}
