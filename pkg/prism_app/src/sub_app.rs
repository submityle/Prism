//! A [`SubApp`]: one [`World`] whose world-owned
//! [`Schedules`] resource holds the phase schedules that drive it, plus an
//! optional one-way [`ExtractFn`] that copies data in from the main world.
//!
//! An [`App`](crate::app::App) owns a [`SubApps`] collection: a *main*
//! sub-app carrying the authoritative simulation world, and an ordered set of
//! labeled *secondary* sub-apps (design §5). The archetypal secondary sub-app
//! is a render sub-app fed each frame by a one-way extract step
//! (design §9, §25.2): after the main sub-app updates, every secondary sub-app
//! runs its extract (`main world → sub world`, read-only w.r.t. authoritative
//! state) and then updates its own world.
//!
//! Per design §5 a sub-app **reuses the `prism_ecs` scheduling graph**: the
//! schedules live inside the world (not a separate hand-rolled registry), and
//! each phase is run by label through
//! [`World::run_schedule`](prism_ecs::world::World::run_schedule).
//!
//! The default path extracts and updates serially. The `pipelined` feature
//! also supports overlapping a secondary world's frame N with the main world's
//! frame N+1; extraction remains on the main thread between complete frames.

use alloc::boxed::Box;
use alloc::vec::Vec;

use crate::rollback::SnapshotHistory;
use prism_ecs::schedule::{
    ContinueReport, Schedule, ScheduleLabel, Schedules, StepReport, Stepping,
};
use prism_ecs::world::World;

use crate::time::{EngineClocks, TimeUpdateStrategy};

use crate::phase_order::{PhaseOrder, PhaseOrderError};
use crate::sub_app_label::{BoxedSubAppLabel, SubAppLabel};

/// The one-way simulation → sub-app data pump (design §9, §25.2).
///
/// Called once per frame per secondary sub-app, *before* that sub-app updates,
/// with `(main_world, sub_world)`. It is the **only** sanctioned seam from the
/// authoritative simulation world into a secondary (e.g. render) world.
///
/// # Invariant (design §21: "extract 单向只读")
///
/// Extraction is **one-way**: it reads from the main world and writes into the
/// sub-app world. The main world is passed by `&mut` only so the closure can
/// run change-detecting queries and advance extraction bookkeeping; the
/// authoritative simulation state must **not** be mutated here, and a secondary
/// sub-app must never write back into the main world. This invariant is a
/// documented contract. Typed `prism_ecs::extract` operations can run inside
/// this closure to maintain entity mappings and changed-component extraction.
///
/// `Send + Sync` is required now even though M3 Inc1 runs serially, so the
/// contract already supports the `pipelined` refinement (M3 Inc2).
pub type ExtractFn = Box<dyn FnMut(&mut World, &mut World) + Send + Sync + 'static>;

/// A self-contained unit of simulation: a [`World`] whose
/// [`Schedules`] resource drives it, plus an optional [`ExtractFn`].
pub struct SubApp {
    /// The ECS world this sub-app simulates. Its [`Schedules`] resource owns
    /// the phase schedules.
    pub world: World,
    phase_order: PhaseOrder,
    /// The one-way extract steps pulling data in from the main world, run in
    /// order once per frame before [`update`](SubApp::update). Empty on the main
    /// sub-app and on any secondary sub-app that does not need to read the main
    /// world.
    ///
    /// A list rather than one closure because every renderer plugin brings its
    /// own slice of extraction (the frame record, each render-asset family, the
    /// scene), and independently authored plugins could not otherwise all be
    /// installed. [`set_extract`](SubApp::set_extract) replaces the list;
    /// [`add_extract`](SubApp::add_extract) appends to it.
    extracts: Vec<ExtractFn>,
    #[cfg(feature = "std")]
    frame_budget_started: Option<std::time::Instant>,
}

impl Default for SubApp {
    fn default() -> Self {
        Self::new()
    }
}

impl SubApp {
    /// Create a sub-app with a fresh [`World`] holding an empty
    /// [`Schedules`] resource and no extract step.
    pub fn new() -> Self {
        let mut world = World::new();
        world.init_resource::<Schedules>();
        Self {
            world,
            phase_order: PhaseOrder::default(),
            extracts: Vec::new(),
            #[cfg(feature = "std")]
            frame_budget_started: None,
        }
    }

    /// Set this sub-app's one-way [`ExtractFn`], replacing any previous ones.
    pub fn set_extract<F>(&mut self, extract: F) -> &mut Self
    where
        F: FnMut(&mut World, &mut World) + Send + Sync + 'static,
    {
        self.extracts.clear();
        self.extracts.push(Box::new(extract));
        self
    }

    /// Append a one-way [`ExtractFn`], keeping the ones already installed.
    ///
    /// Steps run in the order they were added, so a plugin that needs another
    /// plugin's extraction first declares that dependency instead of relying on
    /// registration order.
    pub fn add_extract<F>(&mut self, extract: F) -> &mut Self
    where
        F: FnMut(&mut World, &mut World) + Send + Sync + 'static,
    {
        self.extracts.push(Box::new(extract));
        self
    }

    /// Whether this sub-app has at least one extract step installed.
    #[must_use]
    pub fn has_extract(&self) -> bool {
        !self.extracts.is_empty()
    }

    /// Run this sub-app's extract steps (in order) against `main_world`.
    ///
    /// A no-op when none is installed, so a secondary sub-app that does not read
    /// the main world still participates in the frame.
    pub fn run_extract(&mut self, main_world: &mut World) {
        for extract in &mut self.extracts {
            extract(main_world, &mut self.world);
        }
    }

    /// Run the schedule registered under `label` against this sub-app's world.
    ///
    /// A missing label is a no-op: a phase with no schedule simply does
    /// nothing. This keeps the frame loop total even when a user never adds
    /// systems to, say, `PreUpdate`.
    ///
    /// Under the `multi_thread` feature, a world carrying a
    /// [`TaskPoolResource`](crate::tasks::TaskPoolResource) dispatches the
    /// schedule's conflict-free system waves across that pool (design §8.2);
    /// otherwise (or with the feature off) it runs single-threaded.
    pub fn run_schedule(&mut self, label: impl ScheduleLabel) {
        #[cfg(feature = "multi_thread")]
        if let Some(pool) = self
            .world
            .get_resource::<crate::tasks::TaskPoolResource>()
            .map(|resource| resource.0.clone())
        {
            self.world.run_schedule_parallel(label, &pool);
            return;
        }
        self.world.run_schedule(label);
    }

    /// Number of variable-frame phases, including the native fixed-loop slot.
    pub(crate) fn frame_phase_len(&self) -> usize {
        self.phase_order.frame.len()
    }

    /// Whether a variable-frame phase is the native fixed-loop slot.
    pub(crate) fn frame_phase_is_fixed_driver(&self, index: usize) -> bool {
        self.phase_order
            .frame
            .get(index)
            .is_some_and(|phase| phase.is_fixed_driver)
    }

    /// Step one system in a configured variable-frame phase.
    pub(crate) fn step_frame_phase(&mut self, index: usize, stepping: &mut Stepping) -> StepReport {
        let step = self
            .phase_order
            .frame
            .get(index)
            .map(|phase| phase.step.clone())
            .unwrap_or_else(|| panic!("invalid frame phase index {index}"));
        step(self, stepping)
    }

    pub(crate) fn continue_frame_phase(
        &mut self,
        index: usize,
        stepping: &mut Stepping,
    ) -> ContinueReport {
        let continue_ = self
            .phase_order
            .frame
            .get(index)
            .map(|phase| phase.continue_.clone())
            .unwrap_or_else(|| panic!("invalid frame phase index {index}"));
        continue_(self, stepping)
    }

    // Explicit run_schedule and fixed simulation remain complete; only the
    // variable-frame driver uses this opt-in budget gate.
    pub(crate) fn run_frame_schedule(&mut self, label: impl ScheduleLabel) {
        #[cfg(feature = "std")]
        if let Some(started) = self.frame_budget_started
            && let Some(policy) = self
                .world
                .get_resource::<crate::budget::FrameScheduleBudget>()
        {
            let remaining = policy
                .budget()
                .headroom(started.elapsed())
                .unwrap_or_default();
            #[cfg(feature = "multi_thread")]
            let pool = self
                .world
                .get_resource::<crate::tasks::TaskPoolResource>()
                .map(|resource| resource.0.clone());
            #[cfg(feature = "multi_thread")]
            let deferred = if let Some(pool) = pool {
                self.world
                    .run_schedule_parallel_budgeted(label, &pool, remaining)
            } else {
                self.world.run_schedule_budgeted(label, remaining)
            };
            #[cfg(not(feature = "multi_thread"))]
            let deferred = self.world.run_schedule_budgeted(label, remaining);
            if let Some(policy) = self
                .world
                .get_resource_mut::<crate::budget::FrameScheduleBudget>()
            {
                policy.record_deferred(deferred);
            }
            return;
        }
        self.run_schedule(label);
    }

    // ---- rollback history (design §14 / M5) --------------------------------

    /// Install a bounded world-snapshot history retaining `capacity` frames,
    /// replacing any previous history (design §14, §22 M5).
    ///
    /// # Panics
    /// Panics if `capacity` is zero.
    pub fn init_snapshot_history(&mut self, capacity: usize) -> &mut Self {
        self.world.insert_resource(SnapshotHistory::new(capacity));
        self
    }

    /// This sub-app's snapshot history, if one was installed.
    #[must_use]
    pub fn snapshot_history(&self) -> Option<&SnapshotHistory> {
        self.world.get_resource::<SnapshotHistory>()
    }

    /// Capture this sub-app's world at the next history frame.
    ///
    /// Returns `None` when no history is installed.
    ///
    /// # Panics
    /// Panics (via [`SnapshotHistory::capture`]) if a resident component has no
    /// snapshot clone glue; use
    /// [`try_capture_snapshot`](SubApp::try_capture_snapshot) to report it.
    pub fn capture_snapshot(&mut self) -> Option<u64> {
        let mut history = self.world.remove_resource::<SnapshotHistory>()?;
        let frame = history.capture(&self.world);
        self.world.insert_resource(history);
        Some(frame)
    }

    /// Fallible [`capture_snapshot`](SubApp::capture_snapshot): `None` when no
    /// history is installed, otherwise `Ok(frame)` or `Err(missing component
    /// ids)`.
    pub fn try_capture_snapshot(
        &mut self,
    ) -> Option<Result<u64, Vec<prism_ecs::component::ComponentId>>> {
        let mut history = self.world.remove_resource::<SnapshotHistory>()?;
        let result = history.try_capture(&self.world);
        self.world.insert_resource(history);
        Some(result)
    }

    /// Restore this sub-app's world to `frame` (design §14).
    ///
    /// Rolling back abandons the future branch: on success the frames after
    /// `frame` are dropped and the capture cursor is rebased to `frame`, so a
    /// later [`capture_snapshot`](SubApp::capture_snapshot) records `frame` again
    /// rather than appending past the abandoned history (design §14/§15).
    ///
    /// Returns `false` when no history is installed or `frame` has been evicted
    /// or was never captured; the world is then left untouched.
    pub fn rollback_to(&mut self, frame: u64) -> bool {
        let Some(mut history) = self.world.remove_resource::<SnapshotHistory>() else {
            return false;
        };
        let restored = history.rollback(&mut self.world, frame);
        if !restored {
            self.world.insert_resource(history);
            return false;
        }
        history.truncate_after(frame);
        self.world.insert_resource(history);
        true
    }

    /// Insert `phase` immediately before `anchor` in its startup or frame
    /// sequence, preserving all core phase order (design §7).
    ///
    /// Custom phases can anchor subsequent insertions. Existing schedules are
    /// preserved; absent schedules are created empty. Duplicate phase labels
    /// (including reserved core labels) are rejected before missing anchors,
    /// without modifying either the order or schedules.
    ///
    /// Configure startup insertions before [`App::try_finish`](crate::App::try_finish):
    /// inserting later does not replay startup. Fixed tick phases, fixed hooks,
    /// and exit schedules have separate drivers and are not insertion anchors.
    /// Use [`RunFixedMainLoop`](crate::RunFixedMainLoop) to bracket the fixed loop.
    /// Custom phase timings use their concrete label's type name; values of the
    /// same label type share a diagnostics name but remain distinct schedules.
    pub fn insert_phase_before(
        &mut self,
        anchor: impl ScheduleLabel,
        phase: impl ScheduleLabel + Clone,
    ) -> Result<&mut Self, PhaseOrderError> {
        self.phase_order
            .insert(&mut self.world, anchor, phase, false)?;
        Ok(self)
    }

    /// Insert `phase` immediately after `anchor`. See
    /// [`insert_phase_before`](Self::insert_phase_before) for validation and
    /// lifecycle rules. Repeated insertions after the same anchor place the
    /// newest phase closest to it; anchor to the previous custom phase to append.
    pub fn insert_phase_after(
        &mut self,
        anchor: impl ScheduleLabel,
        phase: impl ScheduleLabel + Clone,
    ) -> Result<&mut Self, PhaseOrderError> {
        self.phase_order
            .insert(&mut self.world, anchor, phase, true)?;
        Ok(self)
    }

    /// Run the startup sequence in order, including inserted phases (design §7).
    /// [`App::try_finish`](crate::App::try_finish) gates this to once; explicitly
    /// calling this method runs the sequence each time.
    pub fn run_startup(&mut self) {
        for index in 0..self.phase_order.startup.len() {
            let run = self.phase_order.startup[index].run.clone();
            run(self);
        }
    }

    // ---- time domain (design §24.9 / §25.4) -------------------------------

    /// Whether this sub-app owns an independent time domain.
    ///
    /// A time domain here means an [`EngineClocks`] resource on this sub-app's
    /// world: [`advance_time`](crate::time::advance_time) only steps a world
    /// that owns one. The main sub-app is given a domain by
    /// [`App::new`](crate::app::App::new); a secondary sub-app has **none** by
    /// default and must opt in via [`init_time_domain`](SubApp::init_time_domain).
    ///
    /// This is the mechanism behind the design §25.4 invariant that *"each
    /// world holds an independent time context … pausing one does not freeze
    /// another"*: because every sub-app advances its **own** clocks, pausing
    /// [`Time<Virtual>`](prism_time::Time) on one world leaves every other
    /// world's clocks untouched.
    #[must_use]
    pub fn has_time_domain(&self) -> bool {
        self.world.get_resource::<EngineClocks>().is_some()
    }

    /// Give this sub-app its own independent time domain (design §24.9 / §25.4).
    ///
    /// Installs a fresh [`EngineClocks`] bundle and a default
    /// [`TimeUpdateStrategy`] on this world, so from the next frame on
    /// [`advance_time`](crate::time::advance_time) steps this sub-app's clocks
    /// independently of every other sub-app. This is what lets a secondary
    /// world (an editor-preview world, an embedded-server world) run, pause or
    /// time-dilate on its own without disturbing the main simulation.
    ///
    /// **Idempotent**: if this sub-app already owns a domain the existing
    /// clocks are left untouched, so calling it on an already-running world
    /// never rewinds elapsed time. A missing [`TimeUpdateStrategy`] is still
    /// filled in, so a domain is always fully formed afterwards.
    pub fn init_time_domain(&mut self) -> &mut Self {
        if self.world.get_resource::<EngineClocks>().is_none() {
            self.world.insert_resource(EngineClocks::new());
        }
        if self.world.get_resource::<TimeUpdateStrategy>().is_none() {
            self.world.insert_resource(TimeUpdateStrategy::default());
        }
        self
    }

    /// Set this sub-app's [`TimeUpdateStrategy`] (design §24.9 / §25.4).
    ///
    /// Controls how *this* world's real clock advances each frame, independent
    /// of other sub-apps: a secondary world can step on
    /// [`ManualDelta`](TimeUpdateStrategy::ManualDelta) while the main world
    /// paces from the wall clock, or vice versa.
    ///
    /// # Panics
    ///
    /// Panics if this sub-app has no time domain yet; call
    /// [`init_time_domain`](SubApp::init_time_domain) first. (A strategy with
    /// no clock to drive would be silently inert, so this fails loudly rather
    /// than pretending to take effect.)
    pub fn set_time_update_strategy(&mut self, strategy: TimeUpdateStrategy) -> &mut Self {
        assert!(
            self.has_time_domain(),
            "set_time_update_strategy: sub-app has no time domain; call init_time_domain first"
        );
        self.world.insert_resource(strategy);
        self
    }

    /// Set this sub-app's fixed-timestep rate in hertz (design §24.9 / §25.4).
    ///
    /// For example `30.0` runs this world's `FixedUpdate` at a 1/30 s step —
    /// an embedded server world can tick at a different rate than the client's
    /// main world.
    ///
    /// # Panics
    ///
    /// Panics if this sub-app has no time domain yet; call
    /// [`init_time_domain`](SubApp::init_time_domain) first.
    pub fn set_fixed_timestep_hz(&mut self, hz: f64) -> &mut Self {
        self.world
            .get_resource_mut::<EngineClocks>()
            .expect(
                "set_fixed_timestep_hz: sub-app has no time domain; call init_time_domain first",
            )
            .fixed_mut()
            .set_timestep_hz(hz);
        self
    }

    /// Set this sub-app's fixed-timestep period to an exact
    /// [`Duration`](prism_time::Duration) (design §24.9 / §25.4).
    ///
    /// Prefer this over [`set_fixed_timestep_hz`](SubApp::set_fixed_timestep_hz)
    /// when the step must match another duration bit-for-bit (see the note on
    /// [`App::set_fixed_timestep`](crate::app::App::set_fixed_timestep)).
    ///
    /// # Panics
    ///
    /// Panics if this sub-app has no time domain yet; call
    /// [`init_time_domain`](SubApp::init_time_domain) first.
    pub fn set_fixed_timestep(&mut self, timestep: prism_time::Duration) -> &mut Self {
        self.world
            .get_resource_mut::<EngineClocks>()
            .expect("set_fixed_timestep: sub-app has no time domain; call init_time_domain first")
            .fixed_mut()
            .set_timestep(timestep);
        self
    }

    /// Run one frame in the full main-frame order (design §7, §21 invariant):
    /// `First → RunFixedMainLoop → PreUpdate → StateTransition → Update →
    /// PostUpdate → Last`.
    ///
    /// Before `First`, [`advance_time`](crate::time::advance_time) steps the
    /// clocks (real → virtual → fixed accumulator). `RunFixedMainLoop` then
    /// drains the fixed accumulator via
    /// [`run_fixed_main_loop`](crate::fixed::run_fixed_main_loop), running the
    /// `FixedMain` tick group once per fixed step. A sub-app without an
    /// [`EngineClocks`] resource simply skips the
    /// time and fixed-loop steps, so a clock-less secondary sub-app still runs
    /// its variable-step phases.
    pub fn update(&mut self) {
        #[cfg(feature = "std")]
        {
            self.frame_budget_started = self
                .world
                .get_resource_mut::<crate::budget::FrameScheduleBudget>()
                .map(|policy| {
                    policy.begin();
                    std::time::Instant::now()
                });
        }
        // Opt-in observability (design §16): when a `FrameDiagnostics` resource
        // is present on this world, run the timed path; otherwise run the plain
        // phase sequence so an un-observed frame pays nothing.
        #[cfg(feature = "std")]
        {
            let instrument = self
                .world
                .get_resource::<crate::diagnostics::FrameDiagnostics>()
                .is_some();
            #[cfg(feature = "trace")]
            let instrument = instrument
                || self
                    .world
                    .get_resource::<crate::trace::PhaseTrace>()
                    .is_some();
            if instrument {
                self.update_instrumented();
                return;
            }
        }
        self.update_phases();
        #[cfg(feature = "std")]
        self.finish_frame_budget();
    }

    #[cfg(feature = "std")]
    fn finish_frame_budget(&mut self) {
        let started = self.frame_budget_started.take();

        #[cfg(feature = "multi_thread")]
        if let Some(qos) = self
            .world
            .get_resource::<crate::tasks::TaskQosResource>()
            .cloned()
        {
            let frame_budget = started.and_then(|frame_start| {
                self.world
                    .get_resource::<crate::budget::FrameScheduleBudget>()
                    .map(|policy| (frame_start, policy.budget()))
            });
            let report = qos.run_frame(|| {
                frame_budget.map(|(frame_start, budget)| {
                    duration_to_nanos(budget.headroom(frame_start.elapsed()).unwrap_or_default())
                })
            });
            let pending_background = qos.pending_in(prism_tasks::Priority::Background);
            if let Some(policy) = self
                .world
                .get_resource_mut::<crate::budget::FrameScheduleBudget>()
            {
                policy.record_tasks(report, pending_background);
            }
        }

        if let Some(started) = started
            && let Some(policy) = self
                .world
                .get_resource_mut::<crate::budget::FrameScheduleBudget>()
        {
            policy.finish(started.elapsed());
        }
    }

    /// Drive the shared frame order without timing (design §7).
    fn update_phases(&mut self) {
        crate::time::advance_time(&mut self.world);
        for index in 0..self.phase_order.frame.len() {
            let run = self.phase_order.frame[index].run.clone();
            run(self);
        }
    }

    /// The frame sequence wrapped in wall-clock timing, taken only when a
    /// [`FrameDiagnostics`](crate::diagnostics::FrameDiagnostics) resource is
    /// present on this world. Records the whole-frame work time plus each
    /// phase's duration; the fixed loop records its own substep count (see
    /// [`run_fixed_main_loop`](crate::fixed::run_fixed_main_loop)).
    #[cfg(feature = "std")]
    fn update_instrumented(&mut self) {
        use crate::diagnostics::FrameDiagnostics;
        use std::time::Instant;

        let frame_start = Instant::now();
        #[cfg(feature = "trace")]
        if let Some(trace) = self.world.get_resource_mut::<crate::trace::PhaseTrace>() {
            trace.begin_frame();
        }
        crate::time::advance_time(&mut self.world);

        for index in 0..self.phase_order.frame.len() {
            self.run_phase_timed(index);
        }

        self.finish_frame_budget();
        let frame_elapsed = frame_start.elapsed();
        if let Some(diag) = self.world.get_resource_mut::<FrameDiagnostics>() {
            diag.record_frame(frame_elapsed);
        }
        #[cfg(feature = "trace")]
        if let Some(trace) = self.world.get_resource_mut::<crate::trace::PhaseTrace>() {
            trace.end_frame(frame_elapsed);
        }
    }

    /// Run one phase schedule, recording its wall-clock duration into the
    /// present [`FrameDiagnostics`](crate::diagnostics::FrameDiagnostics).
    #[cfg(feature = "std")]
    fn run_phase_timed(&mut self, index: usize) {
        let phase = &self.phase_order.frame[index];
        let name = phase.name;
        let run = phase.run.clone();
        let start = std::time::Instant::now();
        run(self);
        let elapsed = start.elapsed();
        if let Some(diag) = self
            .world
            .get_resource_mut::<crate::diagnostics::FrameDiagnostics>()
        {
            diag.record_phase(name, elapsed);
        }
        #[cfg(feature = "trace")]
        if let Some(trace) = self.world.get_resource_mut::<crate::trace::PhaseTrace>() {
            trace.record_phase(name, elapsed);
        }
    }
}

#[cfg(all(feature = "std", feature = "multi_thread"))]
fn duration_to_nanos(duration: std::time::Duration) -> u64 {
    duration.as_nanos().min(u128::from(u64::MAX)) as u64
}

/// The set of sub-apps an [`App`](crate::app::App) drives: one *main* sub-app
/// plus an ordered collection of labeled *secondary* sub-apps (design §5).
///
/// Secondary sub-apps are stored in **insertion order**; each frame they run in
/// that order, so a caller controls extract/update sequencing by insertion
/// order. Each secondary sub-app runs its [`ExtractFn`] (reading the main
/// world) immediately before its own [`update`](SubApp::update).
pub struct SubApps {
    /// The main sub-app carrying the authoritative simulation world.
    pub main: SubApp,
    /// Labeled secondary sub-apps in insertion order.
    secondary: Vec<(BoxedSubAppLabel, SubApp)>,
    /// The cross-thread pipeline driver (design §9, §24.3). `Some` once a
    /// caller opts in via `enable_pipelining`;
    /// `None` means the serial path. Only present under the `pipelined`
    /// feature so a build without it carries zero pipeline state.
    #[cfg(feature = "pipelined")]
    pipeline: Option<crate::pipelined::PipelinedExecutor>,
}

impl Default for SubApps {
    fn default() -> Self {
        Self::new()
    }
}

impl SubApps {
    /// Create a [`SubApps`] with a fresh main sub-app and no secondary sub-apps.
    pub fn new() -> Self {
        Self {
            main: SubApp::new(),
            secondary: Vec::new(),
            #[cfg(feature = "pipelined")]
            pipeline: None,
        }
    }

    /// Create a [`SubApps`] wrapping an existing `main` sub-app.
    pub fn with_main(main: SubApp) -> Self {
        Self {
            main,
            secondary: Vec::new(),
            #[cfg(feature = "pipelined")]
            pipeline: None,
        }
    }

    /// Insert (or replace) a labeled secondary sub-app.
    ///
    /// A new label is appended, preserving insertion order; re-inserting an
    /// existing label replaces that sub-app **in place** (keeping its position).
    pub fn insert(&mut self, label: impl SubAppLabel, sub_app: SubApp) {
        let key = BoxedSubAppLabel::new(label);
        if let Some(slot) = self.secondary.iter_mut().find(|(k, _)| *k == key) {
            slot.1 = sub_app;
        } else {
            self.secondary.push((key, sub_app));
        }
    }

    /// Shared access to a labeled secondary sub-app.
    #[must_use]
    pub fn get(&self, label: impl SubAppLabel) -> Option<&SubApp> {
        let key = BoxedSubAppLabel::new(label);
        self.secondary
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, s)| s)
    }

    /// Mutable access to a labeled secondary sub-app.
    pub fn get_mut(&mut self, label: impl SubAppLabel) -> Option<&mut SubApp> {
        let key = BoxedSubAppLabel::new(label);
        self.secondary
            .iter_mut()
            .find(|(k, _)| *k == key)
            .map(|(_, s)| s)
    }

    /// The number of secondary sub-apps.
    #[must_use]
    pub fn len(&self) -> usize {
        self.secondary.len()
    }

    /// Whether there are no secondary sub-apps.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.secondary.is_empty()
    }

    /// Number of secondary sub-apps in frame order.
    pub(crate) fn secondary_len(&self) -> usize {
        self.secondary.len()
    }

    /// Mutable access to a secondary by stable insertion-order index.
    pub(crate) fn secondary_mut_at(&mut self, index: usize) -> Option<&mut SubApp> {
        self.secondary.get_mut(index).map(|(_, sub_app)| sub_app)
    }

    /// Run one secondary's one-way extract against the completed main world.
    pub(crate) fn run_extract_at(&mut self, index: usize) -> bool {
        let Some((_, sub_app)) = self.secondary.get_mut(index) else {
            return false;
        };
        sub_app.run_extract(&mut self.main.world);
        true
    }

    /// Run startup once for the main sub-app and then each secondary sub-app,
    /// in insertion order (design §7). Extract does not run during startup:
    /// it is the per-frame simulation → render seam, not a one-time init step.
    pub fn run_startup(&mut self) {
        self.main.run_startup();
        for (_, sub_app) in &mut self.secondary {
            sub_app.run_startup();
        }
    }

    /// Run one frame.
    ///
    /// By default this is the **serial** path (`update_serial`):
    /// update the main sub-app, then for each secondary sub-app (in insertion
    /// order) run its extract (`main world → sub world`) followed by its own
    /// update (design §9, §25.2, §23 risk #1).
    ///
    /// When the `pipelined` feature is enabled *and* a caller has opted in via
    /// `enable_pipelining`, it instead drives the
    /// cross-thread pipeline (see the `pipelined` module), overlapping a
    /// secondary's render of frame *N* with the main sub-app's simulation of
    /// frame *N+1*. The pipeline preserves the serial path's extract semantics
    /// (extract still runs on the main thread, one-way, against a complete
    /// frame), so results are identical; only the timing overlaps.
    pub fn update(&mut self) {
        // Opt-in cross-thread pipeline (design §9/§24.3): only taken when the
        // `pipelined` feature is compiled in *and* a caller enabled it. In
        // every other case this falls through to the serial path below, which
        // is therefore the default and the sole path when the feature is off.
        #[cfg(feature = "pipelined")]
        if self.pipeline.is_some() {
            // Disjoint field borrows so the executor can hold `&mut main` and
            // `&mut secondary` at once.
            let Self {
                main,
                secondary,
                pipeline,
            } = self;
            pipeline
                .as_mut()
                .expect("pipeline is Some")
                .drive(main, secondary);
            return;
        }

        self.update_serial();
    }

    /// The serial per-frame path (design §23 risk #1): update the main sub-app,
    /// then for each secondary (in insertion order) run its extract
    /// (`main world → sub world`) followed by its own update.
    ///
    /// The strict ordering — main before every secondary, and extract before
    /// each secondary's update — guarantees a secondary sub-app always reads
    /// the main world's just-finished frame, never a half-updated one. This is
    /// also the body the pipelined path reuses conceptually (its extract step
    /// runs on the main thread at the same synchronization point).
    fn update_serial(&mut self) {
        self.main.update();
        let main_world = &mut self.main.world;

        // Opt-in observability (design §16): when the main world carries a
        // `FrameDiagnostics` resource, time the extract step as one contiguous
        // block and record the total extract cost. Grouping every secondary's
        // extract before every secondary's update is behaviourally identical to
        // interleaving them — extract reads the main world and writes only its
        // own sub-world, while update touches only its own sub-world, so no
        // secondary observes another — and it mirrors the `pipelined`
        // executor's extract/update split, keeping the metric consistent across
        // both paths.
        #[cfg(feature = "std")]
        if main_world
            .get_resource::<crate::diagnostics::FrameDiagnostics>()
            .is_some()
        {
            let start = std::time::Instant::now();
            for (_, sub_app) in &mut self.secondary {
                sub_app.run_extract(main_world);
            }
            let extract = start.elapsed();
            if let Some(diag) =
                main_world.get_resource_mut::<crate::diagnostics::FrameDiagnostics>()
            {
                diag.record_extract(extract);
            }
            for (_, sub_app) in &mut self.secondary {
                sub_app.update();
            }
            return;
        }

        for (_, sub_app) in &mut self.secondary {
            sub_app.run_extract(main_world);
            sub_app.update();
        }
    }

    /// Opt into cross-thread pipelined execution (design §9, §24.3, §25.2).
    ///
    /// After this, [`update`](SubApps::update) overlaps a secondary sub-app's
    /// render of frame *N* with the main sub-app's simulation of frame *N+1*
    /// (see the `pipelined` module). Idempotent: enabling an already-pipelined
    /// collection keeps the existing in-flight state. Only available under the
    /// `pipelined` feature.
    #[cfg(feature = "pipelined")]
    pub fn enable_pipelining(&mut self) {
        if self.pipeline.is_none() {
            self.pipeline = Some(crate::pipelined::PipelinedExecutor::new());
        }
    }

    /// Opt into cross-thread pipelined execution that renders each frame on the
    /// given [`prism_tasks`] worker pool instead of a dedicated lane (design
    /// §24.3): the pipeline runs on the engine's unified work-stealing task
    /// graph. Pass a shared engine pool here. Only available under the
    /// `pipelined-tasks` feature.
    ///
    /// If a frame is already in flight on the current lane it is brought home
    /// first (via [`sync`](SubApps::sync)) before the lane is replaced, so the
    /// secondary sub-apps are never lost — only the render lane changes.
    #[cfg(feature = "pipelined-tasks")]
    pub fn enable_pipelining_with_task_pool(&mut self, pool: prism_tasks::TaskPool) {
        // The secondaries live in the current lane while a frame is in flight;
        // collect them before swapping the lane so replacing it cannot drop
        // them on the floor.
        self.sync();
        self.pipeline = Some(crate::pipelined::PipelinedExecutor::with_task_pool(pool));
    }

    /// Whether cross-thread pipelining is enabled on this collection.
    #[cfg(feature = "pipelined")]
    #[must_use]
    pub fn is_pipelined(&self) -> bool {
        self.pipeline.is_some()
    }

    /// Block until any in-flight render frame finishes, bringing the secondary
    /// sub-apps back to the calling thread so they can be inspected via
    /// [`get`](SubApps::get) / [`get_mut`](SubApps::get_mut). A no-op when
    /// pipelining is disabled or nothing is in flight.
    #[cfg(feature = "pipelined")]
    pub fn sync(&mut self) {
        if let Some(pipeline) = self.pipeline.as_mut() {
            pipeline.sync(&mut self.secondary);
        }
    }
}

/// Step one schedule while preserving the world-owned registry on both normal
/// return and panic unwind. The concrete label remains at the call site so the
/// type-erased schedule key is never double-boxed.
pub(crate) fn step_schedule<L: ScheduleLabel + Clone>(
    world: &mut World,
    label: L,
    stepping: &mut Stepping,
) -> StepReport {
    let schedule = world
        .get_resource_mut::<Schedules>()
        .and_then(|schedules| schedules.remove(label.clone()));
    let Some(schedule) = schedule else {
        return StepReport::FrameBoundary;
    };
    let mut guard = ScheduleStepGuard {
        world,
        label,
        schedule: Some(schedule),
    };
    stepping.step_system(
        guard.schedule.as_mut().expect("leased schedule"),
        guard.world,
    )
}

/// Continue a labeled schedule until its next breakpoint or frame boundary.
pub(crate) fn continue_schedule<L: ScheduleLabel + Clone>(
    world: &mut World,
    label: L,
    stepping: &mut Stepping,
) -> ContinueReport {
    let schedule = world
        .get_resource_mut::<Schedules>()
        .and_then(|schedules| schedules.remove(label.clone()));
    let Some(schedule) = schedule else {
        return ContinueReport {
            ran: Vec::new(),
            stop: prism_ecs::schedule::ContinueStop::FrameBoundary,
        };
    };
    let mut guard = ScheduleStepGuard {
        world,
        label,
        schedule: Some(schedule),
    };
    stepping.continue_frame(
        guard.schedule.as_mut().expect("leased schedule"),
        guard.world,
    )
}

struct ScheduleStepGuard<'a, L: ScheduleLabel + Clone> {
    world: &'a mut World,
    label: L,
    schedule: Option<Schedule>,
}

impl<L: ScheduleLabel + Clone> Drop for ScheduleStepGuard<'_, L> {
    fn drop(&mut self) {
        if let Some(schedule) = self.schedule.take() {
            self.world
                .resource_mut::<Schedules>()
                .insert(self.label.clone(), schedule);
        }
    }
}
