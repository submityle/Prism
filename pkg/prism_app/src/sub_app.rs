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
//! # Pipelining is deferred
//!
//! This milestone (M3 Inc1) runs the main and secondary sub-apps **serially**
//! within one frame, which is exactly what design §23 (risk #1) prescribes:
//! "get serial extract working first, then enable `pipelined`." Running a
//! render sub-app for frame *N* in parallel with the main sub-app simulating
//! frame *N+1* (design §9, §24.3, §25.2) is M3 Inc2 and is honestly absent
//! here, not stubbed.

use prism_ecs::schedule::{ScheduleLabel, Schedules};
use prism_ecs::world::World;

use crate::time::{EngineClocks, TimeUpdateStrategy};

use crate::schedule::{
    First, Last, PostStartup, PostUpdate, PreStartup, PreUpdate, Startup, StateTransition, Update,
};
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
/// documented contract (the richer, trait-driven `ExtractComponent` pipeline of
/// ECS design §23.5 is not yet available in `prism_ecs`, so the seam is defined
/// here as a plain closure rather than faked).
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
    /// The one-way extract step pulling data in from the main world, run once
    /// per frame before [`update`](SubApp::update). `None` on the main sub-app
    /// and on any secondary sub-app that does not need to read the main world.
    extract: Option<ExtractFn>,
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
            extract: None,
        }
    }

    /// Set this sub-app's one-way [`ExtractFn`], replacing any previous one.
    pub fn set_extract<F>(&mut self, extract: F) -> &mut Self
    where
        F: FnMut(&mut World, &mut World) + Send + Sync + 'static,
    {
        self.extract = Some(Box::new(extract));
        self
    }

    /// Whether this sub-app has an extract step installed.
    #[must_use]
    pub fn has_extract(&self) -> bool {
        self.extract.is_some()
    }

    /// Run this sub-app's extract step (if any) against `main_world`.
    ///
    /// A no-op when no [`ExtractFn`] is installed, so a secondary sub-app that
    /// does not read the main world still participates in the frame.
    pub fn run_extract(&mut self, main_world: &mut World) {
        if let Some(extract) = self.extract.as_mut() {
            extract(main_world, &mut self.world);
        }
    }

    /// Run the schedule registered under `label` against this sub-app's world.
    ///
    /// A missing label is a no-op: a phase with no schedule simply does
    /// nothing. This keeps the frame loop total even when a user never adds
    /// systems to, say, `PreUpdate`.
    pub fn run_schedule(&mut self, label: impl ScheduleLabel) {
        self.world.run_schedule(label);
    }

    /// Run the startup phases exactly once, in order
    /// (`PreStartup → Startup → PostStartup`, design §7).
    pub fn run_startup(&mut self) {
        self.run_schedule(PreStartup);
        self.run_schedule(Startup);
        self.run_schedule(PostStartup);
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
        // Opt-in observability (design §16): when a `FrameDiagnostics` resource
        // is present on this world, run the timed path; otherwise run the plain
        // phase sequence so an un-observed frame pays nothing.
        #[cfg(feature = "std")]
        if self
            .world
            .get_resource::<crate::diagnostics::FrameDiagnostics>()
            .is_some()
        {
            self.update_instrumented();
            return;
        }
        self.update_phases();
    }

    /// The plain, un-instrumented per-frame phase sequence (design §7). This is
    /// the hot path when no [`FrameDiagnostics`](crate::diagnostics::FrameDiagnostics)
    /// is installed, and the body the instrumented path mirrors.
    fn update_phases(&mut self) {
        crate::time::advance_time(&mut self.world);
        self.run_schedule(First);
        crate::fixed::run_fixed_main_loop(&mut self.world);
        self.run_schedule(PreUpdate);
        self.run_schedule(StateTransition);
        self.run_schedule(Update);
        self.run_schedule(PostUpdate);
        self.run_schedule(Last);
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
        crate::time::advance_time(&mut self.world);

        self.run_phase_timed(First, "First");

        let fixed_start = Instant::now();
        crate::fixed::run_fixed_main_loop(&mut self.world);
        let fixed_elapsed = fixed_start.elapsed();
        if let Some(diag) = self.world.get_resource_mut::<FrameDiagnostics>() {
            diag.record_phase("RunFixedMainLoop", fixed_elapsed);
        }

        self.run_phase_timed(PreUpdate, "PreUpdate");
        self.run_phase_timed(StateTransition, "StateTransition");
        self.run_phase_timed(Update, "Update");
        self.run_phase_timed(PostUpdate, "PostUpdate");
        self.run_phase_timed(Last, "Last");

        let frame_elapsed = frame_start.elapsed();
        if let Some(diag) = self.world.get_resource_mut::<FrameDiagnostics>() {
            diag.record_frame(frame_elapsed);
        }
    }

    /// Run one phase schedule, recording its wall-clock duration into the
    /// present [`FrameDiagnostics`](crate::diagnostics::FrameDiagnostics).
    #[cfg(feature = "std")]
    fn run_phase_timed(&mut self, label: impl ScheduleLabel, phase: &'static str) {
        let start = std::time::Instant::now();
        self.run_schedule(label);
        let elapsed = start.elapsed();
        if let Some(diag) = self
            .world
            .get_resource_mut::<crate::diagnostics::FrameDiagnostics>()
        {
            diag.record_phase(phase, elapsed);
        }
    }
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
