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

use crate::schedule::{
    First, Last, PostStartup, PostUpdate, PreStartup, PreUpdate, StateTransition, Startup, Update,
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

    /// Run one frame in the full main-frame order (design §7, §21 invariant):
    /// `First → RunFixedMainLoop → PreUpdate → StateTransition → Update →
    /// PostUpdate → Last`.
    ///
    /// Before `First`, [`advance_time`](crate::time::advance_time) steps the
    /// clocks (real → virtual → fixed accumulator). `RunFixedMainLoop` then
    /// drains the fixed accumulator via
    /// [`run_fixed_main_loop`](crate::fixed::run_fixed_main_loop), running the
    /// `FixedMain` tick group once per fixed step. A sub-app without an
    /// [`EngineClocks`](crate::time::EngineClocks) resource simply skips the
    /// time and fixed-loop steps, so a clock-less secondary sub-app still runs
    /// its variable-step phases.
    pub fn update(&mut self) {
        crate::time::advance_time(&mut self.world);
        self.run_schedule(First);
        crate::fixed::run_fixed_main_loop(&mut self.world);
        self.run_schedule(PreUpdate);
        self.run_schedule(StateTransition);
        self.run_schedule(Update);
        self.run_schedule(PostUpdate);
        self.run_schedule(Last);
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
