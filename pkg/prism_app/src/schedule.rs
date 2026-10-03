//! The core phase labels an [`App`](crate::app::App) runs each startup and each frame.
//!
//! # Reusing the `prism_ecs` schedule graph
//!
//! Unlike M0 (which hand-rolled its own label id + registry), a [`SubApp`](crate::sub_app::SubApp)
//! now stores its phase schedules directly in the world-owned
//! [`Schedules`](prism_ecs::schedule::Schedules) resource and runs them by
//! label through [`World::run_schedule`](prism_ecs::world::World::run_schedule)
//! (design §5: *"Schedule reuses the `prism_ecs` scheduling graph"*). Each
//! core phase below is a tiny unit struct that derives the standard traits, so
//! the blanket [`ScheduleLabel`] impl applies
//! automatically — no bespoke id type, and user phases work the same way.
//!
//! # Fixed order invariant (design §21)
//!
//! The per-frame order is an invariant:
//! `First → RunFixedMainLoop → PreUpdate → StateTransition → Update → PostUpdate → Last`.
//! Plugins may insert *relative* to these phases but may not reorder the core
//! sequence. Startup runs exactly once before the first frame:
//! `PreStartup → Startup → PostStartup`.
//!
//! # Honestly deferred
//!
//! - `RunFixedMainLoop` (the fixed-timestep inner loop, design §8) needs
//!   `prism_time` and lands in **M2**. It is intentionally **absent** from the
//!   frame order below rather than stubbed as an empty phase that silently does
//!   nothing — adding it before its accumulator machinery exists would be a
//!   hollow placeholder.
//! - **Dynamic phase insertion** (design §7: a plugin inserting a new phase
//!   *before/after* a core one, which requires an ordered list of boxed phase
//!   labels) is not yet possible through the public API:
//!   [`World::run_schedule`](prism_ecs::world::World::run_schedule) only accepts a
//!   *concrete* label and the boxed-label running path is `pub(crate)` inside
//!   `prism_ecs`. M1 therefore drives the fixed core order via explicit
//!   per-phase `run_schedule` calls; extensible phase ordering is tracked for a
//!   later milestone and is documented as absent, not faked.

use prism_ecs::schedule::ScheduleLabel;

/// Define a zero-sized core-phase label type that auto-implements
/// [`ScheduleLabel`] via the standard derives.
macro_rules! core_phase {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
        pub struct $name;
    };
}

core_phase! {
    /// Runs once, first of all, before [`Startup`].
    PreStartup
}
core_phase! {
    /// Runs once at boot: one-time world setup (spawn initial entities, insert
    /// resources).
    Startup
}
core_phase! {
    /// Runs once, after [`Startup`].
    PostStartup
}
core_phase! {
    /// First phase of every frame.
    First
}
core_phase! {
    /// Before the main [`Update`] work each frame.
    PreUpdate
}
core_phase! {
    /// Applies queued state-machine transitions
    /// ([`apply_state_transition`](prism_ecs::schedule::apply_state_transition)),
    /// between [`PreUpdate`] and [`Update`] (design §7, §11).
    StateTransition
}
core_phase! {
    /// The main per-frame, variable-step work (gameplay, input sampling,
    /// camera smoothing).
    Update
}
core_phase! {
    /// After the main [`Update`] work each frame.
    PostUpdate
}
core_phase! {
    /// Final phase of every frame (frame cleanup / bookkeeping).
    Last
}

/// The startup phases, in run order. Driven exactly once by
/// [`App::run`](crate::app::App::run) before the first frame.
///
/// These are runtime markers for documentation and tests; the actual run is a
/// straight-line sequence of concrete
/// [`World::run_schedule`](prism_ecs::world::World::run_schedule) calls in
/// [`SubApp::run_startup`](crate::sub_app::SubApp::run_startup) (labels are
/// distinct types and cannot share one array).
pub const STARTUP_PHASES: [&str; 3] = ["PreStartup", "Startup", "PostStartup"];

/// The per-frame phases, in run order (design §7). `RunFixedMainLoop` (M2) is
/// honestly absent; see the module docs.
pub const FRAME_PHASES: [&str; 6] =
    ["First", "PreUpdate", "StateTransition", "Update", "PostUpdate", "Last"];

// Compile-time proof that every core phase satisfies the ECS `ScheduleLabel`
// bound, so a drift in the blanket impl's requirements fails the build here
// rather than at every use site.
const _: fn() = || {
    fn assert_label<L: ScheduleLabel>() {}
    assert_label::<PreStartup>();
    assert_label::<Startup>();
    assert_label::<PostStartup>();
    assert_label::<First>();
    assert_label::<PreUpdate>();
    assert_label::<StateTransition>();
    assert_label::<Update>();
    assert_label::<PostUpdate>();
    assert_label::<Last>();
};
