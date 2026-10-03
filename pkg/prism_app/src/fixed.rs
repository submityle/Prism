//! The fixed-timestep inner loop: the `RunFixedMainLoop` frame phase and the
//! `FixedMain` tick group (design §8, §22 M2).
//!
//! # Fix-your-timestep, driven not implemented
//!
//! The accumulator, interpolation `overstep` (alpha), and death-spiral guard
//! all live in [`Time<Fixed>`](prism_time::Time) inside `prism_time` — the
//! single source of truth (design §25.1). This module is purely the *driver*:
//! each frame it drains the accumulator and runs the `FixedMain` tick group
//! once per fixed step.
//!
//! # Where it sits in the frame
//!
//! The per-frame order (design §7, §21 — an **invariant**) is:
//! `First → RunFixedMainLoop → PreUpdate → StateTransition → Update →
//! PostUpdate → Last`. [`run_fixed_main_loop`] *is* the `RunFixedMainLoop`
//! phase: a native driver (not a user schedule) that loops
//! [`Time::<Fixed>::expend`](prism_time::Time::expend) until the accumulator is
//! drained, running the five [`FixedMain`](FIXED_MAIN_PHASES) sub-phases each
//! step.
//!
//! The step count per frame is bounded by `Time<Fixed>`'s `max_substeps`
//! (because [`accumulate`](prism_time::Time::accumulate) caps the accumulator),
//! so a long frame cannot spiral into an unbounded number of steps — it slows
//! down gracefully instead of locking up.
//!
//! # Per-frame hooks around the inner loop
//!
//! Two user schedules bracket the inner loop *within* `RunFixedMainLoop`, each
//! running exactly **once per frame** regardless of how many fixed steps the
//! accumulator yields this frame (including zero):
//!
//! - [`BeforeFixedMainLoop`] runs first, while the default clock still reads
//!   [`Virtual`](prism_time::Virtual) time. Variable-rate systems that feed the
//!   fixed simulation (buffering input the fixed sim drains) or snapshot
//!   pre-step state (storing a *previous* transform for later interpolation)
//!   attach here.
//! - [`AfterFixedMainLoop`] runs last, after the accumulator is drained and the
//!   default clock is restored to `Virtual`. The fixed clock's
//!   [`overstep`](prism_time::Time::overstep) (interpolation alpha) is final for
//!   the frame here, so render-interpolation systems attach to this schedule.
//!
//! Attach *fixed-rate* work (physics, netcode) to the per-step
//! [`FixedMain`](FIXED_MAIN_PHASES) tick group (commonly [`FixedUpdate`])
//! instead; those run once per *step*, not once per frame.
//!
//! # Honestly scoped
//!
//! The two bracket schedules deliver the same user-facing capability as Bevy's
//! `RunFixedMainLoopSystem::{BeforeFixedMainLoop, AfterFixedMainLoop}`. The one
//! remaining difference is a deliberate design choice, documented rather than
//! faked: the accumulator drain itself is a native driver *between* the two
//! hook schedules, not a system a user can reorder relative to other systems in
//! the same schedule. The bracket schedules cover the real use cases (pre-step
//! input, post-step interpolation) without exposing the drain as a reorderable
//! system; a fully system-based driver is a later refinement, not a stub.

use prism_ecs::schedule::ScheduleLabel;
use prism_ecs::world::World;
use prism_time::DefaultSource;

use crate::time::EngineClocks;

/// Define a zero-sized fixed-phase label that auto-implements
/// [`ScheduleLabel`] via the standard derives (mirrors `core_phase!` in
/// [`crate::schedule`]).
macro_rules! fixed_phase {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
        pub struct $name;
    };
}

fixed_phase! {
    /// First sub-phase of every fixed step.
    FixedFirst
}
fixed_phase! {
    /// Before the main [`FixedUpdate`] work of a fixed step.
    FixedPreUpdate
}
fixed_phase! {
    /// The main fixed-rate work of a step: deterministic simulation such as
    /// physics and networking (design §19). The common attachment point for
    /// fixed-timestep systems.
    FixedUpdate
}
fixed_phase! {
    /// After the main [`FixedUpdate`] work of a fixed step.
    FixedPostUpdate
}
fixed_phase! {
    /// Final sub-phase of every fixed step.
    FixedLast
}

/// The `FixedMain` tick group, in run order. Driven once per fixed step by
/// [`run_fixed_main_loop`].
pub const FIXED_MAIN_PHASES: [&str; 5] = [
    "FixedFirst",
    "FixedPreUpdate",
    "FixedUpdate",
    "FixedPostUpdate",
    "FixedLast",
];

fixed_phase! {
    /// Runs **once per frame**, before the fixed inner loop drains the
    /// accumulator, while the default clock still reads
    /// [`Virtual`](prism_time::Virtual) time (design §8).
    ///
    /// Unlike the per-*step* [`FixedMain`](FIXED_MAIN_PHASES) phases
    /// ([`FixedFirst`] … [`FixedLast`]), this bracket schedule fires exactly
    /// once each frame regardless of the step count — including frames that
    /// take zero fixed steps. Attach variable-rate systems that prepare input
    /// for the fixed simulation, or snapshot pre-step state for interpolation.
    BeforeFixedMainLoop
}
fixed_phase! {
    /// Runs **once per frame**, after the fixed inner loop has drained the
    /// accumulator and the default clock is restored to
    /// [`Virtual`](prism_time::Virtual) (design §8).
    ///
    /// Like [`BeforeFixedMainLoop`] it fires exactly once per frame regardless
    /// of step count. The fixed clock's
    /// [`overstep`](prism_time::Time::overstep) (interpolation alpha) is final
    /// for the frame here, so render-interpolation systems attach to this
    /// schedule.
    AfterFixedMainLoop
}

/// Run the `RunFixedMainLoop` phase: drain the fixed accumulator, running the
/// `FixedMain` tick group once per fixed step.
///
/// Preconditions: [`crate::time::advance_time`] has already fed this frame's
/// virtual delta into the fixed accumulator. This function then:
///
/// 1. Points the default clock at [`Fixed`](prism_time::Fixed) so systems in
///    the tick group read the fixed delta from the context-less clock.
/// 2. Loops [`Time::<Fixed>::expend`](prism_time::Time::expend): each successful
///    step advances the fixed clock by one timestep, re-syncs the default
///    clock, and runs `FixedFirst → … → FixedLast`.
/// 3. Restores the default clock to [`Virtual`](prism_time::Virtual) for the
///    remaining variable-step phases, leaving the fixed clock's
///    [`overstep`](prism_time::Time::overstep) as the interpolation alpha the
///    presentation layer reads.
///
/// The resource is re-read on each iteration, so a system that retunes the
/// timestep, pauses, or changes `max_substeps` mid-loop is honored on the next
/// step rather than being clobbered by a stale copy.
///
/// A world without an [`EngineClocks`] resource is a no-op.
pub fn run_fixed_main_loop(world: &mut World) {
    if world.get_resource::<EngineClocks>().is_none() {
        return;
    }

    // Per-frame before-hook (design §8): runs once, before any fixed step,
    // while the default clock still reads virtual time. See `BeforeFixedMainLoop`.
    world.run_schedule(BeforeFixedMainLoop);

    world
        .resource_mut::<EngineClocks>()
        .set_source(DefaultSource::Fixed);

    // Count substeps only when frame diagnostics are being collected (design
    // §16: fixed-step substep count), so the un-observed path pays nothing but
    // a single resource presence check per frame.
    #[cfg(feature = "std")]
    let instrument = world
        .get_resource::<crate::diagnostics::FrameDiagnostics>()
        .is_some();
    #[cfg(feature = "std")]
    let mut substeps: u32 = 0;

    loop {
        let stepped = {
            let clocks = world.resource_mut::<EngineClocks>();
            let stepped = clocks.fixed_mut().expend();
            if stepped {
                clocks.sync_default();
            }
            stepped
        };
        if !stepped {
            break;
        }

        #[cfg(feature = "std")]
        if instrument {
            substeps = substeps.saturating_add(1);
        }

        world.run_schedule(FixedFirst);
        world.run_schedule(FixedPreUpdate);
        world.run_schedule(FixedUpdate);
        world.run_schedule(FixedPostUpdate);
        world.run_schedule(FixedLast);
    }

    let clocks = world.resource_mut::<EngineClocks>();
    clocks.set_source(DefaultSource::Virtual);
    clocks.sync_default();

    // Per-frame after-hook (design §8): runs once, after the accumulator is
    // drained and the default clock is back on virtual time. The fixed clock's
    // overstep (interpolation alpha) is final for the frame. See
    // `AfterFixedMainLoop`.
    world.run_schedule(AfterFixedMainLoop);

    #[cfg(feature = "std")]
    if instrument
        && let Some(diag) = world.get_resource_mut::<crate::diagnostics::FrameDiagnostics>()
    {
        diag.record_substeps(substeps);
    }
}

// Compile-time proof that every fixed phase satisfies the ECS `ScheduleLabel`
// bound, matching the core-phase guard in `crate::schedule`.
const _: fn() = || {
    fn assert_label<L: ScheduleLabel>() {}
    assert_label::<FixedFirst>();
    assert_label::<FixedPreUpdate>();
    assert_label::<FixedUpdate>();
    assert_label::<FixedPostUpdate>();
    assert_label::<FixedLast>();
    assert_label::<BeforeFixedMainLoop>();
    assert_label::<AfterFixedMainLoop>();
};
