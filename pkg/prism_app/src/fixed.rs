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
//! # Honestly deferred
//!
//! `RunFixedMainLoop` here is a native driver, so inserting *user* systems
//! `before` / `after` the inner loop *within* `RunFixedMainLoop` (as Bevy's
//! system-based driver allows) is not yet possible. Attach fixed-rate systems
//! to the [`FixedMain`](FIXED_MAIN_PHASES) tick group (commonly
//! [`FixedUpdate`]) instead. The system-based, user-extensible driver is a
//! later refinement and is documented as absent, not stubbed.

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

    world
        .resource_mut::<EngineClocks>()
        .set_source(DefaultSource::Fixed);

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

        world.run_schedule(FixedFirst);
        world.run_schedule(FixedPreUpdate);
        world.run_schedule(FixedUpdate);
        world.run_schedule(FixedPostUpdate);
        world.run_schedule(FixedLast);
    }

    let clocks = world.resource_mut::<EngineClocks>();
    clocks.set_source(DefaultSource::Virtual);
    clocks.sync_default();
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
};
