//! `prism_app` — Prism's application shell.
//!
//! `prism_app` is the in-house replacement for `bevy_app`: it organizes a set
//! of [`prism_ecs`] worlds and schedules into *a running engine instance*. It
//! defines **who** runs, **in what order**, and **driven by which runner**, and
//! provides [`Plugin`]s as the single unit of engine assembly. It sits above
//! `prism_ecs` (simulation + scheduling) and below the game-runtime framework.
//!
//! The architecture, invariants, and milestones are specified in
//! `docs/prism_app_design_zh.md`.
//!
//! # Milestone status
//!
//! This crate is built in milestones (design §22). **M0** shipped the shell;
//! **M1** layered the full phase order, the plugin dependency graph, the state
//! machine, and buffered events; **M2** adds the fixed-timestep inner loop.
//! Currently implemented:
//!
//! - [`App`] with a monotonic [`PluginsState`] assembly
//!   state machine and a swappable [runner].
//! - A [`SubApps`] collection: a main [`SubApp`] = one
//!   [`World`](prism_ecs::world::World) whose world-owned
//!   [`Schedules`](prism_ecs::schedule::Schedules) resource holds the phase
//!   schedules (design §5: reuse the `prism_ecs` scheduling graph), plus an
//!   ordered set of labeled secondary sub-apps keyed by [`SubAppLabel`]. Each
//!   frame, after the main sub-app updates, every secondary sub-app runs its
//!   one-way [`ExtractFn`] (`main world → sub world`, design §9/§25.2) and then
//!   updates — **serially** for now (design §23 risk #1: serial extract first,
//!   `pipelined` later). Cross-thread pipelined overlap is M3 Inc2 and honestly
//!   absent, not stubbed.
//! - The [`Plugin`] trait (`build` / `ready` / `finish` / `cleanup`, plus
//!   declared [`dependencies`](crate::plugin::Plugin::dependencies)) and an
//!   editable [`PluginGroup`]: [`PluginGroupBuilder`] supports ordered,
//!   idempotent `add` / `add_before` / `add_after` / `set` / `disable` /
//!   `enable`, and resolves members by a topological sort over declared
//!   dependencies with **assembly-time** cycle and missing-dependency
//!   detection (design §24.1, §23 risk #5).
//! - The built-in core [phase labels](crate::schedule) and the full main-frame
//!   loop
//!   (`First → RunFixedMainLoop → PreUpdate → StateTransition → Update →
//!   PostUpdate → Last`), plus a one-time startup
//!   (`PreStartup → Startup → PostStartup`).
//! - The fixed-timestep inner loop (design §8, §22 M2): `RunFixedMainLoop`
//!   drains the fixed accumulator, running the [`FixedMain`](crate::fixed) tick
//!   group (`FixedFirst → FixedPreUpdate → FixedUpdate → FixedPostUpdate →
//!   FixedLast`) once per fixed step with a spiral-of-death cap. The
//!   [`Time<Real>`](prism_time::Time) / [`Time<Virtual>`](prism_time::Time) /
//!   [`Time<Fixed>`](prism_time::Time) clocks are exposed via
//!   [`EngineClocks`] and driven each frame by [`advance_time`](crate::time::advance_time);
//!   the accumulator itself is the single source of truth in `prism_time`
//!   (design §25.1), which this crate only drives.
//! - A finite [state machine](crate::state): [`App::insert_state`] /
//!   [`App::init_state`] wire a [`States`](prism_ecs::schedule::States) type
//!   into the [`StateTransition`] phase, with
//!   `OnEnter`/`OnExit` edges and the [`in_state`](prism_ecs::schedule::in_state)
//!   run condition (design §11).
//! - Buffered cross-frame [events](crate::event): [`App::add_event`] installs
//!   an [`Events<E>`](prism_ecs::event::Events) resource and rotates its double
//!   buffer once per frame in [`First`], so events are readable for the frame
//!   they are sent and the frame after (design §22 M1).
//! - Platform-free runners: [`HeadlessRunner`] and
//!   [`ScheduleRunnerOnce`].
//!
//! Everything on the public surface is a **real, working implementation** —
//! no `todo!()`, `unimplemented!()`, or hollow stubs. Later milestones (sub-app
//! pipelining, windowed / dedicated-server runners, determinism) layer on top
//! without rewriting these foundations. Deferred features are documented as
//! absent, never faked.
//!
//! # `std`
//!
//! M0 requires `std` (the runner / main loop needs it, design §1). The App
//! graph + plugin registry are `no_std + alloc`-capable in principle; relaxing
//! the `std` requirement is future work and is honestly not done yet.
//!
//! # Provenance
//!
//! This crate is engine-agnostic and contains **no Unreal Engine or Unity
//! source or derived code**, and depends on **no `bevy_*` crate**. It borrows
//! only publicly documented architectural shapes.

pub mod app;
pub mod event;
pub mod exit;
pub mod fixed;
pub mod plugin;
pub mod plugin_group;
pub mod plugin_graph;
pub mod runner;
pub mod schedule;
pub mod state;
pub mod sub_app;
pub mod sub_app_label;
pub mod time;

#[cfg(test)]
mod tests;

pub use app::{App, Plugins, PluginsState};
pub use exit::{AppExit, AppExitRequest};
pub use plugin::{Plugin, PluginDependency};
pub use plugin_graph::PluginGraphError;
pub use plugin_group::{PluginGroup, PluginGroupBuilder};
pub use runner::{HeadlessRunner, ScheduleRunnerOnce, run_once};
pub use fixed::{FixedFirst, FixedLast, FixedPostUpdate, FixedPreUpdate, FixedUpdate};
pub use schedule::{
    First, Last, PostStartup, PostUpdate, PreStartup, PreUpdate, StateTransition, Startup, Update,
};
pub use sub_app::{ExtractFn, SubApp, SubApps};
pub use sub_app_label::{BoxedSubAppLabel, SubAppLabel};
pub use time::{EngineClocks, TimeUpdateStrategy};

/// Commonly used exports. Mirrors `bevy_app::prelude` ergonomics to keep the
/// eventual migration a near "change-the-import" exercise, and re-exports the
/// `prism_ecs` prelude so a user needs only one `use`.
pub mod prelude {
    pub use crate::app::{App, Plugins, PluginsState};
    pub use crate::exit::{AppExit, AppExitRequest};
    pub use crate::plugin::{Plugin, PluginDependency};
    pub use crate::plugin_graph::PluginGraphError;
    pub use crate::plugin_group::{PluginGroup, PluginGroupBuilder};
    pub use crate::runner::{HeadlessRunner, ScheduleRunnerOnce};
    pub use crate::fixed::{FixedFirst, FixedLast, FixedPostUpdate, FixedPreUpdate, FixedUpdate};
    pub use crate::schedule::{
        First, Last, PostStartup, PostUpdate, PreStartup, PreUpdate, StateTransition, Startup,
        Update,
    };
    pub use crate::sub_app::{ExtractFn, SubApp, SubApps};
    pub use crate::sub_app_label::{BoxedSubAppLabel, SubAppLabel};
    pub use crate::time::{EngineClocks, TimeUpdateStrategy};

    pub use prism_ecs::prelude::*;
}
