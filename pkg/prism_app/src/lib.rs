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
//! **M1 (in progress)** layers the full phase order and the state machine on
//! top. Currently implemented:
//!
//! - [`App`] with a monotonic [`PluginsState`] assembly
//!   state machine and a swappable [runner].
//! - A single main [`SubApp`] = one [`World`](prism_ecs::world::World) whose
//!   world-owned [`Schedules`](prism_ecs::schedule::Schedules) resource holds
//!   the phase schedules (design §5: reuse the `prism_ecs` scheduling graph).
//! - The [`Plugin`] trait (`build` / `ready` / `finish` / `cleanup`, plus
//!   declared [`dependencies`](crate::plugin::Plugin::dependencies)) and an
//!   editable [`PluginGroup`]: [`PluginGroupBuilder`] supports ordered,
//!   idempotent `add` / `add_before` / `add_after` / `set` / `disable` /
//!   `enable`, and resolves members by a topological sort over declared
//!   dependencies with **assembly-time** cycle and missing-dependency
//!   detection (design §24.1, §23 risk #5).
//! - The built-in core [phase labels](crate::schedule) and the variable-step
//!   main-frame loop
//!   (`First → PreUpdate → StateTransition → Update → PostUpdate → Last`), plus
//!   a one-time startup (`PreStartup → Startup → PostStartup`).
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
//! no `todo!()`, `unimplemented!()`, or hollow stubs. Later milestones (fixed
//! timestep with `RunFixedMainLoop`, sub-app pipelining, windowed /
//! dedicated-server runners, determinism) layer on top without rewriting these
//! foundations. Deferred features are documented as absent, never faked.
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
pub mod plugin;
pub mod plugin_group;
pub mod plugin_graph;
pub mod runner;
pub mod schedule;
pub mod state;
pub mod sub_app;

#[cfg(test)]
mod tests;

pub use app::{App, Plugins, PluginsState};
pub use exit::{AppExit, AppExitRequest};
pub use plugin::{Plugin, PluginDependency};
pub use plugin_graph::PluginGraphError;
pub use plugin_group::{PluginGroup, PluginGroupBuilder};
pub use runner::{HeadlessRunner, ScheduleRunnerOnce, run_once};
pub use schedule::{
    First, Last, PostStartup, PostUpdate, PreStartup, PreUpdate, StateTransition, Startup, Update,
};
pub use sub_app::SubApp;

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
    pub use crate::schedule::{
        First, Last, PostStartup, PostUpdate, PreStartup, PreUpdate, StateTransition, Startup,
        Update,
    };
    pub use crate::sub_app::SubApp;

    pub use prism_ecs::prelude::*;
}
