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
//! This crate is built in milestones (design §22). **M0 (this module set)** is
//! the shell:
//!
//! - [`App`] with a monotonic [`PluginsState`] assembly
//!   state machine and a swappable [runner].
//! - A single main [`SubApp`] = one [`World`](prism_ecs::world::World) whose
//!   world-owned [`Schedules`](prism_ecs::schedule::Schedules) resource holds
//!   the phase schedules (design §5: reuse the `prism_ecs` scheduling graph).
//! - The [`Plugin`] trait (`build` / `ready` / `finish` / `cleanup`) and a
//!   minimal [`PluginGroup`].
//! - The built-in core [phase labels](crate::schedule) and the variable-step
//!   main-frame loop
//!   (`First → PreUpdate → StateTransition → Update → PostUpdate → Last`), plus
//!   a one-time startup (`PreStartup → Startup → PostStartup`).
//! - Platform-free runners: [`HeadlessRunner`] and
//!   [`ScheduleRunnerOnce`].
//!
//! Everything on the M0 public surface is a **real, working implementation** —
//! no `todo!()`, `unimplemented!()`, or hollow stubs. Later milestones (full
//! phase order with `RunFixedMainLoop`/`StateTransition`, plugin dependency
//! graphs and groups editing, fixed timestep, sub-app pipelining, windowed /
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
pub mod exit;
pub mod plugin;
pub mod plugin_group;
pub mod runner;
pub mod schedule;
pub mod sub_app;

#[cfg(test)]
mod tests;

pub use app::{App, Plugins, PluginsState};
pub use exit::{AppExit, AppExitRequest};
pub use plugin::Plugin;
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
    pub use crate::plugin::Plugin;
    pub use crate::plugin_group::{PluginGroup, PluginGroupBuilder};
    pub use crate::runner::{HeadlessRunner, ScheduleRunnerOnce};
    pub use crate::schedule::{
        First, Last, PostStartup, PostUpdate, PreStartup, PreUpdate, StateTransition, Startup,
        Update,
    };
    pub use crate::sub_app::SubApp;

    pub use prism_ecs::prelude::*;
}
