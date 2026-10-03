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
//! machine, and buffered events; **M2** adds the fixed-timestep inner loop;
//! **M4** begins the window/runner tier: drift-free frame pacing, then
//! platform lifecycle events plus the graceful-shutdown path (this increment).
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
//!   updates. The default is **serial** (design §23 risk #1: serial extract
//!   first). Under the `pipelined` feature, `App::enable_pipelined_rendering`
//!   opts into cross-thread overlap (M3 Inc2): a secondary sub-app renders
//!   frame *N* on a worker thread while the main sub-app simulates frame *N+1*
//!   (design §9/§24.3/§25.2, `PipelinedExecutor`). The pipeline keeps the
//!   serial extract semantics (extract still runs on the main thread, one-way,
//!   against a complete frame), so a pipelined run is bit-for-bit equivalent to
//!   the serial run — only the timing overlaps.
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
//!   run condition. Depth features layer on top: [computed states](crate::state::computed)
//!   ([`App::add_computed_state`]) derive a mode from another each frame,
//!   [sub-states](crate::state::sub) ([`App::add_sub_state`]) exist only while a
//!   parent mode is active, [state-scoped entities](crate::state::scoped)
//!   ([`App::enable_state_scoped_entities`]) auto-despawn when their owning mode
//!   leaves, and [transition hooks](crate::state::transition)
//!   ([`App::add_state_transition_hooks`]) run
//!   [`OnTransition`] `from -> to` edges (design §11).
//! - Buffered cross-frame [events](crate::event): [`App::add_event`] installs
//!   an [`Events<E>`](prism_ecs::event::Events) resource and rotates its double
//!   buffer once per frame in [`First`], so events are readable for the frame
//!   they are sent and the frame after (design §22 M1).
#![cfg_attr(feature = "std", doc = "- A drift-free [frame pacer](crate::pacing) (design §13, §22 M4): the")]
#![cfg_attr(not(feature = "std"), doc = "- A drift-free frame pacer (design §13, §22 M4): the")]
#![cfg_attr(feature = "std", doc = "  [`FramePacer`] caps the loop to a [`FrameLimit`] (unlimited / target")]
#![cfg_attr(not(feature = "std"), doc = "  `FramePacer` caps the loop to a `FrameLimit` (unlimited / target")]
//!   FPS / explicit period) by pacing to a *moving* cadence
//!   (`deadline += period`, never `now + period`) so rounding error cannot
//!   accumulate, with an anti-death-spiral clamp that resyncs after a hitch
#![cfg_attr(feature = "std", doc = "  and rolling [`FrameStats`] for the design §16 diagnostics. Present-")]
#![cfg_attr(not(feature = "std"), doc = "  and rolling `FrameStats` for the design §16 diagnostics. Present-")]
//!   timestamp / VRR alignment stays deferred until `prism_window`/RHI can
//!   supply a present estimate (documented in the module). The
//!   [`HeadlessRunner`] can opt into a cap via
#![cfg_attr(feature = "std", doc = "  [`with_frame_limit`](crate::runner::HeadlessRunner::with_frame_limit)")]
#![cfg_attr(not(feature = "std"), doc = "  `with_frame_limit`")]
//!   for a mobile frame limiter or a server tickrate; the default stays
//!   uncapped.
//! - Platform [lifecycle] events (design §12, §22 M4):
//!   [`Suspended`] / [`Resumed`] / [`LowMemory`] / [`FocusChanged`] /
//!   [`WillRenderFirstFrame`] and an [`AppLifecycle`] run-state resource,
//!   registered together by [`App::add_lifecycle_events`] and emittable from a
//!   runner or test via [`App::send_event`]. The windowed runner that forwards
//!   these from the OS is a later M4 increment and is documented as absent, not
//!   stubbed.
//! - A graceful-shutdown path (design §12/§21/§24.5):
//!   [`App::run_shutdown`] runs the dedicated [`Shutdown`] schedule once
//!   (draining each system's deferred commands), materialises reserved
//!   entities, and tears plugins down in **reverse** registration order via
//!   [`Plugin::shutdown`] — distinct from the post-startup, forward-order
//!   [`cleanup`](crate::plugin::Plugin::cleanup). The runners invoke it once the
//!   frame loop ends.
//! - A vetoable exit gate (design §24.5): a pending [`AppExitRequest`] passes
//!   through the [`ExitConfirmation`] schedule that [`App::poll_exit`] runs, so
//!   a confirmation system may [`cancel`](crate::exit::AppExitRequest::cancel)
//!   it ("unsaved changes — really quit?") and keep the app running; the gate
//!   prompts at most once per distinct request.
//! - Platform-free runners: [`HeadlessRunner`], [`ScheduleRunnerOnce`],
#![cfg_attr(feature = "std", doc = "  and the [`DedicatedServerRunner`] (design §10 / §24.4, M5): an")]
#![cfg_attr(not(feature = "std"), doc = "  and the `DedicatedServerRunner` (design §10 / §24.4, M5): an")]
//!   authoritative fixed-tickrate, deterministic simulation heartbeat with
#![cfg_attr(feature = "std", doc = "  no rendering, publishing live [`ServerTickDiagnostics`] so server")]
#![cfg_attr(not(feature = "std"), doc = "  no rendering, publishing live `ServerTickDiagnostics` so server")]
//!   systems can detect tick overload. Networking stays in the
//!   `prism_replication` layer (injected as a plugin), not this runner.
#![cfg_attr(feature = "std", doc = "- Opt-in [observability](crate::diagnostics) (design §16, §22 M6): rolling")]
#![cfg_attr(not(feature = "std"), doc = "- Opt-in observability (design §16, §22 M6): rolling")]
#![cfg_attr(feature = "std", doc = "  [`FrameDiagnostics`] (whole-frame work time, per-phase timing, and the")]
#![cfg_attr(not(feature = "std"), doc = "  `FrameDiagnostics` (whole-frame work time, per-phase timing, and the")]
#![cfg_attr(feature = "std", doc = "  fixed-step substep count) plus per-plugin [`StartupDiagnostics`]")]
#![cfg_attr(not(feature = "std"), doc = "  fixed-step substep count) plus per-plugin `StartupDiagnostics`")]
//!   (`build`/`finish` wall time). Neither is installed by default, so an
//!   un-observed frame pays only a single resource-presence check. Extract
//!   cost, pipeline-overlap rate, and present latency are honestly deferred.
//! - Capability [tiering](crate::capability) (design §3, §17, §24.4):
//!   [`App::new`] probes the environment into a [`Capabilities`] resource
//!   (logical cores, a documented display heuristic, mobile OS, a measured
//!   timer granularity), derives a [`QualityTier`]
//!   (`Server`/`Mobile`/`Desktop`), and seeds a [`RunMode`]
//!   (`Client`/`DedicatedServer`/`EditorEmbedded`/`Headless`, default picked
//!   from the probe), so assembly can gate on real facts rather than
//!   compile-time guesses. The display probe is an honest, overridable
//!   heuristic (true platform display queries belong to the absent
//!   `prism_window`).
//!
//! Everything on the public surface is a **real, working implementation** —
//! no `todo!()`, `unimplemented!()`, or hollow stubs. Later milestones (sub-app
//! pipelining, the windowed runner) layer on top without rewriting these
//! foundations. For record/replay determinism, the seeded RNG, frame hash, and
//! input record/replay have landed, while World-snapshot rollback stays
//! deferred. Deferred features are documented as absent, never faked.
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
pub mod capability;
pub mod cvar;
#[cfg(feature = "std")]
pub mod crash;
pub mod event;
pub mod exit;
pub mod fixed;
pub mod lifecycle;
#[cfg(feature = "determinism")]
pub mod determinism;
#[cfg(feature = "std")]
pub mod diagnostics;
#[cfg(feature = "std")]
pub mod pacing;
pub mod platform_tier;
pub mod plugin;
pub mod plugin_group;
pub mod plugin_graph;
#[cfg(feature = "pipelined")]
pub mod pipelined;
pub mod run_mode;
pub mod runner;
pub mod schedule;
pub mod settings;
pub mod state;
pub mod sub_app;
pub mod sub_app_label;
pub mod time;
#[cfg(feature = "std")]
pub mod watchdog;

#[cfg(test)]
mod tests;

pub use app::{App, Plugins, PluginsState};
pub use capability::{Capabilities, QualityTier};
pub use cvar::{
    Cvar, CvarBounds, CvarCategory, CvarChanged, CvarCliApplied, CvarCliRejection, CvarCliReport,
    CvarError, CvarFlags, CvarRegistry, CvarSetOutcome, CvarSpec, ValidatedWrite,
};
#[cfg(feature = "std")]
pub use crash::{CrashReport, CrashReporter, CrashSink, CrashSnapshot};
#[cfg(feature = "std")]
pub use watchdog::{FrameStall, FrameWatchdog, StallHandler, WatchdogConfig};
pub use exit::{AppExit, AppExitRequest};
pub use plugin::{Plugin, PluginDependency};
pub use plugin_graph::PluginGraphError;
pub use plugin_group::{PluginGroup, PluginGroupBuilder};
#[cfg(feature = "std")]
pub use runner::{DedicatedServerRunner, ServerTickDiagnostics};
pub use runner::{HeadlessRunner, ScheduleRunnerOnce, run_once};
pub use run_mode::RunMode;
pub use fixed::{
    AfterFixedMainLoop, BeforeFixedMainLoop, FixedFirst, FixedLast, FixedPostUpdate,
    FixedPreUpdate, FixedUpdate,
};
pub use lifecycle::{
    AppLifecycle, FocusChanged, LowMemory, Resumed, Suspended, WillRenderFirstFrame,
};
#[cfg(feature = "determinism")]
pub use determinism::{
    DeterministicRng, FrameHash, HashDivergence, InputRecording, RecordedInput, ReplayLog,
    ReplayMode,
};
#[cfg(feature = "std")]
pub use diagnostics::{
    CountWindow, FrameDiagnostics, PluginStartupTiming, StartupDiagnostics,
};
#[cfg(feature = "std")]
pub use pacing::{
    AdaptiveAction, AdaptiveFrameLimiter, FrameLimit, FramePacer, FrameRateLadder, FrameStats,
};
pub use schedule::{
    ExitConfirmation, First, Last, PostStartup, PostUpdate, PreStartup, PreUpdate, Shutdown,
    StateTransition, Startup, Update,
};
pub use platform_tier::PlatformTierProfile;
pub use settings::{SettingChange, SettingChanged, SettingValue, Settings, SettingsLayer};
pub use state::{
    ComputeDepth, ComputedStates, OnTransition, StateScoped, StateTransitionSet, SubStates,
};
#[cfg(feature = "pipelined")]
pub use pipelined::PipelinedExecutor;
pub use sub_app::{ExtractFn, SubApp, SubApps};
pub use sub_app_label::{BoxedSubAppLabel, SubAppLabel};
pub use time::{EngineClocks, TimeUpdateStrategy};

/// Commonly used exports. Mirrors `bevy_app::prelude` ergonomics to keep the
/// eventual migration a near "change-the-import" exercise, and re-exports the
/// `prism_ecs` prelude so a user needs only one `use`.
pub mod prelude {
    pub use crate::app::{App, Plugins, PluginsState};
    pub use crate::capability::{Capabilities, QualityTier};
    pub use crate::cvar::{
        Cvar, CvarBounds, CvarCategory, CvarChanged, CvarCliApplied, CvarCliRejection,
        CvarCliReport, CvarError, CvarFlags, CvarRegistry, CvarSetOutcome, CvarSpec, ValidatedWrite,
    };
    #[cfg(feature = "std")]
    pub use crate::crash::{CrashReport, CrashReporter, CrashSink, CrashSnapshot};
    #[cfg(feature = "std")]
    pub use crate::watchdog::{FrameStall, FrameWatchdog, StallHandler, WatchdogConfig};
    pub use crate::exit::{AppExit, AppExitRequest};
    pub use crate::plugin::{Plugin, PluginDependency};
    pub use crate::plugin_graph::PluginGraphError;
    pub use crate::plugin_group::{PluginGroup, PluginGroupBuilder};
    #[cfg(feature = "std")]
    pub use crate::runner::{DedicatedServerRunner, ServerTickDiagnostics};
    pub use crate::runner::{HeadlessRunner, ScheduleRunnerOnce};
    pub use crate::run_mode::RunMode;
    pub use crate::fixed::{
        AfterFixedMainLoop, BeforeFixedMainLoop, FixedFirst, FixedLast, FixedPostUpdate,
        FixedPreUpdate, FixedUpdate,
    };
    pub use crate::lifecycle::{
        AppLifecycle, FocusChanged, LowMemory, Resumed, Suspended, WillRenderFirstFrame,
    };
    #[cfg(feature = "determinism")]
    pub use crate::determinism::{
        DeterministicRng, FrameHash, HashDivergence, InputRecording, RecordedInput, ReplayLog,
        ReplayMode,
    };
    #[cfg(feature = "std")]
    pub use crate::diagnostics::{
        CountWindow, FrameDiagnostics, PluginStartupTiming, StartupDiagnostics,
    };
    #[cfg(feature = "std")]
    pub use crate::pacing::{
        AdaptiveAction, AdaptiveFrameLimiter, FrameLimit, FramePacer, FrameRateLadder,
        FrameStats,
    };
    pub use crate::schedule::{
        ExitConfirmation, First, Last, PostStartup, PostUpdate, PreStartup, PreUpdate, Shutdown,
        StateTransition, Startup, Update,
    };
    pub use crate::platform_tier::PlatformTierProfile;
    pub use crate::settings::{
        SettingChange, SettingChanged, SettingValue, Settings, SettingsLayer,
    };
    pub use crate::state::{
        ComputeDepth, ComputedStates, OnTransition, StateScoped, StateTransitionSet, SubStates,
    };
    #[cfg(feature = "pipelined")]
    pub use crate::pipelined::PipelinedExecutor;
    pub use crate::sub_app::{ExtractFn, SubApp, SubApps};
    pub use crate::sub_app_label::{BoxedSubAppLabel, SubAppLabel};
    pub use crate::time::{EngineClocks, TimeUpdateStrategy};

    pub use prism_ecs::prelude::*;
}
