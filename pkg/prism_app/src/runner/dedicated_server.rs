//! The dedicated-server runner (design §10 / §24.4).
//!
//! A dedicated server is a headless, authoritative simulation heartbeat: no
//! window, no renderer, no audio — just the main sub-app ticking at a fixed
//! *tickrate* while the network layer feeds it input and ships out snapshots.
//! Where [`HeadlessRunner`](crate::runner::HeadlessRunner) is a general "loop
//! until exit" workhorse (as-fast-as-possible by default, with an *optional*
//! cap), [`DedicatedServerRunner`] encodes the three things that make a server
//! tick *authoritative* rather than merely headless:
//!
//! 1. **Fixed simulated step per tick (determinism).** By default the runner
//!    advances simulated time by exactly one tick period every tick
//!    ([`TimeUpdateStrategy::ManualDelta`]), so the simulation is reproducible
//!    regardless of real-time jitter — the backbone of server-authoritative
//!    rollback/replay (design §15). Opt out with
//!    [`with_wall_clock_time`](DedicatedServerRunner::with_wall_clock_time).
//! 2. **One fixed-update step per tick.** By default it aligns the
//!    [`FixedMain`](crate::fixed) timestep with the tickrate, so a tick drives
//!    exactly one `FixedUpdate` — the classic "定 tickrate 固定步长心跳".
//!    Opt out with
//!    [`without_fixed_timestep_alignment`](DedicatedServerRunner::without_fixed_timestep_alignment).
//! 3. **Real-time pacing + overload diagnostics.** A drift-free [`FramePacer`]
//!    paces the loop to the tickrate in wall-clock time (unless
//!    [`without_real_time_pacing`](DedicatedServerRunner::without_real_time_pacing)
//!    is set, for offline resimulation / tests), and the runner publishes a
//!    live [`ServerTickDiagnostics`] resource so server systems can detect when
//!    a tick overran its budget — i.e. the server is *falling behind* — and
//!    shed load or log it.
//!
//! # Honestly deferred
//!
//! Networking itself (listen/accept, snapshot encode, client reconciliation)
//! is **not** part of this runner: that is the `prism_replication` layer's job
//! and is injected as a plugin. This runner provides the *authoritative
//! heartbeat* those systems run on, nothing more. It is `std`-only because it
//! paces against the platform monotonic clock.

use core::num::NonZeroU32;

use prism_ecs::resource::Resource;
use prism_time::{Duration, Instant};

use crate::app::App;
use crate::exit::AppExit;
use crate::pacing::{FrameLimit, FramePacer};
use crate::time::TimeUpdateStrategy;

/// Live, per-tick health of a [`DedicatedServerRunner`] loop, published into
/// the main world so server systems can react to overload (design §16/§24.4).
///
/// A system takes `Res<ServerTickDiagnostics>` and reads
/// [`is_overloaded`](ServerTickDiagnostics::is_overloaded) /
/// [`overrun_count`](ServerTickDiagnostics::overrun_count) to decide whether to
/// shed work (reduce AoI, drop cosmetic updates) or warn an operator that the
/// authoritative tick can no longer keep real time. The runner inserts this
/// resource before the first tick and refreshes it after every
/// [`App::update`](crate::app::App::update).
#[derive(Clone, Copy, Debug, Default)]
pub struct ServerTickDiagnostics {
    tick: u64,
    tick_period: Duration,
    last_tick_work: Duration,
    overrun_count: u64,
}

impl Resource for ServerTickDiagnostics {}

impl ServerTickDiagnostics {
    /// Number of completed ticks (`App::update` calls) so far.
    #[must_use]
    pub fn tick(&self) -> u64 {
        self.tick
    }

    /// The target wall-clock period of one tick (`1 / tickrate`).
    #[must_use]
    pub fn tick_period(&self) -> Duration {
        self.tick_period
    }

    /// Wall-clock time the most recent tick's work (one `App::update`) took,
    /// excluding any pacing sleep.
    #[must_use]
    pub fn last_tick_work(&self) -> Duration {
        self.last_tick_work
    }

    /// How many ticks so far overran their budget (`last_tick_work >
    /// tick_period`).
    #[must_use]
    pub fn overrun_count(&self) -> u64 {
        self.overrun_count
    }

    /// Whether the most recent tick overran its period — the "server is falling
    /// behind this instant" signal.
    #[must_use]
    pub fn is_overloaded(&self) -> bool {
        !self.tick_period.is_zero() && self.last_tick_work > self.tick_period
    }

    /// Fraction `[0, 1]` of this tick's budget consumed by work. Saturates at
    /// `1.0` when a tick overran, and is `0.0` before the first tick or when the
    /// period is zero.
    #[must_use]
    pub fn budget_used(&self) -> f64 {
        let period = self.tick_period.as_secs_f64();
        if period <= 0.0 {
            return 0.0;
        }
        (self.last_tick_work.as_secs_f64() / period).min(1.0)
    }
}

/// Drives an [`App`] as an authoritative dedicated server: a fixed-tickrate,
/// deterministic simulation heartbeat with no rendering (design §10 / §24.4).
///
/// Construct with [`new`](DedicatedServerRunner::new) and tune with the builder
/// methods, then install it with
/// [`App::set_runner`](crate::app::App::set_runner):
///
/// ```no_run
/// use prism_app::prelude::*;
/// use prism_app::runner::DedicatedServerRunner;
///
/// let runner = DedicatedServerRunner::new(60);
/// App::new()
///     .add_systems(Update, || { /* authoritative step */ })
///     .set_runner(move |app| runner.run(app))
///     .run();
/// ```
#[derive(Clone, Copy, Debug)]
pub struct DedicatedServerRunner {
    tickrate: NonZeroU32,
    max_ticks: Option<u64>,
    deterministic: bool,
    align_fixed_timestep: bool,
    real_time_paced: bool,
}

impl DedicatedServerRunner {
    /// A server runner at `tickrate_hz` ticks per second (e.g. `60` for an
    /// action game, `20`/`30` for a larger-world MMO), with deterministic
    /// per-tick stepping, fixed-timestep alignment, and real-time pacing all on
    /// by default.
    ///
    /// # Panics
    ///
    /// Panics if `tickrate_hz` is `0`: a server with no tickrate is a
    /// programming error, and failing loudly is more honest than silently
    /// picking a rate.
    #[must_use]
    pub fn new(tickrate_hz: u32) -> Self {
        let tickrate =
            NonZeroU32::new(tickrate_hz).expect("DedicatedServerRunner tickrate must be non-zero");
        Self {
            tickrate,
            max_ticks: None,
            deterministic: true,
            align_fixed_timestep: true,
            real_time_paced: true,
        }
    }

    /// Stop after at most `max_ticks` ticks even if no exit was requested
    /// (CI / batch resimulation / tests). Builder-style; the default loops
    /// until a system requests exit.
    #[must_use]
    pub fn with_max_ticks(mut self, max_ticks: u64) -> Self {
        self.max_ticks = Some(max_ticks);
        self
    }

    /// Advance simulated time from the wall clock instead of a fixed per-tick
    /// delta. Builder-style; the default is deterministic
    /// ([`TimeUpdateStrategy::ManualDelta`] of one tick period), which is what
    /// makes server-authoritative replay/rollback (design §15) sound.
    #[must_use]
    pub fn with_wall_clock_time(mut self) -> Self {
        self.deterministic = false;
        self
    }

    /// Leave the [`FixedMain`](crate::fixed) timestep untouched instead of
    /// aligning it to the tickrate. Builder-style; by default the runner sets
    /// the fixed timestep to the tickrate so one tick drives exactly one
    /// `FixedUpdate`.
    #[must_use]
    pub fn without_fixed_timestep_alignment(mut self) -> Self {
        self.align_fixed_timestep = false;
        self
    }

    /// Run ticks back-to-back with no wall-clock pacing (offline
    /// resimulation / fast-forward / deterministic tests). Builder-style; by
    /// default the loop is paced to the tickrate with a drift-free
    /// [`FramePacer`].
    #[must_use]
    pub fn without_real_time_pacing(mut self) -> Self {
        self.real_time_paced = false;
        self
    }

    /// The configured tickrate in hertz.
    #[must_use]
    pub fn tickrate_hz(&self) -> u32 {
        self.tickrate.get()
    }

    /// The tick cap, if any.
    #[must_use]
    pub fn max_ticks(&self) -> Option<u64> {
        self.max_ticks
    }

    /// Whether simulated time steps by a fixed per-tick delta (deterministic).
    #[must_use]
    pub fn is_deterministic(&self) -> bool {
        self.deterministic
    }

    /// Whether the fixed timestep is aligned to the tickrate.
    #[must_use]
    pub fn aligns_fixed_timestep(&self) -> bool {
        self.align_fixed_timestep
    }

    /// Whether the loop is paced to the tickrate in wall-clock time.
    #[must_use]
    pub fn is_real_time_paced(&self) -> bool {
        self.real_time_paced
    }

    /// The wall-clock period of one tick (`1 / tickrate`).
    #[must_use]
    pub fn tick_period(&self) -> Duration {
        FrameLimit::Fps(self.tickrate)
            .period()
            .expect("a non-zero FPS limit always has a period")
    }

    /// Drive `app` as a dedicated server until a system requests exit or the
    /// tick cap is reached, then run the graceful-shutdown path once.
    ///
    /// Configures the time context for an authoritative heartbeat (per the
    /// builder flags), then loops: run one tick ([`App::update`]), refresh
    /// [`ServerTickDiagnostics`], check for exit / the tick cap, and pace to the
    /// tickrate. Returns the requested [`AppExit`] if a system asked to stop,
    /// otherwise [`AppExit::Success`] when the tick cap is reached.
    pub fn run(self, mut app: App) -> AppExit {
        let period = self.tick_period();

        // Authoritative stepping: fixed simulated delta per tick + one fixed
        // step per tick, unless the caller opted out.
        if self.deterministic {
            app.set_time_update_strategy(TimeUpdateStrategy::ManualDelta(period));
        }
        if self.align_fixed_timestep {
            // Align the fixed step to the *exact* tick period so one tick
            // drives exactly one `FixedUpdate`. Deriving the step from the same
            // `Duration` as the manual delta (rather than re-deriving from the
            // tickrate in hertz) keeps the accumulator from drifting a
            // nanosecond short each tick and silently dropping a step.
            app.set_fixed_timestep(period);
        }

        // Publish diagnostics so server systems can observe overload from tick
        // one (seeded with the target period, zero work recorded yet).
        let mut diagnostics = ServerTickDiagnostics {
            tick: 0,
            tick_period: period,
            last_tick_work: Duration::ZERO,
            overrun_count: 0,
        };
        app.insert_resource(diagnostics);

        let limit = if self.real_time_paced {
            FrameLimit::Fps(self.tickrate)
        } else {
            FrameLimit::Off
        };
        let mut pacer = FramePacer::new(limit);

        let exit = loop {
            let started = Instant::now();
            app.update();
            let work = Instant::now().saturating_duration_since(started);

            diagnostics.tick = diagnostics.tick.saturating_add(1);
            diagnostics.last_tick_work = work;
            if work > period {
                diagnostics.overrun_count = diagnostics.overrun_count.saturating_add(1);
            }
            // Refresh the published copy so systems read this tick's health.
            app.insert_resource(diagnostics);

            // Confirmed-exit poll (runs the exit-veto gate on a pending request,
            // design §24.5) rather than the raw observer.
            if let Some(exit) = app.poll_exit() {
                break exit;
            }
            if self.max_ticks.is_some_and(|max| diagnostics.tick >= max) {
                break AppExit::Success;
            }

            // Pace to the tickrate in wall-clock time (no-op when unpaced).
            pacer.throttle();
        };

        // A dedicated server has no render sub-app, but calling this is a
        // harmless no-op and keeps every runner's exit path uniform.
        app.sync_sub_apps();
        // Graceful shutdown once: drain Shutdown schedule + reverse plugin
        // teardown (design §12/§21/§24.5).
        app.run_shutdown();
        exit
    }
}
