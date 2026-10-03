//! Frame and startup observability (design §16): rolling per-frame wall-clock
//! statistics, per-phase timing, fixed-substep counts, and per-plugin startup
//! timing.
//!
//! # What this measures (and what it does not)
//!
//! Design §16 ("可观测性") asks for a *phase flame graph* (per-phase / per-
//! schedule cost), *frame statistics* (frame time, fixed-step substep count,
//! extract cost, pipeline-overlap rate, present latency), and *startup cost*
//! (each plugin's `build` / `finish` time). This module implements the parts
//! that are **genuinely measurable from inside `prism_app`** with nothing but a
//! wall clock:
//!
//! - [`FrameDiagnostics`] — rolling [`FrameStats`] over the whole-frame work
//!   time, a per-phase [`FrameStats`] for each core frame phase, and a
//!   [`CountWindow`] of the fixed-timestep **substep count** per frame.
//! - [`StartupDiagnostics`] — each plugin's [`build`](crate::plugin::Plugin::build)
//!   and [`finish`](crate::plugin::Plugin::finish) wall time, in registration
//!   order, so a slow boot can be attributed to a specific plugin.
//!
//! # Pay only when observing
//!
//! Diagnostics are **opt-in**: neither resource is installed by
//! [`App::new`](crate::app::App::new). The frame loop checks for
//! [`FrameDiagnostics`] once per frame; when it is absent the loop runs the
//! plain, un-instrumented phase sequence and pays nothing. Install the
//! resources explicitly ([`App::init_frame_diagnostics`](crate::app::App::init_frame_diagnostics),
//! [`App::init_startup_diagnostics`](crate::app::App::init_startup_diagnostics)) to start collecting — and install
//! [`StartupDiagnostics`] *before* [`add_plugins`](crate::app::App::add_plugins)
//! so build timings are captured.
//!
//! # Frame-work time vs. loop cadence
//!
//! [`FrameDiagnostics::frame_time`] measures the wall time spent **doing a
//! frame's work** (`First → … → Last`, including the fixed loop), excluding any
//! frame-pacing sleep. That is deliberately distinct from
//! [`FrameStats`] owned by the
//! [`FramePacer`](crate::pacing::FramePacer), which measures the *interval*
//! between frame boundaries (the achieved cadence, including throttle sleep).
//! Work time answers "how long did simulation take?"; cadence answers "how fast
//! are we actually running?". Both are design §16 metrics and both are real.
//!
//! # Honestly deferred
//!
//! - **Extract cost / pipeline-overlap rate / present latency** (design §16)
//!   are *not* measured here. Extract and overlap are properties of the
//!   secondary-sub-app seam and the `pipelined` executor; present latency needs
//!   a `prism_window`/RHI present timestamp that does not exist yet. Measuring
//!   them now would mean fabricating numbers, so they are documented as absent
//!   rather than stubbed.
//! - **Per-system flame graph** (design §16: ECS §16.6 system timing) belongs
//!   to `prism_ecs`'s executor, not this crate; `prism_app` only times at the
//!   phase/schedule granularity it drives.

use std::collections::BTreeMap;
use std::collections::VecDeque;
use std::time::Duration;

use prism_ecs::resource::Resource;

use crate::pacing::FrameStats;

/// Rolling statistics over a bounded window of unsigned counts (design §16).
///
/// The count-oriented sibling of [`FrameStats`]: it tracks the fixed-timestep
/// substep count observed each frame, so a run can report the typical and worst
/// substep load (a value pinned at the `max_substeps` cap signals the fixed
/// loop is saturating and the simulation is falling behind real time).
///
/// Memory is `O(window)` regardless of run length.
#[derive(Clone, Debug)]
pub struct CountWindow {
    window: usize,
    samples: VecDeque<u32>,
    total_samples: u64,
}

impl CountWindow {
    /// A window keeping at most `window` recent counts. A `window` of `0` is
    /// clamped to `1` so there is always room for the last sample.
    #[must_use]
    pub fn new(window: usize) -> Self {
        let window = window.max(1);
        Self {
            window,
            samples: VecDeque::with_capacity(window),
            total_samples: 0,
        }
    }

    /// Record one count, evicting the oldest sample past the window.
    pub fn record(&mut self, count: u32) {
        if self.samples.len() == self.window {
            self.samples.pop_front();
        }
        self.samples.push_back(count);
        self.total_samples = self.total_samples.saturating_add(1);
    }

    /// Total counts recorded over the whole lifetime (not just the window).
    #[must_use]
    pub fn total_samples(&self) -> u64 {
        self.total_samples
    }

    /// Number of samples currently in the window.
    #[must_use]
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Whether no count has been recorded yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// The most recent count, or `None` before the first record.
    #[must_use]
    pub fn last(&self) -> Option<u32> {
        self.samples.back().copied()
    }

    /// Mean count over the window, or `None` when empty.
    #[must_use]
    pub fn average(&self) -> Option<f64> {
        if self.samples.is_empty() {
            return None;
        }
        let total: u64 = self.samples.iter().map(|&c| u64::from(c)).sum();
        Some(total as f64 / self.samples.len() as f64)
    }

    /// Largest count in the window — the worst substep load.
    #[must_use]
    pub fn max(&self) -> Option<u32> {
        self.samples.iter().copied().max()
    }

    /// Smallest count in the window.
    #[must_use]
    pub fn min(&self) -> Option<u32> {
        self.samples.iter().copied().min()
    }
}

/// The core frame phases this crate times, in run order. The fixed loop is
/// timed as the single `RunFixedMainLoop` phase (its per-substep tick group is
/// counted separately via [`FrameDiagnostics::fixed_substeps`]).
pub const TIMED_FRAME_PHASES: [&str; 7] = [
    "First",
    "RunFixedMainLoop",
    "PreUpdate",
    "StateTransition",
    "Update",
    "PostUpdate",
    "Last",
];

/// Rolling per-frame observability for a sub-app (design §16).
///
/// Install it on a sub-app's world ([`App::init_frame_diagnostics`](crate::app::App::init_frame_diagnostics) installs it
/// on the main world) to make [`SubApp::update`](crate::sub_app::SubApp::update)
/// collect, each frame:
///
/// - the whole-frame work time ([`frame_time`](FrameDiagnostics::frame_time)),
/// - the time spent in each core phase ([`phase`](FrameDiagnostics::phase)), and
/// - the fixed-timestep substep count
///   ([`fixed_substeps`](FrameDiagnostics::fixed_substeps)).
///
/// All three are bounded rolling windows, so memory stays `O(window)`.
#[derive(Clone, Debug)]
pub struct FrameDiagnostics {
    window: usize,
    frame_time: FrameStats,
    phases: BTreeMap<&'static str, FrameStats>,
    fixed_substeps: CountWindow,
}

impl FrameDiagnostics {
    /// Default rolling-window length (frames) when none is specified.
    pub const DEFAULT_WINDOW: usize = 128;

    /// Diagnostics keeping the [default](Self::DEFAULT_WINDOW) window length.
    #[must_use]
    pub fn new() -> Self {
        Self::with_window(Self::DEFAULT_WINDOW)
    }

    /// Diagnostics keeping at most `window` recent frames per metric. A
    /// `window` of `0` is clamped to `1`.
    #[must_use]
    pub fn with_window(window: usize) -> Self {
        let window = window.max(1);
        Self {
            window,
            frame_time: FrameStats::new(window),
            phases: BTreeMap::new(),
            fixed_substeps: CountWindow::new(window),
        }
    }

    /// Record one whole-frame work duration.
    pub fn record_frame(&mut self, elapsed: Duration) {
        self.frame_time.record(elapsed);
    }

    /// Record `phase`'s duration this frame, creating its window on first sight.
    pub fn record_phase(&mut self, phase: &'static str, elapsed: Duration) {
        self.phases
            .entry(phase)
            .or_insert_with(|| FrameStats::new(self.window))
            .record(elapsed);
    }

    /// Record how many fixed substeps ran this frame.
    pub fn record_substeps(&mut self, substeps: u32) {
        self.fixed_substeps.record(substeps);
    }

    /// Rolling whole-frame work-time statistics.
    #[must_use]
    pub fn frame_time(&self) -> &FrameStats {
        &self.frame_time
    }

    /// Rolling statistics for one phase by name (see [`TIMED_FRAME_PHASES`]), or
    /// `None` if that phase has not been timed yet.
    #[must_use]
    pub fn phase(&self, phase: &str) -> Option<&FrameStats> {
        self.phases.get(phase)
    }

    /// Iterate every timed phase and its statistics, in deterministic
    /// (alphabetical) name order.
    pub fn phases(&self) -> impl Iterator<Item = (&&'static str, &FrameStats)> {
        self.phases.iter()
    }

    /// Rolling fixed-substep-count statistics.
    #[must_use]
    pub fn fixed_substeps(&self) -> &CountWindow {
        &self.fixed_substeps
    }

    /// The configured rolling-window length.
    #[must_use]
    pub fn window(&self) -> usize {
        self.window
    }

    /// Average frames per second derived from the mean frame-work time, or
    /// `None` before any frame is recorded.
    ///
    /// This is the inverse of the mean *work* time, so it reports the frame
    /// rate the simulation could sustain uncapped — not the paced cadence
    /// (which the [`FramePacer`](crate::pacing::FramePacer) reports). A zero
    /// mean (sub-microsecond frames) yields `None` rather than infinity.
    #[must_use]
    pub fn fps(&self) -> Option<f64> {
        let avg = self.frame_time.average()?.as_secs_f64();
        if avg > 0.0 {
            Some(1.0 / avg)
        } else {
            None
        }
    }
}

impl Default for FrameDiagnostics {
    fn default() -> Self {
        Self::new()
    }
}

impl Resource for FrameDiagnostics {}

/// One plugin's startup wall-clock timing (design §16: "启动耗时").
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PluginStartupTiming {
    /// The plugin's [`name`](crate::plugin::Plugin::name).
    pub name: String,
    /// Wall time spent in the plugin's
    /// [`build`](crate::plugin::Plugin::build).
    pub build: Duration,
    /// Wall time spent in the plugin's
    /// [`finish`](crate::plugin::Plugin::finish). [`Duration::ZERO`] until the
    /// finish phase runs; once it has run every plugin is timed, so a plugin
    /// that overrides nothing records a tiny (near-zero) duration rather than
    /// staying at zero.
    pub finish: Duration,
}

/// Per-plugin startup timing collected during assembly (design §16).
///
/// Install it ([`App::init_startup_diagnostics`](crate::app::App::init_startup_diagnostics)) **before** adding plugins so
/// each [`build`](crate::plugin::Plugin::build) is timed; finish times are
/// filled in when [`App::finish`](crate::app::App::finish) runs. Timings are
/// kept in registration order, so [`slowest_build`](StartupDiagnostics::slowest_build)
/// pinpoints the plugin to optimise first.
#[derive(Clone, Debug, Default)]
pub struct StartupDiagnostics {
    plugins: Vec<PluginStartupTiming>,
}

impl StartupDiagnostics {
    /// An empty startup-timing log.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a plugin's `build` time, appending in registration order.
    pub fn record_build(&mut self, name: impl Into<String>, build: Duration) {
        self.plugins.push(PluginStartupTiming {
            name: name.into(),
            build,
            finish: Duration::ZERO,
        });
    }

    /// Record a plugin's `finish` time against its existing entry.
    ///
    /// Matches the most recent entry with this name whose finish is still
    /// [`Duration::ZERO`], so a (rare) duplicate-named non-unique plugin fills
    /// its own slots in order. A finish with no matching build entry (e.g. the
    /// log was installed after plugins were added) is ignored rather than
    /// invented.
    pub fn record_finish(&mut self, name: &str, finish: Duration) {
        if let Some(entry) = self
            .plugins
            .iter_mut()
            .rev()
            .find(|p| p.name == name && p.finish.is_zero())
        {
            entry.finish = finish;
        }
    }

    /// Every plugin's timing, in registration order.
    #[must_use]
    pub fn plugins(&self) -> &[PluginStartupTiming] {
        &self.plugins
    }

    /// Total wall time across all plugins' `build` phases.
    #[must_use]
    pub fn total_build(&self) -> Duration {
        self.plugins.iter().map(|p| p.build).sum()
    }

    /// Total wall time across all plugins' `finish` phases.
    #[must_use]
    pub fn total_finish(&self) -> Duration {
        self.plugins.iter().map(|p| p.finish).sum()
    }

    /// The plugin whose `build` took longest, or `None` when nothing is logged.
    #[must_use]
    pub fn slowest_build(&self) -> Option<&PluginStartupTiming> {
        self.plugins.iter().max_by_key(|p| p.build)
    }
}

impl Resource for StartupDiagnostics {}

impl crate::app::App {
    /// Install [`FrameDiagnostics`] on the main world (if absent) so the frame
    /// loop collects per-frame, per-phase, and substep statistics.
    ///
    /// Idempotent: an existing diagnostics resource (with its accumulated
    /// history) is never replaced.
    pub fn init_frame_diagnostics(&mut self) -> &mut Self {
        if self.world().get_resource::<FrameDiagnostics>().is_none() {
            self.insert_resource(FrameDiagnostics::new());
        }
        self
    }

    /// Install [`FrameDiagnostics`] with an explicit rolling-window length.
    ///
    /// Idempotent in the same sense as [`init_frame_diagnostics`](crate::app::App::init_frame_diagnostics):
    /// if diagnostics already exist they are kept as-is and `window` is ignored,
    /// so prior history is never discarded.
    pub fn init_frame_diagnostics_with_window(&mut self, window: usize) -> &mut Self {
        if self.world().get_resource::<FrameDiagnostics>().is_none() {
            self.insert_resource(FrameDiagnostics::with_window(window));
        }
        self
    }

    /// Install [`StartupDiagnostics`] on the main world (if absent) so each
    /// plugin's `build`/`finish` time is captured.
    ///
    /// Call this **before** [`add_plugins`](crate::app::App::add_plugins): only plugins
    /// added after the log exists are timed (earlier builds have already run and
    /// are not retroactively invented).
    pub fn init_startup_diagnostics(&mut self) -> &mut Self {
        if self.world().get_resource::<StartupDiagnostics>().is_none() {
            self.insert_resource(StartupDiagnostics::new());
        }
        self
    }

    /// Borrow the main world's [`FrameDiagnostics`], if installed.
    #[must_use]
    pub fn frame_diagnostics(&self) -> Option<&FrameDiagnostics> {
        self.world().get_resource::<FrameDiagnostics>()
    }

    /// Borrow the main world's [`StartupDiagnostics`], if installed.
    #[must_use]
    pub fn startup_diagnostics(&self) -> Option<&StartupDiagnostics> {
        self.world().get_resource::<StartupDiagnostics>()
    }
}
