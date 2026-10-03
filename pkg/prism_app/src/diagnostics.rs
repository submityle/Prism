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
//!   time, a per-phase [`FrameStats`] for each core frame phase, a
//!   [`CountWindow`] of the fixed-timestep **substep count** per frame, the
//!   per-frame secondary-sub-app **extract cost**, and — under the `pipelined`
//!   feature — the simulate/render **pipeline-overlap rate**.
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
//! # Extract cost and pipeline overlap
//!
//! Two more design §16 frame metrics *are* measured, because they are real
//! wall-clock quantities `prism_app` itself drives:
//!
//! - **Extract cost** ([`FrameDiagnostics::extract_time`]) — the per-frame time
//!   spent running every secondary sub-app's
//!   [`ExtractFn`](crate::sub_app::ExtractFn) against the just-simulated main
//!   world (the one-way `main → sub` seam, design §21 / §25.2). Recorded by the
//!   serial path and the `pipelined` executor alike, so it means the same thing
//!   either way.
//! - **Pipeline-overlap rate** ([`FrameDiagnostics::pipeline_overlap_ratio`]) —
//!   under the `pipelined` feature, how much of a secondary's render was hidden
//!   behind the next frame's simulation, derived from the measured simulate
//!   time ([`pipeline_sim`](FrameDiagnostics::pipeline_sim)) and the measured
//!   join-wait tail ([`pipeline_wait`](FrameDiagnostics::pipeline_wait)). It is
//!   recorded only on frames with a prior render actually in flight, so the
//!   first frame is skipped rather than scored as a spurious perfect overlap.
//!
//! # Honestly deferred
//!
//! - **Present latency** (design §16) is *not* measured here: it needs a
//!   `prism_window`/RHI present timestamp that does not exist yet. Measuring it
//!   now would mean fabricating numbers, so it is documented as absent rather
//!   than stubbed.
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
    extract_time: FrameStats,
    pipeline_sim: FrameStats,
    pipeline_wait: FrameStats,
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
            extract_time: FrameStats::new(window),
            pipeline_sim: FrameStats::new(window),
            pipeline_wait: FrameStats::new(window),
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

    /// Record this frame's total secondary-sub-app **extract** time
    /// (design §16: *"extract 耗时"*).
    ///
    /// This is the wall time spent running every secondary sub-app's
    /// [`ExtractFn`](crate::sub_app::ExtractFn) against the just-simulated main
    /// world — the one-way `main → sub` synchronization point (design §21,
    /// §25.2). It is recorded by both the serial per-frame path
    /// ([`SubApps::update`](crate::sub_app::SubApps)) and, under the
    /// `pipelined` feature, the cross-thread executor, so the metric means the
    /// same thing on either path. An app with no secondary sub-apps records a
    /// near-zero duration each frame rather than nothing.
    pub fn record_extract(&mut self, elapsed: Duration) {
        self.extract_time.record(elapsed);
    }

    /// Record one pipelined frame's overlap sample: `sim` is the main-thread
    /// simulation time that ran **while the previous frame's render was in
    /// flight**, and `wait` is the time the main thread then blocked joining
    /// that render (design §16: *"流水线重叠率"*).
    ///
    /// Only the `pipelined` cross-thread executor records this, and only on
    /// frames that actually had a prior render in flight (the first frame has
    /// no overlap to measure, so it is skipped rather than recorded as a
    /// spurious perfect overlap). See
    /// [`pipeline_overlap_ratio`](Self::pipeline_overlap_ratio) for the derived
    /// rate.
    pub fn record_pipeline_overlap(&mut self, sim: Duration, wait: Duration) {
        self.pipeline_sim.record(sim);
        self.pipeline_wait.record(wait);
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

    /// Rolling secondary-sub-app extract-time statistics (design §16).
    ///
    /// Empty ([`FrameStats::is_empty`]) until the first instrumented frame with
    /// at least one secondary sub-app has run.
    #[must_use]
    pub fn extract_time(&self) -> &FrameStats {
        &self.extract_time
    }

    /// Rolling statistics for the pipelined per-frame **simulation** time that
    /// overlapped the previous frame's render (design §16). Empty until the
    /// `pipelined` executor has driven at least one overlapped frame.
    #[must_use]
    pub fn pipeline_sim(&self) -> &FrameStats {
        &self.pipeline_sim
    }

    /// Rolling statistics for the time the main thread **blocked** joining the
    /// overlapped render after simulation finished (design §16). A near-zero
    /// mean means the render fit entirely inside the simulation window; a large
    /// mean means the render overran and stalled the next frame.
    #[must_use]
    pub fn pipeline_wait(&self) -> &FrameStats {
        &self.pipeline_wait
    }

    /// The derived pipeline-overlap ratio in `0.0..=1.0` (design §16:
    /// *"流水线重叠率"*), or `None` before any overlapped frame is recorded.
    ///
    /// Computed from the mean simulation and mean join-wait times as
    /// `mean_sim / (mean_sim + mean_wait)`:
    ///
    /// - `1.0` — the overlapped render finished before simulation did, so it
    ///   was fully hidden behind the next frame's simulation (ideal).
    /// - `< 1.0` — the render overran the simulation window by the wait tail,
    ///   so throughput is bounded by the render, not the simulation.
    ///
    /// A zero mean simulation time (sub-microsecond frames) yields `None`
    /// rather than a meaningless ratio.
    #[must_use]
    pub fn pipeline_overlap_ratio(&self) -> Option<f64> {
        let sim = self.pipeline_sim.average()?.as_secs_f64();
        let wait = self.pipeline_wait.average()?.as_secs_f64();
        let denom = sim + wait;
        if denom > 0.0 {
            Some(sim / denom)
        } else {
            None
        }
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
