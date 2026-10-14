//! Bounded, allocation-aware dock frame instrumentation.
//!
//! The dock crate owns the frame record and its stage ordering, but it does
//! not own a clock. A host starts a [`DockFrameSample`], records the measured
//! boundaries as each downstream stage completes, and commits the completed
//! sample to [`DockRuntimeMetrics`]. The five stages deliberately match the
//! application hand-off points:
//!
//! `Host solve -> Element update -> DrawList lowering -> GPU submit -> present`
//!
//! Host code must provide durations measured by an appropriate real clock.
//! In particular, `gpu_submission_ns` is the measured CPU-side submit call
//! unless the caller explicitly supplies a different documented measurement,
//! and `present_latency_ns` is accepted only as presentation feedback from
//! the window/present layer. No platform timing is synthesized here.

use alloc::vec::Vec;

const MAX_SAMPLES: usize = 256;
const HOST_STAGE: u8 = 1 << 0;
const ELEMENT_STAGE: u8 = 1 << 1;
const DRAW_LIST_STAGE: u8 = 1 << 2;
const GPU_STAGE: u8 = 1 << 3;
const PRESENT_STAGE: u8 = 1 << 4;
const HOST_ELEMENT_STAGES: u8 = HOST_STAGE | ELEMENT_STAGE;
const HOST_ELEMENT_DRAW_LIST_STAGES: u8 = HOST_ELEMENT_STAGES | DRAW_LIST_STAGE;
const HOST_ELEMENT_DRAW_LIST_GPU_STAGES: u8 = HOST_ELEMENT_DRAW_LIST_STAGES | GPU_STAGE;
const COMPLETE_STAGES: u8 =
    HOST_STAGE | ELEMENT_STAGE | DRAW_LIST_STAGE | GPU_STAGE | PRESENT_STAGE;

/// One ordered, host-supplied observation for a dock frame.
///
/// Construct samples with [`DockFrameSample::new`]. The stage methods enforce
/// the renderer hand-off order and reject duplicate or skipped stages. This
/// keeps a partially observed frame from being mistaken for a complete one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DockFrameSample {
    frame_id: u64,
    solve_ns: u64,
    element_update_ns: u64,
    draw_list_ns: u64,
    draw_commands: u32,
    visible_panels: u32,
    factory_builds: u32,
    gpu_submission_ns: u64,
    present_latency_ns: u64,
    dropped_frame: bool,
    stages: u8,
}
impl DockFrameSample {
    /// Starts an empty sample for an application-owned frame identifier.
    #[must_use]
    pub const fn new(frame_id: u64) -> Self {
        Self {
            frame_id,
            solve_ns: 0,
            element_update_ns: 0,
            draw_list_ns: 0,
            draw_commands: 0,
            visible_panels: 0,
            factory_builds: 0,
            gpu_submission_ns: 0,
            present_latency_ns: 0,
            dropped_frame: false,
            stages: 0,
        }
    }

    /// Returns the application-owned frame identifier.
    #[must_use]
    pub const fn frame_id(self) -> u64 {
        self.frame_id
    }

    /// Records the Host layout-solve boundary.
    ///
    /// `duration_ns` must come from the host's monotonic timing source. The
    /// dock crate does not measure or infer it.
    #[must_use]
    pub fn host_solved(&mut self, duration_ns: u64, dropped_frame: bool) -> bool {
        if !self.accept_stage(HOST_STAGE) {
            return false;
        }
        self.solve_ns = duration_ns;
        self.dropped_frame = dropped_frame;
        true
    }

    /// Records the retained `Element` update/reconciliation boundary.
    ///
    /// `duration_ns` must be measured around the actual Element update call.
    /// The counts describe the views/factories touched by that same update,
    /// rather than a later `DrawList` or GPU pass.
    #[must_use]
    pub fn element_updated(
        &mut self,
        duration_ns: u64,
        visible_panels: u32,
        factory_builds: u32,
    ) -> bool {
        if !self.accept_stage(ELEMENT_STAGE) {
            return false;
        }
        self.element_update_ns = duration_ns;
        self.visible_panels = visible_panels;
        self.factory_builds = factory_builds;
        true
    }

    /// Records the `Element -> DrawList` lowering boundary.
    ///
    /// `command_count` is the length of the exact `DrawList` handed to the
    /// renderer, not an estimate of the eventual GPU batch count.
    #[must_use]
    pub fn draw_list_built(&mut self, duration_ns: u64, command_count: u32) -> bool {
        if !self.accept_stage(DRAW_LIST_STAGE) {
            return false;
        }
        self.draw_list_ns = duration_ns;
        self.draw_commands = command_count;
        true
    }

    /// Records the CPU-side GPU submission boundary.
    ///
    /// This is intentionally a submit-call duration. GPU execution time must
    /// be measured by the graphics backend and supplied through this method
    /// only when the caller has converted that measurement to nanoseconds.
    #[must_use]
    pub fn gpu_submitted(&mut self, duration_ns: u64) -> bool {
        if !self.accept_stage(GPU_STAGE) {
            return false;
        }
        self.gpu_submission_ns = duration_ns;
        true
    }

    /// Records presentation feedback for this frame.
    ///
    /// The value must be supplied by the window/present layer. The dock crate
    /// never treats a CPU timestamp or a fixed refresh interval as a present
    /// measurement.
    #[must_use]
    pub fn presented(&mut self, latency_ns: u64) -> bool {
        if !self.accept_stage(PRESENT_STAGE) {
            return false;
        }
        self.present_latency_ns = latency_ns;
        true
    }

    /// Returns whether all five hand-off stages have been recorded.
    #[must_use]
    pub const fn is_complete(self) -> bool {
        self.stages == COMPLETE_STAGES
    }

    /// Finalizes the sample, returning `None` if a stage is missing.
    #[must_use]
    pub const fn finish(self) -> Option<Self> {
        if self.is_complete() {
            Some(self)
        } else {
            None
        }
    }

    /// Layout-solve duration supplied by the host.
    #[must_use]
    pub const fn solve_ns(self) -> u64 {
        self.solve_ns
    }

    /// Retained Element update duration supplied by the host.
    #[must_use]
    pub const fn element_update_ns(self) -> u64 {
        self.element_update_ns
    }

    /// Element-to-DrawList lowering duration supplied by the host.
    #[must_use]
    pub const fn draw_list_ns(self) -> u64 {
        self.draw_list_ns
    }

    /// Number of commands in the exact `DrawList` handed to the renderer.
    #[must_use]
    pub const fn draw_commands(self) -> u32 {
        self.draw_commands
    }

    /// Number of visible content panels reported by the host.
    #[must_use]
    pub const fn visible_panels(self) -> u32 {
        self.visible_panels
    }

    /// Number of view-factory builds reported by the host.
    #[must_use]
    pub const fn factory_builds(self) -> u32 {
        self.factory_builds
    }

    /// GPU submission duration supplied by the caller.
    #[must_use]
    pub const fn gpu_submission_ns(self) -> u64 {
        self.gpu_submission_ns
    }

    /// Present latency supplied by presentation feedback.
    #[must_use]
    pub const fn present_latency_ns(self) -> u64 {
        self.present_latency_ns
    }

    /// Whether the host marked this frame as dropped.
    #[must_use]
    pub const fn dropped_frame(self) -> bool {
        self.dropped_frame
    }

    fn accept_stage(&mut self, stage: u8) -> bool {
        let expected = match self.stages {
            0 => HOST_STAGE,
            HOST_STAGE => ELEMENT_STAGE,
            HOST_ELEMENT_STAGES => DRAW_LIST_STAGE,
            HOST_ELEMENT_DRAW_LIST_STAGES => GPU_STAGE,
            HOST_ELEMENT_DRAW_LIST_GPU_STAGES => PRESENT_STAGE,
            _ => 0,
        };
        if stage != expected {
            return false;
        }
        self.stages |= stage;
        true
    }
}
/// Percentile summary over the retained frame samples.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DockMetricSummary {
    /// Number of samples included in the summary.
    pub samples: usize,
    /// Layout solve P50 in nanoseconds.
    pub solve_p50_ns: u64,
    /// Layout solve P95 in nanoseconds.
    pub solve_p95_ns: u64,
    /// Layout solve P99 in nanoseconds.
    pub solve_p99_ns: u64,
    /// Element update P50 in nanoseconds.
    pub element_p50_ns: u64,
    /// Element update P95 in nanoseconds.
    pub element_p95_ns: u64,
    /// Element update P99 in nanoseconds.
    pub element_p99_ns: u64,
    /// `DrawList` lowering P50 in nanoseconds.
    pub draw_list_p50_ns: u64,
    /// `DrawList` lowering P95 in nanoseconds.
    pub draw_list_p95_ns: u64,
    /// `DrawList` lowering P99 in nanoseconds.
    pub draw_list_p99_ns: u64,
    /// Number of commands handed to the renderer at P50.
    pub draw_commands_p50: u32,
    /// Number of commands handed to the renderer at P95.
    pub draw_commands_p95: u32,
    /// Number of commands handed to the renderer at P99.
    pub draw_commands_p99: u32,
    /// Visible panel count at P50.
    pub visible_panels_p50: u32,
    /// Visible panel count at P95.
    pub visible_panels_p95: u32,
    /// Visible panel count at P99.
    pub visible_panels_p99: u32,
    /// Factory-build count at P50.
    pub factory_builds_p50: u32,
    /// Factory-build count at P95.
    pub factory_builds_p95: u32,
    /// Factory-build count at P99.
    pub factory_builds_p99: u32,
    /// GPU submission P50 in nanoseconds.
    pub gpu_p50_ns: u64,
    /// GPU submission P95 in nanoseconds.
    pub gpu_p95_ns: u64,
    /// GPU submission P99 in nanoseconds.
    pub gpu_p99_ns: u64,
    /// Presentation latency P50 in nanoseconds.
    pub present_p50_ns: u64,
    /// Presentation latency P95 in nanoseconds.
    pub present_p95_ns: u64,
    /// Presentation latency P99 in nanoseconds.
    pub present_p99_ns: u64,
    /// Total dropped-frame count observed by this collector.
    pub dropped_frames: u64,
}

/// Fixed-capacity metrics collector.
#[derive(Clone, Debug)]
pub struct DockRuntimeMetrics {
    samples: Vec<DockFrameSample>,
    oldest: usize,
    dropped_frames: u64,
}

impl Default for DockRuntimeMetrics {
    fn default() -> Self {
        Self::new()
    }
}
impl DockRuntimeMetrics {
    /// Creates an empty collector with storage reserved for the retention cap.
    #[must_use]
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
            oldest: 0,
            dropped_frames: 0,
        }
    }

    /// Starts a frame instrumentation record.
    #[must_use]
    pub const fn begin_frame(frame_id: u64) -> DockFrameSample {
        DockFrameSample::new(frame_id)
    }

    /// Records a complete frame, returning `false` for an incomplete or
    /// incorrectly constructed sample.
    #[must_use]
    pub fn record(&mut self, sample: DockFrameSample) -> bool {
        let Some(sample) = sample.finish() else {
            return false;
        };
        if sample.dropped_frame() {
            self.dropped_frames = self.dropped_frames.saturating_add(1);
        }
        if self.samples.is_empty() {
            self.samples.reserve_exact(MAX_SAMPLES);
        }
        if self.samples.len() < MAX_SAMPLES {
            self.samples.push(sample);
        } else {
            self.samples[self.oldest] = sample;
            self.oldest = (self.oldest + 1) % MAX_SAMPLES;
        }
        true
    }

    /// Returns the number of retained observations.
    #[must_use]
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Returns whether no observations are retained.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Builds deterministic P50/P95/P99 summaries.
    #[must_use]
    pub fn summary(&self) -> DockMetricSummary {
        let mut values = [0; MAX_SAMPLES];
        let mut count_values = [0; MAX_SAMPLES];
        DockMetricSummary {
            samples: self.samples.len(),
            solve_p50_ns: self.percentile(&mut values, DockFrameSample::solve_ns, 50),
            solve_p95_ns: self.percentile(&mut values, DockFrameSample::solve_ns, 95),
            solve_p99_ns: self.percentile(&mut values, DockFrameSample::solve_ns, 99),
            element_p50_ns: self.percentile(&mut values, DockFrameSample::element_update_ns, 50),
            element_p95_ns: self.percentile(&mut values, DockFrameSample::element_update_ns, 95),
            element_p99_ns: self.percentile(&mut values, DockFrameSample::element_update_ns, 99),
            draw_list_p50_ns: self.percentile(&mut values, DockFrameSample::draw_list_ns, 50),
            draw_list_p95_ns: self.percentile(&mut values, DockFrameSample::draw_list_ns, 95),
            draw_list_p99_ns: self.percentile(&mut values, DockFrameSample::draw_list_ns, 99),
            draw_commands_p50: self.percentile_u32(
                &mut count_values,
                DockFrameSample::draw_commands,
                50,
            ),
            draw_commands_p95: self.percentile_u32(
                &mut count_values,
                DockFrameSample::draw_commands,
                95,
            ),
            draw_commands_p99: self.percentile_u32(
                &mut count_values,
                DockFrameSample::draw_commands,
                99,
            ),
            visible_panels_p50: self.percentile_u32(
                &mut count_values,
                DockFrameSample::visible_panels,
                50,
            ),
            visible_panels_p95: self.percentile_u32(
                &mut count_values,
                DockFrameSample::visible_panels,
                95,
            ),
            visible_panels_p99: self.percentile_u32(
                &mut count_values,
                DockFrameSample::visible_panels,
                99,
            ),
            factory_builds_p50: self.percentile_u32(
                &mut count_values,
                DockFrameSample::factory_builds,
                50,
            ),
            factory_builds_p95: self.percentile_u32(
                &mut count_values,
                DockFrameSample::factory_builds,
                95,
            ),
            factory_builds_p99: self.percentile_u32(
                &mut count_values,
                DockFrameSample::factory_builds,
                99,
            ),
            gpu_p50_ns: self.percentile(&mut values, DockFrameSample::gpu_submission_ns, 50),
            gpu_p95_ns: self.percentile(&mut values, DockFrameSample::gpu_submission_ns, 95),
            gpu_p99_ns: self.percentile(&mut values, DockFrameSample::gpu_submission_ns, 99),
            present_p50_ns: self.percentile(&mut values, DockFrameSample::present_latency_ns, 50),
            present_p95_ns: self.percentile(&mut values, DockFrameSample::present_latency_ns, 95),
            present_p99_ns: self.percentile(&mut values, DockFrameSample::present_latency_ns, 99),
            dropped_frames: self.dropped_frames,
        }
    }

    /// Clears samples and dropped-frame counters.
    pub fn clear(&mut self) {
        self.samples.clear();
        self.oldest = 0;
        self.dropped_frames = 0;
    }

    fn percentile(
        &self,
        values: &mut [u64; MAX_SAMPLES],
        value: fn(DockFrameSample) -> u64,
        rank: usize,
    ) -> u64 {
        if self.samples.is_empty() {
            return 0;
        }
        let count = self.samples.len();
        for (offset, slot) in values.iter_mut().take(count).enumerate() {
            let index = if self.samples.len() < MAX_SAMPLES {
                offset
            } else {
                (self.oldest + offset) % self.samples.len()
            };
            *slot = value(self.samples[index]);
        }
        values[..count].sort_unstable();
        percentile_rank(&values[..count], rank)
    }

    fn percentile_u32(
        &self,
        values: &mut [u32; MAX_SAMPLES],
        value: fn(DockFrameSample) -> u32,
        rank: usize,
    ) -> u32 {
        if self.samples.is_empty() {
            return 0;
        }
        let count = self.samples.len();
        for (offset, slot) in values.iter_mut().take(count).enumerate() {
            let index = if self.samples.len() < MAX_SAMPLES {
                offset
            } else {
                (self.oldest + offset) % self.samples.len()
            };
            *slot = value(self.samples[index]);
        }
        values[..count].sort_unstable();
        percentile_rank_u32(&values[..count], rank)
    }
}

fn percentile_rank(values: &[u64], percentile: usize) -> u64 {
    if values.is_empty() {
        return 0;
    }
    let rank = (values.len() - 1)
        .saturating_mul(percentile)
        .saturating_add(99)
        / 100;
    values[rank.min(values.len() - 1)]
}

fn percentile_rank_u32(values: &[u32], percentile: usize) -> u32 {
    if values.is_empty() {
        return 0;
    }
    let rank = (values.len() - 1)
        .saturating_mul(percentile)
        .saturating_add(99)
        / 100;
    values[rank.min(values.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn complete(frame_id: u64, value: u64, dropped: bool) -> DockFrameSample {
        let mut sample = DockFrameSample::new(frame_id);
        assert!(sample.host_solved(value, dropped));
        assert!(sample.element_updated(value * 2, 3, 1));
        assert!(sample.draw_list_built(value * 3, 7));
        assert!(sample.gpu_submitted(value * 4));
        assert!(sample.presented(value * 5));
        assert!(sample.is_complete());
        sample
    }

    #[test]
    fn stage_order_and_completion_are_enforced() {
        let mut sample = DockFrameSample::new(42);
        assert!(!sample.element_updated(2, 0, 0));
        assert!(sample.host_solved(1, false));
        assert!(!sample.host_solved(2, false));
        assert!(sample.element_updated(2, 4, 2));
        assert!(sample.draw_list_built(3, 9));
        assert!(sample.gpu_submitted(4));
        assert!(sample.finish().is_none());
        assert!(sample.presented(5));
        assert_eq!(sample.finish().unwrap().frame_id(), 42);
    }

    #[test]
    fn collector_rejects_partial_frames_and_summarizes_all_boundaries() {
        let mut metrics = DockRuntimeMetrics::new();
        assert!(!metrics.record(DockFrameSample::new(1)));
        assert!(metrics.record(complete(2, 10, false)));
        assert!(metrics.record(complete(3, 20, true)));
        let summary = metrics.summary();
        assert_eq!(summary.samples, 2);
        assert_eq!(summary.solve_p50_ns, 20);
        assert_eq!(summary.draw_list_p50_ns, 60);
        assert_eq!(summary.draw_commands_p50, 7);
        assert_eq!(summary.visible_panels_p50, 3);
        assert_eq!(summary.factory_builds_p50, 1);
        assert_eq!(summary.present_p50_ns, 100);
        assert_eq!(summary.dropped_frames, 1);
    }

    #[test]
    fn retention_is_a_bounded_ring_without_shifting() {
        let mut metrics = DockRuntimeMetrics::new();
        for value in 1..=(MAX_SAMPLES as u64 + 2) {
            assert!(metrics.record(complete(value, value, false)));
        }
        assert_eq!(metrics.len(), MAX_SAMPLES);
        let summary = metrics.summary();
        assert_eq!(summary.solve_p50_ns, 131);
        assert!(summary.solve_p50_ns <= summary.solve_p95_ns);
        assert!(summary.solve_p95_ns <= summary.solve_p99_ns);
    }

    #[test]
    fn clear_resets_ring_and_dropped_counter() {
        let mut metrics = DockRuntimeMetrics::new();
        assert!(metrics.record(complete(1, 10, true)));
        metrics.clear();
        assert!(metrics.is_empty());
        assert_eq!(metrics.summary(), DockMetricSummary::default());
    }
}
