//! Read-back telemetry published by the audio thread over the
//! [telemetry ring](crate::ring).
//!
//! [`TelemetryFrame`] is a small `Copy` snapshot emitted once per processed
//! block. It flows from the single audio-thread producer to an observer thread
//! (an ECS diagnostics system, a profiler overlay, or a remote authoring tool)
//! without allocation or locking. Because the ring is bounded, an observer that
//! falls behind simply misses intermediate frames rather than stalling the
//! audio thread.

/// A per-block snapshot of the audio runtime's state.
///
/// All fields are plain scalars so the frame travels through the lock-free
/// [telemetry ring](crate::ring) by value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TelemetryFrame {
    /// Monotonically increasing index of the processed block, starting at `0`.
    pub block_index: u64,
    /// Playhead position, in frames, *after* this block was rendered.
    pub playhead: u64,
    /// Number of frames rendered in this block.
    pub frames: u32,
    /// Physical (audible) voice count observed at the end of the block.
    pub physical_voices: u32,
    /// Virtual (culled but retained) voice count at the end of the block.
    pub virtual_voices: u32,
    /// Peak absolute sample magnitude of the master output this block.
    pub master_peak: f32,
    /// Root-mean-square level of the master output this block.
    pub master_rms: f32,
    /// Fraction of the block's wall-clock budget spent inside processing, where
    /// `1.0` means the block took exactly its real-time duration to render.
    /// Values above `1.0` indicate an overrun risk.
    pub cpu_load: f32,
}

impl TelemetryFrame {
    /// A zeroed frame, useful as an initial value before the first block runs.
    #[must_use]
    #[inline]
    pub const fn empty() -> Self {
        Self {
            block_index: 0,
            playhead: 0,
            frames: 0,
            physical_voices: 0,
            virtual_voices: 0,
            master_peak: 0.0,
            master_rms: 0.0,
            cpu_load: 0.0,
        }
    }
}

impl Default for TelemetryFrame {
    #[inline]
    fn default() -> Self {
        Self::empty()
    }
}
