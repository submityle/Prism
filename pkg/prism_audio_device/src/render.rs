//! The shared pull-based renderer that bridges the fixed-block
//! [`AudioRuntime`] to the arbitrary, variable-size buffers that device
//! callbacks and file sinks request.
//!
//! Device SDKs hand the callback a buffer of whatever size they please (often
//! not a multiple of the engine's block size), while [`AudioRuntime`] renders
//! exactly one fixed block per call. [`BlockRenderer`] closes that gap: it
//! renders full engine blocks into an internal planar scratch buffer and drains
//! them into caller buffers frame by frame, carrying any partially consumed
//! block across calls. Every render happens up front on demand, so the callback
//! never starves as long as the host thread is scheduled in time.
//!
//! The steady-state path allocates nothing: the scratch buffer and the sample
//! math are sized at construction, matching the real-time contract of
//! [`AudioRuntime::process_block`].

use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
use prism_audio_core::math::Sample;
use prism_audio_rt::AudioRuntime;
use prism_audio_rt::telemetry::TelemetryFrame;

/// Pulls fixed engine blocks from an [`AudioRuntime`] and serves them into
/// arbitrarily sized interleaved output buffers.
///
/// Owns the runtime's real-time half; the paired [`AudioRuntimeClient`] (from
/// [`prism_audio_rt::runtime`]) stays on the control side for commands and
/// telemetry. Move a `BlockRenderer` into the device callback or drive it from
/// the offline file sink.
///
/// [`AudioRuntimeClient`]: prism_audio_rt::AudioRuntimeClient
#[derive(Debug)]
pub struct BlockRenderer {
    /// The real-time runtime half producing planar master output.
    runtime: AudioRuntime,
    /// Planar scratch holding the most recently rendered engine block.
    scratch: AudioBuffer,
    /// Number of valid frames currently held in `scratch`.
    scratch_filled: usize,
    /// Index of the next unconsumed frame within `scratch`.
    scratch_pos: usize,
    /// Output channel layout this renderer produces.
    layout: ChannelLayout,
    /// Telemetry from the most recently rendered block, for callers without a
    /// telemetry-ring consumer (e.g. the offline sink).
    last_telemetry: TelemetryFrame,
}

impl BlockRenderer {
    /// Builds a renderer that produces `layout` output, rendering the runtime
    /// in blocks of `block_frames`.
    ///
    /// `block_frames` is clamped to the runtime's configured maximum block so
    /// [`AudioRuntime::process_block`] never rejects the scratch buffer.
    ///
    /// # Panics
    ///
    /// Panics if `block_frames` is zero.
    #[must_use]
    pub fn new(runtime: AudioRuntime, layout: ChannelLayout, block_frames: usize) -> Self {
        assert!(block_frames > 0, "block_frames must be non-zero");
        let block_frames = block_frames.min(runtime.max_block());
        let block_frames = block_frames.max(1);
        Self {
            runtime,
            scratch: AudioBuffer::new(layout, block_frames),
            scratch_filled: 0,
            scratch_pos: 0,
            layout,
            last_telemetry: TelemetryFrame::empty(),
        }
    }

    /// Returns the output channel layout.
    #[inline]
    #[must_use]
    pub fn layout(&self) -> ChannelLayout {
        self.layout
    }

    /// Returns the number of output channels produced.
    #[inline]
    #[must_use]
    pub fn channels(&self) -> usize {
        self.layout.channel_count()
    }

    /// Returns the telemetry frame from the most recently rendered engine block.
    #[inline]
    #[must_use]
    pub fn last_telemetry(&self) -> TelemetryFrame {
        self.last_telemetry
    }

    /// Immutable access to the underlying runtime (state queries only).
    #[inline]
    #[must_use]
    pub fn runtime(&self) -> &AudioRuntime {
        &self.runtime
    }

    /// Fills the interleaved `out` slice with rendered audio.
    ///
    /// `out.len()` must be `frames * channels()`. Renders as many fresh engine
    /// blocks as needed and interleaves them into `out`. Real-time safe: no
    /// allocation, no locking, and no panics on the steady-state path.
    ///
    /// # Panics
    ///
    /// Panics if `out.len()` is not a multiple of [`BlockRenderer::channels`].
    pub fn render_interleaved(&mut self, out: &mut [Sample]) {
        let channels = self.channels();
        assert!(
            out.len().is_multiple_of(channels),
            "output length {} is not a multiple of channel count {channels}",
            out.len()
        );
        let total_frames = out.len() / channels;
        let mut written = 0usize;

        while written < total_frames {
            if self.scratch_pos >= self.scratch_filled {
                self.render_next_block();
            }
            let available = self.scratch_filled - self.scratch_pos;
            let take = available.min(total_frames - written);

            // Interleave `take` frames starting at `scratch_pos` into `out`
            // starting at frame `written`.
            for channel in 0..channels {
                let src = self.scratch.channel(channel);
                for frame in 0..take {
                    let interleaved_index = (written + frame) * channels + channel;
                    out[interleaved_index] = src[self.scratch_pos + frame];
                }
            }
            self.scratch_pos += take;
            written += take;
        }
    }

    /// Renders exactly `frames` interleaved frames into `out`, which must be
    /// pre-sized to `frames * channels()`.
    ///
    /// A convenience wrapper over [`BlockRenderer::render_interleaved`] used by
    /// the offline sink, sharing the identical render path as live playback.
    ///
    /// # Panics
    ///
    /// Panics if `out.len()` is not exactly `frames * channels()`.
    pub fn render_frames(&mut self, frames: usize, out: &mut [Sample]) {
        assert_eq!(
            out.len(),
            frames * self.channels(),
            "output slice must hold exactly frames * channels samples"
        );
        self.render_interleaved(out);
    }

    /// Renders one fresh engine block into the planar scratch buffer.
    fn render_next_block(&mut self) {
        let frame = self.runtime.process_block(&mut self.scratch);
        self.last_telemetry = frame;
        self.scratch_filled = self.scratch.active_frames();
        self.scratch_pos = 0;
    }
}
