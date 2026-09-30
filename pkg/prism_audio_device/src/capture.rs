//! Lock-free microphone / bus input capture.
//!
//! A capture stream's real-time callback pushes interleaved frames into a
//! bounded lock-free ring via [`CaptureSink`]; a control-side [`CaptureConsumer`]
//! drains whole frames into a planar [`AudioBuffer`] for recording, voice, or
//! spectrum UIs. The sink only ever enqueues *complete* frames, so the ring
//! stays frame-aligned even under overflow: when there is not room for a full
//! frame, the whole frame is dropped and counted rather than desynchronizing
//! the channel interleave.

use alloc::sync::Arc;
use core::sync::atomic::{AtomicU64, Ordering};

use prism_audio_core::buffer::AudioBuffer;
use prism_audio_core::math::Sample;
use prism_audio_rt::ring::{RingConsumer, RingProducer, ring};

/// Real-time producer half of a capture ring. Cheap to clone; move into the
/// device input callback.
#[derive(Clone)]
pub struct CaptureSink {
    /// Producer half of the interleaved-sample ring.
    tx: RingProducer<Sample>,
    /// Count of samples dropped because the ring lacked room for a full frame.
    dropped: Arc<AtomicU64>,
    /// Interleaved channel count the producer writes per frame.
    channels: usize,
}

impl CaptureSink {
    /// Pushes interleaved samples, enqueuing only complete frames.
    ///
    /// Any trailing partial frame in `data` is ignored. Frames that do not fit
    /// are dropped whole and added to the dropped-sample counter, keeping the
    /// ring frame-aligned. Returns the number of samples actually enqueued.
    /// Real-time safe: no allocation, no locking.
    pub fn push_interleaved(&self, data: &[Sample]) -> usize {
        if self.channels == 0 {
            return 0;
        }
        let capacity = self.tx.capacity();
        let mut pushed = 0usize;
        let frames = data.len() / self.channels;
        for frame in 0..frames {
            let free = capacity - self.tx.len();
            if free < self.channels {
                // No room for a whole frame: drop the remainder frame-aligned.
                let remaining_frames = frames - frame;
                self.dropped
                    .fetch_add((remaining_frames * self.channels) as u64, Ordering::Relaxed);
                break;
            }
            let base = frame * self.channels;
            for channel in 0..self.channels {
                // Room was checked above; a failed push here would only occur
                // under an impossible concurrent producer, so account for it as
                // a drop rather than panicking on the real-time thread.
                if self.tx.push(data[base + channel]).is_err() {
                    self.dropped.fetch_add(1, Ordering::Relaxed);
                } else {
                    pushed += 1;
                }
            }
        }
        pushed
    }

    /// Total samples dropped due to ring overflow since construction.
    #[must_use]
    #[inline]
    pub fn dropped_samples(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    /// The interleaved channel count this sink writes per frame.
    #[must_use]
    #[inline]
    pub fn channels(&self) -> usize {
        self.channels
    }
}

impl core::fmt::Debug for CaptureSink {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CaptureSink")
            .field("channels", &self.channels)
            .field("dropped", &self.dropped_samples())
            .field("queued", &self.tx.len())
            .finish()
    }
}

/// Control-side consumer half of a capture ring. Drain from a task or gameplay
/// thread; not real-time safe (borrows a small residual-free frame path but is
/// intended for non-callback threads).
#[derive(Debug)]
pub struct CaptureConsumer {
    /// Consumer half of the interleaved-sample ring.
    rx: RingConsumer<Sample>,
    /// Interleaved channel count matching the paired [`CaptureSink`].
    channels: usize,
}

impl CaptureConsumer {
    /// Drains up to `buffer.capacity_frames()` whole frames into `buffer`,
    /// de-interleaving device channels onto engine channels.
    ///
    /// Surplus device channels are dropped and missing engine channels are
    /// zero-filled. Sets the buffer's active frame count to the number of
    /// frames written and returns that count.
    pub fn drain_into(&mut self, buffer: &mut AudioBuffer) -> usize {
        if self.channels == 0 {
            buffer.set_active_frames(0);
            return 0;
        }
        let available_frames = self.rx.len() / self.channels;
        let frames = available_frames.min(buffer.capacity_frames());
        let engine_channels = buffer.channels();
        let shared = self.channels.min(engine_channels);

        for frame in 0..frames {
            // Pop one whole interleaved frame.
            for channel in 0..self.channels {
                let Some(sample) = self.rx.pop() else {
                    // The producer only enqueues whole frames and we bounded by
                    // observed length, so this is unreachable; treat a miss as
                    // end-of-data and finalize what we have.
                    buffer.set_active_frames(frame);
                    return frame;
                };
                if channel < shared {
                    buffer.channel_mut(channel)[frame] = sample;
                }
            }
            for channel in shared..engine_channels {
                buffer.channel_mut(channel)[frame] = 0.0;
            }
        }
        buffer.set_active_frames(frames);
        frames
    }

    /// The interleaved channel count this consumer expects per frame.
    #[must_use]
    #[inline]
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Number of whole frames currently buffered and ready to drain.
    #[must_use]
    #[inline]
    pub fn buffered_frames(&self) -> usize {
        if self.channels == 0 {
            return 0;
        }
        self.rx.len() / self.channels
    }
}

/// Creates a capture ring holding `capacity_frames` interleaved frames of
/// `channels` each, returning the real-time sink and control-side consumer.
///
/// # Panics
///
/// Panics if `channels` is zero.
#[must_use]
pub fn capture_ring(channels: usize, capacity_frames: usize) -> (CaptureSink, CaptureConsumer) {
    assert!(channels > 0, "capture requires at least one channel");
    let (tx, rx) = ring::<Sample>(capacity_frames.max(1) * channels);
    let sink = CaptureSink {
        tx,
        dropped: Arc::new(AtomicU64::new(0)),
        channels,
    };
    let consumer = CaptureConsumer { rx, channels };
    (sink, consumer)
}

#[cfg(test)]
mod tests {
    use super::capture_ring;
    use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};

    #[test]
    fn frames_round_trip_through_the_capture_ring() {
        let (sink, mut consumer) = capture_ring(2, 8);
        let pushed = sink.push_interleaved(&[0.1, -0.1, 0.2, -0.2, 0.3, -0.3]);
        assert_eq!(pushed, 6);
        assert_eq!(consumer.buffered_frames(), 3);

        let mut buffer = AudioBuffer::new(ChannelLayout::Stereo, 4);
        let frames = consumer.drain_into(&mut buffer);
        assert_eq!(frames, 3);
        assert_eq!(buffer.active_frames(), 3);
        assert_eq!(buffer.channel(0), &[0.1, 0.2, 0.3]);
        assert_eq!(buffer.channel(1), &[-0.1, -0.2, -0.3]);
        assert_eq!(consumer.buffered_frames(), 0);
    }

    #[test]
    fn overflow_drops_whole_frames_and_counts_them() {
        // Room for exactly two stereo frames.
        let (sink, mut consumer) = capture_ring(2, 2);
        let pushed = sink.push_interleaved(&[1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
        assert_eq!(pushed, 4, "only two frames fit");
        assert_eq!(sink.dropped_samples(), 2, "the third frame's two samples dropped");

        let mut buffer = AudioBuffer::new(ChannelLayout::Stereo, 4);
        assert_eq!(consumer.drain_into(&mut buffer), 2);
        assert_eq!(buffer.channel(0), &[1.0, 2.0]);
    }

    #[test]
    fn trailing_partial_frame_is_ignored() {
        let (sink, consumer) = capture_ring(2, 8);
        // Five samples = two whole stereo frames + one dangling sample.
        let pushed = sink.push_interleaved(&[1.0, 1.0, 2.0, 2.0, 3.0]);
        assert_eq!(pushed, 4);
        assert_eq!(consumer.buffered_frames(), 2);
    }
}
