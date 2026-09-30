//! Allocation-free conversion between the engine's planar [`AudioBuffer`] and
//! the interleaved sample layout expected by device SDKs and WAV files.
//!
//! Device callbacks and WAV frames are interleaved (`L R L R ...`), while the
//! render graph works in planar blocks (`L L ... | R R ...`). These helpers do
//! the transpose in place over caller-owned slices, so they never allocate and
//! are safe to call from the real-time audio callback.

use prism_audio_core::buffer::AudioBuffer;
use prism_audio_core::math::Sample;

/// Writes `frames` planar frames from `buffer` into the interleaved `out`
/// slice, mapping engine channels onto device channels.
///
/// The number of device channels is inferred from `out.len() / frames`. When
/// the device has more channels than the buffer, the extra channels are filled
/// with silence; when it has fewer, the surplus engine channels are dropped.
/// This lets a stereo graph feed a mono or surround endpoint without a separate
/// down/up-mix pass in the hot path.
///
/// # Panics
///
/// Panics if `out.len()` is not an exact multiple of `frames` (a malformed
/// device buffer), or if `frames` exceeds the buffer's active frame count.
pub fn planar_to_interleaved(buffer: &AudioBuffer, frames: usize, out: &mut [Sample]) {
    if frames == 0 {
        return;
    }
    assert!(
        out.len().is_multiple_of(frames),
        "interleaved output length {} is not a multiple of frame count {frames}",
        out.len()
    );
    assert!(
        frames <= buffer.active_frames(),
        "requested {frames} frames but buffer only has {} active",
        buffer.active_frames()
    );
    let device_channels = out.len() / frames;
    let engine_channels = buffer.channels();
    let shared = device_channels.min(engine_channels);

    for channel in 0..shared {
        let src = buffer.channel(channel);
        for frame in 0..frames {
            out[frame * device_channels + channel] = src[frame];
        }
    }
    // Silence any device channels the graph does not drive.
    for channel in shared..device_channels {
        for frame in 0..frames {
            out[frame * device_channels + channel] = 0.0;
        }
    }
}

/// Reads `frames` interleaved frames from `input` into the planar `buffer`,
/// mapping device channels onto engine channels and updating the buffer's
/// active frame count.
///
/// The device channel count is inferred from `input.len() / frames`. Surplus
/// device channels are dropped and missing engine channels are zero-filled, so
/// a mono microphone can feed a stereo capture buffer and vice versa.
///
/// # Panics
///
/// Panics if `input.len()` is not an exact multiple of `frames`, or if `frames`
/// exceeds the buffer capacity.
pub fn interleaved_to_planar(input: &[Sample], frames: usize, buffer: &mut AudioBuffer) {
    if frames == 0 {
        buffer.set_active_frames(0);
        return;
    }
    assert!(
        input.len().is_multiple_of(frames),
        "interleaved input length {} is not a multiple of frame count {frames}",
        input.len()
    );
    assert!(
        frames <= buffer.capacity_frames(),
        "requested {frames} frames but buffer capacity is {}",
        buffer.capacity_frames()
    );
    let device_channels = input.len() / frames;
    let engine_channels = buffer.channels();
    let shared = device_channels.min(engine_channels);
    buffer.set_active_frames(frames);

    for channel in 0..shared {
        let dst = buffer.channel_mut(channel);
        for frame in 0..frames {
            dst[frame] = input[frame * device_channels + channel];
        }
    }
    // Zero engine channels the device did not supply.
    for channel in shared..engine_channels {
        let dst = buffer.channel_mut(channel);
        for sample in dst[..frames].iter_mut() {
            *sample = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{interleaved_to_planar, planar_to_interleaved};
    use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};

    #[test]
    fn planar_stereo_round_trips_through_interleaved() {
        let mut planar = AudioBuffer::new(ChannelLayout::Stereo, 4);
        planar.channel_mut(0).copy_from_slice(&[0.1, 0.2, 0.3, 0.4]);
        planar.channel_mut(1).copy_from_slice(&[-0.1, -0.2, -0.3, -0.4]);

        let mut interleaved = [0.0f32; 8];
        planar_to_interleaved(&planar, 4, &mut interleaved);
        assert_eq!(
            interleaved,
            [0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4]
        );

        let mut restored = AudioBuffer::new(ChannelLayout::Stereo, 4);
        interleaved_to_planar(&interleaved, 4, &mut restored);
        assert_eq!(restored.channel(0), &[0.1, 0.2, 0.3, 0.4]);
        assert_eq!(restored.channel(1), &[-0.1, -0.2, -0.3, -0.4]);
    }

    #[test]
    fn extra_device_channels_are_silenced() {
        let mut planar = AudioBuffer::new(ChannelLayout::Mono, 2);
        planar.channel_mut(0).copy_from_slice(&[1.0, 2.0]);

        // Three device channels from a single engine channel.
        let mut interleaved = [9.0f32; 6];
        planar_to_interleaved(&planar, 2, &mut interleaved);
        assert_eq!(interleaved, [1.0, 0.0, 0.0, 2.0, 0.0, 0.0]);
    }

    #[test]
    fn surplus_device_channels_are_dropped_on_capture() {
        // Stereo device input into a mono capture buffer keeps only channel 0.
        let interleaved = [1.0f32, 5.0, 2.0, 6.0, 3.0, 7.0];
        let mut mono = AudioBuffer::new(ChannelLayout::Mono, 3);
        interleaved_to_planar(&interleaved, 3, &mut mono);
        assert_eq!(mono.active_frames(), 3);
        assert_eq!(mono.channel(0), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn zero_frames_is_a_noop_for_output_and_clears_capture() {
        let planar = AudioBuffer::new(ChannelLayout::Stereo, 4);
        let mut out = [7.0f32; 0];
        planar_to_interleaved(&planar, 0, &mut out);

        let mut capture = AudioBuffer::new(ChannelLayout::Stereo, 4);
        interleaved_to_planar(&[], 0, &mut capture);
        assert_eq!(capture.active_frames(), 0);
    }
}
