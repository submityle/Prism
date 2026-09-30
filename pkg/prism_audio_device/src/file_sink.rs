//! Deterministic offline rendering to a WAV file.
//!
//! The offline sink drives the exact same [`BlockRenderer`] pull path as live
//! device playback, so an offline render is sample-identical to what the device
//! backend would produce for the same graph and command stream. This is the
//! basis for golden regression tests and pre-rendered cutscene audio.
//!
//! Output is 32-bit float WAV (`WAVE_FORMAT_IEEE_FLOAT`), preserving the
//! engine's internal sample format without a lossy quantization step.

use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;
use std::path::Path;

use prism_audio_core::math::Sample;

use crate::error::DeviceError;
use crate::render::BlockRenderer;

/// Frames rendered per offline chunk. Large enough to amortize per-chunk
/// overhead, small enough to keep the interleave scratch modest.
const OFFLINE_CHUNK_FRAMES: usize = 1024;

/// Renders `total_frames` frames from `renderer` to a 32-bit float WAV file at
/// `path`.
///
/// The render uses the shared [`BlockRenderer`] path, so it reflects the live
/// graph, master-gain envelope, and any commands already applied. Returns the
/// number of frames written (always `total_frames`).
///
/// # Errors
///
/// Returns [`DeviceError::Wav`] if the file cannot be created, a sample cannot
/// be encoded, or the stream cannot be finalized.
pub fn render_to_wav(
    renderer: &mut BlockRenderer,
    path: impl AsRef<Path>,
    total_frames: u64,
) -> Result<u64, DeviceError> {
    let channels = renderer.channels();
    let spec = hound::WavSpec {
        channels: u16::try_from(channels).unwrap_or(u16::MAX),
        sample_rate: renderer.runtime().sample_rate(),
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer =
        hound::WavWriter::create(path, spec).map_err(|e| DeviceError::Wav(e.to_string()))?;

    let mut chunk: Vec<Sample> = vec![0.0; OFFLINE_CHUNK_FRAMES * channels];
    let mut remaining = total_frames;
    while remaining > 0 {
        let frames = OFFLINE_CHUNK_FRAMES.min(usize::try_from(remaining).unwrap_or(usize::MAX));
        let slice = &mut chunk[..frames * channels];
        renderer.render_frames(frames, slice);
        for &sample in slice.iter() {
            writer
                .write_sample(sample)
                .map_err(|e| DeviceError::Wav(e.to_string()))?;
        }
        remaining -= frames as u64;
    }

    writer
        .finalize()
        .map_err(|e| DeviceError::Wav(e.to_string()))?;
    Ok(total_frames)
}
