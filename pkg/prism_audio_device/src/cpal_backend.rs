//! Live audio output through [`cpal`], driven by a [`BlockRenderer`].
//!
//! [`open_default_output`] opens the platform default output device, matches
//! its channel count to an engine [`ChannelLayout`], and installs a data
//! callback that pulls fixed engine blocks from the shared [`BlockRenderer`]
//! and interleaves them into the device buffer. The callback allocates nothing:
//! a single scratch buffer sized at open time carries samples between the
//! engine's fixed blocks and the host's variable-size callback buffer, and any
//! non-`f32` device format is converted in place through that scratch.
//!
//! The returned [`CpalOutput`] owns the live `cpal::Stream`; playback stops when
//! it is dropped. Because a `cpal::Stream` is not portable across threads on
//! every platform, [`CpalOutput`] is intended to live on the thread that opened
//! it.

use alloc::string::{String, ToString};
use alloc::vec;
use core::sync::atomic::{AtomicU64, Ordering};

use alloc::sync::Arc;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SampleFormat, SizedSample};

use prism_audio_core::buffer::ChannelLayout;
use prism_audio_core::math::Sample;
use prism_audio_rt::AudioRuntime;

use crate::error::DeviceError;
use crate::render::BlockRenderer;

/// Maps a device channel count onto the engine [`ChannelLayout`] with the same
/// number of channels.
///
/// Four channels map to [`ChannelLayout::Quad`] rather than
/// [`ChannelLayout::AmbisonicFoa`]; a raw device endpoint is speaker-fed, not an
/// ambisonic bus.
///
/// # Errors
///
/// Returns [`DeviceError::UnsupportedChannelCount`] for a channel count with no
/// matching layout.
pub fn layout_for_channels(channels: u16) -> Result<ChannelLayout, DeviceError> {
    Ok(match channels {
        1 => ChannelLayout::Mono,
        2 => ChannelLayout::Stereo,
        4 => ChannelLayout::Quad,
        6 => ChannelLayout::Surround5_1,
        8 => ChannelLayout::Surround7_1,
        other => return Err(DeviceError::UnsupportedChannelCount(other as usize)),
    })
}

/// Immutable description of the negotiated output stream.
#[derive(Debug, Clone)]
pub struct OutputStreamInfo {
    /// Human-readable device name, or `"unknown"` if the platform withheld one.
    pub device_name: String,
    /// Stream sample rate in Hz (matches the driving [`AudioRuntime`]).
    pub sample_rate: u32,
    /// Interleaved device channel count.
    pub channels: usize,
    /// Engine layout matched to the device channel count.
    pub layout: ChannelLayout,
    /// Native device sample format the callback converts into.
    pub sample_format: SampleFormat,
}

/// A live `cpal` output stream fed by a [`BlockRenderer`].
///
/// Dropping the value stops and closes the stream. Keep it alive for as long as
/// playback is desired.
pub struct CpalOutput {
    /// The live platform stream. Kept alive; playback stops on drop.
    stream: cpal::Stream,
    /// Negotiated stream description.
    info: OutputStreamInfo,
    /// Count of stream errors reported by the platform error callback
    /// (underruns, device disconnects). Shared with the error callback.
    errors: Arc<AtomicU64>,
}

impl CpalOutput {
    /// The negotiated stream description.
    #[inline]
    #[must_use]
    pub fn info(&self) -> &OutputStreamInfo {
        &self.info
    }

    /// Number of platform stream errors observed since opening.
    #[inline]
    #[must_use]
    pub fn error_count(&self) -> u64 {
        self.errors.load(Ordering::Relaxed)
    }

    /// Resumes playback after [`CpalOutput::pause`].
    ///
    /// # Errors
    ///
    /// Returns [`DeviceError::PlayStream`] if the platform refuses to start.
    pub fn play(&self) -> Result<(), DeviceError> {
        self.stream
            .play()
            .map_err(|e| DeviceError::PlayStream(e.to_string()))
    }

    /// Pauses playback without closing the stream, where the platform supports
    /// it.
    ///
    /// # Errors
    ///
    /// Returns [`DeviceError::PlayStream`] if the platform refuses to pause.
    pub fn pause(&self) -> Result<(), DeviceError> {
        self.stream
            .pause()
            .map_err(|e| DeviceError::PlayStream(e.to_string()))
    }
}

impl core::fmt::Debug for CpalOutput {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CpalOutput")
            .field("info", &self.info)
            .field("errors", &self.error_count())
            .finish_non_exhaustive()
    }
}

/// Opens the platform default output device and starts streaming audio from
/// `runtime`.
///
/// The stream runs at the runtime's sample rate so playback pitch is correct;
/// if the device cannot honor that rate the platform reports a
/// [`DeviceError::BuildStream`] rather than silently resampling. `block_frames`
/// sizes the callback scratch (clamped internally to the runtime's maximum
/// block); the host callback buffer may be any size.
///
/// # Errors
///
/// Returns a [`DeviceError`] if no output device is available, the device
/// configuration cannot be queried, the channel count or sample format is
/// unsupported, or the stream cannot be built or started.
pub fn open_default_output(
    runtime: AudioRuntime,
    block_frames: usize,
) -> Result<CpalOutput, DeviceError> {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or(DeviceError::NoOutputDevice)?;
    let device_name = device
        .description()
        .map(|desc| desc.name().to_string())
        .unwrap_or_else(|_| "unknown".to_string());

    let default_config = device
        .default_output_config()
        .map_err(|e| DeviceError::ConfigQuery(e.to_string()))?;
    let sample_format = default_config.sample_format();
    let channels = default_config.channels();
    let layout = layout_for_channels(channels)?;

    let stream_config = cpal::StreamConfig {
        channels,
        sample_rate: runtime.sample_rate(),
        buffer_size: cpal::BufferSize::Default,
    };

    let info = OutputStreamInfo {
        device_name,
        sample_rate: runtime.sample_rate(),
        channels: channels as usize,
        layout,
        sample_format,
    };

    let renderer = BlockRenderer::new(runtime, layout, block_frames);
    let errors = Arc::new(AtomicU64::new(0));

    let stream = build_output_stream(
        &device,
        &stream_config,
        sample_format,
        renderer,
        block_frames,
        Arc::clone(&errors),
    )?;
    stream
        .play()
        .map_err(|e| DeviceError::PlayStream(e.to_string()))?;

    Ok(CpalOutput {
        stream,
        info,
        errors,
    })
}

/// Dispatches on the device sample format and builds the typed output stream.
fn build_output_stream(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    format: SampleFormat,
    renderer: BlockRenderer,
    block_frames: usize,
    errors: Arc<AtomicU64>,
) -> Result<cpal::Stream, DeviceError> {
    match format {
        SampleFormat::F32 => build_typed::<f32>(device, config, renderer, block_frames, errors),
        SampleFormat::F64 => build_typed::<f64>(device, config, renderer, block_frames, errors),
        SampleFormat::I16 => build_typed::<i16>(device, config, renderer, block_frames, errors),
        SampleFormat::U16 => build_typed::<u16>(device, config, renderer, block_frames, errors),
        SampleFormat::I32 => build_typed::<i32>(device, config, renderer, block_frames, errors),
        SampleFormat::I8 => build_typed::<i8>(device, config, renderer, block_frames, errors),
        SampleFormat::U8 => build_typed::<u8>(device, config, renderer, block_frames, errors),
        other => Err(DeviceError::UnsupportedSampleFormat(
            alloc::format!("{other:?}"),
        )),
    }
}

/// Builds a `cpal` output stream whose callback fills `T` samples from the
/// renderer, converting from the engine's `f32` through a preallocated scratch.
fn build_typed<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    mut renderer: BlockRenderer,
    block_frames: usize,
    errors: Arc<AtomicU64>,
) -> Result<cpal::Stream, DeviceError>
where
    T: SizedSample + FromSample<f32> + Send + 'static,
{
    let channels = renderer.channels();
    let mut scratch = vec![0.0 as Sample; block_frames.max(1) * channels];
    let err_counter = Arc::clone(&errors);

    device
        .build_output_stream(
            config,
            move |data: &mut [T], _info: &cpal::OutputCallbackInfo| {
                fill_output(&mut renderer, &mut scratch, data);
            },
            move |_err| {
                // Count stream errors (underruns, disconnects) for diagnostics;
                // the host may layer richer logging on top of `error_count`.
                err_counter.fetch_add(1, Ordering::Relaxed);
            },
            None,
        )
        .map_err(|e| DeviceError::BuildStream(e.to_string()))
}

/// Fills an interleaved device buffer `data` of arbitrary length by pulling
/// engine blocks through `scratch` (sized `chunk_frames * channels`) and
/// converting each sample into the device format `T`.
///
/// Real-time safe: no allocation, no locking. `scratch` must hold at least one
/// whole frame (`channels` samples).
fn fill_output<T>(renderer: &mut BlockRenderer, scratch: &mut [Sample], data: &mut [T])
where
    T: SizedSample + FromSample<f32>,
{
    let channels = renderer.channels();
    // Largest whole-frame chunk the scratch can hold; always a channel multiple.
    let chunk_samples = (scratch.len() / channels) * channels;
    debug_assert!(chunk_samples >= channels, "scratch smaller than one frame");

    let mut offset = 0;
    while offset < data.len() {
        let take = (data.len() - offset).min(chunk_samples);
        let staged = &mut scratch[..take];
        renderer.render_interleaved(staged);
        for (dst, &src) in data[offset..offset + take].iter_mut().zip(staged.iter()) {
            *dst = T::from_sample(src);
        }
        offset += take;
    }
}

#[cfg(test)]
mod tests {
    use super::{fill_output, layout_for_channels};
    use crate::render::BlockRenderer;
    use cpal::Sample;
    use prism_audio_core::buffer::ChannelLayout;
    use prism_audio_rt::{AudioRuntimeConfig, runtime};

    #[test]
    fn channel_counts_map_to_layouts() {
        assert_eq!(layout_for_channels(1).unwrap(), ChannelLayout::Mono);
        assert_eq!(layout_for_channels(2).unwrap(), ChannelLayout::Stereo);
        assert_eq!(layout_for_channels(4).unwrap(), ChannelLayout::Quad);
        assert_eq!(layout_for_channels(6).unwrap(), ChannelLayout::Surround5_1);
        assert_eq!(layout_for_channels(8).unwrap(), ChannelLayout::Surround7_1);
        assert!(layout_for_channels(3).is_err());
        assert!(layout_for_channels(0).is_err());
    }

    #[test]
    fn f32_conversion_is_identity_and_i16_is_full_scale() {
        assert_eq!(f32::from_sample(0.5f32), 0.5f32);
        // Full-scale positive maps to i16::MAX.
        assert_eq!(i16::from_sample(1.0f32), i16::MAX);
    }

    #[test]
    fn fill_output_serves_buffers_larger_than_the_scratch() {
        // A runtime with no published graph renders silence deterministically.
        let (rt, _client, _collector) = runtime(AudioRuntimeConfig {
            max_block: 16,
            ..AudioRuntimeConfig::default()
        });
        let mut renderer = BlockRenderer::new(rt, ChannelLayout::Stereo, 8);

        // Scratch holds 8 stereo frames; the device buffer asks for 20 frames,
        // forcing multiple chunked renders through the scratch.
        let mut scratch = [0.0f32; 8 * 2];
        let mut data = [7.0f32; 20 * 2];
        fill_output(&mut renderer, &mut scratch, &mut data);

        assert!(data.iter().all(|s| s.is_finite()));
        // No graph => silence on every serviced frame.
        assert!(data.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn fill_output_handles_partial_trailing_chunk() {
        let (rt, _client, _collector) = runtime(AudioRuntimeConfig {
            max_block: 16,
            ..AudioRuntimeConfig::default()
        });
        let mut renderer = BlockRenderer::new(rt, ChannelLayout::Mono, 8);

        let mut scratch = [0.0f32; 8];
        // 13 mono frames: one full 8-frame chunk plus a 5-frame remainder.
        let mut data = [1.0f32; 13];
        fill_output(&mut renderer, &mut scratch, &mut data);
        assert!(data.iter().all(|&s| s == 0.0));
    }
}
