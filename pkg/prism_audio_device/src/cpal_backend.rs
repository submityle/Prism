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

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec;
use core::sync::atomic::{AtomicU64, Ordering};

use alloc::sync::Arc;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SampleFormat, SizedSample};

use prism_audio_core::buffer::ChannelLayout;
use prism_audio_core::math::Sample;
use prism_audio_rt::AudioRuntime;

use crate::backend::{
    AudioDeviceBackend, AudioDeviceFactory, DeviceEvent, DeviceFormat, DeviceSampleFormat,
    DeviceState,
};
use crate::error::DeviceError;
use crate::render::BlockRenderer;

const EVENT_UNDERRUN: u64 = 1;
const EVENT_DISCONNECTED: u64 = 2;
const EVENT_STREAM_ERROR: u64 = 4;

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
    /// Coalesced callback events. The control thread drains this bitset with
    /// [`CpalOutput::poll_event`]; the callback never allocates or locks.
    events: Arc<AtomicU64>,
    state: DeviceState,
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

    /// Drains one coalesced callback event for a control-thread recovery loop.
    ///
    /// CPAL does not provide an xrun frame count through `StreamError`, so
    /// underrun events report `frames: 0`. Reopen/retry policy remains outside
    /// the callback and must create a new [`CpalOutput`] after observing a
    /// disconnect or invalidated stream.
    #[must_use]
    pub fn poll_event(&mut self) -> Option<DeviceEvent> {
        for (mask, event) in [
            (EVENT_DISCONNECTED, DeviceEvent::Disconnected),
            (EVENT_UNDERRUN, DeviceEvent::Underrun { frames: 0 }),
            (EVENT_STREAM_ERROR, DeviceEvent::StreamError),
        ] {
            let previous = self.events.fetch_and(!mask, Ordering::AcqRel);
            if previous & mask != 0 {
                self.state = match event {
                    DeviceEvent::Underrun { .. } => DeviceState::Degraded,
                    DeviceEvent::Disconnected | DeviceEvent::StreamError => DeviceState::Recovering,
                    _ => self.state,
                };
                return Some(event);
            }
        }
        None
    }

    /// Resumes playback after [`CpalOutput::pause`].
    ///
    /// # Errors
    ///
    /// Returns [`DeviceError::PlayStream`] if the platform refuses to start.
    pub fn play(&mut self) -> Result<(), DeviceError> {
        self.start()
    }

    /// Pauses playback without closing the stream, where the platform supports
    /// it.
    ///
    /// # Errors
    ///
    /// Returns [`DeviceError::PlayStream`] if the platform refuses to pause.
    pub fn pause(&mut self) -> Result<(), DeviceError> {
        self.stop()
    }
}

impl AudioDeviceBackend for CpalOutput {
    fn state(&self) -> DeviceState {
        self.state
    }

    fn format(&self) -> DeviceFormat {
        self.info.format()
    }

    fn start(&mut self) -> Result<(), DeviceError> {
        if !matches!(self.state, DeviceState::Stopped) {
            return Err(DeviceError::InvalidState {
                operation: "start",
                state: self.state,
            });
        }
        self.stream
            .play()
            .map_err(|e| DeviceError::PlayStream(e.to_string()))?;
        self.state = DeviceState::Running;
        Ok(())
    }

    fn stop(&mut self) -> Result<(), DeviceError> {
        if !matches!(self.state, DeviceState::Running | DeviceState::Degraded) {
            return Err(DeviceError::InvalidState {
                operation: "stop",
                state: self.state,
            });
        }
        self.stream
            .pause()
            .map_err(|e| DeviceError::StopStream(e.to_string()))?;
        self.state = DeviceState::Stopped;
        Ok(())
    }

    fn poll_event(&mut self) -> Option<DeviceEvent> {
        CpalOutput::poll_event(self)
    }
}

/// Factory for the platform default CPAL output endpoint.
#[derive(Debug, Clone, Copy, Default)]
pub struct CpalOutputFactory;

impl AudioDeviceFactory for CpalOutputFactory {
    fn open(
        &mut self,
        runtime: AudioRuntime,
        block_frames: usize,
    ) -> Result<Box<dyn AudioDeviceBackend>, DeviceError> {
        open_default_output_paused(runtime, block_frames)
            .map(|backend| Box::new(backend) as Box<dyn AudioDeviceBackend>)
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
    let mut output = open_default_output_paused(runtime, block_frames)?;
    output.start()?;
    Ok(output)
}

/// Opens the platform default output device without starting callback delivery.
///
/// This is the construction half used by lifecycle hosts: state is
/// [`DeviceState::Stopped`] until [`AudioDeviceBackend::start`] succeeds.
pub fn open_default_output_paused(
    runtime: AudioRuntime,
    block_frames: usize,
) -> Result<CpalOutput, DeviceError> {
    if block_frames == 0 {
        return Err(DeviceError::InvalidBlockFrames);
    }
    let block_frames = block_frames.min(runtime.max_block());
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
    device_sample_format(sample_format)?;
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
    let events = Arc::new(AtomicU64::new(0));

    let stream = build_output_stream(
        &device,
        &stream_config,
        sample_format,
        renderer,
        block_frames,
        Arc::clone(&errors),
        Arc::clone(&events),
    )?;
    Ok(CpalOutput {
        stream,
        info,
        errors,
        events,
        state: DeviceState::Stopped,
    })
}

impl OutputStreamInfo {
    /// Backend-neutral negotiated stream format.
    #[must_use]
    pub fn format(&self) -> DeviceFormat {
        DeviceFormat {
            sample_rate: self.sample_rate,
            layout: self.layout,
            sample_format: device_sample_format(self.sample_format)
                .expect("unsupported formats are rejected while opening"),
        }
    }
}

fn device_sample_format(format: SampleFormat) -> Result<DeviceSampleFormat, DeviceError> {
    Ok(match format {
        SampleFormat::F32 => DeviceSampleFormat::F32,
        SampleFormat::F64 => DeviceSampleFormat::F64,
        SampleFormat::I8 => DeviceSampleFormat::I8,
        SampleFormat::I16 => DeviceSampleFormat::I16,
        SampleFormat::I32 => DeviceSampleFormat::I32,
        SampleFormat::U8 => DeviceSampleFormat::U8,
        SampleFormat::U16 => DeviceSampleFormat::U16,
        other => {
            return Err(DeviceError::UnsupportedSampleFormat(alloc::format!(
                "{other:?}"
            )))
        }
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
    events: Arc<AtomicU64>,
) -> Result<cpal::Stream, DeviceError> {
    match format {
        SampleFormat::F32 => {
            build_typed::<f32>(device, config, renderer, block_frames, errors, events)
        }
        SampleFormat::F64 => {
            build_typed::<f64>(device, config, renderer, block_frames, errors, events)
        }
        SampleFormat::I16 => {
            build_typed::<i16>(device, config, renderer, block_frames, errors, events)
        }
        SampleFormat::U16 => {
            build_typed::<u16>(device, config, renderer, block_frames, errors, events)
        }
        SampleFormat::I32 => {
            build_typed::<i32>(device, config, renderer, block_frames, errors, events)
        }
        SampleFormat::I8 => {
            build_typed::<i8>(device, config, renderer, block_frames, errors, events)
        }
        SampleFormat::U8 => {
            build_typed::<u8>(device, config, renderer, block_frames, errors, events)
        }
        other => Err(DeviceError::UnsupportedSampleFormat(alloc::format!(
            "{other:?}"
        ))),
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
    events: Arc<AtomicU64>,
) -> Result<cpal::Stream, DeviceError>
where
    T: SizedSample + FromSample<f32> + Send + 'static,
{
    let channels = renderer.channels();
    let mut scratch = vec![0.0 as Sample; block_frames.max(1) * channels];
    let err_counter = Arc::clone(&errors);
    let event_flags = Arc::clone(&events);

    device
        .build_output_stream(
            config,
            move |data: &mut [T], _info: &cpal::OutputCallbackInfo| {
                fill_output(&mut renderer, &mut scratch, data);
            },
            move |err| {
                // CPAL invokes this callback off the control thread. Keep it
                // bounded: count and publish a bit only; never reopen here.
                err_counter.fetch_add(1, Ordering::Relaxed);
                let flag = match err {
                    cpal::StreamError::BufferUnderrun => EVENT_UNDERRUN,
                    cpal::StreamError::DeviceNotAvailable
                    | cpal::StreamError::StreamInvalidated => EVENT_DISCONNECTED,
                    cpal::StreamError::BackendSpecific { .. } => EVENT_STREAM_ERROR,
                };
                event_flags.fetch_or(flag, Ordering::Release);
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

    let whole_samples = data.len() - data.len() % channels;
    let mut offset = 0;
    while offset < whole_samples {
        let take = (whole_samples - offset).min(chunk_samples);
        let staged = &mut scratch[..take];
        renderer.render_interleaved(staged);
        for (dst, &src) in data[offset..offset + take].iter_mut().zip(staged.iter()) {
            *dst = T::from_sample(src);
        }
        offset += take;
    }
    for sample in &mut data[whole_samples..] {
        *sample = T::from_sample(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::{device_sample_format, fill_output, layout_for_channels, open_default_output};
    use crate::backend::DeviceSampleFormat;
    use crate::render::BlockRenderer;
    use crate::DeviceError;
    use cpal::Sample;
    use prism_audio_core::buffer::ChannelLayout;
    use prism_audio_rt::{runtime, AudioRuntimeConfig};

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
    fn cpal_formats_map_to_backend_neutral_formats() {
        assert_eq!(
            device_sample_format(cpal::SampleFormat::F32).unwrap(),
            DeviceSampleFormat::F32
        );
        assert_eq!(
            device_sample_format(cpal::SampleFormat::I16).unwrap(),
            DeviceSampleFormat::I16
        );
        assert_eq!(
            device_sample_format(cpal::SampleFormat::U16).unwrap(),
            DeviceSampleFormat::U16
        );
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

    #[test]
    fn fill_output_silences_a_partial_trailing_frame() {
        let (rt, _client, _collector) = runtime(AudioRuntimeConfig {
            max_block: 16,
            ..AudioRuntimeConfig::default()
        });
        let mut renderer = BlockRenderer::new(rt, ChannelLayout::Stereo, 8);
        let mut scratch = [0.0f32; 8 * 2];
        let mut data = [1.0f32; 5];
        fill_output(&mut renderer, &mut scratch, &mut data);
        assert!(data.iter().all(|&sample| sample == 0.0));
    }

    #[test]
    fn opening_with_zero_block_frames_is_rejected_before_device_access() {
        let (runtime, _client, _collector) = runtime(AudioRuntimeConfig::default());
        assert!(matches!(
            open_default_output(runtime, 0),
            Err(DeviceError::InvalidBlockFrames)
        ));
    }
}
