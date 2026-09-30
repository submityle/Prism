//! Error type shared by the device, capture, and offline-render backends.

use alloc::string::String;
use core::fmt;

/// Errors produced while enumerating devices, building streams, or writing
/// offline renders.
///
/// Platform SDK error strings are captured as owned [`String`]s so the error is
/// `'static` and can cross thread boundaries away from the borrowed SDK types.
#[derive(Debug)]
#[non_exhaustive]
pub enum DeviceError {
    /// The default host exposed no usable output device.
    NoOutputDevice,
    /// The default host exposed no usable input device.
    NoInputDevice,
    /// Querying a device's default or supported stream configuration failed.
    ConfigQuery(String),
    /// Constructing the platform audio stream failed.
    BuildStream(String),
    /// Starting (playing) the platform audio stream failed.
    PlayStream(String),
    /// The renderer's channel count does not match the device stream.
    ChannelCountMismatch {
        /// Channel count the renderer produces.
        renderer: usize,
        /// Channel count the device stream expects.
        device: usize,
    },
    /// The device reported a channel count with no matching engine
    /// [`ChannelLayout`](prism_audio_core::buffer::ChannelLayout).
    UnsupportedChannelCount(usize),
    /// The device stream uses a sample format the backend cannot drive.
    UnsupportedSampleFormat(String),
    /// A WAV encode/decode or file-system error occurred in the offline sink.
    Wav(String),
}

impl fmt::Display for DeviceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceError::NoOutputDevice => f.write_str("no output device available"),
            DeviceError::NoInputDevice => f.write_str("no input device available"),
            DeviceError::ConfigQuery(msg) => write!(f, "device configuration query failed: {msg}"),
            DeviceError::BuildStream(msg) => write!(f, "failed to build audio stream: {msg}"),
            DeviceError::PlayStream(msg) => write!(f, "failed to start audio stream: {msg}"),
            DeviceError::ChannelCountMismatch { renderer, device } => write!(
                f,
                "renderer produces {renderer} channels but the device stream expects {device}"
            ),
            DeviceError::UnsupportedChannelCount(n) => {
                write!(f, "device channel count {n} has no matching engine layout")
            }
            DeviceError::UnsupportedSampleFormat(fmt) => {
                write!(f, "unsupported device sample format: {fmt}")
            }
            DeviceError::Wav(msg) => write!(f, "WAV file error: {msg}"),
        }
    }
}

impl std::error::Error for DeviceError {}
