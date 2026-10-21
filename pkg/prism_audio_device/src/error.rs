//! Error type shared by the device, capture, and offline-render backends.

use alloc::string::String;
use core::fmt;

use crate::backend::DeviceState;

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
    /// The requested stable input endpoint ID is not present in the current catalog.
    InputDeviceNotFound(String),
    /// Querying a device's default or supported stream configuration failed.
    ConfigQuery(String),
    /// Constructing the platform audio stream failed.
    BuildStream(String),
    /// Starting (playing) the platform audio stream failed.
    PlayStream(String),
    /// Pausing/stopping the platform audio stream failed.
    StopStream(String),
    /// A lifecycle operation was requested from an illegal state.
    InvalidState {
        /// Requested lifecycle operation.
        operation: &'static str,
        /// State in which the operation was rejected.
        state: DeviceState,
    },
    /// A backend factory failed before a more specific platform error existed.
    Backend(String),
    /// The renderer's channel count does not match the device stream.
    ChannelCountMismatch {
        /// Channel count the renderer produces.
        renderer: usize,
        /// Channel count the device stream expects.
        device: usize,
    },
    /// Capture ring channel count does not match the selected input endpoint.
    CaptureChannelCountMismatch {
        /// Channel count the capture ring accepts.
        sink: usize,
        /// Channel count the input stream supplies.
        device: usize,
    },
    /// The device reported a channel count with no matching engine
    /// [`ChannelLayout`](prism_audio_core::buffer::ChannelLayout).
    UnsupportedChannelCount(usize),
    /// The device stream uses a sample format the backend cannot drive.
    UnsupportedSampleFormat(String),
    /// A live output backend was requested with no callback scratch frames.
    InvalidBlockFrames,
    /// A WAV encode/decode or file-system error occurred in the offline sink.
    Wav(String),
}

impl fmt::Display for DeviceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceError::NoOutputDevice => f.write_str("no output device available"),
            DeviceError::NoInputDevice => f.write_str("no input device available"),
            DeviceError::InputDeviceNotFound(id) => {
                write!(f, "input device with CPAL ID {id:?} was not found")
            }
            DeviceError::ConfigQuery(msg) => write!(f, "device configuration query failed: {msg}"),
            DeviceError::BuildStream(msg) => write!(f, "failed to build audio stream: {msg}"),
            DeviceError::PlayStream(msg) => write!(f, "failed to start audio stream: {msg}"),
            DeviceError::StopStream(msg) => write!(f, "failed to stop audio stream: {msg}"),
            DeviceError::InvalidState { operation, state } => {
                write!(f, "cannot {operation} audio device while it is {state:?}")
            }
            DeviceError::Backend(msg) => write!(f, "audio backend failed: {msg}"),
            DeviceError::ChannelCountMismatch { renderer, device } => write!(
                f,
                "renderer produces {renderer} channels but the device stream expects {device}"
            ),
            DeviceError::CaptureChannelCountMismatch { sink, device } => write!(
                f,
                "capture ring accepts {sink} channels but the input stream supplies {device}"
            ),
            DeviceError::UnsupportedChannelCount(n) => {
                write!(f, "device channel count {n} has no matching engine layout")
            }
            DeviceError::UnsupportedSampleFormat(fmt) => {
                write!(f, "unsupported device sample format: {fmt}")
            }
            DeviceError::InvalidBlockFrames => {
                f.write_str("audio device block frame count must be non-zero")
            }
            DeviceError::Wav(msg) => write!(f, "WAV file error: {msg}"),
        }
    }
}

impl std::error::Error for DeviceError {}
