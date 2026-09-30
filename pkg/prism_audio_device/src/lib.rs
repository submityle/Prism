//! Device, offline, and capture backends for Prism's next-generation audio
//! engine (**Resonance**).
//!
//! This crate is the outermost seam of the audio stack: it turns the fixed
//! engine blocks produced by [`prism_audio_rt::AudioRuntime`] into the
//! arbitrary, host-driven buffers that platform audio SDKs and offline files
//! consume, and it feeds device input back into the engine's planar buffers.
//!
//! # Modules
//!
//! - [`error`] — the shared [`DeviceError`] returned by every backend.
//! - [`interleave`] — allocation-free planar <-> interleaved transposes shared
//!   by the device, capture, and offline paths.
//! - [`render`] — the [`BlockRenderer`] pull adapter that serves fixed engine
//!   blocks into variable-size interleaved buffers with zero steady-state
//!   allocation.
//! - [`file_sink`] — deterministic offline rendering to 32-bit float WAV,
//!   sharing the exact [`BlockRenderer`] path for sample-identical goldens.
//! - [`capture`] — lock-free interleaved input capture into planar buffers.
//! - [`cpal_backend`] — a live `cpal` output stream driven by a
//!   [`BlockRenderer`] (enabled by the default `cpal-backend` feature).
//!
//! # Real-time contract
//!
//! Everything reachable from a device callback — [`BlockRenderer::render_interleaved`],
//! [`capture::CaptureSink::push_interleaved`], and the interleave helpers — is
//! allocation free, lock free, and panic free on the steady-state path. Scratch
//! buffers are sized once at construction. The offline sink and live device
//! share the identical render path, so an offline WAV render is bit-identical
//! to live playback for the same graph and command stream.
//!
//! # Provenance
//!
//! This crate is engine-agnostic and contains **no Unreal Engine, Unity, Godot,
//! Wwise, or FMOD source or derived code**. It builds on the safe, widely used
//! `cpal` (cross-platform device I/O) and `hound` (WAV) crates and follows
//! standard, publicly documented real-time audio engineering practice.

// The backend files use `alloc::` paths directly (owned error strings, offline
// chunk vectors) even though the crate links `std` for device and file I/O.
extern crate alloc;

pub mod capture;
pub mod error;
pub mod file_sink;
pub mod interleave;
pub mod render;

#[cfg(feature = "cpal-backend")]
pub mod cpal_backend;

pub use capture::{CaptureConsumer, CaptureSink, capture_ring};
pub use error::DeviceError;
pub use file_sink::render_to_wav;
pub use interleave::{interleaved_to_planar, planar_to_interleaved};
pub use render::BlockRenderer;

#[cfg(feature = "cpal-backend")]
pub use cpal_backend::{CpalOutput, OutputStreamInfo, layout_for_channels, open_default_output};
