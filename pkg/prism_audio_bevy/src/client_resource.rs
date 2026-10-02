//! Resources wrapping the two halves returned by
//! [`prism_audio_rt::runtime`]: the clonable client used by ECS systems and the
//! parked audio-thread runtime used by headless/offline setups.
//!
//! [`AudioClient`] is an ordinary [`Resource`] wrapping an
//! [`AudioRuntimeClient`]; it is `Send + Sync` because every ring half is an
//! `Arc<ArrayQueue<_>>`. [`AudioRuntimeHost`] is a non-`Send` resource holding
//! the real-time [`AudioRuntime`] and its [`Collector`]; a device backend would
//! take it and drive it from the audio-callback thread, while this crate's
//! systems never touch it.
//!
//! # Provenance
//!
//! Contains no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; original ECS glue. No AI/ML.
//!
//! # Relationship
//!
//! Owns the client/runtime handles produced by `prism_audio_rt` and exposes
//! them to the plugin's systems.

use alloc::boxed::Box;

use bevy_ecs::resource::Resource;
use prism_audio_core::buffer::AudioBuffer;
use prism_audio_core::graph::AudioGraph;
use prism_audio_rt::{AudioCommand, AudioRuntime, AudioRuntimeClient, Collector, TelemetryFrame};

/// ECS resource wrapping the clonable [`AudioRuntimeClient`].
///
/// Systems read this resource (never mutably) to enqueue commands and drain
/// telemetry; the underlying rings are internally synchronized, so shared
/// (`Res`) access is sufficient and lets command-producing systems run without
/// contending on exclusive access.
#[derive(Resource, Clone, Debug)]
pub struct AudioClient {
    /// The wrapped runtime client handle.
    client: AudioRuntimeClient,
}

impl AudioClient {
    /// Wraps an [`AudioRuntimeClient`] as an ECS resource.
    #[must_use]
    #[inline]
    pub fn new(client: AudioRuntimeClient) -> Self {
        Self { client }
    }

    /// Borrows the underlying [`AudioRuntimeClient`].
    #[must_use]
    #[inline]
    pub fn client(&self) -> &AudioRuntimeClient {
        &self.client
    }

    /// Enqueues a command on the command ring.
    ///
    /// # Errors
    ///
    /// Returns `Err(command)` when the ring is full so the caller can retry or
    /// coalesce next frame; this never blocks.
    #[inline]
    pub fn send(&self, command: AudioCommand) -> Result<(), AudioCommand> {
        self.client.send(command)
    }

    /// Publishes a freshly compiled graph for the audio thread to swap in,
    /// returning any superseded graph for the caller to drop off-thread.
    #[inline]
    pub fn publish_graph(&self, graph: Box<AudioGraph>) -> Option<Box<AudioGraph>> {
        self.client.publish_graph(graph)
    }

    /// Reads the next unread telemetry frame, if any.
    #[must_use]
    #[inline]
    pub fn recv_telemetry(&self) -> Option<TelemetryFrame> {
        self.client.recv_telemetry()
    }
}

/// Non-`Send` resource parking the real-time audio-thread half of the runtime.
///
/// In production a device backend takes the [`AudioRuntime`] and drives
/// [`AudioRuntime::process_block`] from the audio-callback thread. When no
/// device is present (headless, offline rendering, or tests) this host keeps
/// the runtime alongside its [`Collector`] so callers can advance it manually
/// via [`AudioRuntimeHost::pump_block`].
pub struct AudioRuntimeHost {
    /// The real-time runtime that renders blocks and publishes telemetry.
    runtime: AudioRuntime,
    /// The deferred-reclamation collector that drops resources the runtime
    /// retires (for example superseded graphs) off the audio thread.
    collector: Collector,
}

impl AudioRuntimeHost {
    /// Parks a runtime and its collector together.
    #[must_use]
    #[inline]
    pub fn new(runtime: AudioRuntime, collector: Collector) -> Self {
        Self { runtime, collector }
    }

    /// Renders one block into `master_out`, collects any retired resources, and
    /// returns the published [`TelemetryFrame`].
    ///
    /// This is the deviceless/offline entry point; a real device backend would
    /// call [`AudioRuntime::process_block`] directly from its callback instead.
    #[inline]
    pub fn pump_block(&mut self, master_out: &mut AudioBuffer) -> TelemetryFrame {
        let frame = self.runtime.process_block(master_out);
        let _ = self.collector.collect();
        frame
    }

    /// Collects any retired resources without rendering, returning how many
    /// were dropped.
    #[inline]
    pub fn collect(&self) -> usize {
        self.collector.collect()
    }

    /// Borrows the parked runtime for inspection.
    #[must_use]
    #[inline]
    pub fn runtime(&self) -> &AudioRuntime {
        &self.runtime
    }

    /// Exclusively borrows the parked runtime.
    #[inline]
    pub fn runtime_mut(&mut self) -> &mut AudioRuntime {
        &mut self.runtime
    }
}
