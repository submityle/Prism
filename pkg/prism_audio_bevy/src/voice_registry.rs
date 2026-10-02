//! Client-side voice bookkeeping: the deterministic [`VoiceMirror`] of the
//! runtime's pool, the authoritative [`LiveVoices`] entity map, and the
//! [`PendingStops`] retry queue.
//!
//! The runtime mints [`VoiceHandle`]s on the audio thread and the command ring
//! is one-way, so this crate reconstructs the handles by replaying the exact
//! same allocation/release sequence against a private
//! [`VoicePool`](prism_audio_core::voice::VoicePool) in [`VoiceMirror`]. The
//! invariant is strict: a mirror operation runs only *after* its command has
//! been accepted by the ring, in enqueue order, so both pools start identical
//! and apply the identical FIFO command stream. Because the systems are the
//! sole producer of voice-lifecycle commands, the mirror's returned handle is
//! bit-for-bit the handle the runtime will produce.
//!
//! # Provenance
//!
//! Contains no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; original ECS glue. No AI/ML.
//!
//! # Relationship
//!
//! Mirrors [`prism_audio_core::voice::VoicePool`] on the client side and keys
//! live voices by [`bevy_ecs`] [`Entity`].

use alloc::vec::Vec;
use std::collections::HashMap;

use bevy_ecs::entity::Entity;
use bevy_ecs::resource::Resource;
use prism_audio_core::voice::{Importance, VoiceHandle, VoicePool, VoiceRequest};

/// Deterministic client-side mirror of the runtime's voice pool.
///
/// Constructed with the same `capacity`/`max_physical` as the runtime pool and
/// driven with the identical command sequence, so [`VoiceMirror::allocate`]
/// predicts the handle the runtime mints for each
/// [`SpawnVoice`](prism_audio_rt::AudioCommand::SpawnVoice).
#[derive(Resource)]
pub struct VoiceMirror {
    /// Private pool kept in lock-step with the runtime's pool.
    pool: VoicePool,
}

impl VoiceMirror {
    /// Builds a mirror matching a runtime pool of `capacity` voices with at
    /// most `max_physical` audible at once.
    #[must_use]
    #[inline]
    pub fn new(capacity: usize, max_physical: usize) -> Self {
        Self {
            pool: VoicePool::new(capacity, max_physical),
        }
    }

    /// Mirrors a [`SpawnVoice`](prism_audio_rt::AudioCommand::SpawnVoice),
    /// returning the handle the runtime will mint, or `None` if the pool is
    /// full on both sides.
    #[inline]
    pub fn allocate(&mut self, request: VoiceRequest) -> Option<VoiceHandle> {
        self.pool.allocate(request)
    }

    /// Mirrors a [`StopVoice`](prism_audio_rt::AudioCommand::StopVoice),
    /// returning whether the handle resolved.
    #[inline]
    pub fn release(&mut self, handle: VoiceHandle) -> bool {
        self.pool.release(handle)
    }

    /// Mirrors a [`SetVoiceImportance`](prism_audio_rt::AudioCommand::SetVoiceImportance),
    /// returning whether the handle resolved.
    #[inline]
    pub fn set_importance(&mut self, handle: VoiceHandle, importance: Importance) -> bool {
        self.pool.set_importance(handle, importance)
    }

    /// Mirrors a [`SetMaxPhysicalVoices`](prism_audio_rt::AudioCommand::SetMaxPhysicalVoices).
    #[inline]
    pub fn set_max_physical(&mut self, max_physical: usize) {
        self.pool.set_max_physical(max_physical);
    }

    /// Number of allocated (physical + virtual) voices mirrored locally.
    #[must_use]
    #[inline]
    pub fn active_count(&self) -> usize {
        self.pool.active_count()
    }

    /// Number of audible (physical) voices mirrored locally.
    #[must_use]
    #[inline]
    pub fn physical_count(&self) -> usize {
        self.pool.physical_count()
    }
}

/// Authoritative map from an [`Entity`] to the [`VoiceHandle`] of its live
/// voice.
///
/// [`RemovedComponents`](bevy_ecs::prelude::RemovedComponents) yields only the
/// entity (its [`AudioEmitter`](crate::emitter::AudioEmitter) data is already
/// gone), so this map is the source of truth the removal system consults to
/// find which voice to stop.
#[derive(Resource, Debug, Default)]
pub struct LiveVoices {
    /// Live voices keyed by owning entity.
    map: HashMap<Entity, VoiceHandle>,
}

impl LiveVoices {
    /// Records (or replaces) the live voice for `entity`, returning any handle
    /// previously stored for it.
    #[inline]
    pub fn insert(&mut self, entity: Entity, handle: VoiceHandle) -> Option<VoiceHandle> {
        self.map.insert(entity, handle)
    }

    /// Removes and returns the live voice recorded for `entity`, if any.
    #[inline]
    pub fn remove(&mut self, entity: Entity) -> Option<VoiceHandle> {
        self.map.remove(&entity)
    }

    /// Returns the live voice recorded for `entity`, if any.
    #[must_use]
    #[inline]
    pub fn get(&self, entity: Entity) -> Option<VoiceHandle> {
        self.map.get(&entity).copied()
    }

    /// Whether a live voice is recorded for `entity`.
    #[must_use]
    #[inline]
    pub fn contains(&self, entity: Entity) -> bool {
        self.map.contains_key(&entity)
    }

    /// Number of live voices recorded.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Whether no live voices are recorded.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

/// Queue of voice handles whose [`StopVoice`](prism_audio_rt::AudioCommand::StopVoice)
/// could not be enqueued because the ring was full.
///
/// Handles here have already been removed from [`LiveVoices`] but *not* yet
/// released from the [`VoiceMirror`]; the flush system retries them, releasing
/// the mirror slot only once the ring accepts the command.
#[derive(Resource, Debug, Default)]
pub struct PendingStops {
    /// Handles awaiting a retried stop.
    handles: Vec<VoiceHandle>,
}

impl PendingStops {
    /// Queues `handle` for a retried stop next frame.
    #[inline]
    pub fn push(&mut self, handle: VoiceHandle) {
        self.handles.push(handle);
    }

    /// Drains all queued handles, leaving the queue empty.
    #[must_use]
    #[inline]
    pub fn take_all(&mut self) -> Vec<VoiceHandle> {
        core::mem::take(&mut self.handles)
    }

    /// Number of handles awaiting a retried stop.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.handles.len()
    }

    /// Whether no stops are pending.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.handles.is_empty()
    }
}
