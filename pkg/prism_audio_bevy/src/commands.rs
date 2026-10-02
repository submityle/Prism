//! The pure translation layer between ECS state and
//! [`AudioCommand`](prism_audio_rt::AudioCommand)s, plus the thin send helpers
//! that keep the [`VoiceMirror`] in lock-step with the runtime.
//!
//! The functions here hold the whole policy for turning an
//! [`AudioEmitter`](crate::emitter::AudioEmitter) plus listener geometry into a
//! runtime-ready [`VoiceRequest`] and effective [`Importance`]. Distance
//! attenuation uses only [`bevy_math::ops`] so results are deterministic across
//! targets. The send helpers encode the mirror invariant: enqueue first, then
//! mutate the mirror only on acceptance.
//!
//! # Provenance
//!
//! Contains no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; original ECS glue. No AI/ML.
//!
//! # Relationship
//!
//! Called by [`crate::systems`]; reads [`crate::emitter::AudioEmitter`] and
//! writes to the [`crate::client_resource::AudioClient`] ring and the
//! [`crate::voice_registry::VoiceMirror`].

use bevy_math::Vec3;
use bevy_math::ops;
use prism_audio_core::voice::{Importance, VoiceHandle, VoiceRequest};
use prism_audio_rt::AudioCommand;

use crate::client_resource::AudioClient;
use crate::emitter::AudioEmitter;
use crate::voice_registry::VoiceMirror;

/// Clamps an importance to a finite, non-negative value, mapping non-finite
/// inputs to silence.
#[must_use]
#[inline]
fn sanitize(value: Importance) -> Importance {
    if value.is_finite() && value > 0.0 {
        value
    } else {
        0.0
    }
}

/// Computes an emitter's effective importance given its world position and the
/// optional listener position.
///
/// With no listener the base importance (sanitized) is returned unattenuated.
/// Otherwise the Euclidean distance is formed with [`bevy_math::ops::sqrt`] and
/// used to linearly fade the base importance from full at or inside
/// [`AudioEmitter::reference_distance`] down to zero at or beyond
/// [`AudioEmitter::max_distance`]. A non-positive or inverted distance band
/// yields full importance inside the reference radius and silence past it.
#[must_use]
pub fn effective_importance(
    emitter: &AudioEmitter,
    emitter_pos: Vec3,
    listener_pos: Option<Vec3>,
) -> Importance {
    let base = sanitize(emitter.importance);
    let Some(listener_pos) = listener_pos else {
        return base;
    };
    if base <= 0.0 {
        return 0.0;
    }

    let distance = ops::sqrt(emitter_pos.distance_squared(listener_pos));
    let reference = emitter.reference_distance.max(0.0);
    if distance <= reference {
        return base;
    }

    let span = emitter.max_distance - reference;
    if span <= 0.0 || distance >= emitter.max_distance {
        return 0.0;
    }

    let t = ((distance - reference) / span).clamp(0.0, 1.0);
    base * (1.0 - t)
}

/// Builds the [`VoiceRequest`] forwarded to the runtime for `emitter` at the
/// already-computed effective `importance`.
#[must_use]
pub fn build_request(emitter: &AudioEmitter, importance: Importance) -> VoiceRequest {
    VoiceRequest {
        group: emitter.group,
        priority: emitter.priority,
        importance,
        behavior: emitter.behavior,
    }
}

/// Enqueues a [`SpawnVoice`](AudioCommand::SpawnVoice) and, only on acceptance,
/// mirrors the allocation to recover the handle the runtime will mint.
///
/// Returns the predicted [`VoiceHandle`], or `None` if the ring was full or the
/// pool had no free slot.
pub fn spawn_voice(
    client: &AudioClient,
    mirror: &mut VoiceMirror,
    request: VoiceRequest,
) -> Option<VoiceHandle> {
    match client.send(AudioCommand::SpawnVoice { request }) {
        Ok(()) => mirror.allocate(request),
        Err(_) => None,
    }
}

/// Enqueues a [`StopVoice`](AudioCommand::StopVoice) and, only on acceptance,
/// releases the mirror slot. Returns whether the command was enqueued.
#[must_use]
pub fn stop_voice(client: &AudioClient, mirror: &mut VoiceMirror, handle: VoiceHandle) -> bool {
    match client.send(AudioCommand::StopVoice { handle }) {
        Ok(()) => {
            mirror.release(handle);
            true
        }
        Err(_) => false,
    }
}

/// Enqueues a [`SetVoiceImportance`](AudioCommand::SetVoiceImportance) and, only
/// on acceptance, mirrors the change. Returns whether the command was enqueued.
#[must_use]
pub fn set_importance(
    client: &AudioClient,
    mirror: &mut VoiceMirror,
    handle: VoiceHandle,
    importance: Importance,
) -> bool {
    match client.send(AudioCommand::SetVoiceImportance { handle, importance }) {
        Ok(()) => {
            mirror.set_importance(handle, importance);
            true
        }
        Err(_) => false,
    }
}
