//! The ECS systems that drive the command ring and pump telemetry, plus the
//! [`AudioSystems`] set that gives them a single deterministic ordering.
//!
//! The systems run chained in this order every frame: retry backlogged stops,
//! spawn newly playing emitters, retune importance, stop flagged voices, stop
//! voices whose [`AudioEmitter`] was removed, forward master-gain and
//! physical-voice-budget changes, then drain telemetry. Each command-producing
//! step tolerates a full ring without panicking: spawns simply retry next
//! frame, stops are parked in [`PendingStops`], and gain/budget changes latch a
//! retry flag.
//!
//! # Provenance
//!
//! Contains no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; original ECS glue. No AI/ML.
//!
//! # Relationship
//!
//! Reads [`crate::emitter`], [`crate::listener`], [`crate::master_gain`], and
//! [`crate::budget`] state; writes to the [`crate::client_resource::AudioClient`]
//! ring through [`crate::commands`]; fills [`crate::telemetry::AudioTelemetry`].

use bevy_ecs::change_detection::DetectChanges;
use bevy_ecs::entity::Entity;
use bevy_ecs::prelude::{Query, RemovedComponents, Res, ResMut, With};
use bevy_ecs::schedule::SystemSet;
use bevy_ecs::system::Local;
use bevy_transform::components::GlobalTransform;
use prism_audio_rt::AudioCommand;

use crate::budget::PhysicalVoiceBudget;
use crate::client_resource::AudioClient;
use crate::commands::{build_request, effective_importance, set_importance, spawn_voice, stop_voice};
use crate::emitter::AudioEmitter;
use crate::listener::AudioListener;
use crate::master_gain::MasterGain;
use crate::telemetry::AudioTelemetry;
use crate::voice_registry::{LiveVoices, PendingStops, VoiceMirror};

/// Minimum change in effective importance that justifies a new
/// [`SetVoiceImportance`](prism_audio_rt::AudioCommand::SetVoiceImportance),
/// suppressing redundant per-frame churn.
const IMPORTANCE_EPSILON: f32 = 1.0e-4;

/// System set containing every system this crate adds, letting dependent crates
/// order their own work relative to the whole audio bridge.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AudioSystems;

/// Retries stops that previously failed because the command ring was full.
///
/// Handles that still cannot be enqueued are re-parked for the next frame; the
/// mirror slot is released only once the stop is accepted.
pub fn flush_pending_stops(
    client: Res<AudioClient>,
    mut mirror: ResMut<VoiceMirror>,
    mut pending: ResMut<PendingStops>,
) {
    if pending.is_empty() {
        return;
    }
    for handle in pending.take_all() {
        if !stop_voice(&client, &mut mirror, handle) {
            pending.push(handle);
        }
    }
}

/// Spawns a voice for every emitter that wants one but has no live handle.
///
/// The effective importance is derived from the emitter and the first listener
/// found, if any. On acceptance the predicted handle is written back into the
/// emitter and recorded in [`LiveVoices`].
pub fn spawn_voices(
    client: Res<AudioClient>,
    mut mirror: ResMut<VoiceMirror>,
    mut live: ResMut<LiveVoices>,
    listeners: Query<&GlobalTransform, With<AudioListener>>,
    mut emitters: Query<(Entity, &mut AudioEmitter, &GlobalTransform)>,
) {
    let listener_pos = listeners.iter().next().map(GlobalTransform::translation);
    for (entity, mut emitter, transform) in &mut emitters {
        if !emitter.wants_voice() {
            continue;
        }
        let importance = effective_importance(&emitter, transform.translation(), listener_pos);
        let request = build_request(&emitter, importance);
        if let Some(handle) = spawn_voice(&client, &mut mirror, request) {
            emitter.voice = Some(handle);
            emitter.sent_importance = Some(importance);
            live.insert(entity, handle);
        }
    }
}

/// Retunes the importance of each live, playing voice when its distance-faded
/// value has moved past [`IMPORTANCE_EPSILON`] since the last accepted update.
pub fn update_importance(
    client: Res<AudioClient>,
    mut mirror: ResMut<VoiceMirror>,
    listeners: Query<&GlobalTransform, With<AudioListener>>,
    mut emitters: Query<(&mut AudioEmitter, &GlobalTransform)>,
) {
    let listener_pos = listeners.iter().next().map(GlobalTransform::translation);
    for (mut emitter, transform) in &mut emitters {
        let Some(handle) = emitter.voice else {
            continue;
        };
        if emitter.stop_requested || !emitter.playing {
            continue;
        }
        let importance = effective_importance(&emitter, transform.translation(), listener_pos);
        let changed = match emitter.sent_importance {
            Some(previous) => (importance - previous).abs() > IMPORTANCE_EPSILON,
            None => true,
        };
        if changed && set_importance(&client, &mut mirror, handle, importance) {
            emitter.sent_importance = Some(importance);
        }
    }
}

/// Stops the live voice of every emitter that requested a stop or stopped
/// wanting to play, clearing its bookkeeping once the command is accepted.
pub fn stop_flagged_voices(
    client: Res<AudioClient>,
    mut mirror: ResMut<VoiceMirror>,
    mut live: ResMut<LiveVoices>,
    mut emitters: Query<(Entity, &mut AudioEmitter)>,
) {
    for (entity, mut emitter) in &mut emitters {
        if !emitter.wants_stop() {
            continue;
        }
        let Some(handle) = emitter.voice else {
            continue;
        };
        if stop_voice(&client, &mut mirror, handle) {
            emitter.voice = None;
            emitter.sent_importance = None;
            emitter.stop_requested = false;
            live.remove(entity);
        }
    }
}

/// Stops voices whose owning [`AudioEmitter`] component was removed this frame.
///
/// The component data is already gone, so the handle is recovered from
/// [`LiveVoices`]; a full ring parks the handle in [`PendingStops`] for retry.
pub fn stop_removed_voices(
    client: Res<AudioClient>,
    mut mirror: ResMut<VoiceMirror>,
    mut live: ResMut<LiveVoices>,
    mut pending: ResMut<PendingStops>,
    mut removed: RemovedComponents<AudioEmitter>,
) {
    for entity in removed.read() {
        if let Some(handle) = live.remove(entity)
            && !stop_voice(&client, &mut mirror, handle)
        {
            pending.push(handle);
        }
    }
}

/// Forwards master-gain changes to the runtime, latching a retry if the ring is
/// momentarily full so no change is lost.
pub fn apply_master_gain(
    client: Res<AudioClient>,
    master: Res<MasterGain>,
    mut retry: Local<bool>,
) {
    if !master.is_changed() && !*retry {
        return;
    }
    let command = AudioCommand::SetMasterGain {
        linear: master.linear,
        at_frame: 0,
        ramp_frames: master.ramp_frames,
    };
    *retry = client.send(command).is_err();
}

/// Forwards physical-voice-budget changes to the runtime and mirrors them so
/// predicted handles stay exact, latching a retry if the ring is full.
pub fn apply_physical_budget(
    client: Res<AudioClient>,
    mut mirror: ResMut<VoiceMirror>,
    budget: Res<PhysicalVoiceBudget>,
    mut retry: Local<bool>,
) {
    if !budget.is_changed() && !*retry {
        return;
    }
    match client.send(AudioCommand::SetMaxPhysicalVoices {
        max_physical: budget.max,
    }) {
        Ok(()) => {
            mirror.set_max_physical(budget.max);
            *retry = false;
        }
        Err(_) => *retry = true,
    }
}

/// Drains every pending telemetry frame into the [`AudioTelemetry`] resource.
pub fn pump_telemetry(client: Res<AudioClient>, mut telemetry: ResMut<AudioTelemetry>) {
    while let Some(frame) = client.recv_telemetry() {
        telemetry.push(frame);
    }
}
