//! Ordered ECS systems that translate audio state into runtime commands.

use prism_audio_rt::AudioCommand;
use prism_ecs::prelude::{Entity, Local, Query, Res, ResMut, With};
use prism_ecs::schedule::{SystemSet, SystemSetId};
use prism_transform::GlobalTransform;

use crate::budget::PhysicalVoiceBudget;
use crate::client_resource::AudioClient;
use crate::commands::{
    build_request, effective_importance, set_importance, spawn_voice, stop_voice,
};
use crate::emitter::AudioEmitter;
use crate::listener::AudioListener;
use crate::master_gain::MasterGain;
use crate::telemetry::AudioTelemetry;
use crate::voice_registry::{LiveVoices, PendingStops, RemovedAudioEmitters, VoiceMirror};

const IMPORTANCE_EPSILON: f32 = 1.0e-4;

/// Ordering set containing the complete audio bridge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AudioSystems;

impl SystemSet for AudioSystems {
    fn set_id(&self) -> SystemSetId {
        SystemSetId::of::<Self>()
    }
}

/// Retries stops rejected by a full command ring.
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

/// Spawns every emitter that is playing without a mirrored live handle.
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

/// Updates live voice importance only after it crosses the configured epsilon.
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
        let changed = emitter
            .sent_importance
            .is_none_or(|previous| (importance - previous).abs() > IMPORTANCE_EPSILON);
        if changed && set_importance(&client, &mut mirror, handle, importance) {
            emitter.sent_importance = Some(importance);
        }
    }
}

/// Stops emitters that no longer want to play and clears their bookkeeping.
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

/// Stops voices whose emitter was removed, using the lifecycle-observer queue.
pub fn stop_removed_voices(
    client: Res<AudioClient>,
    mut mirror: ResMut<VoiceMirror>,
    mut live: ResMut<LiveVoices>,
    mut pending: ResMut<PendingStops>,
    mut removed: ResMut<RemovedAudioEmitters>,
) {
    for entity in removed.take_all() {
        if let Some(handle) = live.remove(entity)
            && !stop_voice(&client, &mut mirror, handle)
        {
            pending.push(handle);
        }
    }
}

/// Forwards a changed master gain, retrying if the command ring is full.
pub fn apply_master_gain(
    client: Res<AudioClient>,
    master: Res<MasterGain>,
    mut last: Local<Option<MasterGain>>,
    mut retry: Local<bool>,
) {
    if !*retry && last.as_ref().is_some_and(|previous| *previous == *master) {
        return;
    }
    let command = AudioCommand::SetMasterGain {
        linear: master.linear,
        at_frame: 0,
        ramp_frames: master.ramp_frames,
    };
    *retry = client.send(command).is_err();
    if !*retry {
        *last = Some(*master);
    }
}

/// Forwards a changed physical budget and updates the mirror only on success.
pub fn apply_physical_budget(
    client: Res<AudioClient>,
    mut mirror: ResMut<VoiceMirror>,
    budget: Res<PhysicalVoiceBudget>,
    mut last: Local<Option<PhysicalVoiceBudget>>,
    mut retry: Local<bool>,
) {
    if !*retry && last.as_ref().is_some_and(|previous| *previous == *budget) {
        return;
    }
    match client.send_managed(AudioCommand::SetMaxPhysicalVoices {
        max_physical: budget.max,
    }) {
        Ok(()) => {
            mirror.set_max_physical(budget.max);
            *last = Some(*budget);
            *retry = false;
        }
        Err(_) => *retry = true,
    }
}

/// Drains all available runtime telemetry into the bounded ECS history.
pub fn pump_telemetry(client: Res<AudioClient>, mut telemetry: ResMut<AudioTelemetry>) {
    while let Some(frame) = client.recv_telemetry() {
        telemetry.push(frame);
    }
}
