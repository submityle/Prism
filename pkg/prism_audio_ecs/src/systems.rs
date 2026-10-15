//! Exclusive, deterministic control-plane reconciliation. No audio-thread work
//! or locking is added: only accepted bounded-ring commands mutate the mirror.
//! Temporary collections and ECS scans run on the App thread, not the callback.

use alloc::vec::Vec;
use prism_audio_core::voice::VoiceHandle;
use prism_audio_rt::AudioCommand;
use prism_ecs::{
    prelude::*,
    schedule::{SystemSet, SystemSetId},
};
use prism_math::Vec3;
use prism_transform::GlobalTransform;

use crate::commands::{
    build_request, effective_importance, set_importance, spawn_voice, stop_voice,
};
use crate::plugin::BridgeState;
use crate::{
    AudioClient, AudioEmitter, AudioListener, AudioPlayer, AudioTelemetry, LiveVoices, MasterGain,
    PendingStops, PhysicalVoiceBudget, RemovedAudioEmitters, VoiceMirror,
};

/// Ordering set for the native audio bridge in `prism_app::Last`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AudioSystems;

impl SystemSet for AudioSystems {
    fn set_id(&self) -> SystemSetId {
        SystemSetId::of::<Self>()
    }
}

/// Executes the whole bridge after transform propagation and deferred commands.
/// Stops precede allocation. Restore never rewinds the runtime pool: it retires
/// old handles through the FIFO before minting replacements. Tick values and
/// component handle copies are not treated as proof of runtime ownership.
pub fn audio_frame(world: &mut World) {
    if !world
        .get_resource::<AudioClient>()
        .is_some_and(AudioClient::is_open)
        || world.get_resource::<BridgeState>().is_none()
    {
        return;
    }
    crate::player_systems::sync_audio_players(world);
    crate::player_systems::apply_player_disposition(world);
    stop_removed_voices(world);
    stop_flagged_voices(world);
    flush_pending_stops(world);
    apply_physical_budget(world);
    apply_master_gain(world);
    spawn_voices(world);
    update_importance(world);
    prune_stolen(world);
    pump_telemetry(world);
}

fn detach(world: &mut World, entity: Entity, handle: VoiceHandle) {
    world.resource_mut::<LiveVoices>().remove(entity);
    world.resource_mut::<PendingStops>().push(handle);
    if let Some(mut emitter) = world.get_mut::<AudioEmitter>(entity) {
        emitter.voice = None;
        emitter.sent_importance = None;
    }
}

/// Reconciles authoritative ownership against removals, replacements, missing
/// poses, request changes, and wholesale snapshot restore, including same-frame
/// remove/reinsert. Clears copied or restored handle caches before spawning.
pub fn stop_removed_voices(world: &mut World) {
    let generation = world.archetype_generation();
    let reset = world.resource::<BridgeState>().generation != generation;
    if reset {
        let state = world.resource_mut::<BridgeState>();
        state.generation = generation;
        state.gain = None;
        state.budget = None;
    }
    let removed = world.resource_mut::<RemovedAudioEmitters>().take_all();
    let live: Vec<_> = world
        .resource::<LiveVoices>()
        .map
        .iter()
        .map(|(&e, &h)| (e, h))
        .collect();
    for (entity, handle) in live {
        let emitter = world.get::<AudioEmitter>(entity);
        let info = world.resource::<VoiceMirror>().get(handle);
        let invalid = reset
            || removed.contains(&entity)
            || world.get::<GlobalTransform>(entity).is_none()
            || emitter.is_none_or(|emitter| {
                emitter.voice != Some(handle)
                    || info.is_none_or(|info| {
                        info.group != emitter.group
                            || info.priority != emitter.priority
                            || info.behavior != emitter.behavior
                    })
            });
        if invalid {
            detach(world, entity, handle);
        }
    }
    let query = world.query::<(Entity, &AudioEmitter)>();
    let stale: Vec<_> = query
        .iter(world)
        .filter_map(|(entity, emitter)| {
            let expected = world.resource::<LiveVoices>().get(entity);
            (emitter.voice != expected || (expected.is_none() && emitter.sent_importance.is_some()))
                .then_some(entity)
        })
        .collect();
    for entity in stale {
        let expected = world.resource::<LiveVoices>().get(entity);
        let mut emitter = world
            .get_mut::<AudioEmitter>(entity)
            .expect("queried emitter");
        emitter.voice = expected;
        emitter.sent_importance = None;
    }
}

/// Consumes explicit stop requests even before allocation; a stopped emitter
/// remains stopped until the caller explicitly sets `playing = true`.
pub fn stop_flagged_voices(world: &mut World) {
    let query = world.query::<(Entity, &AudioEmitter)>();
    let stopped: Vec<_> = query
        .iter(world)
        .filter_map(|(entity, emitter)| {
            (!emitter.playing || emitter.stop_requested).then_some(entity)
        })
        .collect();
    for entity in stopped {
        if let Some(handle) = world.resource::<LiveVoices>().get(entity) {
            detach(world, entity, handle);
        }
        let mut emitter = world
            .get_mut::<AudioEmitter>(entity)
            .expect("queried emitter");
        emitter.playing = false;
        emitter.stop_requested = false;
    }
}

/// Retries pending stops, retaining mirror slots until the FIFO accepts them.
/// Stolen/already-killed handles need no stop command: their generations no
/// longer resolve in either pool. Allocations wait until this queue is empty.
pub fn flush_pending_stops(world: &mut World) {
    let client = world.resource::<AudioClient>().clone();
    let handles = world.resource_mut::<PendingStops>().take_all();
    for handle in handles {
        let mirror = world.resource_mut::<VoiceMirror>();
        if mirror.get(handle).is_some() && !stop_voice(&client, mirror, handle) {
            world.resource_mut::<PendingStops>().push(handle);
        }
    }
}

fn listener_pos(world: &mut World) -> Option<Vec3> {
    let query = world.query::<(Entity, &AudioListener, &GlobalTransform)>();
    query
        .iter(world)
        .min_by_key(|(entity, _, _)| *entity)
        .map(|(_, _, transform)| transform.translation())
}

/// Allocates playing emitters in stable entity order after pending stops clear.
pub fn spawn_voices(world: &mut World) {
    if !world.resource::<PendingStops>().is_empty() {
        return;
    }
    let listener = listener_pos(world);
    let query = world.query::<(Entity, &AudioEmitter, &GlobalTransform)>();
    let mut requests: Vec<_> = query
        .iter(world)
        .filter(|(_, emitter, _)| emitter.wants_voice())
        .map(|(entity, emitter, pose)| {
            let importance = effective_importance(emitter, pose.translation(), listener);
            (entity, build_request(emitter, importance))
        })
        .collect();
    requests.sort_unstable_by_key(|(entity, _)| *entity);
    let client = world.resource::<AudioClient>().clone();
    for (entity, request) in requests {
        let handle = spawn_voice(&client, world.resource_mut::<VoiceMirror>(), request);
        if let Some(handle) = handle
            && world.resource::<VoiceMirror>().get(handle).is_some()
        {
            world.resource_mut::<LiveVoices>().insert(entity, handle);
            let mut emitter = world
                .get_mut::<AudioEmitter>(entity)
                .expect("queried emitter");
            emitter.voice = Some(handle);
            emitter.sent_importance = Some(request.importance);
        }
    }
}

/// Sends exact importance changes, including sub-epsilon changes, retrying a
/// rejected enqueue without advancing the acknowledged component cache.
pub fn update_importance(world: &mut World) {
    let listener = listener_pos(world);
    let query = world.query::<(Entity, &AudioEmitter, &GlobalTransform)>();
    let updates: Vec<_> = query
        .iter(world)
        .filter_map(|(entity, emitter, pose)| {
            let handle = emitter.voice?;
            let importance = effective_importance(emitter, pose.translation(), listener);
            (emitter.playing
                && !emitter.stop_requested
                && emitter.sent_importance.map(f32::to_bits) != Some(importance.to_bits()))
            .then_some((entity, handle, importance))
        })
        .collect();
    let client = world.resource::<AudioClient>().clone();
    for (entity, handle, importance) in updates {
        if set_importance(
            &client,
            world.resource_mut::<VoiceMirror>(),
            handle,
            importance,
        ) {
            world
                .get_mut::<AudioEmitter>(entity)
                .expect("queried emitter")
                .sent_importance = Some(importance);
        }
    }
}

fn prune_stolen(world: &mut World) {
    let dead: Vec<_> = world
        .resource::<LiveVoices>()
        .map
        .iter()
        .filter_map(|(&entity, &handle)| {
            world
                .resource::<VoiceMirror>()
                .get(handle)
                .is_none()
                .then_some((entity, handle))
        })
        .collect();
    for (entity, handle) in dead {
        detach(world, entity, handle);
    }
    // Newly pruned handles do not resolve, so they can be discarded immediately
    // without issuing commands, even if the command ring is currently full.
    flush_pending_stops(world);
}

/// Forwards desired master gain using exact acknowledged bits, not ECS ticks.
/// Restore clears this acknowledgement; a full FIFO leaves it pending.
pub fn apply_master_gain(world: &mut World) {
    let master = *world.resource::<MasterGain>();
    let key = (master.linear.to_bits(), master.ramp_frames);
    if world.resource::<BridgeState>().gain == Some(key) {
        return;
    }
    if world
        .resource::<AudioClient>()
        .send_managed(AudioCommand::SetMasterGain {
            linear: master.linear,
            at_frame: 0,
            ramp_frames: master.ramp_frames,
        })
        .is_ok()
    {
        world.resource_mut::<BridgeState>().gain = Some(key);
    }
}

/// Forwards budget changes, mirroring only accepted commands. This runs before
/// allocation so both pools apply the same budget to every new request.
pub fn apply_physical_budget(world: &mut World) {
    let max = world.resource::<PhysicalVoiceBudget>().max;
    if world.resource::<BridgeState>().budget == Some(max) {
        return;
    }
    if world
        .resource::<AudioClient>()
        .send_managed(AudioCommand::SetMaxPhysicalVoices { max_physical: max })
        .is_ok()
    {
        world.resource_mut::<VoiceMirror>().set_max_physical(max);
        world.resource_mut::<BridgeState>().budget = Some(max);
    }
}

/// Drains available telemetry into the bounded ECS history on the App thread.
pub fn pump_telemetry(world: &mut World) {
    let client = world.resource::<AudioClient>().clone();
    while let Some(frame) = client.recv_telemetry() {
        world
            .resource_mut::<VoiceMirror>()
            .advance(frame.frames as u64);
        world.resource_mut::<AudioTelemetry>().push(frame);
    }
    sync_player_playheads(world);
}

/// Projects accepted voice playheads from the deterministic mirror onto ECS
/// players. Asset-backed players without a source route intentionally remain at
/// their last playhead (normally zero); they do not advance as fake playback.
fn sync_player_playheads(world: &mut World) {
    let entities: Vec<_> = world
        .query::<(Entity, &AudioPlayer)>()
        .iter(world)
        .map(|(entity, _)| entity)
        .collect();
    let updates: Vec<_> = {
        let live = world.resource::<LiveVoices>();
        let mirror = world.resource::<VoiceMirror>();
        entities
            .into_iter()
            .filter_map(|entity| {
                let handle = live.get(entity)?;
                let elapsed = mirror.get(handle)?.elapsed;
                Some((entity, elapsed))
            })
            .collect()
    };
    for (entity, playhead) in updates {
        if let Some(mut player) = world.get_mut::<AudioPlayer>(entity) {
            player.playhead = playhead;
        }
    }
}
