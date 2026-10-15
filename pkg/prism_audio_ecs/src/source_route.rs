//! ECS-to-graph source routing.
//!
//! This module is the resident-media route supplied by the ECS crate. It keeps
//! the boundary deliberately narrow: the asset store supplies decoder-verified
//! PCM, the core graph supplies DSP nodes, and this crate only reconciles ECS
//! playback state with that graph. Streaming decoders and project-specific
//! event/container policies remain explicit extension points rather than fake
//! successful voices.

use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use prism_audio_assets::asset::{AudioAssetHandle, AudioAssetResidency};
use prism_audio_core::buffer::ChannelLayout;
use prism_audio_core::graph::{
    AudioGraph, AudioNode, GraphError, PortRef, ProcessIo, RenderContext,
};
use prism_audio_core::math::Sample;
use prism_audio_core::nodes::sources::LoopMode;
use prism_audio_core::nodes::{GainNode, SamplePlayerNode, StereoPanNode, SumNode};
use prism_ecs::prelude::{Entity, Resource, World};
use prism_transform::GlobalTransform;

use crate::assets::{AudioAssetStore, AudioSourceRouteError, ResidentAudioSource};
use crate::client_resource::{AudioClient, AudioRuntimeHost};
use crate::commands::effective_importance;
use crate::listener::AudioListener;
use crate::playback::{PlaybackMode, PlaybackSettings};
use crate::player::{derive_emitter, AudioPlaybackState, AudioPlayer};

/// Requested output layout for the ECS graph route.
///
/// The current built-in resident route has a stereo spatializer. Other layouts
/// remain valid for an externally supplied graph, but are rejected here with a
/// diagnostic instead of being silently downmixed or causing an audio-thread
/// layout assertion.
#[derive(Resource, Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioOutputLayout(
    /// The physical layout expected by the graph hand-off consumer.
    pub ChannelLayout,
);

impl Default for AudioOutputLayout {
    fn default() -> Self {
        Self(ChannelLayout::Stereo)
    }
}

/// Errors that prevent a compiled resident route from being published.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AudioGraphRouteError {
    /// The runtime host was not installed in the world.
    MissingRuntime,
    /// The asset store was not installed in the world.
    MissingAssetStore,
    /// The built-in route only knows how to spatialize to stereo.
    UnsupportedOutputLayout(ChannelLayout),
    /// A source has more channels than the built-in route can describe.
    UnsupportedSourceChannels(usize),
    /// Graph validation or compilation failed before publication.
    Graph(GraphError),
    /// The runtime closed before the graph could be handed off.
    PublishRejected,
}

/// Control-side state for the resident graph hand-off.
///
/// The state also owns the small lock-free progress bridges written by routed
/// source nodes. The audio callback never touches ECS or this resource; it only
/// performs relaxed atomic stores on those bridges.
#[derive(Resource, Debug, Default)]
pub struct AudioGraphRouteState {
    signature: u64,
    initialized: bool,
    last_error: Option<AudioGraphRouteError>,
    bindings: Vec<RouteBinding>,
}

impl AudioGraphRouteState {
    /// Returns the last graph publication/build error, if any.
    #[must_use]
    pub fn last_error(&self) -> Option<AudioGraphRouteError> {
        self.last_error.clone()
    }

    /// Returns whether at least one graph has been published for the current
    /// route signature.
    #[must_use]
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }
}

#[derive(Debug, Default)]
struct RouteProgress {
    playhead: AtomicU64,
    finished: AtomicBool,
}

#[derive(Debug)]
struct RouteBinding {
    entity: Entity,
    progress: Arc<RouteProgress>,
    last_synced_playhead: u64,
}

/// A source node that reports progress without reaching into ECS from the
/// render thread. Compilation resets the node to the requested control-side
/// start frame, so graph publication remains deterministic.
struct RoutedSampleNode {
    inner: SamplePlayerNode,
    start_frame: f64,
    playing: bool,
    one_shot: bool,
    progress: Arc<RouteProgress>,
}

impl RoutedSampleNode {
    fn new(
        source: ResidentAudioSource,
        start_frame: u64,
        playing: bool,
        mode: PlaybackMode,
        progress: Arc<RouteProgress>,
    ) -> Self {
        let mut inner = SamplePlayerNode::new(source.channels, source.stream.sample_rate);
        let one_shot = !mode.loops();
        if mode.loops() {
            inner.set_loop_mode(LoopMode::Forward {
                start: 0,
                end: inner.frame_len().saturating_sub(1),
            });
        }
        inner.seek(start_frame as f64);
        if playing {
            inner.play();
        }
        progress.playhead.store(start_frame, Ordering::Relaxed);
        progress.finished.store(false, Ordering::Relaxed);
        Self {
            inner,
            start_frame: start_frame as f64,
            playing,
            one_shot,
            progress,
        }
    }

    fn publish_progress(&self) {
        self.progress
            .playhead
            .store(self.inner.position() as u64, Ordering::Relaxed);
        if self.one_shot && !self.inner.is_playing() {
            self.progress.finished.store(true, Ordering::Release);
        }
    }
}

impl AudioNode for RoutedSampleNode {
    fn process(&mut self, ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        self.inner.process(ctx, io);
        self.publish_progress();
    }

    fn reset(&mut self) {
        self.inner.reset();
        self.inner.seek(self.start_frame);
        if self.playing {
            self.inner.play();
        }
        self.progress
            .playhead
            .store(self.start_frame as u64, Ordering::Relaxed);
        self.progress.finished.store(false, Ordering::Relaxed);
    }
}

#[derive(Clone, Copy)]
struct Candidate {
    entity: Entity,
    handle: AudioAssetHandle,
    mode: PlaybackMode,
    active: bool,
    start_frame: u64,
    gain: Sample,
    pan: Sample,
    route_epoch: u64,
}

struct BuiltGraph {
    graph: AudioGraph,
    bindings: Vec<RouteBinding>,
}

fn listener_pose(world: &mut World) -> Option<GlobalTransform> {
    let query = world.query::<(Entity, &AudioListener, &GlobalTransform)>();
    query
        .iter(world)
        .min_by_key(|(entity, _, _)| *entity)
        .map(|(_, _, transform)| *transform)
}

fn mix_hash(mut hash: u64, value: u64) -> u64 {
    hash ^= value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    hash = hash.rotate_left(13).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    hash ^ (hash >> 29)
}

fn layout_tag(layout: ChannelLayout) -> u64 {
    match layout {
        ChannelLayout::Mono => 1,
        ChannelLayout::Stereo => 2,
        ChannelLayout::Quad => 3,
        ChannelLayout::Surround5_1 => 4,
        ChannelLayout::Surround7_1 => 5,
        ChannelLayout::AmbisonicFoa => 6,
        _ => 0,
    }
}

fn mode_tag(mode: PlaybackMode) -> u64 {
    match mode {
        PlaybackMode::Once => 1,
        PlaybackMode::Loop => 2,
        PlaybackMode::Despawn => 3,
        PlaybackMode::Remove => 4,
    }
}

fn signature(output: ChannelLayout, revision: u64, candidates: &[Candidate]) -> u64 {
    let mut hash = mix_hash(0x5052_4953_4d41_5544, layout_tag(output));
    hash = mix_hash(hash, revision);
    for candidate in candidates {
        hash = mix_hash(hash, candidate.entity.to_bits());
        hash = mix_hash(hash, candidate.handle.bank.0 as u64);
        hash = mix_hash(hash, candidate.handle.entry.0 as u64);
        hash = mix_hash(hash, mode_tag(candidate.mode));
        hash = mix_hash(hash, candidate.active as u64);
        hash = mix_hash(hash, candidate.gain.to_bits() as u64);
        hash = mix_hash(hash, candidate.pan.to_bits() as u64);
        hash = mix_hash(hash, candidate.route_epoch);
    }
    hash
}

fn stereo_pan(emitter: Option<&GlobalTransform>, listener: Option<GlobalTransform>) -> Sample {
    let (Some(emitter), Some(listener)) = (emitter, listener) else {
        return 0.0;
    };
    let delta = emitter.translation() - listener.translation();
    let distance_squared = delta.length_squared();
    if !distance_squared.is_finite() || distance_squared <= f32::EPSILON {
        return 0.0;
    }
    let distance = libm::sqrtf(distance_squared);
    let right = listener.compute_transform().local_x();
    let pan = delta.dot(right) / distance;
    if pan.is_finite() {
        pan.clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

fn sync_route_progress(world: &mut World) -> bool {
    let snapshots: Vec<_> = world
        .get_resource::<AudioGraphRouteState>()
        .map(|state| {
            state
                .bindings
                .iter()
                .map(|binding| {
                    (
                        binding.entity,
                        Arc::clone(&binding.progress),
                        binding.last_synced_playhead,
                    )
                })
                .collect()
        })
        .unwrap_or_default();
    if snapshots.is_empty() {
        return false;
    }

    let mut updates = Vec::with_capacity(snapshots.len());
    for (entity, progress, last_synced_playhead) in snapshots {
        let Some(mut player) = world.get_mut::<AudioPlayer>(entity) else {
            continue;
        };
        let external_seek = player.playhead != last_synced_playhead;
        if !external_seek {
            player.playhead = progress.playhead.load(Ordering::Acquire);
            if progress.finished.load(Ordering::Acquire) && player.playing {
                player.playing = false;
                player.stop_requested = true;
            }
        }
        updates.push((entity, player.playhead, external_seek));
    }

    let force_rebuild = updates.iter().any(|(_, _, external_seek)| *external_seek);
    let state = world.resource_mut::<AudioGraphRouteState>();
    for (entity, playhead, _) in updates {
        if let Some(binding) = state
            .bindings
            .iter_mut()
            .find(|binding| binding.entity == entity)
        {
            binding.last_synced_playhead = playhead;
        }
    }
    force_rebuild
}

fn collect_candidates(world: &mut World, listener: Option<GlobalTransform>) -> Vec<Candidate> {
    let query = world.query::<(
        Entity,
        &AudioPlayer,
        Option<&PlaybackSettings>,
        Option<&GlobalTransform>,
    )>();
    let mut candidates = Vec::new();
    for (entity, player, settings, transform) in query.iter(world) {
        let Some(handle) = player.asset else {
            continue;
        };
        let settings = settings.copied().unwrap_or_default();
        let emitter = derive_emitter(player, &settings);
        let position = transform
            .map(GlobalTransform::translation)
            .unwrap_or_default();
        let listener_position = listener.map(|pose| pose.translation());
        let gain = effective_importance(&emitter, position, listener_position);
        let pan = if settings.spatial {
            stereo_pan(transform, listener)
        } else {
            0.0
        };
        candidates.push(Candidate {
            entity,
            handle,
            mode: settings.mode,
            active: player.playing && !player.stop_requested,
            start_frame: player.playhead,
            gain,
            pan,
            route_epoch: player.route_epoch,
        });
    }
    candidates.sort_unstable_by_key(|candidate| candidate.entity);
    candidates
}

fn set_player_state(world: &mut World, entity: Entity, state: AudioPlaybackState) {
    if let Some(mut player) = world.get_mut::<AudioPlayer>(entity) {
        player.state = state;
    }
}

fn source_error_state(error: AudioSourceRouteError) -> AudioPlaybackState {
    match error {
        AudioSourceRouteError::NotLoaded => AudioPlaybackState::WaitingForAsset,
        AudioSourceRouteError::StreamingRequiresBackend
        | AudioSourceRouteError::InvalidStreamMetadata
        | AudioSourceRouteError::InvalidResidentPcm
        | AudioSourceRouteError::UnsupportedChannelCount => AudioPlaybackState::AwaitingSourceRoute,
    }
}

fn build_graph(
    sample_rate: u32,
    max_block: usize,
    output: ChannelLayout,
    sources: Vec<(Candidate, ResidentAudioSource)>,
) -> Result<BuiltGraph, AudioGraphRouteError> {
    if !sources.is_empty() && output != ChannelLayout::Stereo {
        return Err(AudioGraphRouteError::UnsupportedOutputLayout(output));
    }
    let mut graph = AudioGraph::new(sample_rate, max_block);
    let mixer = graph.add_node(
        Box::new(SumNode::new()),
        vec![output; sources.len()],
        vec![output],
    );
    let mut bindings = Vec::with_capacity(sources.len());
    for (index, (candidate, source)) in sources.into_iter().enumerate() {
        let channels = source.channels.len();
        let source_layout = match channels {
            1 => ChannelLayout::Mono,
            2 => ChannelLayout::Stereo,
            _ => {
                return Err(AudioGraphRouteError::UnsupportedSourceChannels(channels));
            }
        };
        let progress = Arc::new(RouteProgress::default());
        let source_node = graph.add_node(
            Box::new(RoutedSampleNode::new(
                source,
                candidate.start_frame,
                candidate.active,
                candidate.mode,
                Arc::clone(&progress),
            )),
            Vec::new(),
            vec![source_layout],
        );
        let output_node = if channels == 1 {
            let pan = graph.add_node(
                Box::new(StereoPanNode::new(candidate.pan)),
                vec![ChannelLayout::Mono],
                vec![ChannelLayout::Stereo],
            );
            graph
                .connect(PortRef::new(source_node, 0), PortRef::new(pan, 0))
                .map_err(AudioGraphRouteError::Graph)?;
            pan
        } else {
            source_node
        };
        let gain = graph.add_node(
            Box::new(GainNode::new(candidate.gain)),
            vec![ChannelLayout::Stereo],
            vec![ChannelLayout::Stereo],
        );
        graph
            .connect(PortRef::new(output_node, 0), PortRef::new(gain, 0))
            .map_err(AudioGraphRouteError::Graph)?;
        graph
            .connect(PortRef::new(gain, 0), PortRef::new(mixer, index))
            .map_err(AudioGraphRouteError::Graph)?;
        bindings.push(RouteBinding {
            entity: candidate.entity,
            progress,
            last_synced_playhead: candidate.start_frame,
        });
    }
    graph
        .set_master(PortRef::new(mixer, 0))
        .map_err(AudioGraphRouteError::Graph)?;
    graph.compile().map_err(AudioGraphRouteError::Graph)?;
    Ok(BuiltGraph { graph, bindings })
}

/// Reconciles resident asset-backed players into the real render graph.
///
/// The function is intentionally infallible at the ECS-system boundary. Any
/// build or publication failure is retained in [`AudioGraphRouteState`] and the
/// affected player remains in an explicit waiting/unavailable state; an invalid
/// asset or a closed runtime can therefore never panic the app thread.
pub fn sync_audio_source_routes(world: &mut World) {
    if world.get_resource::<AudioGraphRouteState>().is_none() {
        return;
    }
    let force_rebuild = sync_route_progress(world);
    let output = world
        .get_resource::<AudioOutputLayout>()
        .copied()
        .unwrap_or_default()
        .0;
    let listener = listener_pose(world);
    let candidates = collect_candidates(world, listener);
    let revision = world
        .get_resource::<AudioAssetStore>()
        .map(AudioAssetStore::revision)
        .unwrap_or_default();
    let key = signature(output, revision, &candidates);
    if world
        .get_resource::<AudioGraphRouteState>()
        .is_some_and(|state| state.initialized && state.signature == key && !force_rebuild)
    {
        return;
    }

    if output != ChannelLayout::Stereo {
        for candidate in &candidates {
            if candidate.active {
                set_player_state(
                    world,
                    candidate.entity,
                    AudioPlaybackState::AwaitingSourceRoute,
                );
            }
        }
        let state = world.resource_mut::<AudioGraphRouteState>();
        state.last_error = Some(AudioGraphRouteError::UnsupportedOutputLayout(output));
        state.initialized = false;
        return;
    }

    let Some((sample_rate, max_block)) = world
        .get_non_send::<AudioRuntimeHost>()
        .map(|host| (host.runtime().sample_rate(), host.runtime().max_block()))
    else {
        world.resource_mut::<AudioGraphRouteState>().last_error =
            Some(AudioGraphRouteError::MissingRuntime);
        return;
    };
    let Some(store) = world.get_resource::<AudioAssetStore>() else {
        world.resource_mut::<AudioGraphRouteState>().last_error =
            Some(AudioGraphRouteError::MissingAssetStore);
        return;
    };

    let mut sources = Vec::new();
    let mut source_errors = Vec::new();
    let mut source_error_entities = Vec::new();
    let mut ready_entities = Vec::new();
    for candidate in &candidates {
        let Some(resolved) = store.resolve(candidate.handle) else {
            source_errors.push((candidate.entity, AudioSourceRouteError::NotLoaded));
            source_error_entities.push(candidate.entity);
            continue;
        };
        if resolved.stream.channels == 0 || resolved.stream.sample_rate == 0 {
            source_errors.push((
                candidate.entity,
                AudioSourceRouteError::InvalidStreamMetadata,
            ));
            source_error_entities.push(candidate.entity);
            continue;
        }
        if !matches!(resolved.residency, AudioAssetResidency::Resident) {
            source_errors.push((
                candidate.entity,
                AudioSourceRouteError::StreamingRequiresBackend,
            ));
            source_error_entities.push(candidate.entity);
            continue;
        }
        if !matches!(resolved.stream.channels, 1 | 2) {
            source_errors.push((
                candidate.entity,
                AudioSourceRouteError::UnsupportedChannelCount,
            ));
            source_error_entities.push(candidate.entity);
            continue;
        }
        if candidate.active {
            match store.resident_source(candidate.handle) {
                Ok(source) => {
                    sources.push((*candidate, source));
                    ready_entities.push(candidate.entity);
                }
                Err(error) => {
                    source_errors.push((candidate.entity, error));
                    source_error_entities.push(candidate.entity);
                }
            }
        } else {
            ready_entities.push(candidate.entity);
        }
    }
    for (entity, error) in source_errors {
        set_player_state(world, entity, source_error_state(error));
    }
    for candidate in &candidates {
        if candidate.active && !source_error_entities.contains(&candidate.entity) {
            set_player_state(
                world,
                candidate.entity,
                AudioPlaybackState::AwaitingSourceRoute,
            );
        }
    }

    // Do not publish an empty Stereo graph merely because this ECS bridge is
    // installed. Other users may own the runtime graph (including a Mono
    // offline graph), and an empty route must never overwrite that graph. Once
    // this bridge has published sources, an empty graph is still published on
    // teardown so the old resident route is silenced.
    let clear_previous_route = {
        let state = world.resource::<AudioGraphRouteState>();
        state.initialized && !state.bindings.is_empty()
    };
    if sources.is_empty() && !clear_previous_route {
        for entity in ready_entities {
            set_player_state(world, entity, AudioPlaybackState::Ready);
        }
        return;
    }

    let built = match build_graph(sample_rate, max_block, output, sources) {
        Ok(built) => built,
        Err(error) => {
            world.resource_mut::<AudioGraphRouteState>().last_error = Some(error);
            return;
        }
    };
    let Some(client) = world.get_resource::<AudioClient>().cloned() else {
        world.resource_mut::<AudioGraphRouteState>().last_error =
            Some(AudioGraphRouteError::PublishRejected);
        return;
    };
    let published = client.try_publish_graph(Box::new(built.graph));
    if published.is_err() {
        world.resource_mut::<AudioGraphRouteState>().last_error =
            Some(AudioGraphRouteError::PublishRejected);
        return;
    }

    for entity in ready_entities {
        set_player_state(world, entity, AudioPlaybackState::Ready);
    }
    let state = world.resource_mut::<AudioGraphRouteState>();
    state.signature = key;
    state.initialized = true;
    state.last_error = None;
    state.bindings = built.bindings;
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::String;
    use prism_app::prelude::App;
    use prism_audio_assets::bank::{
        BankId, BankManifest, EntryId, MediaEntry, MediaFormat, Residency,
    };
    use prism_audio_assets::codec::CodecTag;
    use prism_audio_core::buffer::AudioBuffer;
    use prism_audio_core::voice::VoiceGroup;
    use prism_transform::GlobalTransform;

    use crate::assets::AudioAssetStore;
    use crate::client_resource::AudioRuntimeHost;
    use crate::plugin::AudioRuntimePlugin;

    fn app() -> App {
        let mut app = App::new();
        app.add_plugins(AudioRuntimePlugin::default());
        app
    }

    fn wav_mono(samples: &[i16]) -> Vec<u8> {
        let mut data = Vec::with_capacity(samples.len() * 2);
        for sample in samples {
            data.extend_from_slice(&sample.to_le_bytes());
        }
        let mut wav = Vec::with_capacity(44 + data.len());
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&48_000u32.to_le_bytes());
        wav.extend_from_slice(&(samples.len() as u32 * 2).to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&(data.len() as u32).to_le_bytes());
        wav.extend_from_slice(&data);
        wav
    }

    fn load_bank(app: &mut App, residency: Residency, samples: &[i16]) -> AudioAssetHandle {
        let blob = wav_mono(samples);
        let manifest =
            BankManifest::new(BankId(7), String::from("test"), 1).with_media(MediaEntry::new(
                EntryId(3),
                String::from("clip"),
                MediaFormat::new(CodecTag::Pcm, 1, 48_000, Some(samples.len() as u64)),
                residency,
                0,
                blob.len() as u64,
            ));
        let handle = AudioAssetHandle::new(BankId(7), EntryId(3));
        app.world_mut()
            .resource_mut::<AudioAssetStore>()
            .load_bank(manifest, &blob)
            .expect("test bank loads");
        handle
    }

    fn pump(app: &mut App, frames: usize) -> AudioBuffer {
        let mut output = AudioBuffer::new(ChannelLayout::Stereo, frames);
        app.world_mut()
            .get_non_send_mut::<AudioRuntimeHost>()
            .expect("runtime host")
            .pump_block(&mut output);
        output
    }

    #[test]
    fn resident_route_publishes_real_pcm_and_progress() {
        let mut app = app();
        let handle = load_bank(&mut app, Residency::Memory, &[16_384, 16_384, 16_384]);
        let entity = app.world_mut().spawn((
            AudioPlayer::from_asset(handle, VoiceGroup(0)),
            PlaybackSettings::ONCE.non_spatial(),
            GlobalTransform::default(),
        ));
        app.update();
        let output = pump(&mut app, 2);
        assert!(output.channel(0)[0] > 0.0);
        app.update();
        let player = app.world().get::<AudioPlayer>(entity).expect("player");
        assert!(player.playhead > 0 || !player.playing);
        assert_eq!(player.state, AudioPlaybackState::Ready);
    }

    #[test]
    fn streaming_media_awaits_a_stream_route_backend() {
        let mut app = app();
        let handle = load_bank(
            &mut app,
            Residency::Streaming { prefetch_frames: 2 },
            &[16_384, 16_384],
        );
        let entity = app.world_mut().spawn((
            AudioPlayer::from_asset(handle, VoiceGroup(0)),
            GlobalTransform::default(),
        ));
        app.update();
        assert_eq!(
            app.world()
                .get::<AudioPlayer>(entity)
                .expect("player")
                .state,
            AudioPlaybackState::AwaitingSourceRoute
        );
    }

    #[test]
    fn unsupported_output_layout_does_not_panic_or_publish_a_route() {
        let mut app = app();
        let handle = load_bank(&mut app, Residency::Memory, &[16_384]);
        app.world_mut()
            .insert_resource(AudioOutputLayout(ChannelLayout::Quad));
        let entity = app.world_mut().spawn((
            AudioPlayer::from_asset(handle, VoiceGroup(0)),
            GlobalTransform::default(),
        ));
        app.update();
        assert_eq!(
            app.world()
                .get::<AudioPlayer>(entity)
                .expect("player")
                .state,
            AudioPlaybackState::AwaitingSourceRoute
        );
        assert_eq!(
            app.world().resource::<AudioGraphRouteState>().last_error(),
            Some(AudioGraphRouteError::UnsupportedOutputLayout(
                ChannelLayout::Quad
            ))
        );
    }
}
