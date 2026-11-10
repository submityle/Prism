//! ECS-to-graph source routing.
//!
//! The asset store supplies decoder-verified resident PCM or retained encoded
//! media. Resident sources are copied into the graph at publication time;
//! streaming sources use a bounded background decoder and a lock-free chunk
//! queue, so the render callback only consumes already-decoded samples.

use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::OnceLock;
use std::thread;
use std::time::Duration;

use crossbeam_queue::ArrayQueue;
use prism_audio_assets::asset::{AudioAssetHandle, AudioAssetResidency};
use prism_audio_assets::streaming::prefetch::PrefetchStream;
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
use crate::client_resource::{AudioClient, AudioRuntimeMetadata};
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

/// Errors that prevent a compiled source route from being published.
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

/// Control-side state for the ECS graph hand-off.
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
    failed: AtomicBool,
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

const STREAM_CHUNK_FRAMES: usize = 1024;
const STREAM_QUEUE_CHUNKS: usize = 8;

/// Lock-free hand-off between one decoder producer and one graph consumer.
/// The queues carry preallocated sample vectors, so the consumer never
/// allocates or locks while rendering.
struct StreamBridge {
    ready: ArrayQueue<Vec<Sample>>,
    free: ArrayQueue<Vec<Sample>>,
    stop: AtomicBool,
    ended: AtomicBool,
    failed: AtomicBool,
    worker: OnceLock<thread::Thread>,
}

struct StreamingSource {
    bridge: Arc<StreamBridge>,
    channels: usize,
    sample_rate: u32,
}

enum ProducedChunk {
    Produced,
    NoBuffer,
    Ended,
}

fn produce_stream_chunk(
    stream: &mut PrefetchStream,
    bridge: &StreamBridge,
    chunk_frames: usize,
) -> Result<ProducedChunk, prism_audio_assets::codec::decoder::DecodeError> {
    let Some(mut buffer) = bridge.free.pop() else {
        return Ok(ProducedChunk::NoBuffer);
    };
    if stream.buffered_frames() == 0 && !stream.is_finished() {
        stream.pump(chunk_frames)?;
    }
    let frames = stream.buffered_frames().min(chunk_frames);
    if frames == 0 {
        let _ = bridge.free.push(buffer);
        return Ok(ProducedChunk::Ended);
    }

    buffer.clear();
    buffer.resize(frames * stream.channels(), 0.0);
    let result = stream.read(&mut buffer);
    if result.frames_silenced != 0 || result.frames_from_data != frames {
        let _ = bridge.free.push(buffer);
        return Err(prism_audio_assets::codec::decoder::DecodeError::UnexpectedEof);
    }
    match bridge.ready.push(buffer) {
        Ok(()) => Ok(ProducedChunk::Produced),
        Err(buffer) => {
            let _ = bridge.free.push(buffer);
            Ok(ProducedChunk::Produced)
        }
    }
}

fn wait_for_stream_consumer(bridge: &StreamBridge) {
    if bridge.worker.get().is_some() {
        thread::park_timeout(Duration::from_millis(1));
    } else {
        thread::yield_now();
    }
}

fn run_stream_worker(
    mut stream: PrefetchStream,
    bridge: Arc<StreamBridge>,
    looping: bool,
    chunk_frames: usize,
) {
    loop {
        if bridge.stop.load(Ordering::Acquire) {
            return;
        }
        if bridge.ready.len() >= STREAM_QUEUE_CHUNKS {
            wait_for_stream_consumer(&bridge);
            continue;
        }
        match produce_stream_chunk(&mut stream, &bridge, chunk_frames) {
            Ok(ProducedChunk::Produced) => {}
            Ok(ProducedChunk::NoBuffer) => wait_for_stream_consumer(&bridge),
            Ok(ProducedChunk::Ended) => {
                if looping {
                    if stream.restart().is_err() {
                        bridge.failed.store(true, Ordering::Release);
                        bridge.ended.store(true, Ordering::Release);
                        return;
                    }
                } else {
                    bridge.ended.store(true, Ordering::Release);
                    return;
                }
            }
            Err(_) => {
                bridge.failed.store(true, Ordering::Release);
                bridge.ended.store(true, Ordering::Release);
                return;
            }
        }
    }
}

impl StreamingSource {
    fn new(
        mut stream: PrefetchStream,
        start_frame: u64,
        mode: PlaybackMode,
    ) -> Result<Self, AudioSourceRouteError> {
        if start_frame != 0 {
            stream
                .seek(start_frame)
                .map_err(|_| AudioSourceRouteError::StreamingWorkerUnavailable)?;
        }
        let channels = stream.channels();
        let sample_rate = stream.info().sample_rate.max(1);
        let bridge = Arc::new(StreamBridge {
            ready: ArrayQueue::new(STREAM_QUEUE_CHUNKS),
            free: ArrayQueue::new(STREAM_QUEUE_CHUNKS + 2),
            stop: AtomicBool::new(false),
            ended: AtomicBool::new(false),
            failed: AtomicBool::new(false),
            worker: OnceLock::new(),
        });
        let chunk_samples = STREAM_CHUNK_FRAMES
            .checked_mul(channels)
            .ok_or(AudioSourceRouteError::StreamingWorkerUnavailable)?;
        for _ in 0..(STREAM_QUEUE_CHUNKS + 2) {
            bridge
                .free
                .push(Vec::with_capacity(chunk_samples))
                .map_err(|_| AudioSourceRouteError::StreamingWorkerUnavailable)?;
        }

        // Fill the first chunk synchronously so publication has deterministic
        // zero-latency startup; subsequent chunks are produced in the worker.
        match produce_stream_chunk(&mut stream, &bridge, STREAM_CHUNK_FRAMES) {
            Ok(ProducedChunk::Produced) if !mode.loops() && stream.is_finished() => {
                bridge.ended.store(true, Ordering::Release);
            }
            Ok(ProducedChunk::Produced) => {}
            Ok(ProducedChunk::NoBuffer) => {}
            Ok(ProducedChunk::Ended) => bridge.ended.store(true, Ordering::Release),
            Err(_) => {
                bridge.failed.store(true, Ordering::Release);
                bridge.ended.store(true, Ordering::Release);
            }
        }

        let worker_bridge = Arc::clone(&bridge);
        let chunk_frames = STREAM_CHUNK_FRAMES;
        let worker = thread::Builder::new()
            .name("prism-audio-stream".into())
            .spawn(move || run_stream_worker(stream, worker_bridge, mode.loops(), chunk_frames))
            .map_err(|_| AudioSourceRouteError::StreamingWorkerUnavailable)?;
        let _ = bridge.worker.set(worker.thread().clone());
        drop(worker);
        Ok(Self {
            bridge,
            channels,
            sample_rate,
        })
    }
}

struct RoutedStreamingNode {
    source: StreamingSource,
    fifo: Vec<Sample>,
    fifo_capacity_frames: usize,
    read_frame: usize,
    write_frame: usize,
    buffered_frames: usize,
    pending: Option<Vec<Sample>>,
    pending_frame: usize,
    source_position: f64,
    rendered_frames: u64,
    initial_playing: bool,
    playing: bool,
    progress: Arc<RouteProgress>,
}

impl RoutedStreamingNode {
    fn new(
        source: StreamingSource,
        start_frame: u64,
        playing: bool,
        max_block: usize,
        progress: Arc<RouteProgress>,
    ) -> Self {
        let fifo_capacity_frames = max_block
            .saturating_mul(8)
            .max(STREAM_CHUNK_FRAMES.saturating_mul(2));
        let fifo = alloc::vec![0.0; fifo_capacity_frames * source.channels];
        progress.playhead.store(start_frame, Ordering::Relaxed);
        progress.finished.store(false, Ordering::Relaxed);
        progress.failed.store(false, Ordering::Relaxed);
        Self {
            source,
            fifo,
            fifo_capacity_frames,
            read_frame: 0,
            write_frame: 0,
            buffered_frames: 0,
            pending: None,
            pending_frame: 0,
            source_position: 0.0,
            rendered_frames: start_frame,
            initial_playing: playing,
            playing,
            progress,
        }
    }

    fn return_chunk(&self, chunk: Vec<Sample>) {
        let _ = self.source.bridge.free.push(chunk);
        if let Some(worker) = self.source.bridge.worker.get() {
            worker.unpark();
        }
    }

    fn fill_fifo(&mut self) {
        loop {
            if self.pending.is_none() {
                self.pending = self.source.bridge.ready.pop();
                self.pending_frame = 0;
            }
            let Some(chunk) = self.pending.as_ref() else {
                return;
            };
            let chunk_frames = chunk.len() / self.source.channels;
            if chunk_frames == 0 {
                if let Some(chunk) = self.pending.take() {
                    self.return_chunk(chunk);
                }
                continue;
            }
            let writable = (self.fifo_capacity_frames - self.buffered_frames)
                .min(chunk_frames.saturating_sub(self.pending_frame));
            if writable == 0 {
                return;
            }
            for frame in 0..writable {
                let source_frame = self.pending_frame + frame;
                let destination_frame = (self.write_frame + frame) % self.fifo_capacity_frames;
                let source_start = source_frame * self.source.channels;
                let destination_start = destination_frame * self.source.channels;
                self.fifo[destination_start..destination_start + self.source.channels]
                    .copy_from_slice(&chunk[source_start..source_start + self.source.channels]);
            }
            self.write_frame = (self.write_frame + writable) % self.fifo_capacity_frames;
            self.buffered_frames += writable;
            self.pending_frame += writable;
            if self.pending_frame == chunk_frames {
                let Some(chunk) = self.pending.take() else {
                    return;
                };
                self.pending_frame = 0;
                self.return_chunk(chunk);
            }
        }
    }

    #[inline]
    fn sample(&self, frame: usize, channel: usize) -> Option<Sample> {
        if frame >= self.buffered_frames || channel >= self.source.channels {
            return None;
        }
        let index = ((self.read_frame + frame) % self.fifo_capacity_frames) * self.source.channels
            + channel;
        Some(self.fifo[index])
    }

    fn consume(&mut self, frames: usize) {
        let consumed = frames.min(self.buffered_frames);
        self.read_frame = (self.read_frame + consumed) % self.fifo_capacity_frames;
        self.buffered_frames -= consumed;
        self.rendered_frames = self.rendered_frames.saturating_add(consumed as u64);
    }

    fn publish_progress(&self) {
        self.progress
            .playhead
            .store(self.rendered_frames, Ordering::Relaxed);
        if self.source.bridge.failed.load(Ordering::Acquire) {
            self.progress.failed.store(true, Ordering::Release);
        }
        if self.source.bridge.ended.load(Ordering::Acquire)
            && self.buffered_frames == 0
            && self.pending.is_none()
        {
            self.progress.finished.store(true, Ordering::Release);
        }
    }
}

impl AudioNode for RoutedStreamingNode {
    fn process(&mut self, ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let out = io.output(0);
        for channel in 0..out.channels() {
            out.channel_mut(channel).fill(0.0);
        }
        if !self.playing {
            self.publish_progress();
            return;
        }
        let ratio = f64::from(self.source.sample_rate) / f64::from(ctx.sample_rate.max(1));
        for frame in 0..out.active_frames() {
            self.fill_fifo();
            let base = self.source_position as usize;
            if base >= self.buffered_frames {
                if self.source.bridge.ended.load(Ordering::Acquire) {
                    self.playing = false;
                }
                continue;
            }
            let frac = self.source_position - base as f64;
            for channel in 0..out.channels().min(self.source.channels) {
                let a = self.sample(base, channel).unwrap_or(0.0);
                let b = self.sample(base.saturating_add(1), channel).unwrap_or(a);
                out.channel_mut(channel)[frame] = a + (b - a) * frac as Sample;
            }
            self.source_position += ratio;
            let consumed = self.source_position as usize;
            if consumed != 0 {
                let actual = consumed.min(self.buffered_frames);
                self.consume(consumed);
                self.source_position -= actual as f64;
                if self.source_position < 0.0 {
                    self.source_position = 0.0;
                }
            }
        }
        self.publish_progress();
    }

    fn reset(&mut self) {
        // `AudioGraph::compile` calls reset after the source has been built.
        // Keep the synchronously prefetched queue available for that first
        // block; clearing it here would turn a valid stream into silence.
        self.playing = self.initial_playing;
        self.source_position = 0.0;
        self.read_frame = 0;
        self.write_frame = 0;
        self.buffered_frames = 0;
        self.progress
            .playhead
            .store(self.rendered_frames, Ordering::Relaxed);
        self.progress.finished.store(false, Ordering::Release);
        self.progress.failed.store(false, Ordering::Release);
    }
}

impl Drop for RoutedStreamingNode {
    fn drop(&mut self) {
        self.source.bridge.stop.store(true, Ordering::Release);
        if let Some(worker) = self.source.bridge.worker.get() {
            worker.unpark();
        }
        if let Some(chunk) = self.pending.take() {
            self.return_chunk(chunk);
        }
        while let Some(chunk) = self.source.bridge.ready.pop() {
            self.return_chunk(chunk);
        }
    }
}

impl Drop for StreamingSource {
    fn drop(&mut self) {
        self.bridge.stop.store(true, Ordering::Release);
        if let Some(worker) = self.bridge.worker.get() {
            worker.unpark();
        }
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
            if progress.failed.load(Ordering::Acquire) {
                player.state = AudioPlaybackState::AwaitingSourceRoute;
                player.playing = false;
                player.stop_requested = true;
            } else if progress.finished.load(Ordering::Acquire) && player.playing {
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
        | AudioSourceRouteError::StreamingDecode(_)
        | AudioSourceRouteError::StreamingWorkerUnavailable
        | AudioSourceRouteError::InvalidStreamMetadata
        | AudioSourceRouteError::InvalidResidentPcm
        | AudioSourceRouteError::UnsupportedChannelCount => AudioPlaybackState::AwaitingSourceRoute,
    }
}

enum RouteSource {
    Resident(ResidentAudioSource),
    Streaming(StreamingSource),
}

fn build_graph(
    sample_rate: u32,
    max_block: usize,
    output: ChannelLayout,
    sources: Vec<(Candidate, RouteSource)>,
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
        let (channels, source_node, progress) = match source {
            RouteSource::Resident(source) => {
                let channels = source.channels.len();
                let progress = Arc::new(RouteProgress::default());
                let node = graph.add_node(
                    Box::new(RoutedSampleNode::new(
                        source,
                        candidate.start_frame,
                        candidate.active,
                        candidate.mode,
                        Arc::clone(&progress),
                    )),
                    Vec::new(),
                    vec![match channels {
                        1 => ChannelLayout::Mono,
                        2 => ChannelLayout::Stereo,
                        _ => {
                            return Err(AudioGraphRouteError::UnsupportedSourceChannels(channels));
                        }
                    }],
                );
                (channels, node, progress)
            }
            RouteSource::Streaming(source) => {
                let channels = source.channels;
                let progress = Arc::new(RouteProgress::default());
                let node = graph.add_node(
                    Box::new(RoutedStreamingNode::new(
                        source,
                        candidate.start_frame,
                        candidate.active,
                        max_block,
                        Arc::clone(&progress),
                    )),
                    Vec::new(),
                    vec![match channels {
                        1 => ChannelLayout::Mono,
                        2 => ChannelLayout::Stereo,
                        _ => {
                            return Err(AudioGraphRouteError::UnsupportedSourceChannels(channels));
                        }
                    }],
                );
                (channels, node, progress)
            }
        };
        let source_layout = match channels {
            1 => ChannelLayout::Mono,
            2 => ChannelLayout::Stereo,
            _ => {
                return Err(AudioGraphRouteError::UnsupportedSourceChannels(channels));
            }
        };
        let output_node = if channels == 1 {
            let pan = graph.add_node(
                Box::new(StereoPanNode::new(candidate.pan)),
                vec![source_layout],
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

/// Reconciles asset-backed players into the real render graph.
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
        .get_resource::<AudioRuntimeMetadata>()
        .map(|metadata| (metadata.sample_rate, metadata.max_block))
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
        if !matches!(resolved.stream.channels, 1 | 2) {
            source_errors.push((
                candidate.entity,
                AudioSourceRouteError::UnsupportedChannelCount,
            ));
            source_error_entities.push(candidate.entity);
            continue;
        }
        if candidate.active {
            match resolved.residency {
                AudioAssetResidency::Resident => match store.resident_source(candidate.handle) {
                    Ok(source) => {
                        sources.push((*candidate, RouteSource::Resident(source)));
                        ready_entities.push(candidate.entity);
                    }
                    Err(error) => {
                        source_errors.push((candidate.entity, error));
                        source_error_entities.push(candidate.entity);
                    }
                },
                AudioAssetResidency::Streaming => {
                    match store.open_stream(candidate.handle, STREAM_CHUNK_FRAMES.saturating_mul(2))
                    {
                        Ok(stream) => match StreamingSource::new(
                            stream,
                            candidate.start_frame,
                            candidate.mode,
                        ) {
                            Ok(source) => {
                                sources.push((*candidate, RouteSource::Streaming(source)));
                                ready_entities.push(candidate.entity);
                            }
                            Err(error) => {
                                source_errors.push((candidate.entity, error));
                                source_error_entities.push(candidate.entity);
                            }
                        },
                        Err(error) => {
                            source_errors.push((candidate.entity, error));
                            source_error_entities.push(candidate.entity);
                        }
                    }
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
    fn streaming_media_publishes_real_pcm_and_reaches_end_of_stream() {
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
            AudioPlaybackState::Ready
        );
        let output = pump(&mut app, 2);
        assert!(
            output.channel(0).iter().any(|sample| *sample > 0.1),
            "stream output: {:?}, route: {:?}",
            output.channel(0),
            app.world().resource::<AudioGraphRouteState>().last_error()
        );
        app.update();
        let player = app.world().get::<AudioPlayer>(entity).expect("player");
        assert!(!player.playing);
        assert_eq!(player.state, AudioPlaybackState::Ready);
    }

    #[test]
    fn streaming_loop_restarts_from_the_decoder_without_fake_voice() {
        let mut app = app();
        let handle = load_bank(
            &mut app,
            Residency::Streaming { prefetch_frames: 1 },
            &[16_384, 8_192],
        );
        let entity = app.world_mut().spawn((
            AudioPlayer::from_asset(handle, VoiceGroup(0)),
            PlaybackSettings::LOOP.non_spatial(),
            GlobalTransform::default(),
        ));
        app.update();
        assert_eq!(
            app.world()
                .get::<AudioPlayer>(entity)
                .expect("player")
                .state,
            AudioPlaybackState::Ready
        );
        let output = pump(&mut app, 1);
        assert!(
            output.channel(0)[0] > 0.1,
            "stream output: {:?}, route: {:?}",
            output.channel(0),
            app.world().resource::<AudioGraphRouteState>().last_error()
        );
        app.update();
        let player = app.world().get::<AudioPlayer>(entity).expect("player");
        assert!(player.playing);
        assert!(!player.stop_requested);
    }

    #[test]
    fn streaming_route_decodes_chunks_after_the_initial_prefetch() {
        let mut app = app();
        let mut samples = vec![16_384i16; 2];
        samples.extend(vec![-16_384i16; 1_500]);
        let handle = load_bank(
            &mut app,
            Residency::Streaming { prefetch_frames: 2 },
            &samples,
        );
        let entity = app.world_mut().spawn((
            AudioPlayer::from_asset(handle, VoiceGroup(0)),
            PlaybackSettings::ONCE.non_spatial(),
            GlobalTransform::default(),
        ));
        app.update();
        let first = pump(&mut app, 2);
        assert!(first.channel(0).iter().all(|sample| *sample > 0.1));

        let mut saw_decoded_tail = false;
        for _ in 0..64 {
            thread::yield_now();
            let output = pump(&mut app, 64);
            if output.channel(0).iter().any(|sample| *sample < -0.1) {
                saw_decoded_tail = true;
                break;
            }
        }
        assert!(
            saw_decoded_tail,
            "background decoder never published the tail"
        );
        assert!(app
            .world()
            .get::<AudioPlayer>(entity)
            .is_some_and(|player| player.state == AudioPlaybackState::Ready));
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
