//! The real-time runtime that lives on the audio-callback thread and the client
//! handle that task threads use to drive it.
//!
//! [`AudioRuntime`] owns the live [`AudioGraph`] and [`VoicePool`], drains the
//! [command ring](crate::ring), renders a block, applies a sample-accurate
//! master-gain envelope, and publishes a [`TelemetryFrame`]. Every step on this
//! path is allocation-, lock-, and panic-free. Construction returns three
//! cooperating halves:
//!
//! - [`AudioRuntime`] — moved onto the audio-callback thread.
//! - [`AudioRuntimeClient`] — cloned onto any task/ECS thread to send commands,
//!   publish graphs, and read telemetry.
//! - [`Collector`] — parked on a task thread to drop resources the audio thread
//!   retires (e.g. superseded graphs).

use prism_audio_core::buffer::AudioBuffer;
use prism_audio_core::graph::AudioGraph;
use prism_audio_core::math::Sample;
use prism_audio_core::param::{Ramp, Smoothed};
use prism_audio_core::voice::VoicePool;

use crate::command::AudioCommand;
use crate::epoch::{Collector, GraphConsumer, GraphHandoff, GraphProducer, RetireQueue, Retirer};
use crate::ring::{RingConsumer, RingProducer, ring};
use crate::telemetry::TelemetryFrame;

/// Maximum number of master-gain changes honoured per block. Additional changes
/// drained in the same block are coalesced onto the last accepted slot's frame
/// rather than dropped, so no gain command is ever lost.
const MAX_GAIN_CHANGES_PER_BLOCK: usize = 64;

/// Configuration for constructing an [`AudioRuntime`] and its companions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioRuntimeConfig {
    /// Output sample rate in Hz.
    pub sample_rate: u32,
    /// Maximum number of frames the audio thread will render per block. Sizes
    /// the internal gain-envelope scratch so the hot path never allocates.
    pub max_block: usize,
    /// Capacity of the command ring (task threads -> audio thread).
    pub command_capacity: usize,
    /// Capacity of the telemetry ring (audio thread -> observer thread).
    pub telemetry_capacity: usize,
    /// Capacity of the deferred-reclamation queue (retired resources awaiting
    /// collection off the audio thread).
    pub retire_capacity: usize,
    /// Total number of pooled voices (physical + virtual).
    pub voice_capacity: usize,
    /// Maximum number of simultaneously audible (physical) voices.
    pub max_physical_voices: usize,
}

impl Default for AudioRuntimeConfig {
    #[inline]
    fn default() -> Self {
        Self {
            sample_rate: 48_000,
            max_block: 512,
            command_capacity: 1024,
            telemetry_capacity: 256,
            retire_capacity: 64,
            voice_capacity: 256,
            max_physical_voices: 64,
        }
    }
}

/// A pending, frame-scheduled master-gain change staged during
/// [`AudioRuntime::begin_block`] and consumed during rendering.
#[derive(Debug, Clone, Copy)]
struct GainChange {
    /// Frame offset within the block at which the ramp begins.
    at_frame: u32,
    /// Sanitized linear target gain.
    target: Sample,
    /// Ramp length in frames (`0` snaps immediately).
    ramp_frames: u32,
}

/// The client handle used by task/ECS threads to drive an [`AudioRuntime`].
///
/// Cloning yields another handle sharing the same rings and hand-off, so many
/// task threads can submit commands to the single audio thread.
#[derive(Debug, Clone)]
pub struct AudioRuntimeClient {
    /// Producer half of the command ring.
    command_tx: RingProducer<AudioCommand>,
    /// Consumer half of the telemetry ring.
    telemetry_rx: RingConsumer<TelemetryFrame>,
    /// Producer half of the capacity-one graph hand-off.
    graph_tx: GraphProducer,
}

impl AudioRuntimeClient {
    /// Sends a control command to the audio thread.
    ///
    /// # Errors
    ///
    /// Returns `Err(command)` when the command ring is full so the caller can
    /// retry or coalesce; never blocks.
    #[inline]
    pub fn send(&self, command: AudioCommand) -> Result<(), AudioCommand> {
        self.command_tx.push(command)
    }

    /// Publishes a freshly compiled graph for the audio thread to swap in.
    ///
    /// If a previously published graph has not been consumed yet it is handed
    /// back as `Some(previous)` for the caller to drop on this thread; the new
    /// graph always wins.
    #[inline]
    pub fn publish_graph(&self, graph: Box<AudioGraph>) -> Option<Box<AudioGraph>> {
        self.graph_tx.publish(graph)
    }

    /// Reads the most recent telemetry frame, if one is available. Returns the
    /// oldest un-read frame; call in a loop to drain the backlog.
    #[inline]
    #[must_use]
    pub fn recv_telemetry(&self) -> Option<TelemetryFrame> {
        self.telemetry_rx.pop()
    }
}

/// The real-time audio-thread runtime. See the [module docs](self) for the
/// end-to-end model and the real-time contract.
pub struct AudioRuntime {
    /// Output sample rate in Hz.
    sample_rate: u32,
    /// Maximum frames rendered per block; bounds `gain_scratch`.
    max_block: usize,
    /// Consumer half of the command ring.
    command_rx: RingConsumer<AudioCommand>,
    /// Producer half of the telemetry ring.
    telemetry_tx: RingProducer<TelemetryFrame>,
    /// Consumer half of the graph hand-off.
    graph_rx: GraphConsumer,
    /// Retirement producer for superseded resources.
    retire_tx: Retirer,
    /// The live render graph, if one has been published.
    graph: Option<Box<AudioGraph>>,
    /// A resource that could not be retired last block (queue was full) and
    /// must be re-offered before the audio thread drops anything itself.
    pending_retire: Option<Box<dyn core::any::Any + Send>>,
    /// Voice bookkeeping shared with higher layers.
    voices: VoicePool,
    /// Sample-accurate master-gain envelope.
    master_gain: Smoothed,
    /// Frame-scheduled gain changes staged for the current block.
    gain_changes: [GainChange; MAX_GAIN_CHANGES_PER_BLOCK],
    /// Number of valid entries in `gain_changes`.
    gain_change_count: usize,
    /// Per-sample gain envelope scratch, sized to `max_block` at construction.
    gain_scratch: Box<[Sample]>,
    /// Monotonic block index.
    block_index: u64,
    /// Playhead in frames.
    playhead: u64,
    /// Upper bound on commands drained per block, to keep worst-case work
    /// bounded even if a task thread floods the ring.
    command_drain_cap: usize,
}

/// Constructs an [`AudioRuntime`] together with a client handle and a
/// collector. Move the runtime onto the audio thread, clone the client onto
/// task threads, and park the collector on a task thread.
#[must_use]
pub fn runtime(config: AudioRuntimeConfig) -> (AudioRuntime, AudioRuntimeClient, Collector) {
    let (command_tx, command_rx) = ring::<AudioCommand>(config.command_capacity);
    let (telemetry_tx, telemetry_rx) = ring::<TelemetryFrame>(config.telemetry_capacity);
    let (graph_tx, graph_rx) = GraphHandoff::new();
    let (retire_tx, collector) = RetireQueue::new(config.retire_capacity);

    let max_block = config.max_block.max(1);
    let runtime = AudioRuntime {
        sample_rate: config.sample_rate.max(1),
        max_block,
        command_rx,
        telemetry_tx,
        graph_rx,
        retire_tx,
        graph: None,
        pending_retire: None,
        voices: VoicePool::new(config.voice_capacity, config.max_physical_voices),
        master_gain: Smoothed::new(1.0),
        gain_changes: [GainChange {
            at_frame: 0,
            target: 1.0,
            ramp_frames: 0,
        }; MAX_GAIN_CHANGES_PER_BLOCK],
        gain_change_count: 0,
        gain_scratch: vec![0.0; max_block].into_boxed_slice(),
        block_index: 0,
        playhead: 0,
        command_drain_cap: config.command_capacity.max(1),
    };
    let client = AudioRuntimeClient {
        command_tx,
        telemetry_rx,
        graph_tx,
    };
    (runtime, client, collector)
}

impl core::fmt::Debug for AudioRuntime {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // `AudioGraph` and the retired `dyn Any` payload are not `Debug`; report
        // their presence rather than their contents.
        f.debug_struct("AudioRuntime")
            .field("sample_rate", &self.sample_rate)
            .field("max_block", &self.max_block)
            .field("has_graph", &self.graph.is_some())
            .field("pending_retire", &self.pending_retire.is_some())
            .field("block_index", &self.block_index)
            .field("playhead", &self.playhead)
            .finish_non_exhaustive()
    }
}

impl AudioRuntime {
    /// The configured output sample rate in Hz.
    #[must_use]
    #[inline]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Read-only access to the voice pool for diagnostics and tests.
    #[must_use]
    #[inline]
    pub fn voices(&self) -> &VoicePool {
        &self.voices
    }

    /// The number of blocks rendered so far.
    #[must_use]
    #[inline]
    pub fn block_index(&self) -> u64 {
        self.block_index
    }

    /// The playhead position in frames.
    #[must_use]
    #[inline]
    pub fn playhead(&self) -> u64 {
        self.playhead
    }

    /// Whether a live graph is currently installed.
    #[must_use]
    #[inline]
    pub fn has_graph(&self) -> bool {
        self.graph.is_some()
    }

    /// Renders one block into `master_out`, returning the telemetry frame that
    /// was also published to the telemetry ring.
    ///
    /// This is the single audio-thread entry point. It swaps in any published
    /// graph, drains and applies pending commands, renders the graph, applies
    /// the sample-accurate master-gain envelope, advances voice timers, and
    /// emits telemetry — all allocation-, lock-, and panic-free.
    pub fn process_block(&mut self, master_out: &mut AudioBuffer) -> TelemetryFrame {
        let start = std::time::Instant::now();
        let frames = master_out.capacity_frames().min(self.max_block);
        master_out.set_active_frames(frames);

        self.begin_block(frames);
        self.render(frames, master_out);
        self.apply_master_gain(frames, master_out);
        self.voices.advance(frames as u64);
        self.playhead += frames as u64;

        let frame = self.build_telemetry(frames, master_out, start);
        // A dropped telemetry frame only costs an observer a snapshot; it must
        // never stall the audio thread, so a full ring is silently tolerated.
        let _ = self.telemetry_tx.push(frame);
        self.block_index += 1;
        frame
    }

    /// Swaps in a published graph (retiring the old one) and drains pending
    /// commands into voice/gain state.
    fn begin_block(&mut self, frames: usize) {
        self.swap_graph();
        self.drain_commands(frames);
    }

    /// Installs a newly published graph, if any, and retires the previous one
    /// without dropping it on the audio thread.
    fn swap_graph(&mut self) {
        // Re-offer any resource that could not be retired previously.
        if let Some(resource) = self.pending_retire.take()
            && let Err(resource) = self.retire_tx.retire(resource)
        {
            self.pending_retire = Some(resource);
        }
        // Only accept a new graph once the retire queue has drained our
        // holdover, guaranteeing we never have to drop on this thread.
        if self.pending_retire.is_none()
            && let Some(new_graph) = self.graph_rx.take()
            && let Some(old_graph) = self.graph.replace(new_graph)
        {
            let retired: Box<dyn core::any::Any + Send> = old_graph;
            if let Err(resource) = self.retire_tx.retire(retired) {
                self.pending_retire = Some(resource);
            }
        }
    }

    /// Drains up to `command_drain_cap` commands, applying voice/config changes
    /// immediately and staging frame-scheduled gain changes.
    fn drain_commands(&mut self, frames: usize) {
        self.gain_change_count = 0;
        let mut drained = 0;
        while drained < self.command_drain_cap {
            let Some(command) = self.command_rx.pop() else {
                break;
            };
            drained += 1;
            self.apply_command(command, frames);
        }
    }

    /// Applies a single drained command.
    fn apply_command(&mut self, command: AudioCommand, frames: usize) {
        match command {
            AudioCommand::SpawnVoice { request } => {
                let _ = self.voices.allocate(request);
            }
            AudioCommand::StopVoice { handle } => {
                let _ = self.voices.release(handle);
            }
            AudioCommand::SetVoiceImportance { handle, importance } => {
                let _ = self.voices.set_importance(handle, importance);
            }
            AudioCommand::SetMaxPhysicalVoices { max_physical } => {
                self.voices.set_max_physical(max_physical);
            }
            AudioCommand::SetMasterGain {
                linear,
                at_frame,
                ramp_frames,
            } => self.stage_gain_change(linear, at_frame, ramp_frames, frames),
        }
    }

    /// Stages a frame-scheduled master-gain change, keeping `gain_changes`
    /// sorted by `at_frame`.
    fn stage_gain_change(&mut self, linear: f32, at_frame: u32, ramp_frames: u32, frames: usize) {
        let last_frame = frames.saturating_sub(1) as u32;
        let change = GainChange {
            at_frame: at_frame.min(last_frame),
            target: sanitize_gain(linear),
            ramp_frames,
        };
        if self.gain_change_count < MAX_GAIN_CHANGES_PER_BLOCK {
            // Insertion sort into the sorted prefix.
            let mut i = self.gain_change_count;
            while i > 0 && self.gain_changes[i - 1].at_frame > change.at_frame {
                self.gain_changes[i] = self.gain_changes[i - 1];
                i -= 1;
            }
            self.gain_changes[i] = change;
            self.gain_change_count += 1;
        } else {
            // Ring is saturated for this block: coalesce onto the last slot so
            // the most recent target still takes effect rather than being lost.
            self.gain_changes[MAX_GAIN_CHANGES_PER_BLOCK - 1] = change;
        }
    }

    /// Renders the live graph into `master_out`, or clears it when no graph is
    /// installed.
    fn render(&mut self, frames: usize, master_out: &mut AudioBuffer) {
        if let Some(graph) = self.graph.as_mut() {
            graph.process(frames, self.playhead, master_out);
        } else {
            master_out.clear();
        }
    }

    /// Applies the sample-accurate master-gain envelope across all channels.
    fn apply_master_gain(&mut self, frames: usize, master_out: &mut AudioBuffer) {
        let mut cursor = 0;
        for (frame, slot) in self.gain_scratch[..frames].iter_mut().enumerate() {
            while cursor < self.gain_change_count
                && self.gain_changes[cursor].at_frame as usize <= frame
            {
                let change = self.gain_changes[cursor];
                let ramp = if change.ramp_frames == 0 {
                    Ramp::Immediate
                } else {
                    Ramp::Linear {
                        samples: change.ramp_frames,
                    }
                };
                self.master_gain.set_target(change.target, ramp);
                cursor += 1;
            }
            *slot = self.master_gain.next_sample();
        }
        let channels = master_out.channels();
        for channel in 0..channels {
            let data = master_out.channel_mut(channel);
            for (sample, gain) in data[..frames].iter_mut().zip(self.gain_scratch[..frames].iter()) {
                *sample *= *gain;
            }
        }
    }

    /// Computes and returns the telemetry frame for the block just rendered.
    fn build_telemetry(
        &self,
        frames: usize,
        master_out: &AudioBuffer,
        start: std::time::Instant,
    ) -> TelemetryFrame {
        let (peak, rms) = peak_rms(master_out, frames);
        let elapsed = start.elapsed().as_secs_f32();
        let budget = frames.max(1) as f32 / self.sample_rate as f32;
        let cpu_load = if budget > 0.0 { elapsed / budget } else { 0.0 };
        TelemetryFrame {
            block_index: self.block_index,
            playhead: self.playhead,
            frames: frames as u32,
            physical_voices: self.voices.physical_count() as u32,
            virtual_voices: self.voices.virtual_count() as u32,
            master_peak: peak,
            master_rms: rms,
            cpu_load,
        }
    }
}

/// Sanitizes a linear gain to a finite, non-negative value; non-finite or
/// negative inputs collapse to silence.
#[inline]
fn sanitize_gain(x: f32) -> f32 {
    if x.is_finite() && x >= 0.0 { x } else { 0.0 }
}

/// Computes the peak absolute magnitude and root-mean-square level of the first
/// `frames` frames across all channels of `buffer`.
fn peak_rms(buffer: &AudioBuffer, frames: usize) -> (Sample, Sample) {
    let channels = buffer.channels();
    if channels == 0 || frames == 0 {
        return (0.0, 0.0);
    }
    let mut peak = 0.0f32;
    let mut sum_sq = 0.0f32;
    for channel in 0..channels {
        for &sample in &buffer.channel(channel)[..frames] {
            let magnitude = sample.abs();
            if magnitude > peak {
                peak = magnitude;
            }
            sum_sq += sample * sample;
        }
    }
    let count = (channels * frames) as f32;
    let rms = (sum_sq / count).sqrt();
    (peak, rms)
}

#[cfg(test)]
mod tests {
    use super::{AudioRuntimeConfig, runtime};
    use crate::command::AudioCommand;
    use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
    use prism_audio_core::graph::AudioGraph;
    use prism_audio_core::voice::{VoiceGroup, VoiceRequest};

    fn stereo_block(frames: usize) -> AudioBuffer {
        AudioBuffer::new(ChannelLayout::Stereo, frames)
    }

    #[test]
    fn empty_runtime_renders_silence_and_publishes_telemetry() {
        let (mut rt, client, _collector) = runtime(AudioRuntimeConfig::default());
        let mut out = stereo_block(256);
        let frame = rt.process_block(&mut out);
        assert_eq!(frame.block_index, 0);
        assert_eq!(frame.frames, 256);
        assert_eq!(frame.playhead, 256);
        assert_eq!(frame.master_peak, 0.0);
        assert!(!rt.has_graph());
        let observed = client.recv_telemetry().expect("telemetry published");
        assert_eq!(observed, frame);
        assert_eq!(rt.playhead(), 256);
    }

    #[test]
    fn spawn_and_stop_voice_commands_move_pool_counts() {
        let (mut rt, client, _collector) = runtime(AudioRuntimeConfig::default());
        let request = VoiceRequest::new(VoiceGroup(1), 0.8);
        client
            .send(AudioCommand::SpawnVoice { request })
            .expect("command ring has room");
        let mut out = stereo_block(128);
        let frame = rt.process_block(&mut out);
        assert_eq!(frame.physical_voices, 1);
        assert_eq!(rt.voices().active_count(), 1);
    }

    #[test]
    fn master_gain_ramp_is_applied_to_output() {
        let (mut rt, client, _collector) = runtime(AudioRuntimeConfig::default());
        // Publish a DC graph so the master output is a constant 1.0 before gain.
        let graph = Box::new(build_dc_graph());
        assert!(client.publish_graph(graph).is_none());
        // Immediately halve the master gain from frame 0.
        client
            .send(AudioCommand::SetMasterGain {
                linear: 0.5,
                at_frame: 0,
                ramp_frames: 0,
            })
            .expect("command ring has room");
        let mut out = stereo_block(64);
        rt.process_block(&mut out);
        assert!(rt.has_graph());
        for &sample in &out.channel(0)[..64] {
            assert!((sample - 0.5).abs() < 1.0e-6, "sample={sample}");
        }
    }

    #[test]
    fn max_physical_voices_command_reaches_pool() {
        let (mut rt, client, _collector) = runtime(AudioRuntimeConfig::default());
        client
            .send(AudioCommand::SetMaxPhysicalVoices { max_physical: 3 })
            .expect("command ring has room");
        let mut out = stereo_block(32);
        rt.process_block(&mut out);
        assert_eq!(rt.voices().max_physical(), 3);
    }

    /// Builds a compiled graph whose master output is a constant `1.0` (DC).
    fn build_dc_graph() -> AudioGraph {
        use prism_audio_core::graph::{AudioNode, PortRef, ProcessIo, RenderContext};

        /// A node that writes a constant `1.0` to its single stereo output.
        struct Dc;
        impl AudioNode for Dc {
            fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
                let out = io.output(0);
                for channel in 0..out.channels() {
                    for sample in out.channel_mut(channel) {
                        *sample = 1.0;
                    }
                }
            }
        }

        let mut graph = AudioGraph::new(48_000, 512);
        let dc = graph.add_node(Box::new(Dc), vec![], vec![ChannelLayout::Stereo]);
        graph
            .set_master(PortRef::new(dc, 0))
            .expect("master port is valid");
        graph.compile().expect("graph compiles");
        graph
    }
}
