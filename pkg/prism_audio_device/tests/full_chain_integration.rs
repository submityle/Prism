//! Production-grade, end-to-end integration coverage for the real public API of
//! `prism_audio_device`: the device-agnostic render chain that turns the fixed
//! engine blocks of a [`prism_audio_rt::AudioRuntime`] into the arbitrary,
//! host-driven interleaved buffers consumed by platform audio SDKs and offline
//! WAV files, plus the lock-free capture path and the device channel-layout
//! negotiation.
//!
//! Each test drives one concern of the whole data flow with real engine types:
//! device layout negotiation, format/channel mapping, the [`BlockRenderer`]
//! pull adapter across variable callback buffer sizes, sample-exact
//! determinism, offline-versus-live equivalence, master-gain control-plane
//! commands, telemetry read-back, lock-free capture loopback with overflow
//! handling, and the error-reporting path.
//!
//! # Provenance
//!
//! This test file is original work authored for Prism. It contains no source or
//! derived code from Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, MPEG, Google Resonance, Web Audio, or Project Acoustics, and no
//! AI/ML models or generated coefficients. Where it exercises standard,
//! publicly documented audio engineering ideas (interleaving, planar buffers,
//! lock-free rings, equal-channel layouts), only the public idea is borrowed,
//! never any third-party implementation.
//!
//! # Relationship
//!
//! Covers the behaviour described by the `prism_audio_device` crate docs
//! (modules `render`, `file_sink`, `capture`, `interleave`, `cpal_backend`,
//! `error`) end to end. It depends only on the real public API of
//! `prism_audio_device`, `prism_audio_core` (buffers, the render graph, and the
//! band-limited oscillator source), and `prism_audio_rt` (the runtime, its
//! control-plane commands, and telemetry), plus `hound` for reading back the
//! offline WAV golden.

use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
use prism_audio_core::graph::{AudioGraph, PortRef};
use prism_audio_core::math::Sample;
use prism_audio_core::nodes::sources::{OscillatorNode, Waveform};
use prism_audio_core::voice::{VoiceGroup, VoiceRequest};
use prism_audio_device::{
    capture_ring, interleaved_to_planar, layout_for_channels, planar_to_interleaved, render_to_wav,
    BlockRenderer, DeviceError,
};
use prism_audio_rt::{runtime, AudioCommand, AudioRuntimeClient, AudioRuntimeConfig};

/// Sample rate shared by every pipeline so playhead maths stay exact.
const SR: u32 = 48_000;
/// Engine maximum block; the renderer clamps its scratch to this.
const MAX_BLOCK: usize = 512;
/// Comparison tolerance for approximate (non-bit-exact) sample checks.
const EPS: Sample = 1.0e-6;

/// Branch-free absolute value: the clippy rules forbid `std` float math on the
/// hot comparison path, so magnitude is computed by hand.
#[inline]
fn fabs(x: Sample) -> Sample {
    if x < 0.0 {
        -x
    } else {
        x
    }
}

/// Returns `true` when `a` and `b` are within [`EPS`] of each other.
#[inline]
fn close(a: Sample, b: Sample) -> bool {
    fabs(a - b) < EPS
}

/// Largest absolute magnitude across a slice, computed with the hand-written
/// [`fabs`] rather than `Sample::abs`.
fn peak(samples: &[Sample]) -> Sample {
    let mut hi = 0.0;
    for &s in samples {
        let m = fabs(s);
        if m > hi {
            hi = m;
        }
    }
    hi
}

/// Builds a freshly compiled stereo oscillator graph driving a brand-new
/// runtime, returning the real-time renderer half and the control-side client.
///
/// This is the real production assembly: construct the runtime, build and
/// compile an [`AudioGraph`] whose master is a band-limited [`OscillatorNode`],
/// publish the graph across the lock-free hand-off, and wrap the runtime in a
/// [`BlockRenderer`] driving `block_frames`-sized engine blocks.
fn build_pipeline(
    waveform: Waveform,
    freq: Sample,
    amp: Sample,
    block_frames: usize,
) -> (BlockRenderer, AudioRuntimeClient) {
    let cfg = AudioRuntimeConfig {
        sample_rate: SR,
        max_block: MAX_BLOCK,
        ..AudioRuntimeConfig::default()
    };
    let (rt, client, collector) = runtime(cfg);
    // The collector drops retired graphs off the audio thread; keep it alive
    // for the lifetime of the pipeline by leaking it into a long-lived home.
    core::mem::forget(collector);

    let mut graph = AudioGraph::new(SR, MAX_BLOCK);
    let osc = graph.add_node(
        Box::new(OscillatorNode::new(waveform, freq, amp)),
        Vec::new(),
        vec![ChannelLayout::Stereo],
    );
    graph
        .set_master(PortRef::new(osc, 0))
        .expect("master port is valid");
    graph.compile().expect("acyclic graph compiles");
    assert!(graph.is_compiled());
    assert!(client.publish_graph(Box::new(graph)).is_none());

    let renderer = BlockRenderer::new(rt, ChannelLayout::Stereo, block_frames);
    (renderer, client)
}

/// Renders exactly `frames` interleaved stereo frames through the shared pull
/// path and returns the owned interleaved buffer.
fn render_vec(renderer: &mut BlockRenderer, frames: usize) -> Vec<Sample> {
    let mut out = vec![0.0; frames * renderer.channels()];
    renderer.render_frames(frames, &mut out);
    out
}

#[test]
fn end_to_end_oscillator_graph_drives_block_renderer() {
    let (mut renderer, _client) = build_pipeline(Waveform::Sine, 440.0, 0.5, 128);
    assert_eq!(renderer.layout(), ChannelLayout::Stereo);
    assert_eq!(renderer.channels(), 2);

    // Two full engine blocks worth of frames (256 = 2 * 128).
    let out = render_vec(&mut renderer, 256);
    assert_eq!(out.len(), 512);

    // A real 0.5-amplitude sine must produce audible, bounded output.
    let p = peak(&out);
    assert!(p > 0.1, "oscillator produced near-silence: peak {p}");
    assert!(p <= 0.5 + EPS, "oscillator exceeded its amplitude: peak {p}");

    // Telemetry from the most recent block reflects the real render.
    let tel = renderer.last_telemetry();
    assert_eq!(tel.frames, 128, "last block rendered a full engine block");
    assert_eq!(tel.playhead, 256, "two 128-frame blocks advanced the playhead");
    assert!(tel.master_peak > 0.1, "telemetry peak tracks the signal");
    assert!(renderer.runtime().has_graph(), "graph was swapped in");
}

#[test]
fn identical_pipelines_are_bit_for_bit_deterministic() {
    let (mut a, _ca) = build_pipeline(Waveform::Saw, 110.0, 0.4, 96);
    let (mut b, _cb) = build_pipeline(Waveform::Saw, 110.0, 0.4, 96);

    // Render the same length through two independently constructed pipelines.
    let out_a = render_vec(&mut a, 2000);
    let out_b = render_vec(&mut b, 2000);

    assert_eq!(out_a.len(), out_b.len());
    // Determinism is a bit-exact property: compare raw IEEE-754 bit patterns so
    // the check is integer-exact and never hides a 1-ULP divergence.
    for (i, (x, y)) in out_a.iter().zip(out_b.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "sample {i} diverged between identical pipelines"
        );
    }
    // Guard against the degenerate "both silent" pass.
    assert!(peak(&out_a) > 0.05, "deterministic signal must be non-trivial");
}

#[test]
fn variable_callback_buffer_sizes_match_single_render() {
    // A device callback hands the renderer arbitrary, non-block-aligned sizes.
    // The partial-block carry must make the concatenated stream identical to a
    // single large render of the same total length.
    let block = 128;
    let chunk_sizes = [1usize, 7, 13, 64, 100, 128, 129, 255, 303];
    let total: usize = chunk_sizes.iter().sum();

    let (mut streamed, _cs) = build_pipeline(Waveform::Square, 220.0, 0.3, block);
    let (mut single, _cb) = build_pipeline(Waveform::Square, 220.0, 0.3, block);

    let mut streamed_out: Vec<Sample> = Vec::with_capacity(total * 2);
    for &frames in &chunk_sizes {
        let piece = render_vec(&mut streamed, frames);
        streamed_out.extend_from_slice(&piece);
    }
    let single_out = render_vec(&mut single, total);

    assert_eq!(streamed_out.len(), single_out.len());
    for (i, (x, y)) in streamed_out.iter().zip(single_out.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "block-boundary carry corrupted sample {i}"
        );
    }
}

#[test]
fn offline_wav_render_matches_live_pull_path() {
    let block = 160;
    let total: u64 = 4096;

    // Live reference: pull the stream straight through the renderer.
    let (mut live, _cl) = build_pipeline(Waveform::Triangle, 330.0, 0.45, block);
    let live_out = render_vec(&mut live, total as usize);

    // Offline sink: identical pipeline rendered to a 32-bit float WAV.
    let (mut offline, _co) = build_pipeline(Waveform::Triangle, 330.0, 0.45, block);
    let path = std::env::temp_dir().join("prism_full_chain_offline.wav");
    let written = render_to_wav(&mut offline, &path, total).expect("offline render succeeds");
    assert_eq!(written, total);

    let reader = hound::WavReader::open(&path).expect("golden WAV opens");
    let spec = reader.spec();
    assert_eq!(spec.channels, 2);
    assert_eq!(spec.sample_rate, SR);
    assert_eq!(spec.bits_per_sample, 32);
    assert_eq!(spec.sample_format, hound::SampleFormat::Float);

    let wav: Vec<Sample> = reader
        .into_samples::<f32>()
        .collect::<Result<Vec<_>, _>>()
        .expect("all WAV samples decode");
    let _ = std::fs::remove_file(&path);

    assert_eq!(wav.len(), live_out.len(), "offline frame count matches live");
    // The offline sink shares the exact BlockRenderer path, so a 32-bit float
    // WAV is a lossless container: the readback must be bit-identical.
    for (i, (disk, mem)) in wav.iter().zip(live_out.iter()).enumerate() {
        assert_eq!(
            disk.to_bits(),
            mem.to_bits(),
            "offline sample {i} differs from live render"
        );
    }
    assert!(peak(&wav) > 0.05, "offline render must contain real audio");
}

#[test]
fn master_gain_command_mutes_the_end_to_end_signal() {
    // Unity-gain reference.
    let (mut loud, _cl) = build_pipeline(Waveform::Sine, 500.0, 0.6, 128);
    let loud_out = render_vec(&mut loud, 512);
    assert!(peak(&loud_out) > 0.2, "unity-gain pipeline is audible");

    // Identical pipeline, but a control-plane command snaps master gain to zero
    // before the first block is pulled, so the whole stream is silent.
    let (mut muted, client) = build_pipeline(Waveform::Sine, 500.0, 0.6, 128);
    client
        .send(AudioCommand::SetMasterGain {
            linear: 0.0,
            at_frame: 0,
            ramp_frames: 0,
        })
        .expect("command ring has room");
    let muted_out = render_vec(&mut muted, 512);

    for (i, &s) in muted_out.iter().enumerate() {
        assert!(close(s, 0.0), "master mute leaked signal at sample {i}: {s}");
    }
    assert!(close(muted.last_telemetry().master_peak, 0.0));
}

#[test]
fn spawn_voice_command_is_observable_in_renderer_telemetry() {
    let (mut renderer, client) = build_pipeline(Waveform::Sine, 440.0, 0.3, 128);

    // Spawn two voices through the lock-free command ring before rendering.
    for importance in [0.9, 0.5] {
        client
            .send(AudioCommand::SpawnVoice {
                request: VoiceRequest::new(VoiceGroup(1), importance),
            })
            .expect("command ring has room");
    }

    let _ = render_vec(&mut renderer, 128);
    let tel = renderer.last_telemetry();
    assert_eq!(tel.physical_voices, 2, "both voices became physical");
    assert_eq!(tel.virtual_voices, 0);
    assert_eq!(
        renderer.runtime().voices().active_count(),
        2,
        "pool state agrees with telemetry"
    );

    // The same frame is published to the control-side telemetry ring.
    let observed = client.recv_telemetry().expect("telemetry frame published");
    assert_eq!(observed.physical_voices, 2);
    assert_eq!(observed.block_index, 0);
}

#[test]
fn planar_engine_block_interleaves_onto_device_channels() {
    // Drive a bare runtime (no BlockRenderer) so we can inspect the planar
    // master block directly, then map it onto device channel counts.
    let cfg = AudioRuntimeConfig {
        sample_rate: SR,
        max_block: MAX_BLOCK,
        ..AudioRuntimeConfig::default()
    };
    let (mut rt, client, collector) = runtime(cfg);
    core::mem::forget(collector);

    let mut graph = AudioGraph::new(SR, MAX_BLOCK);
    let osc = graph.add_node(
        Box::new(OscillatorNode::new(Waveform::Sine, 1000.0, 0.5)),
        Vec::new(),
        vec![ChannelLayout::Stereo],
    );
    graph.set_master(PortRef::new(osc, 0)).expect("valid master");
    graph.compile().expect("compiles");
    assert!(client.publish_graph(Box::new(graph)).is_none());

    let frames = 64;
    let mut planar = AudioBuffer::new(ChannelLayout::Stereo, frames);
    let tel = rt.process_block(&mut planar);
    assert_eq!(tel.frames as usize, frames);
    assert_eq!(planar.active_frames(), frames);

    // Stereo engine block -> 4-channel device buffer: channels 0/1 carry the
    // signal, channels 2/3 are silenced.
    let device_channels = 4;
    let mut quad = vec![9.0; frames * device_channels];
    planar_to_interleaved(&planar, frames, &mut quad);
    for frame in 0..frames {
        let base = frame * device_channels;
        assert!(close(quad[base], planar.channel(0)[frame]));
        assert!(close(quad[base + 1], planar.channel(1)[frame]));
        assert!(close(quad[base + 2], 0.0), "extra device channel silenced");
        assert!(close(quad[base + 3], 0.0), "extra device channel silenced");
    }

    // Interleave back into a mono capture buffer: the surplus right channel is
    // dropped, keeping only the left (channel 0).
    let mut mono = AudioBuffer::new(ChannelLayout::Mono, frames);
    let mut stereo_dev = vec![0.0; frames * 2];
    planar_to_interleaved(&planar, frames, &mut stereo_dev);
    interleaved_to_planar(&stereo_dev, frames, &mut mono);
    assert_eq!(mono.active_frames(), frames);
    for frame in 0..frames {
        assert!(close(mono.channel(0)[frame], planar.channel(0)[frame]));
    }
}

#[test]
fn capture_loopback_round_trips_engine_audio() {
    // Render real engine audio, feed it through the lock-free capture ring as
    // if it were a device input callback, and drain it back into a planar
    // buffer — a full output-to-input loopback over the real public API.
    let (mut renderer, _client) = build_pipeline(Waveform::Saw, 200.0, 0.4, 128);
    let frames = 300;
    let interleaved = render_vec(&mut renderer, frames);
    assert!(peak(&interleaved) > 0.05);

    let (sink, mut consumer) = capture_ring(2, frames);
    assert_eq!(sink.channels(), 2);
    let pushed = sink.push_interleaved(&interleaved);
    assert_eq!(pushed, interleaved.len(), "all frames fit the ring");
    assert_eq!(sink.dropped_samples(), 0, "no overflow on an exact-fit ring");
    assert_eq!(consumer.buffered_frames(), frames);

    let mut recorded = AudioBuffer::new(ChannelLayout::Stereo, frames);
    let drained = consumer.drain_into(&mut recorded);
    assert_eq!(drained, frames);
    assert_eq!(recorded.active_frames(), frames);
    assert_eq!(consumer.buffered_frames(), 0);

    // De-interleaving must reconstruct the original planar frames exactly.
    for frame in 0..frames {
        let left = interleaved[frame * 2];
        let right = interleaved[frame * 2 + 1];
        assert!(close(recorded.channel(0)[frame], left), "left frame {frame}");
        assert!(close(recorded.channel(1)[frame], right), "right frame {frame}");
    }
}

#[test]
fn capture_overflow_drops_whole_frames_and_counts_them() {
    // A ring with room for only two stereo frames must drop surplus frames
    // whole (frame-aligned) under overload rather than desynchronising the
    // channel interleave.
    let (sink, mut consumer) = capture_ring(2, 2);
    let pushed = sink.push_interleaved(&[1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0]);
    assert_eq!(pushed, 4, "only two of four frames fit");
    assert_eq!(
        sink.dropped_samples(),
        4,
        "the two dropped stereo frames count four samples"
    );

    let mut buffer = AudioBuffer::new(ChannelLayout::Stereo, 4);
    assert_eq!(consumer.drain_into(&mut buffer), 2);
    assert!(close(buffer.channel(0)[0], 1.0));
    assert!(close(buffer.channel(0)[1], 2.0));
    assert!(close(buffer.channel(1)[0], -1.0));
    assert!(close(buffer.channel(1)[1], -2.0));

    // A trailing partial frame (odd sample count) is ignored, not half-queued.
    let (sink2, consumer2) = capture_ring(2, 8);
    assert_eq!(sink2.push_interleaved(&[5.0, 5.0, 6.0, 6.0, 7.0]), 4);
    assert_eq!(consumer2.buffered_frames(), 2);
}

#[test]
fn layout_negotiation_maps_supported_channel_counts_and_rejects_others() {
    // The device backend negotiates format by mapping a reported device channel
    // count onto an engine ChannelLayout. Supported endpoints map cleanly.
    let supported = [
        (1u16, ChannelLayout::Mono),
        (2, ChannelLayout::Stereo),
        (4, ChannelLayout::Quad),
        (6, ChannelLayout::Surround5_1),
        (8, ChannelLayout::Surround7_1),
    ];
    for (channels, layout) in supported {
        let got = layout_for_channels(channels).expect("supported channel count");
        assert_eq!(got, layout);
        assert_eq!(got.channel_count(), channels as usize);
    }

    // Odd / unsupported endpoints are rejected with the exact channel count.
    for bad in [0u16, 3, 5, 7, 9] {
        match layout_for_channels(bad) {
            Err(DeviceError::UnsupportedChannelCount(n)) => {
                assert_eq!(n, bad as usize, "error reports the offending count");
            }
            other => panic!("expected UnsupportedChannelCount for {bad}, got {other:?}"),
        }
    }
}

#[test]
fn device_error_is_thread_safe_and_formats_each_variant() {
    // Every backend error must be a `'static`, `Send` value (SDK strings are
    // owned) so it can cross off the audio/device thread, and must render a
    // human-readable message.
    fn assert_send_static<T: Send + 'static>(_: &T) {}

    let errors = [
        DeviceError::NoOutputDevice,
        DeviceError::NoInputDevice,
        DeviceError::ConfigQuery("rate unsupported".into()),
        DeviceError::BuildStream("alsa refused".into()),
        DeviceError::PlayStream("device busy".into()),
        DeviceError::ChannelCountMismatch {
            renderer: 2,
            device: 6,
        },
        DeviceError::UnsupportedChannelCount(3),
        DeviceError::UnsupportedSampleFormat("u24".into()),
        DeviceError::Wav("truncated header".into()),
    ];

    for err in &errors {
        assert_send_static(err);
        let text = format!("{err}");
        assert!(!text.is_empty(), "every variant renders a message");
    }

    // Spot-check that the mismatch variant surfaces both channel counts.
    let mismatch = format!(
        "{}",
        DeviceError::ChannelCountMismatch {
            renderer: 2,
            device: 6,
        }
    );
    assert!(mismatch.contains('2') && mismatch.contains('6'));

    // The error implements the standard trait and can be moved to a thread.
    let moved = DeviceError::NoOutputDevice;
    let handle = std::thread::spawn(move || {
        let dyn_err: &dyn std::error::Error = &moved;
        format!("{dyn_err}")
    });
    let msg = handle.join().expect("thread joins");
    assert!(msg.contains("no output device"));
}

