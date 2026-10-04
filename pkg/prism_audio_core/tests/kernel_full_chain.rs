//! AAA whole-kernel integration coverage for `prism_audio_core`.
//!
//! These tests exercise the public DSP kernel the way higher layers do: they
//! assemble a real [`AudioGraph`] out of concrete source/transform nodes,
//! compile it, and pull blocks through the allocation-free block renderer,
//! then assert the render-graph summing, layout validation, and determinism
//! contracts. They additionally drive the sample-accurate scheduling
//! (`EventScheduler`, `NamedClock`, `Transport`), the fixed-capacity
//! `VoicePool` stealing/virtualization policy, and the click-free `Smoothed`
//! parameter so the full control-plane-to-audio-plane path is covered end to
//! end rather than per unit.
//!
//! # Provenance
//!
//! Original work authored for Prism. Contains no Unreal Engine, Unity, Godot,
//! Wwise, FMOD, Steam Audio, Dolby, MPEG, Google Resonance, Web Audio, or
//! Project Acoustics source or derived code, and no AI/ML. It relies only on
//! the crate's own public API and standard, publicly documented
//! signal-processing concepts expressed independently.
//!
//! # Relationship
//!
//! Validates the unified render graph (design doc "render graph" and
//! "scheduling/voice management" chapters) through `prism_audio_core`'s public
//! surface: [`AudioGraph`], the concrete [`nodes`] vocabulary, [`VoicePool`],
//! [`EventScheduler`], [`NamedClock`], [`Transport`], and [`Smoothed`].

use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
use prism_audio_core::graph::{AudioGraph, GraphError, PortRef};
use prism_audio_core::nodes::biquad::{BiquadKind, BiquadNode};
use prism_audio_core::nodes::gain::GainNode;
use prism_audio_core::nodes::sources::{OscillatorNode, Waveform};
use prism_audio_core::param::{Ramp, Smoothed};
use prism_audio_core::scheduler::{EventScheduler, Grid, NamedClock};
use prism_audio_core::time::{TimeSignature, Transport};
use prism_audio_core::voice::{VirtualBehavior, VoiceGroup, VoicePool, VoiceRequest, VoiceState};

const SR: u32 = 48_000;
const MAX_BLOCK: usize = 512;

/// Branchless absolute value; the workspace lints forbid `f32::abs` in tests
/// to keep signal math deterministic and intrinsic-free.
fn fabs(x: f32) -> f32 {
    if x < 0.0 {
        -x
    } else {
        x
    }
}

/// Builds the canonical `oscillator -> gain -> low-pass` stereo mix graph used
/// by several tests. Returns the compiled graph ready for `process`.
fn build_voice_chain(freq: f32, amp: f32, gain: f32) -> AudioGraph {
    let mut graph = AudioGraph::new(SR, MAX_BLOCK);
    let osc = graph.add_node(
        Box::new(OscillatorNode::new(Waveform::Sine, freq, amp)),
        Vec::new(),
        vec![ChannelLayout::Stereo],
    );
    let trim = graph.add_node(
        Box::new(GainNode::new(gain)),
        vec![ChannelLayout::Stereo],
        vec![ChannelLayout::Stereo],
    );
    let lp = graph.add_node(
        Box::new(BiquadNode::new(
            BiquadKind::LowPass,
            SR,
            1_200.0,
            0.707,
            0.0,
            2,
        )),
        vec![ChannelLayout::Stereo],
        vec![ChannelLayout::Stereo],
    );
    graph
        .connect(PortRef::new(osc, 0), PortRef::new(trim, 0))
        .expect("source feeds trim");
    graph
        .connect(PortRef::new(trim, 0), PortRef::new(lp, 0))
        .expect("trim feeds filter");
    graph
        .set_master(PortRef::new(lp, 0))
        .expect("filter is master");
    graph.compile().expect("acyclic graph compiles");
    graph
}

#[test]
fn compiled_voice_chain_renders_finite_bounded_signal() {
    let mut graph = build_voice_chain(220.0, 0.5, 0.5);
    let mut master = AudioBuffer::new(ChannelLayout::Stereo, MAX_BLOCK);

    let mut playhead = 0u64;
    let mut energy = 0.0f32;
    for _ in 0..8 {
        graph.process(128, playhead, &mut master);
        for ch in 0..master.channels() {
            for &s in master.channel(ch) {
                assert!(s.is_finite(), "sample must be finite");
                assert!(fabs(s) <= 1.0, "trimmed signal stays bounded");
                energy += s * s;
            }
        }
        playhead += 128;
    }
    assert!(energy > 0.0, "a live oscillator chain must not be silent");
}

#[test]
fn fan_in_connections_sum_into_a_shared_input_port() {
    // Two unity sources summed into one trim input should produce more energy
    // than a single source through the same port.
    fn energy_of(sources: usize) -> f32 {
        let mut graph = AudioGraph::new(SR, MAX_BLOCK);
        let trim = graph.add_node(
            Box::new(GainNode::new(1.0)),
            vec![ChannelLayout::Stereo],
            vec![ChannelLayout::Stereo],
        );
        for _ in 0..sources {
            let osc = graph.add_node(
                Box::new(OscillatorNode::new(Waveform::Sine, 110.0, 0.25)),
                Vec::new(),
                vec![ChannelLayout::Stereo],
            );
            graph
                .connect(PortRef::new(osc, 0), PortRef::new(trim, 0))
                .expect("source fans into shared input");
        }
        graph
            .set_master(PortRef::new(trim, 0))
            .expect("trim is master");
        graph.compile().expect("compiles");
        let mut master = AudioBuffer::new(ChannelLayout::Stereo, MAX_BLOCK);
        graph.process(256, 0, &mut master);
        let mut e = 0.0f32;
        for &s in master.channel(0) {
            e += s * s;
        }
        e
    }
    let one = energy_of(1);
    let two = energy_of(2);
    assert!(one > 0.0, "single source produces signal");
    // Two phase-aligned identical sources sum constructively (≈4x energy).
    assert!(two > one * 2.0, "summed fan-in increases port energy");
}

#[test]
fn connecting_mismatched_layouts_is_rejected() {
    let mut graph = AudioGraph::new(SR, MAX_BLOCK);
    let mono_src = graph.add_node(
        Box::new(OscillatorNode::new(Waveform::Sine, 440.0, 0.5)),
        Vec::new(),
        vec![ChannelLayout::Mono],
    );
    let stereo_trim = graph.add_node(
        Box::new(GainNode::new(1.0)),
        vec![ChannelLayout::Stereo],
        vec![ChannelLayout::Stereo],
    );
    let err = graph
        .connect(PortRef::new(mono_src, 0), PortRef::new(stereo_trim, 0))
        .expect_err("mono into stereo must be rejected");
    assert!(matches!(err, GraphError::LayoutMismatch { .. }));
}

#[test]
fn render_graph_is_deterministic_bit_identical() {
    let mut a = build_voice_chain(330.0, 0.4, 0.75);
    let mut b = build_voice_chain(330.0, 0.4, 0.75);
    let mut buf_a = AudioBuffer::new(ChannelLayout::Stereo, MAX_BLOCK);
    let mut buf_b = AudioBuffer::new(ChannelLayout::Stereo, MAX_BLOCK);

    let mut playhead = 0u64;
    for _ in 0..6 {
        a.process(200, playhead, &mut buf_a);
        b.process(200, playhead, &mut buf_b);
        for ch in 0..buf_a.channels() {
            let ca = buf_a.channel(ch);
            let cb = buf_b.channel(ch);
            for (sa, sb) in ca.iter().zip(cb) {
                // Bit-identical reproducibility is an integer comparison.
                assert_eq!(sa.to_bits(), sb.to_bits());
            }
        }
        playhead += 200;
    }
}

#[test]
fn smoothed_gain_ramps_monotonically_without_overshoot() {
    let mut g = Smoothed::new(0.0);
    g.set_target(1.0, Ramp::Linear { samples: 64 });
    let mut prev = g.current();
    for _ in 0..64 {
        let v = g.next_sample();
        assert!(v + 1e-6 >= prev, "linear ramp never steps backward");
        assert!(v <= 1.0 + 1e-6, "linear ramp never overshoots the target");
        prev = v;
    }
    assert!(g.is_settled(), "ramp settles at its length");
    assert!(fabs(g.current() - 1.0) < 1e-6, "settles exactly on target");
}

#[test]
fn voice_pool_steals_lowest_priority_when_oversubscribed() {
    let mut pool = VoicePool::new(16, 2);
    let group = VoiceGroup(0);
    let make = |priority: u8| VoiceRequest {
        group,
        priority,
        importance: priority as f32,
        behavior: VirtualBehavior::ContinueVirtual,
    };

    let low = pool.allocate(make(10)).expect("first voice fits physically");
    let mid = pool.allocate(make(20)).expect("second voice fits physically");
    assert_eq!(pool.physical_count(), 2);

    // A higher-priority request with no free physical slot steals the lowest.
    let high = pool.allocate(make(30)).expect("high-priority voice allocates");
    assert_eq!(pool.physical_count(), 2, "physical budget is respected");
    assert!(pool.is_physical(high), "winner is physical");
    assert!(pool.is_physical(mid), "mid priority survives");
    assert!(pool.is_virtual(low), "lowest priority is virtualized, not killed");
    assert_eq!(
        pool.get(low).expect("virtual voice still tracked").state,
        VoiceState::Virtual
    );
}

#[test]
fn voice_pool_group_limit_caps_concurrency() {
    let mut pool = VoicePool::new(16, 16);
    let group = VoiceGroup(7);
    pool.set_group_limit(group, 2, prism_audio_core::voice::LimitPolicy::RejectNewest);
    let req = |priority: u8| VoiceRequest {
        group,
        priority,
        importance: priority as f32,
        behavior: VirtualBehavior::Kill,
    };
    let _a = pool.allocate(req(5));
    let _b = pool.allocate(req(6));
    let _c = pool.allocate(req(7));
    assert!(
        pool.group_count(group) <= 2,
        "group limit caps simultaneous members"
    );
}

#[test]
fn event_scheduler_fires_events_at_sample_accurate_offsets() {
    let mut sched: EventScheduler<u32> = EventScheduler::with_capacity(8);
    sched.schedule(130, 1).expect("schedule within capacity");
    sched.schedule(10, 2).expect("schedule within capacity");
    sched.schedule(300, 3).expect("schedule within capacity");

    // First block [0,128): only the event at sample 10 is due, at offset 10.
    let mut fired: Vec<(usize, u32)> = Vec::new();
    sched.drain_due(0, 128, |offset, payload| fired.push((offset, payload)));
    assert_eq!(fired, vec![(10, 2)]);

    // Second block [128,256): the event at sample 130 fires at offset 2.
    fired.clear();
    sched.drain_due(128, 128, |offset, payload| fired.push((offset, payload)));
    assert_eq!(fired, vec![(2, 1)]);

    // Third block [256,384): the event at 300 fires at offset 44.
    fired.clear();
    sched.drain_due(256, 128, |offset, payload| fired.push((offset, payload)));
    assert_eq!(fired, vec![(44, 3)]);
    assert!(sched.is_empty(), "all events consumed");
}

#[test]
fn named_clock_quantizes_to_beat_and_bar_boundaries() {
    // 120 BPM, 4/4 at 48 kHz: one beat = 24000 samples, one bar = 96000.
    let clock = NamedClock::new(SR, 0, 120.0, TimeSignature::default());
    let spb = clock.samples_per_beat();
    let spbar = clock.samples_per_bar();
    assert!(fabs(spb as f32 - 24_000.0) < 1.0);
    assert!(fabs(spbar as f32 - 96_000.0) < 1.0);

    // A target just past beat 1 snaps up to beat 2 (sample 48000).
    let q_beat = clock.quantize(24_001, Grid::Beat);
    assert_eq!(q_beat, 48_000);
    // A target just past bar 0 snaps up to bar 1 (sample 96000).
    let q_bar = clock.quantize(1, Grid::Bar);
    assert_eq!(q_bar, 96_000);
    // Immediate grid is a pass-through.
    assert_eq!(clock.quantize(12_345, Grid::Immediate), 12_345);
}

#[test]
fn transport_tracks_musical_position_as_it_advances() {
    let mut transport = Transport::new(SR);
    assert_eq!(transport.playhead(), 0);
    transport.set_tempo_bpm(120.0);
    let spb = transport.samples_per_beat();
    assert!(fabs(spb as f32 - 24_000.0) < 1.0);
    transport.advance(24_000);
    assert_eq!(transport.playhead(), 24_000, "advance moves the playhead");
    transport.advance(1_000);
    assert_eq!(transport.playhead(), 25_000);
}
