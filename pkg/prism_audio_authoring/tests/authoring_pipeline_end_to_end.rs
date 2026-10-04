//! End-to-end integration tests driving the real public main chains of the
//! `prism_audio_authoring` crate: the section 11 Patch procedural graph
//! (`PatchBuilder` to `compile` to `CompiledPatch`), its host-graph adapter
//! (`PatchNode` as a `prism_audio_core::graph::AudioNode`), the section 12
//! modulation matrix (`ModMatrix`), the section 46.8 auto-mix ducking runtime
//! (`AutoMixRuleset` to `CompiledAutoMix`), and the section 42 quality-paged
//! Patch (`PagedPatch` with `PageSelector`). Each test exercises one concern of
//! the authoring pipeline through its public surface only, asserting real
//! rendered output, deterministic bit-identical behaviour, documented
//! invariants, and the structural error paths.
//!
//! # Provenance
//! Original work. No Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, MPEG, Google Resonance Audio, Web Audio, or Project Acoustics source
//! or derived code is present, and no AI or ML techniques are used. Only ideas
//! from public, openly published standards informed the design; no third-party
//! source was copied or adapted.
//!
//! # Relationship
//! Validates the public contracts of design section 11 (Patch graph and its
//! deterministic compiler), section 12 (modulation buses, curves, LFO sources,
//! and the modulation matrix), section 42 (quality-paged tiered compilation),
//! and section 46.8 (auto-mix ducking). Depends on `prism_audio_core` for the
//! runtime graph primitives (`AudioBuffer`, `ChannelLayout`, `AudioNode`,
//! `ProcessIo`, `RenderContext`, `GraphError`) and the deterministic
//! `LfoWaveform`, all of which Cargo exposes to this same-package integration
//! test as ordinary dependencies.

use prism_audio_authoring::automix::{decibels_to_linear, AutoMixRuleset, DuckingRule};
use prism_audio_authoring::modulation::{
    BusId, Curve, EnvelopeConfig, LfoModulator, ModContext, ModMatrix, ModMix, ModRoute, Polarity,
    RouteInput,
};
use prism_audio_authoring::pages::{
    PageDecision, PageSelector, PagedPatch, PatchPage, QualityLevel,
};
use prism_audio_authoring::patch::{
    compile, CompiledPatch, NodeKind, OscWaveform, PatchBuilder, PatchDescription, PatchError,
};
use prism_audio_core::graph::{AudioNode, GraphError, ProcessIo, RenderContext};
use prism_audio_core::modulation::LfoWaveform;
use prism_audio_core::{AudioBuffer, ChannelLayout};

/// Render sample rate used across the suite.
const SR: u32 = 48_000;
/// Block size used across the suite.
const BLOCK: usize = 256;
/// Loose tolerance for approximate float comparisons.
const EPS: f32 = 1.0e-3;

/// Branch-based absolute value avoiding the forbidden `f32::abs` intrinsic.
fn fabs(x: f32) -> f32 {
    if x < 0.0 {
        -x
    } else {
        x
    }
}

/// Returns `true` when `a` and `b` are within `eps` of one another.
fn approx(a: f32, b: f32, eps: f32) -> bool {
    fabs(a - b) < eps
}

/// Asserts that `compile` rejected the description with exactly `expected`.
///
/// A dedicated helper is required because `CompiledPatch` implements neither
/// `Debug` nor `PartialEq`, so `assert_eq!` on the whole `Result` is impossible;
/// only the `PatchError` arm is compared.
fn expect_err(result: Result<CompiledPatch, PatchError>, expected: &PatchError) {
    match result {
        Ok(_) => panic!("expected compile error {expected:?}, got a successful compile"),
        Err(ref actual) => assert_eq!(actual, expected),
    }
}

/// Peak absolute magnitude of a sample slice, without std float math.
fn peak(samples: &[f32]) -> f32 {
    let mut m = 0.0_f32;
    for s in samples {
        let a = fabs(*s);
        if a > m {
            m = a;
        }
    }
    m
}

/// Builds a single-`Constant` patch wired straight to the master output.
fn constant_patch(value: f32) -> PatchDescription {
    let mut b = PatchBuilder::new();
    let c = b.add_node(NodeKind::Constant, "dc");
    b.set_param(c, "value", value);
    b.add_output(c, 0);
    b.build()
}

/// Wraps `inner` inside one more `SubPatch` nesting layer.
fn wrap_subpatch(inner: PatchDescription) -> PatchDescription {
    let mut b = PatchBuilder::new();
    let n = b.add_node(NodeKind::SubPatch(Box::new(inner)), "nested");
    b.add_output(n, 0);
    b.build()
}

#[test]
fn end_to_end_patch_compiles_and_renders_known_dc() {
    // The simplest full main chain: build -> validate -> flatten -> graph.
    let desc = constant_patch(0.5);
    let mut compiled = compile(&desc, SR, BLOCK).expect("constant patch compiles");
    assert_eq!(compiled.output_count(), 1);

    let mut out = AudioBuffer::new(ChannelLayout::Mono, BLOCK);
    compiled.process(BLOCK, 0, &mut out);

    // A smoothed constant whose target equals its start emits an exact DC level,
    // so every rendered sample is bit-identical to the requested value.
    let want = 0.5_f32.to_bits();
    for s in out.channel(0) {
        assert_eq!(s.to_bits(), want);
    }
}

#[test]
fn signal_chain_oscillator_gain_lowpass_produces_bounded_audio() {
    // osc -> gain -> one-pole lowpass -> master out, a representative DSP chain.
    let mut b = PatchBuilder::new();
    let osc = b.add_node(
        NodeKind::Oscillator {
            waveform: OscWaveform::Sine,
        },
        "osc",
    );
    b.set_param(osc, "frequency", 220.0);
    b.set_param(osc, "amplitude", 0.8);
    let gain = b.add_node(NodeKind::Gain, "gain");
    b.set_param(gain, "gain", 0.5);
    let lpf = b.add_node(NodeKind::OnePoleLowpass, "lpf");
    b.set_param(lpf, "cutoff_hz", 2_000.0);
    b.connect(osc, 0, gain, 0);
    b.connect(gain, 0, lpf, 0);
    b.add_output(lpf, 0);
    let desc = b.build();

    let mut compiled = compile(&desc, SR, BLOCK).expect("chain compiles");
    assert_eq!(compiled.output_count(), 1);

    let mut out = AudioBuffer::new(ChannelLayout::Mono, BLOCK);
    // Render several blocks so smoothing and the filter settle into motion.
    let mut playhead = 0_u64;
    let mut observed = 0.0_f32;
    for _ in 0..8 {
        compiled.process(BLOCK, playhead, &mut out);
        let p = peak(out.channel(0));
        if p > observed {
            observed = p;
        }
        playhead += BLOCK as u64;
    }

    // Signal is audible yet bounded by the amplitude * gain budget (0.8 * 0.5).
    assert!(observed > 0.02, "expected audible output, got {observed}");
    assert!(observed <= 0.4 + EPS, "exceeded gain budget, got {observed}");
}

#[test]
fn compilation_is_bit_identical_across_two_runs() {
    // Two independent compilations of the same description must render the exact
    // same bits sample for sample: the compiler and runtime are deterministic.
    let mut b = PatchBuilder::new();
    let osc = b.add_node(
        NodeKind::Oscillator {
            waveform: OscWaveform::Sawtooth,
        },
        "osc",
    );
    b.set_param(osc, "frequency", 110.0);
    b.set_param(osc, "amplitude", 0.6);
    let gain = b.add_node(NodeKind::Gain, "gain");
    b.set_param(gain, "gain", 0.75);
    b.connect(osc, 0, gain, 0);
    b.add_output(gain, 0);
    let desc = b.build();

    let mut a = compile(&desc, SR, BLOCK).expect("compile a");
    let mut c = compile(&desc, SR, BLOCK).expect("compile c");

    let mut out_a = AudioBuffer::new(ChannelLayout::Mono, BLOCK);
    let mut out_c = AudioBuffer::new(ChannelLayout::Mono, BLOCK);
    let mut playhead = 0_u64;
    for _ in 0..8 {
        a.process(BLOCK, playhead, &mut out_a);
        c.process(BLOCK, playhead, &mut out_c);
        for (sa, sc) in out_a.channel(0).iter().zip(out_c.channel(0).iter()) {
            assert_eq!(sa.to_bits(), sc.to_bits());
        }
        playhead += BLOCK as u64;
    }
}

#[test]
fn exposed_parameter_drives_output_value() {
    // A host-exposed parameter handle must steer the live rendered value.
    let mut b = PatchBuilder::new();
    let c = b.add_node(NodeKind::Constant, "dc");
    b.expose_param("level", c, "value");
    b.add_output(c, 0);
    let desc = b.build();

    let mut compiled = compile(&desc, SR, BLOCK).expect("compiles");
    let level = compiled.param("level").expect("exposed param present").clone();
    assert_eq!(level.name(), "level");
    level.set(0.75);

    let mut out = AudioBuffer::new(ChannelLayout::Mono, BLOCK);
    compiled.process(BLOCK, 0, &mut out);

    // A linear ramp over the block lands exactly on the target at the final
    // sample, so the last rendered value is bit-identical to the request.
    let last = *out.channel(0).last().expect("non-empty block");
    assert_eq!(last.to_bits(), 0.75_f32.to_bits());
}

#[test]
fn gate_trigger_opens_adsr_envelope() {
    // Constant(1.0) -> AdsrAmp -> out, with the gate exposed to the host.
    let mut b = PatchBuilder::new();
    let c = b.add_node(NodeKind::Constant, "dc");
    b.set_param(c, "value", 1.0);
    let env = b.add_node(
        NodeKind::AdsrAmp {
            config: EnvelopeConfig::default(),
        },
        "env",
    );
    b.connect(c, 0, env, 0);
    b.expose_trigger("gate", env);
    b.add_output(env, 0);
    let desc = b.build();

    let mut compiled = compile(&desc, SR, BLOCK).expect("compiles");
    let gate = compiled.trigger("gate").expect("exposed gate present").clone();
    assert_eq!(gate.name(), "gate");

    // With the gate shut the envelope is idle: the output is exact silence.
    let mut out = AudioBuffer::new(ChannelLayout::Mono, BLOCK);
    compiled.process(BLOCK, 0, &mut out);
    assert_eq!(peak(out.channel(0)).to_bits(), 0.0_f32.to_bits());

    // Opening the gate begins the attack; after a few blocks there is signal.
    gate.set_gate(true);
    let mut playhead = BLOCK as u64;
    let mut observed = 0.0_f32;
    for _ in 0..8 {
        compiled.process(BLOCK, playhead, &mut out);
        let p = peak(out.channel(0));
        if p > observed {
            observed = p;
        }
        playhead += BLOCK as u64;
    }
    assert!(observed > 0.01, "gate open should produce signal, got {observed}");
}

#[test]
fn nested_subpatch_flattens_and_renders() {
    // A SubPatch(Constant 0.25) feeding a Gain exercises compile-time flattening.
    let inner = constant_patch(0.25);
    let mut b = PatchBuilder::new();
    let sub = b.add_node(NodeKind::SubPatch(Box::new(inner)), "inner");
    let gain = b.add_node(NodeKind::Gain, "gain");
    b.set_param(gain, "gain", 1.0);
    b.connect(sub, 0, gain, 0);
    b.add_output(gain, 0);
    let desc = b.build();

    let mut compiled = compile(&desc, SR, BLOCK).expect("nested patch compiles");
    let mut out = AudioBuffer::new(ChannelLayout::Mono, BLOCK);
    // Second block so both smoothed stages are fully settled.
    compiled.process(BLOCK, 0, &mut out);
    compiled.process(BLOCK, BLOCK as u64, &mut out);

    let last = *out.channel(0).last().expect("non-empty block");
    assert!(approx(last, 0.25, EPS), "nested DC mismatch: {last}");
}

#[test]
fn arithmetic_primitives_hold_invariants() {
    // Product multiplies its two inputs: 0.5 * 0.4 == 0.2.
    let mut pb = PatchBuilder::new();
    let a = pb.add_node(NodeKind::Constant, "a");
    pb.set_param(a, "value", 0.5);
    let bb = pb.add_node(NodeKind::Constant, "b");
    pb.set_param(bb, "value", 0.4);
    let prod = pb.add_node(NodeKind::Product, "prod");
    pb.connect(a, 0, prod, 0);
    pb.connect(bb, 0, prod, 1);
    pb.add_output(prod, 0);
    let pdesc = pb.build();
    let mut pcompiled = compile(&pdesc, SR, BLOCK).expect("product compiles");
    let mut pout = AudioBuffer::new(ChannelLayout::Mono, BLOCK);
    pcompiled.process(BLOCK, 0, &mut pout);
    pcompiled.process(BLOCK, BLOCK as u64, &mut pout);
    let plast = *pout.channel(0).last().expect("non-empty");
    assert!(approx(plast, 0.2, EPS), "product mismatch: {plast}");

    // Sum adds its three inputs: 0.1 + 0.2 + 0.3 == 0.6.
    let mut sb = PatchBuilder::new();
    let s0 = sb.add_node(NodeKind::Constant, "s0");
    sb.set_param(s0, "value", 0.1);
    let s1 = sb.add_node(NodeKind::Constant, "s1");
    sb.set_param(s1, "value", 0.2);
    let s2 = sb.add_node(NodeKind::Constant, "s2");
    sb.set_param(s2, "value", 0.3);
    let sum = sb.add_node(NodeKind::Sum { inputs: 3 }, "sum");
    sb.connect(s0, 0, sum, 0);
    sb.connect(s1, 0, sum, 1);
    sb.connect(s2, 0, sum, 2);
    sb.add_output(sum, 0);
    let sdesc = sb.build();
    let mut scompiled = compile(&sdesc, SR, BLOCK).expect("sum compiles");
    let mut sout = AudioBuffer::new(ChannelLayout::Mono, BLOCK);
    scompiled.process(BLOCK, 0, &mut sout);
    scompiled.process(BLOCK, BLOCK as u64, &mut sout);
    let slast = *sout.channel(0).last().expect("non-empty");
    assert!(approx(slast, 0.6, EPS), "sum mismatch: {slast}");
}

#[test]
fn compiled_patch_runs_as_audio_node_in_host_graph() {
    // into_node() yields a PatchNode usable as an AudioNode inside a host graph.
    let mut b = PatchBuilder::new();
    let osc = b.add_node(
        NodeKind::Oscillator {
            waveform: OscWaveform::Triangle,
        },
        "osc",
    );
    b.set_param(osc, "frequency", 330.0);
    b.set_param(osc, "amplitude", 0.9);
    let gain = b.add_node(NodeKind::Gain, "gain");
    b.set_param(gain, "gain", 0.5);
    b.connect(osc, 0, gain, 0);
    b.add_output(gain, 0);
    let desc = b.build();

    let compiled = compile(&desc, SR, BLOCK).expect("compiles");
    let mut node = compiled.into_node();

    // Host contract: zero inputs, one mono output with active frames preset.
    let mut outputs = [AudioBuffer::new(ChannelLayout::Mono, BLOCK)];
    outputs[0].set_active_frames(BLOCK);
    let inputs: [AudioBuffer; 0] = [];
    let mut observed = 0.0_f32;
    let mut playhead = 0_u64;
    for _ in 0..8 {
        let ctx = RenderContext {
            sample_rate: SR,
            frames: BLOCK,
            playhead,
        };
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx, &mut io);
        let p = peak(outputs[0].channel(0));
        if p > observed {
            observed = p;
        }
        playhead += BLOCK as u64;
    }
    assert!(observed > 0.02, "host-graph render was silent: {observed}");
}

#[test]
fn validation_rejects_malformed_descriptions() {
    // NoOutput: a node with no declared audio output.
    let mut b = PatchBuilder::new();
    b.add_node(NodeKind::Constant, "dc");
    expect_err(compile(&b.build(), SR, BLOCK), &PatchError::NoOutput);

    // UnknownNode: a connection targeting an index outside `nodes`.
    let mut b = PatchBuilder::new();
    let g = b.add_node(NodeKind::Gain, "g");
    b.connect(g, 0, 99, 0);
    b.add_output(g, 0);
    expect_err(compile(&b.build(), SR, BLOCK), &PatchError::UnknownNode(99));

    // ConnectionIntoSource: wiring into a pure source (Constant has no inputs).
    let mut b = PatchBuilder::new();
    let src = b.add_node(NodeKind::Constant, "src");
    let sink = b.add_node(NodeKind::Constant, "sink");
    b.connect(src, 0, sink, 0);
    b.add_output(src, 0);
    expect_err(
        compile(&b.build(), SR, BLOCK),
        &PatchError::ConnectionIntoSource { node: sink },
    );

    // PortOutOfRange: a destination input port beyond the node's input count.
    let mut b = PatchBuilder::new();
    let src = b.add_node(NodeKind::Constant, "src");
    let g = b.add_node(NodeKind::Gain, "g");
    b.connect(src, 0, g, 3);
    b.add_output(g, 0);
    expect_err(
        compile(&b.build(), SR, BLOCK),
        &PatchError::PortOutOfRange { node: g, port: 3 },
    );

    // UnknownParam: exposing a parameter the node does not have.
    let mut b = PatchBuilder::new();
    let g = b.add_node(NodeKind::Gain, "g");
    b.expose_param("x", g, "nonexistent");
    b.add_output(g, 0);
    assert!(matches!(
        compile(&b.build(), SR, BLOCK),
        Err(PatchError::UnknownParam { .. })
    ));

    // UnknownTrigger: exposing a trigger on a node without a gate input.
    let mut b = PatchBuilder::new();
    let g = b.add_node(NodeKind::Gain, "g");
    b.expose_trigger("t", g);
    b.add_output(g, 0);
    expect_err(
        compile(&b.build(), SR, BLOCK),
        &PatchError::UnknownTrigger { node: g },
    );

    // NestingTooDeep: five SubPatch layers exceed MAX_NESTING_DEPTH (4).
    let mut deep = constant_patch(0.1);
    for _ in 0..5 {
        deep = wrap_subpatch(deep);
    }
    expect_err(
        compile(&deep, SR, BLOCK),
        &PatchError::NestingTooDeep { max: 4 },
    );
}

#[test]
fn cycle_is_rejected_by_core_compiler() {
    // A feedback loop between two gains passes structural validation but is
    // rejected by the prism_audio_core graph compiler as a cycle.
    let mut b = PatchBuilder::new();
    let a = b.add_node(NodeKind::Gain, "a");
    let c = b.add_node(NodeKind::Gain, "b");
    b.connect(a, 0, c, 0);
    b.connect(c, 0, a, 0);
    b.add_output(a, 0);
    let desc = b.build();

    expect_err(
        compile(&desc, SR, BLOCK),
        &PatchError::GraphBuild(GraphError::Cycle),
    );
}

#[test]
fn automix_ducking_recovers_and_ducks_to_floor() {
    // Dialogue ducks music by 12 dB; the music category tracks the documented
    // idle-unity and active-floor contract.
    let mut set = AutoMixRuleset::new();
    let dialogue = set.add_category("dialogue", 100);
    let music = set.add_category("music", 10);
    set.add_rule(
        DuckingRule::new(dialogue, music, 12.0)
            .with_attack(0.01)
            .with_release(0.1),
    );
    let mut mix = set.compile(SR).expect("ruleset compiles");
    assert_eq!(mix.category_count(), 2);

    // Idle: music recovers to unity gain and zero reduction.
    for _ in 0..50 {
        mix.advance(BLOCK as u32);
    }
    assert!(approx(mix.gain(music), 1.0, EPS));
    assert!(approx(mix.reduction(music), 0.0, EPS));

    // Active dialogue drives music to its 12 dB floor; dialogue is never ducked.
    mix.set_active(dialogue, true);
    for _ in 0..200 {
        mix.advance(BLOCK as u32);
    }
    let floor = decibels_to_linear(-12.0);
    let ducked = mix.gain(music);
    assert!(approx(ducked, floor, EPS), "floor mismatch: {ducked} vs {floor}");
    assert!(approx(mix.gain(dialogue), 1.0, EPS));
    let reduction = mix.reduction(music);
    assert!((0.0..1.0).contains(&reduction), "reduction out of range: {reduction}");

    // Release recovers back toward unity once the trigger stops.
    mix.set_active(dialogue, false);
    for _ in 0..200 {
        mix.advance(BLOCK as u32);
    }
    let recovered = mix.gain(music);
    assert!(recovered > ducked, "release did not recover: {recovered}");
    assert!(approx(recovered, 1.0, EPS));
}

#[test]
fn automix_strongest_duck_wins() {
    // Two triggers target one category; the deeper (12 dB) duck must dominate
    // the shallower (6 dB) duck via the strongest-reduction fold.
    let mut set = AutoMixRuleset::new();
    let dialogue = set.add_category("dialogue", 100);
    let ui = set.add_category("ui", 80);
    let music = set.add_category("music", 10);
    set.add_rule(DuckingRule::new(dialogue, music, 12.0).with_attack(0.005));
    set.add_rule(DuckingRule::new(ui, music, 6.0).with_attack(0.005));
    let mut mix = set.compile(SR).expect("compiles");

    mix.set_active(dialogue, true);
    mix.set_active(ui, true);
    for _ in 0..200 {
        mix.advance(BLOCK as u32);
    }

    let deep = decibels_to_linear(-12.0);
    let shallow = decibels_to_linear(-6.0);
    let music_gain = mix.gain(music);
    assert!(approx(music_gain, deep, EPS), "expected 12 dB floor, got {music_gain}");
    assert!(music_gain < shallow, "deeper duck did not win: {music_gain}");
}

#[test]
fn pages_resolve_select_and_compile_resolved_page() {
    // A quality ladder: a cheap base page and a rich high page, each a real
    // compilable Patch description keyed by quality level.
    let base = PatchPage::new(QualityLevel::LOWEST, "base".to_string(), constant_patch(0.1));
    let high = PatchPage::new(QualityLevel::new(3), "high".to_string(), constant_patch(0.9));
    let paged = PagedPatch::new(vec![base, high]).expect("valid ladder");

    // Resolution picks the richest page not exceeding the requested level.
    assert_eq!(paged.resolve(QualityLevel::LOWEST).label(), "base");
    assert_eq!(paged.resolve(QualityLevel::new(5)).label(), "high");

    // Hysteresis: a differing request must persist before the switch commits.
    let mut selector = PageSelector::for_level(&paged, QualityLevel::LOWEST, 2, 0);
    let first = selector.observe(&paged, QualityLevel::new(3), 0);
    assert!(matches!(first, PageDecision::Deferred { .. }));
    let second = selector.observe(&paged, QualityLevel::new(3), 1);
    let to_index = match second {
        PageDecision::Switched { to_index, .. } => to_index,
        other => panic!("expected a committed switch, got {other:?}"),
    };

    // Compile the resolved page and confirm it renders that page's DC level.
    let resolved = &paged.pages()[to_index];
    assert_eq!(resolved.label(), "high");
    let mut compiled = compile(resolved.description(), SR, BLOCK).expect("page compiles");
    let mut out = AudioBuffer::new(ChannelLayout::Mono, BLOCK);
    compiled.process(BLOCK, 0, &mut out);
    compiled.process(BLOCK, BLOCK as u64, &mut out);
    let last = *out.channel(0).last().expect("non-empty");
    assert!(approx(last, 0.9, EPS), "resolved page DC mismatch: {last}");
}

#[test]
fn modulation_matrix_tick_is_bit_identical() {
    // Two identically configured modulation matrices must produce bit-identical
    // bus values on every tick: the matrix is fully deterministic.
    fn build() -> (ModMatrix, BusId) {
        let mut m = ModMatrix::new();
        let bus = m.add_bus("depth", 0.0);
        let src = m.add_source(Box::new(LfoModulator::new(
            SR,
            5.0,
            LfoWaveform::Triangle,
            true,
        )));
        m.add_route(
            ModRoute::new(RouteInput::Source(src), bus, 1.0)
                .with_polarity(Polarity::Unipolar)
                .with_curve(Curve::Linear)
                .with_mix(ModMix::Add),
        );
        (m, bus)
    }

    let (mut a, bus_a) = build();
    let (mut c, bus_c) = build();
    let ctx = ModContext::new(SR, BLOCK as u32);
    for _ in 0..64 {
        a.tick(&ctx);
        c.tick(&ctx);
        assert_eq!(a.bus_value(bus_a).to_bits(), c.bus_value(bus_c).to_bits());
    }
}
