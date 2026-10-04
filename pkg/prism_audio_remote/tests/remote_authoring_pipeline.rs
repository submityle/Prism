//! End-to-end integration coverage for the `prism_audio_remote` live-authoring
//! chain: authentication and allow-list gating, whitelisted command validation,
//! bidirectional transport round-trips, session capability negotiation,
//! read-only telemetry mirroring, smoothed live tuning driven by real
//! `prism_audio_core` ramp machinery, and authoring-data hot-reload
//! classification. Each test drives the crate's real public API across module
//! boundaries so a remote edit flows exactly as it would from an external tool.
//!
//! # Provenance
//! Original work. Contains no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, Dolby, MPEG, Google Resonance Audio, Web Audio, or Project Acoustics
//! source or derived code, and no AI/ML. Public, standardized ideas (bounded
//! queues, conservative capability intersection, constant-time secret compare)
//! are reused only as concepts, not as code.
//!
//! # Relationship
//! Exercises the public contracts of `prism_audio_remote` (the `auth`,
//! `command`, `session`, `telemetry`, `transport`, `tuning`, and `hotreload`
//! modules) together with the `Smoothed` and `Ramp` parameter types re-exported
//! through `prism_audio_core`. It never edits the crate under test; it only
//! consumes its published surface the way a live editor would.

use prism_audio_remote::auth::{AccessToken, AuthError, Authenticator, TOKEN_LEN};
use prism_audio_remote::command::{
    AuthoringCommand, BusId, CapabilitySet, CommandError, CommandKind, EventId, ParameterTarget,
    SnapshotId, StateGroupId, StateId, BUS_GAIN_DB_MAX, PARAMETER_MAX,
};
use prism_audio_remote::hotreload::{AssetId, AssetKind, ChangeKind, HotReloadEvent};
use prism_audio_remote::session::{Capabilities, RemoteSession, SessionState};
use prism_audio_remote::telemetry::{BusLevel, EventMarker, TelemetryMirror, TelemetrySnapshot};
use prism_audio_remote::transport::{AuthoringTransport, InProcessTransport, TransportError};
use prism_audio_remote::tuning::{LiveTuner, Smoothing, TuningTarget};

/// Absolute-value helper that avoids the standard-library float math path.
fn fabs(x: f32) -> f32 {
    if x < 0.0 {
        -x
    } else {
        x
    }
}

/// Tolerance for approximate float comparisons on smoothed glide values.
const EPS: f32 = 1.0e-4;

/// Returns `true` when `a` and `b` agree within [`EPS`].
fn close(a: f32, b: f32) -> bool {
    fabs(a - b) < EPS
}

/// Builds a deterministic 32-byte access token from a seed.
fn token(seed: u8) -> AccessToken {
    let mut bytes = [0u8; TOKEN_LEN];
    for (i, b) in bytes.iter_mut().enumerate() {
        *b = seed ^ (i as u8).wrapping_mul(31);
    }
    AccessToken::new(bytes)
}

/// Translates a validated authoring write into the matching live-tuning edit,
/// reproducing the engine-side command dispatch from the remote command set
/// onto the smoothed tuning channels. Returns the coarse kind that was applied.
fn apply_to_tuner(tuner: &mut LiveTuner, cmd: &AuthoringCommand, smoothing: Smoothing) -> CommandKind {
    match *cmd {
        AuthoringCommand::SetParameter { target, value } => {
            tuner.tune(TuningTarget::Rtpc(target.0), value, smoothing);
        }
        AuthoringCommand::SetBusGain { bus, gain_db } => {
            tuner.tune(TuningTarget::BusGain(bus.0), gain_db, smoothing);
        }
        AuthoringCommand::TriggerEvent { .. }
        | AuthoringCommand::SetState { .. }
        | AuthoringCommand::SwapSnapshot { .. } => {}
    }
    cmd.kind()
}

/// Derives the per-session write allow-list from negotiated session
/// capabilities, mirroring how the negotiated write flag gates the command
/// path.
fn caps_from_session(negotiated: &Capabilities) -> CapabilitySet {
    if negotiated.allow_write {
        CapabilitySet::full()
    } else {
        CapabilitySet::denied()
    }
}

#[test]
fn authorized_command_dispatches_through_transport_to_engine() {
    // Main chain: a tool authenticates, authorizes a whitelisted write, pushes
    // it onto the authoring link, and the engine endpoint receives it intact.
    let secret = token(7);
    let auth = Authenticator::with_gate(true, secret, CapabilitySet::full());

    let (mut tool, mut engine) =
        InProcessTransport::<AuthoringCommand, TelemetrySnapshot>::pair();

    let cmd = AuthoringCommand::SetParameter {
        target: ParameterTarget(11),
        value: 0.5,
    };

    // The command only reaches the wire after it clears the security boundary.
    assert_eq!(auth.authorize(&secret, &cmd), Ok(()));
    assert_eq!(tool.send(cmd), Ok(()));

    let received = engine.poll().expect("engine should receive the command");
    assert_eq!(received, cmd);
    assert_eq!(received.kind(), CommandKind::SetParameter);
    assert!(engine.poll().is_none());
}

#[test]
fn every_command_variant_survives_transport_round_trip() {
    // Round-trip consistency: each whitelisted variant must emerge byte-for-byte
    // equal on the engine side, in submission order.
    let (mut tool, mut engine) = InProcessTransport::<AuthoringCommand, u8>::pair();

    let commands = [
        AuthoringCommand::SetParameter {
            target: ParameterTarget(1),
            value: -0.25,
        },
        AuthoringCommand::TriggerEvent { event: EventId(2) },
        AuthoringCommand::SetState {
            group: StateGroupId(3),
            state: StateId(4),
        },
        AuthoringCommand::SwapSnapshot {
            snapshot: SnapshotId(5),
        },
        AuthoringCommand::SetBusGain {
            bus: BusId(6),
            gain_db: -6.0,
        },
    ];

    for cmd in commands.iter() {
        assert_eq!(tool.send(*cmd), Ok(()));
    }

    for cmd in commands.iter() {
        assert_eq!(engine.poll(), Some(*cmd));
    }
    assert!(engine.poll().is_none());
}

#[test]
fn denied_capability_blocks_command_before_the_wire() {
    // Error path: an under-privileged session is refused at authorization and
    // must never place traffic on the link.
    let secret = token(3);
    let caps = CapabilitySet {
        writes_enabled: true,
        trigger_event: true,
        ..CapabilitySet::denied()
    };
    let auth = Authenticator::with_gate(true, secret, caps);

    let (mut tool, mut engine) = InProcessTransport::<AuthoringCommand, u8>::pair();

    let blocked = AuthoringCommand::SwapSnapshot {
        snapshot: SnapshotId(99),
    };
    assert_eq!(
        auth.authorize(&secret, &blocked),
        Err(AuthError::Rejected(CommandError::NotPermitted(
            CommandKind::SwapSnapshot
        )))
    );

    // A conscientious tool does not send what it could not authorize.
    if auth.authorize(&secret, &blocked).is_ok() {
        let _ = tool.send(blocked);
    }
    assert_eq!(engine.poll(), None);
}

#[test]
fn wrong_token_fails_closed_and_disabled_build_refuses_all() {
    // Error path on the authentication stage, independent of the allow-list.
    let auth = Authenticator::with_gate(true, token(1), CapabilitySet::full());
    assert_eq!(auth.authenticate(&token(2)), Err(AuthError::InvalidToken));
    assert_eq!(auth.authenticate(&token(1)), Ok(()));

    let release_like = Authenticator::with_gate(false, token(1), CapabilitySet::full());
    let cmd = AuthoringCommand::TriggerEvent { event: EventId(1) };
    assert_eq!(
        release_like.authorize(&token(1), &cmd),
        Err(AuthError::RemoteDisabled)
    );
    assert!(!release_like.is_enabled());
}

#[test]
fn session_negotiation_gates_the_write_capability() {
    // State/boundary chain: the conservative capability intersection decides
    // whether writes are possible at all, which then drives the allow-list.
    let tool_caps = Capabilities {
        max_bandwidth_bytes_per_sec: 1_000_000,
        sample_rate: 48_000,
        telemetry_hz: 60,
        allow_write: true,
    };
    let device_caps = Capabilities {
        max_bandwidth_bytes_per_sec: 256_000,
        sample_rate: 44_100,
        telemetry_hz: 30,
        allow_write: false,
    };

    let agreed = tool_caps.negotiate(&device_caps);
    assert_eq!(agreed.max_bandwidth_bytes_per_sec, 256_000);
    assert_eq!(agreed.sample_rate, 44_100);
    assert_eq!(agreed.telemetry_hz, 30);
    assert!(!agreed.allow_write);

    // A read-only negotiation yields a deny-all allow-list.
    let read_only = caps_from_session(&agreed);
    let write = AuthoringCommand::SetBusGain {
        bus: BusId(1),
        gain_db: 0.0,
    };
    assert_eq!(write.validate(&read_only), Err(CommandError::WritesDisabled));

    // When both ends permit writes, the negotiated allow-list admits them.
    let both_write = tool_caps.negotiate(&tool_caps);
    assert!(both_write.allow_write);
    assert_eq!(write.validate(&caps_from_session(&both_write)), Ok(()));
}

#[test]
fn full_session_lifecycle_runs_authorized_tuning_pipeline() {
    // The complete chain: handshake, negotiate, authorize two writes, dispatch
    // them over the transport, apply them to the smoothed tuner, publish
    // telemetry back, and mirror it on the tool side.
    let mut session = RemoteSession::new();
    assert_eq!(session.state(), SessionState::Disconnected);
    session.begin_handshake().expect("handshake should start");

    let tool_caps = Capabilities {
        max_bandwidth_bytes_per_sec: 512_000,
        sample_rate: 48_000,
        telemetry_hz: 60,
        allow_write: true,
    };
    let device_caps = Capabilities {
        max_bandwidth_bytes_per_sec: 384_000,
        sample_rate: 48_000,
        telemetry_hz: 30,
        allow_write: true,
    };
    let agreed = tool_caps.negotiate(&device_caps);
    session.complete_handshake(agreed).expect("handshake completes");
    assert!(session.is_connected());

    let secret = token(21);
    let auth = Authenticator::with_gate(true, secret, caps_from_session(&agreed));

    let (mut tool, mut engine) =
        InProcessTransport::<AuthoringCommand, TelemetrySnapshot>::pair();

    let writes = [
        AuthoringCommand::SetParameter {
            target: ParameterTarget(1),
            value: 0.8,
        },
        AuthoringCommand::SetBusGain {
            bus: BusId(2),
            gain_db: -3.0,
        },
    ];
    for cmd in writes.iter() {
        assert_eq!(auth.authorize(&secret, cmd), Ok(()));
        assert_eq!(tool.send(*cmd), Ok(()));
    }

    // Engine side: drain commands, apply to the real smoothed tuner.
    let mut tuner = LiveTuner::new(agreed.sample_rate);
    while let Some(cmd) = engine.poll() {
        apply_to_tuner(&mut tuner, &cmd, Smoothing::LinearSeconds(0.01));
    }
    // First edits seed settled, so the audible values match the requests.
    assert!(tuner.is_settled());
    assert!(close(tuner.current(TuningTarget::Rtpc(1)).unwrap(), 0.8));
    assert!(close(tuner.current(TuningTarget::BusGain(2)).unwrap(), -3.0));

    // Engine publishes one telemetry snapshot; the tool mirrors it read-only.
    let snapshot = TelemetrySnapshot {
        active_voices: 3,
        cpu_load: 0.4,
        bus_levels: alloc_bus_levels(),
        event_timeline: alloc_event_timeline(),
        ..TelemetrySnapshot::empty()
    };
    assert_eq!(engine.send(snapshot.clone()), Ok(()));

    let mut mirror = TelemetryMirror::new();
    let decoded = tool.poll().expect("tool should receive telemetry");
    mirror.update(decoded);
    assert_eq!(mirror.generation(), 1);
    assert_eq!(mirror.snapshot().generation, 1);
    assert_eq!(mirror.snapshot().active_voices, 3);
    assert!(close(mirror.snapshot().cpu_load, 0.4));
    assert_eq!(mirror.snapshot().bus_levels, snapshot.bus_levels);
    assert_eq!(mirror.snapshot().event_timeline, snapshot.event_timeline);

    session.close().expect("session closes");
    assert_eq!(session.state(), SessionState::Closed);
}

/// Builds a small, deterministic set of bus level readings for telemetry tests.
fn alloc_bus_levels() -> Vec<BusLevel> {
    vec![
        BusLevel {
            bus: BusId(1),
            peak_db: -6.0,
            rms_db: -12.0,
        },
        BusLevel {
            bus: BusId(2),
            peak_db: -3.0,
            rms_db: -9.0,
        },
    ]
}

/// Builds a short, ordered event timeline for telemetry tests.
fn alloc_event_timeline() -> Vec<EventMarker> {
    vec![
        EventMarker {
            event: EventId(5),
            time_samples: 44_100,
        },
        EventMarker {
            event: EventId(6),
            time_samples: 48_000,
        },
    ]
}

#[test]
fn telemetry_mirror_generation_is_monotonic_and_authoritative() {
    // State-mirror invariant: every update bumps the generation and overwrites
    // any caller-supplied generation with the mirror's own authoritative value.
    let mut mirror = TelemetryMirror::new();
    assert_eq!(mirror.generation(), 0);

    let (mut tool, mut engine) = InProcessTransport::<u8, TelemetrySnapshot>::pair();

    for step in 0..4u32 {
        // The engine lies about the generation; the mirror must ignore it.
        let snap = TelemetrySnapshot {
            generation: 9_999,
            active_voices: step,
            cpu_load: 0.1,
            ..TelemetrySnapshot::empty()
        };
        engine.send(snap).expect("engine telemetry send");
        let decoded = tool.poll().expect("telemetry available");
        mirror.update(decoded);

        let expected_gen = u64::from(step) + 1;
        assert_eq!(mirror.generation(), expected_gen);
        assert_eq!(mirror.snapshot().generation, expected_gen);
        assert_eq!(mirror.snapshot().active_voices, step);
    }
}

#[test]
fn live_tuner_glide_is_bit_identical_across_runs() {
    // Determinism: two independent tuners fed the same edits produce the same
    // audible samples bit-for-bit at every tick.
    fn run() -> Vec<u32> {
        let mut tuner = LiveTuner::new(1_000);
        tuner.tune(TuningTarget::Rtpc(1), 0.0, Smoothing::Immediate);
        tuner.tune(TuningTarget::Rtpc(1), 1.0, Smoothing::ExponentialSeconds(0.01));
        tuner.tune(TuningTarget::BusGain(2), -12.0, Smoothing::Immediate);
        tuner.tune(TuningTarget::BusGain(2), -3.0, Smoothing::LinearSeconds(0.02));

        let mut trace = Vec::new();
        for _ in 0..64 {
            tuner.advance(1);
            let a = tuner.current(TuningTarget::Rtpc(1)).unwrap();
            let b = tuner.current(TuningTarget::BusGain(2)).unwrap();
            trace.push(a.to_bits());
            trace.push(b.to_bits());
        }
        trace
    }

    let first = run();
    let second = run();
    assert_eq!(first, second);
    assert!(!first.is_empty());
}

#[test]
fn live_tuner_pending_writeback_dedupes_and_sorts() {
    // State-mirror invariant for the authoring write-back set: one entry per
    // target holding the final requested value, drained in target order.
    let mut tuner = LiveTuner::new(48_000);
    tuner.tune(TuningTarget::BusGain(2), -6.0, Smoothing::Immediate);
    tuner.tune(TuningTarget::Rtpc(1), 0.1, Smoothing::Immediate);
    tuner.tune(TuningTarget::Rtpc(1), 0.9, Smoothing::Immediate);
    tuner.tune(TuningTarget::Attenuation(3), 0.25, Smoothing::Immediate);
    assert_eq!(tuner.pending_len(), 3);

    let drained = tuner.take_pending();
    assert_eq!(drained.len(), 3);

    // Sorted by target: Rtpc < BusGain < Attenuation per the enum discriminant.
    assert_eq!(drained[0].0, TuningTarget::Rtpc(1));
    assert_eq!(drained[1].0, TuningTarget::BusGain(2));
    assert_eq!(drained[2].0, TuningTarget::Attenuation(3));

    // The deduped Rtpc entry keeps the latest requested value, not the first.
    assert!(close(drained[0].1, 0.9));
    assert_eq!(tuner.pending_len(), 0);
}

#[test]
fn out_of_range_and_non_finite_writes_are_refused() {
    // Boundary path: the value-range gate rejects both saturated and
    // non-finite writes while admitting the inclusive edges.
    let caps = CapabilitySet::full();

    let too_hot = AuthoringCommand::SetParameter {
        target: ParameterTarget(1),
        value: PARAMETER_MAX * 2.0,
    };
    assert_eq!(too_hot.validate(&caps), Err(CommandError::OutOfRange));

    let edge = AuthoringCommand::SetBusGain {
        bus: BusId(1),
        gain_db: BUS_GAIN_DB_MAX,
    };
    assert_eq!(edge.validate(&caps), Ok(()));

    let not_finite = AuthoringCommand::SetBusGain {
        bus: BusId(1),
        gain_db: f32::INFINITY,
    };
    assert_eq!(not_finite.validate(&caps), Err(CommandError::NonFinite));
}

#[test]
fn transport_capacity_backpressure_and_disconnect_are_observable() {
    // Boundary and error paths of the link itself: a saturated queue reports
    // `Full`, and a dropped peer reports `Disconnected`.
    let (mut tool, engine) = InProcessTransport::<AuthoringCommand, u8>::pair_with_capacity(2);
    let fill = AuthoringCommand::TriggerEvent { event: EventId(1) };
    assert_eq!(tool.send(fill), Ok(()));
    assert_eq!(tool.send(fill), Ok(()));
    assert_eq!(tool.send(fill), Err(TransportError::Full));

    assert!(tool.is_connected());
    drop(engine);
    assert!(!tool.is_connected());
    assert_eq!(tool.send(fill), Err(TransportError::Disconnected));
}

#[test]
fn hot_reload_events_order_and_classify_plan_rebuilds() {
    // Authoring hot-reload chain: revisions stay strictly increasing for
    // ordering/dedup, and only graph-structural or removal changes demand a
    // recompiled execution plan.
    let events = [
        HotReloadEvent::new(AssetId(10), AssetKind::Event, ChangeKind::Modified, 1),
        HotReloadEvent::new(AssetId(11), AssetKind::Patch, ChangeKind::Modified, 2),
        HotReloadEvent::new(AssetId(12), AssetKind::Container, ChangeKind::Added, 3),
        HotReloadEvent::new(AssetId(13), AssetKind::Event, ChangeKind::Removed, 4),
        HotReloadEvent::new(AssetId(14), AssetKind::Bank, ChangeKind::Modified, 5),
    ];
    let expected_rebuild = [false, true, true, true, true];

    let mut previous: Option<u64> = None;
    for (ev, want) in events.iter().zip(expected_rebuild.iter()) {
        assert_eq!(ev.requires_exec_plan_rebuild(), *want);
        if let Some(prev) = previous {
            assert!(ev.revision > prev);
        }
        previous = Some(ev.revision);
    }

    // Structural classification is independent of the change kind.
    assert!(AssetKind::Container.is_graph_structural());
    assert!(AssetKind::Bank.is_graph_structural());
    assert!(AssetKind::Patch.is_graph_structural());
    assert!(!AssetKind::Event.is_graph_structural());
}
