use super::*;
use prism_ecs::component::Component;
use prism_ecs::world::World;

#[derive(Clone, Hash)]
struct Value(u64);
impl Component for Value {}

fn history(capacity: usize, frames: u64) -> SnapshotHistory {
    let mut world = World::new();
    world.spawn(Value(0));
    world.register_snapshot_component_hashable::<Value>();
    let mut history = SnapshotHistory::new(capacity);
    for _ in 0..frames {
        history.capture(&world);
    }
    history
}

fn update(tick: u64, hash: u64, sequence: u64) -> FrameHashUpdate {
    FrameHashUpdate {
        frame: FrameHash { tick, hash },
        sequence,
    }
}

#[test]
fn frame_hash_wire_is_exact_and_malformed_envelopes_are_rejected() {
    let value = update(u64::MAX, 0x1234, 91);
    let encoded = super::super::encode_frame_hash(value);
    assert_eq!(encoded.len(), 25);
    assert!(
        matches!(super::super::decode_event(&encoded), Ok(super::super::EventEnvelope::FrameHash(decoded)) if decoded == value)
    );
    assert!(super::super::decode_event(&encoded[..24]).is_err());
    let mut extra = encoded;
    extra.push(0);
    assert!(super::super::decode_event(&extra).is_err());
}

#[test]
fn out_of_order_hashes_find_first_known_divergence_and_corrections_clear_it() {
    let history = history(4, 4);
    let mut audit = FrameHashAudit::new(4);
    audit.sync_history(&history, 0).unwrap();
    assert!(!audit.comparison().retained_window_matches());
    assert_eq!(audit.comparison().missing_remote, 4);
    for tick in [3, 0, 2, 1] {
        let hash = history.state_hash(tick).unwrap() ^ u64::from(tick == 1 || tick == 3);
        assert_eq!(
            audit.receive(update(tick, hash, tick + 1)),
            Ok(FrameHashReceipt::Inserted)
        );
    }
    assert!(matches!(
        audit.comparison().first_divergence,
        Some(TickDivergence::Hash { tick: 1, .. })
    ));
    let correction = update(1, history.state_hash(1).unwrap(), 10);
    assert_eq!(audit.receive(correction), Ok(FrameHashReceipt::Replaced));
    assert_eq!(audit.receive(correction), Ok(FrameHashReceipt::Duplicate));
    assert_eq!(
        audit.receive(update(1, correction.frame.hash ^ 1, 2)),
        Ok(FrameHashReceipt::Stale)
    );
    assert_eq!(
        audit.receive(update(1, correction.frame.hash ^ 1, 10)),
        Err(FrameHashAuditError::ConflictingDuplicate {
            tick: 1,
            sequence: 10
        })
    );
    assert!(matches!(
        audit.comparison().first_divergence,
        Some(TickDivergence::Hash { tick: 3, .. })
    ));
    audit
        .receive(update(3, history.state_hash(3).unwrap(), 11))
        .unwrap();
    assert!(audit.comparison().retained_window_matches());
}

#[test]
fn missing_and_future_ticks_are_not_agreement_and_remote_memory_is_bounded() {
    let history = history(2, 2);
    let mut audit = FrameHashAudit::new(2);
    audit.sync_history(&history, 0).unwrap();
    audit.receive(update(2, 1, 1)).unwrap();
    audit.receive(update(3, 1, 2)).unwrap();
    assert_eq!(audit.comparison().compared, 0);
    assert_eq!(audit.comparison().pending_remote, 2);
    assert!(!audit.comparison().retained_window_matches());
    assert_eq!(
        audit.receive(update(0, 1, 3)),
        Err(FrameHashAuditError::RemoteWindowFull { tick: 0 })
    );
    assert_eq!(
        audit.receive(update(4, 1, 4)),
        Err(FrameHashAuditError::OutsideWindow {
            tick: 4,
            end_exclusive: 4
        })
    );
    assert_eq!(audit.remote.len(), 2);
}

#[test]
fn retained_window_reports_unsent_expiry_and_refuses_expired_remote_frames() {
    let mut history = history(2, 2);
    let mut audit = FrameHashAudit::new(2);
    audit.sync_history(&history, 0).unwrap();
    audit
        .receive(update(0, history.state_hash(0).unwrap(), 1))
        .unwrap();
    history.capture(&World::new());
    assert_eq!(
        audit.sync_history(&history, 2),
        Err(FrameHashAuditError::UnsentFrameExpired { tick: 0 })
    );
    assert_eq!(audit.local_log().len(), 2);
    assert_eq!(audit.pending_len(), 2);
    assert!(audit.remote(0).is_none());
    assert_eq!(
        audit.receive(update(0, 7, 10)),
        Err(FrameHashAuditError::Expired { tick: 0, oldest: 1 })
    );
}

#[test]
fn truncated_future_and_replayed_hashes_replace_unsent_branch() {
    let mut history = history(8, 4);
    let mut audit = FrameHashAudit::new(8);
    audit.sync_history(&history, 0).unwrap();
    let old = audit.local(2).unwrap();
    history.truncate_after(1);
    history.seek(2);
    audit.sync_history(&history, 2).unwrap();
    assert!(audit.local(2).is_none());
    assert_eq!(audit.pending_len(), 2);
    history.capture(&World::new());
    audit.sync_history(&history, 2).unwrap();
    let correction = audit.local(2).unwrap();
    assert!(correction.sequence > old.sequence);
    assert_ne!(correction.frame.hash, old.frame.hash);
    assert_eq!(audit.pending_len(), 3);
    assert_eq!(audit.pending.get(&2), Some(&correction));
    assert!(audit.local(3).is_none());
}

#[test]
fn sequence_exhaustion_is_explicit_and_never_wraps() {
    let history = history(2, 1);
    let mut audit = FrameHashAudit::new(2);
    audit.sequence = u64::MAX;
    assert_eq!(
        audit.sync_history(&history, 0),
        Err(FrameHashAuditError::SequenceExhausted)
    );
    assert!(audit.local(0).is_none());
    assert_eq!(audit.pending_len(), 0);
    assert_eq!(audit.comparison().missing_local, 1);
    assert!(!audit.comparison().retained_window_matches());

    // Exhaustion during a correction must remove the previously valid hash,
    // rather than continuing to compare the abandoned branch as a match.
    let mut audit = FrameHashAudit::new(2);
    audit.sync_history(&history, 0).unwrap();
    let original = audit.local(0).unwrap();
    audit.receive(original).unwrap();
    assert!(audit.comparison().retained_window_matches());
    audit.sequence = u64::MAX;
    assert_eq!(
        audit.record_retained(0, original.frame.hash ^ 1),
        Err(FrameHashAuditError::SequenceExhausted)
    );
    assert!(audit.local(0).is_none());
    assert_eq!(audit.pending_len(), 0);
    assert!(!audit.comparison().retained_window_matches());
}

#[test]
fn clone_only_component_values_are_not_claimed_as_hashed() {
    let mut world = World::new();
    let entity = world.spawn(Value(0));
    world.register_snapshot_component::<Value>();
    world.get_mut::<Value>(entity).unwrap().0 = 1;
    let before = world.snapshot();
    world.get_mut::<Value>(entity).unwrap().0 = 2;
    let after = world.snapshot();
    assert!(!before.value_hash_complete());
    assert!(!after.value_hash_complete());
    assert_eq!(
        before.state_hash(),
        after.state_hash(),
        "same-tick metadata cannot hash clone-only values"
    );
}
