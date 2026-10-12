//! Peer reconciliation: detect a desync and roll back onto the peer's inputs
//! (design §14, §15).
//!
//! [`reconcile_with_peer`] closes the rollback loop with the last piece the
//! design asks for: compare the local per-frame state hashes
//! ([`FrameHash`]) against a peer's, find the first frame they disagree on, and
//! [`reconcile`](crate::rollback_replay::RollbackReplay::reconcile) to it using
//! the peer's authoritative inputs — so a desync is repaired deterministically
//! instead of drifting apart.
//!
//! The peer is supplied through the small [`PeerAuthoritative`] trait, so the
//! driver carries no networking dependency: a real transport fills in the hashes
//! and inputs it has received, and a test supplies a fake.

use crate::determinism::FrameHash;
use crate::rollback_replay::RollbackReplay;
use prism_ecs::world::World;

/// The authoritative per-frame facts a peer can supply during reconciliation.
pub trait PeerAuthoritative<F> {
    /// The peer's state hash for `frame`, or `None` until it is received.
    fn hash(&self, frame: u64) -> Option<u64>;
    /// The peer's authoritative input for `frame`, or `None` until received.
    fn input(&self, frame: u64) -> Option<F>;
}

/// The outcome of [`reconcile_with_peer`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReconcileOutcome {
    /// Local and peer hashes agreed across every comparable retained frame (or
    /// the rollback window held no frames to compare).
    Agreed,
    /// Diverged at `frame`, then reconciled; the session now stands at `current`.
    Reconciled {
        /// The first divergent (and rolled-back-to) frame.
        frame: u64,
        /// The session's frame number after re-simulation.
        current: u64,
    },
    /// Diverged at `frame` but could not be repaired — the peer had not supplied
    /// every input in `frame..current`, or the rollback was refused (evicted).
    Unreconciled {
        /// The first divergent frame that could not be repaired.
        frame: u64,
    },
}

/// Compare the local per-frame hashes against a peer's over the retained
/// rollback window and, on the first disagreement, roll back to that frame and
/// re-simulate with the peer's authoritative inputs.
///
/// `local` and `session` must share a frame numbering (both advance once per
/// fixed step from frame `0`). `simulate` advances the world by exactly one
/// frame from a given input, matching
/// [`RollbackReplay::commit`](crate::rollback_replay::RollbackReplay::commit)'s
/// commit-then-simulate contract.
///
/// The scan only covers frames still retained in the session's snapshot window
/// (older frames cannot be rolled back to), and it stops at the first frame where
/// *both* sides have a hash and those hashes differ; a frame missing a hash on
/// **either** side (the peer has not sent one yet, or the local hash ring's
/// window is narrower than the snapshot window) is skipped — so keep the local
/// hash window at least as wide as the snapshot window to compare every retained
/// frame.
pub fn reconcile_with_peer<F, P>(
    session: &mut RollbackReplay<F>,
    world: &mut World,
    local: &FrameHash,
    peer: &P,
    mut simulate: impl FnMut(&mut World, &F),
) -> ReconcileOutcome
where
    F: Clone,
    P: PeerAuthoritative<F>,
{
    let current = session.frame();
    let Some(oldest) = session.history().oldest_frame() else {
        return ReconcileOutcome::Agreed;
    };

    let Some(frame) = (oldest..current).find(|&f| {
        matches!((local.hash_at(f), peer.hash(f)), (Some(a), Some(b)) if a != b)
    }) else {
        return ReconcileOutcome::Agreed;
    };

    let mut inputs = Vec::with_capacity((current - frame) as usize);
    for f in frame..current {
        match peer.input(f) {
            Some(input) => inputs.push(input),
            None => return ReconcileOutcome::Unreconciled { frame },
        }
    }

    match session.reconcile(world, frame, &inputs, &mut simulate) {
        Some(new_current) => ReconcileOutcome::Reconciled {
            frame,
            current: new_current,
        },
        None => ReconcileOutcome::Unreconciled { frame },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ecs::component::Component;
    use prism_ecs::entity::Entity;

    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    struct Counter(i32);
    impl Component for Counter {}

    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    struct Marker;
    impl Component for Marker {}

    fn world() -> (World, Entity) {
        let mut w = World::new();
        w.register_snapshot_component_hashable::<Counter>();
        w.register_snapshot_component::<Marker>();
        let e = w.spawn((Counter(0), Marker));
        (w, e)
    }

    fn sim(world: &mut World, e: Entity, input: i32) {
        world.get_mut::<Counter>(e).unwrap().0 += input;
    }

    fn forward(rr: &mut RollbackReplay<i32>, world: &mut World, e: Entity, input: i32) {
        rr.commit(world, input);
        sim(world, e, input);
    }

    /// A local [`FrameHash`] with one finalized frame per value.
    fn local_hash(values: &[u64]) -> FrameHash {
        let mut hash = FrameHash::with_window(values.len().max(1));
        for &value in values {
            hash.write_u64(value);
            hash.finalize_frame();
        }
        hash
    }

    struct FakePeer {
        hashes: Vec<u64>,
        inputs: Vec<i32>,
    }

    impl PeerAuthoritative<i32> for FakePeer {
        fn hash(&self, frame: u64) -> Option<u64> {
            self.hashes.get(frame as usize).copied()
        }
        fn input(&self, frame: u64) -> Option<i32> {
            self.inputs.get(frame as usize).copied()
        }
    }

    /// A peer whose hashes match `local` except frame `diverge`, which it flips.
    fn diverging_peer(local: &FrameHash, frames: u64, diverge: u64, inputs: Vec<i32>) -> FakePeer {
        let mut hashes: Vec<u64> = (0..frames).map(|f| local.hash_at(f).unwrap()).collect();
        hashes[diverge as usize] ^= 0x9E37_79B9_7F4A_7C15;
        FakePeer { hashes, inputs }
    }

    #[test]
    fn agrees_when_hashes_match() {
        let (mut w, e) = world();
        let mut rr = RollbackReplay::<i32>::new(8);
        forward(&mut rr, &mut w, e, 1);
        forward(&mut rr, &mut w, e, 2);
        forward(&mut rr, &mut w, e, 3);
        let local = local_hash(&[10, 20, 30]);
        let peer = FakePeer {
            hashes: (0..3).map(|f| local.hash_at(f).unwrap()).collect(),
            inputs: vec![1, 2, 3],
        };
        let outcome =
            reconcile_with_peer(&mut rr, &mut w, &local, &peer, |w, i| sim(w, e, *i));
        assert_eq!(outcome, ReconcileOutcome::Agreed);
        assert_eq!(w.get::<Counter>(e).unwrap().0, 6);
    }

    #[test]
    fn reconciles_to_the_first_divergent_frame_with_peer_inputs() {
        let (mut w, e) = world();
        let mut rr = RollbackReplay::<i32>::new(8);
        forward(&mut rr, &mut w, e, 1);
        forward(&mut rr, &mut w, e, 2);
        forward(&mut rr, &mut w, e, 3);

        let local = local_hash(&[10, 20, 30]);
        // Peer says frame 1 diverged and supplies the authoritative inputs
        // (frame 1 is 10, not the locally-applied 2).
        let peer = diverging_peer(&local, 3, 1, vec![1, 10, 3]);
        let outcome =
            reconcile_with_peer(&mut rr, &mut w, &local, &peer, |w, i| sim(w, e, *i));
        assert_eq!(
            outcome,
            ReconcileOutcome::Reconciled {
                frame: 1,
                current: 3
            }
        );
        assert_eq!(w.get::<Counter>(e).unwrap().0, 14); // 1 + 10 + 3
        let inputs: Vec<i32> = rr.inputs().iter().map(|r| r.frame).collect();
        assert_eq!(inputs, vec![1, 10, 3]);
    }

    #[test]
    fn missing_peer_inputs_leave_the_world_untouched() {
        let (mut w, e) = world();
        let mut rr = RollbackReplay::<i32>::new(8);
        forward(&mut rr, &mut w, e, 1);
        forward(&mut rr, &mut w, e, 2);
        forward(&mut rr, &mut w, e, 3);

        let local = local_hash(&[10, 20, 30]);
        // Peer flags the divergence at frame 1 but is missing frame 2's input.
        let peer = FakePeer {
            hashes: {
                let mut h: Vec<u64> = (0..3).map(|f| local.hash_at(f).unwrap()).collect();
                h[1] ^= 1;
                h
            },
            inputs: vec![1, 10], // only frames 0 and 1
        };
        let outcome =
            reconcile_with_peer(&mut rr, &mut w, &local, &peer, |w, i| sim(w, e, *i));
        assert_eq!(outcome, ReconcileOutcome::Unreconciled { frame: 1 });
        assert_eq!(w.get::<Counter>(e).unwrap().0, 6);
        assert_eq!(rr.frame(), 3);
    }
}
