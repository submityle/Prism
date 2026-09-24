//! Deterministic state hashing for online desync detection and localization.
//!
//! A networked, deterministic simulation stays in lock-step only if every peer
//! computes bit-identical state. To detect (and *locate*) divergence cheaply,
//! each peer hashes its published [`StateSnapshot`] every frame and exchanges a
//! single [`StateHash`]; a mismatch means a desync. [`locate_divergence`] then
//! narrows the mismatch to the first differing body so debugging does not start
//! from a whole-world diff.
//!
//! The hash is a plain FNV-1a over the raw IEEE-754 bit patterns of each body's
//! generation, pose, and velocities in slot order. Hashing the exact bits (via
//! [`f32::to_bits`]) keeps the hash a faithful function of state without any
//! floating-point rounding of its own, so identical states always hash equal.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. FNV-1a is
//! a public-domain, non-cryptographic hash implemented from its published
//! specification.

use crate::snapshot::StateSnapshot;

/// FNV-1a 64-bit offset basis.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a 64-bit prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// A 64-bit deterministic digest of a [`StateSnapshot`].
///
/// Two snapshots hash to the same [`StateHash`] iff their per-slot occupancy,
/// generations, poses, and velocities have identical bit patterns.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct StateHash(pub u64);

/// Incremental FNV-1a accumulator over raw bytes.
#[derive(Clone, Copy, Debug)]
struct Fnv1a(u64);

impl Fnv1a {
    /// Starts a new accumulator at the FNV offset basis.
    const fn new() -> Fnv1a {
        Fnv1a(FNV_OFFSET)
    }

    /// Folds a single byte into the digest.
    fn write_u8(&mut self, byte: u8) {
        self.0 ^= u64::from(byte);
        self.0 = self.0.wrapping_mul(FNV_PRIME);
    }

    /// Folds all four bytes of a `u32` (little-endian) into the digest.
    fn write_u32(&mut self, value: u32) {
        for byte in value.to_le_bytes() {
            self.write_u8(byte);
        }
    }

    /// Folds an `f32` by hashing its IEEE-754 bit pattern.
    fn write_f32(&mut self, value: f32) {
        self.write_u32(value.to_bits());
    }

    /// Folds a three-component vector by hashing each component.
    fn write_vec3(&mut self, value: glam::Vec3) {
        for c in value.to_array() {
            self.write_f32(c);
        }
    }

    /// Returns the accumulated digest.
    const fn finish(self) -> u64 {
        self.0
    }
}

/// Hashes the state stored at a single `slot` of `snapshot`.
fn hash_slot(snapshot: &StateSnapshot, slot: usize) -> u64 {
    let mut h = Fnv1a::new();
    let occupied = snapshot.slot_occupied(slot);
    h.write_u8(u8::from(occupied));
    if occupied {
        h.write_u32(snapshot.generation_at_slot(slot).unwrap_or(0));
        if let Some(pose) = snapshot.pose_at_slot(slot) {
            h.write_vec3(pose.position);
            for c in pose.orientation.to_array() {
                h.write_f32(c);
            }
        }
        h.write_vec3(snapshot.linear_velocity_at_slot(slot));
        h.write_vec3(snapshot.angular_velocity_at_slot(slot));
    }
    h.finish()
}

/// Computes the whole-snapshot [`StateHash`].
///
/// The slot count is mixed in first so that two snapshots with a different
/// number of slots cannot collide by coincidence, then every slot's per-slot
/// digest is folded in slot order.
#[must_use]
pub fn hash_state(snapshot: &StateSnapshot) -> StateHash {
    let mut h = Fnv1a::new();
    h.write_u32(snapshot.slot_count() as u32);
    for slot in 0..snapshot.slot_count() {
        // Fold each slot's digest in as eight little-endian bytes.
        for byte in hash_slot(snapshot, slot).to_le_bytes() {
            h.write_u8(byte);
        }
    }
    StateHash(h.finish())
}

/// Returns the first slot at which `a` and `b` diverge, or `None` if the two
/// snapshots hash identically slot-for-slot.
///
/// A differing slot count is itself a divergence: the extra slots on the longer
/// snapshot are compared against an empty slot, so the first surplus (or first
/// mismatching) slot index is reported. This is the localization step run after
/// [`hash_state`] flags a whole-world mismatch.
#[must_use]
pub fn locate_divergence(a: &StateSnapshot, b: &StateSnapshot) -> Option<usize> {
    let n = a.slot_count().max(b.slot_count());
    for slot in 0..n {
        let ha = slot_digest_or_empty(a, slot);
        let hb = slot_digest_or_empty(b, slot);
        if ha != hb {
            return Some(slot);
        }
    }
    None
}

/// Per-slot digest, treating an out-of-range slot as an empty (unoccupied) one.
fn slot_digest_or_empty(snapshot: &StateSnapshot, slot: usize) -> u64 {
    if slot < snapshot.slot_count() {
        hash_slot(snapshot, slot)
    } else {
        let mut h = Fnv1a::new();
        h.write_u8(0);
        h.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::body::BodyDesc;
    use crate::world::PhysicsWorld;
    use glam::Vec3;

    #[test]
    fn identical_states_hash_equal() {
        let mut world = PhysicsWorld::default();
        world.spawn(BodyDesc::dynamic_at(Vec3::new(1.0, 2.0, 3.0)));
        world.spawn(BodyDesc::dynamic_at(Vec3::new(-4.0, 0.0, 5.0)));
        let a = StateSnapshot::capture(&world);
        let b = StateSnapshot::capture(&world);
        assert_eq!(hash_state(&a), hash_state(&b));
        assert!(locate_divergence(&a, &b).is_none());
    }

    #[test]
    fn moved_body_changes_hash_and_is_located() {
        let mut world = PhysicsWorld::default();
        world.spawn(BodyDesc::dynamic_at(Vec3::ZERO));
        let h1 = world.spawn(BodyDesc::dynamic_at(Vec3::ONE));
        let before = StateSnapshot::capture(&world);
        // Perturb only the second body (slot 1).
        world.bodies.set_position(h1, Vec3::new(1.0, 1.0, 1.001));
        let after = StateSnapshot::capture(&world);
        assert_ne!(hash_state(&before), hash_state(&after));
        assert_eq!(locate_divergence(&before, &after), Some(1));
    }

    #[test]
    fn differing_slot_count_is_a_divergence() {
        let mut world = PhysicsWorld::default();
        world.spawn(BodyDesc::dynamic_at(Vec3::ZERO));
        let small = StateSnapshot::capture(&world);
        world.spawn(BodyDesc::dynamic_at(Vec3::ONE));
        let big = StateSnapshot::capture(&world);
        assert_ne!(hash_state(&small), hash_state(&big));
        assert_eq!(locate_divergence(&small, &big), Some(1));
    }

    #[test]
    fn velocity_difference_changes_hash() {
        let mut a_world = PhysicsWorld::default();
        let ha = a_world.spawn(BodyDesc::dynamic_at(Vec3::ZERO));
        let mut b_world = PhysicsWorld::default();
        let hb = b_world.spawn(BodyDesc::dynamic_at(Vec3::ZERO));
        a_world.bodies.set_linear_velocity(ha, Vec3::X);
        b_world.bodies.set_linear_velocity(hb, Vec3::Y);
        let a = StateSnapshot::capture(&a_world);
        let b = StateSnapshot::capture(&b_world);
        assert_ne!(hash_state(&a), hash_state(&b));
    }
}
