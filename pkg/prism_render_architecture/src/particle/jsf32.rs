//! Bob Jenkins' Small Fast `PRNG` (`jsf32`, 32-bit variant).
//!
//! This module provides a pure-integer `u32` random number generator
//! (`RNG`). It uses only `wrapping_*` arithmetic and `rotate_left`, with no
//! floating point, no transcendental functions, and no heap allocation.
//!
//! The algorithm advances four `u32` words of state (`a`, `b`, `c`, `d`).
//! Seeding initializes `a` to a fixed constant and `b`, `c`, `d` to the seed,
//! then discards 20 outputs so the state is well mixed.

/// Bob Jenkins' Small Fast `PRNG` (`Jsf32`), 32-bit variant.
///
/// The four fields form the internal state. Instances are compared and copied
/// by value, which keeps snapshot and determinism checks straightforward.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Jsf32 {
    a: u32,
    b: u32,
    c: u32,
    d: u32,
}

impl Jsf32 {
    /// Fixed seeding constant used for the `a` word.
    const SEED_CONST: u32 = 0xf1ea_5eed;

    /// Number of outputs discarded during seeding to mix the state.
    const SEED_ROUNDS: usize = 20;

    /// Create a new `Jsf32` from a 32-bit seed.
    ///
    /// The `a` word is set to a fixed constant and `b`, `c`, `d` are set to the
    /// seed. The generator is then advanced [`Self::SEED_ROUNDS`] times.
    #[must_use]
    pub fn new(seed: u32) -> Self {
        let mut s = Self {
            a: Self::SEED_CONST,
            b: seed,
            c: seed,
            d: seed,
        };
        for _ in 0..Self::SEED_ROUNDS {
            let _ = s.next_u32();
        }
        s
    }

    /// Create a `Jsf32` directly from a raw four-word state.
    ///
    /// This performs no seeding rounds; it is intended for restoring a
    /// previously captured state (see [`Self::state`]).
    #[must_use]
    pub fn from_state(a: u32, b: u32, c: u32, d: u32) -> Self {
        Self { a, b, c, d }
    }

    /// Return the current raw four-word state as `(a, b, c, d)`.
    ///
    /// Pairing this with [`Self::from_state`] allows exact snapshots and
    /// reproduction of a sequence from any point.
    #[must_use]
    pub fn state(&self) -> (u32, u32, u32, u32) {
        (self.a, self.b, self.c, self.d)
    }

    /// Advance the state and return the next `u32` output.
    pub fn next_u32(&mut self) -> u32 {
        let e = self.a.wrapping_sub(self.b.rotate_left(27));
        self.a = self.b ^ self.c.rotate_left(17);
        self.b = self.c.wrapping_add(self.d);
        self.c = self.d.wrapping_add(e);
        self.d = e.wrapping_add(self.a);
        self.d
    }
}

#[cfg(test)]
mod tests {
    use super::Jsf32;

    // Hard reference vectors (first five outputs for each seed). These are
    // enshrined and must all be hit exactly.
    const REF_SEED_0: [u32; 5] = [
        0x1a9b_6c07,
        0x9a55_0895,
        0xf12b_e876,
        0x0902_ba19,
        0x20f1_a244,
    ];
    const REF_SEED_1: [u32; 5] = [
        0xa251_32f4,
        0x1efa_0761,
        0x332b_56b3,
        0xd1ae_db87,
        0x4c4d_7156,
    ];
    const REF_SEED_X: [u32; 5] = [
        0x4324_435b,
        0x2820_3161,
        0xe6d1_95a6,
        0x31e5_3a77,
        0x7c50_cdfb,
    ];

    const SEED_X: u32 = 0x1234_5678;

    fn take5(seed: u32) -> [u32; 5] {
        let mut r = Jsf32::new(seed);
        [
            r.next_u32(),
            r.next_u32(),
            r.next_u32(),
            r.next_u32(),
            r.next_u32(),
        ]
    }

    // --- Hard vectors: seed 0, element by element ---

    #[test]
    fn ref_seed0_elem0() {
        assert_eq!(Jsf32::new(0).next_u32(), REF_SEED_0[0]);
    }

    #[test]
    fn ref_seed0_elem1() {
        assert_eq!(take5(0)[1], REF_SEED_0[1]);
    }

    #[test]
    fn ref_seed0_elem2() {
        assert_eq!(take5(0)[2], REF_SEED_0[2]);
    }

    #[test]
    fn ref_seed0_elem3() {
        assert_eq!(take5(0)[3], REF_SEED_0[3]);
    }

    #[test]
    fn ref_seed0_elem4() {
        assert_eq!(take5(0)[4], REF_SEED_0[4]);
    }

    #[test]
    fn ref_seed0_full() {
        assert_eq!(take5(0), REF_SEED_0);
    }

    // --- Hard vectors: seed 1, element by element ---

    #[test]
    fn ref_seed1_elem0() {
        assert_eq!(Jsf32::new(1).next_u32(), REF_SEED_1[0]);
    }

    #[test]
    fn ref_seed1_elem1() {
        assert_eq!(take5(1)[1], REF_SEED_1[1]);
    }

    #[test]
    fn ref_seed1_elem2() {
        assert_eq!(take5(1)[2], REF_SEED_1[2]);
    }

    #[test]
    fn ref_seed1_elem3() {
        assert_eq!(take5(1)[3], REF_SEED_1[3]);
    }

    #[test]
    fn ref_seed1_elem4() {
        assert_eq!(take5(1)[4], REF_SEED_1[4]);
    }

    #[test]
    fn ref_seed1_full() {
        assert_eq!(take5(1), REF_SEED_1);
    }

    // --- Hard vectors: seed 0x12345678, element by element ---

    #[test]
    fn ref_seedx_elem0() {
        assert_eq!(Jsf32::new(SEED_X).next_u32(), REF_SEED_X[0]);
    }

    #[test]
    fn ref_seedx_elem1() {
        assert_eq!(take5(SEED_X)[1], REF_SEED_X[1]);
    }

    #[test]
    fn ref_seedx_elem2() {
        assert_eq!(take5(SEED_X)[2], REF_SEED_X[2]);
    }

    #[test]
    fn ref_seedx_elem3() {
        assert_eq!(take5(SEED_X)[3], REF_SEED_X[3]);
    }

    #[test]
    fn ref_seedx_elem4() {
        assert_eq!(take5(SEED_X)[4], REF_SEED_X[4]);
    }

    #[test]
    fn ref_seedx_full() {
        assert_eq!(take5(SEED_X), REF_SEED_X);
    }

    // --- Determinism: identical seeds yield identical sequences ---

    #[test]
    fn determinism_seed0() {
        assert_eq!(take5(0), take5(0));
    }

    #[test]
    fn determinism_seed1() {
        assert_eq!(take5(1), take5(1));
    }

    #[test]
    fn determinism_seedx() {
        assert_eq!(take5(SEED_X), take5(SEED_X));
    }

    #[test]
    fn determinism_long_run() {
        let mut p = Jsf32::new(0xdead_beef);
        let mut q = Jsf32::new(0xdead_beef);
        for _ in 0..1000 {
            assert_eq!(p.next_u32(), q.next_u32());
        }
    }

    #[test]
    fn determinism_state_tracks() {
        let mut p = Jsf32::new(42);
        let mut q = Jsf32::new(42);
        for _ in 0..64 {
            let _ = p.next_u32();
            let _ = q.next_u32();
            assert_eq!(p.state(), q.state());
        }
    }

    // --- Different seeds yield different sequences ---

    #[test]
    fn different_seeds_0_and_1() {
        assert_ne!(take5(0), take5(1));
    }

    #[test]
    fn different_seeds_adjacent() {
        assert_ne!(take5(100), take5(101));
    }

    #[test]
    fn different_seeds_large() {
        assert_ne!(take5(0xffff_ffff), take5(0x7fff_ffff));
    }

    #[test]
    fn different_seeds_zero_vs_x() {
        assert_ne!(take5(0), take5(SEED_X));
    }

    #[test]
    fn different_seeds_states_diverge() {
        let a = Jsf32::new(7);
        let b = Jsf32::new(8);
        assert_ne!(a.state(), b.state());
    }

    // --- from_state consistency ---

    #[test]
    fn from_state_roundtrips_new0() {
        let seeded = Jsf32::new(0);
        let (a, b, c, d) = seeded.state();
        let rebuilt = Jsf32::from_state(a, b, c, d);
        assert_eq!(seeded, rebuilt);
    }

    #[test]
    fn from_state_roundtrips_new1() {
        let seeded = Jsf32::new(1);
        let (a, b, c, d) = seeded.state();
        assert_eq!(Jsf32::from_state(a, b, c, d), seeded);
    }

    #[test]
    fn from_state_roundtrips_newx() {
        let seeded = Jsf32::new(SEED_X);
        let (a, b, c, d) = seeded.state();
        assert_eq!(Jsf32::from_state(a, b, c, d), seeded);
    }

    #[test]
    fn from_state_reproduces_seed0_outputs() {
        let (a, b, c, d) = Jsf32::new(0).state();
        let mut rebuilt = Jsf32::from_state(a, b, c, d);
        let out = [
            rebuilt.next_u32(),
            rebuilt.next_u32(),
            rebuilt.next_u32(),
            rebuilt.next_u32(),
            rebuilt.next_u32(),
        ];
        assert_eq!(out, REF_SEED_0);
    }

    #[test]
    fn from_state_custom_fields() {
        let r = Jsf32::from_state(1, 2, 3, 4);
        assert_eq!(r.state(), (1, 2, 3, 4));
    }

    #[test]
    fn from_state_no_seeding_rounds() {
        // from_state must not advance the state the way new() does.
        let raw = Jsf32::from_state(0xf1ea_5eed, 0, 0, 0);
        assert_eq!(raw.state(), (0xf1ea_5eed, 0, 0, 0));
    }

    #[test]
    fn from_state_matches_manual_new0() {
        // Reproduce new(0) seeding manually via from_state + rounds.
        let mut manual = Jsf32::from_state(0xf1ea_5eed, 0, 0, 0);
        for _ in 0..20 {
            let _ = manual.next_u32();
        }
        assert_eq!(manual, Jsf32::new(0));
    }

    // --- Snapshot / reproduction ---

    #[test]
    fn snapshot_reproduces_tail() {
        let mut p = Jsf32::new(123);
        for _ in 0..50 {
            let _ = p.next_u32();
        }
        let (a, b, c, d) = p.state();
        let mut resumed = Jsf32::from_state(a, b, c, d);
        for _ in 0..50 {
            assert_eq!(p.next_u32(), resumed.next_u32());
        }
    }

    #[test]
    fn snapshot_mid_sequence() {
        let mut p = Jsf32::new(0xabcd);
        let mut collected = [0_u32; 10];
        for slot in &mut collected {
            *slot = p.next_u32();
        }
        // Re-run from a fresh generator and compare.
        let mut q = Jsf32::new(0xabcd);
        let mut again = [0_u32; 10];
        for slot in &mut again {
            *slot = q.next_u32();
        }
        assert_eq!(collected, again);
    }

    #[test]
    fn snapshot_state_restore_equivalence() {
        let mut p = Jsf32::new(555);
        for _ in 0..17 {
            let _ = p.next_u32();
        }
        let saved = p.state();
        let next_from_live = p.next_u32();
        let mut restored = Jsf32::from_state(saved.0, saved.1, saved.2, saved.3);
        assert_eq!(restored.next_u32(), next_from_live);
    }

    // --- General behavior ---

    #[test]
    fn next_advances_state() {
        let mut p = Jsf32::new(9);
        let before = p.state();
        let _ = p.next_u32();
        assert_ne!(p.state(), before);
    }

    #[test]
    fn sequence_not_constant() {
        let mut p = Jsf32::new(0);
        let first = p.next_u32();
        let mut all_same = true;
        for _ in 0..32 {
            if p.next_u32() != first {
                all_same = false;
            }
        }
        assert!(!all_same);
    }

    #[test]
    fn two_instances_independent() {
        let mut p = Jsf32::new(1);
        let mut q = Jsf32::new(1);
        let _ = p.next_u32();
        // Advancing p must not affect q.
        assert_eq!(q.next_u32(), REF_SEED_1[0]);
    }

    #[test]
    fn copy_preserves_sequence() {
        let p = Jsf32::new(77);
        let mut original = p;
        let mut copy = p;
        for _ in 0..20 {
            assert_eq!(original.next_u32(), copy.next_u32());
        }
    }

    #[test]
    fn seed_const_in_initial_state_before_rounds() {
        let raw = Jsf32::from_state(Jsf32::SEED_CONST, 5, 5, 5);
        assert_eq!(raw.state().0, 0xf1ea_5eed);
    }

    #[test]
    fn distinct_outputs_present() {
        // Across a short run the generator should produce more than one value.
        let mut p = Jsf32::new(314);
        let a = p.next_u32();
        let b = p.next_u32();
        let c = p.next_u32();
        let distinct = (a != b) || (b != c) || (a != c);
        assert!(distinct);
    }

    #[test]
    fn long_run_terminates_and_mixes() {
        let mut p = Jsf32::new(0x5555_aaaa);
        let mut acc = 0_u32;
        for _ in 0..10_000 {
            acc = acc.wrapping_add(p.next_u32());
        }
        // The accumulated value is state-dependent; just assert the run
        // completed by checking the state changed from its seeded start.
        assert_ne!(p.state(), Jsf32::new(0x5555_aaaa).state());
    }
}
