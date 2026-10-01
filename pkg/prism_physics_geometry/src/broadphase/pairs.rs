//! Overlapping-pair generation and a persistent pair cache.

use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use crate::bvh::DynamicBvh;

/// A candidate collision pair between two proxy payloads.
///
/// The two payloads are stored in canonical order (`a < b`) so that a pair has
/// a single representation regardless of discovery order, which makes pairs
/// comparable and hashable for deduplication.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct BroadPhasePair {
    /// Smaller of the two payloads.
    pub a: u64,
    /// Larger of the two payloads.
    pub b: u64,
}

impl BroadPhasePair {
    /// Creates a pair from two payloads, ordering them so that `a < b`.
    ///
    /// # Panics
    ///
    /// Panics if `first == second`, since a proxy cannot pair with itself.
    #[inline]
    pub fn new(first: u64, second: u64) -> BroadPhasePair {
        assert!(
            first != second,
            "a broad-phase pair must join two distinct proxies"
        );
        if first < second {
            BroadPhasePair {
                a: first,
                b: second,
            }
        } else {
            BroadPhasePair {
                a: second,
                b: first,
            }
        }
    }
}

/// Generates every overlapping leaf fat-box pair in `bvh`.
///
/// The result is deduplicated and sorted in ascending [`BroadPhasePair`] order.
/// Pairs come from [`DynamicBvh::query_self_pairs`], a simultaneous tree
/// descent that reports each overlapping pair once, so the cost scales with the
/// number of actual overlaps rather than `O(n^2)` and avoids re-descending the
/// whole tree per leaf.
pub fn generate_pairs(bvh: &DynamicBvh) -> Vec<BroadPhasePair> {
    let mut set: BTreeSet<BroadPhasePair> = BTreeSet::new();
    bvh.query_self_pairs(&mut |a, b| {
        set.insert(BroadPhasePair::new(a, b));
    });
    set.into_iter().collect()
}

/// The set difference between two consecutive frames of overlapping pairs.
#[derive(Clone, Default, PartialEq, Eq, Debug)]
pub struct PairChanges {
    /// Pairs that overlap this frame but did not last frame.
    pub started: Vec<BroadPhasePair>,
    /// Pairs that overlapped last frame but no longer do.
    pub ended: Vec<BroadPhasePair>,
}

/// A persistent cache of active overlapping pairs across frames.
///
/// Call [`PersistentBroadPhase::update`] once per frame with the current tree;
/// it returns the pairs that started or ended relative to the previous frame
/// and updates the cached active set.
#[derive(Clone, Default, Debug)]
pub struct PersistentBroadPhase {
    /// The set of pairs overlapping as of the most recent update.
    active: BTreeSet<BroadPhasePair>,
}

impl PersistentBroadPhase {
    /// Creates a persistent broad-phase with no active pairs.
    #[inline]
    pub fn new() -> PersistentBroadPhase {
        PersistentBroadPhase {
            active: BTreeSet::new(),
        }
    }

    /// Recomputes overlapping pairs for `bvh` and returns the delta relative to
    /// the previously cached active set.
    pub fn update(&mut self, bvh: &DynamicBvh) -> PairChanges {
        let current: BTreeSet<BroadPhasePair> = generate_pairs(bvh).into_iter().collect();
        let started: Vec<BroadPhasePair> = current.difference(&self.active).copied().collect();
        let ended: Vec<BroadPhasePair> = self.active.difference(&current).copied().collect();
        self.active = current;
        PairChanges { started, ended }
    }

    /// Returns the currently active set of overlapping pairs.
    #[inline]
    pub fn active(&self) -> &BTreeSet<BroadPhasePair> {
        &self.active
    }
}

#[cfg(test)]
mod tests {
    use super::{generate_pairs, BroadPhasePair, PersistentBroadPhase};
    use crate::bounding::Aabb;
    use crate::bvh::DynamicBvh;
    use alloc::collections::BTreeSet;
    use alloc::vec::Vec;
    use glam::Vec3;

    struct Rng(u64);
    impl Rng {
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn next_f32(&mut self) -> f32 {
            (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
        }
        fn range(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (hi - lo) * self.next_f32()
        }
    }

    fn random_box(rng: &mut Rng) -> Aabb {
        let c = Vec3::new(
            rng.range(-20.0, 20.0),
            rng.range(-20.0, 20.0),
            rng.range(-20.0, 20.0),
        );
        let h = Vec3::splat(rng.range(1.0, 3.0));
        Aabb::from_center_half_extents(c, h)
    }

    #[test]
    fn pair_new_is_canonical() {
        assert_eq!(BroadPhasePair::new(5, 2), BroadPhasePair::new(2, 5));
        let p = BroadPhasePair::new(9, 4);
        assert_eq!(p.a, 4);
        assert_eq!(p.b, 9);
    }

    #[test]
    #[should_panic(expected = "distinct")]
    fn pair_new_rejects_self_pair() {
        let _ = BroadPhasePair::new(7, 7);
    }

    #[test]
    fn generate_pairs_matches_brute_force() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        let mut tree = DynamicBvh::new();
        let mut fats: Vec<(u64, Aabb)> = Vec::new();
        for i in 0..80u64 {
            let id = tree.insert(random_box(&mut rng), i);
            fats.push((i, tree.get_aabb(id).unwrap()));
        }

        let got: BTreeSet<BroadPhasePair> = generate_pairs(&tree).into_iter().collect();

        let mut expected: BTreeSet<BroadPhasePair> = BTreeSet::new();
        for i in 0..fats.len() {
            for j in (i + 1)..fats.len() {
                if fats[i].1.intersects(&fats[j].1) {
                    expected.insert(BroadPhasePair::new(fats[i].0, fats[j].0));
                }
            }
        }
        assert_eq!(got, expected);

        // Result is sorted.
        let v = generate_pairs(&tree);
        let mut sorted = v.clone();
        sorted.sort();
        assert_eq!(v, sorted);
    }

    #[test]
    fn persistent_started_and_ended() {
        let mut tree = DynamicBvh::new();
        // Frame 1: proxies 1 and 2 overlap; 3 is far away.
        let p1 = tree.insert(
            Aabb::from_center_half_extents(Vec3::ZERO, Vec3::splat(1.0)),
            1,
        );
        let _p2 = tree.insert(
            Aabb::from_center_half_extents(Vec3::new(0.5, 0.0, 0.0), Vec3::splat(1.0)),
            2,
        );
        let p3 = tree.insert(
            Aabb::from_center_half_extents(Vec3::new(100.0, 0.0, 0.0), Vec3::splat(1.0)),
            3,
        );

        let mut broad = PersistentBroadPhase::new();
        let f1 = broad.update(&tree);
        assert!(f1.started.contains(&BroadPhasePair::new(1, 2)));
        assert!(f1.ended.is_empty());
        assert!(broad.active().contains(&BroadPhasePair::new(1, 2)));

        // Frame 2: move proxy 3 next to 1/2 (now overlaps both) and move 1 away
        // so that the (1,2) pair ends.
        assert!(tree.update(
            p3,
            Aabb::from_center_half_extents(Vec3::new(0.25, 0.0, 0.0), Vec3::splat(1.0))
        ));
        assert!(tree.update(
            p1,
            Aabb::from_center_half_extents(Vec3::new(80.0, 0.0, 0.0), Vec3::splat(1.0))
        ));

        let f2 = broad.update(&tree);
        let started: BTreeSet<BroadPhasePair> = f2.started.iter().copied().collect();
        let ended: BTreeSet<BroadPhasePair> = f2.ended.iter().copied().collect();
        assert!(started.contains(&BroadPhasePair::new(2, 3)));
        assert!(ended.contains(&BroadPhasePair::new(1, 2)));
        assert!(broad.active().contains(&BroadPhasePair::new(2, 3)));
        assert!(!broad.active().contains(&BroadPhasePair::new(1, 2)));
    }
}
