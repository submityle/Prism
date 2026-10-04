//! Reusable uniform spatial-hash broad phase for sphere clouds.
//!
//! Several discrete-element resolvers in this crate (Hertzian contact,
//! Cundall–Strack tangential history, bonded particles, capillary bridges,
//! rolling granular contact) each need the same first step: given a cloud of
//! particle centres and radii, enumerate the unordered pairs that are close
//! enough to be worth a precise narrow-phase test, without paying the full
//! `O(n^2)` cost of testing every pair. Historically each resolver inlined its
//! own uniform-grid binning; this module factors that broad phase out into a
//! single reusable, independently tested component.
//!
//! # Algorithm
//!
//! Each particle has an axis-aligned bounding box of half-extent
//! `radius + margin/2`. The grid cell size is chosen as
//! `2 * max_radius + margin`, which is the largest possible box diameter plus
//! the search margin. With that cell size, two particles whose expanded boxes
//! overlap must lie in the same or an adjacent cell, so each particle only has
//! to test the 27 cells of its `3x3x3` neighbourhood. For every candidate the
//! broad phase performs a cheap axis-aligned-bounding-box overlap test (sphere
//! contact implies box overlap, so the test is conservative — it never drops a
//! true contact pair).
//!
//! # Determinism
//!
//! Each unordered pair is emitted exactly once, as `(a, b)` with `a < b`, and
//! the returned list is sorted ascending. The output therefore does not depend
//! on hash-bucket iteration order, which keeps downstream force accumulation
//! reproducible run to run.
//!
//! # Reuse
//!
//! The grid owns its bucket map so repeated calls reuse the allocation instead
//! of rebuilding a fresh `HashMap` every frame.

use std::collections::HashMap;

use glam::Vec3;

/// A reusable uniform spatial-hash broad phase for sphere clouds.
///
/// Construct once and call [`candidate_pairs`](Self::candidate_pairs) each
/// frame; the internal bucket map is cleared and refilled in place so the
/// allocation is amortised across calls.
#[derive(Clone, Debug, Default)]
pub struct UniformGridBroadphase {
    /// Scratch bucket map, reused across calls. Maps an integer cell
    /// coordinate to the indices of the particles binned into it.
    cells: HashMap<[i32; 3], Vec<u32>>,
}

impl UniformGridBroadphase {
    /// Create an empty broad phase.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enumerate the unordered candidate pairs whose expanded bounding boxes
    /// overlap.
    ///
    /// `positions` and `radii` must have equal length; every position must be
    /// finite and every radius must be finite and strictly positive. `margin`
    /// is the extra separation (beyond touching) at which a pair is still
    /// considered a candidate — pass `0.0` for pure contact, or e.g. a
    /// capillary rupture distance to keep near-touching pairs in the list. It
    /// must be finite and non-negative.
    ///
    /// Returns `None` if any input is invalid. On success returns the pairs as
    /// `(a, b)` with `a < b`, deduplicated and sorted ascending. An empty cloud
    /// yields an empty list.
    pub fn candidate_pairs(
        &mut self,
        positions: &[Vec3],
        radii: &[f32],
        margin: f32,
    ) -> Option<Vec<(u32, u32)>> {
        let n = positions.len();
        if radii.len() != n {
            return None;
        }
        if !margin.is_finite() || margin < 0.0 {
            return None;
        }
        for (pos, &r) in positions.iter().zip(radii.iter()) {
            if !pos.is_finite() || !r.is_finite() || r <= 0.0 {
                return None;
            }
        }
        if n == 0 {
            return Some(Vec::new());
        }

        // Grid origin at the minimum corner of the cloud; cell size large
        // enough that any overlapping pair lands within a 3x3x3 neighbourhood.
        let mut origin = positions[0];
        let mut max_radius = radii[0];
        for &p in positions.iter() {
            origin = origin.min(p);
        }
        for &r in radii.iter() {
            if r > max_radius {
                max_radius = r;
            }
        }
        let cell_size = 2.0 * max_radius + margin;

        self.cells.clear();
        for (i, &pos) in positions.iter().enumerate() {
            let cell = cell_of(pos, origin, cell_size);
            self.cells.entry(cell).or_default().push(i as u32);
        }

        let mut pairs: Vec<(u32, u32)> = Vec::new();
        for (i, &pos_i) in positions.iter().enumerate() {
            let ci = cell_of(pos_i, origin, cell_size);
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        let neighbour = [ci[0] + dx, ci[1] + dy, ci[2] + dz];
                        let Some(bucket) = self.cells.get(&neighbour) else {
                            continue;
                        };
                        for &j in bucket {
                            let ju = j as usize;
                            // Emit each unordered pair once (j strictly after i).
                            if ju > i
                                && aabb_overlap(pos_i, radii[i], positions[ju], radii[ju], margin)
                            {
                                pairs.push((i as u32, j));
                            }
                        }
                    }
                }
            }
        }
        pairs.sort_unstable();
        Some(pairs)
    }
}

/// Integer grid cell coordinate of a point relative to `origin` at `cell_size`.
fn cell_of(point: Vec3, origin: Vec3, cell_size: f32) -> [i32; 3] {
    let rel = (point - origin) / cell_size;
    [
        rel.x.floor() as i32,
        rel.y.floor() as i32,
        rel.z.floor() as i32,
    ]
}

/// Whether two spheres' bounding boxes, each grown by `margin / 2`, overlap.
///
/// Equivalent to testing `|a - b| <= r_a + r_b + margin` on every axis.
fn aabb_overlap(pa: Vec3, ra: f32, pb: Vec3, rb: f32, margin: f32) -> bool {
    let reach = ra + rb + margin;
    (pa.x - pb.x).abs() <= reach && (pa.y - pb.y).abs() <= reach && (pa.z - pb.z).abs() <= reach
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference brute-force broad phase using the same predicate, for
    /// equivalence testing.
    fn brute_force(positions: &[Vec3], radii: &[f32], margin: f32) -> Vec<(u32, u32)> {
        let mut pairs = Vec::new();
        for i in 0..positions.len() {
            for j in (i + 1)..positions.len() {
                if aabb_overlap(positions[i], radii[i], positions[j], radii[j], margin) {
                    pairs.push((i as u32, j as u32));
                }
            }
        }
        pairs.sort_unstable();
        pairs
    }

    #[test]
    fn empty_cloud_has_no_pairs() {
        let mut bp = UniformGridBroadphase::new();
        assert_eq!(bp.candidate_pairs(&[], &[], 0.0), Some(Vec::new()));
    }

    #[test]
    fn single_particle_has_no_pairs() {
        let mut bp = UniformGridBroadphase::new();
        let pairs = bp.candidate_pairs(&[Vec3::ZERO], &[1.0], 0.0).unwrap();
        assert!(pairs.is_empty());
    }

    #[test]
    fn rejects_invalid_inputs() {
        let mut bp = UniformGridBroadphase::new();
        // Length mismatch.
        assert!(bp
            .candidate_pairs(&[Vec3::ZERO], &[1.0, 1.0], 0.0)
            .is_none());
        // Negative margin.
        assert!(bp.candidate_pairs(&[Vec3::ZERO], &[1.0], -0.1).is_none());
        // Non-finite margin.
        assert!(bp
            .candidate_pairs(&[Vec3::ZERO], &[1.0], f32::NAN)
            .is_none());
        // Non-positive radius.
        assert!(bp.candidate_pairs(&[Vec3::ZERO], &[0.0], 0.0).is_none());
        // Non-finite position.
        assert!(bp
            .candidate_pairs(&[Vec3::new(f32::INFINITY, 0.0, 0.0)], &[1.0], 0.0)
            .is_none());
    }

    #[test]
    fn touching_pair_is_reported() {
        let mut bp = UniformGridBroadphase::new();
        // Two unit spheres exactly touching (centres 2 apart).
        let positions = [Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0)];
        let radii = [1.0, 1.0];
        let pairs = bp.candidate_pairs(&positions, &radii, 0.0).unwrap();
        assert_eq!(pairs, vec![(0, 1)]);
    }

    #[test]
    fn far_pair_is_not_reported() {
        let mut bp = UniformGridBroadphase::new();
        // Centres 10 apart, radii 1: well beyond contact and margin.
        let positions = [Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0)];
        let radii = [1.0, 1.0];
        let pairs = bp.candidate_pairs(&positions, &radii, 0.5).unwrap();
        assert!(pairs.is_empty());
    }

    #[test]
    fn margin_includes_near_touching_pair() {
        let mut bp = UniformGridBroadphase::new();
        // Centres 2.4 apart, radii 1 => gap 0.4. Not touching, but within a
        // 0.5 margin.
        let positions = [Vec3::ZERO, Vec3::new(2.4, 0.0, 0.0)];
        let radii = [1.0, 1.0];
        assert!(bp
            .candidate_pairs(&positions, &radii, 0.0)
            .unwrap()
            .is_empty());
        assert_eq!(
            bp.candidate_pairs(&positions, &radii, 0.5).unwrap(),
            vec![(0, 1)]
        );
    }

    #[test]
    fn output_is_sorted_and_unique() {
        let mut bp = UniformGridBroadphase::new();
        // A tight cluster where several pairs overlap.
        let positions = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
        ];
        let radii = [0.6, 0.6, 0.6, 0.6];
        let pairs = bp.candidate_pairs(&positions, &radii, 0.0).unwrap();
        for w in pairs.windows(2) {
            assert!(w[0] < w[1], "pairs must be strictly ascending/unique");
        }
        for &(a, b) in &pairs {
            assert!(a < b, "pair must be ordered a < b");
        }
    }

    #[test]
    fn matches_brute_force_on_a_lattice() {
        // A deterministic 4x4x4 lattice of jittered spheres so neighbourhoods
        // are populated and the grid path is exercised against brute force.
        let mut positions = Vec::new();
        let mut radii = Vec::new();
        let mut seed: u32 = 0x1234_5678;
        let mut next = || {
            // Simple LCG for reproducible jitter in [-0.1, 0.1].
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((seed >> 8) as f32 / (1u32 << 24) as f32) * 0.2 - 0.1
        };
        for x in 0..4 {
            for y in 0..4 {
                for z in 0..4 {
                    positions.push(Vec3::new(
                        x as f32 + next(),
                        y as f32 + next(),
                        z as f32 + next(),
                    ));
                    radii.push(0.55);
                }
            }
        }

        let mut bp = UniformGridBroadphase::new();
        for &margin in &[0.0_f32, 0.3, 0.9] {
            let grid = bp.candidate_pairs(&positions, &radii, margin).unwrap();
            let brute = brute_force(&positions, &radii, margin);
            assert_eq!(grid, brute, "grid broad phase disagrees at margin {margin}");
        }
    }

    #[test]
    fn mixed_radii_are_handled() {
        let mut bp = UniformGridBroadphase::new();
        // One large sphere overlapping two small ones; the two small ones are
        // far from each other.
        let positions = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.5, 0.0, 0.0),
            Vec3::new(-2.5, 0.0, 0.0),
        ];
        let radii = [2.0, 0.6, 0.6];
        let grid = bp.candidate_pairs(&positions, &radii, 0.0).unwrap();
        let brute = brute_force(&positions, &radii, 0.0);
        assert_eq!(grid, brute);
        // Large (0) touches both small ones; smalls do not touch each other.
        assert_eq!(grid, vec![(0, 1), (0, 2)]);
    }

    #[test]
    fn repeated_calls_reuse_and_stay_consistent() {
        let mut bp = UniformGridBroadphase::new();
        let positions = [Vec3::ZERO, Vec3::new(1.5, 0.0, 0.0)];
        let radii = [1.0, 1.0];
        let first = bp.candidate_pairs(&positions, &radii, 0.0).unwrap();
        // A different cloud, then back to the first: results must be stable.
        let _ = bp.candidate_pairs(&[Vec3::ZERO, Vec3::new(100.0, 0.0, 0.0)], &[1.0, 1.0], 0.0);
        let again = bp.candidate_pairs(&positions, &radii, 0.0).unwrap();
        assert_eq!(first, again);
        assert_eq!(first, vec![(0, 1)]);
    }
}
