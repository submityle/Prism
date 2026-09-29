//! Ocean surface level-of-detail: `clipmap` rings, geomorph blending, and
//! spectral cascade distance weighting.
//!
//! An open ocean cannot tessellate the whole visible surface at ripple
//! density: the patch under the camera needs the finest grid, while the
//! horizon needs only a coarse height field. Following `WaveWorks`/`Crest`
//! practice, the surface is a set of camera-centred concentric rings
//! (`clipmap`), each ring covering a geometric annulus at half the density of
//! the ring inside it. To avoid vertex popping when a patch crosses a ring
//! boundary, geometry morphs continuously (geomorphing) across a band near the
//! outer edge of each ring.
//!
//! Independently, the spectral displacement is built from several physical
//! cascades (large swell down to capillary ripples). Fine cascades carry
//! detail that is invisible at distance, so each cascade fades out beyond its
//! own reach; coarser cascades persist farther. This keeps distant water from
//! shimmering with sub-pixel detail while the near surface stays crisp.
//!
//! Everything here is a pure, deterministic classification over caller-supplied
//! distances (meters from the camera). There is no projection or transcendental
//! math; only comparisons, division, and integer powers of the ring growth
//! factor. Ring selection is monotonic in distance and morph weights are
//! clamped to `0..=1`, so the geometry stage can trust the outputs without
//! re-validating them.

use alloc::vec::Vec;

use super::{WaterBodyHandle, EPS};

/// Concentric-ring (`clipmap`) layout around the camera.
///
/// Ring `0` is the innermost, finest patch reaching out to `inner_radius`.
/// Each subsequent ring's outer radius grows by `radius_growth` (`> 1`), so
/// ring `i` reaches `inner_radius * radius_growth^i`. The outermost
/// `morph_fraction` of every ring's radial band is the geomorph zone where its
/// geometry blends toward the next coarser ring.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OceanClipmapConfig {
    /// Number of concentric rings; `0` yields a degenerate single-ring result.
    pub ring_count: u32,
    /// Outer radius of ring `0`, in meters (`> 0`).
    pub inner_radius: f32,
    /// Geometric growth factor of the ring outer radius per level (`> 1`).
    pub radius_growth: f32,
    /// Fraction of each ring's radial band used for geomorph blending, in
    /// `0..=1`. `0` disables morphing (hard ring boundaries).
    pub morph_fraction: f32,
}

impl OceanClipmapConfig {
    /// Highest valid ring index (`ring_count - 1`), or `0` when no rings exist.
    #[must_use]
    pub fn last_ring(self) -> u32 {
        self.ring_count.saturating_sub(1)
    }

    /// Outer radius of a ring, clamped to the last ring for out-of-range input.
    ///
    /// Computed as `inner_radius * radius_growth^ring` using repeated
    /// multiplication (no `powf`), so the result is exactly reproducible.
    #[must_use]
    pub fn ring_outer_radius(self, ring: u32) -> f32 {
        let ring = ring.min(self.last_ring());
        let mut radius = self.inner_radius;
        let mut level = 0;
        while level < ring {
            radius *= self.radius_growth;
            level += 1;
        }
        radius
    }

    /// Inner radius of a ring: `0` for ring `0`, else the previous ring's outer
    /// radius. This is where the ring's radial band begins.
    #[must_use]
    pub fn ring_inner_radius(self, ring: u32) -> f32 {
        if ring == 0 {
            0.0
        } else {
            self.ring_outer_radius(ring - 1)
        }
    }
}

/// Selects the `clipmap` ring that owns a given camera distance.
///
/// Returns the first ring whose outer radius is at least `distance`; distances
/// beyond the outermost ring clamp to the last ring. The result is monotonic
/// non-decreasing in `distance`, so moving away from the camera never selects a
/// finer ring.
#[must_use]
pub fn select_clipmap_ring(distance: f32, cfg: OceanClipmapConfig) -> u32 {
    let last = cfg.last_ring();
    let mut ring = 0;
    while ring < last {
        if distance <= cfg.ring_outer_radius(ring) {
            return ring;
        }
        ring += 1;
    }
    last
}

/// Continuous geomorph weight for a distance, in `0..=1`.
///
/// `0` in the inner part of the selected ring (geometry uses this ring as-is);
/// it ramps linearly to `1` across the outermost `morph_fraction` of the ring's
/// band, where the geometry has fully morphed toward the next coarser ring.
/// Distances past the outermost ring saturate at `1`. A degenerate morph band
/// (zero width) returns `0`.
#[must_use]
pub fn clipmap_morph_weight(distance: f32, cfg: OceanClipmapConfig) -> f32 {
    let ring = select_clipmap_ring(distance, cfg);
    let inner = cfg.ring_inner_radius(ring);
    let outer = cfg.ring_outer_radius(ring);
    let band = outer - inner;
    if band <= EPS || cfg.morph_fraction <= EPS {
        // No usable band or morphing disabled: past the last ring saturate,
        // otherwise stay unmorphed.
        return if distance > outer { 1.0 } else { 0.0 };
    }
    let morph_start = outer - band * cfg.morph_fraction.min(1.0);
    if distance <= morph_start {
        return 0.0;
    }
    let denom = outer - morph_start;
    if denom <= EPS {
        return 1.0;
    }
    clamp01((distance - morph_start) / denom)
}

/// Camera-centred spectral cascade fade layout.
///
/// Cascade `0` is the finest (shortest wavelength) and fades first; each higher
/// index is coarser and reaches `reach_per_cascade` meters farther before it
/// begins to fade. `fade_range` is the width over which any cascade ramps from
/// full weight to zero.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OceanCascadeConfig {
    /// Number of spectral cascades stacked into the displacement.
    pub cascade_count: u32,
    /// Distance at which the finest cascade (index `0`) begins to fade.
    pub fade_start: f32,
    /// Width of the linear fade from full weight to zero, in meters (`> 0`).
    pub fade_range: f32,
    /// Extra reach granted to each coarser cascade before it fades.
    pub reach_per_cascade: f32,
}

impl OceanCascadeConfig {
    /// Distance at which a cascade begins fading (`fade_start` plus its reach).
    #[must_use]
    pub fn fade_begin(self, cascade: u32) -> f32 {
        self.fade_start + (cascade as f32) * self.reach_per_cascade
    }
}

/// Distance weight for a single spectral cascade, in `0..=1`.
///
/// Full weight up to the cascade's `fade_begin`, then a linear ramp down to `0`
/// across `fade_range`, and `0` beyond. The weight is monotonic non-increasing
/// in `distance`; at any fixed distance a coarser cascade (higher index) weighs
/// at least as much as a finer one, so distant water keeps swell and drops
/// ripples.
#[must_use]
pub fn cascade_distance_weight(cascade: u32, distance: f32, cfg: OceanCascadeConfig) -> f32 {
    let begin = cfg.fade_begin(cascade);
    if distance <= begin {
        return 1.0;
    }
    if cfg.fade_range <= EPS {
        return 0.0;
    }
    let t = (distance - begin) / cfg.fade_range;
    clamp01(1.0 - t)
}

/// Fills `out` with the per-cascade distance weights at a distance.
///
/// Writes `min(out.len(), cfg.cascade_count)` entries in cascade order; extra
/// slots in `out` are left untouched and a short slice is filled as far as it
/// reaches, so a mismatched buffer never panics. Returns the number of weights
/// written.
pub fn cascade_weights_into(distance: f32, cfg: OceanCascadeConfig, out: &mut [f32]) -> usize {
    let count = (cfg.cascade_count as usize).min(out.len());
    let mut index = 0;
    while index < count {
        out[index] = cascade_distance_weight(index as u32, distance, cfg);
        index += 1;
    }
    count
}

/// The resolved ocean LOD for one surface patch this frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OceanPatchLod {
    /// Water body this patch belongs to.
    pub body: WaterBodyHandle,
    /// Selected `clipmap` ring.
    pub ring: u32,
    /// Geomorph weight toward the next coarser ring, in `0..=1`.
    pub morph: f32,
}

/// Resolves the `clipmap` ring and geomorph weight for a patch at a distance.
#[must_use]
pub fn resolve_ocean_patch(
    body: WaterBodyHandle,
    distance: f32,
    cfg: OceanClipmapConfig,
) -> OceanPatchLod {
    OceanPatchLod {
        body,
        ring: select_clipmap_ring(distance, cfg),
        morph: clipmap_morph_weight(distance, cfg),
    }
}

/// Ocean patches partitioned by the `clipmap` ring selected for them.
///
/// One bucket per ring (indexed by ring number) lets the geometry stage build
/// a single indirect draw per ring density. Input order is preserved within
/// each bucket so draw submission is deterministic frame to frame.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct OceanClipmapPlan {
    rings: Vec<Vec<OceanPatchLod>>,
}

impl OceanClipmapPlan {
    /// Creates an empty plan with one bucket per ring (at least one bucket).
    #[must_use]
    pub fn with_ring_count(ring_count: u32) -> Self {
        let buckets = (ring_count.max(1)) as usize;
        let mut rings = Vec::with_capacity(buckets);
        let mut index = 0;
        while index < buckets {
            rings.push(Vec::new());
            index += 1;
        }
        Self { rings }
    }

    /// Number of ring buckets in this plan.
    #[must_use]
    pub fn ring_count(&self) -> usize {
        self.rings.len()
    }

    /// Total patches across every ring bucket.
    #[must_use]
    pub fn total(&self) -> usize {
        self.rings.iter().map(Vec::len).sum()
    }

    /// Returns `true` when no patch landed in any bucket.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rings.iter().all(Vec::is_empty)
    }

    /// The bucket backing a given ring, or an empty slice when out of range.
    #[must_use]
    pub fn bucket(&self, ring: u32) -> &[OceanPatchLod] {
        self.rings.get(ring as usize).map_or(&[], Vec::as_slice)
    }

    /// Number of patches routed to a given ring.
    #[must_use]
    pub fn count_of_ring(&self, ring: u32) -> usize {
        self.bucket(ring).len()
    }

    /// Appends a resolved patch to its ring bucket.
    ///
    /// A ring index at or beyond the bucket count is clamped to the last
    /// bucket rather than panicking, so a stale ring never crashes binning.
    pub fn push(&mut self, patch: OceanPatchLod) {
        if self.rings.is_empty() {
            self.rings.push(Vec::new());
        }
        let last = self.rings.len() - 1;
        let index = (patch.ring as usize).min(last);
        self.rings[index].push(patch);
    }
}

/// Resolves and bins a set of ocean patches by their per-patch camera distance.
///
/// `distances[i]` is the camera distance for `bodies[i]`. A body with no
/// matching distance entry is skipped rather than panicking, so a stale or
/// short distance slice cannot crash LOD selection. Input order is preserved
/// within each ring bucket.
#[must_use]
pub fn bin_ocean_patches(
    bodies: &[WaterBodyHandle],
    distances: &[f32],
    cfg: OceanClipmapConfig,
) -> OceanClipmapPlan {
    let mut plan = OceanClipmapPlan::with_ring_count(cfg.ring_count);
    for (index, &body) in bodies.iter().enumerate() {
        let Some(&distance) = distances.get(index) else {
            continue;
        };
        plan.push(resolve_ocean_patch(body, distance, cfg));
    }
    plan
}

/// Clamps a value to the `0..=1` range without branching on float equality.
fn clamp01(value: f32) -> f32 {
    value.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const CLIPMAP: OceanClipmapConfig = OceanClipmapConfig {
        ring_count: 4,
        inner_radius: 32.0,
        radius_growth: 2.0,
        morph_fraction: 0.25,
    };

    const CASCADES: OceanCascadeConfig = OceanCascadeConfig {
        cascade_count: 4,
        fade_start: 100.0,
        fade_range: 50.0,
        reach_per_cascade: 200.0,
    };

    #[test]
    fn ring_outer_radius_grows_geometrically_and_clamps() {
        assert_eq!(CLIPMAP.ring_outer_radius(0), 32.0);
        assert_eq!(CLIPMAP.ring_outer_radius(1), 64.0);
        assert_eq!(CLIPMAP.ring_outer_radius(2), 128.0);
        assert_eq!(CLIPMAP.ring_outer_radius(3), 256.0);
        // Out-of-range ring clamps to the last ring.
        assert_eq!(CLIPMAP.ring_outer_radius(99), 256.0);
    }

    #[test]
    fn ring_selection_is_monotonic_in_distance() {
        let mut prev = select_clipmap_ring(0.0, CLIPMAP);
        let mut distance = 0.0;
        while distance <= 400.0 {
            let ring = select_clipmap_ring(distance, CLIPMAP);
            assert!(ring >= prev, "ring must not decrease with distance");
            assert!(ring <= CLIPMAP.last_ring());
            prev = ring;
            distance += 3.0;
        }
        // Near-camera lands in ring 0, far beyond the horizon clamps to last.
        assert_eq!(select_clipmap_ring(1.0, CLIPMAP), 0);
        assert_eq!(select_clipmap_ring(10_000.0, CLIPMAP), CLIPMAP.last_ring());
    }

    #[test]
    fn morph_weight_stays_in_unit_range_and_ramps_at_boundary() {
        let mut distance = 0.0;
        while distance <= 400.0 {
            let w = clipmap_morph_weight(distance, CLIPMAP);
            assert!((0.0..=1.0).contains(&w), "morph weight out of range: {w}");
            distance += 1.0;
        }
        // Ring 0 spans 0..32 with a 25% morph band => morph starts at 24.
        assert_eq!(clipmap_morph_weight(10.0, CLIPMAP), 0.0);
        assert_eq!(clipmap_morph_weight(24.0, CLIPMAP), 0.0);
        let mid = clipmap_morph_weight(28.0, CLIPMAP);
        assert!(mid > 0.0 && mid < 1.0);
        assert!((clipmap_morph_weight(32.0, CLIPMAP) - 1.0).abs() < EPS);
    }

    #[test]
    fn morph_weight_is_monotonic_inside_a_ring() {
        // Sweep the interior of ring 1 (64..128, morph band starts at 112).
        let mut prev = clipmap_morph_weight(112.0, CLIPMAP);
        let mut distance = 112.0;
        while distance <= 128.0 {
            let w = clipmap_morph_weight(distance, CLIPMAP);
            assert!(w + EPS >= prev, "morph must not decrease within the band");
            prev = w;
            distance += 0.5;
        }
    }

    #[test]
    fn zero_morph_fraction_disables_blending() {
        let cfg = OceanClipmapConfig {
            morph_fraction: 0.0,
            ..CLIPMAP
        };
        assert_eq!(clipmap_morph_weight(30.0, cfg), 0.0);
        assert_eq!(clipmap_morph_weight(60.0, cfg), 0.0);
    }

    #[test]
    fn cascade_weight_is_monotonic_non_increasing_and_bounded() {
        for cascade in 0..CASCADES.cascade_count {
            let mut prev = cascade_distance_weight(cascade, 0.0, CASCADES);
            let mut distance = 0.0;
            while distance <= 1_200.0 {
                let w = cascade_distance_weight(cascade, distance, CASCADES);
                assert!((0.0..=1.0).contains(&w), "cascade weight out of range: {w}");
                assert!(w <= prev + EPS, "cascade weight must not increase");
                prev = w;
                distance += 5.0;
            }
        }
    }

    #[test]
    fn coarser_cascade_persists_at_least_as_far() {
        let mut distance = 0.0;
        while distance <= 1_200.0 {
            let mut prev = cascade_distance_weight(0, distance, CASCADES);
            for cascade in 1..CASCADES.cascade_count {
                let w = cascade_distance_weight(cascade, distance, CASCADES);
                assert!(
                    w + EPS >= prev,
                    "coarser cascade must weigh at least as much"
                );
                prev = w;
            }
            distance += 10.0;
        }
    }

    #[test]
    fn cascade_weights_into_fills_buffer_and_tolerates_mismatch() {
        let mut full = [0.0_f32; 4];
        assert_eq!(cascade_weights_into(120.0, CASCADES, &mut full), 4);
        for (cascade, &w) in full.iter().enumerate() {
            assert_eq!(w, cascade_distance_weight(cascade as u32, 120.0, CASCADES));
        }
        // Short buffer: fill only what fits, no panic.
        let mut short = [0.0_f32; 2];
        assert_eq!(cascade_weights_into(120.0, CASCADES, &mut short), 2);
        // Oversized buffer: extra slots untouched.
        let mut big = [-1.0_f32; 6];
        assert_eq!(cascade_weights_into(120.0, CASCADES, &mut big), 4);
        assert_eq!(big[4], -1.0);
        assert_eq!(big[5], -1.0);
    }

    #[test]
    fn binning_preserves_order_and_skips_missing_distances() {
        let bodies = [
            WaterBodyHandle(1),
            WaterBodyHandle(2),
            WaterBodyHandle(3),
            WaterBodyHandle(4),
        ];
        // Distances: near, near, far, and a fourth with no entry (skipped).
        let distances = [5.0, 8.0, 10_000.0];
        let plan = bin_ocean_patches(&bodies, &distances, CLIPMAP);
        assert_eq!(plan.total(), 3, "body 4 has no distance and is skipped");
        assert_eq!(plan.ring_count(), 4);
        // Bodies 1 and 2 land in ring 0 in input order.
        let ring0 = plan.bucket(0);
        assert_eq!(ring0.len(), 2);
        assert_eq!(ring0[0].body, WaterBodyHandle(1));
        assert_eq!(ring0[1].body, WaterBodyHandle(2));
        // Body 3 lands in the last ring.
        assert_eq!(plan.count_of_ring(CLIPMAP.last_ring()), 1);
        assert_eq!(plan.bucket(CLIPMAP.last_ring())[0].body, WaterBodyHandle(3));
    }

    #[test]
    fn out_of_range_ring_and_bucket_do_not_panic() {
        let mut plan = OceanClipmapPlan::with_ring_count(2);
        // A patch tagged with a ring past the bucket count clamps to the last.
        plan.push(OceanPatchLod {
            body: WaterBodyHandle(7),
            ring: 99,
            morph: 0.5,
        });
        assert_eq!(plan.total(), 1);
        assert_eq!(plan.count_of_ring(1), 1);
        // Querying an out-of-range ring returns an empty slice.
        assert!(plan.bucket(50).is_empty());
    }

    #[test]
    fn degenerate_zero_ring_config_is_safe() {
        let cfg = OceanClipmapConfig {
            ring_count: 0,
            ..CLIPMAP
        };
        assert_eq!(cfg.last_ring(), 0);
        assert_eq!(select_clipmap_ring(500.0, cfg), 0);
        let w = clipmap_morph_weight(500.0, cfg);
        assert!((0.0..=1.0).contains(&w));
        let plan = OceanClipmapPlan::with_ring_count(0);
        assert_eq!(plan.ring_count(), 1);
        assert!(plan.is_empty());
    }
}
