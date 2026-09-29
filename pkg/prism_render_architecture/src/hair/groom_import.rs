//! Groom import: normalize raw authored guide strands into a uniform,
//! solver-ready layout.
//!
//! Authored grooms (Alembic caches from `UE5` Groom, AMD `TressFX` assets,
//! DCC exports) arrive as guide *polylines* with arbitrary and non-uniform
//! control-point counts and uneven spacing. The strand solver
//! ([`super::dynamics`]) and the guide-to-render interpolator
//! ([`super::interpolation`]) both assume a *uniform* number of control points
//! per guide with roughly even arc-length spacing, so rest lengths are stable
//! and blends line up point-for-point. Production engines therefore resample
//! every guide on import; this module owns that deterministic step:
//!
//! 1. **Arc-length resampling** — [`resample_strand`] reparameterizes one raw
//!    polyline to a fixed control-point count by even arc length, pinning the
//!    first and last output to the true root and tip so no strand shrinks.
//! 2. **Batch normalization** — [`resample_groom`] resamples many raw guides
//!    (given as flat points plus per-strand ranges) into one flat,
//!    fixed-stride [`ResampledGroom`], carrying each strand's authored
//!    attributes (radius taper, root UV, seed) forward and skipping malformed
//!    ranges instead of panicking.
//! 3. **Rest lengths** — [`strand_rest_lengths`] derives the per-segment rest
//!    lengths a resampled guide feeds straight into the XPBD edge constraint.
//!
//! Everything is array-in / array-out, deterministic (fixed evaluation order,
//! no real randomness), and panic-free on empty, single-point, degenerate
//! (all-coincident), or out-of-range input.

use alloc::vec::Vec;

use super::dynamics::Vec3;

/// Authored per-strand attributes carried through import.
///
/// These ride alongside geometry but never affect the resampling math; they are
/// what the raster/shading side reads to taper strand width and drive
/// per-strand randomization (`seed`) and root-space texturing (`root_uv`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StrandAttributes {
    /// Strand radius at the root (world units); clamped to `>= 0` on build.
    pub root_radius: f32,
    /// Strand radius at the tip (world units); clamped to `>= 0` on build.
    pub tip_radius: f32,
    /// Root anchor UV on the scalp mesh, for root-space texturing/masks.
    pub root_uv: [f32; 2],
    /// Deterministic per-strand seed for downstream randomization.
    pub seed: u32,
}

impl StrandAttributes {
    /// Builds attributes, clamping both radii to be non-negative.
    #[must_use]
    pub fn new(root_radius: f32, tip_radius: f32, root_uv: [f32; 2], seed: u32) -> Self {
        Self {
            root_radius: root_radius.max(0.0),
            tip_radius: tip_radius.max(0.0),
            root_uv,
            seed,
        }
    }

    /// Linearly interpolated radius at normalized length `t` (`0` = root,
    /// `1` = tip); `t` is clamped to `0..=1`.
    #[must_use]
    pub fn radius_at(&self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        self.root_radius + (self.tip_radius - self.root_radius) * t
    }
}

impl Default for StrandAttributes {
    fn default() -> Self {
        Self {
            root_radius: 0.0,
            tip_radius: 0.0,
            root_uv: [0.0, 0.0],
            seed: 0,
        }
    }
}

/// Describes one raw guide strand inside a shared flat point buffer.
///
/// `start..start + len` indexes the raw points of this strand; ranges that fall
/// outside the buffer are skipped by [`resample_groom`] rather than panicking,
/// so a stale offset table cannot crash import.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RawStrandRange {
    /// Index of this strand's first raw point in the shared buffer.
    pub start: usize,
    /// Number of raw points in this strand.
    pub len: usize,
    /// Authored attributes carried onto the resampled strand.
    pub attributes: StrandAttributes,
}

/// A resampled groom: fixed-stride guide control points plus per-strand
/// attributes.
///
/// `positions` is flat, strand-major: strand `i`'s control points are
/// `positions[i * points_per_strand .. (i + 1) * points_per_strand]`, always
/// `points_per_strand` long. `attributes[i]` are the authored attributes for
/// that strand, so the two run parallel and equal length
/// (`attributes.len() == strand_count()`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ResampledGroom {
    /// Uniform control-point count per strand (always `>= 2`).
    pub points_per_strand: usize,
    /// Flat strand-major control points; length `strand_count * points_per_strand`.
    pub positions: Vec<Vec3>,
    /// Per-strand authored attributes; length `strand_count`.
    pub attributes: Vec<StrandAttributes>,
}

impl ResampledGroom {
    /// Number of resampled strands.
    #[must_use]
    pub fn strand_count(&self) -> usize {
        self.attributes.len()
    }

    /// Returns `true` when no strand survived import.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.attributes.is_empty()
    }

    /// Borrows strand `i`'s control points, or `None` when out of range.
    #[must_use]
    pub fn strand(&self, i: usize) -> Option<&[Vec3]> {
        if i >= self.strand_count() {
            return None;
        }
        let base = i * self.points_per_strand;
        Some(&self.positions[base..base + self.points_per_strand])
    }
}

/// Linear interpolation between two control points (no `Vec3::lerp` in the
/// solver math type, so it is spelled out here in fixed order).
fn lerp(a: Vec3, b: Vec3, t: f32) -> Vec3 {
    a.add(b.sub(a).scale(t))
}

/// Resamples one raw guide polyline to `target_points` control points spaced
/// evenly by arc length.
///
/// The first and last outputs are pinned to the true root and tip so the strand
/// keeps its full length; interior points land at equal arc-length fractions.
/// `target_points` is clamped to at least `2` (a strand needs a root and a
/// tip). Degenerate input is handled without panicking:
///
/// - empty `raw` yields an empty `Vec` (the caller drops the strand);
/// - a single raw point, or an all-coincident polyline (zero total length),
///   yields `target_points` copies of the root, i.e. a valid zero-length strand
///   whose rest lengths are all `0`.
#[must_use]
pub fn resample_strand(raw: &[Vec3], target_points: u32) -> Vec<Vec3> {
    let target = (target_points.max(2)) as usize;
    if raw.is_empty() {
        return Vec::new();
    }
    if raw.len() == 1 {
        let mut out = Vec::with_capacity(target);
        out.resize(target, raw[0]);
        return out;
    }

    // Cumulative arc length along the raw polyline; `cumulative[i]` is the
    // distance from the root to raw point `i`.
    let mut cumulative = Vec::with_capacity(raw.len());
    cumulative.push(0.0_f32);
    let mut total = 0.0_f32;
    for pair in raw.windows(2) {
        total += pair[1].sub(pair[0]).length();
        cumulative.push(total);
    }

    let mut out = Vec::with_capacity(target);
    if total <= 0.0 {
        // All points coincide: emit the root `target` times (zero-length strand).
        out.resize(target, raw[0]);
        return out;
    }

    let last_raw = raw.len() - 1;
    let mut seg = 0usize;
    for j in 0..target {
        // Target arc-length distance for output j, exact at both ends.
        let d = if j + 1 == target {
            total
        } else {
            total * (j as f32) / ((target - 1) as f32)
        };
        // Advance the segment cursor until d lies in [cumulative[seg], cumulative[seg+1]].
        while seg < last_raw && cumulative[seg + 1] < d {
            seg += 1;
        }
        let d0 = cumulative[seg];
        let d1 = cumulative[seg + 1];
        let span = d1 - d0;
        let t = if span > 0.0 { (d - d0) / span } else { 0.0 };
        out.push(lerp(raw[seg], raw[seg + 1], t));
    }
    out
}

/// Resamples many raw guide strands into one fixed-stride [`ResampledGroom`].
///
/// `points` is a shared flat buffer of raw control points; each entry in
/// `strands` slices out one guide (`start..start + len`) plus its authored
/// attributes. Every valid guide is resampled to `target_points`
/// (clamped `>= 2`) evenly by arc length and appended in input order. A strand
/// whose range is empty or runs past the buffer is skipped (not emitted, not
/// panicked on), so `strand_count()` counts only the strands that survived.
#[must_use]
pub fn resample_groom(
    points: &[Vec3],
    strands: &[RawStrandRange],
    target_points: u32,
) -> ResampledGroom {
    let points_per_strand = (target_points.max(2)) as usize;
    let mut positions = Vec::new();
    let mut attributes = Vec::new();

    for range in strands {
        let Some(end) = range.start.checked_add(range.len) else {
            continue;
        };
        if range.len == 0 || end > points.len() {
            continue;
        }
        let raw = &points[range.start..end];
        let resampled = resample_strand(raw, target_points);
        if resampled.len() != points_per_strand {
            // Only empty raw produces a mismatch, and that is filtered above;
            // guard defensively so the flat stride invariant always holds.
            continue;
        }
        positions.extend_from_slice(&resampled);
        attributes.push(range.attributes);
    }

    ResampledGroom {
        points_per_strand,
        positions,
        attributes,
    }
}

/// Per-segment rest lengths of a resampled strand.
///
/// Returns one length per edge (`strand.len().saturating_sub(1)` entries): the
/// distance between consecutive control points, which the XPBD edge constraint
/// in [`super::dynamics`] uses as its target. A strand shorter than two points
/// has no edges and yields an empty `Vec`. Never panics.
#[must_use]
pub fn strand_rest_lengths(strand: &[Vec3]) -> Vec<f32> {
    if strand.len() < 2 {
        return Vec::new();
    }
    let mut lengths = Vec::with_capacity(strand.len() - 1);
    for pair in strand.windows(2) {
        lengths.push(pair[1].sub(pair[0]).length());
    }
    lengths
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-4;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    fn close_v(a: Vec3, b: Vec3) -> bool {
        close(a.x, b.x) && close(a.y, b.y) && close(a.z, b.z)
    }

    #[test]
    fn resample_pins_root_and_tip_and_counts() {
        let raw = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 3.0, 0.0),
            Vec3::new(0.0, 6.0, 0.0),
        ];
        let out = resample_strand(&raw, 5);
        assert_eq!(out.len(), 5);
        assert!(close_v(out[0], raw[0]), "root moved");
        assert!(close_v(out[4], raw[3]), "tip moved");
    }

    #[test]
    fn resample_spaces_points_evenly_by_arc_length() {
        // A straight line of total length 6: 4 output points -> spacing 2 each.
        let raw = [Vec3::new(0.0, 0.0, 0.0), Vec3::new(6.0, 0.0, 0.0)];
        let out = resample_strand(&raw, 4);
        let lens = strand_rest_lengths(&out);
        assert_eq!(lens.len(), 3);
        for l in lens {
            assert!(close(l, 2.0), "uneven spacing: {l}");
        }
    }

    #[test]
    fn resample_reparameterizes_nonuniform_input() {
        // Bunched-up raw sampling on a straight length-4 line; resampling to 5
        // points must give even spacing 1.0 regardless of raw clustering.
        let raw = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.1, 0.0, 0.0),
            Vec3::new(0.2, 0.0, 0.0),
            Vec3::new(4.0, 0.0, 0.0),
        ];
        let out = resample_strand(&raw, 5);
        let lens = strand_rest_lengths(&out);
        assert_eq!(lens.len(), 4);
        for l in lens {
            assert!(close(l, 1.0), "expected even 1.0 spacing, got {l}");
        }
    }

    #[test]
    fn target_points_clamped_to_two() {
        let raw = [Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)];
        let out = resample_strand(&raw, 0);
        assert_eq!(out.len(), 2);
        assert!(close_v(out[0], raw[0]));
        assert!(close_v(out[1], raw[1]));
    }

    #[test]
    fn empty_and_single_and_coincident_never_panic() {
        assert!(resample_strand(&[], 5).is_empty());

        let single = resample_strand(&[Vec3::new(2.0, 3.0, 4.0)], 4);
        assert_eq!(single.len(), 4);
        for p in &single {
            assert!(close_v(*p, Vec3::new(2.0, 3.0, 4.0)));
        }

        let coincident = resample_strand(&[Vec3::new(1.0, 1.0, 1.0), Vec3::new(1.0, 1.0, 1.0)], 3);
        assert_eq!(coincident.len(), 3);
        for p in &coincident {
            assert!(close_v(*p, Vec3::new(1.0, 1.0, 1.0)));
        }
        // A zero-length strand yields all-zero rest lengths.
        for l in strand_rest_lengths(&coincident) {
            assert!(close(l, 0.0));
        }
    }

    #[test]
    fn resample_groom_flattens_with_fixed_stride_and_carries_attributes() {
        let points = [
            // Strand 0: straight length-2 line.
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            // Strand 1: straight length-3 line (3 raw points).
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.5, 0.0),
            Vec3::new(0.0, 3.0, 0.0),
        ];
        let strands = [
            RawStrandRange {
                start: 0,
                len: 2,
                attributes: StrandAttributes::new(0.02, 0.005, [0.1, 0.2], 7),
            },
            RawStrandRange {
                start: 2,
                len: 3,
                attributes: StrandAttributes::new(0.03, 0.01, [0.3, 0.4], 9),
            },
        ];
        let groom = resample_groom(&points, &strands, 3);
        assert_eq!(groom.points_per_strand, 3);
        assert_eq!(groom.strand_count(), 2);
        assert_eq!(groom.positions.len(), 6);
        assert!(!groom.is_empty());

        let s0 = groom.strand(0).expect("strand 0");
        assert!(close_v(s0[0], Vec3::new(0.0, 0.0, 0.0)));
        assert!(close_v(s0[1], Vec3::new(1.0, 0.0, 0.0)));
        assert!(close_v(s0[2], Vec3::new(2.0, 0.0, 0.0)));

        assert_eq!(groom.attributes[0].seed, 7);
        assert_eq!(groom.attributes[1].seed, 9);
        assert!(close(groom.attributes[1].radius_at(0.0), 0.03));
        assert!(close(groom.attributes[1].radius_at(1.0), 0.01));
        assert!(close(groom.attributes[1].radius_at(0.5), 0.02));

        assert!(groom.strand(2).is_none());
    }

    #[test]
    fn resample_groom_skips_out_of_range_and_empty_ranges() {
        let points = [Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)];
        let strands = [
            RawStrandRange {
                start: 0,
                len: 2,
                attributes: StrandAttributes::default(),
            },
            // Empty range: skipped.
            RawStrandRange {
                start: 0,
                len: 0,
                attributes: StrandAttributes::default(),
            },
            // Runs past the buffer: skipped, not panicked on.
            RawStrandRange {
                start: 1,
                len: 5,
                attributes: StrandAttributes::default(),
            },
        ];
        let groom = resample_groom(&points, &strands, 4);
        assert_eq!(groom.strand_count(), 1);
        assert_eq!(groom.positions.len(), 4);
    }

    #[test]
    fn resample_is_order_independent_of_raw_density() {
        // Two rawsamplings of the same straight length-10 line must resample to
        // identical uniform guides.
        let coarse = [Vec3::new(0.0, 0.0, 0.0), Vec3::new(10.0, 0.0, 0.0)];
        let fine = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(7.0, 0.0, 0.0),
            Vec3::new(10.0, 0.0, 0.0),
        ];
        let a = resample_strand(&coarse, 6);
        let b = resample_strand(&fine, 6);
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(&b) {
            assert!(close_v(*x, *y), "resample diverged by raw density");
        }
    }
}
