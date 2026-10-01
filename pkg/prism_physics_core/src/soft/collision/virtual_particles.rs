//! `NvCloth`-style virtual particles for cloth self-collision.
//!
//! The discrete self-collision tier in [`self_collision`](super::self_collision)
//! is a point-to-point spatial hash: it separates cloth *vertices* that come
//! within a fabric `thickness` of one another. That catches most layer
//! stacking, but it has a well-known blind spot: a lone vertex can slip straight
//! *through the interior of a large triangle* without ever coming within
//! `thickness` of any of that triangle's three corner vertices, so no
//! point-to-point pair ever fires and the layers interpenetrate.
//!
//! NVIDIA `NvCloth` closes that gap without paying for a full continuous
//! triangle-triangle test by seeding each triangle with a handful of **virtual
//! particles**: fixed barycentric sample points on the face (its centroid and
//! edge midpoints) that are fed into the *same* uniform spatial hash as the
//! real vertices. A vertex diving through a face now finds a virtual particle
//! sitting on that face and is pushed back out, while the correction applied to
//! the virtual particle is scattered back onto the triangle's three real
//! vertices by the barycentric weights so momentum and mass are conserved.
//!
//! This module owns that tier as pure array-in / array-out math over the
//! particle store's raw columns, mirroring the determinism guarantees of
//! [`self_collision`](super::self_collision):
//!
//! * The sample list has a fixed order — every real particle first (index
//!   `0..n`), then the virtual particles in generation order — so the spatial
//!   hash buckets, the ascending [`BTreeMap`] cell traversal and the `b > a`
//!   pair test are all deterministic.
//! * Corrections are Gauss-Seidel (applied in place as pairs are found) and
//!   sample positions are recomputed from the live vertex positions each time,
//!   so the same inputs always produce bit-identical outputs.
//! * A sample pair that shares any *active* real vertex (a vertex carrying
//!   positive barycentric weight in either sample) is skipped, which suppresses
//!   a triangle colliding with its own vertices or with an adjacent triangle
//!   that shares an edge/corner — only genuinely non-incident geometry is
//!   separated.
//! * Pinned vertices (`inverse_mass <= 0`) never move; a sample whose active
//!   vertices are all pinned is immovable and its free partner takes the whole
//!   correction. Coincident samples separate along a fixed `+X` axis, and every
//!   degenerate input falls back deterministically and never produces [`f32::NAN`].
//!
//! Passing an empty virtual-particle slice makes
//! [`resolve_self_collision_virtual`] behave exactly like the point-to-point
//! [`resolve_self_collision`](super::resolve_self_collision): the real-vertex
//! samples still collide with one another, so this is a strict superset of the
//! base tier rather than a replacement.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. Seeding
//! triangle faces with fixed barycentric sample points and feeding them into
//! the shared uniform spatial hash is the standard `NvCloth` technique; the
//! mass-conserving barycentric scatter is elementary rigid-body mechanics.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use glam::Vec3;

use crate::math::scalar::Real;

use super::{cell_of, EPS_LEN_SQ};

/// A reusable table of barycentric sample weights applied to every triangle.
///
/// Each row is a `[w0, w1, w2]` barycentric coordinate on the triangle
/// `[v0, v1, v2]`; the entries are sanitized on construction so every row is
/// finite, non-negative and sums to exactly `1.0`. One [`VirtualParticle`] is
/// emitted per `(triangle, row)` pair by [`generate_virtual_particles`], so a
/// four-row pattern turns a `t`-triangle mesh into `4 * t` virtual particles.
#[derive(Clone, Debug, PartialEq)]
pub struct VirtualParticlePattern {
    weights: Vec<[Real; 3]>,
}

impl VirtualParticlePattern {
    /// The `NvCloth` default: the face centroid plus the three edge midpoints.
    ///
    /// These four samples are the classic `NvCloth` seeding — the centroid
    /// guards the middle of the face and the three edge midpoints guard the
    /// thin strips near each edge, which is where a vertex is most likely to
    /// tunnel between two corner vertices.
    #[must_use]
    pub fn nvcloth_default() -> Self {
        Self::from_weights(&[
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
        ])
    }

    /// Builds a pattern from raw barycentric rows, sanitizing each row.
    ///
    /// Every row is clamped to non-negative finite components and then
    /// renormalized to sum to `1.0`; a row that cannot be normalized (all zero,
    /// negative or non-finite) falls back to the centroid `[1/3, 1/3, 1/3]` so
    /// the pattern never carries a degenerate weight triple. An empty input
    /// yields an empty pattern (which generates no virtual particles).
    #[must_use]
    pub fn from_weights(rows: &[[Real; 3]]) -> Self {
        let weights = rows.iter().map(|row| sanitize_weights(*row)).collect();
        Self { weights }
    }

    /// The number of sample rows (virtual particles emitted per triangle).
    #[must_use]
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    /// Returns `true` when the pattern has no rows and generates nothing.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }

    /// The sanitized barycentric rows, in table order.
    #[must_use]
    pub fn rows(&self) -> &[[Real; 3]] {
        &self.weights
    }
}

impl Default for VirtualParticlePattern {
    fn default() -> Self {
        Self::nvcloth_default()
    }
}

/// Clamps a barycentric row to finite non-negative components summing to `1.0`.
///
/// Non-finite or negative components are treated as `0.0`; if the surviving
/// components sum to a positive value the row is scaled to sum to `1.0`,
/// otherwise the row falls back to the centroid so it is never degenerate.
fn sanitize_weights(row: [Real; 3]) -> [Real; 3] {
    let mut clamped = [0.0; 3];
    let mut sum = 0.0;
    for (out, &w) in clamped.iter_mut().zip(row.iter()) {
        let value = if w.is_finite() && w > 0.0 { w } else { 0.0 };
        *out = value;
        sum += value;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        [clamped[0] * inv, clamped[1] * inv, clamped[2] * inv]
    } else {
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
    }
}

/// A barycentric sample point bound to one triangle's three real vertices.
///
/// The sample's world position is `Σ weights[k] * position[verts[k]]`, and any
/// correction the solver applies to that position is scattered back onto the
/// three vertices by [`resolve_self_collision_virtual`]. `verts` are always the
/// three distinct corner indices of a non-degenerate triangle; `weights` come
/// straight from the [`VirtualParticlePattern`] row that produced this particle
/// and therefore sum to `1.0`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct VirtualParticle {
    /// The triangle's three real vertex indices.
    pub verts: [u32; 3],
    /// The barycentric weights of this sample on `verts` (sum `1.0`).
    pub weights: [Real; 3],
}

/// Seeds every triangle with the pattern's virtual particles.
///
/// Emits one [`VirtualParticle`] per `(triangle, pattern row)` pair, so the
/// result has length `triangles.len() * pattern.len()` for meshes of
/// non-degenerate triangles. Degenerate triangles (any two corner indices
/// equal) are skipped: their zero-area face cannot host a meaningful
/// barycentric sample and would break the mass-conserving scatter, so they
/// contribute no virtual particles and the result may be shorter.
#[must_use]
pub fn generate_virtual_particles(
    triangles: &[[u32; 3]],
    pattern: &VirtualParticlePattern,
) -> Vec<VirtualParticle> {
    let mut out = Vec::with_capacity(triangles.len().saturating_mul(pattern.len()));
    for &tri in triangles {
        if tri[0] == tri[1] || tri[1] == tri[2] || tri[0] == tri[2] {
            continue;
        }
        for &weights in pattern.rows() {
            out.push(VirtualParticle {
                verts: tri,
                weights,
            });
        }
    }
    out
}

/// A unified collision sample: either a real particle or a virtual particle.
///
/// Real particles are encoded as `verts = [i, i, i]`, `weights = [1, 0, 0]` so
/// they flow through the exact same position/inverse-mass/scatter math as
/// virtual particles: their single active vertex is `i`, their sample position
/// is `position[i]`, and a correction scatters entirely back onto `i`.
#[derive(Clone, Copy)]
struct Sample {
    verts: [u32; 3],
    weights: [Real; 3],
}

impl Sample {
    /// The real-particle sample for vertex `index`.
    fn real(index: u32) -> Self {
        Self {
            verts: [index, index, index],
            weights: [1.0, 0.0, 0.0],
        }
    }

    /// The virtual-particle sample for `vp`.
    fn virtual_particle(vp: VirtualParticle) -> Self {
        Self {
            verts: vp.verts,
            weights: vp.weights,
        }
    }

    /// The live world position `Σ weights[k] * position[verts[k]]`.
    fn position(&self, positions: &[Vec3]) -> Vec3 {
        let mut pos = Vec3::ZERO;
        for k in 0..3 {
            let w = self.weights[k];
            if w == 0.0 {
                continue;
            }
            pos += positions[self.verts[k] as usize] * w;
        }
        pos
    }

    /// The effective inverse mass `Σ weights[k]^2 * inverse_mass[verts[k]]`.
    ///
    /// This is the inverse mass the sample presents to a normal push: moving
    /// the sample by `dP` costs the least energy when each vertex `k` moves by
    /// `(weights[k] * inverse_mass / eff) * dP`, and the resulting sample
    /// displacement is exactly `dP` (see [`Sample::scatter`]).
    fn inverse_mass_eff(&self, inverse_masses: &[Real]) -> Real {
        let mut eff = 0.0;
        for k in 0..3 {
            let w = self.weights[k];
            if w == 0.0 {
                continue;
            }
            let im = inverse_masses[self.verts[k] as usize].max(0.0);
            eff += w * w * im;
        }
        eff
    }

    /// Scatters a sample-space displacement `dp` back onto the real vertices.
    ///
    /// Each active vertex `k` receives `(weights[k] * inverse_mass / eff) * dp`,
    /// which is the mass-weighted distribution whose weighted sum reproduces
    /// `dp` at the sample. A sample whose active vertices are all pinned has
    /// `eff <= 0` and does not move.
    fn scatter(&self, positions: &mut [Vec3], inverse_masses: &[Real], dp: Vec3) {
        let eff = self.inverse_mass_eff(inverse_masses);
        if eff <= 0.0 {
            return;
        }
        for k in 0..3 {
            let w = self.weights[k];
            if w == 0.0 {
                continue;
            }
            let j = self.verts[k] as usize;
            let im = inverse_masses[j].max(0.0);
            if im <= 0.0 {
                continue;
            }
            let coeff = w * im / eff;
            positions[j] += dp * coeff;
        }
    }
}

/// Returns `true` when two samples share any active (positive-weight) vertex.
///
/// Incident samples — a vertex against a triangle it belongs to, or two
/// triangles sharing an edge or corner — must not be separated, because the
/// spatial hash would otherwise fight the mesh's own topology. Only the active
/// vertices (weight `> 0`) participate, so a real particle `[i, i, i]` with
/// weights `[1, 0, 0]` counts as touching just vertex `i`.
fn shares_active_vertex(a: &Sample, b: &Sample) -> bool {
    for ka in 0..3 {
        if a.weights[ka] <= 0.0 {
            continue;
        }
        for kb in 0..3 {
            if b.weights[kb] <= 0.0 {
                continue;
            }
            if a.verts[ka] == b.verts[kb] {
                return true;
            }
        }
    }
    false
}

/// Which sample pairs a virtual-particle self-collision sweep resolves.
#[derive(Clone, Copy, PartialEq, Eq)]
enum PairScope {
    /// Every pair, including real-vertex versus real-vertex, so the sweep is a
    /// self-contained self-collision tier.
    All,
    /// Only pairs where at least one sample is a virtual particle, so the sweep
    /// augments an existing point-to-point pass without re-resolving (and
    /// thereby stripping the friction from) its real-vertex pairs.
    VirtualOnly,
}

/// Runs the full virtual-particle self-collision tier (see [`resolve_core`]).
///
/// This resolves every sample pair — real-vertex versus real-vertex included —
/// so with an empty `virtuals` slice it reduces exactly to the point-to-point
/// [`resolve_self_collision`](super::resolve_self_collision), and with virtual
/// particles it additionally catches vertices tunnelling through triangle
/// interiors. Use this when the virtual tier is the *only* self-collision pass;
/// use [`resolve_self_collision_virtual_augment`] to layer it on top of the
/// friction point-to-point pass.
///
/// `positions` is the particle position column and `inverse_masses` the
/// index-aligned inverse-mass column (`0` marks a pinned particle). A
/// non-positive `cell_size` or `thickness`, an `inverse_masses` slice whose
/// length differs from `positions`, or fewer than two samples is a no-op.
pub fn resolve_self_collision_virtual(
    positions: &mut [Vec3],
    inverse_masses: &[Real],
    virtuals: &[VirtualParticle],
    cell_size: Real,
    thickness: Real,
) {
    resolve_core(
        positions,
        inverse_masses,
        virtuals,
        cell_size,
        thickness,
        PairScope::All,
    );
}

/// Augments an existing point-to-point self-collision pass with virtual
/// particles, resolving only pairs where at least one sample is virtual.
///
/// The friction point-to-point tier
/// ([`resolve_self_collision_with_friction`](super::resolve_self_collision_with_friction))
/// already separates and rubs real-vertex pairs, so re-resolving them here
/// would undo their tangential friction. This pass therefore skips real-vertex
/// versus real-vertex pairs and adds only the vertex-versus-face and
/// face-versus-face coverage the point tier cannot see. An empty `virtuals`
/// slice is a no-op.
pub fn resolve_self_collision_virtual_augment(
    positions: &mut [Vec3],
    inverse_masses: &[Real],
    virtuals: &[VirtualParticle],
    cell_size: Real,
    thickness: Real,
) {
    resolve_core(
        positions,
        inverse_masses,
        virtuals,
        cell_size,
        thickness,
        PairScope::VirtualOnly,
    );
}

/// Resolves cloth self-collision with `NvCloth`-style virtual particles.
///
/// Real particles (`0..positions.len()`) and the supplied `virtuals` are folded
/// into one sample list — reals first, then virtuals in order — and bucketed
/// into a deterministic uniform spatial hash of side `cell_size`. Each sample
/// tests only its 27-cell neighborhood, every unordered pair is visited once
/// (`b > a`), and a pair closer than `thickness` is separated along the line
/// joining the two sample positions, split by effective inverse mass. The
/// per-sample correction is then scattered back onto that sample's real
/// vertices by its barycentric weights, so a virtual particle's push moves the
/// three triangle corners rather than a phantom point.
///
/// Pairs that share an active vertex are skipped (a triangle never fights its
/// own or an adjacent triangle's corners). Coincident samples separate along
/// `+X`. Pinned vertices never move; if both samples of a pair are immovable
/// the pair is a no-op. A non-positive `cell_size` or `thickness`, a mismatched
/// `inverse_masses` length, or fewer than two samples, is a no-op.
fn resolve_core(
    positions: &mut [Vec3],
    inverse_masses: &[Real],
    virtuals: &[VirtualParticle],
    cell_size: Real,
    thickness: Real,
    scope: PairScope,
) {
    if cell_size <= 0.0 || thickness <= 0.0 || inverse_masses.len() != positions.len() {
        return;
    }
    let real_count = positions.len();

    // Fixed sample order: every real particle first, then the virtual
    // particles in generation order. Out-of-range virtual particles are
    // dropped so the scatter never indexes past the vertex buffer.
    let mut samples: Vec<Sample> = Vec::with_capacity(real_count.saturating_add(virtuals.len()));
    for i in 0..real_count {
        samples.push(Sample::real(i as u32));
    }
    for &vp in virtuals {
        let in_range = vp.verts.iter().all(|&v| (v as usize) < real_count);
        if in_range {
            samples.push(Sample::virtual_particle(vp));
        }
    }
    if samples.len() < 2 {
        return;
    }

    // Bucket samples by their initial position. Indices are pushed in ascending
    // order, so both the cell traversal and per-bucket traversal are stable.
    let mut grid: BTreeMap<(i32, i32, i32), Vec<u32>> = BTreeMap::new();
    for (index, sample) in samples.iter().enumerate() {
        let cell = cell_of(sample.position(positions), cell_size);
        grid.entry(cell).or_default().push(index as u32);
    }

    let thickness_sq = thickness * thickness;
    for (&cell, bucket) in &grid {
        for &a in bucket {
            let ai = a as usize;
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        let neighbor = (cell.0 + dx, cell.1 + dy, cell.2 + dz);
                        let Some(nbucket) = grid.get(&neighbor) else {
                            continue;
                        };
                        for &b in nbucket {
                            if b <= a {
                                continue;
                            }
                            let bi = b as usize;
                            if scope == PairScope::VirtualOnly
                                && ai < real_count
                                && bi < real_count
                            {
                                // Both samples are real vertices; the friction
                                // point-to-point tier already resolved this
                                // pair, so the augment sweep skips it.
                                continue;
                            }
                            resolve_sample_pair(
                                positions,
                                inverse_masses,
                                &samples,
                                ai,
                                bi,
                                thickness,
                                thickness_sq,
                            );
                        }
                    }
                }
            }
        }
    }
}

/// Separates sample pair `(ai, bi)` if closer than `thickness`.
///
/// Positions are recomputed from the live vertex buffer (Gauss-Seidel), the
/// penetration is split by effective inverse mass so pinned samples stay put,
/// and each half is scattered back onto its real vertices. Incident pairs and
/// zero-mobility pairs are skipped; coincident samples separate along `+X`.
fn resolve_sample_pair(
    positions: &mut [Vec3],
    inverse_masses: &[Real],
    samples: &[Sample],
    ai: usize,
    bi: usize,
    thickness: Real,
    thickness_sq: Real,
) {
    let sample_a = samples[ai];
    let sample_b = samples[bi];
    if shares_active_vertex(&sample_a, &sample_b) {
        return;
    }

    let pa = sample_a.position(positions);
    let pb = sample_b.position(positions);
    let delta = pb - pa;
    let dist_sq = delta.length_squared();
    if dist_sq >= thickness_sq {
        return;
    }

    let wa = sample_a.inverse_mass_eff(inverse_masses);
    let wb = sample_b.inverse_mass_eff(inverse_masses);
    let w_sum = wa + wb;
    if w_sum <= 0.0 {
        // Both samples immovable (all active vertices pinned).
        return;
    }

    let (dir, penetration) = if dist_sq <= EPS_LEN_SQ {
        // Coincident samples: separate along a fixed axis by the full
        // thickness so the result is deterministic and never `NaN`.
        (Vec3::new(1.0, 0.0, 0.0), thickness)
    } else {
        let dist = dist_sq.sqrt();
        (delta * (1.0 / dist), thickness - dist)
    };

    // `dir` points from A toward B; push the samples apart along it, weighted
    // so the lighter (larger inverse mass) sample yields more.
    let dp_a = dir * (-penetration * (wa / w_sum));
    let dp_b = dir * (penetration * (wb / w_sum));
    sample_a.scatter(positions, inverse_masses, dp_a);
    sample_b.scatter(positions, inverse_masses, dp_b);
}

#[cfg(test)]
mod tests {
    use super::super::resolve_self_collision;
    use super::*;
    use alloc::vec;

    #[test]
    fn default_pattern_rows_are_normalized_and_non_negative() {
        let pattern = VirtualParticlePattern::nvcloth_default();
        assert_eq!(pattern.len(), 4);
        for row in pattern.rows() {
            let sum: Real = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6, "row {row:?} sums to {sum}");
            assert!(row.iter().all(|&w| w >= 0.0), "row {row:?} has a negative");
        }
    }

    #[test]
    fn sanitize_clamps_negatives_and_renormalizes() {
        let pattern = VirtualParticlePattern::from_weights(&[[2.0, -1.0, 0.0]]);
        let row = pattern.rows()[0];
        // The negative is dropped, the rest renormalized: [2, 0, 0] -> [1, 0, 0].
        assert_eq!(row, [1.0, 0.0, 0.0]);
    }

    #[test]
    fn sanitize_falls_back_to_centroid_for_degenerate_rows() {
        let pattern = VirtualParticlePattern::from_weights(&[
            [0.0, 0.0, 0.0],
            [Real::NAN, -1.0, Real::INFINITY],
        ]);
        for row in pattern.rows() {
            assert_eq!(*row, [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]);
        }
    }

    #[test]
    fn generate_emits_one_particle_per_triangle_row() {
        let pattern = VirtualParticlePattern::nvcloth_default();
        let triangles = [[0u32, 1, 2], [1, 2, 3]];
        let virtuals = generate_virtual_particles(&triangles, &pattern);
        assert_eq!(virtuals.len(), triangles.len() * pattern.len());
        assert_eq!(virtuals[0].verts, [0, 1, 2]);
        assert_eq!(virtuals[pattern.len()].verts, [1, 2, 3]);
    }

    #[test]
    fn generate_skips_degenerate_triangles() {
        let pattern = VirtualParticlePattern::nvcloth_default();
        let triangles = [[0u32, 0, 1], [2, 3, 4]];
        let virtuals = generate_virtual_particles(&triangles, &pattern);
        // Only the second, non-degenerate triangle contributes.
        assert_eq!(virtuals.len(), pattern.len());
        assert!(virtuals.iter().all(|v| v.verts == [2, 3, 4]));
    }

    /// The headline case: a vertex diving through a large triangle's interior
    /// is missed by the point-to-point tier but caught by a virtual particle.
    #[test]
    fn virtual_particle_catches_a_vertex_through_a_triangle() {
        // Big triangle (indices 0,1,2) in the z=0 plane, intruder (index 3)
        // just above its centroid, well inside `thickness` of the face but far
        // from every corner vertex.
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(4.0, 0.0, 0.0),
            Vec3::new(0.0, 4.0, 0.0),
            Vec3::new(4.0 / 3.0, 4.0 / 3.0, 0.05),
        ];
        let inverse_masses = vec![1.0, 1.0, 1.0, 1.0];
        let thickness = 0.2;
        let cell_size = 1.0;

        // Point-to-point tier: the intruder is far from all corners, so it
        // never moves.
        let mut plain = positions.clone();
        resolve_self_collision(&mut plain, &inverse_masses, cell_size, thickness);
        assert_eq!(plain[3], positions[3]);

        // Virtual tier: the centroid virtual particle sits under the intruder
        // and pushes it back out along +z.
        let mut virt = positions.clone();
        let pattern = VirtualParticlePattern::nvcloth_default();
        let virtuals = generate_virtual_particles(&[[0, 1, 2]], &pattern);
        resolve_self_collision_virtual(&mut virt, &inverse_masses, &virtuals, cell_size, thickness);
        assert!(
            virt[3].z > positions[3].z + 1e-4,
            "intruder z should be pushed out, got {}",
            virt[3].z
        );
    }

    #[test]
    fn pinned_triangle_only_moves_the_intruder() {
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(4.0, 0.0, 0.0),
            Vec3::new(0.0, 4.0, 0.0),
            Vec3::new(4.0 / 3.0, 4.0 / 3.0, 0.05),
        ];
        let inverse_masses = vec![0.0, 0.0, 0.0, 1.0];
        let thickness = 0.2;
        let mut particles = positions.clone();
        let virtuals =
            generate_virtual_particles(&[[0, 1, 2]], &VirtualParticlePattern::nvcloth_default());
        resolve_self_collision_virtual(&mut particles, &inverse_masses, &virtuals, 1.0, thickness);

        // The pinned corners are untouched.
        for i in 0..3 {
            assert_eq!(particles[i], positions[i]);
        }
        // The intruder absorbs the whole correction, ending a full thickness out.
        assert!((particles[3].z - thickness).abs() < 1e-5);
    }

    #[test]
    fn correction_scatters_onto_all_three_triangle_vertices() {
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(4.0, 0.0, 0.0),
            Vec3::new(0.0, 4.0, 0.0),
            Vec3::new(4.0 / 3.0, 4.0 / 3.0, 0.05),
        ];
        let inverse_masses = vec![1.0, 1.0, 1.0, 1.0];
        let mut particles = positions.clone();
        let virtuals =
            generate_virtual_particles(&[[0, 1, 2]], &VirtualParticlePattern::nvcloth_default());
        resolve_self_collision_virtual(&mut particles, &inverse_masses, &virtuals, 1.0, 0.2);

        // Every free corner shares the reaction, so all three move in -z.
        for i in 0..3 {
            assert!(
                particles[i].z < positions[i].z - 1e-6,
                "corner {i} should recoil in -z, got {}",
                particles[i].z
            );
        }
        assert!(particles[3].z > positions[3].z);
    }

    #[test]
    fn incident_vertex_of_the_triangle_is_not_separated() {
        // Vertex 0 is a corner of the triangle, so its real sample shares an
        // active vertex with every one of the triangle's virtual particles and
        // must never be pushed by them.
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(4.0, 0.0, 0.0),
            Vec3::new(0.0, 4.0, 0.0),
        ];
        let inverse_masses = vec![1.0, 1.0, 1.0];
        let mut particles = positions.clone();
        let virtuals =
            generate_virtual_particles(&[[0, 1, 2]], &VirtualParticlePattern::nvcloth_default());
        resolve_self_collision_virtual(&mut particles, &inverse_masses, &virtuals, 4.0, 0.5);
        for i in 0..3 {
            assert_eq!(particles[i], positions[i]);
        }
    }

    #[test]
    fn two_virtual_layers_separate_and_scatter_to_disjoint_vertices() {
        // Two parallel triangles (disjoint vertex sets 0..3 and 3..6) stacked
        // 0.05 apart in z: their centroid virtual particles collide and push
        // the two faces apart.
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(4.0, 0.0, 0.0),
            Vec3::new(0.0, 4.0, 0.0),
            Vec3::new(0.0, 0.0, 0.05),
            Vec3::new(4.0, 0.0, 0.05),
            Vec3::new(0.0, 4.0, 0.05),
        ];
        let inverse_masses = vec![1.0; 6];
        let mut particles = positions.clone();
        let virtuals = generate_virtual_particles(
            &[[0, 1, 2], [3, 4, 5]],
            &VirtualParticlePattern::nvcloth_default(),
        );
        resolve_self_collision_virtual(&mut particles, &inverse_masses, &virtuals, 1.0, 0.2);

        // Lower face recoils to -z, upper face to +z: the gap widens.
        for i in 0..3 {
            assert!(particles[i].z < positions[i].z - 1e-6);
        }
        for i in 3..6 {
            assert!(particles[i].z > positions[i].z + 1e-6);
        }
    }

    #[test]
    fn empty_virtuals_matches_point_to_point_self_collision() {
        // With no virtual particles the virtual tier must reduce exactly to the
        // point-to-point solver on a pair of near-coincident free vertices.
        let positions = vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.05, 0.0, 0.0)];
        let inverse_masses = vec![1.0, 1.0];
        let mut plain = positions.clone();
        resolve_self_collision(&mut plain, &inverse_masses, 1.0, 0.2);

        let mut virt = positions.clone();
        resolve_self_collision_virtual(&mut virt, &inverse_masses, &[], 1.0, 0.2);

        for i in 0..2 {
            assert_eq!(virt[i], plain[i]);
        }
    }

    #[test]
    fn augment_skips_real_real_but_catches_face_penetration() {
        // Triangle 0,1,2 with an intruder (3) above its centroid, plus a bare
        // pair of near-coincident free vertices (4,5) that belong to no
        // triangle. The augment sweep must push the intruder out (real-vs-face)
        // yet leave the bare real-vs-real pair to the point-to-point tier.
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(4.0, 0.0, 0.0),
            Vec3::new(0.0, 4.0, 0.0),
            Vec3::new(4.0 / 3.0, 4.0 / 3.0, 0.05),
            Vec3::new(10.0, 10.0, 10.0),
            Vec3::new(10.03, 10.0, 10.0),
        ];
        let inverse_masses = vec![1.0; 6];
        let mut particles = positions.clone();
        let virtuals =
            generate_virtual_particles(&[[0, 1, 2]], &VirtualParticlePattern::nvcloth_default());
        resolve_self_collision_virtual_augment(&mut particles, &inverse_masses, &virtuals, 1.0, 0.2);

        // Face penetration resolved: the intruder is pushed out along +z.
        assert!(particles[3].z > positions[3].z + 1e-4);
        // The bare real-vertex pair is untouched by the augment sweep.
        assert_eq!(particles[4], positions[4]);
        assert_eq!(particles[5], positions[5]);
    }

    #[test]
    fn augment_with_empty_virtuals_is_a_full_noop() {
        // No virtual particles means no pair involves a virtual sample, so the
        // augment sweep must not touch even a colliding real-vertex pair.
        let positions = vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.05, 0.0, 0.0)];
        let inverse_masses = vec![1.0, 1.0];
        let mut particles = positions.clone();
        resolve_self_collision_virtual_augment(&mut particles, &inverse_masses, &[], 1.0, 0.2);
        assert_eq!(particles, positions);
    }

    #[test]
    fn resolution_is_deterministic() {
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(4.0, 0.0, 0.0),
            Vec3::new(0.0, 4.0, 0.0),
            Vec3::new(4.0 / 3.0, 4.0 / 3.0, 0.05),
        ];
        let inverse_masses = vec![1.0; 4];
        let virtuals =
            generate_virtual_particles(&[[0, 1, 2]], &VirtualParticlePattern::nvcloth_default());

        let mut first = positions.clone();
        resolve_self_collision_virtual(&mut first, &inverse_masses, &virtuals, 1.0, 0.2);
        let mut second = positions.clone();
        resolve_self_collision_virtual(&mut second, &inverse_masses, &virtuals, 1.0, 0.2);

        for i in 0..4 {
            assert_eq!(first[i], second[i]);
        }
    }

    #[test]
    fn non_positive_parameters_and_short_input_are_noops() {
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(4.0, 0.0, 0.0),
            Vec3::new(0.0, 4.0, 0.0),
            Vec3::new(4.0 / 3.0, 4.0 / 3.0, 0.05),
        ];
        let inverse_masses = vec![1.0; 4];
        let virtuals =
            generate_virtual_particles(&[[0, 1, 2]], &VirtualParticlePattern::nvcloth_default());

        let mut zero_cell = positions.clone();
        resolve_self_collision_virtual(&mut zero_cell, &inverse_masses, &virtuals, 0.0, 0.2);
        assert_eq!(zero_cell, positions);

        let mut zero_thickness = positions.clone();
        resolve_self_collision_virtual(&mut zero_thickness, &inverse_masses, &virtuals, 1.0, 0.0);
        assert_eq!(zero_thickness, positions);

        let mut single = vec![Vec3::new(0.0, 0.0, 0.0)];
        resolve_self_collision_virtual(&mut single, &[1.0], &[], 1.0, 0.2);
        assert_eq!(single[0], Vec3::ZERO);
    }
}
