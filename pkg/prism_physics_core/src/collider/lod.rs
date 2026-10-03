//! Error-bounded collision-LOD chain generation.
//!
//! A cooked collision mesh is rarely used at a single resolution. Engines keep
//! a *chain* of progressively coarser proxies and pick the cheapest one whose
//! deviation from the original still fits the simulation's accuracy budget:
//! distant or fast-moving bodies query a coarse level while nearby resting
//! bodies query a fine one. `PhysX` cooks graded meshes, `Jolt` ships mesh LOD
//! groups, and Chaos drives collision detail off a Nanite-style cluster
//! hierarchy.
//!
//! [`build_lod_chain`] turns a single triangle soup into a [`MeshLodChain`]:
//! level 0 is the (compacted) base mesh and each deeper level is produced by
//! quadric-error [`decimate_mesh`](super::decimate::decimate_mesh) at a smaller
//! triangle budget. Every generated level records a measured *geometric error*
//! -- a symmetric, vertex-and-centroid-sampled approximation of the two-sided
//! Hausdorff distance to the base mesh, computed with [`MeshBvh`] closest-point
//! queries -- so callers can select a level by accuracy at runtime via
//! [`MeshLodChain::select_for_error`].
//!
//! The algorithm composes standard quadric decimation with a sampled Hausdorff
//! estimate; nothing here is derived from Unreal Engine source.

use super::decimate::{decimate_mesh, DecimateParams, DecimateTarget};
use super::mesh_bvh::MeshBvh;
use glam::Vec3;

/// Coarsest triangle count a generated level is allowed to target: a closed
/// surface cannot drop below a tetrahedron.
pub const MIN_LOD_TRIANGLES: usize = 4;

/// A single level of a [`MeshLodChain`].
#[derive(Clone, Debug, PartialEq)]
pub struct MeshLod {
    /// Vertex positions for this level, re-indexed from zero.
    pub vertices: Vec<Vec3>,
    /// Triangle indices (triples) into [`MeshLod::vertices`].
    pub indices: Vec<u32>,
    /// Symmetric, sampled Hausdorff distance from this level back to the base
    /// mesh, in world units. Level 0 (the base) always reports `0.0`.
    pub error: f32,
}

impl MeshLod {
    /// Number of triangles in this level.
    #[must_use]
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Number of vertices in this level.
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }
}

/// How the triangle budget decreases down a [`MeshLodChain`].
#[derive(Clone, Debug, PartialEq)]
pub enum LodSchedule {
    /// Explicit triangle budgets for the levels below the base, in the order
    /// they should appear. Values are clamped to `[MIN_LOD_TRIANGLES, base-1]`
    /// and reduced to a strictly decreasing sequence before use.
    TriangleCounts(Vec<usize>),
    /// Geometric decay: level `i` (for `i` in `1..=levels`) targets
    /// `base_triangles * factor^i`. `factor` is clamped to `(0, 1)` and the
    /// chain stops early once a target would fall below [`MIN_LOD_TRIANGLES`]
    /// or stops decreasing.
    Geometric {
        /// Per-level survival fraction, clamped to the open interval `(0, 1)`.
        factor: f32,
        /// Maximum number of levels to generate below the base.
        levels: usize,
    },
}

/// Parameters controlling [`build_lod_chain`].
#[derive(Clone, Debug)]
pub struct LodChainParams {
    /// The triangle-budget schedule for the generated levels.
    pub schedule: LodSchedule,
    /// Quadric-error ceiling forwarded to each [`decimate_mesh`] call. Use
    /// [`f32::INFINITY`] to decimate purely to the triangle budget.
    pub max_error: f32,
    /// Stop generating levels once a level's measured geometric error exceeds
    /// this ceiling (coarser levels only degrade further). Use
    /// [`f32::INFINITY`] to keep every scheduled level.
    pub max_geometric_error: f32,
}

impl LodChainParams {
    /// A geometric schedule of `levels` levels, each keeping `factor` of the
    /// previous triangle count, with no error ceilings.
    #[must_use]
    pub fn geometric(factor: f32, levels: usize) -> Self {
        Self {
            schedule: LodSchedule::Geometric { factor, levels },
            max_error: f32::INFINITY,
            max_geometric_error: f32::INFINITY,
        }
    }

    /// An explicit per-level triangle-budget schedule with no error ceilings.
    #[must_use]
    pub fn triangle_counts(counts: Vec<usize>) -> Self {
        Self {
            schedule: LodSchedule::TriangleCounts(counts),
            max_error: f32::INFINITY,
            max_geometric_error: f32::INFINITY,
        }
    }
}

/// A base mesh plus a sequence of progressively coarser, error-graded collision
/// proxies produced by [`build_lod_chain`].
#[derive(Clone, Debug, PartialEq)]
pub struct MeshLodChain {
    /// Levels ordered finest first: `levels[0]` is the base mesh and each
    /// subsequent level has fewer triangles and a larger geometric error.
    pub levels: Vec<MeshLod>,
}

impl MeshLodChain {
    /// Number of levels, including the base.
    #[must_use]
    pub fn len(&self) -> usize {
        self.levels.len()
    }

    /// Whether the chain is empty. A chain built by [`build_lod_chain`] always
    /// has at least the base level, so this only reports `true` for a chain
    /// constructed by hand.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.levels.is_empty()
    }

    /// The finest (base) level.
    #[must_use]
    pub fn base(&self) -> Option<&MeshLod> {
        self.levels.first()
    }

    /// The coarsest generated level.
    #[must_use]
    pub fn coarsest(&self) -> Option<&MeshLod> {
        self.levels.last()
    }

    /// Borrow a level by index, finest first.
    #[must_use]
    pub fn level(&self, index: usize) -> Option<&MeshLod> {
        self.levels.get(index)
    }

    /// Select the coarsest (cheapest) level whose geometric error does not
    /// exceed `max_error`, falling back to the base level when even it does
    /// not qualify (the base error is always `0.0`, so this only returns
    /// `None` for an empty chain).
    #[must_use]
    pub fn select_for_error(&self, max_error: f32) -> Option<&MeshLod> {
        let mut chosen = self.levels.first();
        for level in &self.levels {
            if level.error <= max_error {
                chosen = Some(level);
            } else {
                break;
            }
        }
        chosen
    }
}

/// Builds an error-graded [`MeshLodChain`] from a triangle soup.
///
/// Returns `None` when the input is empty, the index buffer is not a whole
/// number of triangles, an index is out of range, or the base mesh is too
/// degenerate to build an acceleration structure for error measurement.
#[must_use]
pub fn build_lod_chain(
    vertices: &[Vec3],
    indices: &[u32],
    params: &LodChainParams,
) -> Option<MeshLodChain> {
    // Compact and validate the base mesh by running a no-op decimation (a
    // target at or above the input count returns the mesh compacted but
    // otherwise unchanged).
    let base = decimate_mesh(
        vertices,
        indices,
        DecimateParams {
            target: DecimateTarget::TriangleCount(usize::MAX),
            max_error: f32::INFINITY,
        },
    )?;
    let base_tris = base.triangle_count();
    if base_tris == 0 {
        return None;
    }

    let base_triples = flat_to_triples(&base.indices);
    let base_bvh = MeshBvh::build(&base.vertices, &base_triples)?;
    let base_samples = sample_points(&base.vertices, &base_triples);

    let mut levels = Vec::new();
    levels.push(MeshLod {
        vertices: base.vertices.clone(),
        indices: base.indices.clone(),
        error: 0.0,
    });

    let targets = resolve_targets(&params.schedule, base_tris);
    for target in targets {
        let Some(dec) = decimate_mesh(
            &base.vertices,
            &base.indices,
            DecimateParams {
                target: DecimateTarget::TriangleCount(target),
                max_error: params.max_error,
            },
        ) else {
            continue;
        };
        if dec.triangle_count() == 0 || dec.triangle_count() >= base_tris {
            continue;
        }

        let triples = flat_to_triples(&dec.indices);
        let Some(bvh) = MeshBvh::build(&dec.vertices, &triples) else {
            continue;
        };
        let samples = sample_points(&dec.vertices, &triples);
        let forward = one_sided_error(&samples, &base_bvh);
        let backward = one_sided_error(&base_samples, &bvh);
        let error = forward.max(backward);

        if error > params.max_geometric_error {
            break;
        }

        levels.push(MeshLod {
            vertices: dec.vertices,
            indices: dec.indices,
            error,
        });
    }

    Some(MeshLodChain { levels })
}

/// Expands a schedule into a strictly decreasing list of triangle budgets, each
/// in `[MIN_LOD_TRIANGLES, base_tris - 1]`.
fn resolve_targets(schedule: &LodSchedule, base_tris: usize) -> Vec<usize> {
    let ceiling = base_tris.saturating_sub(1);
    if ceiling < MIN_LOD_TRIANGLES {
        return Vec::new();
    }

    let raw: Vec<usize> = match schedule {
        LodSchedule::TriangleCounts(counts) => counts.clone(),
        LodSchedule::Geometric { factor, levels } => {
            let f = factor.clamp(f32::MIN_POSITIVE, 1.0 - f32::EPSILON);
            let mut acc = base_tris as f32;
            let mut out = Vec::with_capacity(*levels);
            for _ in 0..*levels {
                acc *= f;
                out.push(acc.round() as usize);
            }
            out
        }
    };

    let mut targets = Vec::with_capacity(raw.len());
    let mut last = base_tris;
    for value in raw {
        let clamped = value.clamp(MIN_LOD_TRIANGLES, ceiling);
        if clamped < last {
            targets.push(clamped);
            last = clamped;
        }
    }
    targets
}

/// Groups a flat triangle-index buffer into triples for [`MeshBvh::build`].
fn flat_to_triples(indices: &[u32]) -> Vec<[u32; 3]> {
    indices
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect()
}

/// Builds the point set sampled when estimating geometric error: every vertex
/// plus every triangle centroid. Centroids catch large flat faces whose
/// vertices alone would understate the deviation.
fn sample_points(vertices: &[Vec3], triples: &[[u32; 3]]) -> Vec<Vec3> {
    let mut points = Vec::with_capacity(vertices.len() + triples.len());
    points.extend_from_slice(vertices);
    for &[a, b, c] in triples {
        let centroid = (vertices[a as usize] + vertices[b as usize] + vertices[c as usize]) / 3.0;
        points.push(centroid);
    }
    points
}

/// Largest distance from any sample point to the closest point on `target`.
fn one_sided_error(samples: &[Vec3], target: &MeshBvh) -> f32 {
    let mut worst = 0.0f32;
    for &p in samples {
        if let Some(hit) = target.closest_point(p) {
            worst = worst.max(hit.distance_sq.sqrt());
        }
    }
    worst
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Builds a closed, welded icosahedron subdivided `levels` times, i.e. a
    /// watertight 2-manifold unit sphere.
    fn icosphere(levels: u32) -> (Vec<Vec3>, Vec<u32>) {
        let t = (1.0 + 5.0_f32.sqrt()) * 0.5;
        let mut verts: Vec<Vec3> = vec![
            Vec3::new(-1.0, t, 0.0),
            Vec3::new(1.0, t, 0.0),
            Vec3::new(-1.0, -t, 0.0),
            Vec3::new(1.0, -t, 0.0),
            Vec3::new(0.0, -1.0, t),
            Vec3::new(0.0, 1.0, t),
            Vec3::new(0.0, -1.0, -t),
            Vec3::new(0.0, 1.0, -t),
            Vec3::new(t, 0.0, -1.0),
            Vec3::new(t, 0.0, 1.0),
            Vec3::new(-t, 0.0, -1.0),
            Vec3::new(-t, 0.0, 1.0),
        ];
        for v in &mut verts {
            *v = v.normalize();
        }
        let mut faces: Vec<[u32; 3]> = vec![
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ];
        for _ in 0..levels {
            let mut cache: HashMap<(u32, u32), u32> = HashMap::new();
            let mut next: Vec<[u32; 3]> = Vec::with_capacity(faces.len() * 4);
            let mut midpoint = |a: u32, b: u32, verts: &mut Vec<Vec3>| -> u32 {
                let key = if a < b { (a, b) } else { (b, a) };
                if let Some(&m) = cache.get(&key) {
                    return m;
                }
                let m = verts.len() as u32;
                let p = ((verts[a as usize] + verts[b as usize]) * 0.5).normalize();
                verts.push(p);
                cache.insert(key, m);
                m
            };
            for f in &faces {
                let a = midpoint(f[0], f[1], &mut verts);
                let b = midpoint(f[1], f[2], &mut verts);
                let c = midpoint(f[2], f[0], &mut verts);
                next.push([f[0], a, c]);
                next.push([f[1], b, a]);
                next.push([f[2], c, b]);
                next.push([a, b, c]);
            }
            faces = next;
        }
        let indices: Vec<u32> = faces.into_iter().flatten().collect();
        (verts, indices)
    }

    #[test]
    fn rejects_bad_input() {
        let p = LodChainParams::geometric(0.5, 3);
        assert!(build_lod_chain(&[], &[0, 1, 2], &p).is_none());
        let v = vec![Vec3::ZERO, Vec3::X, Vec3::Y];
        assert!(build_lod_chain(&v, &[], &p).is_none());
        assert!(build_lod_chain(&v, &[0, 1], &p).is_none());
        assert!(build_lod_chain(&v, &[0, 1, 9], &p).is_none());
    }

    #[test]
    fn base_level_is_exact_and_first() {
        let (v, i) = icosphere(2); // 320 triangles
        let chain = build_lod_chain(&v, &i, &LodChainParams::geometric(0.5, 3)).expect("chain");
        let base = chain.base().expect("base");
        assert_eq!(base.error, 0.0);
        assert_eq!(base.triangle_count(), i.len() / 3);
    }

    #[test]
    fn geometric_chain_is_strictly_coarser_and_monotone() {
        let (v, i) = icosphere(3); // 1280 triangles
        let chain = build_lod_chain(&v, &i, &LodChainParams::geometric(0.5, 4)).expect("chain");
        assert!(
            chain.len() >= 2,
            "expected generated levels, got {}",
            chain.len()
        );

        let mut prev_tris = chain.base().unwrap().triangle_count();
        let mut prev_err = 0.0f32;
        for level in chain.levels.iter().skip(1) {
            assert!(
                level.triangle_count() < prev_tris,
                "triangle count did not decrease: {} then {}",
                prev_tris,
                level.triangle_count()
            );
            // Fewer triangles can only deviate further from the original.
            assert!(
                level.error + 1.0e-4 >= prev_err,
                "error regressed: {prev_err} then {}",
                level.error
            );
            prev_tris = level.triangle_count();
            prev_err = level.error;
        }
    }

    #[test]
    fn sphere_errors_stay_below_radius() {
        let (v, i) = icosphere(3);
        let chain = build_lod_chain(&v, &i, &LodChainParams::geometric(0.5, 4)).expect("chain");
        // A unit sphere's coarsest proxy should still hug the surface well under
        // its radius; a blown-up error would mean decimation lost the shape.
        for level in &chain.levels {
            assert!(
                level.error < 1.0,
                "level with {} tris deviates {} (>= unit radius)",
                level.triangle_count(),
                level.error
            );
        }
    }

    #[test]
    fn select_for_error_trades_accuracy_for_cost() {
        let (v, i) = icosphere(3);
        let chain = build_lod_chain(&v, &i, &LodChainParams::geometric(0.5, 4)).expect("chain");

        // A zero budget forces the exact base level.
        let exact = chain.select_for_error(0.0).expect("exact");
        assert_eq!(exact.error, 0.0);
        assert_eq!(
            exact.triangle_count(),
            chain.base().unwrap().triangle_count()
        );

        // A generous budget picks the coarsest (cheapest) level.
        let coarse = chain.select_for_error(f32::INFINITY).expect("coarse");
        assert_eq!(
            coarse.triangle_count(),
            chain.coarsest().unwrap().triangle_count()
        );
    }

    #[test]
    fn explicit_counts_are_clamped_and_decreasing() {
        let (v, i) = icosphere(2); // 320 triangles
                                   // Mixed order with an out-of-range entry; resolver must clamp and keep a
                                   // strictly decreasing sequence.
        let params = LodChainParams::triangle_counts(vec![160, 160, 80, 1, 40]);
        let chain = build_lod_chain(&v, &i, &params).expect("chain");
        let tris: Vec<usize> = chain.levels.iter().map(MeshLod::triangle_count).collect();
        for w in tris.windows(2) {
            assert!(w[1] < w[0], "levels not strictly decreasing: {tris:?}");
        }
        // The `1` entry was clamped up to MIN_LOD_TRIANGLES.
        assert!(tris.last().copied().unwrap() >= MIN_LOD_TRIANGLES);
    }

    #[test]
    fn max_geometric_error_truncates_chain() {
        let (v, i) = icosphere(3);
        let unbounded = build_lod_chain(&v, &i, &LodChainParams::geometric(0.5, 5)).expect("chain");
        let budget = unbounded
            .levels
            .get(1)
            .map(|l| l.error)
            .expect("at least one generated level");

        let mut params = LodChainParams::geometric(0.5, 5);
        // Allow only levels at least as accurate as the first generated level.
        params.max_geometric_error = budget;
        let bounded = build_lod_chain(&v, &i, &params).expect("chain");

        assert!(bounded.len() <= unbounded.len());
        for level in &bounded.levels {
            assert!(
                level.error <= budget + 1.0e-5,
                "kept over-budget level {}",
                level.error
            );
        }
    }

    #[test]
    fn single_triangle_has_no_coarser_levels() {
        let v = vec![Vec3::ZERO, Vec3::X, Vec3::Y];
        let chain =
            build_lod_chain(&v, &[0, 1, 2], &LodChainParams::geometric(0.5, 3)).expect("chain");
        assert_eq!(chain.len(), 1);
        assert_eq!(chain.base().unwrap().triangle_count(), 1);
    }
}
