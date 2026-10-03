//! Volume-preserving mesh smoothing (Taubin lambda|mu).
//!
//! Cooked collision meshes are often noisy: scanned geometry, marching-cubes
//! output from a signed distance field, or decimated art all carry high-
//! frequency jitter that wastes contact-solver iterations and produces
//! unstable normals. Classic Laplacian smoothing removes that jitter but
//! shrinks the shape — every pass pulls each vertex toward the average of its
//! neighbours, so a convex surface slowly collapses inward.
//!
//! Taubin's lambda|mu smoothing fixes the shrinkage. Each iteration applies a
//! shrinking Laplacian step with a positive factor `lambda`, then an inflating
//! step with a negative factor `mu` whose magnitude is slightly larger. The
//! pair acts as a low-pass filter on the mesh signal: high-frequency noise is
//! attenuated while the low-frequency shape (and hence the enclosed volume) is
//! preserved. Setting `mu` to zero recovers plain Laplacian smoothing.
//!
//! Connectivity uses uniform (umbrella) weights built from the triangle edges,
//! which is robust to irregular triangulations and needs no cotangent areas.
//! Boundary vertices — those on an edge used by a single triangle — can be
//! pinned so open patches keep their silhouette.
//!
//! This is pure triangle-soup geometry with no coupling to the collision
//! pipeline, and nothing here is derived from Unreal Engine source.

use alloc::collections::{BTreeMap, BTreeSet};

use glam::Vec3;

/// Parameters controlling [`taubin_smooth`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SmoothingParams {
    /// Number of lambda|mu iterations to apply.
    pub iterations: u32,
    /// Positive shrinking factor, typically around `0.33`.
    pub lambda: f32,
    /// Negative inflating factor, typically around `-0.34`. Set to `0.0` for
    /// plain Laplacian smoothing (which shrinks the mesh).
    pub mu: f32,
    /// When `true`, boundary vertices are held fixed so open meshes keep their
    /// outline.
    pub pin_boundary: bool,
}

impl Default for SmoothingParams {
    fn default() -> Self {
        Self {
            iterations: 10,
            lambda: 0.33,
            mu: -0.34,
            pin_boundary: true,
        }
    }
}

/// Smooth a triangle mesh with Taubin's lambda|mu filter.
///
/// `positions` and `indices` describe a triangle mesh; the returned vector has
/// one smoothed position per input vertex, in the same order. Returns [`None`]
/// when the mesh is empty, an index is out of range, or the parameters are not
/// finite.
///
/// The result is deterministic for a given input.
#[must_use]
pub fn taubin_smooth(
    positions: &[Vec3],
    indices: &[[u32; 3]],
    params: &SmoothingParams,
) -> Option<Vec<Vec3>> {
    if positions.is_empty() || indices.is_empty() {
        return None;
    }
    if !(params.lambda.is_finite() && params.mu.is_finite()) {
        return None;
    }
    let n = positions.len();
    for tri in indices {
        for &v in tri {
            if (v as usize) >= n {
                return None;
            }
        }
    }

    let adjacency = build_adjacency(indices, n);
    let pinned = if params.pin_boundary {
        boundary_flags(indices, n)
    } else {
        vec![false; n]
    };

    let mut current = positions.to_vec();
    let mut scratch = current.clone();
    for _ in 0..params.iterations {
        smooth_pass(&current, &adjacency, &pinned, params.lambda, &mut scratch);
        core::mem::swap(&mut current, &mut scratch);
        if params.mu != 0.0 {
            smooth_pass(&current, &adjacency, &pinned, params.mu, &mut scratch);
            core::mem::swap(&mut current, &mut scratch);
        }
    }
    Some(current)
}

/// Build the per-vertex neighbour lists from triangle edges.
fn build_adjacency(indices: &[[u32; 3]], vertex_count: usize) -> Vec<Vec<u32>> {
    let mut sets: Vec<BTreeSet<u32>> = vec![BTreeSet::new(); vertex_count];
    for tri in indices {
        for e in 0..3 {
            let a = tri[e];
            let b = tri[(e + 1) % 3];
            if a != b {
                sets[a as usize].insert(b);
                sets[b as usize].insert(a);
            }
        }
    }
    sets.into_iter().map(|s| s.into_iter().collect()).collect()
}

/// Flag vertices that touch a boundary edge (an edge used by one triangle).
fn boundary_flags(indices: &[[u32; 3]], vertex_count: usize) -> Vec<bool> {
    let mut edge_use: BTreeMap<(u32, u32), u32> = BTreeMap::new();
    for tri in indices {
        for e in 0..3 {
            let a = tri[e];
            let b = tri[(e + 1) % 3];
            if a == b {
                continue;
            }
            let key = if a < b { (a, b) } else { (b, a) };
            *edge_use.entry(key).or_insert(0) += 1;
        }
    }
    let mut flags = vec![false; vertex_count];
    for ((a, b), count) in edge_use {
        if count == 1 {
            flags[a as usize] = true;
            flags[b as usize] = true;
        }
    }
    flags
}

/// One umbrella-weighted smoothing pass with the given factor.
fn smooth_pass(
    positions: &[Vec3],
    adjacency: &[Vec<u32>],
    pinned: &[bool],
    factor: f32,
    out: &mut [Vec3],
) {
    for (i, position) in positions.iter().enumerate() {
        let neighbours = &adjacency[i];
        if pinned[i] || neighbours.is_empty() {
            out[i] = *position;
            continue;
        }
        let mut centroid = Vec3::ZERO;
        for &j in neighbours {
            centroid += positions[j as usize];
        }
        centroid /= neighbours.len() as f32;
        out[i] = *position + factor * (centroid - *position);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A flat `res x res` grid in the XZ plane with per-vertex y set by `height`.
    fn grid(res: usize, height: impl Fn(usize, usize) -> f32) -> (Vec<Vec3>, Vec<[u32; 3]>) {
        let mut verts = Vec::with_capacity(res * res);
        for z in 0..res {
            for x in 0..res {
                let fx = x as f32 / (res - 1) as f32;
                let fz = z as f32 / (res - 1) as f32;
                verts.push(Vec3::new(fx, height(x, z), fz));
            }
        }
        let mut tris = Vec::new();
        for z in 0..res - 1 {
            for x in 0..res - 1 {
                let i = (z * res + x) as u32;
                let right = i + 1;
                let down = i + res as u32;
                let diag = down + 1;
                tris.push([i, right, diag]);
                tris.push([i, diag, down]);
            }
        }
        (verts, tris)
    }

    /// A unit icosphere (icosahedron subdivided once, projected to the sphere).
    fn icosphere() -> (Vec<Vec3>, Vec<[u32; 3]>) {
        let t = (1.0 + 5.0_f32.sqrt()) / 2.0;
        let mut verts = vec![
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
        let faces: [[u32; 3]; 20] = [
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
        let mut midpoints: BTreeMap<(u32, u32), u32> = BTreeMap::new();
        let mut tris = Vec::new();
        let mut midpoint = |a: u32, b: u32, verts: &mut Vec<Vec3>| -> u32 {
            let key = if a < b { (a, b) } else { (b, a) };
            if let Some(&m) = midpoints.get(&key) {
                return m;
            }
            let mid = ((verts[a as usize] + verts[b as usize]) * 0.5).normalize();
            let idx = verts.len() as u32;
            verts.push(mid);
            midpoints.insert(key, idx);
            idx
        };
        for f in faces {
            let a = midpoint(f[0], f[1], &mut verts);
            let b = midpoint(f[1], f[2], &mut verts);
            let c = midpoint(f[2], f[0], &mut verts);
            tris.push([f[0], a, c]);
            tris.push([f[1], b, a]);
            tris.push([f[2], c, b]);
            tris.push([a, b, c]);
        }
        (verts, tris)
    }

    fn mean_radius(verts: &[Vec3]) -> f32 {
        let centroid = verts.iter().copied().sum::<Vec3>() / verts.len() as f32;
        verts.iter().map(|v| v.distance(centroid)).sum::<f32>() / verts.len() as f32
    }

    #[test]
    fn rejects_bad_input() {
        let p = SmoothingParams::default();
        assert!(taubin_smooth(&[], &[[0, 1, 2]], &p).is_none());
        assert!(taubin_smooth(&[Vec3::ZERO; 3], &[], &p).is_none());
        // Out-of-range index.
        assert!(taubin_smooth(&[Vec3::ZERO; 3], &[[0, 1, 9]], &p).is_none());
        // Non-finite parameters.
        let bad = SmoothingParams {
            lambda: f32::NAN,
            ..p
        };
        assert!(taubin_smooth(&[Vec3::ZERO; 3], &[[0, 1, 2]], &bad).is_none());
    }

    #[test]
    fn flat_grid_is_left_essentially_unchanged() {
        let (verts, tris) = grid(6, |_, _| 0.0);
        let out = taubin_smooth(&verts, &tris, &SmoothingParams::default()).expect("smooths");
        let max_shift = verts
            .iter()
            .zip(&out)
            .map(|(a, b)| a.distance(*b))
            .fold(0.0_f32, f32::max);
        assert!(max_shift < 1.0e-4, "flat grid moved by {max_shift}");
    }

    #[test]
    fn interior_noise_is_attenuated() {
        // Deterministic alternating out-of-plane spikes on a flat grid.
        let res = 9;
        let (verts, tris) = grid(res, |x, z| if (x + z) % 2 == 0 { 0.2 } else { -0.2 });
        let out = taubin_smooth(&verts, &tris, &SmoothingParams::default()).expect("smooths");

        // Compare interior-vertex out-of-plane magnitude before and after.
        let is_interior = |x: usize, z: usize| x > 0 && z > 0 && x < res - 1 && z < res - 1;
        let mut before = 0.0_f32;
        let mut after = 0.0_f32;
        for z in 0..res {
            for x in 0..res {
                if is_interior(x, z) {
                    let i = z * res + x;
                    before += verts[i].y.abs();
                    after += out[i].y.abs();
                }
            }
        }
        assert!(
            after < 0.25 * before,
            "noise {before} -> {after} not attenuated"
        );
    }

    #[test]
    fn pinned_boundary_stays_fixed() {
        let res = 5;
        let (verts, tris) = grid(res, |x, z| if (x + z) % 2 == 0 { 0.1 } else { -0.1 });
        let out = taubin_smooth(&verts, &tris, &SmoothingParams::default()).expect("smooths");
        for z in 0..res {
            for x in 0..res {
                let on_boundary = x == 0 || z == 0 || x == res - 1 || z == res - 1;
                if on_boundary {
                    let i = z * res + x;
                    assert!(
                        verts[i].distance(out[i]) < 1.0e-6,
                        "boundary vertex {i} moved"
                    );
                }
            }
        }
    }

    #[test]
    fn taubin_preserves_volume_better_than_plain_laplacian() {
        let (verts, tris) = icosphere();
        let orig = mean_radius(&verts);

        let taubin = taubin_smooth(
            &verts,
            &tris,
            &SmoothingParams {
                iterations: 12,
                pin_boundary: false,
                ..SmoothingParams::default()
            },
        )
        .expect("taubin");
        let plain = taubin_smooth(
            &verts,
            &tris,
            &SmoothingParams {
                iterations: 12,
                lambda: 0.33,
                mu: 0.0,
                pin_boundary: false,
            },
        )
        .expect("laplacian");

        let r_taubin = mean_radius(&taubin);
        let r_plain = mean_radius(&plain);
        // Plain Laplacian shrinks a convex closed surface.
        assert!(
            r_plain < orig,
            "laplacian did not shrink: {r_plain} vs {orig}"
        );
        // Taubin preserves the radius far better.
        assert!(
            r_taubin > r_plain,
            "taubin {r_taubin} should shrink less than laplacian {r_plain}"
        );
        assert!(
            (r_taubin - orig).abs() < 0.05 * orig,
            "taubin radius {r_taubin} drifted from {orig}"
        );
    }

    #[test]
    fn result_is_deterministic() {
        let (verts, tris) = icosphere();
        let p = SmoothingParams {
            pin_boundary: false,
            ..SmoothingParams::default()
        };
        let a = taubin_smooth(&verts, &tris, &p).expect("smooths");
        let b = taubin_smooth(&verts, &tris, &p).expect("smooths");
        assert_eq!(a, b);
    }
}
