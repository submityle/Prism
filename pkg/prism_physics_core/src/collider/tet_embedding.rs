//! Barycentric embedding of a point set inside a tetrahedral cage mesh.
//!
//! A common soft-body workflow drives a detailed render surface with a coarse
//! tetrahedral "cage": each render vertex is bound once, in the rest pose, to
//! the tetrahedron that contains it, storing the four barycentric weights of
//! that containment. When the cage later deforms, every bound point is
//! reconstructed as the same barycentric blend of the (now moved) cage
//! vertices. Because barycentric interpolation is affine, a point rigidly
//! transformed with its cage follows exactly, and a smoothly deformed cage
//! carries its embedded geometry without tearing.
//!
//! Points that fall outside every tetrahedron (slightly proud of the cage,
//! or in a concavity) are bound to the tetrahedron that comes closest, with
//! their weights projected onto the simplex so they stay non-negative and sum
//! to one. The embedding is therefore total: every input point receives a
//! binding.
//!
//! The module is pure rest-pose geometry plus an affine reconstruction; it
//! holds no simulation state. All of it is standard barycentric interpolation;
//! nothing here is derived from Unreal Engine source.

use glam::{Mat3, Vec3};

/// Below this absolute determinant of its edge matrix a tetrahedron is treated
/// as flat and skipped: it cannot contain a point unambiguously.
const DEGENERATE_DET: f32 = 1e-10;

/// Parameters controlling point location and embedding.
#[derive(Clone, Copy, Debug)]
pub struct TetEmbeddingParams {
    /// A point is accepted as inside a tetrahedron when every barycentric
    /// weight is at least `-containment_eps`. A small positive value absorbs
    /// round-off so points exactly on a shared face bind to one of the
    /// neighbouring tets rather than falling through to the nearest-tet
    /// fallback.
    pub containment_eps: f32,
}

impl TetEmbeddingParams {
    /// Creates parameters with the given containment tolerance.
    #[must_use]
    pub fn new(containment_eps: f32) -> Self {
        Self { containment_eps }
    }
}

impl Default for TetEmbeddingParams {
    fn default() -> Self {
        Self {
            containment_eps: 1e-4,
        }
    }
}

/// One embedded point's binding into the cage mesh.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TetBinding {
    /// Index of the cage tetrahedron the point is bound to.
    pub tet: u32,
    /// The four cage-vertex indices of [`Self::tet`], in the tet's own order.
    pub nodes: [u32; 4],
    /// Barycentric weights for [`Self::nodes`]. Always non-negative and summing
    /// to one, so the binding is a convex blend of its four cage vertices.
    pub weights: [f32; 4],
}

impl TetBinding {
    /// Reconstructs the point from a cage pose, as the barycentric blend of the
    /// four node positions. Returns `None` when a node index is out of range
    /// for `cage`.
    #[must_use]
    pub fn interpolate(&self, cage: &[Vec3]) -> Option<Vec3> {
        let mut p = Vec3::ZERO;
        for i in 0..4 {
            let node = *self.nodes.get(i)? as usize;
            p += self.weights[i] * *cage.get(node)?;
        }
        Some(p)
    }
}

/// A whole point set embedded into a tetrahedral cage.
#[derive(Clone, Debug, PartialEq)]
pub struct TetEmbedding {
    /// One binding per embedded point, in input order.
    pub bindings: Vec<TetBinding>,
}

impl TetEmbedding {
    /// Number of embedded points.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.bindings.len()
    }

    /// Reconstructs every embedded point from a cage pose. Returns `None` when
    /// any binding references a cage vertex out of range for `cage`.
    #[must_use]
    pub fn deform(&self, cage: &[Vec3]) -> Option<Vec<Vec3>> {
        let mut out = Vec::with_capacity(self.bindings.len());
        for binding in &self.bindings {
            out.push(binding.interpolate(cage)?);
        }
        Some(out)
    }
}

/// Barycentric weights `(w0, w1, w2, w3)` of `p` in the tet `(v0, v1, v2, v3)`,
/// or `None` when the tet is degenerate. Weights may be negative when `p` is
/// outside the tet.
fn barycentric(v0: Vec3, v1: Vec3, v2: Vec3, v3: Vec3, p: Vec3) -> Option<[f32; 4]> {
    let dm = Mat3::from_cols(v1 - v0, v2 - v0, v3 - v0);
    if dm.determinant().abs() < DEGENERATE_DET {
        return None;
    }
    let lambda = dm.inverse() * (p - v0);
    let w0 = 1.0 - lambda.x - lambda.y - lambda.z;
    Some([w0, lambda.x, lambda.y, lambda.z])
}

/// How far outside the simplex a weight vector lies: the summed magnitude of
/// its negative components. Zero exactly when every weight is non-negative.
fn outside_amount(w: &[f32; 4]) -> f32 {
    let mut amount = 0.0;
    for &x in w {
        if x < 0.0 {
            amount -= x;
        }
    }
    amount
}

/// Projects weights onto the simplex: clamps negatives to zero and renormalises
/// so they sum to one.
fn project_to_simplex(mut w: [f32; 4]) -> [f32; 4] {
    for x in &mut w {
        if *x < 0.0 {
            *x = 0.0;
        }
    }
    let sum = w[0] + w[1] + w[2] + w[3];
    if sum > 0.0 {
        for x in &mut w {
            *x /= sum;
        }
    }
    w
}

/// Locates the tetrahedron that best contains `p` and returns its binding.
///
/// The first tetrahedron that contains `p` (every weight at least
/// `-params.containment_eps`) is used. If none contains it, the point is bound
/// to the tetrahedron whose weights lie least outside the simplex, with those
/// weights projected back onto it. Returns `None` only when `tets` is empty,
/// an index is out of range, or every tetrahedron is degenerate.
#[must_use]
pub fn locate_point(
    vertices: &[Vec3],
    tets: &[[u32; 4]],
    p: Vec3,
    params: &TetEmbeddingParams,
) -> Option<TetBinding> {
    if tets.is_empty() {
        return None;
    }
    let n = vertices.len();
    let mut best: Option<(f32, usize, [f32; 4])> = None;

    for (ti, t) in tets.iter().enumerate() {
        if t.iter().any(|&vi| vi as usize >= n) {
            return None;
        }
        let Some(w) = barycentric(
            vertices[t[0] as usize],
            vertices[t[1] as usize],
            vertices[t[2] as usize],
            vertices[t[3] as usize],
            p,
        ) else {
            continue;
        };

        let outside = outside_amount(&w);
        if outside <= params.containment_eps {
            return Some(TetBinding {
                tet: ti as u32,
                nodes: *t,
                weights: project_to_simplex(w),
            });
        }

        let improved = match best {
            Some((best_outside, _, _)) => outside < best_outside,
            None => true,
        };
        if improved {
            best = Some((outside, ti, w));
        }
    }

    best.map(|(_, ti, w)| TetBinding {
        tet: ti as u32,
        nodes: tets[ti],
        weights: project_to_simplex(w),
    })
}

/// Embeds every point of `points` into the cage mesh `(cage_vertices, tets)`.
///
/// Returns `None` when `tets` is empty, a tet index is out of range, or every
/// tetrahedron is degenerate (so no point can be located). Otherwise every
/// point receives a binding, using the nearest tetrahedron for points outside
/// the cage.
#[must_use]
pub fn build_tet_embedding(
    cage_vertices: &[Vec3],
    tets: &[[u32; 4]],
    points: &[Vec3],
    params: &TetEmbeddingParams,
) -> Option<TetEmbedding> {
    let mut bindings = Vec::with_capacity(points.len());
    for &p in points {
        bindings.push(locate_point(cage_vertices, tets, p, params)?);
    }
    Some(TetEmbedding { bindings })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::tetrahedralize::{tetrahedralize, TetMeshParams};

    const UNIT: [Vec3; 4] = [
        Vec3::ZERO,
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
    ];

    fn cube_surface(h: f32) -> (Vec<Vec3>, Vec<[u32; 3]>) {
        let verts = vec![
            Vec3::new(-h, -h, -h),
            Vec3::new(h, -h, -h),
            Vec3::new(h, h, -h),
            Vec3::new(-h, h, -h),
            Vec3::new(-h, -h, h),
            Vec3::new(h, -h, h),
            Vec3::new(h, h, h),
            Vec3::new(-h, h, h),
        ];
        let idx = vec![
            [0u32, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [3, 7, 6],
            [3, 6, 2],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ];
        (verts, idx)
    }

    #[test]
    fn centroid_has_equal_weights() {
        let tets = [[0u32, 1, 2, 3]];
        let centroid = (UNIT[0] + UNIT[1] + UNIT[2] + UNIT[3]) / 4.0;
        let b = locate_point(&UNIT, &tets, centroid, &TetEmbeddingParams::default()).unwrap();
        for w in b.weights {
            assert!((w - 0.25).abs() < 1e-5, "weight {w}");
        }
    }

    #[test]
    fn vertex_binds_to_itself() {
        let tets = [[0u32, 1, 2, 3]];
        let b = locate_point(&UNIT, &tets, UNIT[2], &TetEmbeddingParams::default()).unwrap();
        // Node 2 of the tet carries all the weight.
        assert!((b.weights[2] - 1.0).abs() < 1e-4);
        let rest: Vec<Vec3> = UNIT.to_vec();
        let p = b.interpolate(&rest).unwrap();
        assert!((p - UNIT[2]).length() < 1e-4);
    }

    #[test]
    fn weights_are_convex_everywhere() {
        let tets = [[0u32, 1, 2, 3]];
        let params = TetEmbeddingParams::default();
        for p in [
            Vec3::new(0.1, 0.1, 0.1),
            Vec3::new(5.0, 5.0, 5.0),
            Vec3::new(-2.0, 0.3, 0.1),
        ] {
            let b = locate_point(&UNIT, &tets, p, &params).unwrap();
            let sum: f32 = b.weights.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "sum {sum}");
            assert!(b.weights.iter().all(|&w| w >= 0.0));
        }
    }

    #[test]
    fn interpolation_reconstructs_interior_points() {
        let (v, i) = cube_surface(1.0);
        let mesh = tetrahedralize(&v, &i, &TetMeshParams::new(12)).unwrap();
        let params = TetEmbeddingParams::default();
        let points = [
            Vec3::new(0.1, -0.2, 0.3),
            Vec3::new(-0.4, 0.1, -0.1),
            Vec3::new(0.5, 0.5, -0.5),
        ];
        let emb = build_tet_embedding(&mesh.vertices, &mesh.tets, &points, &params).unwrap();
        let rebuilt = emb.deform(&mesh.vertices).unwrap();
        for (p, r) in points.iter().zip(&rebuilt) {
            assert!((*p - *r).length() < 1e-3, "point {p:?} rebuilt {r:?}");
        }
    }

    #[test]
    fn embedding_follows_an_affine_deformation() {
        let (v, i) = cube_surface(1.0);
        let mesh = tetrahedralize(&v, &i, &TetMeshParams::new(12)).unwrap();
        let params = TetEmbeddingParams::default();
        let points = [Vec3::new(0.2, 0.1, -0.3), Vec3::new(-0.3, 0.4, 0.2)];
        let emb = build_tet_embedding(&mesh.vertices, &mesh.tets, &points, &params).unwrap();

        // Transform the cage; embedded points must follow exactly.
        let a = Mat3::from_rotation_y(0.7) * Mat3::from_diagonal(Vec3::new(1.5, 0.8, 1.2));
        let t = Vec3::new(2.0, -1.0, 0.5);
        let moved: Vec<Vec3> = mesh.vertices.iter().map(|&x| a * x + t).collect();
        let rebuilt = emb.deform(&moved).unwrap();
        for (p, r) in points.iter().zip(&rebuilt) {
            let expected = a * *p + t;
            assert!(
                (expected - *r).length() < 1e-2,
                "expected {expected:?} got {r:?}"
            );
        }
    }

    #[test]
    fn outside_point_uses_nearest_tet() {
        let tets = [[0u32, 1, 2, 3]];
        let params = TetEmbeddingParams::default();
        let far = Vec3::new(10.0, 10.0, 10.0);
        let b = locate_point(&UNIT, &tets, far, &params).unwrap();
        let sum: f32 = b.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(b.weights.iter().all(|&w| w >= 0.0));
    }

    #[test]
    fn structural_errors_return_none() {
        let params = TetEmbeddingParams::default();
        assert!(locate_point(&UNIT, &[], Vec3::ZERO, &params).is_none());
        assert!(locate_point(&UNIT, &[[0u32, 1, 2, 9]], Vec3::ZERO, &params).is_none());
        // A single flat tetrahedron can locate no point.
        let flat = [Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::new(1.0, 1.0, 0.0)];
        assert!(locate_point(&flat, &[[0u32, 1, 2, 3]], Vec3::ZERO, &params).is_none());
    }

    #[test]
    fn is_deterministic() {
        let (v, i) = cube_surface(1.0);
        let mesh = tetrahedralize(&v, &i, &TetMeshParams::new(10)).unwrap();
        let params = TetEmbeddingParams::default();
        let points = [Vec3::new(0.1, 0.2, 0.3), Vec3::new(-0.2, -0.1, 0.4)];
        let a = build_tet_embedding(&mesh.vertices, &mesh.tets, &points, &params).unwrap();
        let b = build_tet_embedding(&mesh.vertices, &mesh.tets, &points, &params).unwrap();
        assert_eq!(a, b);
    }
}
