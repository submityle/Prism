//! Render-mesh barycentric embedding onto the coarse sim mesh.
//!
//! A cloth garment carries two meshes: a low-resolution *sim mesh* whose
//! vertices are the solved particles, and a high-resolution *render mesh* that
//! carries the visible silhouette, seams, and shading detail. The render mesh
//! is never solved; each of its vertices is frozen into a triangle of the sim
//! mesh through barycentric coordinates plus a signed offset along that
//! triangle's face normal, exactly as production cloth pipelines embed a skin
//! into their simulation cage. Once bound, driving the render mesh is a pure,
//! deterministic function of the current sim positions: reconstruct the point
//! inside its host triangle and push it back out along the (re-evaluated) face
//! normal to restore thickness.
//!
//! Everything here is allocation-light and free of transcendental math: the
//! only root used is `sqrt`, inherited from [`Vec3::normalize_or_zero`]. All
//! out-of-range indices degrade to a defined result instead of panicking, and
//! degenerate (zero-area) host triangles fall back to their first vertex so no
//! `NaN` can enter the render stream.

use alloc::vec::Vec;

use super::{Vec3, EPS_LEN_SQ};

/// Binding of one render-mesh vertex to a single sim-mesh triangle.
///
/// The vertex is expressed as barycentric weights over the three sim particles
/// in [`BarycentricBinding::tri`] plus a signed distance along the triangle
/// face normal ([`BarycentricBinding::normal_offset`]) that preserves the
/// render surface's offset from the sim cage (garment thickness). The weights
/// nominally sum to one; they are stored rather than recomputed so embedding is
/// a cheap, branch-light dot product per vertex.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BarycentricBinding {
    /// Indices of the three sim-mesh particles forming the host triangle.
    pub tri: [u32; 3],
    /// Barycentric weights `(w0, w1, w2)` for the host triangle; they sum to
    /// approximately one for a well-formed binding.
    pub bary: (f32, f32, f32),
    /// Signed distance along the host triangle's face normal, restoring the
    /// render vertex's offset from the sim surface (cloth thickness).
    pub normal_offset: f32,
}

impl BarycentricBinding {
    /// Builds a binding from explicit indices, weights, and normal offset.
    #[must_use]
    pub fn new(tri: [u32; 3], bary: (f32, f32, f32), normal_offset: f32) -> Self {
        Self {
            tri,
            bary,
            normal_offset,
        }
    }
}

/// Fetches the three host-triangle positions, or `None` if any index is out of
/// range. Keeping this in one place means every entry point degrades to a
/// defined result on a stale or truncated index list rather than panicking.
fn host_positions(tri: [u32; 3], positions: &[Vec3]) -> Option<[Vec3; 3]> {
    let p0 = *positions.get(tri[0] as usize)?;
    let p1 = *positions.get(tri[1] as usize)?;
    let p2 = *positions.get(tri[2] as usize)?;
    Some([p0, p1, p2])
}

/// Unit face normal of the triangle `(a, b, c)`, or [`Vec3::ZERO`] when the
/// triangle is degenerate (its edges are parallel or coincident).
fn face_normal(a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    b.sub(a).cross(c.sub(a)).normalize_or_zero()
}

/// Computes the barycentric coordinates of `p` projected onto the plane of
/// triangle `(a, b, c)`.
///
/// The result `(w0, w1, w2)` satisfies `w0 + w1 + w2 == 1` (up to rounding) and
/// reconstructs the in-plane projection of `p` as `w0*a + w1*b + w2*c`. The
/// solve is the Gram / Cramer form over the two edge vectors, so it uses only
/// `dot`, `cross`, and `sub`; no transcendental functions appear. A degenerate
/// triangle whose edges span (numerically) zero area returns `(1.0, 0.0, 0.0)`,
/// pinning the point to the first vertex instead of dividing by zero.
#[must_use]
pub fn compute_barycentric(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> (f32, f32, f32) {
    let v0 = b.sub(a);
    let v1 = c.sub(a);
    let v2 = p.sub(a);

    let d00 = v0.dot(v0);
    let d01 = v0.dot(v1);
    let d11 = v1.dot(v1);
    let d20 = v2.dot(v0);
    let d21 = v2.dot(v1);

    // The Gram determinant equals the squared length of `v0 x v1`, i.e. the
    // squared doubled triangle area; comparing it against the shared squared
    // epsilon rejects zero-area triangles consistently with normalization.
    let denom = d00 * d11 - d01 * d01;
    if denom.abs() < EPS_LEN_SQ {
        return (1.0, 0.0, 0.0);
    }

    let inv = 1.0 / denom;
    let w1 = (d11 * d20 - d01 * d21) * inv;
    let w2 = (d00 * d21 - d01 * d20) * inv;
    let w0 = 1.0 - w1 - w2;
    (w0, w1, w2)
}

/// Binds a render-mesh vertex at world position `p` to the sim triangle named
/// by `tri`, reading the sim particle positions from `positions`.
///
/// Returns `None` when any triangle index is out of range (a stale or truncated
/// position list), so callers can skip the vertex without panicking. Otherwise
/// the barycentric weights capture the in-plane location and `normal_offset`
/// captures the signed distance of `p` above or below the triangle plane, so a
/// later [`embed_render_vertex`] reproduces `p` exactly when the sim mesh has
/// not moved.
#[must_use]
pub fn bind_render_vertex(
    p: Vec3,
    tri: [u32; 3],
    positions: &[Vec3],
) -> Option<BarycentricBinding> {
    let [a, b, c] = host_positions(tri, positions)?;
    let bary = compute_barycentric(p, a, b, c);
    let normal = face_normal(a, b, c);
    // `a` lies in the plane, so this is the signed plane distance; a degenerate
    // triangle yields a zero normal and therefore a zero offset.
    let normal_offset = p.sub(a).dot(normal);
    Some(BarycentricBinding::new(tri, bary, normal_offset))
}

/// Evaluates one render-mesh vertex from its binding and the current sim
/// positions.
///
/// Reconstructs the in-plane point `w0*p0 + w1*p1 + w2*p2` and pushes it along
/// the re-evaluated face normal by the stored offset, so the render surface
/// tracks both the deformation and the orientation of its host triangle.
/// Returns [`Vec3::ZERO`] when any triangle index is out of range, never
/// panicking; a degenerate host triangle contributes a zero normal, so the
/// offset term simply vanishes rather than producing `NaN`.
#[must_use]
pub fn embed_render_vertex(binding: &BarycentricBinding, positions: &[Vec3]) -> Vec3 {
    let Some([p0, p1, p2]) = host_positions(binding.tri, positions) else {
        return Vec3::ZERO;
    };

    let (w0, w1, w2) = binding.bary;
    let base = p0.scale(w0).add(p1.scale(w1)).add(p2.scale(w2));
    let normal = face_normal(p0, p1, p2);
    base.add(normal.scale(binding.normal_offset))
}

/// Embeds a whole render mesh into the sim positions, writing one output vertex
/// per binding into `out`.
///
/// `out` is cleared first and then filled in binding order, so the traversal is
/// `O(n)` and fully deterministic. Out-of-range bindings emit [`Vec3::ZERO`]
/// (see [`embed_render_vertex`]) rather than being skipped, keeping the output
/// index-aligned with `bindings`.
pub fn embed_render_mesh(bindings: &[BarycentricBinding], positions: &[Vec3], out: &mut Vec<Vec3>) {
    out.clear();
    out.reserve(bindings.len());
    for binding in bindings {
        out.push(embed_render_vertex(binding, positions));
    }
}

/// Convenience wrapper over [`embed_render_mesh`] that allocates and returns the
/// embedded render positions, index-aligned with `bindings`.
#[must_use]
pub fn embedded_render_positions(bindings: &[BarycentricBinding], positions: &[Vec3]) -> Vec<Vec3> {
    let mut out = Vec::with_capacity(bindings.len());
    for binding in bindings {
        out.push(embed_render_vertex(binding, positions));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    fn approx_vec(a: Vec3, b: Vec3) -> bool {
        approx(a.x, b.x) && approx(a.y, b.y) && approx(a.z, b.z)
    }

    fn approx_bary(got: (f32, f32, f32), want: (f32, f32, f32)) -> bool {
        approx(got.0, want.0) && approx(got.1, want.1) && approx(got.2, want.2)
    }

    fn tri() -> (Vec3, Vec3, Vec3) {
        (
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.0, 3.0, 0.0),
        )
    }

    #[test]
    fn barycentric_at_vertices_is_a_unit_basis() {
        let (a, b, c) = tri();
        assert!(approx_bary(
            compute_barycentric(a, a, b, c),
            (1.0, 0.0, 0.0)
        ));
        assert!(approx_bary(
            compute_barycentric(b, a, b, c),
            (0.0, 1.0, 0.0)
        ));
        assert!(approx_bary(
            compute_barycentric(c, a, b, c),
            (0.0, 0.0, 1.0)
        ));
    }

    #[test]
    fn barycentric_at_centroid_is_thirds() {
        let (a, b, c) = tri();
        let centroid = a.add(b).add(c).scale(1.0 / 3.0);
        let bary = compute_barycentric(centroid, a, b, c);
        assert!(approx_bary(bary, (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)));
        assert!(approx(bary.0 + bary.1 + bary.2, 1.0));
    }

    #[test]
    fn barycentric_weights_always_sum_to_one() {
        let (a, b, c) = tri();
        let p = Vec3::new(0.7, 0.4, 0.0);
        let (w0, w1, w2) = compute_barycentric(p, a, b, c);
        assert!(approx(w0 + w1 + w2, 1.0));
    }

    #[test]
    fn barycentric_projects_off_plane_point() {
        // A point lifted along +Z projects to the same in-plane barycentrics.
        let (a, b, c) = tri();
        let on_plane = Vec3::new(0.5, 0.5, 0.0);
        let lifted = Vec3::new(0.5, 0.5, 4.0);
        assert!(approx_bary(
            compute_barycentric(lifted, a, b, c),
            compute_barycentric(on_plane, a, b, c),
        ));
    }

    #[test]
    fn degenerate_triangle_falls_back_without_nan() {
        let a = Vec3::new(1.0, 1.0, 1.0);
        let bary = compute_barycentric(Vec3::new(5.0, 2.0, 3.0), a, a, a);
        assert!(approx_bary(bary, (1.0, 0.0, 0.0)));
        assert!(!bary.0.is_nan() && !bary.1.is_nan() && !bary.2.is_nan());
    }

    #[test]
    fn embed_is_linearly_exact_at_rest() {
        // Binding then embedding against the same positions reproduces `p`.
        let positions = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.0, 3.0, 0.0),
        ];
        let p = Vec3::new(0.6, 0.9, 0.0);
        let binding = bind_render_vertex(p, [0, 1, 2], &positions).expect("in range");
        let embedded = embed_render_vertex(&binding, &positions);
        assert!(approx_vec(embedded, p));
    }

    #[test]
    fn embed_follows_translated_sim_mesh() {
        // Rigidly translating every sim vertex translates the render vertex by
        // the same amount (barycentric combinations are affine).
        let rest = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.0, 3.0, 0.0),
        ];
        let p = Vec3::new(0.6, 0.9, 0.0);
        let binding = bind_render_vertex(p, [0, 1, 2], &rest).expect("in range");

        let shift = Vec3::new(10.0, -4.0, 2.5);
        let moved = [rest[0].add(shift), rest[1].add(shift), rest[2].add(shift)];
        let embedded = embed_render_vertex(&binding, &moved);
        assert!(approx_vec(embedded, p.add(shift)));
    }

    #[test]
    fn normal_offset_restores_thickness() {
        let positions = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.0, 3.0, 0.0),
        ];
        // Face normal points along +Z; lift the render vertex above the plane.
        let p = Vec3::new(0.5, 0.5, 1.25);
        let binding = bind_render_vertex(p, [0, 1, 2], &positions).expect("in range");
        assert!(approx(binding.normal_offset, 1.25));
        let embedded = embed_render_vertex(&binding, &positions);
        assert!(approx_vec(embedded, p));
    }

    #[test]
    fn embed_offset_tracks_rotated_normal() {
        // Bind against an XY triangle (normal +Z), then re-evaluate against a
        // triangle rotated into the XZ plane (normal -Y): the offset must follow
        // the new normal, not the old one.
        let rest = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let p = Vec3::new(0.25, 0.25, 2.0);
        let binding = bind_render_vertex(p, [0, 1, 2], &rest).expect("in range");
        assert!(approx(binding.normal_offset, 2.0));

        let rotated = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        // In-plane point is (0.25, 0, 0.25); normal is cross(x, z) = (0,-1,0).
        let expected = Vec3::new(0.25, -2.0, 0.25);
        let embedded = embed_render_vertex(&binding, &rotated);
        assert!(approx_vec(embedded, expected));
    }

    #[test]
    fn bind_out_of_range_index_is_none() {
        let positions = [Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0)];
        assert!(bind_render_vertex(Vec3::ZERO, [0, 1, 9], &positions).is_none());
    }

    #[test]
    fn embed_out_of_range_index_is_zero() {
        let positions = [Vec3::new(1.0, 2.0, 3.0), Vec3::new(4.0, 5.0, 6.0)];
        let binding = BarycentricBinding::new([0, 1, 7], (0.5, 0.25, 0.25), 1.0);
        assert_eq!(embed_render_vertex(&binding, &positions), Vec3::ZERO);
    }

    #[test]
    fn batch_embed_is_index_aligned_and_deterministic() {
        let positions = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
        ];
        let p0 = Vec3::new(0.5, 0.5, 0.0);
        let p1 = Vec3::new(1.0, 0.25, 0.0);
        let b0 = bind_render_vertex(p0, [0, 1, 2], &positions).expect("in range");
        let b1 = bind_render_vertex(p1, [0, 1, 2], &positions).expect("in range");
        let out_of_range = BarycentricBinding::new([0, 1, 5], (0.3, 0.3, 0.4), 0.0);
        let bindings = [b0, b1, out_of_range];

        let mut out = Vec::new();
        embed_render_mesh(&bindings, &positions, &mut out);
        assert_eq!(out.len(), 3);
        assert!(approx_vec(out[0], p0));
        assert!(approx_vec(out[1], p1));
        assert_eq!(out[2], Vec3::ZERO);

        // The returning wrapper agrees exactly, and re-running clears stale data.
        let again = embedded_render_positions(&bindings, &positions);
        assert_eq!(again, out);
        embed_render_mesh(&bindings, &positions, &mut out);
        assert_eq!(out.len(), 3);
    }
}
