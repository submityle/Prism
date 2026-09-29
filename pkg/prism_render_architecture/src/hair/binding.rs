//! Groom root binding to a skinned scalp mesh.
//!
//! A groom is *authored* in the rest pose of the character it grows on, but the
//! character's scalp is a skinned mesh that deforms every frame. For the hair to
//! ride the head instead of floating in place, each strand root must be pinned
//! to the surface it grew from and follow that surface as it skins. This is the
//! job of a *binding asset* in `UE5` Groom and of the root-skinning step in
//! `TressFX`: at import time each root is projected onto the closest scalp
//! triangle and stored as a barycentric attachment plus a signed height along
//! the face normal; at runtime the deformed triangle is re-read and the root
//! world transform is reconstructed from it.
//!
//! This module owns the *geometry* of that attachment as a deterministic,
//! array-in/array-out service so it can be validated with CPU golden tests. The
//! resolved root positions feed the pinned particles of [`super::dynamics`]
//! (`inverse_mass == 0`), and the resolved root frames orient the strand's
//! rest-pose shape. It does not own the skinning itself: the caller supplies the
//! already-skinned vertex positions each frame.
//!
//! ## Contract
//! - Binding is topology-stable: the triangle index list is identical between
//!   bind time and every resolve, only the vertex positions change.
//! - The round trip is exact: resolving against the same (rest) vertices used to
//!   bind reproduces the original root position (up to float rounding), which is
//!   the golden property the tests pin.
//! - Degenerate input (no triangles, out-of-range indices, zero-area faces)
//!   never panics; it yields a safe identity attachment or frame instead.

use alloc::vec::Vec;

use super::interpolation::Vec3;

/// Sentinel triangle index meaning "this root could not be bound".
///
/// Emitted by [`bind_roots`] when the scalp mesh has no triangles to project
/// onto. [`resolve_root_frames`] treats it (and any out-of-range index) as an
/// identity attachment so a malformed binding degrades gracefully instead of
/// panicking.
pub const UNBOUND: u32 = u32::MAX;

/// Squared-length threshold below which a direction is treated as degenerate.
///
/// Matches the guard style used elsewhere in `hair/`: compare squared lengths
/// against a small epsilon rather than testing floats for equality.
const EPS_LEN_SQ: f32 = 1.0e-12;

/// One strand root's attachment to the scalp mesh.
///
/// Stored once at bind time and replayed every frame against the deformed
/// vertices. The attachment is intentionally tiny (one index + four floats) so a
/// full groom's bindings stay cache-friendly.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshBinding {
    /// Index into the triangle list the root is attached to, or [`UNBOUND`].
    pub triangle: u32,
    /// Barycentric coordinates of the projected root within `triangle`.
    ///
    /// The three weights are clamped to the triangle and sum to one (for a valid
    /// binding), so the interpolated point always lies on the face.
    pub bary: [f32; 3],
    /// Signed distance from the projected point to the root along the face
    /// normal, preserving how far the authored root floated off the surface.
    pub height: f32,
}

impl MeshBinding {
    /// An identity attachment that resolves to the world origin.
    pub const UNBOUND: MeshBinding = MeshBinding {
        triangle: UNBOUND,
        bary: [0.0, 0.0, 0.0],
        height: 0.0,
    };

    /// Returns `true` when this attachment failed to bind to any triangle.
    #[must_use]
    pub fn is_bound(&self) -> bool {
        self.triangle != UNBOUND
    }
}

/// A resolved root world transform: an origin plus a right-handed orthonormal
/// basis.
///
/// `normal` follows the deformed face normal, `tangent` follows the face's first
/// edge, and `bitangent` completes the frame. Downstream the position pins the
/// root particle and the basis rotates the strand's authored rest shape so the
/// whole strand swings with the scalp, not just its root point.
#[derive(Clone, Copy, Debug)]
pub struct RootFrame {
    /// World-space root position on (or offset from) the deformed surface.
    pub position: Vec3,
    /// Unit tangent along the attached face's first edge.
    pub tangent: Vec3,
    /// Unit outward face normal at the attachment.
    pub normal: Vec3,
    /// Unit bitangent completing the right-handed basis.
    pub bitangent: Vec3,
}

impl RootFrame {
    /// The identity frame at the origin, used for unbound or degenerate roots.
    pub const IDENTITY: RootFrame = RootFrame {
        position: Vec3::ZERO,
        tangent: Vec3::new(1.0, 0.0, 0.0),
        normal: Vec3::new(0.0, 1.0, 0.0),
        bitangent: Vec3::new(0.0, 0.0, 1.0),
    };
}

/// Fetches a triangle's three corner positions, or `None` when the index or any
/// vertex index is out of range.
fn triangle_corners(
    triangle: u32,
    vertices: &[Vec3],
    triangles: &[[u32; 3]],
) -> Option<(Vec3, Vec3, Vec3)> {
    let tri = *triangles.get(triangle as usize)?;
    let a = *vertices.get(tri[0] as usize)?;
    let b = *vertices.get(tri[1] as usize)?;
    let c = *vertices.get(tri[2] as usize)?;
    Some((a, b, c))
}

/// Returns the point on triangle `abc` closest to `p`, plus its barycentric
/// coordinates `(u, v, w)` such that the point is `u*a + v*b + w*c`.
///
/// This is the standard Voronoi-region test from Ericson's *Real-Time Collision
/// Detection*: it decides whether the projection falls in a vertex, edge, or
/// face region and clamps accordingly, so the result is always on the triangle
/// even for obtuse faces or points far outside. Degenerate (zero-area) faces
/// collapse gracefully onto vertex `a`.
#[must_use]
pub fn closest_point_on_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> (Vec3, [f32; 3]) {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;

    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    // Vertex region A.
    if d1 <= 0.0 && d2 <= 0.0 {
        return (a, [1.0, 0.0, 0.0]);
    }

    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    // Vertex region B.
    if d3 >= 0.0 && d4 <= d3 {
        return (b, [0.0, 1.0, 0.0]);
    }

    // Edge region AB.
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let denom = d1 - d3;
        let v = if denom > EPS_LEN_SQ { d1 / denom } else { 0.0 };
        let point = a + ab.scale(v);
        return (point, [1.0 - v, v, 0.0]);
    }

    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    // Vertex region C.
    if d6 >= 0.0 && d5 <= d6 {
        return (c, [0.0, 0.0, 1.0]);
    }

    // Edge region AC.
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let denom = d2 - d6;
        let w = if denom > EPS_LEN_SQ { d2 / denom } else { 0.0 };
        let point = a + ac.scale(w);
        return (point, [1.0 - w, 0.0, w]);
    }

    // Edge region BC.
    let va = d3 * d6 - d5 * d4;
    let bc_num = d4 - d3;
    let bc_den = (d4 - d3) + (d5 - d6);
    if va <= 0.0 && bc_num >= 0.0 && (d5 - d6) >= 0.0 {
        let w = if bc_den > EPS_LEN_SQ {
            bc_num / bc_den
        } else {
            0.0
        };
        let point = b + (c - b).scale(w);
        return (point, [0.0, 1.0 - w, w]);
    }

    // Interior face region: barycentric via the precomputed numerators.
    let denom_sum = va + vb + vc;
    if denom_sum > EPS_LEN_SQ {
        let inv = 1.0 / denom_sum;
        let v = vb * inv;
        let w = vc * inv;
        let point = a + ab.scale(v) + ac.scale(w);
        (point, [1.0 - v - w, v, w])
    } else {
        // Zero-area triangle: fall back to vertex A.
        (a, [1.0, 0.0, 0.0])
    }
}

/// Interpolates a triangle's rest position from a barycentric weight triple.
fn bary_point(bary: [f32; 3], a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    a.scale(bary[0]) + b.scale(bary[1]) + c.scale(bary[2])
}

/// Unit outward normal of triangle `abc`, or `None` for a zero-area face.
fn face_normal(a: Vec3, b: Vec3, c: Vec3) -> Option<Vec3> {
    let n = (b - a).cross(c - a);
    if n.dot(n) > EPS_LEN_SQ {
        Some(n.normalize_or(Vec3::new(0.0, 1.0, 0.0)))
    } else {
        None
    }
}

/// Binds each strand root to the closest triangle of the rest-pose scalp mesh.
///
/// For every root it brute-force scans all triangles for the one whose closest
/// surface point is nearest (deterministic: ties resolve to the lowest triangle
/// index because the scan keeps the first strict minimum), then records the
/// barycentric projection and the signed height along that face's normal.
///
/// Returns one [`MeshBinding`] per root, in input order. When the mesh has no
/// triangles every root binds to [`MeshBinding::UNBOUND`]; roots are never
/// dropped, so indices stay aligned with the caller's strand array.
#[must_use]
pub fn bind_roots(roots: &[Vec3], vertices: &[Vec3], triangles: &[[u32; 3]]) -> Vec<MeshBinding> {
    let mut out = Vec::with_capacity(roots.len());
    for &root in roots {
        out.push(bind_one_root(root, vertices, triangles));
    }
    out
}

/// Binds a single root; factored out so [`bind_roots`] stays a thin loop.
fn bind_one_root(root: Vec3, vertices: &[Vec3], triangles: &[[u32; 3]]) -> MeshBinding {
    let mut best: Option<(f32, u32, [f32; 3], f32)> = None;
    for (index, _tri) in triangles.iter().enumerate() {
        let idx = index as u32;
        let Some((a, b, c)) = triangle_corners(idx, vertices, triangles) else {
            continue;
        };
        let (point, bary) = closest_point_on_triangle(root, a, b, c);
        let offset = root - point;
        let dist_sq = offset.dot(offset);
        let closer = match best {
            Some((best_dist, ..)) => dist_sq < best_dist,
            None => true,
        };
        if closer {
            // Signed height keeps the authored float-off distance: positive when
            // the root sits on the outward-normal side of the face.
            let height = face_normal(a, b, c).map_or(0.0, |n| offset.dot(n));
            best = Some((dist_sq, idx, bary, height));
        }
    }
    match best {
        Some((_, triangle, bary, height)) => MeshBinding {
            triangle,
            bary,
            height,
        },
        None => MeshBinding::UNBOUND,
    }
}

/// Resolves each binding against the current (deformed/skinned) scalp vertices,
/// producing a world-space root frame per strand.
///
/// The triangle topology must match the one used by [`bind_roots`]; only the
/// vertex positions change between frames. Unbound or out-of-range bindings, and
/// zero-area deformed faces, resolve to [`RootFrame::IDENTITY`] so a bad binding
/// never panics and never poisons neighbouring strands.
#[must_use]
pub fn resolve_root_frames(
    bindings: &[MeshBinding],
    vertices: &[Vec3],
    triangles: &[[u32; 3]],
) -> Vec<RootFrame> {
    let mut out = Vec::with_capacity(bindings.len());
    for binding in bindings {
        out.push(resolve_one_frame(*binding, vertices, triangles));
    }
    out
}

/// Convenience wrapper returning only the resolved root positions.
///
/// The positions are what pin the root particles in [`super::dynamics`]; callers
/// that do not need the orientation basis can skip building it downstream.
#[must_use]
pub fn resolve_root_positions(
    bindings: &[MeshBinding],
    vertices: &[Vec3],
    triangles: &[[u32; 3]],
) -> Vec<Vec3> {
    resolve_root_frames(bindings, vertices, triangles)
        .into_iter()
        .map(|frame| frame.position)
        .collect()
}

/// Resolves a single binding; factored out so the public loops stay thin.
fn resolve_one_frame(binding: MeshBinding, vertices: &[Vec3], triangles: &[[u32; 3]]) -> RootFrame {
    if !binding.is_bound() {
        return RootFrame::IDENTITY;
    }
    let Some((a, b, c)) = triangle_corners(binding.triangle, vertices, triangles) else {
        return RootFrame::IDENTITY;
    };
    let Some(normal) = face_normal(a, b, c) else {
        return RootFrame::IDENTITY;
    };
    let surface = bary_point(binding.bary, a, b, c);
    let position = surface + normal.scale(binding.height);
    // Tangent follows the first edge; re-orthonormalize against the normal so the
    // basis stays right-handed even when the edge is not perpendicular to it.
    let tangent = (b - a).normalize_or(Vec3::new(1.0, 0.0, 0.0));
    let bitangent = normal.cross(tangent).normalize_or(Vec3::new(0.0, 0.0, 1.0));
    let tangent = bitangent.cross(normal).normalize_or(tangent);
    RootFrame {
        position,
        tangent,
        normal,
        bitangent,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3::new(x, y, z)
    }

    /// A single unit right-triangle in the y = 0 plane, normal pointing +y.
    fn unit_scalp() -> (Vec<Vec3>, Vec<[u32; 3]>) {
        // Winding [0, 2, 1] makes the face normal point +y (a scalp patch whose
        // outward side faces up), so the resolved frame normal is unambiguous.
        let verts = alloc::vec![v(0.0, 0.0, 0.0), v(1.0, 0.0, 0.0), v(0.0, 0.0, 1.0)];
        let tris = alloc::vec![[0u32, 2, 1]];
        (verts, tris)
    }

    fn approx(a: Vec3, b: Vec3, eps: f32) -> bool {
        (a - b).length() <= eps
    }

    #[test]
    fn binds_root_on_surface_at_zero_height() {
        let (verts, tris) = unit_scalp();
        // Point inside the triangle, right on the plane.
        let roots = alloc::vec![v(0.25, 0.0, 0.25)];
        let bindings = bind_roots(&roots, &verts, &tris);
        assert_eq!(bindings.len(), 1);
        assert!(bindings[0].is_bound());
        assert_eq!(bindings[0].triangle, 0);
        // Height is ~0 since the root is on the surface.
        assert!(bindings[0].height.abs() <= 1.0e-6);
        // Barycentric weights sum to one.
        let s: f32 = bindings[0].bary.iter().sum();
        assert!((s - 1.0).abs() <= 1.0e-6);
    }

    #[test]
    fn round_trip_reproduces_root_position() {
        let (verts, tris) = unit_scalp();
        // A root floating above the surface keeps its height on resolve.
        let roots = alloc::vec![v(0.3, 0.5, 0.2), v(0.1, 0.0, 0.1)];
        let bindings = bind_roots(&roots, &verts, &tris);
        let positions = resolve_root_positions(&bindings, &verts, &tris);
        assert_eq!(positions.len(), 2);
        for (resolved, original) in positions.iter().zip(roots.iter()) {
            assert!(
                approx(*resolved, *original, 1.0e-5),
                "resolved {resolved:?} != original {original:?}",
            );
        }
    }

    #[test]
    fn root_follows_rigid_translation_of_scalp() {
        let (verts, tris) = unit_scalp();
        let roots = alloc::vec![v(0.3, 0.4, 0.2)];
        let bindings = bind_roots(&roots, &verts, &tris);
        // Skin: rigidly translate every vertex by the same offset.
        let shift = v(2.0, -1.0, 3.0);
        let deformed: Vec<Vec3> = verts.iter().map(|&p| p + shift).collect();
        let positions = resolve_root_positions(&bindings, &deformed, &tris);
        // The root should ride the surface by exactly the same translation.
        assert!(approx(positions[0], roots[0] + shift, 1.0e-5));
    }

    #[test]
    fn resolved_frame_is_orthonormal_and_matches_face_normal() {
        let (verts, tris) = unit_scalp();
        let roots = alloc::vec![v(0.25, 0.1, 0.25)];
        let bindings = bind_roots(&roots, &verts, &tris);
        let frames = resolve_root_frames(&bindings, &verts, &tris);
        let f = frames[0];
        // Orthonormal: unit length and mutually perpendicular.
        assert!((f.tangent.length() - 1.0).abs() <= 1.0e-5);
        assert!((f.normal.length() - 1.0).abs() <= 1.0e-5);
        assert!((f.bitangent.length() - 1.0).abs() <= 1.0e-5);
        assert!(f.tangent.dot(f.normal).abs() <= 1.0e-5);
        assert!(f.tangent.dot(f.bitangent).abs() <= 1.0e-5);
        assert!(f.normal.dot(f.bitangent).abs() <= 1.0e-5);
        // Face lies in y = 0, so the normal is ±y; our winding gives +y.
        assert!(approx(f.normal, v(0.0, 1.0, 0.0), 1.0e-5));
    }

    #[test]
    fn empty_mesh_leaves_roots_unbound_without_panic() {
        let roots = alloc::vec![v(0.0, 0.0, 0.0), v(1.0, 2.0, 3.0)];
        let bindings = bind_roots(&roots, &[], &[]);
        assert_eq!(bindings.len(), 2);
        assert!(bindings.iter().all(|b| !b.is_bound()));
        // Unbound bindings resolve to the identity frame at the origin.
        let frames = resolve_root_frames(&bindings, &[], &[]);
        assert!(approx(frames[0].position, Vec3::ZERO, 0.0));
    }

    #[test]
    fn out_of_range_binding_resolves_to_identity() {
        let (verts, tris) = unit_scalp();
        let bogus = alloc::vec![MeshBinding {
            triangle: 99,
            bary: [0.5, 0.25, 0.25],
            height: 1.0,
        }];
        let frames = resolve_root_frames(&bogus, &verts, &tris);
        assert_eq!(frames.len(), 1);
        assert!(approx(
            frames[0].position,
            RootFrame::IDENTITY.position,
            0.0
        ));
    }

    #[test]
    fn degenerate_face_does_not_panic_and_stays_finite() {
        // All three vertices coincide: a zero-area face.
        let verts = alloc::vec![v(1.0, 1.0, 1.0), v(1.0, 1.0, 1.0), v(1.0, 1.0, 1.0)];
        let tris = alloc::vec![[0u32, 1, 2]];
        let roots = alloc::vec![v(0.0, 0.0, 0.0)];
        let bindings = bind_roots(&roots, &verts, &tris);
        let frames = resolve_root_frames(&bindings, &verts, &tris);
        assert_eq!(frames.len(), 1);
        assert!(frames[0].position.x.is_finite());
        assert!(frames[0].position.y.is_finite());
        assert!(frames[0].position.z.is_finite());
    }

    #[test]
    fn closest_point_clamps_outside_projection_to_edge() {
        let a = v(0.0, 0.0, 0.0);
        let b = v(1.0, 0.0, 0.0);
        let c = v(0.0, 0.0, 1.0);
        // A point off the +x side projects onto edge AB, clamped within it.
        let (point, bary) = closest_point_on_triangle(v(2.0, 0.0, -1.0), a, b, c);
        assert!(point.x <= 1.0 + 1.0e-6 && point.x >= -1.0e-6);
        // Weight on C should be zero for an edge-AB projection.
        assert!(bary[2].abs() <= 1.0e-6);
        let s: f32 = bary.iter().sum();
        assert!((s - 1.0).abs() <= 1.0e-5);
    }

    #[test]
    fn picks_nearest_of_several_triangles_deterministically() {
        // Two coplanar triangles side by side; root sits above the second one.
        let verts = alloc::vec![
            v(0.0, 0.0, 0.0),
            v(1.0, 0.0, 0.0),
            v(0.0, 0.0, 1.0),
            v(5.0, 0.0, 0.0),
            v(6.0, 0.0, 0.0),
            v(5.0, 0.0, 1.0),
        ];
        let tris = alloc::vec![[0u32, 1, 2], [3, 4, 5]];
        let roots = alloc::vec![v(5.3, 0.2, 0.2)];
        let bindings = bind_roots(&roots, &verts, &tris);
        assert_eq!(bindings[0].triangle, 1);
        // Re-binding the same input is bit-for-bit identical.
        let again = bind_roots(&roots, &verts, &tris);
        assert_eq!(bindings, again);
    }
}
