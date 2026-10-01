//! Follicle binding: attach hair roots to an animated scalp mesh and transfer
//! that mesh's deformation (skinning, blendshapes, head motion) to the strand
//! roots deterministically.
//!
//! A groom is authored once in a rest pose, but the scalp it grows from is an
//! animated skinned mesh. Each follicle (hair root) is bound to one scalp
//! triangle by its **barycentric coordinates** plus a small signed **offset
//! along the surface normal** (roots usually sit a hair's breadth above the
//! skin). When the scalp triangle deforms, the root position and its local
//! tangent frame are reconstructed from the deformed triangle, so hair stays
//! glued to the scalp without sliding or poking through, and follows
//! expressions and head turns. The strand *body* is then simulated by the
//! dynamics solver; only the root is driven here.
//!
//! This is a pure, deterministic, panic-free mapping (array in, array out),
//! mirroring the contract of [`crate::hair::melanin`]: the barycentric solve,
//! the interpolated frame and the re-orthonormalisation are all plain vector
//! arithmetic (the only root is `f32::sqrt`, which is permitted). Degenerate
//! triangles and non-finite inputs fall back to safe values instead of
//! panicking. The owning renderer supplies the deformed triangle each frame
//! (from the skinning pass); this module owns no budget and performs no
//! transcendental math.

use alloc::vec::Vec;

/// Small epsilon used to detect degenerate (zero-area / zero-length) inputs.
const DEGENERATE_EPS: f32 = 1e-12;

#[inline]
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
fn scale(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline]
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn length(a: [f32; 3]) -> f32 {
    dot(a, a).sqrt()
}

/// Normalise `a`, returning `fallback` when `a` is (near) zero-length or
/// non-finite so the result is always a finite unit-ish vector.
fn normalize_or(a: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let len = length(a);
    if len > DEGENERATE_EPS && len.is_finite() {
        scale(a, 1.0 / len)
    } else {
        fallback
    }
}

#[inline]
fn finite3(a: [f32; 3]) -> bool {
    a[0].is_finite() && a[1].is_finite() && a[2].is_finite()
}

/// A scalp triangle with a per-vertex local frame. `positions[i]` is the world
/// position of vertex `i`; `normals[i]` / `tangents[i]` are that vertex's local
/// frame used to orient the follicle. The frame vectors need not be perfectly
/// unit-length or orthogonal on input; [`transfer_frame`] re-orthonormalises.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TriangleFrame {
    /// World-space positions of the three triangle vertices.
    pub positions: [[f32; 3]; 3],
    /// Per-vertex surface normals.
    pub normals: [[f32; 3]; 3],
    /// Per-vertex surface tangents.
    pub tangents: [[f32; 3]; 3],
}

/// Barycentric weights `(u, v, w)` for a point relative to a triangle, where
/// `u` weights vertex 0, `v` vertex 1 and `w` vertex 2. [`Barycentric::sanitized`]
/// clamps negatives and renormalises so the weights are non-negative and sum to
/// one (an interior, on-surface binding).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Barycentric {
    /// Weight on vertex 0.
    pub u: f32,
    /// Weight on vertex 1.
    pub v: f32,
    /// Weight on vertex 2.
    pub w: f32,
}

impl Barycentric {
    /// The triangle centroid weighting, used as a safe fallback for degenerate
    /// triangles.
    pub const CENTROID: Self = Self {
        u: 1.0 / 3.0,
        v: 1.0 / 3.0,
        w: 1.0 / 3.0,
    };

    /// Non-negative weights that sum to one. Non-finite components are treated
    /// as zero; an all-zero (or non-finite) set falls back to [`Self::CENTROID`].
    #[must_use]
    pub fn sanitized(self) -> Self {
        let clamp0 = |x: f32| if x.is_finite() && x > 0.0 { x } else { 0.0 };
        let u = clamp0(self.u);
        let v = clamp0(self.v);
        let w = clamp0(self.w);
        let sum = u + v + w;
        if sum > DEGENERATE_EPS {
            Self {
                u: u / sum,
                v: v / sum,
                w: w / sum,
            }
        } else {
            Self::CENTROID
        }
    }
}

/// A follicle's binding to one scalp triangle: barycentric position on the
/// triangle plus a signed offset along the (interpolated) surface normal, so a
/// root that sits slightly above the skin stays at that height after
/// deformation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FollicleBinding {
    /// Barycentric weights locating the root within the bound triangle.
    pub bary: Barycentric,
    /// Signed distance of the root above the surface along the normal.
    pub normal_offset: f32,
}

/// Interpolate a per-vertex vector attribute by barycentric weights.
fn bary_mix(bary: Barycentric, attr: [[f32; 3]; 3]) -> [f32; 3] {
    add(
        add(scale(attr[0], bary.u), scale(attr[1], bary.v)),
        scale(attr[2], bary.w),
    )
}

/// Compute the barycentric coordinates of `point` projected onto the plane of
/// `tri` (Ericson's method). Degenerate (zero-area) triangles return
/// [`Barycentric::CENTROID`] rather than dividing by zero.
#[must_use]
pub fn compute_barycentric(point: [f32; 3], tri: TriangleFrame) -> Barycentric {
    let a = tri.positions[0];
    let b = tri.positions[1];
    let c = tri.positions[2];
    let v0 = sub(b, a);
    let v1 = sub(c, a);
    let v2 = sub(point, a);
    let d00 = dot(v0, v0);
    let d01 = dot(v0, v1);
    let d11 = dot(v1, v1);
    let d20 = dot(v2, v0);
    let d21 = dot(v2, v1);
    let denom = d00 * d11 - d01 * d01;
    if denom.abs() <= DEGENERATE_EPS || !denom.is_finite() {
        return Barycentric::CENTROID;
    }
    let inv = 1.0 / denom;
    let v = (d11 * d20 - d01 * d21) * inv;
    let w = (d00 * d21 - d01 * d20) * inv;
    let u = 1.0 - v - w;
    Barycentric { u, v, w }
}

/// Bind a hair root to a scalp triangle: capture its barycentric position and
/// its signed height above the surface along the interpolated normal. The
/// stored weights are sanitised so the binding is always on-surface.
#[must_use]
pub fn bind_follicle(root: [f32; 3], tri: TriangleFrame) -> FollicleBinding {
    let bary = compute_barycentric(root, tri).sanitized();
    let surface = bary_mix(bary, tri.positions);
    let normal = normalize_or(bary_mix(bary, tri.normals), [0.0, 0.0, 1.0]);
    let offset = dot(sub(root, surface), normal);
    let normal_offset = if offset.is_finite() { offset } else { 0.0 };
    FollicleBinding {
        bary,
        normal_offset,
    }
}

/// Reconstruct the world-space root position from a binding and the (possibly
/// deformed) triangle: barycentric surface point plus the stored normal offset
/// along the deformed interpolated normal.
#[must_use]
pub fn transfer_root(binding: FollicleBinding, deformed: TriangleFrame) -> [f32; 3] {
    let bary = binding.bary.sanitized();
    let surface = bary_mix(bary, deformed.positions);
    let normal = normalize_or(bary_mix(bary, deformed.normals), [0.0, 0.0, 1.0]);
    let out = add(surface, scale(normal, binding.normal_offset));
    if finite3(out) {
        out
    } else {
        surface
    }
}

/// An orthonormal local frame transferred to the follicle root: `normal` and
/// `tangent` are unit and perpendicular; `bitangent` completes a right-handed
/// basis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FollicleFrame {
    /// Unit surface normal at the root.
    pub normal: [f32; 3],
    /// Unit surface tangent, orthogonal to `normal`.
    pub tangent: [f32; 3],
    /// Unit bitangent completing the right-handed basis.
    pub bitangent: [f32; 3],
}

/// Transfer and re-orthonormalise the local frame onto the deformed triangle by
/// barycentric interpolation followed by Gram-Schmidt. Degenerate interpolated
/// vectors fall back to canonical axes so the result is always a finite
/// orthonormal basis.
#[must_use]
pub fn transfer_frame(binding: FollicleBinding, deformed: TriangleFrame) -> FollicleFrame {
    let bary = binding.bary.sanitized();
    let normal = normalize_or(bary_mix(bary, deformed.normals), [0.0, 0.0, 1.0]);
    let raw_tangent = bary_mix(bary, deformed.tangents);
    // Gram-Schmidt: remove the normal component from the tangent.
    let projected = sub(raw_tangent, scale(normal, dot(raw_tangent, normal)));
    // Fallback tangent is any axis not parallel to the normal.
    let fallback = if normal[0].abs() < 0.9 {
        normalize_or(cross(normal, [1.0, 0.0, 0.0]), [0.0, 1.0, 0.0])
    } else {
        normalize_or(cross(normal, [0.0, 1.0, 0.0]), [1.0, 0.0, 0.0])
    };
    let tangent = normalize_or(projected, fallback);
    let bitangent = normalize_or(cross(normal, tangent), [0.0, 1.0, 0.0]);
    FollicleFrame {
        normal,
        tangent,
        bitangent,
    }
}

/// Transfer many bindings against one deformed triangle, preserving order. An
/// empty input yields an empty output.
#[must_use]
pub fn transfer_root_map(bindings: &[FollicleBinding], deformed: TriangleFrame) -> Vec<[f32; 3]> {
    bindings
        .iter()
        .map(|&b| transfer_root(b, deformed))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    fn close3(a: [f32; 3], b: [f32; 3]) -> bool {
        close(a[0], b[0]) && close(a[1], b[1]) && close(a[2], b[2])
    }

    fn flat_tri() -> TriangleFrame {
        TriangleFrame {
            positions: [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            normals: [[0.0, 0.0, 1.0]; 3],
            tangents: [[1.0, 0.0, 0.0]; 3],
        }
    }

    #[test]
    fn barycentric_recovers_vertices_and_centroid() {
        let tri = flat_tri();
        let b0 = compute_barycentric([0.0, 0.0, 0.0], tri);
        assert!(close(b0.u, 1.0) && close(b0.v, 0.0) && close(b0.w, 0.0));
        let b1 = compute_barycentric([1.0, 0.0, 0.0], tri);
        assert!(close(b1.u, 0.0) && close(b1.v, 1.0) && close(b1.w, 0.0));
        let bc = compute_barycentric([1.0 / 3.0, 1.0 / 3.0, 0.0], tri);
        assert!(close(bc.u, 1.0 / 3.0) && close(bc.v, 1.0 / 3.0) && close(bc.w, 1.0 / 3.0));
    }

    #[test]
    fn barycentric_ignores_height_via_projection() {
        let tri = flat_tri();
        // Same xy as a vertex but lifted along z: projects back to that vertex.
        let b = compute_barycentric([1.0, 0.0, 5.0], tri);
        assert!(close(b.u, 0.0) && close(b.v, 1.0) && close(b.w, 0.0));
    }

    #[test]
    fn sanitize_clamps_and_renormalises() {
        let s = Barycentric {
            u: -1.0,
            v: 1.0,
            w: 3.0,
        }
        .sanitized();
        assert!(close(s.u, 0.0));
        assert!(close(s.u + s.v + s.w, 1.0));
        assert!(close(s.v, 0.25) && close(s.w, 0.75));
        // All-zero / non-finite falls back to centroid.
        let z = Barycentric {
            u: 0.0,
            v: f32::NAN,
            w: -2.0,
        }
        .sanitized();
        assert_eq!(z, Barycentric::CENTROID);
    }

    #[test]
    fn bind_then_transfer_identity_recovers_root() {
        let tri = flat_tri();
        let root = [0.25, 0.25, 0.5];
        let binding = bind_follicle(root, tri);
        // Offset is the height above the surface.
        assert!(close(binding.normal_offset, 0.5));
        // Transferring against the same (undeformed) triangle recovers the root.
        let back = transfer_root(binding, tri);
        assert!(close3(back, root), "{back:?}");
    }

    #[test]
    fn transfer_follows_translation() {
        let tri = flat_tri();
        let root = [0.25, 0.25, 0.5];
        let binding = bind_follicle(root, tri);
        let shift = [10.0, -3.0, 2.0];
        let moved = TriangleFrame {
            positions: [
                add(tri.positions[0], shift),
                add(tri.positions[1], shift),
                add(tri.positions[2], shift),
            ],
            normals: tri.normals,
            tangents: tri.tangents,
        };
        let out = transfer_root(binding, moved);
        assert!(close3(out, add(root, shift)), "{out:?}");
    }

    #[test]
    fn transfer_frame_is_orthonormal() {
        let tri = flat_tri();
        let binding = bind_follicle([0.3, 0.3, 0.1], tri);
        let f = transfer_frame(binding, tri);
        assert!(close(length(f.normal), 1.0));
        assert!(close(length(f.tangent), 1.0));
        assert!(close(length(f.bitangent), 1.0));
        assert!(close(dot(f.normal, f.tangent), 0.0));
        assert!(close(dot(f.normal, f.bitangent), 0.0));
        assert!(close(dot(f.tangent, f.bitangent), 0.0));
    }

    #[test]
    fn degenerate_triangle_does_not_panic() {
        let degen = TriangleFrame {
            positions: [[1.0, 1.0, 1.0]; 3],
            normals: [[0.0, 0.0, 0.0]; 3],
            tangents: [[0.0, 0.0, 0.0]; 3],
        };
        let b = compute_barycentric([2.0, 2.0, 2.0], degen);
        assert_eq!(b, Barycentric::CENTROID);
        let binding = bind_follicle([2.0, 2.0, 2.0], degen);
        let root = transfer_root(binding, degen);
        assert!(finite3(root));
        let f = transfer_frame(binding, degen);
        assert!(close(length(f.normal), 1.0));
        assert!(close(length(f.tangent), 1.0));
    }

    #[test]
    fn non_finite_root_offset_sanitized() {
        let tri = flat_tri();
        let binding = bind_follicle([f32::NAN, 0.0, 1.0], tri);
        assert!(binding.normal_offset.is_finite());
        let out = transfer_root(binding, tri);
        assert!(finite3(out));
    }

    #[test]
    fn transfer_map_matches_scalar_and_preserves_order() {
        let tri = flat_tri();
        let bindings = [
            bind_follicle([0.0, 0.0, 0.0], tri),
            bind_follicle([0.5, 0.5, 1.0], tri),
            bind_follicle([0.2, 0.1, -0.3], tri),
        ];
        let mapped = transfer_root_map(&bindings, tri);
        assert_eq!(mapped.len(), bindings.len());
        for (i, b) in bindings.iter().enumerate() {
            assert!(close3(mapped[i], transfer_root(*b, tri)));
        }
    }

    #[test]
    fn empty_map_is_empty_without_panic() {
        assert!(transfer_root_map(&[], flat_tri()).is_empty());
    }
}
