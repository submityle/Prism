//! Strand-to-ribbon meshing: the `Cards` LOD proxy geometry.
//!
//! Once a render strand is too thin (or too distant) to justify a full
//! sub-pixel software raster, the [`super::lod`] ladder swaps it for a
//! camera-card / ribbon proxy: a flat quad strip that follows the strand's
//! centerline and carries its width. This module builds that ribbon on the
//! CPU, deterministically, from three inputs that the earlier pipeline stages
//! already produce:
//!
//! - the strand control points (render strands from [`super::interpolation`]),
//! - the coherent per-vertex frames from [`super::frames`] (the bitangent is
//!   the widening direction; the tangent rides along for anisotropic shading),
//! - a per-point radius, either given directly or tapered from the authored
//!   [`super::groom_import::StrandAttributes`] root/tip radii.
//!
//! The ribbon is *view-independent*: it widens along the strand's own
//! rotation-minimizing bitangent rather than toward the camera, so the mesh is
//! stable across frames and can be built once and cached. (A view-facing
//! billboard variant is a GPU concern and is not owned here.) Output is a plain
//! indexed triangle list — flat position/tangent/UV arrays plus `u32` indices —
//! ready to hand to the visibility pass. Everything is array-in / array-out and
//! panic-free: mismatched or too-short inputs yield an empty mesh.

use alloc::vec::Vec;

use super::frames::StrandFrame;
use super::groom_import::StrandAttributes;
use super::interpolation::Vec3;

/// An indexed triangle-list ribbon for one strand.
///
/// Vertices are laid out two per control point: for control point `i`, the left
/// edge is vertex `2*i` and the right edge is vertex `2*i + 1`. The three
/// per-vertex arrays run parallel and share a length of `2 * point_count`;
/// `indices` addresses them as a triangle list (`6 * (point_count - 1)`
/// entries, two triangles per segment).
#[derive(Clone, Debug, Default)]
pub struct RibbonMesh {
    /// Ribbon edge positions; two per control point (left, then right).
    pub positions: Vec<Vec3>,
    /// Strand tangent per vertex, for anisotropic shading; parallel to `positions`.
    pub tangents: Vec<Vec3>,
    /// Per-vertex UVs: `u` is `0` on the left edge and `1` on the right, `v`
    /// runs `0` at the root to `1` at the tip along arc length.
    pub uvs: Vec<[f32; 2]>,
    /// Triangle-list indices into the vertex arrays.
    pub indices: Vec<u32>,
}

impl RibbonMesh {
    /// Number of vertices (`2 * point_count`).
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    /// Number of triangles (`indices.len() / 3`).
    #[must_use]
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Returns `true` when the ribbon has no geometry.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }
}

/// Builds a ribbon from explicit per-point radii.
///
/// `points`, `frames`, and `radii` must all describe the same strand and share
/// a length of at least two; otherwise (length mismatch or a single point) an
/// empty [`RibbonMesh`] is returned rather than panicking. Each control point
/// contributes two edge vertices offset `±radius` along the frame bitangent;
/// consecutive points are stitched with two triangles of consistent winding.
#[must_use]
pub fn build_ribbon(points: &[Vec3], frames: &[StrandFrame], radii: &[f32]) -> RibbonMesh {
    let n = points.len();
    if n < 2 || frames.len() != n || radii.len() != n {
        return RibbonMesh::default();
    }

    // Arc length along the centerline for the v coordinate.
    let mut cumulative = Vec::with_capacity(n);
    cumulative.push(0.0_f32);
    let mut total = 0.0_f32;
    for i in 1..n {
        total += (points[i] - points[i - 1]).length();
        cumulative.push(total);
    }

    let mut positions = Vec::with_capacity(2 * n);
    let mut tangents = Vec::with_capacity(2 * n);
    let mut uvs = Vec::with_capacity(2 * n);
    for i in 0..n {
        let offset = frames[i].bitangent.scale(radii[i]);
        positions.push(points[i] - offset);
        positions.push(points[i] + offset);
        tangents.push(frames[i].tangent);
        tangents.push(frames[i].tangent);
        let v = if total > 0.0 {
            cumulative[i] / total
        } else {
            // Zero-length strand: fall back to even parameter spacing.
            i as f32 / ((n - 1) as f32)
        };
        uvs.push([0.0, v]);
        uvs.push([1.0, v]);
    }

    let mut indices = Vec::with_capacity(6 * (n - 1));
    for i in 0..n - 1 {
        let l0 = (2 * i) as u32;
        let r0 = l0 + 1;
        let l1 = l0 + 2;
        let r1 = l0 + 3;
        // Two triangles per segment, consistent (l0, r0, l1) / (r0, r1, l1).
        indices.extend_from_slice(&[l0, r0, l1, r0, r1, l1]);
    }

    RibbonMesh {
        positions,
        tangents,
        uvs,
        indices,
    }
}

/// Builds a ribbon whose per-point radius is tapered from authored attributes.
///
/// The radius at each control point is [`StrandAttributes::radius_at`] evaluated
/// at that point's normalized arc-length position (`0` = root, `1` = tip), so
/// the ribbon narrows from root to tip exactly as the groom was authored. Falls
/// back to an empty mesh on mismatched or too-short input, like [`build_ribbon`].
#[must_use]
pub fn build_ribbon_tapered(
    points: &[Vec3],
    frames: &[StrandFrame],
    attributes: &StrandAttributes,
) -> RibbonMesh {
    let n = points.len();
    if n < 2 || frames.len() != n {
        return RibbonMesh::default();
    }
    let mut radii = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f32 / ((n - 1) as f32);
        radii.push(attributes.radius_at(t));
    }
    build_ribbon(points, frames, &radii)
}

#[cfg(test)]
mod tests {
    use super::super::frames::build_strand_frames;
    use super::*;

    const EPS: f32 = 1e-4;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    fn close_v(a: Vec3, b: Vec3) -> bool {
        close(a.x, b.x) && close(a.y, b.y) && close(a.z, b.z)
    }

    fn straight_strand() -> ([Vec3; 4], Vec<StrandFrame>) {
        let points = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(0.0, 3.0, 0.0),
        ];
        let frames = build_strand_frames(&points);
        (points, frames)
    }

    #[test]
    fn counts_are_consistent() {
        let (points, frames) = straight_strand();
        let radii = [0.1_f32; 4];
        let mesh = build_ribbon(&points, &frames, &radii);
        assert_eq!(mesh.vertex_count(), 8);
        assert_eq!(mesh.positions.len(), mesh.tangents.len());
        assert_eq!(mesh.positions.len(), mesh.uvs.len());
        assert_eq!(mesh.indices.len(), 6 * 3);
        assert_eq!(mesh.triangle_count(), 2 * 3);
        assert!(!mesh.is_empty());
    }

    #[test]
    fn edges_are_symmetric_about_centerline_and_span_width() {
        let (points, frames) = straight_strand();
        let radii = [0.2_f32; 4];
        let mesh = build_ribbon(&points, &frames, &radii);
        for i in 0..points.len() {
            let left = mesh.positions[2 * i];
            let right = mesh.positions[2 * i + 1];
            let mid = (left + right).scale(0.5);
            assert!(close_v(mid, points[i]), "edge midpoint off centerline");
            assert!(close((right - left).length(), 0.4), "width != 2*radius");
        }
    }

    #[test]
    fn uv_v_is_monotonic_root_to_tip() {
        let (points, frames) = straight_strand();
        let radii = [0.1_f32; 4];
        let mesh = build_ribbon(&points, &frames, &radii);
        assert!(close(mesh.uvs[0][1], 0.0), "root v != 0");
        let last = mesh.uvs.len() - 1;
        assert!(close(mesh.uvs[last][1], 1.0), "tip v != 1");
        // u is 0 on the left edge, 1 on the right edge.
        for pair in mesh.uvs.chunks_exact(2) {
            assert!(close(pair[0][0], 0.0));
            assert!(close(pair[1][0], 1.0));
        }
        // v never decreases.
        let mut prev = -1.0_f32;
        for uv in mesh.uvs.chunks_exact(2) {
            let v = uv[0][1];
            assert!(v + EPS >= prev, "v decreased");
            prev = v;
        }
    }

    #[test]
    fn all_indices_are_in_range() {
        let (points, frames) = straight_strand();
        let radii = [0.1_f32; 4];
        let mesh = build_ribbon(&points, &frames, &radii);
        let vcount = mesh.vertex_count() as u32;
        for &idx in &mesh.indices {
            assert!(idx < vcount, "index out of range");
        }
    }

    #[test]
    fn tapered_radius_narrows_to_tip() {
        let (points, frames) = straight_strand();
        let attrs = StrandAttributes::new(0.3, 0.05, [0.0, 0.0], 0);
        let mesh = build_ribbon_tapered(&points, &frames, &attrs);
        // Root width uses root radius; tip width uses tip radius.
        let root_w = (mesh.positions[1] - mesh.positions[0]).length();
        let n = points.len();
        let tip_w = (mesh.positions[2 * n - 1] - mesh.positions[2 * n - 2]).length();
        assert!(close(root_w, 0.6), "root width wrong: {root_w}");
        assert!(close(tip_w, 0.1), "tip width wrong: {tip_w}");
        assert!(tip_w < root_w, "ribbon did not narrow");
    }

    #[test]
    fn degenerate_inputs_yield_empty_mesh() {
        // Fewer than two points.
        let one = [Vec3::new(0.0, 0.0, 0.0)];
        let f1 = build_strand_frames(&one);
        assert!(build_ribbon(&one, &f1, &[0.1]).is_empty());

        // Length mismatch: frames/radii do not match points.
        let (points, frames) = straight_strand();
        assert!(build_ribbon(&points, &frames, &[0.1, 0.1]).is_empty());
        assert!(build_ribbon(&points, &frames[..2], &[0.1; 4]).is_empty());
    }

    #[test]
    fn zero_length_strand_uses_even_v_spacing_without_panic() {
        let points = [Vec3::new(1.0, 1.0, 1.0); 3];
        let frames = build_strand_frames(&points);
        let mesh = build_ribbon(&points, &frames, &[0.1; 3]);
        assert_eq!(mesh.vertex_count(), 6);
        assert!(close(mesh.uvs[0][1], 0.0));
        assert!(close(mesh.uvs[2][1], 0.5));
        assert!(close(mesh.uvs[4][1], 1.0));
    }

    #[test]
    fn ribbon_is_deterministic() {
        let (points, frames) = straight_strand();
        let radii = [0.15_f32; 4];
        let a = build_ribbon(&points, &frames, &radii);
        let b = build_ribbon(&points, &frames, &radii);
        assert_eq!(a.indices, b.indices);
        assert_eq!(a.positions.len(), b.positions.len());
        for (x, y) in a.positions.iter().zip(&b.positions) {
            assert_eq!(x.x.to_bits(), y.x.to_bits());
            assert_eq!(x.y.to_bits(), y.y.to_bits());
            assert_eq!(x.z.to_bits(), y.z.to_bits());
        }
    }
}
