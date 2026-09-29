//! Mesh-tier LOD proxy geometry: the coarsest rung of the hair LOD ladder.
//!
//! The [`super::lod`] ladder degrades a strand through four rungs as it recedes
//! from the camera: full sub-pixel strands, reduced strands, flat `Cards`
//! ribbons ([`super::ribbon`]), and finally a solid **`Mesh`** shell — a low
//! triangle-count tube swept along the strand centerline. At `Mesh` range a
//! group of hairs is far enough that individual fibers are indistinguishable,
//! so `UE5` Groom and comparable engines collapse them onto baked shell
//! geometry that reads as a silhouette and catches light cheaply. This module
//! builds that shell on the CPU, deterministically.
//!
//! Unlike the flat ribbon, the shell has volume: each control point contributes
//! a four-corner rectangular cross-section oriented by the strand's coherent
//! frame ([`super::frames`]) — the bitangent spans the width, the normal spans
//! the thickness. Consecutive sections are stitched into a closed box tube with
//! flat root/tip caps. A rectangular (rather than round) section is deliberate:
//! it needs no trigonometry, keeps the vertex budget at four per ring, and the
//! diagonal per-corner normals are more than enough fidelity for geometry that
//! only ever appears at distance.
//!
//! Output is an indexed triangle list with parallel position / normal / UV
//! arrays, ready for the visibility pass. Everything is array-in / array-out
//! and panic-free: mismatched or too-short input yields an empty mesh.

use alloc::vec::Vec;

use super::frames::StrandFrame;
use super::groom_import::StrandAttributes;
use super::interpolation::Vec3;

/// Per-corner `(width_sign, thickness_sign)` walked once around the rectangular
/// cross-section. Corner `j` sits at `width_sign * bitangent + thickness_sign *
/// normal`; the order traces the rectangle so consecutive corners share an
/// edge, which the side-face stitching below relies on for outward winding.
const CORNER_SIGNS: [(f32, f32); 4] = [(1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)];

/// An indexed triangle-list shell tube for one strand.
///
/// Vertices are laid out four per control point ("ring"): control point `r`
/// owns vertices `4*r .. 4*r + 4`, one per rectangle corner in [`CORNER_SIGNS`]
/// order. The three per-vertex arrays run parallel and share a length of
/// `4 * point_count`; `indices` addresses them as a triangle list of
/// `8 * (point_count - 1) + 4` triangles (eight side triangles per segment plus
/// two triangles for each of the root and tip caps).
#[derive(Clone, Debug, Default)]
pub struct ShellMesh {
    /// Shell surface positions; four per control point, one per corner.
    pub positions: Vec<Vec3>,
    /// Outward per-vertex normals; parallel to `positions`.
    pub normals: Vec<Vec3>,
    /// Per-vertex UVs: `u` walks the cross-section perimeter in quarter steps
    /// (`0`, `0.25`, `0.5`, `0.75`), `v` runs `0` at the root to `1` at the tip
    /// along arc length.
    pub uvs: Vec<[f32; 2]>,
    /// Triangle-list indices into the vertex arrays.
    pub indices: Vec<u32>,
}

impl ShellMesh {
    /// Number of vertices (`4 * point_count`).
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    /// Number of triangles (`indices.len() / 3`).
    #[must_use]
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Returns `true` when the shell has no geometry.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }
}

/// Builds a shell tube with a uniform rectangular cross-section.
///
/// `half_width` is the half-extent along each frame's bitangent, `half_thickness`
/// the half-extent along each frame's normal. `points` and `frames` must
/// describe the same strand and share a length of at least two; otherwise a
/// length mismatch or a single point yields an empty [`ShellMesh`] rather than
/// panicking.
#[must_use]
pub fn build_shell(
    points: &[Vec3],
    frames: &[StrandFrame],
    half_width: f32,
    half_thickness: f32,
) -> ShellMesh {
    let n = points.len();
    if n < 2 {
        return ShellMesh::default();
    }
    let mut sections = Vec::with_capacity(n);
    for _ in 0..n {
        sections.push((half_width, half_thickness));
    }
    build_from_sections(points, frames, &sections)
}

/// Builds a shell tube whose cross-section tapers from authored attributes.
///
/// The half-width and half-thickness at each ring are both
/// [`StrandAttributes::radius_at`] evaluated at that ring's normalized
/// arc-length position (`0` = root, `1` = tip), producing a square section that
/// narrows from root to tip exactly as the groom was authored. Falls back to an
/// empty mesh on mismatched or too-short input, like [`build_shell`].
#[must_use]
pub fn build_shell_tapered(
    points: &[Vec3],
    frames: &[StrandFrame],
    attributes: &StrandAttributes,
) -> ShellMesh {
    let n = points.len();
    if n < 2 {
        return ShellMesh::default();
    }
    let mut sections = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f32 / ((n - 1) as f32);
        let r = attributes.radius_at(t);
        sections.push((r, r));
    }
    build_from_sections(points, frames, &sections)
}

/// Shared core: builds the shell from per-ring `(half_width, half_thickness)`
/// sections. Returns an empty mesh when the parallel arrays disagree in length
/// or the strand is too short.
fn build_from_sections(
    points: &[Vec3],
    frames: &[StrandFrame],
    sections: &[(f32, f32)],
) -> ShellMesh {
    let n = points.len();
    if n < 2 || frames.len() != n || sections.len() != n {
        return ShellMesh::default();
    }

    // Arc length along the centerline for the v coordinate.
    let mut cumulative = Vec::with_capacity(n);
    cumulative.push(0.0_f32);
    let mut total = 0.0_f32;
    for i in 1..n {
        total += (points[i] - points[i - 1]).length();
        cumulative.push(total);
    }

    let mut positions = Vec::with_capacity(4 * n);
    let mut normals = Vec::with_capacity(4 * n);
    let mut uvs = Vec::with_capacity(4 * n);
    for i in 0..n {
        let frame = &frames[i];
        let (half_width, half_thickness) = sections[i];
        let v = if total > 0.0 {
            cumulative[i] / total
        } else {
            // Zero-length strand: fall back to even parameter spacing.
            i as f32 / ((n - 1) as f32)
        };
        for (j, &(sign_w, sign_h)) in CORNER_SIGNS.iter().enumerate() {
            let offset = frame.bitangent.scale(sign_w * half_width)
                + frame.normal.scale(sign_h * half_thickness);
            positions.push(points[i] + offset);
            // Diagonal outward normal at the rectangle corner; the ±bitangent /
            // ±normal sum has length sqrt(2) for orthonormal frames, so the
            // normalize is always well-defined (fallback guards degenerate frames).
            let outward = frame.bitangent.scale(sign_w) + frame.normal.scale(sign_h);
            normals.push(outward.normalize_or(frame.normal));
            uvs.push([j as f32 * 0.25, v]);
        }
    }

    // Eight side triangles per segment plus two triangles for each cap.
    let mut indices = Vec::with_capacity(3 * (8 * (n - 1) + 4));
    for r in 0..n - 1 {
        let base = 4 * r;
        let next = 4 * (r + 1);
        for s in 0..4usize {
            let sn = (s + 1) % 4;
            let a = (base + s) as u32;
            let b = (base + sn) as u32;
            let c = (next + sn) as u32;
            let d = (next + s) as u32;
            // Quad (a, b, c, d) wound so its face normal points outward.
            indices.extend_from_slice(&[a, b, c, a, c, d]);
        }
    }
    // Root cap (ring 0): wound so its face normal points along -tangent.
    indices.extend_from_slice(&[0, 2, 1, 0, 3, 2]);
    // Tip cap (last ring): reversed winding, face normal along +tangent.
    let tip = (4 * (n - 1)) as u32;
    indices.extend_from_slice(&[tip, tip + 1, tip + 2, tip, tip + 2, tip + 3]);

    ShellMesh {
        positions,
        normals,
        uvs,
        indices,
    }
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
        let mesh = build_shell(&points, &frames, 0.1, 0.05);
        // Four vertices per ring, four rings.
        assert_eq!(mesh.vertex_count(), 16);
        assert_eq!(mesh.positions.len(), mesh.normals.len());
        assert_eq!(mesh.positions.len(), mesh.uvs.len());
        // Eight side triangles per of three segments, plus two caps of two.
        assert_eq!(mesh.triangle_count(), 8 * 3 + 4);
        assert_eq!(mesh.indices.len(), 3 * (8 * 3 + 4));
        assert!(!mesh.is_empty());
    }

    #[test]
    fn corners_sit_on_the_section_box() {
        let (points, frames) = straight_strand();
        let hw = 0.2_f32;
        let ht = 0.07_f32;
        let mesh = build_shell(&points, &frames, hw, ht);
        for i in 0..points.len() {
            let frame = &frames[i];
            for (j, &(sign_w, sign_h)) in CORNER_SIGNS.iter().enumerate() {
                let offset = mesh.positions[4 * i + j] - points[i];
                // The offset decomposes exactly onto the frame axes.
                assert!(
                    close(offset.dot(frame.bitangent), sign_w * hw),
                    "width extent"
                );
                assert!(
                    close(offset.dot(frame.normal), sign_h * ht),
                    "thickness extent"
                );
                assert!(close(offset.dot(frame.tangent), 0.0), "no tangent drift");
            }
        }
    }

    #[test]
    fn corners_are_symmetric_about_centerline() {
        let (points, frames) = straight_strand();
        let mesh = build_shell(&points, &frames, 0.15, 0.15);
        for (i, p) in points.iter().enumerate() {
            // Corner 0 and corner 2 are diagonally opposite, as are 1 and 3.
            let c0 = mesh.positions[4 * i];
            let c2 = mesh.positions[4 * i + 2];
            assert!(close_v((c0 + c2).scale(0.5), *p), "0/2 not centered");
            let c1 = mesh.positions[4 * i + 1];
            let c3 = mesh.positions[4 * i + 3];
            assert!(close_v((c1 + c3).scale(0.5), *p), "1/3 not centered");
        }
    }

    #[test]
    fn normals_are_unit_length() {
        let (points, frames) = straight_strand();
        let mesh = build_shell(&points, &frames, 0.1, 0.03);
        for nrm in &mesh.normals {
            assert!(close(nrm.length(), 1.0), "normal not unit length");
        }
    }

    #[test]
    fn uv_layout_walks_perimeter_and_arc() {
        let (points, frames) = straight_strand();
        let mesh = build_shell(&points, &frames, 0.1, 0.1);
        // u steps a quarter of the perimeter per corner.
        for i in 0..points.len() {
            for j in 0..4 {
                assert!(
                    close(mesh.uvs[4 * i + j][0], j as f32 * 0.25),
                    "u perimeter"
                );
            }
        }
        // v is 0 at the root ring and 1 at the tip ring, non-decreasing.
        assert!(close(mesh.uvs[0][1], 0.0), "root v != 0");
        let last = mesh.uvs.len() - 1;
        assert!(close(mesh.uvs[last][1], 1.0), "tip v != 1");
        let mut prev = -1.0_f32;
        for uv in &mesh.uvs {
            assert!(uv[1] >= prev - EPS, "v decreased along the shell");
            prev = uv[1];
        }
    }

    #[test]
    fn tapered_section_hits_authored_radii() {
        let (points, frames) = straight_strand();
        let attributes = StrandAttributes {
            root_radius: 0.3,
            tip_radius: 0.05,
            root_uv: [0.0, 0.0],
            seed: 0,
        };
        let mesh = build_shell_tapered(&points, &frames, &attributes);
        // Root ring (i = 0) uses the root radius for both extents.
        let root_off = mesh.positions[0] - points[0];
        assert!(
            close(root_off.dot(frames[0].bitangent).abs(), 0.3),
            "root width"
        );
        assert!(
            close(root_off.dot(frames[0].normal).abs(), 0.3),
            "root thickness"
        );
        // Tip ring uses the tip radius.
        let last_ring = points.len() - 1;
        let tip_off = mesh.positions[4 * last_ring] - points[last_ring];
        assert!(
            close(tip_off.dot(frames[last_ring].bitangent).abs(), 0.05),
            "tip width"
        );
    }

    #[test]
    fn build_is_deterministic() {
        let (points, frames) = straight_strand();
        let a = build_shell(&points, &frames, 0.12, 0.04);
        let b = build_shell(&points, &frames, 0.12, 0.04);
        assert_eq!(a.indices, b.indices);
        assert_eq!(a.positions.len(), b.positions.len());
        for (p, q) in a.positions.iter().zip(b.positions.iter()) {
            // interpolation::Vec3 has no PartialEq; compare bit patterns.
            assert_eq!(p.x.to_bits(), q.x.to_bits());
            assert_eq!(p.y.to_bits(), q.y.to_bits());
            assert_eq!(p.z.to_bits(), q.z.to_bits());
        }
    }

    #[test]
    fn indices_stay_in_range() {
        let (points, frames) = straight_strand();
        let mesh = build_shell(&points, &frames, 0.1, 0.1);
        let vertex_count = mesh.vertex_count() as u32;
        for &idx in &mesh.indices {
            assert!(idx < vertex_count, "index out of range");
        }
    }

    #[test]
    fn degenerate_input_yields_empty_mesh() {
        let single = [Vec3::new(0.0, 0.0, 0.0)];
        let frames = build_strand_frames(&single);
        assert!(build_shell(&single, &frames, 0.1, 0.1).is_empty());

        let empty: [Vec3; 0] = [];
        let empty_frames: Vec<StrandFrame> = Vec::new();
        assert!(build_shell(&empty, &empty_frames, 0.1, 0.1).is_empty());

        // Length mismatch between points and frames is rejected, not panicked.
        let (points, _) = straight_strand();
        let short_frames = build_strand_frames(&points[..2]);
        assert!(build_shell(&points, &short_frames, 0.1, 0.1).is_empty());
    }
}
