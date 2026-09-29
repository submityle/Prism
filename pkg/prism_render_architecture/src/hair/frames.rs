//! Coherent per-vertex strand frames via rotation-minimizing transport.
//!
//! A render strand is a polyline, but everything downstream that gives it
//! *width* or *anisotropy* needs a stable orthonormal frame at every control
//! point, not just a tangent:
//!
//! - **Ribbon / card expansion** offsets each control point along the frame's
//!   bitangent to build the two edges of a view-independent ribbon (the
//!   strand-to-card LOD proxy). If the offset direction twists arbitrarily
//!   from point to point, the ribbon self-folds and shimmers.
//! - **Anisotropic shading** (Kajiya-Kay / Marschner / Chiang) reads the
//!   strand tangent, but the highlight *shift* and any tangent-space texturing
//!   need the full frame, coherent along the strand and stable across frames.
//!
//! The naive per-point orthonormal basis (pick an arbitrary helper axis, cross
//! twice) flips direction wherever the tangent crosses the helper axis, so the
//! frame is discontinuous. Production hair engines (`UE5` Groom, AMD `TressFX`)
//! instead carry one reference direction *along* the strand with minimal twist.
//!
//! This module implements the **double-reflection rotation-minimizing frame**
//! (Wang et al. 2008): start from one orthonormal frame at the root, then
//! propagate the reference normal point-to-point with two reflections so it
//! never rotates about the tangent more than the curve geometry forces. The
//! result is deterministic, array-in / array-out, and panic-free on empty,
//! single-point, or degenerate (coincident-point) strands.

use alloc::vec::Vec;

use super::interpolation::Vec3;

/// Fallback tangent for degenerate strands (points that coincide or a strand
/// with a single point): an arbitrary but fixed unit direction so the frame
/// stays orthonormal and deterministic instead of collapsing.
const FALLBACK_TANGENT: Vec3 = Vec3::new(0.0, 1.0, 0.0);

/// An orthonormal frame at one strand control point.
///
/// The three axes are mutually perpendicular unit vectors with
/// `bitangent == tangent × normal`, forming a right-handed basis. `tangent`
/// runs along the strand (root → tip); `normal` is the rotation-minimizing
/// reference direction; `bitangent` is the ribbon-widening direction.
#[derive(Clone, Copy, Debug)]
pub struct StrandFrame {
    /// Unit direction along the strand at this control point (root → tip).
    pub tangent: Vec3,
    /// Rotation-minimizing reference normal, perpendicular to `tangent`.
    pub normal: Vec3,
    /// `tangent × normal`; the direction a ribbon widens along.
    pub bitangent: Vec3,
}

impl StrandFrame {
    /// The two ribbon edge positions for a control point at `center` with the
    /// given `half_width`, offset symmetrically along the bitangent.
    ///
    /// Feeds strand-to-card / ribbon LOD expansion: `half_width` is typically
    /// the strand radius, so the returned pair spans the full strand width.
    #[must_use]
    pub fn ribbon_edges(&self, center: Vec3, half_width: f32) -> (Vec3, Vec3) {
        let offset = self.bitangent.scale(half_width);
        (center - offset, center + offset)
    }
}

/// Builds a unit reference normal perpendicular to `tangent`.
///
/// Picks whichever cardinal axis is least aligned with the tangent as a helper,
/// so the cross product is well-conditioned, then normalizes. Used only to seed
/// the root frame; interior frames are transported, not rebuilt.
fn orthonormal_reference(tangent: Vec3) -> Vec3 {
    let helper = if tangent.x.abs() < 0.9 {
        Vec3::new(1.0, 0.0, 0.0)
    } else {
        Vec3::new(0.0, 1.0, 0.0)
    };
    tangent.cross(helper).normalize_or(Vec3::new(0.0, 0.0, 1.0))
}

/// Forward-difference unit tangents at each control point.
///
/// Interior and root points use the direction to the next point; the tip uses
/// the direction from the previous point. Coincident points fall back to a
/// fixed unit direction so the tangent is always defined. This matches the
/// tangent convention used by the guide-to-render interpolator.
#[must_use]
pub fn strand_tangents(points: &[Vec3]) -> Vec<Vec3> {
    let n = points.len();
    let mut tangents = Vec::with_capacity(n);
    for i in 0..n {
        let raw = if n <= 1 {
            FALLBACK_TANGENT
        } else if i + 1 < n {
            points[i + 1] - points[i]
        } else {
            points[i] - points[i - 1]
        };
        tangents.push(raw.normalize_or(FALLBACK_TANGENT));
    }
    tangents
}

/// Reflects `v` in the plane whose normal is `axis` (with `c = axis·axis`).
///
/// The double-reflection step needs `v - (2·(axis·v)/c)·axis`; the caller has
/// already checked `c` is non-degenerate.
fn reflect(v: Vec3, axis: Vec3, c: f32) -> Vec3 {
    v - axis.scale(2.0 * axis.dot(v) / c)
}

/// Computes a rotation-minimizing frame at every control point of `points`.
///
/// Returns one [`StrandFrame`] per input point (same length as `points`). The
/// root frame is seeded from an arbitrary reference normal; every later frame
/// is propagated by the double-reflection method so the normal tracks the
/// strand with minimal twist about the tangent. Each returned frame is
/// re-orthonormalized against its tangent for numerical hygiene.
///
/// Deterministic and panic-free:
/// - empty input yields an empty `Vec`;
/// - a single point yields one frame around the fallback tangent;
/// - coincident points reuse the previous reference direction rather than
///   dividing by zero.
#[must_use]
pub fn build_strand_frames(points: &[Vec3]) -> Vec<StrandFrame> {
    let n = points.len();
    if n == 0 {
        return Vec::new();
    }

    let tangents = strand_tangents(points);
    let mut frames = Vec::with_capacity(n);

    let t0 = tangents[0];
    let r0 = orthonormal_reference(t0);
    let b0 = t0.cross(r0).normalize_or(Vec3::new(0.0, 0.0, 1.0));
    frames.push(StrandFrame {
        tangent: t0,
        normal: r0,
        bitangent: b0,
    });

    for i in 0..n.saturating_sub(1) {
        let prev = frames[i];
        let t_next = tangents[i + 1];

        // Reflection 1: reflect the previous normal and tangent across the
        // plane bisecting the offset between the two control points.
        let v1 = points[i + 1] - points[i];
        let c1 = v1.dot(v1);
        let (r_l, t_l) = if c1 > f32::EPSILON {
            (reflect(prev.normal, v1, c1), reflect(prev.tangent, v1, c1))
        } else {
            // Coincident points: nothing to transport across, keep the frame.
            (prev.normal, prev.tangent)
        };

        // Reflection 2: reflect again so the transported tangent lands exactly
        // on the next tangent, carrying the normal along with minimal twist.
        let v2 = t_next - t_l;
        let c2 = v2.dot(v2);
        let r_reflected = if c2 > f32::EPSILON {
            reflect(r_l, v2, c2)
        } else {
            r_l
        };

        // Re-orthonormalize the normal against the next tangent, then rebuild
        // the bitangent, so accumulated float error never skews the basis.
        let r_perp = r_reflected - t_next.scale(t_next.dot(r_reflected));
        let normal = r_perp.normalize_or(orthonormal_reference(t_next));
        let bitangent = t_next.cross(normal).normalize_or(Vec3::new(0.0, 0.0, 1.0));

        frames.push(StrandFrame {
            tangent: t_next,
            normal,
            bitangent,
        });
    }

    frames
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

    fn assert_orthonormal(f: &StrandFrame) {
        assert!(close(f.tangent.length(), 1.0), "tangent not unit");
        assert!(close(f.normal.length(), 1.0), "normal not unit");
        assert!(close(f.bitangent.length(), 1.0), "bitangent not unit");
        assert!(close(f.tangent.dot(f.normal), 0.0), "t·n != 0");
        assert!(close(f.tangent.dot(f.bitangent), 0.0), "t·b != 0");
        assert!(close(f.normal.dot(f.bitangent), 0.0), "n·b != 0");
        // Right-handed: bitangent == tangent × normal.
        assert!(
            close_v(f.bitangent, f.tangent.cross(f.normal)),
            "not right-handed"
        );
    }

    #[test]
    fn frame_count_matches_points() {
        let points = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
        ];
        let frames = build_strand_frames(&points);
        assert_eq!(frames.len(), points.len());
    }

    #[test]
    fn straight_line_frame_is_constant() {
        // A straight strand has zero curvature, so an RMF must not rotate at
        // all: every frame equals the root frame.
        let points = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(3.0, 0.0, 0.0),
        ];
        let frames = build_strand_frames(&points);
        let first = frames[0];
        for f in &frames {
            assert_orthonormal(f);
            assert!(close_v(f.tangent, first.tangent), "tangent drifted");
            assert!(close_v(f.normal, first.normal), "normal twisted");
            assert!(close_v(f.bitangent, first.bitangent), "bitangent twisted");
        }
        // Tangent points along +x, the strand direction.
        assert!(close_v(first.tangent, Vec3::new(1.0, 0.0, 0.0)));
    }

    #[test]
    fn tangents_follow_forward_difference() {
        let points = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 2.0),
            Vec3::new(0.0, 3.0, 2.0),
        ];
        let tangents = strand_tangents(&points);
        assert_eq!(tangents.len(), 3);
        assert!(close_v(tangents[0], Vec3::new(0.0, 0.0, 1.0)));
        assert!(close_v(tangents[1], Vec3::new(0.0, 1.0, 0.0)));
        // Tip uses backward difference: same as the previous segment here.
        assert!(close_v(tangents[2], Vec3::new(0.0, 1.0, 0.0)));
    }

    #[test]
    fn planar_curve_keeps_out_of_plane_normal_constant() {
        // A curve confined to the XY plane: an RMF must not twist about the
        // tangent, so the out-of-plane reference axis (Z) stays constant along
        // the whole strand. This is the defining minimal-rotation property.
        let points = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.5, 0.0),
            Vec3::new(2.0, 1.5, 0.0),
            Vec3::new(3.0, 1.7, 0.0),
            Vec3::new(4.0, 1.2, 0.0),
        ];
        let frames = build_strand_frames(&points);
        let seed_normal = frames[0].normal;
        // Seed frame's normal is the out-of-plane axis for an in-plane tangent.
        assert!(
            close(seed_normal.z.abs(), 1.0),
            "seed normal not out-of-plane"
        );
        for f in &frames {
            assert_orthonormal(f);
            assert!(close_v(f.normal, seed_normal), "normal twisted out of RMF");
        }
    }

    #[test]
    fn empty_and_single_never_panic() {
        assert!(build_strand_frames(&[]).is_empty());

        let single = build_strand_frames(&[Vec3::new(5.0, 6.0, 7.0)]);
        assert_eq!(single.len(), 1);
        assert_orthonormal(&single[0]);
        assert!(close_v(single[0].tangent, FALLBACK_TANGENT));
    }

    #[test]
    fn coincident_points_never_panic_and_stay_orthonormal() {
        let points = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let frames = build_strand_frames(&points);
        assert_eq!(frames.len(), 4);
        for f in &frames {
            assert_orthonormal(f);
        }
    }

    #[test]
    fn ribbon_edges_are_symmetric_and_span_width() {
        let frames = build_strand_frames(&[Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)]);
        let f = frames[0];
        let center = Vec3::new(1.0, 2.0, 3.0);
        let (left, right) = f.ribbon_edges(center, 0.25);
        // Midpoint is the center.
        let mid = (left + right).scale(0.5);
        assert!(close_v(mid, center), "ribbon not centered");
        // Full span equals twice the half-width.
        assert!(close((right - left).length(), 0.5), "ribbon width wrong");
        // Edges lie along the bitangent.
        let dir = (right - left).normalize_or(Vec3::ZERO);
        assert!(
            close(dir.dot(f.bitangent).abs(), 1.0),
            "edges off bitangent"
        );
    }

    fn bits_v(v: Vec3) -> [u32; 3] {
        [v.x.to_bits(), v.y.to_bits(), v.z.to_bits()]
    }

    #[test]
    fn frames_are_deterministic() {
        let points = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.4, 1.0, 0.2),
            Vec3::new(1.0, 1.8, 0.9),
            Vec3::new(1.3, 2.9, 1.1),
        ];
        let a = build_strand_frames(&points);
        let b = build_strand_frames(&points);
        assert_eq!(a.len(), b.len());
        // Bit-exact reproducibility (Vec3 has no PartialEq; compare raw bits).
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(bits_v(x.tangent), bits_v(y.tangent));
            assert_eq!(bits_v(x.normal), bits_v(y.normal));
            assert_eq!(bits_v(x.bitangent), bits_v(y.bitangent));
        }
    }
}
