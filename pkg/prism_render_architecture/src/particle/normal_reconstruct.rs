//! View-space normal reconstruction from a linear depth buffer for the particle
//! shading and screen-space effects passes (design §16-§21).
//!
//! Deferred and screen-space effects (`SSAO`, contact shadows, screen-space
//! reflections, edge-aware blurs) frequently need a per-pixel *view-space
//! normal* but only have a depth buffer to work from — reconstructing the
//! surface orientation from depth avoids paying for a dedicated normal
//! `G-buffer` target. This module owns the `CPU`-verifiable reference for that
//! reconstruction so a future `GPU` kernel can match it bit for bit. It mirrors
//! the improved depth-to-normal technique popularised by Valve (and used by
//! Unreal's screen-space passes) at the algorithm level, without reusing any
//! vendor code.
//!
//! # Pipeline: depth → view position → cross product
//! 1. **Unproject.** A pixel is addressed by its `UV` in `0..=1`. Turning the
//!    `UV` into normalized device coordinates (`ndc = uv * 2 - 1`) and scaling
//!    by the frustum half-extents at the sampled linear depth reprojects it to
//!    a view-space position ([`view_pos_from_depth`]). The half-extents are the
//!    *precomputed* `tan(FOV/2)` scalars passed in through
//!    [`NormalReconstructParams`]; this module never evaluates a tangent itself,
//!    keeping the math to linear/rational operations only.
//! 2. **Differentiate.** Two view-space edge vectors are formed from neighboring
//!    pixels (a horizontal and a vertical difference). Because the reprojection
//!    is affine in `UV` at a fixed depth and linear in depth, these differences
//!    approximate the surface tangents.
//! 3. **Cross.** The surface normal is `normalize(cross(ddx, ddy))`. With `+u`
//!    to the right and `+v` upward the horizontal difference points roughly
//!    along `+x` and the vertical along `+y`, so a plane squarely facing the
//!    camera reconstructs the `+Z` view-space normal (pointing back toward the
//!    camera at the origin).
//!
//! # Naive vs. improved (4-tap) reconstruction
//! [`reconstruct_normal_naive`] uses a single forward difference on the right
//! and up neighbors. It is cheap but produces a badly skewed normal wherever one
//! of those two neighbors straddles a depth discontinuity (a silhouette edge),
//! because the edge vector then spans two unrelated surfaces.
//!
//! [`reconstruct_normal_improved`] samples four neighbors (left/right and
//! down/up) and, for each axis, keeps the side whose depth is *closest* to the
//! center depth — the side most likely to lie on the same surface. This is the
//! Valve "best 4-tap" selection: it suppresses the false normals that appear
//! along depth edges while costing only two extra depth fetches.
//!
//! # Deliberately out of scope
//! This module reconstructs a normal *from depth* only. It does **not** build
//! per-particle `billboard` orientation frames (the camera-/velocity-facing
//! `right`/`up` basis lives in [`super::orientation_basis`] and is a different
//! problem — orienting a sprite quad, not reading geometry from depth). It does
//! **not** evaluate signed-distance-field gradients ([`super::sdf`]) and does
//! **not** touch irradiance-probe spherical harmonics ([`super::gi_probe`]). It
//! imports nothing beyond the shared `std430` layout helpers.
//!
//! # Determinism
//! The only non-`+ - * /` primitive used is `f32::sqrt` (inside
//! [`normalize3`]); there are no transcendental functions and no platform math,
//! so the reference is deterministic. `f32` values are never compared with
//! `==`/`!=`; comparisons go through [`CMP_EPS`] or ordinary `<` ordering.

use alloc::vec::Vec;

use crate::particle::gpu_layout::{storage_bytes, VEC4_STRIDE};

/// Epsilon for tolerant `f32` comparisons; direct `==`/`!=` is forbidden.
///
/// Also serves as the squared-length floor below which [`normalize3`] treats a
/// vector as degenerate and returns the zero vector instead of dividing.
pub const CMP_EPS: f32 = 1e-6;

/// Byte stride of one [`NormalReconstructParams`] record in a `std430` storage
/// buffer: the two half-extent scalars pad up to a single `vec4` slot.
pub const NORMAL_PARAMS_STRIDE: usize = VEC4_STRIDE;

/// Subtracts two 3-component vectors component-wise.
#[must_use]
fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Right-handed cross product of two 3-component vectors.
#[must_use]
fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Normalizes a 3-component vector, guarding against division by zero.
///
/// Returns the input scaled to unit length. When the squared length falls below
/// [`CMP_EPS`] the vector is treated as degenerate and the zero vector is
/// returned, so a collapsed edge (for example a perfectly flat run of equal
/// depths that cancels the cross product) yields `[0, 0, 0]` rather than a
/// `NaN`. This is the only place `f32::sqrt` is used.
#[must_use]
pub fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    if len_sq < CMP_EPS {
        return [0.0, 0.0, 0.0];
    }
    let inv_len = 1.0 / len_sq.sqrt();
    [v[0] * inv_len, v[1] * inv_len, v[2] * inv_len]
}

/// Frustum reprojection parameters for turning depth into a view-space position.
///
/// Both fields are the *precomputed* `tan(FOV/2)` half-extents of the view
/// frustum — horizontal and vertical — supplied by the caller so this module
/// never evaluates a tangent. At a linear view depth `d`, a pixel at the edge of
/// the frustum (`ndc = ±1`) sits at `±half_tan * d` in view space, which is
/// exactly what [`view_pos_from_depth`] scales by.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NormalReconstructParams {
    /// Precomputed `tan(FOV_x / 2)`: the view-space half-width at unit depth.
    pub half_tan_fov_x: f32,
    /// Precomputed `tan(FOV_y / 2)`: the view-space half-height at unit depth.
    pub half_tan_fov_y: f32,
}

impl NormalReconstructParams {
    /// Builds a parameter set from the two precomputed half-extent scalars.
    #[must_use]
    pub const fn new(half_tan_fov_x: f32, half_tan_fov_y: f32) -> Self {
        Self {
            half_tan_fov_x,
            half_tan_fov_y,
        }
    }

    /// Packs the parameters into their `std430` `vec4`-aligned word layout.
    ///
    /// Layout: `[half_tan_fov_x, half_tan_fov_y, pad, pad]` as raw `u32` words
    /// (the two `f32` fields via `f32::to_bits`), filling one `vec4` slot that
    /// matches [`NORMAL_PARAMS_STRIDE`]. The trailing words are padding.
    #[must_use]
    pub fn to_std430(&self) -> [u32; 4] {
        [
            self.half_tan_fov_x.to_bits(),
            self.half_tan_fov_y.to_bits(),
            0,
            0,
        ]
    }
}

/// The five depth samples the improved reconstruction reads: the center pixel
/// and its four axis-aligned neighbors.
///
/// All depths are *linear view depths* (positive distance in front of the
/// camera), matching the `linear_depth` argument of [`view_pos_from_depth`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DepthTaps {
    /// Linear depth at the pixel being shaded.
    pub center: f32,
    /// Linear depth one pixel to the left (`-u`).
    pub left: f32,
    /// Linear depth one pixel to the right (`+u`).
    pub right: f32,
    /// Linear depth one pixel down (`-v`).
    pub down: f32,
    /// Linear depth one pixel up (`+v`).
    pub up: f32,
}

impl DepthTaps {
    /// Gathers the five neighborhood depths into one record.
    #[must_use]
    pub const fn new(center: f32, left: f32, right: f32, down: f32, up: f32) -> Self {
        Self {
            center,
            left,
            right,
            down,
            up,
        }
    }
}

/// Reprojects a pixel to its view-space position from a linear depth.
///
/// `uv` addresses the pixel in `0..=1`; `linear_depth` is the positive distance
/// in front of the camera. The `UV` is mapped to normalized device coordinates
/// (`ndc = uv * 2 - 1`) and scaled by the frustum half-extents at that depth,
/// with the camera looking down `-Z`, so the returned position is
/// `[ndc_x * half_tan_fov_x * d, ndc_y * half_tan_fov_y * d, -d]`. The
/// arithmetic is purely linear/rational — no tangent is evaluated here.
#[must_use]
pub fn view_pos_from_depth(
    uv: [f32; 2],
    linear_depth: f32,
    params: &NormalReconstructParams,
) -> [f32; 3] {
    let ndc_x = uv[0] * 2.0 - 1.0;
    let ndc_y = uv[1] * 2.0 - 1.0;
    let x = ndc_x * params.half_tan_fov_x * linear_depth;
    let y = ndc_y * params.half_tan_fov_y * linear_depth;
    let z = -linear_depth;
    [x, y, z]
}

/// Reconstructs a view-space normal with the naive two-tap forward difference.
///
/// The horizontal edge vector is `P(right) - P(center)` and the vertical edge
/// vector is `P(up) - P(center)`, each reprojected through
/// [`view_pos_from_depth`]. The normal is `normalize(cross(ddx, ddy))`, which
/// points along `+Z` for a plane squarely facing the camera. `texel` is the
/// `UV`-space step to the neighbor `(du, dv)`.
///
/// This is the cheap reference: it always trusts the right and up neighbors, so
/// it produces a skewed normal when either neighbor lies across a depth
/// discontinuity. Use [`reconstruct_normal_improved`] near silhouette edges.
#[must_use]
pub fn reconstruct_normal_naive(
    uv: [f32; 2],
    texel: [f32; 2],
    depth_center: f32,
    depth_right: f32,
    depth_up: f32,
    params: &NormalReconstructParams,
) -> [f32; 3] {
    let p_c = view_pos_from_depth(uv, depth_center, params);
    let p_r = view_pos_from_depth([uv[0] + texel[0], uv[1]], depth_right, params);
    let p_u = view_pos_from_depth([uv[0], uv[1] + texel[1]], depth_up, params);
    let ddx = sub3(p_r, p_c);
    let ddy = sub3(p_u, p_c);
    normalize3(cross3(ddx, ddy))
}

/// Reconstructs a view-space normal with the improved Valve 4-tap selection.
///
/// For each axis the neighbor whose linear depth is *closest* to the center
/// depth is kept — the side most likely to sit on the same surface — and the
/// edge vector is oriented to point along `+x` (horizontal) or `+y` (vertical):
///
/// - Horizontal: if `|center - left| < |right - center|` use `P(center) -
///   P(left)`, else `P(right) - P(center)`.
/// - Vertical: if `|center - down| < |up - center|` use `P(center) - P(down)`,
///   else `P(up) - P(center)`.
///
/// The normal is `normalize(cross(ddx, ddy))`. Choosing the nearer neighbor
/// keeps the edge vectors on the shaded surface, so the reconstruction stays
/// correct across depth discontinuities where [`reconstruct_normal_naive`]
/// breaks down. `texel` is the `UV`-space step to a neighbor `(du, dv)`.
#[must_use]
pub fn reconstruct_normal_improved(
    uv: [f32; 2],
    texel: [f32; 2],
    taps: &DepthTaps,
    params: &NormalReconstructParams,
) -> [f32; 3] {
    let p_c = view_pos_from_depth(uv, taps.center, params);
    let p_l = view_pos_from_depth([uv[0] - texel[0], uv[1]], taps.left, params);
    let p_r = view_pos_from_depth([uv[0] + texel[0], uv[1]], taps.right, params);
    let p_d = view_pos_from_depth([uv[0], uv[1] - texel[1]], taps.down, params);
    let p_u = view_pos_from_depth([uv[0], uv[1] + texel[1]], taps.up, params);

    let left_closer = (taps.center - taps.left).abs() < (taps.right - taps.center).abs();
    let ddx = if left_closer {
        sub3(p_c, p_l)
    } else {
        sub3(p_r, p_c)
    };

    let down_closer = (taps.center - taps.down).abs() < (taps.up - taps.center).abs();
    let ddy = if down_closer {
        sub3(p_c, p_d)
    } else {
        sub3(p_u, p_c)
    };

    normalize3(cross3(ddx, ddy))
}

/// Packs a slice of [`NormalReconstructParams`] into contiguous `std430` words.
///
/// Each record contributes the four words of
/// [`NormalReconstructParams::to_std430`], so the returned buffer is
/// `params.len() * 4` words long and binds directly as a `vec4`-strided storage
/// array on the `GPU`.
#[must_use]
pub fn pack_params_std430(params: &[NormalReconstructParams]) -> Vec<u32> {
    let mut out = Vec::with_capacity(params.len().saturating_mul(4));
    for p in params {
        out.extend_from_slice(&p.to_std430());
    }
    out
}

/// Total byte size of a `std430` storage buffer holding `count` packed
/// [`NormalReconstructParams`] records.
///
/// Uses [`NORMAL_PARAMS_STRIDE`] and the shared clamp-to-one-element rule from
/// [`storage_bytes`], so an empty set still yields a valid `GPU` binding.
#[must_use]
pub fn params_buffer_bytes(count: usize) -> usize {
    storage_bytes(NORMAL_PARAMS_STRIDE, count)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute tolerance for the float assertions in this module's tests.
    const TEST_EPS: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < TEST_EPS
    }

    fn approx_vec(a: [f32; 3], b: [f32; 3]) -> bool {
        approx_eq(a[0], b[0]) && approx_eq(a[1], b[1]) && approx_eq(a[2], b[2])
    }

    fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    fn length3(v: [f32; 3]) -> f32 {
        (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
    }

    fn test_params() -> NormalReconstructParams {
        // A ~90° horizontal / vertical field of view (half-extent 1.0), which
        // keeps the reprojection arithmetic easy to reason about.
        NormalReconstructParams::new(1.0, 1.0)
    }

    #[test]
    fn cmp_eps_governs_normalize_guard() {
        // A vector whose squared length sits just below CMP_EPS is treated as
        // degenerate and collapses to zero; one just above keeps unit length.
        let below_len = (CMP_EPS * 0.5).sqrt();
        let above_len = (CMP_EPS * 4.0).sqrt();
        assert_eq!(normalize3([below_len, 0.0, 0.0]), [0.0, 0.0, 0.0]);
        let kept = normalize3([above_len, 0.0, 0.0]);
        assert!(approx_eq(length3(kept), 1.0));
    }

    #[test]
    fn view_pos_center_pixel_is_on_axis() {
        // UV (0.5, 0.5) maps to ndc (0, 0): the position sits on the -Z axis.
        let params = test_params();
        let p = view_pos_from_depth([0.5, 0.5], 4.0, &params);
        assert!(approx_eq(p[0], 0.0));
        assert!(approx_eq(p[1], 0.0));
        assert!(approx_eq(p[2], -4.0));
    }

    #[test]
    fn view_pos_z_is_negative_linear_depth() {
        let params = test_params();
        let p = view_pos_from_depth([0.2, 0.9], 7.5, &params);
        assert!(approx_eq(p[2], -7.5));
    }

    #[test]
    fn view_pos_edges_scale_with_half_extent() {
        // ndc x = +1 at uv.x = 1.0, so x = half_tan_fov_x * depth.
        let params = NormalReconstructParams::new(2.0, 0.5);
        let p = view_pos_from_depth([1.0, 1.0], 3.0, &params);
        assert!(approx_eq(p[0], 2.0 * 3.0));
        assert!(approx_eq(p[1], 0.5 * 3.0));
    }

    #[test]
    fn view_pos_round_trip_recovers_uv_and_depth() {
        // Reprojecting then inverting the closed form must be consistent.
        let params = NormalReconstructParams::new(1.3, 0.8);
        let uv = [0.37, 0.62];
        let depth = 5.25;
        let p = view_pos_from_depth(uv, depth, &params);

        let recovered_depth = -p[2];
        assert!(approx_eq(recovered_depth, depth));

        let ndc_x = p[0] / (params.half_tan_fov_x * recovered_depth);
        let ndc_y = p[1] / (params.half_tan_fov_y * recovered_depth);
        let recovered_u = (ndc_x + 1.0) * 0.5;
        let recovered_v = (ndc_y + 1.0) * 0.5;
        assert!(approx_eq(recovered_u, uv[0]));
        assert!(approx_eq(recovered_v, uv[1]));
    }

    #[test]
    fn normalize_arbitrary_vector_has_unit_length() {
        let n = normalize3([3.0, -4.0, 12.0]);
        assert!(approx_eq(length3(n), 1.0));
    }

    #[test]
    fn normalize_zero_vector_guards_to_zero() {
        assert_eq!(normalize3([0.0, 0.0, 0.0]), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn normalize_tiny_vector_below_eps_guards_to_zero() {
        // Squared length far below CMP_EPS collapses to the zero vector.
        let n = normalize3([1e-5, -1e-5, 0.0]);
        assert_eq!(n, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn normalize_unit_axis_is_unchanged() {
        assert!(approx_vec(normalize3([0.0, 0.0, 5.0]), [0.0, 0.0, 1.0]));
    }

    #[test]
    fn naive_facing_plane_reconstructs_plus_z() {
        // Constant depth ⇒ a plane squarely facing the camera ⇒ +Z normal.
        let params = test_params();
        let n = reconstruct_normal_naive([0.5, 0.5], [0.01, 0.01], 4.0, 4.0, 4.0, &params);
        assert!(approx_vec(n, [0.0, 0.0, 1.0]));
    }

    #[test]
    fn improved_facing_plane_reconstructs_plus_z() {
        let params = test_params();
        let taps = DepthTaps::new(4.0, 4.0, 4.0, 4.0, 4.0);
        let n = reconstruct_normal_improved([0.5, 0.5], [0.01, 0.01], &taps, &params);
        assert!(approx_vec(n, [0.0, 0.0, 1.0]));
    }

    #[test]
    fn naive_normal_is_unit_length() {
        let params = test_params();
        let n = reconstruct_normal_naive([0.3, 0.7], [0.02, 0.02], 3.0, 3.4, 2.6, &params);
        assert!(approx_eq(length3(n), 1.0));
    }

    #[test]
    fn improved_normal_is_unit_length() {
        let params = test_params();
        let taps = DepthTaps::new(3.0, 2.7, 3.4, 2.6, 3.2);
        let n = reconstruct_normal_improved([0.3, 0.7], [0.02, 0.02], &taps, &params);
        assert!(approx_eq(length3(n), 1.0));
    }

    #[test]
    fn naive_tilted_plane_matches_independent_cross() {
        // A plane whose depth increases to the right and up. The reconstructed
        // normal must equal the cross product of the same two view-space edges
        // computed independently, and still face the camera (+z component).
        let params = test_params();
        let uv = [0.5, 0.5];
        let texel = [0.05, 0.05];
        let d_c = 4.0;
        let d_r = 4.3;
        let d_u = 4.2;

        let p_c = view_pos_from_depth(uv, d_c, &params);
        let p_r = view_pos_from_depth([uv[0] + texel[0], uv[1]], d_r, &params);
        let p_u = view_pos_from_depth([uv[0], uv[1] + texel[1]], d_u, &params);
        let expected = normalize3(cross3(sub3(p_r, p_c), sub3(p_u, p_c)));

        let n = reconstruct_normal_naive(uv, texel, d_c, d_r, d_u, &params);
        assert!(approx_vec(n, expected));
        assert!(n[2] > 0.0);
    }

    #[test]
    fn improved_picks_nearest_neighbor_on_each_axis() {
        // Right and up neighbors are the near ones (matching the plane); left
        // and down are far (a discontinuity). The improved result must equal
        // the reconstruction that uses only the near right/up neighbors.
        let params = test_params();
        let uv = [0.5, 0.5];
        let texel = [0.05, 0.05];
        let taps = DepthTaps::new(4.0, 40.0, 4.25, 40.0, 4.2);

        let improved = reconstruct_normal_improved(uv, texel, &taps, &params);
        let via_right_up =
            reconstruct_normal_naive(uv, texel, taps.center, taps.right, taps.up, &params);
        assert!(approx_vec(improved, via_right_up));
    }

    #[test]
    fn improved_beats_naive_at_depth_discontinuity() {
        // The left/down neighbors lie on the shaded facing plane; the right/up
        // neighbors jump far away (a silhouette edge). The true normal is +Z.
        // Naive trusts the right/up jump and skews badly; improved keeps the
        // near left/down neighbors and stays close to +Z.
        let params = test_params();
        let uv = [0.5, 0.5];
        let texel = [0.05, 0.05];
        let truth = [0.0, 0.0, 1.0];

        let d_center = 4.0;
        let d_near = 4.0; // left and down: same facing plane
        let d_far = 24.0; // right and up: far background

        let naive = reconstruct_normal_naive(uv, texel, d_center, d_far, d_far, &params);
        let taps = DepthTaps::new(d_center, d_near, d_far, d_near, d_far);
        let improved = reconstruct_normal_improved(uv, texel, &taps, &params);

        let naive_alignment = dot3(naive, truth);
        let improved_alignment = dot3(improved, truth);
        assert!(improved_alignment > naive_alignment);
        assert!(approx_vec(improved, truth));
    }

    #[test]
    fn improved_degenerate_flat_collapse_guards_to_zero() {
        // All depths identical and a zero texel step ⇒ both edge vectors are
        // zero ⇒ the cross product collapses and normalize3 guards to zero.
        let params = test_params();
        let taps = DepthTaps::new(4.0, 4.0, 4.0, 4.0, 4.0);
        let n = reconstruct_normal_improved([0.5, 0.5], [0.0, 0.0], &taps, &params);
        assert_eq!(n, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn std430_layout_encodes_fields_and_padding() {
        let params = NormalReconstructParams::new(1.25, 0.75);
        let words = params.to_std430();
        assert_eq!(words[0], 1.25_f32.to_bits());
        assert_eq!(words[1], 0.75_f32.to_bits());
        assert_eq!(words[2], 0);
        assert_eq!(words[3], 0);
    }

    #[test]
    fn params_stride_is_one_vec4() {
        assert_eq!(NORMAL_PARAMS_STRIDE, VEC4_STRIDE);
        assert_eq!(NORMAL_PARAMS_STRIDE, 16);
    }

    #[test]
    fn params_buffer_bytes_clamps_empty_to_one_element() {
        assert_eq!(params_buffer_bytes(0), NORMAL_PARAMS_STRIDE);
        assert_eq!(params_buffer_bytes(3), NORMAL_PARAMS_STRIDE * 3);
    }

    #[test]
    fn pack_params_std430_concatenates_records() {
        let a = NormalReconstructParams::new(1.0, 2.0);
        let b = NormalReconstructParams::new(3.0, 4.0);
        let packed = pack_params_std430(&[a, b]);
        assert_eq!(packed.len(), 8);
        assert_eq!(&packed[0..4], &a.to_std430());
        assert_eq!(&packed[4..8], &b.to_std430());
    }

    #[test]
    fn pack_params_std430_empty_is_empty() {
        let packed = pack_params_std430(&[]);
        assert!(packed.is_empty());
    }
}
