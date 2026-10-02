//! World-space **triplanar** normal blending (Whiteout method).
//!
//! Triplanar mapping textures a surface by projecting the material down the
//! three world axes (the `YZ`, `ZX` and `XY` planes) and blending the three
//! samples by weights derived from the geometric normal, so steep terrain,
//! procedurally-placed meshes and UV-less geometry get seamless detail with no
//! authored UVs. Blending three *tangent-space* normal-map samples naively (in
//! their own planar frames) tilts the detail the wrong way on every face; the
//! **Whiteout** triplanar blend (Golus) first reorients each projected tangent
//! normal against the world geometric normal with the same partial-derivative
//! (whiteout) rule used by [`blend_whiteout`](super::blend_whiteout), then
//! swizzles each result into world orientation before the weighted sum.
//!
//! Given the world geometric normal `n` and the three projected tangent normals
//! `nx`, `ny`, `nz` (`z` up in each plane's own frame), each is combined as
//! * `X`: `(nx.x + n.z, nx.y + n.y, |nx.z| * n.x)` then swizzled `.zyx`;
//! * `Y`: `(ny.x + n.x, ny.y + n.z, |ny.z| * n.y)` then swizzled `.xzy`;
//! * `Z`: `(nz.x + n.x, nz.y + n.y, |nz.z| * n.z)` then swizzled `.xyz`;
//!
//! and the three are mixed by `blend = normalize_L1(|n|^sharpness)` (a partition
//! of unity that `sharpness` sharpens toward the dominant axis), then the sum is
//! renormalised. Two structural identities fall straight out and anchor the
//! tests: flat detail normals `(0,0,1)` on every plane reconstruct the
//! geometric normal exactly, and a surface facing a single axis returns that
//! axis' projected sample alone.
//!
//! Pure analytic `f32` math, no AI/ML, so a CPU golden matches a GPU compute
//! twin to floating-point tolerance.
//!
//! # References
//! * Ben Golus, "Normal Mapping for a Triplanar Shader" (Whiteout blend), 2017.
//! * Akenine-Moller et al., *Real-Time Rendering* 4th ed., Section 20.5.1
//!   (triplanar / UV-less projection).

use bevy_math::ops;

/// Triplanar blend weights `normalize_L1(|n|^sharpness)` for a geometric normal.
///
/// The weights are non-negative and sum to `1` (a partition of unity), so the
/// blended normal never exceeds the inputs' envelope. `sharpness` (clamped to
/// `>= 0`) sharpens the transition toward the dominant axis; `0` yields an equal
/// `1/3` split. A degenerate zero normal falls back to the equal split.
#[must_use]
pub fn triplanar_weights(n_geo: [f32; 3], sharpness: f32) -> [f32; 3] {
    let k = sharpness.max(0.0);
    let mut w = [
        ops::powf(n_geo[0].abs(), k),
        ops::powf(n_geo[1].abs(), k),
        ops::powf(n_geo[2].abs(), k),
    ];
    let sum = w[0] + w[1] + w[2];
    if sum > 1.0e-20 {
        let inv = 1.0 / sum;
        w[0] *= inv;
        w[1] *= inv;
        w[2] *= inv;
    } else {
        w = [1.0 / 3.0; 3];
    }
    w
}

/// Renormalise a vector, falling back to `(0,0,1)` when it is degenerate.
#[inline]
#[must_use]
fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    if len2 > 1.0e-20 {
        let inv = 1.0 / ops::sqrt(len2);
        [v[0] * inv, v[1] * inv, v[2] * inv]
    } else {
        [0.0, 0.0, 1.0]
    }
}

/// Blend three projected tangent-space normals into a single world-space normal
/// with the Whiteout triplanar method.
///
/// `n_geo` is the world geometric (interpolated vertex) normal; `normal_x`,
/// `normal_y`, `normal_z` are the tangent-space normals sampled from the
/// material projected along the world `X`, `Y` and `Z` axes respectively (each
/// `z`-up in its own plane). `sharpness` controls the blend contrast (see
/// [`triplanar_weights`]). The result is a unit world-space normal.
#[must_use]
pub fn blend_triplanar_whiteout(
    n_geo: [f32; 3],
    normal_x: [f32; 3],
    normal_y: [f32; 3],
    normal_z: [f32; 3],
    sharpness: f32,
) -> [f32; 3] {
    let w = triplanar_weights(n_geo, sharpness);

    // Reorient each projected tangent normal against the world normal with the
    // whiteout rule (sum xy, multiply |z|), then swizzle into world orientation.
    let tx = [
        normal_x[0] + n_geo[2],
        normal_x[1] + n_geo[1],
        normal_x[2].abs() * n_geo[0],
    ];
    let ty = [
        normal_y[0] + n_geo[0],
        normal_y[1] + n_geo[2],
        normal_y[2].abs() * n_geo[1],
    ];
    let tz = [
        normal_z[0] + n_geo[0],
        normal_z[1] + n_geo[1],
        normal_z[2].abs() * n_geo[2],
    ];

    // X -> .zyx, Y -> .xzy, Z -> .xyz.
    let sx = [tx[2], tx[1], tx[0]];
    let sy = [ty[0], ty[2], ty[1]];
    let sz = tz;

    normalize3([
        sx[0] * w[0] + sy[0] * w[1] + sz[0] * w[2],
        sx[1] * w[0] + sy[1] * w[1] + sz[1] * w[2],
        sx[2] * w[0] + sy[2] * w[1] + sz[2] * w[2],
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    const UP: [f32; 3] = [0.0, 0.0, 1.0];
    const EPS: f32 = 1.0e-5;

    fn unit(v: [f32; 3]) -> [f32; 3] {
        let inv = 1.0 / ops::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        [v[0] * inv, v[1] * inv, v[2] * inv]
    }

    fn is_unit(n: [f32; 3]) {
        let len = ops::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        assert!((len - 1.0).abs() < EPS, "len={len} n={n:?}");
    }

    fn close(a: [f32; 3], b: [f32; 3]) {
        for i in 0..3 {
            assert!((a[i] - b[i]).abs() < EPS, "a={a:?} b={b:?}");
        }
    }

    fn geo_normals() -> Vec<[f32; 3]> {
        [
            unit([0.2, 0.6, 0.8]),
            unit([-0.5, 0.3, 0.9]),
            unit([0.7, -0.7, 0.1]),
            unit([-0.1, -0.9, 0.4]),
            unit([1.0, 0.0, 0.0]),
            unit([0.0, 1.0, 0.0]),
            unit([0.0, 0.0, 1.0]),
        ]
        .to_vec()
    }

    #[test]
    fn weights_are_a_partition_of_unity() {
        for n in geo_normals() {
            for &k in &[0.0f32, 1.0, 4.0, 16.0] {
                let w = triplanar_weights(n, k);
                assert!(w.iter().all(|&x| x >= 0.0), "neg {w:?}");
                assert!((w[0] + w[1] + w[2] - 1.0).abs() < EPS, "sum {w:?}");
            }
        }
    }

    #[test]
    fn zero_sharpness_is_equal_split() {
        let w = triplanar_weights(unit([0.3, -0.7, 0.65]), 0.0);
        close(w, [1.0 / 3.0; 3]);
    }

    #[test]
    fn negative_sharpness_clamps_to_zero() {
        let n = unit([0.3, -0.7, 0.65]);
        close(triplanar_weights(n, -5.0), triplanar_weights(n, 0.0));
    }

    #[test]
    fn degenerate_zero_normal_falls_back_to_equal_split() {
        close(triplanar_weights([0.0, 0.0, 0.0], 4.0), [1.0 / 3.0; 3]);
    }

    #[test]
    fn sharpness_concentrates_on_dominant_axis() {
        // |n| = (0.6, 0.48, 0.64)-ish: z is dominant. Higher sharpness must not
        // decrease the dominant-axis weight.
        let n = unit([0.6, 0.5, 0.7]);
        let dom = 2; // z has the largest magnitude.
        let w_lo = triplanar_weights(n, 1.0);
        let w_hi = triplanar_weights(n, 8.0);
        assert!(w_hi[dom] >= w_lo[dom] - EPS, "lo={w_lo:?} hi={w_hi:?}");
        assert!(w_hi[dom] > 0.5, "hi={w_hi:?}");
    }

    #[test]
    fn flat_details_reconstruct_the_geometric_normal() {
        // Every plane sees a flat (0,0,1) tangent normal: the blend must return
        // the geometric normal exactly. This exercises all three swizzles AND
        // the weight normalisation at once (strong anti-fake oracle).
        for n in geo_normals() {
            for &k in &[0.0f32, 1.0, 4.0, 16.0] {
                let out = blend_triplanar_whiteout(n, UP, UP, UP, k);
                close(out, n);
                is_unit(out);
            }
        }
    }

    #[test]
    fn pure_z_face_returns_z_sample() {
        // Facing +Z (sharp blend) -> weight (0,0,1); for a +z tangent normal the
        // whiteout swizzle is the identity, so the result equals that sample.
        let nz = unit([0.25, -0.15, 0.95]);
        let out = blend_triplanar_whiteout([0.0, 0.0, 1.0], UP, UP, nz, 8.0);
        close(out, nz);
    }

    #[test]
    fn pure_x_face_matches_the_x_swizzle() {
        // Facing +X -> weight (1,0,0); result = normalize(|nx.z|, nx.y, nx.x).
        let nx = unit([0.3, -0.2, 0.9]);
        let out = blend_triplanar_whiteout([1.0, 0.0, 0.0], nx, UP, UP, 8.0);
        close(out, unit([nx[2].abs(), nx[1], nx[0]]));
    }

    #[test]
    fn pure_y_face_matches_the_y_swizzle() {
        // Facing +Y -> weight (0,1,0); result = normalize(ny.x, |ny.z|, ny.y).
        let ny = unit([-0.4, 0.1, 0.9]);
        let out = blend_triplanar_whiteout([0.0, 1.0, 0.0], UP, ny, UP, 8.0);
        close(out, unit([ny[0], ny[2].abs(), ny[1]]));
    }

    #[test]
    fn output_is_always_unit() {
        let nx = unit([0.3, -0.2, 0.9]);
        let ny = unit([-0.4, 0.1, 0.9]);
        let nz = unit([0.25, -0.15, 0.95]);
        for n in geo_normals() {
            for &k in &[0.0f32, 2.0, 8.0] {
                is_unit(blend_triplanar_whiteout(n, nx, ny, nz, k));
            }
        }
    }
}
