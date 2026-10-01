//! Blending a detail (mesostructure) normal onto a base normal.
//!
//! AAA materials layer a high-frequency detail normal over a base normal
//! (tiling detail, decals, damage). Naive addition flattens or over-steepens
//! the result; the techniques here are the standard tangent-space blends, in
//! ascending order of correctness/cost:
//!
//! * [`blend_linear`] -- add then renormalise; cheapest, loses detail slope.
//! * [`blend_udn`] -- "unity detail normal": sum `xy`, keep base `z`.
//! * [`blend_whiteout`] -- partial-derivative (whiteout) blend: sum `xy`,
//!   multiply `z`; a good quality/cost tradeoff.
//! * [`blend_rnm`] -- **Reoriented Normal Mapping** (Barre-Brisebois & Hill):
//!   reflects the detail into the base's frame, the most faithful of the four.
//!
//! All operate on unit tangent-space normals (`z` up) and return a unit normal.
//! Pure analytic math, no AI/ML; a CPU golden matches a GPU twin to
//! floating-point tolerance.
//!
//! # References
//! * Barre-Brisebois & Hill, "Blending in Detail" (reoriented normal mapping),
//!   Self-Shadow blog, 2012.
//! * Akenine-Moller et al., *Real-Time Rendering* 4th ed., Section 6.7.2.

#[inline]
fn normalize_or_base(v: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let len2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    if len2 > 1.0e-12 {
        let inv = 1.0 / len2.sqrt();
        [v[0] * inv, v[1] * inv, v[2] * inv]
    } else {
        fallback
    }
}

#[inline]
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Linear blend: `normalize(base + detail)`.
///
/// Cheapest option; tends to wash out the detail slope because the two `z`
/// components add. Included mostly for comparison / low-end paths.
#[must_use]
pub fn blend_linear(base: [f32; 3], detail: [f32; 3]) -> [f32; 3] {
    normalize_or_base(
        [base[0] + detail[0], base[1] + detail[1], base[2] + detail[2]],
        base,
    )
}

/// Unity detail-normal (UDN) blend: sum the `xy` slopes, keep the base `z`.
#[must_use]
pub fn blend_udn(base: [f32; 3], detail: [f32; 3]) -> [f32; 3] {
    normalize_or_base([base[0] + detail[0], base[1] + detail[1], base[2]], base)
}

/// Whiteout (partial-derivative) blend: sum the `xy` slopes, multiply `z`.
///
/// Equivalent to adding the surface gradients, which is the physically
/// motivated way to combine two height-field perturbations.
#[must_use]
pub fn blend_whiteout(base: [f32; 3], detail: [f32; 3]) -> [f32; 3] {
    normalize_or_base(
        [
            base[0] + detail[0],
            base[1] + detail[1],
            base[2] * detail[2],
        ],
        base,
    )
}

/// Reoriented Normal Mapping (RNM): rotate the detail normal into the base
/// normal's tangent frame, then renormalise.
///
/// With unit inputs this satisfies the two identities a correct blend must
/// have: a flat detail `(0,0,1)` returns `base`, and a flat base `(0,0,1)`
/// returns `detail`.
#[must_use]
pub fn blend_rnm(base: [f32; 3], detail: [f32; 3]) -> [f32; 3] {
    let t = [base[0], base[1], base[2] + 1.0];
    let u = [-detail[0], -detail[1], detail[2]];
    let d = dot3(t, u);
    let r = [
        t[0] * d - u[0] * t[2],
        t[1] * d - u[1] * t[2],
        t[2] * d - u[2] * t[2],
    ];
    normalize_or_base(r, base)
}

#[cfg(test)]
mod tests {
    use super::*;

    const UP: [f32; 3] = [0.0, 0.0, 1.0];

    fn is_unit(n: [f32; 3]) {
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        assert!((len - 1.0).abs() < 1.0e-5, "len={len} n={n:?}");
    }

    fn close(a: [f32; 3], b: [f32; 3]) {
        for i in 0..3 {
            assert!((a[i] - b[i]).abs() < 1.0e-5, "a={a:?} b={b:?}");
        }
    }

    fn unit(v: [f32; 3]) -> [f32; 3] {
        let inv = 1.0 / (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        [v[0] * inv, v[1] * inv, v[2] * inv]
    }

    #[test]
    fn rnm_flat_detail_returns_base() {
        let base = unit([0.3, -0.2, 0.9]);
        close(blend_rnm(base, UP), base);
        is_unit(blend_rnm(base, UP));
    }

    #[test]
    fn rnm_flat_base_returns_detail() {
        let detail = unit([-0.4, 0.1, 0.9]);
        close(blend_rnm(UP, detail), detail);
    }

    #[test]
    fn whiteout_and_udn_flat_detail_keep_base_direction() {
        let base = unit([0.2, 0.3, 0.9]);
        // Flat detail (0,0,1): whiteout keeps z*1, udn keeps z; both renormalise
        // to base because xy are unchanged and z scales uniformly.
        close(blend_whiteout(base, UP), base);
        close(blend_udn(base, UP), base);
    }

    #[test]
    fn all_blends_return_unit_normals() {
        let base = unit([0.5, -0.5, 0.707]);
        let detail = unit([-0.3, 0.4, 0.866]);
        is_unit(blend_linear(base, detail));
        is_unit(blend_udn(base, detail));
        is_unit(blend_whiteout(base, detail));
        is_unit(blend_rnm(base, detail));
    }

    #[test]
    fn degenerate_opposite_normals_fall_back_to_base() {
        // base + detail == 0 would be unnormalisable; linear blend falls back.
        let base = [0.0, 0.0, 1.0];
        let detail = [0.0, 0.0, -1.0];
        close(blend_linear(base, detail), base);
    }
}
