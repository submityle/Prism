//! The analytic per-slice visibility integral at the heart of GTAO.
//!
//! Jimenez et al., "Practical Real-Time Strategies for Accurate Indirect
//! Occlusion" (SIGGRAPH 2016), integrate the cosine-weighted visible arc of a
//! single slice in closed form once the two horizon angles are known.  This
//! module encodes that closed form plus the horizon combination rule, and is
//! the golden reference the `gtao.wesl` compute twin must reproduce bit-for-bit
//! (every transcendental routes through `bevy_math::ops` for determinism).

use bevy_math::ops;

/// `pi / 2`; the widest a horizon can open on one side of the normal.
pub(crate) const HALF_PI: f32 = core::f32::consts::FRAC_PI_2;

/// Folds one marched sample's (already distance-attenuated) horizon cosine into
/// the running horizon cosine.
///
/// GTAO's horizon search keeps the *highest* horizon on each side: a larger
/// cosine means the occluder rises closer to the view direction and blocks
/// more of the hemisphere.  Distance attenuation is applied to the sample
/// *before* this call so far occluders contribute a smaller rise, which keeps
/// the max monotonic and free of the halos that eroding the horizon would
/// introduce.
pub(crate) fn combine_horizon(current: f32, sample_cos: f32) -> f32 {
    current.max(sample_cos)
}

/// Closed-form visibility of one slice.
///
/// * `cos_h1` / `cos_h2` are the horizon cosines measured against the view
///   vector on the negative / positive side of the slice axis.
/// * `gamma` is the signed angle of the normal projected into the slice plane,
///   measured from the view vector toward the positive axis.
/// * `proj_len` is the length of that projected normal, weighting the slice by
///   how much of the normal actually lies in this plane.
///
/// Returns the slice's contribution to visibility; averaging the contribution
/// of every slice yields ambient visibility in `[0, 1]` (occlusion is
/// `1 - visibility`).
pub(crate) fn slice_visibility(cos_h1: f32, cos_h2: f32, gamma: f32, proj_len: f32) -> f32 {
    // Raw horizon angles: negative side is below the view vector, positive above.
    let h1_raw = -ops::acos(cos_h1.clamp(-1.0, 1.0));
    let h2_raw = ops::acos(cos_h2.clamp(-1.0, 1.0));

    // Clamp both horizons into the hemisphere around the projected normal so
    // the arc never dips behind the surface.
    let h1 = gamma + (h1_raw - gamma).max(-HALF_PI);
    let h2 = gamma + (h2_raw - gamma).min(HALF_PI);

    let sin_gamma = ops::sin(gamma);
    let cos_gamma = ops::cos(gamma);

    // Jimenez 2016, the integrated cosine-weighted arc for each half.
    let term1 = 0.25 * (-ops::cos(2.0 * h1 - gamma) + cos_gamma + 2.0 * h1 * sin_gamma);
    let term2 = 0.25 * (-ops::cos(2.0 * h2 - gamma) + cos_gamma + 2.0 * h2 * sin_gamma);

    proj_len * (term1 + term2)
}

/// Distance attenuation applied to a horizon sample.
///
/// Occluders nearer than `falloff_start` count fully; between `falloff_start`
/// and `radius` their contribution eases smoothly to zero (`XeGTAO`'s
/// `falloffRange`), which stops the search radius from producing a hard AO
/// discontinuity.  `dist` and the bounds are all world-space lengths.
pub(crate) fn distance_weight(dist: f32, falloff_start: f32, radius: f32) -> f32 {
    if dist <= falloff_start {
        return 1.0;
    }
    let span = (radius - falloff_start).max(1.0e-4);
    let t = ((dist - falloff_start) / span).clamp(0.0, 1.0);
    // Smooth quadratic ease-out to zero at the radius.
    1.0 - t * t
}
