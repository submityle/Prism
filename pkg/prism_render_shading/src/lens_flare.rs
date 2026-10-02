//! Backend-neutral CPU golden for a screen-space lens-flare / glare
//! post-processing pass (the coloured `ghost` discs and the radial `halo`
//! that a real lens throws when a very bright feature is in frame).
//!
//! Lens flare is a shared post-processing base, not a peer of the PBR/NPR
//! shading fronts: like bloom it runs on the resolved, pre-exposed `HDR`
//! radiance (see [`crate::exposure`]) and re-images the brightest pixels as
//! lens artefacts. Every illumination model — physically based or stylized —
//! writes into the same `HDR` buffer this pass reads, so one implementation
//! serves them all. It shares bloom's bright-tail isolation
//! ([`crate::bloom::prefilter`]) but, instead of a symmetric blur, mirrors the
//! bright features back through the optical centre to fake internal lens
//! reflections.
//!
//! The pipeline is the well-known screen-space `ghost` + `halo` construction
//! (John Chapman's *Pseudo Lens Flare* article and `UE`'s
//! `PostProcessLensFlare`), reduced to the closed-form primitives a compute
//! pass repeats per sample:
//!
//! * **Threshold prefilter.** A soft, hue-preserving isolation of the bright
//!   tail: only the luminance above `threshold` survives, scaled back into the
//!   input's own colour so no `ghost` is ever tinted by the threshold.
//! * **Ghost sampling.** Each `ghost` is the bright buffer read at a `uv`
//!   stepped along the vector toward the optical centre `(0.5, 0.5)`; a
//!   `dispersal` factor times the `ghost` index spaces the discs out so a
//!   single bright source becomes a chain of reflections through the centre.
//! * **Halo.** A single radial ring sampled a fixed distance along the
//!   centre-facing direction, the wide coloured arc that hugs the frame.
//! * **Chromatic offset.** An optional per-channel radial micro-shift that
//!   fringes the `ghost` discs red/blue, the dispersion a real lens shows.
//! * **Radial / vignette falloff.** Smooth edge weights (own `3t^2 - 2t^3`
//!   smoothstep, no library call) so the artefacts fade toward the frame edge
//!   rather than clipping hard.
//! * **Composite.** The accumulated flare is added over the scene by an artist
//!   `intensity`.
//!
//! Everything is pure `f32` maths — no transcendental calls, radial distances
//! use `sqrt` — mirrored arm-for-arm by `shaders/lens_flare.wesl` (same
//! primitives, same constants, componentwise vector ops matching the
//! per-channel Rust) so the CPU golden and the GPU twin agree.

/// Rec. 709 luminance weights (linear `sRGB` primaries), used to isolate the
/// bright tail. Kept local so the lens-flare golden is self-contained.
pub const LENS_FLARE_LUMINANCE_WEIGHTS: [f32; 3] = [0.2126, 0.7152, 0.0722];

/// Optical centre of the frame in normalised `uv` space; every `ghost` and the
/// `halo` are mirrored/aligned through this point.
pub const LENS_FLARE_CENTER: [f32; 2] = [0.5, 0.5];

/// Rec. 709 relative luminance of a linear RGB radiance sample.
#[must_use]
pub fn luminance(rgb: [f32; 3]) -> f32 {
    rgb[0] * LENS_FLARE_LUMINANCE_WEIGHTS[0]
        + rgb[1] * LENS_FLARE_LUMINANCE_WEIGHTS[1]
        + rgb[2] * LENS_FLARE_LUMINANCE_WEIGHTS[2]
}

/// Own smoothstep `t^2 * (3 - 2t)` on the normalised, clamped
/// `t = (x - edge0) / (edge1 - edge0)`. Written out rather than using a library
/// smoothstep so the CPU golden and the WESL twin stay bit-identical. Degenerate
/// `edge0 == edge1` falls back to a hard step at the edge.
#[must_use]
pub fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let span = edge1 - edge0;
    if span == 0.0 {
        return if x < edge0 { 0.0 } else { 1.0 };
    }
    let t = ((x - edge0) / span).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Per-channel linear interpolation `a + (b - a) * t`, written out (not a
/// library `mix`) so it matches the WESL twin exactly. `t` is not clamped.
#[must_use]
pub fn mix3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

/// Soft, hue-preserving isolation of the bright tail. The contribution is the
/// luminance surplus over `threshold` normalised by the input luminance, so the
/// output keeps the input's channel ratios (no threshold tint) and is `0` for
/// anything at or below `threshold`:
///
/// ```text
/// contribution = max(0, luminance(rgb) - threshold) / max(luminance(rgb), 1e-6)
/// out          = rgb * contribution
/// ```
#[must_use]
pub fn threshold_prefilter(rgb: [f32; 3], threshold: f32) -> [f32; 3] {
    let luma = luminance(rgb);
    let contribution = (luma - threshold).max(0.0) / luma.max(1.0e-6);
    [
        rgb[0] * contribution,
        rgb[1] * contribution,
        rgb[2] * contribution,
    ]
}

/// Sampling `uv` for the `ghost_index`-th `ghost`: step from `uv` toward the
/// optical centre by `dispersal * ghost_index` of the centre-facing vector.
///
/// `ghost_index == 0` returns `uv` unchanged (the source itself); each further
/// index lands one `dispersal` step deeper toward — and eventually past — the
/// centre, so a single bright source becomes an evenly spaced chain of
/// reflections.
#[must_use]
pub fn ghost_uv(uv: [f32; 2], ghost_index: f32, dispersal: f32) -> [f32; 2] {
    let dir = [LENS_FLARE_CENTER[0] - uv[0], LENS_FLARE_CENTER[1] - uv[1]];
    let step = dispersal * ghost_index;
    [uv[0] + dir[0] * step, uv[1] + dir[1] * step]
}

/// Sampling `uv` for the `halo` ring: step a fixed `halo_width` along the unit
/// vector pointing from `uv` to the optical centre. A `uv` exactly at the
/// centre has no direction and is returned unchanged.
#[must_use]
pub fn halo_uv(uv: [f32; 2], halo_width: f32) -> [f32; 2] {
    let dir = [LENS_FLARE_CENTER[0] - uv[0], LENS_FLARE_CENTER[1] - uv[1]];
    let len = (dir[0] * dir[0] + dir[1] * dir[1]).sqrt();
    if len < 1.0e-6 {
        return uv;
    }
    let n = [dir[0] / len, dir[1] / len];
    [uv[0] + n[0] * halo_width, uv[1] + n[1] * halo_width]
}

/// Radial weight that peaks at the optical centre and smoothly falls to `0` by
/// half a frame out: `1 - smoothstep(0, 0.5, distance(uv, centre))`. Uses the
/// own [`smoothstep`], so the falloff matches the WESL twin.
#[must_use]
pub fn radial_weight(uv: [f32; 2]) -> f32 {
    let dx = uv[0] - LENS_FLARE_CENTER[0];
    let dy = uv[1] - LENS_FLARE_CENTER[1];
    let dist = (dx * dx + dy * dy).sqrt();
    1.0 - smoothstep(0.0, 0.5, dist)
}

/// Vignette multiplier in `[1 - strength, 1]`: full weight at the centre,
/// darkening toward the frame edge by `strength * smoothstep(0, 1, 2 * dist)`.
/// `strength == 0` is a no-op (returns `1`).
#[must_use]
pub fn vignette_weight(uv: [f32; 2], strength: f32) -> f32 {
    let dx = uv[0] - LENS_FLARE_CENTER[0];
    let dy = uv[1] - LENS_FLARE_CENTER[1];
    let dist = (dx * dx + dy * dy).sqrt();
    let edge = smoothstep(0.0, 1.0, 2.0 * dist);
    1.0 - strength * edge
}

/// Per-channel radial micro-offset for a chromatic (dispersed) `ghost`. The
/// `channel` selects a signed lobe — red (`0`) shifts one way, green (`1`) is
/// unshifted, blue (`2`) shifts the other — by `distortion * (channel - 1)`
/// along the centre-facing vector, so sampling the three channels at these
/// offsets fringes the disc red/blue like real lens dispersion.
#[must_use]
pub fn chromatic_ghost_offset(uv: [f32; 2], distortion: f32, channel: u32) -> [f32; 2] {
    let dir = [LENS_FLARE_CENTER[0] - uv[0], LENS_FLARE_CENTER[1] - uv[1]];
    let lobe = channel as f32 - 1.0;
    let step = distortion * lobe;
    [uv[0] + dir[0] * step, uv[1] + dir[1] * step]
}

/// Accumulate weighted `ghost` samples: `sum(color_i * weight_i)`. This is the
/// pure-CPU form used by the golden tests; the GPU twin accumulates the same
/// sum by looping the sampler over the `ghost` chain. An empty slice sums to
/// black.
#[must_use]
pub fn accumulate_ghosts(samples: &[([f32; 3], f32)]) -> [f32; 3] {
    let mut out = [0.0_f32; 3];
    for &(color, weight) in samples {
        out[0] += color[0] * weight;
        out[1] += color[1] * weight;
        out[2] += color[2] * weight;
    }
    out
}

/// Composite the accumulated `flare` over the `scene` additively:
/// `scene + flare * intensity`. `intensity == 0` returns the untouched scene;
/// `intensity` is not clamped so callers may push past `1`.
#[must_use]
pub fn apply_lens_flare(scene_rgb: [f32; 3], flare_rgb: [f32; 3], intensity: f32) -> [f32; 3] {
    [
        scene_rgb[0] + flare_rgb[0] * intensity,
        scene_rgb[1] + flare_rgb[1] * intensity,
        scene_rgb[2] + flare_rgb[2] * intensity,
    ]
}

/// Artist controls for the lens-flare pass. `Default` is a neutral, disabled
/// flare (all zero): `intensity == 0` leaves the scene untouched in
/// [`apply_lens_flare`].
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct LensFlareParams {
    /// Blend weight of the accumulated flare over the scene in
    /// [`apply_lens_flare`]. `0` disables the pass.
    pub intensity: f32,
    /// Luminance above which pixels contribute to the flare in
    /// [`threshold_prefilter`].
    pub threshold: f32,
    /// Number of `ghost` discs sampled along the centre-facing vector.
    pub ghost_count: u32,
    /// Spacing of the `ghost` chain passed to [`ghost_uv`].
    pub dispersal: f32,
    /// Radial offset of the `halo` ring passed to [`halo_uv`].
    pub halo_width: f32,
    /// Per-channel chromatic dispersion passed to [`chromatic_ghost_offset`].
    pub distortion: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) {
        assert!((a - b).abs() < 1.0e-5, "expected {b}, got {a}");
    }

    fn approx2(a: [f32; 2], b: [f32; 2]) {
        approx(a[0], b[0]);
        approx(a[1], b[1]);
    }

    fn approx3(a: [f32; 3], b: [f32; 3]) {
        approx(a[0], b[0]);
        approx(a[1], b[1]);
        approx(a[2], b[2]);
    }

    // --- luminance ---

    #[test]
    fn luminance_matches_rec709() {
        approx(luminance([1.0, 1.0, 1.0]), 1.0);
        approx(luminance([1.0, 0.0, 0.0]), 0.2126);
        approx(luminance([0.0, 1.0, 0.0]), 0.7152);
        approx(luminance([0.0, 0.0, 1.0]), 0.0722);
    }

    // --- smoothstep ---

    #[test]
    fn smoothstep_endpoints() {
        approx(smoothstep(0.0, 1.0, 0.0), 0.0);
        approx(smoothstep(0.0, 1.0, 1.0), 1.0);
    }

    #[test]
    fn smoothstep_midpoint_is_half() {
        approx(smoothstep(0.0, 1.0, 0.5), 0.5);
    }

    #[test]
    fn smoothstep_clamps_outside_range() {
        approx(smoothstep(0.0, 1.0, -2.0), 0.0);
        approx(smoothstep(0.0, 1.0, 3.0), 1.0);
    }

    #[test]
    fn smoothstep_degenerate_span_is_hard_step() {
        approx(smoothstep(0.5, 0.5, 0.4), 0.0);
        approx(smoothstep(0.5, 0.5, 0.6), 1.0);
    }

    // --- mix3 ---

    #[test]
    fn mix3_endpoints() {
        let a = [0.1, 0.2, 0.3];
        let b = [0.9, 0.8, 0.7];
        approx3(mix3(a, b, 0.0), a);
        approx3(mix3(a, b, 1.0), b);
    }

    #[test]
    fn mix3_midpoint_is_average() {
        approx3(
            mix3([0.0, 0.0, 0.0], [1.0, 0.5, 0.25], 0.5),
            [0.5, 0.25, 0.125],
        );
    }

    // --- threshold_prefilter ---

    #[test]
    fn threshold_prefilter_below_threshold_is_zero() {
        // Luminance under the threshold contributes nothing.
        approx3(threshold_prefilter([0.1, 0.1, 0.1], 1.0), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn threshold_prefilter_at_threshold_is_zero() {
        // Exactly at the threshold the surplus is zero.
        let luma = 1.0;
        approx3(
            threshold_prefilter([luma, luma, luma], 1.0),
            [0.0, 0.0, 0.0],
        );
    }

    #[test]
    fn threshold_prefilter_preserves_hue() {
        // The contribution is a scalar, so channel ratios survive.
        let color = [3.0, 1.5, 0.75];
        let out = threshold_prefilter(color, 1.0);
        approx(out[0] / out[1], color[0] / color[1]);
        approx(out[1] / out[2], color[1] / color[2]);
    }

    #[test]
    fn threshold_prefilter_white_passes_surplus() {
        // Grey white: luminance == component. Surplus fraction (l - t) / l.
        let out = threshold_prefilter([4.0, 4.0, 4.0], 1.0);
        let expected = 4.0 * (4.0 - 1.0) / 4.0;
        approx3(out, [expected, expected, expected]);
    }

    // --- ghost_uv ---

    #[test]
    fn ghost_uv_index_zero_is_identity() {
        let uv = [0.2, 0.7];
        approx2(ghost_uv(uv, 0.0, 0.6), uv);
    }

    #[test]
    fn ghost_uv_moves_toward_center() {
        // A positive step pulls the sample toward (0.5, 0.5).
        let uv = [0.0, 0.0];
        let g = ghost_uv(uv, 1.0, 0.5);
        approx2(g, [0.25, 0.25]);
    }

    #[test]
    fn ghost_uv_scales_with_index() {
        // Index 2 is twice the displacement of index 1 for the same dispersal.
        let uv = [0.1, 0.3];
        let g1 = ghost_uv(uv, 1.0, 0.4);
        let g2 = ghost_uv(uv, 2.0, 0.4);
        approx2(
            [g2[0] - uv[0], g2[1] - uv[1]],
            [2.0 * (g1[0] - uv[0]), 2.0 * (g1[1] - uv[1])],
        );
    }

    #[test]
    fn ghost_uv_from_center_is_stable() {
        // The centre has no centre-facing vector, so any step keeps it put.
        approx2(ghost_uv(LENS_FLARE_CENTER, 3.0, 0.5), LENS_FLARE_CENTER);
    }

    // --- halo_uv ---

    #[test]
    fn halo_uv_offsets_by_width_along_center_dir() {
        // From (0, 0.5) the centre-facing unit vector is (+1, 0).
        let uv = [0.0, 0.5];
        approx2(halo_uv(uv, 0.2), [0.2, 0.5]);
    }

    #[test]
    fn halo_uv_center_is_stable() {
        approx2(halo_uv(LENS_FLARE_CENTER, 0.3), LENS_FLARE_CENTER);
    }

    #[test]
    fn halo_uv_offset_length_matches_width() {
        let uv = [0.1, 0.2];
        let h = halo_uv(uv, 0.15);
        let ox = h[0] - uv[0];
        let oy = h[1] - uv[1];
        let len = (ox * ox + oy * oy).sqrt();
        approx(len, 0.15);
    }

    // --- radial_weight ---

    #[test]
    fn radial_weight_center_is_one() {
        approx(radial_weight(LENS_FLARE_CENTER), 1.0);
    }

    #[test]
    fn radial_weight_half_out_is_zero() {
        // Distance 0.5 from the centre reaches the smoothstep upper edge.
        approx(radial_weight([1.0, 0.5]), 0.0);
    }

    #[test]
    fn radial_weight_monotonic_decreasing() {
        let mut prev = f32::INFINITY;
        for i in 0..=20 {
            let d = i as f32 / 40.0; // 0 .. 0.5 along +x
            let w = radial_weight([0.5 + d, 0.5]);
            assert!(w <= prev, "radial weight not decreasing at {i}");
            prev = w;
        }
    }

    // --- vignette_weight ---

    #[test]
    fn vignette_weight_center_is_one() {
        approx(vignette_weight(LENS_FLARE_CENTER, 1.0), 1.0);
    }

    #[test]
    fn vignette_weight_strength_zero_is_one() {
        approx(vignette_weight([1.0, 1.0], 0.0), 1.0);
    }

    #[test]
    fn vignette_weight_darkens_toward_edge() {
        let center = vignette_weight(LENS_FLARE_CENTER, 1.0);
        let edge = vignette_weight([1.0, 0.5], 1.0);
        assert!(edge < center, "vignette should darken toward the edge");
    }

    // --- chromatic_ghost_offset ---

    #[test]
    fn chromatic_ghost_offset_green_is_identity() {
        // Channel 1 (green) has a zero lobe, so it is unshifted.
        let uv = [0.2, 0.8];
        approx2(chromatic_ghost_offset(uv, 0.5, 1), uv);
    }

    #[test]
    fn chromatic_ghost_offset_red_blue_are_symmetric() {
        // Red (0) and blue (2) shift equal-and-opposite about green.
        let uv = [0.2, 0.8];
        let r = chromatic_ghost_offset(uv, 0.3, 0);
        let b = chromatic_ghost_offset(uv, 0.3, 2);
        approx2(
            [r[0] - uv[0], r[1] - uv[1]],
            [-(b[0] - uv[0]), -(b[1] - uv[1])],
        );
    }

    // --- accumulate_ghosts ---

    #[test]
    fn accumulate_ghosts_empty_is_black() {
        approx3(accumulate_ghosts(&[]), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn accumulate_ghosts_sums_weighted() {
        let samples = [
            ([1.0, 0.0, 0.0], 0.5),
            ([0.0, 2.0, 0.0], 0.25),
            ([0.0, 0.0, 4.0], 0.125),
        ];
        approx3(accumulate_ghosts(&samples), [0.5, 0.5, 0.5]);
    }

    // --- apply_lens_flare ---

    #[test]
    fn apply_lens_flare_intensity_zero_is_scene() {
        let scene = [0.2, 0.4, 0.6];
        approx3(apply_lens_flare(scene, [9.0, 9.0, 9.0], 0.0), scene);
    }

    #[test]
    fn apply_lens_flare_is_additive() {
        let scene = [0.2, 0.4, 0.6];
        let flare = [0.1, 0.2, 0.3];
        approx3(apply_lens_flare(scene, flare, 2.0), [0.4, 0.8, 1.2]);
    }

    // --- params ---

    #[test]
    fn lens_flare_params_default_is_disabled() {
        let p = LensFlareParams::default();
        approx(p.intensity, 0.0);
        approx(p.threshold, 0.0);
        assert_eq!(p.ghost_count, 0);
        approx(p.dispersal, 0.0);
        approx(p.halo_width, 0.0);
        approx(p.distortion, 0.0);
        // Disabled intensity leaves the scene untouched regardless of flare.
        approx3(
            apply_lens_flare([0.3, 0.3, 0.3], [9.0, 9.0, 9.0], p.intensity),
            [0.3, 0.3, 0.3],
        );
    }
}
