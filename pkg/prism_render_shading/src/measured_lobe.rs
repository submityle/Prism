//! The measured (tabulated) BRDF/BTF lobe's per-light response (§17.6).
//!
//! `measured_brdf` is the kernel — Rusinkiewicz coordinates, grid mapping and
//! trilinear interpolation over a table. This module is the *lobe* half: it
//! turns that kernel into a radiance contribution for one direct light, with the
//! authored weight/scale and the surface frame applied, so the shading resolve
//! (CPU) and `measured_lobe.wesl` (GPU) are the same response.
//!
//! # Why the table is not in the ABI
//!
//! A measured BRDF is *data*: a MERL-sized table is 90·90·180 RGB cells. It
//! ships as an ordinary material texture (a `provider`, bound with the
//! measured-BRDF semantic), and the lobe block carries only the response
//! parameters — weight, the table's difference-azimuth resolution, and a scale
//! that maps the data's own normalisation into the renderer's units. That keeps
//! the über block small and the table replaceable per material.
//!
//! # Units
//!
//! Measured data is stored as RGB BRDF values, so the contribution is the
//! textbook `f_r · (n·l) · E` — the same shape the analytic lobes use — times
//! the authored weight and scale.

use crate::lighting::{DirectLightSample, ShadingFrame};
use crate::measured_brdf::MeasuredBrdf;
use crate::vecmath::{dot, normalize_or};

/// Evaluate the measured lobe for one direct light.
///
/// `frame` supplies the surface basis: the kernel works in a local frame whose
/// `+Z` is the shading normal and `+X` the tangent, so the light and view
/// directions are rotated into it here. Returns zero for an inert weight, a
/// light below the horizon (the table is only defined for the upper hemisphere),
/// or a degenerate frame, so a material that never authors the lobe is an exact
/// identity.
#[must_use]
pub fn evaluate_measured_direct(
    table: &MeasuredBrdf,
    scale: f32,
    weight: f32,
    frame: &ShadingFrame,
    light: &DirectLightSample,
) -> [f32; 3] {
    let w = weight.clamp(0.0, 1.0);
    if w <= 0.0 {
        return [0.0; 3];
    }
    let n = normalize_or(frame.normal, [0.0, 0.0, 1.0]);
    let v = normalize_or(frame.view, n);
    let l = normalize_or(light.direction, n);
    let n_dot_l = dot(n, l);
    if n_dot_l <= 0.0 {
        return [0.0; 3];
    }
    // Rotate both directions into the surface's local frame.
    let local = |d: [f32; 3]| [dot(frame.tangent, d), dot(frame.bitangent, d), dot(n, d)];
    let f = table.evaluate(local(l), local(v));
    let energy = n_dot_l * w * scale.max(0.0);
    [
        f[0] * energy * light.illuminance[0] * light.visibility.clamp(0.0, 1.0),
        f[1] * energy * light.illuminance[1] * light.visibility.clamp(0.0, 1.0),
        f[2] * energy * light.illuminance[2] * light.visibility.clamp(0.0, 1.0),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::measured_brdf::{MeasuredBrdfDims, SyntheticBrdfParams};

    fn table() -> MeasuredBrdf {
        MeasuredBrdf::from_analytic(MeasuredBrdfDims::MERL, SyntheticBrdfParams::default())
    }

    fn frame() -> ShadingFrame {
        ShadingFrame {
            normal: [0.0, 1.0, 0.0],
            view: [0.0, 1.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
            bitangent: [0.0, 0.0, -1.0],
        }
    }

    fn light() -> DirectLightSample {
        DirectLightSample {
            direction: [0.0, 1.0, 0.0],
            illuminance: [2.0; 3],
            visibility: 1.0,
        }
    }

    #[test]
    fn measured_lobe_is_the_table_times_the_lighting_terms() {
        let table = table();
        let value = evaluate_measured_direct(&table, 1.5, 0.7, &frame(), &light());
        // Reference: the same kernel call the twin makes, with the same framing.
        let f = table.evaluate([0.0, 0.0, 1.0], [0.0, 0.0, 1.0]);
        for channel in 0..3 {
            let want = f[channel] * 1.0 * 0.7 * 1.5 * 2.0;
            assert!(
                (value[channel] - want).abs() <= 1.0e-6 * want.abs().max(1.0),
                "channel {channel}: {} vs {want}",
                value[channel]
            );
        }
    }

    #[test]
    fn inert_weight_back_facing_and_black_lights_contribute_nothing() {
        let table = table();
        assert_eq!(
            evaluate_measured_direct(&table, 1.0, 0.0, &frame(), &light()),
            [0.0; 3]
        );
        let back_facing = DirectLightSample {
            direction: [0.0, -1.0, 0.0],
            ..light()
        };
        assert_eq!(
            evaluate_measured_direct(&table, 1.0, 1.0, &frame(), &back_facing),
            [0.0; 3]
        );
        let unlit = DirectLightSample {
            illuminance: [0.0; 3],
            ..light()
        };
        assert_eq!(
            evaluate_measured_direct(&table, 1.0, 1.0, &frame(), &unlit),
            [0.0; 3]
        );
        let shadowed = DirectLightSample {
            visibility: 0.0,
            ..light()
        };
        assert_eq!(
            evaluate_measured_direct(&table, 1.0, 1.0, &frame(), &shadowed),
            [0.0; 3]
        );
    }

    #[test]
    fn the_lobe_hosts_the_tilted_view_the_table_says_it_should() {
        // A tilted view must change the response (the table is not constant) and
        // always stay non-negative and finite — the resolve composes it into HDR
        // radiance, where a negative or NaN term would poison the pixel.
        let table = table();
        let upright = evaluate_measured_direct(&table, 1.0, 1.0, &frame(), &light());
        let tilted_frame = ShadingFrame {
            view: [0.6, 0.8, 0.0],
            ..frame()
        };
        let tilted = evaluate_measured_direct(&table, 1.0, 1.0, &tilted_frame, &light());
        assert_ne!(upright, tilted);
        for value in upright.iter().chain(tilted.iter()) {
            assert!(value.is_finite() && *value >= 0.0, "{value}");
        }
    }
}
