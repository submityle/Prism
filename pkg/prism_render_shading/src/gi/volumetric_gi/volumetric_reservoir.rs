//! Volumetric ReSTIR: streaming RIS over froxel light candidates with temporal
//! reprojection — CPU golden.
//!
//! Direct volumetric lighting integrates, at every froxel, the in-scattered
//! radiance from all lights.  With a tiny per-froxel sample budget this is
//! prohibitive, so Volumetric ReSTIR (Lin & Yuksel 2021, *Real-Time Volumetric
//! Rendering with Reservoir Resampling*) recycles the per-froxel light samples
//! across the frustum and across frames with weighted-reservoir sampling.  This
//! module is the backend-neutral numerical reference for that resampling.
//!
//! It reuses the generic [`Reservoir`](crate::gi::screen_probe::restir::Reservoir)
//! container from the screen-probe ReSTIR reference and layers a
//! participating-media *target function* on top: the scalar density a froxel
//! resamples towards is the luminance of its in-scattered, transmittance-
//! weighted radiance, built from [`super::scattering`].
//!
//! * [`VolumetricSample`] is the reservoir payload: a light position and its
//!   emitted RGB radiance.
//! * [`MediumParams`] bundles the homogeneous medium's `sigma_s`, `sigma_t`,
//!   and HG anisotropy `g`.
//! * [`integrand`] is the per-sample in-scattered radiance (vector) and
//!   [`scatter_target`] its scalar luminance target `p_hat`.
//! * [`resample_froxel`] runs the streaming RIS loop over a batch of candidate
//!   light samples for one froxel and finalises the contribution weight.
//! * [`temporal_resample`] folds a reprojected previous-frame reservoir into the
//!   current one with an M-cap, realising temporal reuse.
//! * [`estimate_radiance`] turns a finalised reservoir into the unbiased
//!   in-scattered radiance estimate `integrand * W`.
//!
//! # Conventions
//! * The scatter point and a view direction `view_dir` (pointing from the
//!   froxel towards the camera) define the scattering geometry; the phase angle
//!   cosine is `dot(wi, view_dir)` with `wi` the unit direction to the light.
//! * `source_pdf` is the density the candidate was drawn from; the RIS weight
//!   is `target / source_pdf`.  Non-positive or non-finite source pdfs make the
//!   candidate degenerate and it is skipped.
//! * Mismatched candidate / pdf / uniform slice lengths are handled by iterating
//!   over the shortest common prefix, so the function never indexes out of
//!   bounds.
//! * All weights and radiances are finite and non-negative; every division is
//!   guarded, inheriting the reservoir reference's `NaN`-free guarantees.
//! * Every function is a deterministic pure function: no RNG, no I/O, no GPU,
//!   and no `unsafe`.  Transcendental maths goes through [`bevy_math::ops`].

use bevy_math::Vec3;

use super::scattering::{beer_lambert, henyey_greenstein};
use crate::gi::screen_probe::restir::{luminance, Reservoir};

/// Homogeneous participating-medium parameters for a froxel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MediumParams {
    /// Scattering coefficient `sigma_s` (redirected radiance per unit length).
    pub sigma_s: f32,
    /// Extinction coefficient `sigma_t = sigma_s + sigma_a` (attenuation rate).
    pub sigma_t: f32,
    /// Henyey-Greenstein anisotropy `g` in `(-1, 1)`.
    pub g: f32,
}

impl MediumParams {
    /// Creates medium parameters, clamping `sigma_s` / `sigma_t` non-negative
    /// (and `sigma_t >= sigma_s`), and `g` just inside `(-1, 1)`.
    #[inline]
    pub fn new(sigma_s: f32, sigma_t: f32, g: f32) -> Self {
        let sigma_s = sanitize_scalar(sigma_s);
        let sigma_t = sanitize_scalar(sigma_t).max(sigma_s);
        let g = if g.is_finite() {
            g.clamp(-0.999_9, 0.999_9)
        } else {
            0.0
        };
        Self {
            sigma_s,
            sigma_t,
            g,
        }
    }
}

/// A single volumetric light candidate (the reservoir payload).
///
/// Records the light's world-space position and the linear RGB radiance it
/// emits towards the froxel.  All fields are `f32`-backed [`Vec3`]s to match the
/// GPU sample-buffer twin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VolumetricSample {
    /// World-space position of the sampled light.
    pub light_position: Vec3,
    /// Linear RGB radiance emitted by the light towards the froxel.
    pub radiance: Vec3,
}

impl VolumetricSample {
    /// A zero-energy sample that contributes nothing.
    pub const ZERO: Self = Self {
        light_position: Vec3::ZERO,
        radiance: Vec3::ZERO,
    };
}

impl Default for VolumetricSample {
    #[inline]
    fn default() -> Self {
        Self::ZERO
    }
}

/// Per-sample in-scattered, transmittance-weighted radiance (vector integrand).
///
/// Computes `sigma_s * p(cos_theta, g) * T(sigma_t, dist) * radiance`, where the
/// phase cosine uses the unit direction from `scatter_pos` to the light and the
/// supplied `view_dir`, `dist` is the froxel-to-light distance, and `T` is the
/// Beer-Lambert transmittance along it.  Returns zero for a coincident light or
/// a degenerate view direction, and is always finite and non-negative.
#[inline]
pub fn integrand(
    sample: &VolumetricSample,
    scatter_pos: Vec3,
    view_dir: Vec3,
    medium: MediumParams,
) -> Vec3 {
    let to_light = sample.light_position - scatter_pos;
    let dist_sq = to_light.length_squared();
    if !dist_sq.is_finite() || dist_sq <= f32::MIN_POSITIVE {
        return Vec3::ZERO;
    }
    let dist = dist_sq.sqrt();
    let wi = to_light * dist.recip();
    let wo = normalize_or_zero(view_dir);
    let cos_theta = wi.dot(wo);
    let phase = henyey_greenstein(cos_theta, medium.g);
    let transmittance = beer_lambert(medium.sigma_t, dist);
    let scale = medium.sigma_s * phase * transmittance;
    sanitize_rgb(sample.radiance * scale)
}

/// Scalar resampling target `p_hat` for a volumetric sample: the luminance of
/// its [`integrand`].
///
/// Always finite and non-negative; a dark, occluded, or degenerate sample
/// yields `0`, which the reservoir then discards.
#[inline]
pub fn scatter_target(
    sample: &VolumetricSample,
    scatter_pos: Vec3,
    view_dir: Vec3,
    medium: MediumParams,
) -> f32 {
    let t = luminance(integrand(sample, scatter_pos, view_dir, medium));
    if t.is_finite() {
        t.max(0.0)
    } else {
        0.0
    }
}

/// Streaming RIS over a batch of candidate light samples for one froxel.
///
/// Folds each `candidates[i]` into a fresh reservoir with resampling weight
/// `scatter_target / source_pdfs[i]`, driven by the uniform `uniforms[i]`, then
/// finalises the contribution weight against the surviving sample's target.
/// The three slices are consumed over their shortest common prefix.  Candidates
/// with a non-positive / non-finite source pdf are skipped.
#[inline]
pub fn resample_froxel(
    candidates: &[VolumetricSample],
    source_pdfs: &[f32],
    uniforms: &[f32],
    scatter_pos: Vec3,
    view_dir: Vec3,
    medium: MediumParams,
) -> Reservoir<VolumetricSample> {
    let mut reservoir = Reservoir::<VolumetricSample>::new();
    let n = candidates.len().min(source_pdfs.len()).min(uniforms.len());
    for i in 0..n {
        let candidate = candidates[i];
        let source_pdf = source_pdfs[i];
        if !source_pdf.is_finite() || source_pdf <= 0.0 {
            continue;
        }
        let target = scatter_target(&candidate, scatter_pos, view_dir, medium);
        let weight = target / source_pdf;
        reservoir.update(candidate, weight, uniforms[i]);
    }
    finalize(&mut reservoir, scatter_pos, view_dir, medium);
    reservoir
}

/// Temporal reuse: fold a reprojected previous-frame `history` reservoir into
/// `current`, then re-finalise.
///
/// `history` is first confidence-capped to `m_cap` (the ReSTIR M-cap bounding
/// temporal history); its surviving sample's target is re-evaluated in the
/// current froxel's domain and merged in with the uniform `u`.  After the merge
/// the contribution weight is recomputed for whichever sample now survives, so
/// the returned reservoir is a ready-to-use temporal estimator.
#[inline]
pub fn temporal_resample(
    current: &mut Reservoir<VolumetricSample>,
    history: &Reservoir<VolumetricSample>,
    scatter_pos: Vec3,
    view_dir: Vec3,
    medium: MediumParams,
    m_cap: f32,
    u: f32,
) {
    let mut prev = *history;
    prev.cap_confidence(m_cap);
    let history_target = match prev.sample() {
        Some(s) => scatter_target(&s, scatter_pos, view_dir, medium),
        None => 0.0,
    };
    current.merge(&prev, history_target, u);
    finalize(current, scatter_pos, view_dir, medium);
}

/// Unbiased in-scattered radiance estimate from a finalised reservoir:
/// `integrand(selected) * W`.
///
/// Returns zero when the reservoir is empty or its contribution weight is zero;
/// always finite and non-negative.
#[inline]
pub fn estimate_radiance(
    reservoir: &Reservoir<VolumetricSample>,
    scatter_pos: Vec3,
    view_dir: Vec3,
    medium: MediumParams,
) -> Vec3 {
    match reservoir.sample() {
        Some(s) => {
            let f = integrand(&s, scatter_pos, view_dir, medium);
            sanitize_rgb(f * reservoir.contribution_weight())
        }
        None => Vec3::ZERO,
    }
}

/// Re-evaluates the surviving sample's target and finalises the reservoir's
/// contribution weight.
#[inline]
fn finalize(
    reservoir: &mut Reservoir<VolumetricSample>,
    scatter_pos: Vec3,
    view_dir: Vec3,
    medium: MediumParams,
) {
    let p_hat = match reservoir.sample() {
        Some(s) => scatter_target(&s, scatter_pos, view_dir, medium),
        None => 0.0,
    };
    reservoir.finalize_weight(p_hat);
}

/// Normalises `v`, returning zero for a degenerate (near-zero / non-finite)
/// input so downstream dot products stay finite.
#[inline]
fn normalize_or_zero(v: Vec3) -> Vec3 {
    let len_sq = v.length_squared();
    if len_sq.is_finite() && len_sq > f32::MIN_POSITIVE {
        v * len_sq.sqrt().recip()
    } else {
        Vec3::ZERO
    }
}

/// Clamps a scalar coefficient non-negative and finite.
#[inline]
fn sanitize_scalar(value: f32) -> f32 {
    if value.is_finite() {
        value.max(0.0)
    } else {
        0.0
    }
}

/// Replaces non-finite channels with `0` and clamps every channel non-negative.
#[inline]
fn sanitize_rgb(rgb: Vec3) -> Vec3 {
    Vec3::new(
        if rgb.x.is_finite() { rgb.x.max(0.0) } else { 0.0 },
        if rgb.y.is_finite() { rgb.y.max(0.0) } else { 0.0 },
        if rgb.z.is_finite() { rgb.z.max(0.0) } else { 0.0 },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn medium() -> MediumParams {
        MediumParams::new(0.5, 1.0, 0.0)
    }

    #[test]
    fn medium_new_clamps() {
        let m = MediumParams::new(-1.0, 0.2, 5.0);
        assert_eq!(m.sigma_s, 0.0);
        assert!(m.sigma_t >= m.sigma_s);
        assert!(m.g.abs() < 1.0);
        let m2 = MediumParams::new(2.0, 1.0, f32::NAN);
        // sigma_t is lifted to at least sigma_s.
        assert!(m2.sigma_t >= m2.sigma_s);
        assert_eq!(m2.g, 0.0);
    }

    #[test]
    fn integrand_zero_for_coincident_light() {
        let s = VolumetricSample {
            light_position: Vec3::ZERO,
            radiance: Vec3::ONE,
        };
        let out = integrand(&s, Vec3::ZERO, Vec3::Z, medium());
        assert_eq!(out, Vec3::ZERO);
    }

    #[test]
    fn integrand_matches_manual_formula() {
        let m = MediumParams::new(0.5, 1.0, 0.0);
        let scatter = Vec3::ZERO;
        let view = Vec3::Z;
        let s = VolumetricSample {
            light_position: Vec3::new(0.0, 0.0, 2.0),
            radiance: Vec3::new(3.0, 1.0, 0.5),
        };
        let out = integrand(&s, scatter, view, m);
        // wi = +Z, cos = 1, isotropic phase = 1/4pi, dist = 2.
        let phase = henyey_greenstein(1.0, 0.0);
        let t = beer_lambert(1.0, 2.0);
        let scale = 0.5 * phase * t;
        let expected = Vec3::new(3.0, 1.0, 0.5) * scale;
        assert!((out - expected).abs().max_element() < 1e-6, "out={out} exp={expected}");
    }

    #[test]
    fn scatter_target_is_non_negative_and_finite() {
        let s = VolumetricSample {
            light_position: Vec3::new(1.0, 0.0, 1.0),
            radiance: Vec3::new(-1.0, 2.0, 3.0),
        };
        let t = scatter_target(&s, Vec3::ZERO, Vec3::Z, medium());
        assert!(t.is_finite() && t >= 0.0);
    }

    #[test]
    fn resample_selects_from_candidates_and_is_unbiased_on_average() {
        // Two candidates with equal uniform source pdf; brighter one dominates.
        let scatter = Vec3::ZERO;
        let view = Vec3::Z;
        let m = medium();
        let candidates = [
            VolumetricSample {
                light_position: Vec3::new(0.0, 0.0, 1.0),
                radiance: Vec3::splat(0.1),
            },
            VolumetricSample {
                light_position: Vec3::new(0.0, 0.0, 1.0),
                radiance: Vec3::splat(10.0),
            },
        ];
        // With fixed discrete candidates the RIS estimator averages
        // `(1/M) * sum f(x_i)/pdf_i`, so a source pdf of `1/M` per candidate
        // (the uniform candidate-selection probability) recovers the full sum.
        let pdfs = [0.5, 0.5];

        // Monte-Carlo-average the single-sample RIS estimate over many uniforms;
        // it must approach the exact two-sample integrand sum.
        let exact = integrand(&candidates[0], scatter, view, m)
            + integrand(&candidates[1], scatter, view, m);
        let trials = 4000;
        let mut acc = Vec3::ZERO;
        for i in 0..trials {
            // Deterministic low-discrepancy-ish stream of distinct uniforms.
            let u0 = (i as f32 * 0.618_034) % 1.0;
            let u1 = (i as f32 * 0.381_966 + 0.5) % 1.0;
            let r = resample_froxel(&candidates, &pdfs, &[u0, u1], scatter, view, m);
            acc += estimate_radiance(&r, scatter, view, m);
        }
        let avg = acc / trials as f32;
        assert!(
            (avg - exact).abs().max_element() < 5e-2,
            "avg={avg} exact={exact}"
        );
    }

    #[test]
    fn resample_skips_degenerate_source_pdfs() {
        let scatter = Vec3::ZERO;
        let view = Vec3::Z;
        let m = medium();
        let candidates = [VolumetricSample {
            light_position: Vec3::new(0.0, 0.0, 1.0),
            radiance: Vec3::splat(5.0),
        }];
        // Non-positive source pdf -> candidate skipped -> empty reservoir.
        let r = resample_froxel(&candidates, &[0.0], &[0.5], scatter, view, m);
        assert!(r.is_empty());
        assert_eq!(r.contribution_weight(), 0.0);
    }

    #[test]
    fn resample_handles_mismatched_slice_lengths() {
        let scatter = Vec3::ZERO;
        let view = Vec3::Z;
        let m = medium();
        let candidates = [
            VolumetricSample {
                light_position: Vec3::new(0.0, 0.0, 1.0),
                radiance: Vec3::splat(1.0),
            },
            VolumetricSample {
                light_position: Vec3::new(0.0, 0.0, 2.0),
                radiance: Vec3::splat(1.0),
            },
        ];
        // Only one pdf / uniform: the second candidate is never visited.
        let r = resample_froxel(&candidates, &[1.0], &[0.3], scatter, view, m);
        assert_eq!(r.confidence(), 1.0);
    }

    #[test]
    fn temporal_resample_can_adopt_history_sample() {
        let scatter = Vec3::ZERO;
        let view = Vec3::Z;
        let m = medium();
        // Current reservoir: a dim sample.
        let mut current = resample_froxel(
            &[VolumetricSample {
                light_position: Vec3::new(0.0, 0.0, 1.0),
                radiance: Vec3::splat(0.01),
            }],
            &[1.0],
            &[0.5],
            scatter,
            view,
            m,
        );
        // History reservoir: a strong sample with accumulated confidence.
        let mut history = resample_froxel(
            &[VolumetricSample {
                light_position: Vec3::new(0.0, 0.0, 1.0),
                radiance: Vec3::splat(100.0),
            }],
            &[1.0],
            &[0.5],
            scatter,
            view,
            m,
        );
        for _ in 0..9 {
            // Inflate confidence by re-merging itself (temporal history buildup).
            let clone = history;
            let p = history.sample().map(|s| scatter_target(&s, scatter, view, m)).unwrap_or(0.0);
            history.merge(&clone, p, 1.0);
        }
        // u = 0 forces selecting the strong history sample.
        temporal_resample(&mut current, &history, scatter, view, m, 20.0, 0.0);
        assert_eq!(
            current.sample().map(|s| s.radiance),
            Some(Vec3::splat(100.0))
        );
        let est = estimate_radiance(&current, scatter, view, m);
        assert!(est.is_finite() && est.max_element() > 0.0);
    }

    #[test]
    fn temporal_resample_caps_history_confidence() {
        let scatter = Vec3::ZERO;
        let view = Vec3::Z;
        let m = medium();
        let mut current = Reservoir::<VolumetricSample>::new();
        let mut history = resample_froxel(
            &[VolumetricSample {
                light_position: Vec3::new(0.0, 0.0, 1.0),
                radiance: Vec3::splat(1.0),
            }],
            &[1.0],
            &[0.5],
            scatter,
            view,
            m,
        );
        // Pump the history confidence well above the cap.
        for _ in 0..200 {
            let clone = history;
            let p = history.sample().map(|s| scatter_target(&s, scatter, view, m)).unwrap_or(0.0);
            history.merge(&clone, p, 1.0);
        }
        temporal_resample(&mut current, &history, scatter, view, m, 20.0, 0.0);
        // Current started empty, so its confidence is exactly the capped history.
        assert!(current.confidence() <= 20.0 + 1e-6, "m={}", current.confidence());
    }

    #[test]
    fn estimate_radiance_empty_is_zero() {
        let r = Reservoir::<VolumetricSample>::new();
        let out = estimate_radiance(&r, Vec3::ZERO, Vec3::Z, medium());
        assert_eq!(out, Vec3::ZERO);
    }

    #[test]
    fn results_are_deterministic() {
        let scatter = Vec3::ZERO;
        let view = Vec3::Z;
        let m = medium();
        let candidates = [VolumetricSample {
            light_position: Vec3::new(0.5, 0.2, 1.0),
            radiance: Vec3::new(1.0, 2.0, 3.0),
        }];
        let a = resample_froxel(&candidates, &[0.7], &[0.4], scatter, view, m);
        let b = resample_froxel(&candidates, &[0.7], &[0.4], scatter, view, m);
        assert_eq!(a, b);
        assert_eq!(
            estimate_radiance(&a, scatter, view, m),
            estimate_radiance(&b, scatter, view, m)
        );
    }

    #[test]
    fn no_nan_on_degenerate_inputs() {
        let m = MediumParams::new(f32::NAN, f32::INFINITY, f32::NAN);
        let s = VolumetricSample {
            light_position: Vec3::splat(f32::NAN),
            radiance: Vec3::splat(f32::INFINITY),
        };
        let out = integrand(&s, Vec3::splat(f32::NAN), Vec3::splat(f32::NAN), m);
        assert!(out.is_finite());
        assert!(scatter_target(&s, Vec3::ZERO, Vec3::Z, m).is_finite());
    }
}
