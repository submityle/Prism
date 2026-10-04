//! Physically-based layered-glass transparency resolve.
//!
//! This is the resolve backing [`crate::transparency::TransparencyPath::LayeredGlass`]:
//! the path the router selects for stacked dielectric slabs (window panes,
//! vehicle glass, display covers) where the thin single-weight blends of
//! `weighted`/`moment` `OIT` would lose the sharp `Fresnel` rim and inter-pane
//! inter-reflections that read as "real glass". Instead of blending coverage,
//! it evaluates the exact collimated radiative transfer through a stack of
//! parallel absorbing dielectric plates and returns per-channel reflectance and
//! transmittance.
//!
//! The physics is the classical *adding method* (Stokes' equations for a pile
//! of plates): each plate's reflectance/transmittance already sums its infinite
//! internal inter-reflections in closed form via a geometric series, and
//! adjacent plates are combined by the same recurrence across their air gap.
//! `Fresnel` is the exact unpolarized dielectric reflectance (average of the s-
//! and p-polarized terms), and bulk attenuation is per-channel `Beer-Lambert`.
//! Energy is conserved: for a loss-free stack `R + T == 1` on every channel.
//!
//! Determinism: only `sqrt`, arithmetic, and the workspace's reproducible
//! [`crate::water::exp_approx`] are used; no forbidden `f32` transcendental
//! intrinsic appears.

use crate::water::exp_approx;

/// A single parallel dielectric slab in the stack.
#[derive(Clone, Copy, Debug)]
pub struct GlassLayer {
    /// Refractive index of the slab (air outside is assumed `1.0`).
    pub ior: f32,
    /// Physical thickness along the slab normal, in metres.
    pub thickness: f32,
    /// Per-channel bulk absorption coefficient `sigma_a` (`1/m`), giving the
    /// slab its tint via `Beer-Lambert` attenuation.
    pub absorption: [f32; 3],
}

impl GlassLayer {
    /// A clear slab of the given index and thickness (no absorption tint).
    #[must_use]
    pub fn clear(ior: f32, thickness: f32) -> Self {
        Self {
            ior,
            thickness,
            absorption: [0.0; 3],
        }
    }
}

/// Per-channel reflectance/transmittance of a slab or stack.
///
/// Both are in `[0, 1]` per channel; `1 - reflect - transmit` is the absorbed
/// fraction (zero for a loss-free dielectric).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReflectTransmit {
    /// Fraction of incident radiance reflected back toward the viewer.
    pub reflect: [f32; 3],
    /// Fraction of incident radiance transmitted through to the far side.
    pub transmit: [f32; 3],
}

/// Exact unpolarized `Fresnel` reflectance at a dielectric interface.
///
/// `cos_i` is the cosine of the incidence angle in the medium of index `n1`;
/// `n2` is the index on the far side. Returns `1.0` under total internal
/// reflection. The result averages the s- and p-polarized reflectances, which
/// is the correct unpolarized response for environment lighting.
#[must_use]
fn fresnel_dielectric(cos_i: f32, n1: f32, n2: f32) -> f32 {
    let ci = cos_i.clamp(0.0, 1.0);
    let eta = n1 / n2;
    let sin_t2 = eta * eta * (1.0 - ci * ci);
    if sin_t2 >= 1.0 {
        return 1.0; // Total internal reflection.
    }
    let ct = (1.0 - sin_t2).sqrt();
    let rs = (n1 * ci - n2 * ct) / (n1 * ci + n2 * ct);
    let rp = (n1 * ct - n2 * ci) / (n1 * ct + n2 * ci);
    0.5 * (rs * rs + rp * rp)
}

/// Cosine of the refracted angle inside a slab of index `n2` for air-side
/// incidence cosine `cos_i`, or `None` under total internal reflection.
#[must_use]
fn refracted_cos(cos_i: f32, n1: f32, n2: f32) -> Option<f32> {
    let ci = cos_i.clamp(0.0, 1.0);
    let eta = n1 / n2;
    let sin_t2 = eta * eta * (1.0 - ci * ci);
    if sin_t2 >= 1.0 {
        return None;
    }
    Some((1.0 - sin_t2).sqrt())
}

/// Reflectance/transmittance of one slab, summing its internal
/// inter-reflections in closed form.
///
/// `cos_i` is the air-side incidence cosine. The two slab faces share the same
/// interface `Fresnel` `f` (reciprocity), separated by the one-way bulk
/// transmittance `tau = exp(-sigma_a * thickness / cos_t)`. The geometric
/// series over internal bounces gives, per channel:
///
/// ```text
/// T = (1 - f)^2 * tau / (1 - f^2 * tau^2)
/// R = f + (1 - f)^2 * f * tau^2 / (1 - f^2 * tau^2)
/// ```
#[must_use]
pub fn slab_response(layer: &GlassLayer, cos_i: f32) -> ReflectTransmit {
    let f = fresnel_dielectric(cos_i, 1.0, layer.ior);
    let Some(cos_t) = refracted_cos(cos_i, 1.0, layer.ior) else {
        // Cannot physically occur entering from air, but stay well-defined.
        return ReflectTransmit {
            reflect: [1.0; 3],
            transmit: [0.0; 3],
        };
    };
    let inv_cos_t = if cos_t <= 1.0e-4 { 1.0e4 } else { 1.0 / cos_t };
    let one_minus_f = 1.0 - f;
    let mut reflect = [0.0_f32; 3];
    let mut transmit = [0.0_f32; 3];
    for c in 0..3 {
        let tau = exp_approx(-layer.absorption[c] * layer.thickness * inv_cos_t).clamp(0.0, 1.0);
        let denom = 1.0 - f * f * tau * tau;
        let denom = if denom.abs() <= 1.0e-6 { 1.0e-6 } else { denom };
        transmit[c] = (one_minus_f * one_minus_f * tau / denom).clamp(0.0, 1.0);
        reflect[c] =
            (f + one_minus_f * one_minus_f * f * tau * tau / denom).clamp(0.0, 1.0);
    }
    ReflectTransmit { reflect, transmit }
}

/// Combines two stacked responses (`front` nearer the viewer) across their air
/// gap using the adding recurrence, per channel:
///
/// ```text
/// R = R_f + T_f^2 * R_b / (1 - R_f * R_b)
/// T = T_f * T_b / (1 - R_f * R_b)
/// ```
#[must_use]
fn add_layers(front: &ReflectTransmit, back: &ReflectTransmit) -> ReflectTransmit {
    let mut reflect = [0.0_f32; 3];
    let mut transmit = [0.0_f32; 3];
    for c in 0..3 {
        let rf = front.reflect[c];
        let tf = front.transmit[c];
        let rb = back.reflect[c];
        let tb = back.transmit[c];
        let denom = 1.0 - rf * rb;
        let denom = if denom.abs() <= 1.0e-6 { 1.0e-6 } else { denom };
        reflect[c] = (rf + tf * tf * rb / denom).clamp(0.0, 1.0);
        transmit[c] = (tf * tb / denom).clamp(0.0, 1.0);
    }
    ReflectTransmit { reflect, transmit }
}

/// Reflectance/transmittance of the whole ordered stack (index `0` is the slab
/// nearest the viewer), evaluated at air-side incidence cosine `cos_i`.
///
/// Returns a fully transparent response for an empty stack.
#[must_use]
pub fn stack_response(layers: &[GlassLayer], cos_i: f32) -> ReflectTransmit {
    let mut acc = ReflectTransmit {
        reflect: [0.0; 3],
        transmit: [1.0; 3],
    };
    let mut first = true;
    for layer in layers {
        let slab = slab_response(layer, cos_i);
        if first {
            acc = slab;
            first = false;
        } else {
            acc = add_layers(&acc, &slab);
        }
    }
    acc
}

/// Composites layered glass over the scene.
///
/// `background` is the radiance behind the glass (already shaded), `reflection`
/// is the environment radiance seen in the glass. The result is
/// `transmit ⊙ background + reflect ⊙ reflection`, per channel, which is energy
/// consistent: a loss-free stack neither creates nor destroys radiance.
#[must_use]
pub fn resolve(
    layers: &[GlassLayer],
    cos_i: f32,
    background: [f32; 3],
    reflection: [f32; 3],
) -> [f32; 3] {
    let rt = stack_response(layers, cos_i);
    let mut out = [0.0_f32; 3];
    for c in 0..3 {
        out[c] = rt.transmit[c] * background[c] + rt.reflect[c] * reflection[c];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn normal_incidence_fresnel_matches_schlick_f0() {
        // At normal incidence F = ((n1-n2)/(n1+n2))^2; for n=1.5 that is 0.04.
        let f = fresnel_dielectric(1.0, 1.0, 1.5);
        assert!(approx(f, 0.04, 1e-3), "F0 = {f}");
    }

    #[test]
    fn grazing_incidence_reflects_everything() {
        let f = fresnel_dielectric(0.0, 1.0, 1.5);
        assert!(approx(f, 1.0, 1e-3), "grazing F = {f}");
    }

    #[test]
    fn clear_single_pane_conserves_energy() {
        // No absorption -> R + T == 1 on every channel, every angle.
        let pane = GlassLayer::clear(1.5, 0.006);
        for &ci in &[1.0_f32, 0.8, 0.5, 0.2, 0.05] {
            let rt = slab_response(&pane, ci);
            for c in 0..3 {
                assert!(
                    approx(rt.reflect[c] + rt.transmit[c], 1.0, 2e-3),
                    "ci={ci} c={c} R+T={}",
                    rt.reflect[c] + rt.transmit[c]
                );
            }
        }
    }

    #[test]
    fn clear_pane_transmits_most_at_normal() {
        // n=1.5 pane at normal: two 4% interfaces -> T ~ 0.92-0.93.
        let pane = GlassLayer::clear(1.5, 0.006);
        let rt = slab_response(&pane, 1.0);
        assert!(rt.transmit[0] > 0.9 && rt.transmit[0] < 0.94, "T={}", rt.transmit[0]);
    }

    #[test]
    fn absorption_reduces_transmission() {
        let clear = GlassLayer::clear(1.5, 0.02);
        let tinted = GlassLayer {
            ior: 1.5,
            thickness: 0.02,
            absorption: [40.0, 10.0, 5.0], // strong red absorption
        };
        let tc = slab_response(&clear, 1.0);
        let tt = slab_response(&tinted, 1.0);
        assert!(tt.transmit[0] < tc.transmit[0], "tinted should transmit less red");
        // Red absorbed hardest -> less red than blue through the tint.
        assert!(tt.transmit[0] < tt.transmit[2], "red < blue transmit");
    }

    #[test]
    fn more_panes_transmit_less() {
        let pane = GlassLayer::clear(1.5, 0.006);
        let one = stack_response(&[pane], 1.0);
        let three = stack_response(&[pane, pane, pane], 1.0);
        assert!(
            three.transmit[0] < one.transmit[0],
            "3 panes {} should transmit less than 1 pane {}",
            three.transmit[0],
            one.transmit[0]
        );
        // Still energy consistent for the clear stack.
        for c in 0..3 {
            assert!(approx(three.reflect[c] + three.transmit[c], 1.0, 3e-3));
        }
    }

    #[test]
    fn empty_stack_is_fully_transparent() {
        let rt = stack_response(&[], 1.0);
        assert_eq!(rt.transmit, [1.0; 3]);
        assert_eq!(rt.reflect, [0.0; 3]);
        let bg = [0.3, 0.5, 0.7];
        assert_eq!(resolve(&[], 1.0, bg, [1.0; 3]), bg);
    }

    #[test]
    fn resolve_blends_background_and_reflection() {
        let panes = vec![GlassLayer::clear(1.5, 0.006)];
        let bg = [0.2, 0.2, 0.2];
        let refl = [1.0, 1.0, 1.0];
        let out = resolve(&panes, 1.0, bg, refl);
        let rt = stack_response(&panes, 1.0);
        for c in 0..3 {
            let expect = rt.transmit[c] * bg[c] + rt.reflect[c] * refl[c];
            assert!(approx(out[c], expect, 1e-6));
        }
        // Output stays within the lit range for these inputs.
        for c in &out {
            assert!(*c >= 0.0 && *c <= 1.0, "out channel {c}");
        }
    }

    #[test]
    fn stack_adding_is_associative_for_identical_panes() {
        // Adding pane-by-pane must equal nesting, a basic consistency check.
        let pane = GlassLayer::clear(1.52, 0.004);
        let seq = stack_response(&[pane, pane], 0.7);
        let manual = add_layers(&slab_response(&pane, 0.7), &slab_response(&pane, 0.7));
        for c in 0..3 {
            assert!(approx(seq.reflect[c], manual.reflect[c], 1e-6));
            assert!(approx(seq.transmit[c], manual.transmit[c], 1e-6));
        }
    }
}
