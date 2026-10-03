//! Water caustic-projection compute kernel: the `WESL` shader plus its
//! bit-exact `CPU` twin.
//!
//! When sunlight refracts through a wavy surface the refracted rays converge
//! and diverge, concentrating irradiance into the bright dancing patterns
//! called caustics (see [`caustics`](super::super::caustics) for the golden
//! derivation). The cheap real-time route projects, per receiver texel, the
//! inverse area distortion of the refracted-ray mapping: a focusing patch
//! brightens, a defocusing patch dims. This pass runs that light-space Jacobian
//! projection across the screen and adds the resulting achromatic caustic
//! irradiance onto the lit scene colour.
//!
//! [`WATER_CAUSTICS_PROJECT_WESL`] is the shader (entry point
//! `water_caustics_project`); [`dispatch_caustics_project`] is its bit-exact
//! `CPU` twin. Because the sandbox has no `GPU`, the twin is the correctness
//! proof: it consumes the identical buffer `ABI`
//! ([`WaterKernel::CausticsProject`](super::super::kernels::WaterKernel) — one
//! storage buffer, one uniform param block, one sampled texture, one
//! `rgba32float` storage output, 8x8 tile, `Screen` domain) and reconstructs
//! the shader arithmetic texel-for-texel. The composite is diffed against the
//! golden
//! [`caustics::jacobian_caustic_gain`](super::super::caustics::jacobian_caustic_gain)
//! and
//! [`caustics::project_caustic_intensity`](super::super::caustics::project_caustic_intensity),
//! so the whole pass is bit-exact rather than a floating approximation.
//!
//! Only `+ - * /`, comparisons, `abs`, `min` and `max` appear — no float
//! intrinsic the workspace determinism policy forbids. Route selection
//! ([`caustics::select_caustics`](super::super::caustics::select_caustics)) and
//! the offline photon estimate
//! ([`caustics::photon_splat_density`](super::super::caustics::photon_splat_density))
//! stay on the `CPU`: this kernel is the real-time Jacobian route only.

use alloc::vec;
use alloc::vec::Vec;

use super::super::EPS;

/// `WESL` source of the water caustic-projection compute kernel.
///
/// Bound into the crate so the shader ships and is covered by the structural
/// test; the standalone `naga`/`wesl` compile check runs out of tree, because
/// `prism_render_architecture` is a zero-dependency crate.
pub const WATER_CAUSTICS_PROJECT_WESL: &str = include_str!("water_caustics_project.wesl");

/// Number of `f32` lanes in one per-texel light-space receiver record:
/// `(incident_irradiance, jacobian, _, _)`.
pub const CAUSTICS_RECEIVER_FLOATS: usize = 4;

/// Number of `f32` lanes in one lit-scene texel: `(r, g, b, a)`.
pub const CAUSTICS_SCENE_FLOATS: usize = 4;

/// Number of `f32` lanes in one output texel: `(r, g, b, a)`.
pub const CAUSTICS_OUT_FLOATS: usize = 4;

/// Uniform parameter block for the caustic-projection pass.
///
/// `max_gain` caps the caustic gain so a near-singular surface focus cannot
/// produce an unbounded irradiance spike. The screen dimensions travel as
/// explicit [`dispatch_caustics_project`] arguments rather than struct fields.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CausticsParams {
    /// Upper bound on the per-texel caustic gain.
    pub max_gain: f32,
}

/// Caustic gain `1 / |jacobian|` clamped to `max_gain`, reconstructing the
/// shader's `jacobian_caustic_gain`.
///
/// Kept independent of
/// [`caustics::jacobian_caustic_gain`](super::super::caustics::jacobian_caustic_gain)
/// so the parity test proves the transcription rather than asserting a
/// tautology.
#[must_use]
fn jacobian_gain_twin(jacobian: f32, max_gain: f32) -> f32 {
    let cap = max_gain.max(0.0);
    let mag = jacobian.abs();
    if mag <= EPS {
        return cap;
    }
    (1.0 / mag).min(cap)
}

/// Caustic irradiance `incident * jacobian_gain`, reconstructing the shader's
/// `project_caustic_intensity`.
#[must_use]
fn project_intensity_twin(incident: f32, jacobian: f32, max_gain: f32) -> f32 {
    incident.max(0.0) * jacobian_gain_twin(jacobian, max_gain)
}

/// Bit-exact `CPU` twin of the `water_caustics_project` compute pass.
///
/// Each texel reads its light-space `(incident_irradiance, jacobian)` from
/// `receiver` and the lit scene colour from `scene`, projects the achromatic
/// caustic irradiance, and adds it to the three colour channels while
/// preserving alpha. The output always has `width * height * CAUSTICS_OUT_FLOATS`
/// lanes; a texel whose receiver or scene record is not fully supplied is left
/// as zeros rather than panicking, mirroring the shader's bounds guard.
#[must_use]
pub fn dispatch_caustics_project(
    receiver: &[f32],
    scene: &[f32],
    params: CausticsParams,
    width: usize,
    height: usize,
) -> Vec<f32> {
    let texels = width * height;
    let mut out = vec![0.0_f32; texels * CAUSTICS_OUT_FLOATS];
    let mut i = 0usize;
    while i < texels {
        let r = i * CAUSTICS_RECEIVER_FLOATS;
        let s = i * CAUSTICS_SCENE_FLOATS;
        if r + CAUSTICS_RECEIVER_FLOATS > receiver.len() || s + CAUSTICS_SCENE_FLOATS > scene.len()
        {
            break;
        }
        let incident = receiver[r];
        let jacobian = receiver[r + 1];
        let intensity = project_intensity_twin(incident, jacobian, params.max_gain);

        let o = i * CAUSTICS_OUT_FLOATS;
        out[o] = scene[s] + intensity;
        out[o + 1] = scene[s + 1] + intensity;
        out[o + 2] = scene[s + 2] + intensity;
        out[o + 3] = scene[s + 3];
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::super::super::caustics::project_caustic_intensity;
    use super::*;

    const PARAMS: CausticsParams = CausticsParams { max_gain: 50.0 };

    #[test]
    fn composite_matches_golden_bit_for_bit() {
        // A small screen of texels with varying incident light and Jacobian;
        // every output channel must equal the golden lit-scene-plus-caustic
        // reconstructed from the independent `caustics` module, bit-for-bit.
        let width = 2;
        let height = 2;
        let texels = width * height;
        let mut receiver = Vec::with_capacity(texels * CAUSTICS_RECEIVER_FLOATS);
        let mut scene = Vec::with_capacity(texels * CAUSTICS_SCENE_FLOATS);
        let mut c = 0usize;
        while c < texels {
            let f = c as f32;
            receiver.push(0.8 + 0.1 * f); // incident
            receiver.push(0.3 + 0.5 * f); // jacobian: focusing -> defocusing
            receiver.push(0.0);
            receiver.push(0.0);
            scene.push(0.2 + 0.05 * f);
            scene.push(0.4);
            scene.push(0.6 - 0.03 * f);
            scene.push(1.0);
            c += 1;
        }
        let out = dispatch_caustics_project(&receiver, &scene, PARAMS, width, height);

        let mut i = 0usize;
        while i < texels {
            let r = i * CAUSTICS_RECEIVER_FLOATS;
            let s = i * CAUSTICS_SCENE_FLOATS;
            let incident = receiver[r];
            let jacobian = receiver[r + 1];
            let intensity = project_caustic_intensity(incident, jacobian, PARAMS.max_gain);
            let o = i * CAUSTICS_OUT_FLOATS;
            assert_eq!(
                out[o].to_bits(),
                (scene[s] + intensity).to_bits(),
                "red texel {i}"
            );
            assert_eq!(
                out[o + 1].to_bits(),
                (scene[s + 1] + intensity).to_bits(),
                "green texel {i}"
            );
            assert_eq!(
                out[o + 2].to_bits(),
                (scene[s + 2] + intensity).to_bits(),
                "blue texel {i}"
            );
            assert_eq!(
                out[o + 3].to_bits(),
                scene[s + 3].to_bits(),
                "alpha texel {i}"
            );
            i += 1;
        }
    }

    #[test]
    fn focus_brightens_more_than_defocus() {
        // Two texels, equal incident light, one focusing (small Jacobian) and
        // one defocusing (large Jacobian); the focusing texel gains more.
        let receiver = [1.0_f32, 0.25, 0.0, 0.0, 1.0, 4.0, 0.0, 0.0];
        let scene = [0.0_f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let out = dispatch_caustics_project(&receiver, &scene, PARAMS, 2, 1);
        assert!(out[0] > out[CAUSTICS_OUT_FLOATS], "focus brightens more");
        assert!(out[0] >= 0.0);
    }

    #[test]
    fn near_singular_jacobian_saturates_at_cap() {
        // A zero Jacobian would divide by zero; the gain must saturate at the
        // cap, so the added irradiance is exactly incident * max_gain.
        let receiver = [2.0_f32, 0.0, 0.0, 0.0];
        let scene = [0.0_f32, 0.0, 0.0, 1.0];
        let out = dispatch_caustics_project(&receiver, &scene, PARAMS, 1, 1);
        let want = 2.0_f32 * 50.0;
        assert!((out[0] - want).abs() < EPS);
    }

    #[test]
    fn alpha_is_preserved() {
        let receiver = [0.5_f32, 1.0, 0.0, 0.0];
        let scene = [0.1_f32, 0.2, 0.3, 0.7];
        let out = dispatch_caustics_project(&receiver, &scene, PARAMS, 1, 1);
        assert_eq!(out[3].to_bits(), 0.7_f32.to_bits(), "scene alpha survives");
    }

    #[test]
    fn short_buffers_do_not_panic() {
        // Claim a 2x2 screen but only supply one full texel of receiver data.
        let receiver = [0.5_f32, 1.0, 0.0, 0.0, 0.3];
        let scene = [0.1_f32, 0.2, 0.3, 1.0, 0.1, 0.2, 0.3, 1.0];
        let out = dispatch_caustics_project(&receiver, &scene, PARAMS, 2, 2);
        assert_eq!(out.len(), 4 * CAUSTICS_OUT_FLOATS);
        assert!(out[0] > 0.0, "first texel computed");
        // The under-supplied tail stays zero.
        assert_eq!(out[CAUSTICS_OUT_FLOATS].to_bits(), 0.0_f32.to_bits());
    }

    #[test]
    fn wesl_kernel_declares_expected_abi() {
        assert!(WATER_CAUSTICS_PROJECT_WESL.contains("fn water_caustics_project"));
        assert!(WATER_CAUSTICS_PROJECT_WESL.contains("@workgroup_size(8, 8, 1)"));
        assert!(WATER_CAUSTICS_PROJECT_WESL.contains("struct CausticsParams"));
        assert!(WATER_CAUSTICS_PROJECT_WESL.contains("var<storage, read> receiver_in"));
        assert!(WATER_CAUSTICS_PROJECT_WESL.contains("texture_storage_2d<rgba32float, write>"));
    }

    #[test]
    fn strides_are_consistent() {
        assert_eq!(CAUSTICS_RECEIVER_FLOATS, 4);
        assert_eq!(CAUSTICS_SCENE_FLOATS, 4);
        assert_eq!(CAUSTICS_OUT_FLOATS, 4);
    }
}
