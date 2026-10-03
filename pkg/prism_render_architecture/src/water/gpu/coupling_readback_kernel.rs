//! Water two-way-coupling read-back compute kernel: the `WESL` shader plus its
//! bit-exact `CPU` twin.
//!
//! Two-way fluid/rigid coupling needs the fluid's reaction on every floating
//! body: the `Archimedes` buoyancy that lifts it, the quadratic form drag that
//! resists its motion, the added-mass inertia of the water it drags along, and
//! the fraction of its momentum stamped back into the fluid (see
//! [`coupling`](super::super::coupling) for the golden derivation). Rather than
//! reading the whole field back per frame, the renderer batches the per-body
//! force evaluation into one bounded read-back pass: one invocation per body
//! turns its sampled state into the four coupling scalars.
//!
//! [`WATER_COUPLING_READBACK_WESL`] is the shader (entry point
//! `water_coupling_readback`); [`dispatch_coupling_readback`] is its bit-exact
//! `CPU` twin. Because the sandbox has no `GPU`, the twin is the correctness
//! proof: it consumes the identical buffer `ABI`
//! ([`WaterKernel::CouplingReadback`](super::super::kernels::WaterKernel) — two
//! storage buffers, one uniform param block, no textures, 64x1x1 brick,
//! `Particle` domain) and reconstructs the shader arithmetic body-for-body. The
//! outputs are diffed against the golden
//! [`coupling::buoyancy_force`](super::super::coupling::buoyancy_force),
//! [`coupling::drag_force`](super::super::coupling::drag_force),
//! [`coupling::added_mass`](super::super::coupling::added_mass) and
//! [`coupling::source_writeback_fraction`](super::super::coupling::source_writeback_fraction),
//! so the whole pass is bit-exact rather than a floating approximation.
//!
//! Only `+ - * /`, comparisons and `clamp`/`max` appear — no float intrinsic
//! the workspace determinism policy forbids. The per-frame sub-step / batch
//! schedule ([`coupling::plan_coupling`](super::super::coupling::plan_coupling))
//! stays on the `CPU`: it is one scalar per frame, not a per-body quantity, so
//! it is not part of this kernel.

use alloc::vec;
use alloc::vec::Vec;

use super::super::EPS;

/// `WESL` source of the water two-way-coupling read-back compute kernel.
///
/// Bound into the crate so the shader ships and is covered by the structural
/// test; the standalone `naga`/`wesl` compile check runs out of tree, because
/// `prism_render_architecture` is a zero-dependency crate.
pub const WATER_COUPLING_READBACK_WESL: &str = include_str!("water_coupling_readback.wesl");

/// Number of `f32` lanes in one per-body input record: `(submerged_volume,
/// total_volume, cross_section, rel_speed)`.
pub const COUPLING_BODY_FLOATS: usize = 4;

/// Number of `f32` lanes in one per-body output record: `(buoyancy, drag,
/// added_mass, writeback_fraction)`.
pub const COUPLING_FORCE_FLOATS: usize = 4;

/// Uniform parameter block for the coupling read-back pass.
///
/// `fluid_density` is the water density driving every force; `drag_coeff` and
/// `added_mass_coeff` are the dimensionless form-drag and added-mass
/// coefficients; `gravity` is the acceleration feeding the buoyancy weight. The
/// body count travels as an explicit [`dispatch_coupling_readback`] argument
/// rather than a struct field, matching the shader's `params.body_count` guard.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CouplingReadbackParams {
    /// Fluid (water) density.
    pub fluid_density: f32,
    /// Dimensionless quadratic form-drag coefficient.
    pub drag_coeff: f32,
    /// Dimensionless added-mass coefficient.
    pub added_mass_coeff: f32,
    /// Gravitational acceleration feeding the buoyancy weight.
    pub gravity: f32,
}

/// `Archimedes` buoyancy magnitude `fluid_density * submerged_volume *
/// gravity`, reconstructing the shader's `buoyancy_force`.
///
/// Kept independent of
/// [`coupling::buoyancy_force`](super::super::coupling::buoyancy_force) so the
/// parity test proves the transcription rather than asserting a tautology.
#[must_use]
fn buoyancy_twin(fluid_density: f32, submerged_volume: f32, gravity: f32) -> f32 {
    fluid_density.max(0.0) * submerged_volume.max(0.0) * gravity.max(0.0)
}

/// Quadratic form drag `0.5 * drag_coeff * fluid_density * area *
/// rel_speed^2`, reconstructing the shader's `drag_force`.
#[must_use]
fn drag_twin(drag_coeff: f32, fluid_density: f32, area: f32, rel_speed: f32) -> f32 {
    let v = rel_speed.max(0.0);
    0.5 * drag_coeff.max(0.0) * fluid_density.max(0.0) * area.max(0.0) * v * v
}

/// Added-mass reaction `added_mass_coeff * fluid_density * displaced_volume`,
/// reconstructing the shader's `added_mass`.
#[must_use]
fn added_mass_twin(added_mass_coeff: f32, fluid_density: f32, displaced_volume: f32) -> f32 {
    added_mass_coeff.max(0.0) * fluid_density.max(0.0) * displaced_volume.max(0.0)
}

/// Momentum write-back fraction `submerged_volume / total_volume` clamped to
/// `0..=1`, reconstructing the shader's `source_writeback_fraction`.
#[must_use]
fn writeback_fraction_twin(submerged_volume: f32, total_volume: f32) -> f32 {
    let total = total_volume.max(0.0);
    if total <= EPS {
        return 0.0;
    }
    (submerged_volume.max(0.0) / total).clamp(0.0, 1.0)
}

/// Bit-exact `CPU` twin of the `water_coupling_readback` compute pass.
///
/// Each body's four-lane state `(submerged_volume, total_volume, cross_section,
/// rel_speed)` becomes four coupling scalars `(buoyancy, drag, added_mass,
/// writeback_fraction)`. The drag area is the body's cross-section; the added
/// mass uses the submerged (displaced) volume, matching the golden per-frame
/// assembly. The output always has `body_count * COUPLING_FORCE_FLOATS` lanes;
/// a body whose input record is not fully supplied is left as zeros rather than
/// panicking, mirroring the shader's bounds guard.
#[must_use]
pub fn dispatch_coupling_readback(
    bodies: &[f32],
    params: CouplingReadbackParams,
    body_count: usize,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; body_count * COUPLING_FORCE_FLOATS];
    let mut i = 0usize;
    while i < body_count {
        let b = i * COUPLING_BODY_FLOATS;
        if b + COUPLING_BODY_FLOATS > bodies.len() {
            break;
        }
        let submerged_volume = bodies[b];
        let total_volume = bodies[b + 1];
        let cross_section = bodies[b + 2];
        let rel_speed = bodies[b + 3];

        let buoyancy = buoyancy_twin(params.fluid_density, submerged_volume, params.gravity);
        let drag = drag_twin(
            params.drag_coeff,
            params.fluid_density,
            cross_section,
            rel_speed,
        );
        let am = added_mass_twin(
            params.added_mass_coeff,
            params.fluid_density,
            submerged_volume,
        );
        let frac = writeback_fraction_twin(submerged_volume, total_volume);

        let o = i * COUPLING_FORCE_FLOATS;
        out[o] = buoyancy;
        out[o + 1] = drag;
        out[o + 2] = am;
        out[o + 3] = frac;
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::super::super::coupling::{
        added_mass, buoyancy_force, drag_force, source_writeback_fraction,
    };
    use super::super::super::GRAVITY;
    use super::*;

    const PARAMS: CouplingReadbackParams = CouplingReadbackParams {
        fluid_density: 1000.0,
        drag_coeff: 1.2,
        added_mass_coeff: 0.5,
        gravity: GRAVITY,
    };

    #[test]
    fn forces_match_golden_bit_for_bit() {
        // A handful of bodies with varying submersion, cross-section and speed;
        // every output lane must equal the golden `coupling` force reconstructed
        // from the independent module, bit-for-bit.
        let bodies = [
            0.1_f32, 1.0, 2.0, 1.0, // partly submerged, slow
            0.5, 0.5, 1.5, 3.0, // fully submerged, fast
            0.0, 2.0, 0.8, 0.0, // clear of the water, still
            2.5, 2.0, 4.0, 5.5, // over-submerged, churning
        ];
        let count = bodies.len() / COUPLING_BODY_FLOATS;
        let out = dispatch_coupling_readback(&bodies, PARAMS, count);

        let mut i = 0usize;
        while i < count {
            let b = i * COUPLING_BODY_FLOATS;
            let submerged = bodies[b];
            let total = bodies[b + 1];
            let area = bodies[b + 2];
            let speed = bodies[b + 3];

            let want_buoyancy = buoyancy_force(PARAMS.fluid_density, submerged, GRAVITY);
            let want_drag = drag_force(PARAMS.drag_coeff, PARAMS.fluid_density, area, speed);
            let want_am = added_mass(PARAMS.added_mass_coeff, PARAMS.fluid_density, submerged);
            let want_frac = source_writeback_fraction(submerged, total);

            let o = i * COUPLING_FORCE_FLOATS;
            assert_eq!(
                out[o].to_bits(),
                want_buoyancy.to_bits(),
                "buoyancy body {i}"
            );
            assert_eq!(out[o + 1].to_bits(), want_drag.to_bits(), "drag body {i}");
            assert_eq!(
                out[o + 2].to_bits(),
                want_am.to_bits(),
                "added_mass body {i}"
            );
            assert_eq!(
                out[o + 3].to_bits(),
                want_frac.to_bits(),
                "fraction body {i}"
            );
            i += 1;
        }
    }

    #[test]
    fn buoyancy_rises_with_submerged_volume() {
        // Two otherwise-identical bodies at different submersion depths: the
        // deeper one must feel more lift.
        let bodies = [0.1_f32, 1.0, 1.0, 0.0, 0.6, 1.0, 1.0, 0.0];
        let out = dispatch_coupling_readback(&bodies, PARAMS, 2);
        assert!(
            out[COUPLING_FORCE_FLOATS] > out[0],
            "deeper body lifts more"
        );
        assert!(out[0] >= 0.0);
    }

    #[test]
    fn drag_grows_with_speed_squared() {
        // Doubling the relative speed must quadruple the quadratic drag.
        let bodies = [0.5_f32, 1.0, 2.0, 1.0, 0.5, 1.0, 2.0, 2.0];
        let out = dispatch_coupling_readback(&bodies, PARAMS, 2);
        let slow = out[1];
        let fast = out[COUPLING_FORCE_FLOATS + 1];
        assert!((fast - 4.0 * slow).abs() < 1e-2 * fast.max(1.0));
        assert!(slow >= 0.0);
    }

    #[test]
    fn writeback_fraction_clamps_and_guards_zero_total() {
        // Over-submersion clamps to one; a degenerate zero total volume is inert.
        let bodies = [3.0_f32, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let out = dispatch_coupling_readback(&bodies, PARAMS, 2);
        assert!((out[3] - 1.0).abs() < EPS, "over-submersion clamps to one");
        assert!(
            out[COUPLING_FORCE_FLOATS + 3].abs() < EPS,
            "zero total is inert"
        );
    }

    #[test]
    fn short_buffers_do_not_panic() {
        // Claim three bodies but only supply one-and-a-half records.
        let bodies = [0.5_f32, 1.0, 2.0, 1.0, 0.3, 1.0];
        let out = dispatch_coupling_readback(&bodies, PARAMS, 3);
        assert_eq!(out.len(), 3 * COUPLING_FORCE_FLOATS);
        // The first body is populated; the under-supplied tail stays zero.
        assert!(out[0] > 0.0, "first body computed");
        assert_eq!(out[COUPLING_FORCE_FLOATS].to_bits(), 0.0_f32.to_bits());
        assert_eq!(out[2 * COUPLING_FORCE_FLOATS].to_bits(), 0.0_f32.to_bits());
    }

    #[test]
    fn wesl_kernel_declares_expected_abi() {
        assert!(WATER_COUPLING_READBACK_WESL.contains("fn water_coupling_readback"));
        assert!(WATER_COUPLING_READBACK_WESL.contains("@workgroup_size(64, 1, 1)"));
        assert!(WATER_COUPLING_READBACK_WESL.contains("struct CouplingReadbackParams"));
        assert!(WATER_COUPLING_READBACK_WESL.contains("var<storage, read_write> force_out"));
    }

    #[test]
    fn strides_are_consistent() {
        assert_eq!(COUPLING_BODY_FLOATS, 4);
        assert_eq!(COUPLING_FORCE_FLOATS, 4);
    }
}
