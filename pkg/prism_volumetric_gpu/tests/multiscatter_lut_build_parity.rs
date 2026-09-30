//! Real-device parity for the multi-scatter energy-gain `LUT` bake twin:
//! [`GpuMultiScatterLutBuild`] must reproduce the `CPU` golden
//! [`MultiScatterLut::build_energy_gain`](prism_render_architecture::volumetric::multiscatter::MultiScatterLut::build_energy_gain)
//! for every cell, across several table shapes (including degenerate
//! single-cell axes) and several `octave` schedules (including the
//! zero-octave fallback).
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL
//! plus a base-two `exp` polynomial, so it needs no optional device feature.
//!
//! # Reading back the golden cells
//!
//! The `CPU` [`MultiScatterLut`] keeps its `data` private, exposing only
//! `dims` and a `trilinear` `sample`. At an axis coordinate that lands exactly
//! on a grid node the interpolation fraction is zero, so `sample` returns that
//! node's stored cell. The test therefore reconstructs each cell coordinate
//! with the same `axis_value` lerp the bake uses and reads the golden cell via
//! `lut.sample(cos, depth, albedo)`, comparing it to the `GPU` flat cell at the
//! matching row-major index (`cos` outer, `depth` middle, `albedo` inner).
//!
//! # Parity criterion
//!
//! The gain combines a `sqrt` diffusion reflectance, an `exp`-based
//! optical-depth ramp, and a Henyey-Greenstein octave phase weight. `CPU` and
//! `GPU` share the same base-two `exp` polynomial and the same algebra, so they
//! agree to a small absolute tolerance (`< 1e-4`); every value must also stay
//! in `[0, 1]` (energy conserving).
//!
//! Provenance: `Frostbite`-style pre-integrated multiple-scattering energy gain
//! plus Wrenninge-style `octave` decay and standard Henyey-Greenstein phase; no
//! Unreal Engine source or derived code.

use prism_render_architecture::volumetric::multiscatter::MultiScatterLut;
use prism_render_architecture::volumetric::scatter::OctaveParams;
use prism_volumetric_gpu::{GpuContext, GpuMultiScatterLutBuild};

/// Fixed axis ranges baked into `MultiScatterLut::new`. The upper `depth` bound
/// is the crate-private `DEFAULT_MAX_OPTICAL_DEPTH`, mirrored here.
const MINS: [f32; 3] = [-1.0, 0.0, 0.0];
const MAXS: [f32; 3] = [1.0, 8.0, 1.0];

/// Sample value of axis coordinate `i` on a `dim`-cell axis spanning
/// `[min, max]`, mirroring the crate-private `axis_value`.
fn axis_value(min: f32, max: f32, dim: usize, i: usize) -> f32 {
    if dim <= 1 {
        min
    } else {
        min + (max - min) * (i as f32 / (dim - 1) as f32)
    }
}

/// Asserts the `GPU` baked table matches the `CPU` golden cell for cell, read
/// back via `sample` at exact grid nodes, and that every value is in `[0, 1]`.
fn assert_parity(dims: [usize; 3], params: OctaveParams, gpu: &[f32]) {
    let expected_len = dims[0] * dims[1] * dims[2];
    assert_eq!(gpu.len(), expected_len, "one cell per grid point");

    let lut = MultiScatterLut::build_energy_gain(dims, params);
    assert_eq!(lut.dims(), dims, "table dims should be preserved");

    for ic in 0..dims[0] {
        let cos = axis_value(MINS[0], MAXS[0], dims[0], ic);
        for id in 0..dims[1] {
            let depth = axis_value(MINS[1], MAXS[1], dims[1], id);
            for ia in 0..dims[2] {
                let albedo = axis_value(MINS[2], MAXS[2], dims[2], ia);
                let flat_idx = (ic * dims[1] + id) * dims[2] + ia;
                let cpu = lut.sample(cos, depth, albedo);
                let got = gpu[flat_idx];
                assert!(
                    (got - cpu).abs() < 1e-4,
                    "energy-gain bake mismatch at cell {flat_idx} \
                     (cos={cos}, depth={depth}, albedo={albedo}): gpu {got}, cpu {cpu}"
                );
                assert!(
                    (0.0..=1.0).contains(&got),
                    "gpu gain must stay in [0, 1]: {got}"
                );
            }
        }
    }
}

#[test]
#[expect(clippy::print_stderr, reason = "surface a skip notice on headless CI")]
fn multiscatter_lut_build_matches_cpu_default_schedule() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping multiscatter_lut_build parity: no wgpu adapter");
        return;
    };
    let kernel = GpuMultiScatterLutBuild::new(&ctx);
    let params = OctaveParams::DEFAULT;

    // A cubic table and two anisotropic shapes exercise the flat-index
    // decomposition on non-square strides.
    for dims in [[8usize, 8, 8], [4, 6, 5], [2, 2, 2]] {
        let udims = [dims[0] as u32, dims[1] as u32, dims[2] as u32];
        let gpu = kernel.eval(&ctx, udims, params);
        assert_parity(dims, params, &gpu);
    }
}

#[test]
#[expect(clippy::print_stderr, reason = "surface a skip notice on headless CI")]
fn multiscatter_lut_build_matches_cpu_degenerate_axes() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping multiscatter_lut_build parity: no wgpu adapter");
        return;
    };
    let kernel = GpuMultiScatterLutBuild::new(&ctx);
    let params = OctaveParams::DEFAULT;

    // Single-cell axes must collapse to the axis minimum on both sides.
    for dims in [[1usize, 4, 4], [4, 1, 4], [4, 4, 1], [1, 1, 1]] {
        let udims = [dims[0] as u32, dims[1] as u32, dims[2] as u32];
        let gpu = kernel.eval(&ctx, udims, params);
        assert_parity(dims, params, &gpu);
    }
}

#[test]
#[expect(clippy::print_stderr, reason = "surface a skip notice on headless CI")]
fn multiscatter_lut_build_matches_cpu_custom_and_zero_octaves() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping multiscatter_lut_build parity: no wgpu adapter");
        return;
    };
    let kernel = GpuMultiScatterLutBuild::new(&ctx);
    let dims = [5usize, 7, 6];
    let udims = [dims[0] as u32, dims[1] as u32, dims[2] as u32];

    // A custom decay schedule and the zero-octave fallback (phase weight -> 1).
    let custom = OctaveParams {
        attenuation: 0.4,
        contribution: 0.65,
        eccentricity_attenuation: 0.3,
        octave_count: 6,
    };
    let zero = OctaveParams {
        attenuation: 0.5,
        contribution: 0.5,
        eccentricity_attenuation: 0.5,
        octave_count: 0,
    };
    for params in [custom, zero] {
        let gpu = kernel.eval(&ctx, udims, params);
        assert_parity(dims, params, &gpu);
    }
}
