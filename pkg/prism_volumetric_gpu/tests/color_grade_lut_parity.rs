//! Real-device parity for the 3D colour-grading `LUT` twin:
//! [`GpuColorGradeLut`](prism_volumetric_gpu::color_grade_lut::GpuColorGradeLut)
//! must reproduce the `CPU` golden
//! [`color_grade_lut`](prism_render_architecture::particle::color_grade_lut)
//! across both the trilinear 8-corner filter and the tetrahedral 6-cell filter.
//!
//! The fixtures cover identity cubes of several sizes (`2`, `3`, `5`), randomly
//! baked cubes whose texels carry arbitrary graded colours, and inputs crafted
//! to drive the tetrahedral selector through all six tetrahedra. Every random
//! input is reject-sampled so each channel's in-cell fraction lands in
//! `[0.15, 0.85]` — clear of the [`floor`](f32::floor) lattice ties and the
//! `0`/`1` clamp endpoints — and the three fractions stay mutually separated, so
//! the `CPU` and `GPU` pick the identical lattice cell and the identical
//! tetrahedron. The random data is drawn from a host-side `u64` `LCG`, so the
//! fixtures stay pure and need no `bevy_math` and no transcendental math.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Both filters thread the fractions through multiplies and adds, so `CPU` and
//! `GPU` are not bit-exact; every graded channel is compared under tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`).
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::color_grade_lut`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::color_grade_lut::{ColorGradeLut, Rgb};
use prism_volumetric_gpu::color_grade_lut::{ColorGradeLutQuery, GpuColorGradeLut};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous quantities.
const ABS_EPS: f32 = 1.0e-4;
/// Relative tolerance for the continuous quantities.
const REL_EPS: f32 = 1.0e-3;
/// Floor for the relative-tolerance denominator.
const REL_FLOOR: f32 = 1.0e-6;
/// Minimum in-cell fraction kept by the reject sampler (clear of a tie / clamp).
const FRAC_LO: f32 = 0.15;
/// Maximum in-cell fraction kept by the reject sampler.
const FRAC_HI: f32 = 0.85;
/// Minimum separation between any two in-cell fractions (clear of a tetrahedral
/// branch tie).
const FRAC_SEP: f32 = 0.08;

/// Mixed absolute / relative tolerance comparison for one `f32` lane.
fn approx(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= ABS_EPS {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= REL_EPS
}

/// Tolerant comparison of two `RGB` triples.
fn approx3(a: [f32; 3], b: [f32; 3]) -> bool {
    approx(a[0], b[0]) && approx(a[1], b[1]) && approx(a[2], b[2])
}

/// A tiny host-side `u64` linear congruential generator (`LCG`), used only for
/// pure integer arithmetic so the fixtures invoke no transcendental math.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Advances the state and returns the next `u64`.
    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    /// Returns the next `f32` in `[0, 1)` using the top `24` state bits.
    fn next_unit(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32;
        (bits as f32) / ((1u32 << 24) as f32)
    }
}

/// The in-cell fraction of `channel` for a cube of `size` lattice points.
fn frac_of(channel: f32, size: usize) -> f32 {
    let last = (size - 1) as f32;
    let scaled = channel.clamp(0.0, 1.0) * last;
    scaled - scaled.floor()
}

/// Whether every channel's in-cell fraction lands clear of a lattice tie / clamp
/// endpoint and the three fractions stay mutually separated.
fn well_separated(input: [f32; 3], size: usize) -> bool {
    let fr = frac_of(input[0], size);
    let fg = frac_of(input[1], size);
    let fb = frac_of(input[2], size);
    let in_band = |f: f32| (FRAC_LO..=FRAC_HI).contains(&f);
    if !(in_band(fr) && in_band(fg) && in_band(fb)) {
        return false;
    }
    (fr - fg).abs() >= FRAC_SEP && (fg - fb).abs() >= FRAC_SEP && (fr - fb).abs() >= FRAC_SEP
}

/// Draws `n` reject-sampled inputs whose per-channel fractions are well clear of
/// the ties and endpoints for a cube of `size` lattice points.
fn sampled_inputs(rng: &mut Lcg, size: usize, n: usize) -> Vec<[f32; 3]> {
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let candidate = [rng.next_unit(), rng.next_unit(), rng.next_unit()];
        if well_separated(candidate, size) {
            out.push(candidate);
        }
    }
    out
}

/// Packs a cube into the flat `red`-fastest `f32` layout the twin expects, in
/// the golden texel order.
fn pack(lut: &ColorGradeLut) -> Vec<f32> {
    lut.texels()
        .iter()
        .flat_map(|texel| texel.to_array())
        .collect()
}

/// Bakes a randomly graded cube of the requested `size`.
fn random_cube(rng: &mut Lcg, size: usize) -> ColorGradeLut {
    let count = size * size * size;
    let mut texels = Vec::with_capacity(count);
    for _ in 0..count {
        texels.push(Rgb::new(rng.next_unit(), rng.next_unit(), rng.next_unit()));
    }
    ColorGradeLut::from_texels(size, texels).expect("texel count matches size^3")
}

/// Asserts the twin reproduces both golden filters for every input against the
/// supplied cube.
fn assert_parity(
    gpu: &GpuColorGradeLut,
    ctx: &GpuContext,
    lut: &ColorGradeLut,
    inputs: &[[f32; 3]],
) {
    let data = pack(lut);
    let queries: Vec<ColorGradeLutQuery> = inputs
        .iter()
        .map(|&input| ColorGradeLutQuery { input })
        .collect();
    let got = gpu.evaluate(ctx, &data, lut.size(), &queries);
    assert_eq!(got.len(), inputs.len(), "one result per input");
    for (i, &input) in inputs.iter().enumerate() {
        let key = Rgb::new(input[0], input[1], input[2]);
        let cpu_tri = lut.sample_trilinear(key).to_array();
        let cpu_tet = lut.sample_tetrahedral(key).to_array();
        assert!(
            approx3(got[i].trilinear, cpu_tri),
            "trilinear mismatch at {input:?}: gpu {:?} vs cpu {cpu_tri:?}",
            got[i].trilinear
        );
        assert!(
            approx3(got[i].tetrahedral, cpu_tet),
            "tetrahedral mismatch at {input:?}: gpu {:?} vs cpu {cpu_tet:?}",
            got[i].tetrahedral
        );
    }
}

/// Builds an input that sits in the `(1,1,1)` cell of a `size`-`5` cube with the
/// requested per-channel fractions, keeping each channel clear of the clamp
/// endpoints.
fn cell_input(fr: f32, fg: f32, fb: f32) -> [f32; 3] {
    // last == 4 for a size-5 cube; channel = (base + frac) / last with base = 1.
    let last = 4.0;
    [(1.0 + fr) / last, (1.0 + fg) / last, (1.0 + fb) / last]
}

#[test]
fn identity_cubes_reproduce_input() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuColorGradeLut::new(&ctx);
    let mut rng = Lcg::new(0x1234_5678_9abc_def0);
    for &size in &[2usize, 3, 5] {
        let lut = ColorGradeLut::identity(size);
        let inputs = sampled_inputs(&mut rng, size, 16);
        assert_parity(&gpu, &ctx, &lut, &inputs);
    }
}

#[test]
fn random_cubes_match_both_filters() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuColorGradeLut::new(&ctx);
    let mut rng = Lcg::new(0x0bad_f00d_dead_beef);
    for &size in &[3usize, 5] {
        let lut = random_cube(&mut rng, size);
        let inputs = sampled_inputs(&mut rng, size, 24);
        assert_parity(&gpu, &ctx, &lut, &inputs);
    }
}

#[test]
fn tetrahedral_branches_all_covered() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuColorGradeLut::new(&ctx);
    let mut rng = Lcg::new(0x00c0_ffee_1337_4242);
    let lut = random_cube(&mut rng, 5);
    // Three distinct fractions with 0.25 gaps, permuted to visit each of the six
    // tetrahedra; all land in [0.15, 0.85] and stay far from the branch ties.
    let (hi, mid, lo) = (0.75, 0.5, 0.25);
    let inputs = [
        cell_input(hi, mid, lo), // fr > fg > fb
        cell_input(hi, lo, mid), // fr > fb > fg
        cell_input(mid, lo, hi), // fb > fr > fg
        cell_input(lo, mid, hi), // fb > fg > fr
        cell_input(lo, hi, mid), // fg > fb > fr
        cell_input(mid, hi, lo), // fg > fr > fb
    ];
    assert_parity(&gpu, &ctx, &lut, &inputs);
}

#[test]
fn degenerate_single_texel_cube() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuColorGradeLut::new(&ctx);
    // A size-1 cube has a single lattice point returned for every input.
    let lut = ColorGradeLut::identity(1);
    let inputs = [[0.3, 0.7, 0.1], [0.5, 0.5, 0.5], [0.9, 0.2, 0.4]];
    assert_parity(&gpu, &ctx, &lut, &inputs);
}

#[test]
fn empty_cube_returns_black_guard() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuColorGradeLut::new(&ctx);
    // The empty cube is short-circuited on the host to the reference black guard.
    let queries = [
        ColorGradeLutQuery {
            input: [0.5, 0.5, 0.5],
        },
        ColorGradeLutQuery {
            input: [0.2, 0.8, 0.4],
        },
    ];
    let got = gpu.evaluate(&ctx, &[], 0, &queries);
    assert_eq!(got.len(), queries.len());
    for r in &got {
        assert!(approx3(r.trilinear, [0.0, 0.0, 0.0]));
        assert!(approx3(r.tetrahedral, [0.0, 0.0, 0.0]));
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuColorGradeLut::new(&ctx);
    let lut = ColorGradeLut::identity(3);
    let data = pack(&lut);
    // No dispatch is issued and the result vector is empty.
    assert!(gpu.evaluate(&ctx, &data, lut.size(), &[]).is_empty());
}
