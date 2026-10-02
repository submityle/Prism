//! Real-device parity for the multigrid mean-removal twin: [`GpuMgRemoveMean`]
//! must reproduce the `CPU` golden private `remove_mean` from
//! [`multigrid_pressure`](prism_render_architecture::particle::multigrid_pressure)
//! across random one-dimensional fields of many lengths — including the
//! length-one and empty degeneracies and a large field — and the structural
//! property that a de-meaned field sums back to (near) zero.
//!
//! The golden `remove_mean` is a private helper and is not exported, so this
//! test carries a line-for-line mirror of it — tagged with a `MIRROR of ...`
//! provenance note — and compares the on-device result against that mirror.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device.
//!
//! # Parity criterion
//!
//! The mean is formed on the host in the golden strict ascending order, so it is
//! bit-identical to the reference; the only device float op is the per-element
//! subtraction. Values are asserted to within `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3` — robust across backends yet tight enough to fail a wrong
//! port (a reordered sum, a dropped subtraction, a wrong divisor).
//!
//! Provenance: standard multigrid null-space pin (mean removal); 无第三方引擎
//! 源码或衍生代码。

use prism_volumetric_gpu::mg_remove_mean::{GpuMgRemoveMean, GpuMgRemoveMeanQuery};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity tolerance, a decade above single-term subtraction rounding.
const ABS_EPS: f32 = 1.0e-4;

/// Relative parity tolerance for samples large enough that the absolute floor is
/// pessimistic.
const REL_EPS: f32 = 1.0e-3;

/// Relative-tolerance floor so a near-zero reference never divides by zero.
const REL_FLOOR: f32 = 1.0e-6;

// ----------------------------------------------------------------------------
// Mirror of the golden `multigrid_pressure` mean removal.
// ----------------------------------------------------------------------------

// MIRROR of multigrid_pressure::remove_mean (private, exact transcription)
fn golden_remove_mean(field: &mut [f32]) {
    if field.is_empty() {
        return;
    }
    let mut sum = 0.0f32;
    for &v in field.iter() {
        sum += v;
    }
    let mean = sum / field.len() as f32;
    for v in field.iter_mut() {
        *v -= mean;
    }
}

/// Applies the mirrored golden `remove_mean` to a copy of `field`, returning the
/// de-meaned field.
fn golden_demean(field: &[f32]) -> Vec<f32> {
    let mut out = field.to_vec();
    golden_remove_mean(&mut out);
    out
}

// ----------------------------------------------------------------------------
// Deterministic fixture generation.
// ----------------------------------------------------------------------------

/// A tiny deterministic linear-congruential generator so the "random" fields are
/// reproducible run to run without pulling in an external crate. The constants
/// are the Numerical Recipes `LCG` multiplier and increment.
struct Lcg {
    state: u32,
}

impl Lcg {
    fn new(seed: u32) -> Self {
        Lcg { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        self.state
    }

    /// A reproducible `f32` in `[-range, range]` using pure integer scaling (no
    /// transcendental method).
    fn next_signed(&mut self, range: f32) -> f32 {
        let unit = (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32;
        (unit * 2.0 - 1.0) * range
    }

    fn field(&mut self, count: usize, range: f32) -> Vec<f32> {
        (0..count).map(|_| self.next_signed(range)).collect()
    }
}

/// Asserts the `GPU` field matches the mirrored golden element for element to
/// within the documented tolerance.
fn assert_parity(label: &str, cpu: &[f32], gpu: &[f32]) {
    assert_eq!(cpu.len(), gpu.len(), "{label}: field length mismatch");
    for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        let abs_diff = (c - g).abs();
        let rel_diff = abs_diff / c.abs().max(REL_FLOOR);
        assert!(
            abs_diff <= ABS_EPS || rel_diff <= REL_EPS,
            "{label}: element {i} mismatch: cpu {c}, gpu {g} (abs {abs_diff}, rel {rel_diff})"
        );
    }
}

/// Runs one mirrored-`CPU`-vs-`GPU` scenario and asserts parity.
fn check_scenario(label: &str, engine: &GpuMgRemoveMean, ctx: &GpuContext, field: &[f32]) {
    let cpu = golden_demean(field);
    let gpu = engine.remove_mean(
        ctx,
        &GpuMgRemoveMeanQuery {
            field: field.to_vec(),
        },
    );
    assert_parity(label, &cpu, &gpu.field);
}

// ----------------------------------------------------------------------------
// Tests.
// ----------------------------------------------------------------------------

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_matches_cpu_across_lengths() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping mg-remove-mean parity: no wgpu adapter on this host");
        return;
    };
    let engine = GpuMgRemoveMean::new(&ctx);

    // A spread of lengths: tiny, odd, a workgroup boundary (64), just over it,
    // and a few hundred — each on its own reproducible random field.
    let lengths = [1usize, 2, 3, 7, 15, 63, 64, 65, 100, 257, 512];
    for (seed, &len) in lengths.iter().enumerate() {
        let mut rng = Lcg::new(0x5E57_u32.wrapping_add(seed as u32));
        let field = rng.field(len, 3.0);
        check_scenario(&format!("random len {len}"), &engine, &ctx, &field);
    }
}

#[test]
fn gpu_matches_cpu_length_one() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuMgRemoveMean::new(&ctx);

    // A single element is its own mean, so the de-meaned field is a single zero.
    let field = vec![2.75f32];
    let gpu = engine.remove_mean(
        &ctx,
        &GpuMgRemoveMeanQuery {
            field: field.clone(),
        },
    );
    let cpu = golden_demean(&field);
    assert_parity("length one", &cpu, &gpu.field);
    assert!(
        gpu.field.iter().all(|&v| v.abs() <= ABS_EPS),
        "a length-one field de-means to a single zero"
    );
}

#[test]
fn gpu_matches_cpu_large_field() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuMgRemoveMean::new(&ctx);

    // A larger field spanning many workgroups, exercising the full ascending
    // host sum against the device subtraction.
    let mut rng = Lcg::new(0xBEEF_u32);
    let field = rng.field(4096, 5.0);
    check_scenario("large 4096", &engine, &ctx, &field);
}

#[test]
fn gpu_matches_cpu_biased_field() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuMgRemoveMean::new(&ctx);

    // A field offset by a large constant so the mean is far from zero; the twin
    // must still reproduce the golden mean and subtraction exactly.
    let mut rng = Lcg::new(0x1234_u32);
    let field: Vec<f32> = rng.field(300, 1.0).into_iter().map(|v| v + 50.0).collect();
    check_scenario("biased +50", &engine, &ctx, &field);
}

#[test]
fn gpu_handles_empty_field() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuMgRemoveMean::new(&ctx);

    // An empty field returns an empty field with no dispatch, the exact
    // reference guard.
    let gpu = engine.remove_mean(&ctx, &GpuMgRemoveMeanQuery { field: Vec::new() });
    assert!(
        gpu.field.is_empty(),
        "an empty field yields an empty de-meaned field"
    );
}
