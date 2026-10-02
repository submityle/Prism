//! Real-device parity for the signed `Q24.8` multiply twin:
//! [`GpuFixedPointQ24_8Muldiv`](prism_volumetric_gpu::fixed_point_q24_8_muldiv::GpuFixedPointQ24_8Muldiv)
//! must reproduce the `CPU` golden
//! [`mul`](prism_render_architecture::particle::fixed_point_q24_8::mul) over the
//! whole [`i32`] operand range.
//!
//! The golden multiply widens to [`i64`], forms the full `64`-bit product,
//! arithmetic-right-shifts by `8` and wrap-narrows to [`i32`] with no round bias
//! and no saturation. The twin emulates that `64`-bit path with a `(hi, lo)`
//! `u32` two's-complement pair, so the fixtures must exercise the whole span —
//! not a convenient small sub-range — to prove the emulation is faithful.
//!
//! # Fixtures
//!
//! The deterministic integer-`LCG` sweep draws operand pairs across the full
//! [`i32`] range (both signs, all magnitudes), so products routinely overflow
//! `32` bits and exercise the wrapping low-`32`-bit narrowing. Dedicated
//! fixtures pin the structural corners: `0`, `±256` (the `Q24.8` `1.0`), `128`
//! (`0.5`), small negatives, every sign combination (to exercise the
//! two's-complement negation), magnitudes whose product exceeds [`i32`], and the
//! extremes [`i32::MIN`] and [`i32::MAX`] (where the magnitude of `i32::MIN` must
//! round-trip as `0x8000_0000`). The `LCG` lives on the host in `u64`; the
//! kernel stays pure `u32`, and the fixtures use no external math library and no
//! transcendental method.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable
//! core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Every step is exact integer bit algebra with no rounding and no floating
//! point, so `CPU` and `GPU` must agree bit for bit. The comparison is an exact
//! `==` on every [`i32`] output with no tolerance: any mismatch is a genuine
//! port bug (a wrong limb, a dropped carry, a mis-sized shift, a missing sign
//! negation).
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::fixed_point_q24_8`
//! 的 `mul` 真机 parity；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::fixed_point_q24_8::{mul, Q24_8};
use prism_volumetric_gpu::fixed_point_q24_8_muldiv::{
    FixedPointQ24_8MulQuery, GpuFixedPointQ24_8Muldiv,
};
use prism_volumetric_gpu::GpuContext;

/// Small linear-congruential generator for deterministic random samples. The
/// `u64` state lives on the host; the kernel stays pure `u32`. The constants are
/// the Numerical Recipes multiplier and increment.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Advances the generator and returns the next raw word.
    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.state >> 32) as u32
    }

    /// A reproducible [`i32`] spanning the full signed range (both signs, all
    /// magnitudes), including the extremes.
    fn next_i32(&mut self) -> i32 {
        self.next_u32() as i32
    }
}

/// The golden `raw` result for one operand pair, calling the reference `mul`
/// directly so the twin is pinned lane for lane.
fn cpu_reference(a: i32, b: i32) -> i32 {
    mul(Q24_8 { raw: a }, Q24_8 { raw: b }).raw
}

/// Runs one batch on the device and asserts every lane matches the golden `mul`
/// bit for bit.
fn check_batch(
    label: &str,
    engine: &GpuFixedPointQ24_8Muldiv,
    ctx: &GpuContext,
    queries: &[FixedPointQ24_8MulQuery],
) {
    let gpu = engine.run(ctx, queries);
    assert_eq!(gpu.len(), queries.len(), "{label}: result count");
    for (i, q) in queries.iter().enumerate() {
        let cpu = cpu_reference(q.a, q.b);
        assert_eq!(
            gpu[i], cpu,
            "{label}: lane {i} mismatch for a={}, b={} (cpu {cpu}, gpu {})",
            q.a, q.b, gpu[i]
        );
    }
}

/// The `Q24.8` `raw` of `1.0` (`256`), i.e. [`ONE`] in the golden module.
const ONE_RAW: i32 = 256;

/// The `Q24.8` `raw` of `0.5` (`128`), i.e. [`HALF`] in the golden module.
const HALF_RAW: i32 = 128;

/// Builds a query from two raw operands.
fn q(a: i32, b: i32) -> FixedPointQ24_8MulQuery {
    FixedPointQ24_8MulQuery { a, b }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_matches_cpu_across_full_range_random() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping fixed-point Q24.8 multiply parity: no wgpu adapter on this host");
        return;
    };
    let engine = GpuFixedPointQ24_8Muldiv::new(&ctx);
    for (seed, count) in [(1usize, 97usize), (2, 256), (3, 513), (4, 1024)] {
        let mut rng = Lcg::new(0x9E37_79B9_7F4A_7C15u64.wrapping_add(seed as u64));
        let queries: Vec<FixedPointQ24_8MulQuery> = (0..count)
            .map(|_| q(rng.next_i32(), rng.next_i32()))
            .collect();
        check_batch(
            &format!("random full-range batch {seed} (count={count})"),
            &engine,
            &ctx,
            &queries,
        );
    }
}

#[test]
fn gpu_matches_cpu_on_structural_corners() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuFixedPointQ24_8Muldiv::new(&ctx);
    let queries = vec![
        // Zero on either side.
        q(0, 0),
        q(0, ONE_RAW),
        q(ONE_RAW, 0),
        q(0, i32::MIN),
        q(i32::MAX, 0),
        // Identity and half scaling, both signs.
        q(ONE_RAW, 12_345),
        q(-ONE_RAW, 12_345),
        q(ONE_RAW, -12_345),
        q(HALF_RAW, 1_000),
        q(HALF_RAW, -1_000),
        q(HALF_RAW, HALF_RAW),
        // Small integers in Q24.8 and the documented golden example (-3 * 4).
        q(-3 * ONE_RAW, 4 * ONE_RAW),
        q(3 * ONE_RAW, -4 * ONE_RAW),
        q(-3 * ONE_RAW, -4 * ONE_RAW),
        // Every sign combination with the same magnitudes.
        q(123_456, 654_321),
        q(-123_456, 654_321),
        q(123_456, -654_321),
        q(-123_456, -654_321),
    ];
    check_batch("structural corners", &engine, &ctx, &queries);
}

#[test]
fn gpu_matches_cpu_on_overflowing_products() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuFixedPointQ24_8Muldiv::new(&ctx);
    // Magnitudes large enough that the raw i64 product exceeds i32, so the
    // low-32-bit wrapping narrowing after the >> 8 is exercised in every sign
    // combination.
    let big = vec![
        1_000_000i32,
        16_777_216,
        100_000_000,
        2_000_000_000,
        i32::MAX,
        i32::MIN,
        i32::MIN + 1,
        -2_000_000_000,
        715_827_883,
        1_518_500_250,
    ];
    let mut queries = Vec::new();
    for &a in &big {
        for &b in &big {
            queries.push(q(a, b));
        }
    }
    check_batch("overflowing products", &engine, &ctx, &queries);
}

#[test]
fn gpu_matches_cpu_on_extreme_pairs() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuFixedPointQ24_8Muldiv::new(&ctx);
    // The i32 extremes against each other and against the unit scales: i32::MIN
    // must round-trip through the magnitude 0x8000_0000 exactly.
    let queries = vec![
        q(i32::MIN, i32::MIN),
        q(i32::MIN, i32::MAX),
        q(i32::MAX, i32::MIN),
        q(i32::MAX, i32::MAX),
        q(i32::MIN, ONE_RAW),
        q(i32::MIN, -ONE_RAW),
        q(i32::MAX, -ONE_RAW),
        q(i32::MIN, 1),
        q(i32::MAX, 1),
        q(i32::MIN, -1),
        q(i32::MAX, -1),
        q(-1, -1),
        q(1, -1),
    ];
    check_batch("extreme pairs", &engine, &ctx, &queries);
}
