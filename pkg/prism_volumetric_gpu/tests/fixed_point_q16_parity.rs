//! Real-device parity for the signed `Q16.16` integer twin:
//! [`GpuFixedPointQ16`] must reproduce the `CPU` golden
//! [`fixed_point_q16`](prism_render_architecture::particle::fixed_point_q16)
//! pure-[`i32`] functions bit for bit across random batches, the saturation
//! edges, the floor/frac fractional cases, the min/max/clamp selectors, a single
//! query, and the degenerate empty batch.
//!
//! The tests skip (with a printed notice on the first) when the host has no
//! `wgpu` adapter, so the suite stays green everywhere while still exercising
//! the full dispatch-and-readback on any real device. The kernel is portable
//! core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Every twinned operation is exact integer arithmetic — two's-complement
//! wrapping add/subtract/negate, saturating add/subtract/absolute, min/max,
//! clamp and bit masking — so the `CPU` and `GPU` agree bit for bit. The parity
//! test therefore asserts a strict `==` on every output word with no tolerance:
//! any mismatch is a genuine port defect.
//!
//! # Oracle
//!
//! The expected value for each query is the golden `pub` function evaluated on
//! the same operands, so the twin is pinned directly against the reference
//! rather than a re-transcribed copy. The golden operates on the `Q16_16`
//! wrapper, so each raw operand is wrapped before the call and the result's
//! backing integer is read back out.
//!
//! # Fixtures
//!
//! The deterministic `u64` `LCG` fixtures draw operands from several regimes —
//! zero, small signed values, sub-unit fractions, mid-range, full-range and
//! values a few units inside `i32::MIN`/`i32::MAX` — so the saturating
//! operations exercise both their overflow and non-overflow paths and the
//! absolute operation hits its `i32::MIN` saturation. The fixtures use pure
//! integer arithmetic with no external math library and no transcendental
//! method. Integer parity has no boundary fuzz, so no reject-sampling margin is
//! needed; the clamp fixture only orders its bounds so `lo <= hi` as the
//! reference contract requires.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::fixed_point_q16`
//! 的纯 `i32` 子集真机 parity；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::fixed_point_q16::{
    q_abs, q_add, q_add_sat, q_clamp, q_floor, q_frac, q_from_int, q_max, q_min, q_neg, q_sub,
    q_sub_sat, q_to_int_floor, q_to_int_trunc, Q16_16,
};
use prism_volumetric_gpu::fixed_point_q16::{GpuFixedPointQ16, GpuQ16Op, GpuQ16Query};
use prism_volumetric_gpu::GpuContext;

/// Every operation the twin supports, in a fixed order so a random batch cycles
/// through all `14` with each dispatch.
const OP_TABLE: [GpuQ16Op; 14] = [
    GpuQ16Op::FromInt,
    GpuQ16Op::ToIntTrunc,
    GpuQ16Op::ToIntFloor,
    GpuQ16Op::Add,
    GpuQ16Op::Sub,
    GpuQ16Op::Neg,
    GpuQ16Op::AddSat,
    GpuQ16Op::SubSat,
    GpuQ16Op::Abs,
    GpuQ16Op::Min,
    GpuQ16Op::Max,
    GpuQ16Op::Clamp,
    GpuQ16Op::Floor,
    GpuQ16Op::Frac,
];

/// A tiny deterministic `64`-bit linear-congruential generator so the "random"
/// fixtures are reproducible run to run without pulling in an external crate.
/// The constants are the common `PCG`/`MMIX` `LCG` multiplier and increment.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg { state: seed }
    }

    /// Advances the generator and returns the next raw word.
    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    /// A reproducible full-range [`i32`] drawn from the high bits.
    fn next_i32(&mut self) -> i32 {
        (self.next_u64() >> 32) as u32 as i32
    }

    /// A reproducible operand drawn from one of several regimes so the batch
    /// covers zero, small signed values, sub-unit fractions, mid-range,
    /// full-range and values a few units inside `i32::MIN`/`i32::MAX`. The near
    /// extremes drive both the overflow and non-overflow paths of the saturating
    /// operations.
    fn next_value(&mut self) -> i32 {
        let regime = self.next_u64() % 8;
        let jitter = (self.next_u64() % 5) as i32;
        match regime {
            0 => 0,
            1 => (self.next_u64() % 2_001) as i32 - 1_000,
            2 => i32::MAX - jitter,
            3 => i32::MIN + jitter,
            4 => self.next_i32() / 2,
            5 => (self.next_u64() % 262_145) as i32 - 131_072,
            6 => self.next_i32(),
            _ => ((self.next_u64() % 21) as i32 - 10) * Q16_16::ONE,
        }
    }
}

/// The golden expected value for one query: the `pub` reference function
/// evaluated on the same operands, wrapping each raw operand into `Q16_16` and
/// reading the backing integer back out.
fn cpu_expected(q: &GpuQ16Query) -> i32 {
    match q.op {
        GpuQ16Op::FromInt => q_from_int(q.a).0,
        GpuQ16Op::ToIntTrunc => q_to_int_trunc(Q16_16(q.a)),
        GpuQ16Op::ToIntFloor => q_to_int_floor(Q16_16(q.a)),
        GpuQ16Op::Add => q_add(Q16_16(q.a), Q16_16(q.b)).0,
        GpuQ16Op::Sub => q_sub(Q16_16(q.a), Q16_16(q.b)).0,
        GpuQ16Op::Neg => q_neg(Q16_16(q.a)).0,
        GpuQ16Op::AddSat => q_add_sat(Q16_16(q.a), Q16_16(q.b)).0,
        GpuQ16Op::SubSat => q_sub_sat(Q16_16(q.a), Q16_16(q.b)).0,
        GpuQ16Op::Abs => q_abs(Q16_16(q.a)).0,
        GpuQ16Op::Min => q_min(Q16_16(q.a), Q16_16(q.b)).0,
        GpuQ16Op::Max => q_max(Q16_16(q.a), Q16_16(q.b)).0,
        GpuQ16Op::Clamp => q_clamp(Q16_16(q.a), Q16_16(q.b), Q16_16(q.c)).0,
        GpuQ16Op::Floor => q_floor(Q16_16(q.a)).0,
        GpuQ16Op::Frac => q_frac(Q16_16(q.a)).0,
    }
}

/// Runs one batch on the device and asserts every lane matches the golden
/// function exactly.
fn check_batch(label: &str, engine: &GpuFixedPointQ16, ctx: &GpuContext, queries: &[GpuQ16Query]) {
    let gpu = engine.run(ctx, queries);
    assert_eq!(gpu.len(), queries.len(), "{label}: result count");
    for (i, q) in queries.iter().enumerate() {
        let cpu = cpu_expected(q);
        assert_eq!(
            gpu[i],
            cpu,
            "{label}[{i}] op {op:?} a={a} b={b} c={c}: cpu {cpu}, gpu {g}",
            op = q.op,
            a = q.a,
            b = q.b,
            c = q.c,
            g = gpu[i],
        );
    }
}

/// Builds a reproducible batch of `count` queries, cycling through all `14`
/// operations and drawing each operand from the mixed-regime generator. The
/// clamp lane orders its bounds so `lo <= hi` as the reference contract requires.
fn random_batch(rng: &mut Lcg, count: usize) -> Vec<GpuQ16Query> {
    let mut queries = Vec::with_capacity(count);
    for i in 0..count {
        let op = OP_TABLE[i % OP_TABLE.len()];
        let a = rng.next_value();
        let mut b = rng.next_value();
        let mut c = rng.next_value();
        if matches!(op, GpuQ16Op::Clamp) && b > c {
            core::mem::swap(&mut b, &mut c);
        }
        queries.push(GpuQ16Query { a, b, c, op });
    }
    queries
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_matches_cpu_across_random_batches() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping fixed-point-q16 parity: no wgpu adapter on this host");
        return;
    };
    let engine = GpuFixedPointQ16::new(&ctx);
    for (seed, count) in [(1usize, 70usize), (2, 128), (3, 257), (4, 64)] {
        let mut rng = Lcg::new(0x51ED_0000_u64.wrapping_add(seed as u64));
        let queries = random_batch(&mut rng, count);
        check_batch(
            &format!("random batch {seed} (count={count})"),
            &engine,
            &ctx,
            &queries,
        );
    }
}

#[test]
fn gpu_matches_cpu_on_saturation_edges() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuFixedPointQ16::new(&ctx);
    // Both the overflow and non-overflow path of every saturating operation,
    // plus the i32::MIN absolute/negate wrap.
    let queries = vec![
        // add_sat: positive overflow, negative overflow, no overflow.
        GpuQ16Query {
            a: i32::MAX,
            b: 1,
            c: 0,
            op: GpuQ16Op::AddSat,
        },
        GpuQ16Query {
            a: i32::MIN,
            b: -1,
            c: 0,
            op: GpuQ16Op::AddSat,
        },
        GpuQ16Query {
            a: 1_000,
            b: -500,
            c: 0,
            op: GpuQ16Op::AddSat,
        },
        // sub_sat: positive overflow, negative overflow, no overflow.
        GpuQ16Query {
            a: i32::MAX,
            b: -1,
            c: 0,
            op: GpuQ16Op::SubSat,
        },
        GpuQ16Query {
            a: i32::MIN,
            b: 1,
            c: 0,
            op: GpuQ16Op::SubSat,
        },
        GpuQ16Query {
            a: -500,
            b: 500,
            c: 0,
            op: GpuQ16Op::SubSat,
        },
        // abs: i32::MIN saturates, negative, positive, zero.
        GpuQ16Query {
            a: i32::MIN,
            b: 0,
            c: 0,
            op: GpuQ16Op::Abs,
        },
        GpuQ16Query {
            a: -65_536,
            b: 0,
            c: 0,
            op: GpuQ16Op::Abs,
        },
        GpuQ16Query {
            a: 65_536,
            b: 0,
            c: 0,
            op: GpuQ16Op::Abs,
        },
        // neg: i32::MIN wraps to itself, ordinary negate.
        GpuQ16Query {
            a: i32::MIN,
            b: 0,
            c: 0,
            op: GpuQ16Op::Neg,
        },
        GpuQ16Query {
            a: 123_456,
            b: 0,
            c: 0,
            op: GpuQ16Op::Neg,
        },
        // wrapping add/sub at the extremes.
        GpuQ16Query {
            a: i32::MAX,
            b: 1,
            c: 0,
            op: GpuQ16Op::Add,
        },
        GpuQ16Query {
            a: i32::MIN,
            b: 1,
            c: 0,
            op: GpuQ16Op::Sub,
        },
    ];
    check_batch("saturation edges", &engine, &ctx, &queries);
}

#[test]
fn gpu_matches_cpu_on_floor_and_frac() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuFixedPointQ16::new(&ctx);
    // 1.75, -0.25, -2.6, exact integers and zero, exercising the positive and
    // negative floor/frac semantics (fraction stays in [0, 1)).
    let values = [
        Q16_16::ONE + Q16_16::ONE * 3 / 4, // 1.75
        -(Q16_16::ONE / 4),                // -0.25
        -(Q16_16::ONE * 13 / 5),           // -2.6
        3 * Q16_16::ONE,                   // 3.0
        0,
        -(Q16_16::ONE / 2), // -0.5
    ];
    let mut queries = Vec::new();
    for &v in &values {
        queries.push(GpuQ16Query {
            a: v,
            b: 0,
            c: 0,
            op: GpuQ16Op::Floor,
        });
        queries.push(GpuQ16Query {
            a: v,
            b: 0,
            c: 0,
            op: GpuQ16Op::Frac,
        });
        queries.push(GpuQ16Query {
            a: v,
            b: 0,
            c: 0,
            op: GpuQ16Op::ToIntTrunc,
        });
        queries.push(GpuQ16Query {
            a: v,
            b: 0,
            c: 0,
            op: GpuQ16Op::ToIntFloor,
        });
    }
    check_batch("floor and frac", &engine, &ctx, &queries);
}

#[test]
fn gpu_matches_cpu_on_min_max_clamp() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuFixedPointQ16::new(&ctx);
    let lo = -Q16_16::ONE;
    let hi = Q16_16::ONE;
    let queries = vec![
        GpuQ16Query {
            a: -2 * Q16_16::ONE,
            b: hi,
            c: 0,
            op: GpuQ16Op::Min,
        },
        GpuQ16Query {
            a: -2 * Q16_16::ONE,
            b: hi,
            c: 0,
            op: GpuQ16Op::Max,
        },
        // Clamp below, above and inside the band.
        GpuQ16Query {
            a: -5 * Q16_16::ONE,
            b: lo,
            c: hi,
            op: GpuQ16Op::Clamp,
        },
        GpuQ16Query {
            a: 5 * Q16_16::ONE,
            b: lo,
            c: hi,
            op: GpuQ16Op::Clamp,
        },
        GpuQ16Query {
            a: Q16_16::ONE / 4,
            b: lo,
            c: hi,
            op: GpuQ16Op::Clamp,
        },
        // from_int round-trips.
        GpuQ16Query {
            a: 5,
            b: 0,
            c: 0,
            op: GpuQ16Op::FromInt,
        },
        GpuQ16Query {
            a: -7,
            b: 0,
            c: 0,
            op: GpuQ16Op::FromInt,
        },
    ];
    check_batch("min max clamp", &engine, &ctx, &queries);
}

#[test]
fn gpu_matches_cpu_on_single_query() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuFixedPointQ16::new(&ctx);
    let queries = vec![GpuQ16Query {
        a: 3 * Q16_16::ONE / 2,
        b: Q16_16::ONE / 2,
        c: 0,
        op: GpuQ16Op::Add,
    }];
    check_batch("single query", &engine, &ctx, &queries);
}

#[test]
fn gpu_matches_cpu_on_empty_batch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuFixedPointQ16::new(&ctx);
    let out = engine.run(&ctx, &[]);
    assert!(
        out.is_empty(),
        "empty batch must short-circuit to an empty vec"
    );
}
