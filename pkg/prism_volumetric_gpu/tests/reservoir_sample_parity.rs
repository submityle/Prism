//! Real-device parity for the weighted reservoir sampling (`WRS` / `RIS`) twin:
//! [`GpuReservoirSample`](prism_volumetric_gpu::reservoir_sample::GpuReservoirSample)
//! must reproduce the `CPU` golden
//! [`reservoir_sample`](prism_render_architecture::particle::reservoir_sample)
//! per element across the stream-sampling and merge transforms.
//!
//! The fixtures cover a single candidate, equal-weight and increasing-weight
//! multi-candidate streams, a zero-weight candidate (counted but never
//! selected), an empty stream (`m == 0` degenerate), several independent
//! streams dispatched together, and a spread of seeds. The merge fixtures fuse
//! singleton and multi-count reservoirs, including an empty operand.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernels are portable core-`WGSL`, so they need no optional device
//! feature.
//!
//! # Parity criterion
//!
//! The `splitmix32` `RNG` is pure wrapping `u32` algebra, so both sides draw
//! the identical uniform word stream and fold the identical terms in the same
//! order. The held sample index (`y`) and the count (`m`) must therefore agree
//! exactly and are compared with `==`. The `f32` accumulators `w_sum` and the
//! unbiased contribution weight `W` are compared with the tolerance
//! `abs_diff <= 1e-4 || rel_diff <= 1e-3` (relative floor `1e-6`) because an
//! independent `GPU` may fuse or round the `f32` adds and the final divide
//! differently. `WGSL` has no `u64` and the reference is already `u32`-only, so
//! nothing is out of scope here.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::reservoir_sample`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::reservoir_sample::{Reservoir, Rng};
use prism_volumetric_gpu::reservoir_sample::{GpuReservoirSample, ReservoirValue};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the `f32` accumulators.
const ABS_TOL: f32 = 1e-4;
/// Relative tolerance for the `f32` accumulators.
const REL_TOL: f32 = 1e-3;
/// Relative-difference denominator floor, guarding the near-zero case.
const REL_FLOOR: f32 = 1e-6;

/// Returns `true` when `a` and `b` agree within the `f32` accumulator
/// tolerance: an absolute gap at or under [`ABS_TOL`], or a relative gap at or
/// under [`REL_TOL`] against a floored magnitude.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= ABS_TOL {
        return true;
    }
    let denom = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / denom <= REL_TOL
}

/// Asserts a `GPU` reservoir matches the `CPU` golden: `y` and `m` exactly,
/// `w_sum` and `W` within tolerance.
fn assert_reservoir(got: ReservoirValue, want: &Reservoir, ctx: &str) {
    assert_eq!(got.sample, want.sample, "sample (y) mismatch: {ctx}");
    assert_eq!(got.m, want.m, "count (m) mismatch: {ctx}");
    assert!(
        close(got.w_sum, want.w_sum),
        "w_sum mismatch: {ctx} got {} want {}",
        got.w_sum,
        want.w_sum
    );
    assert!(
        close(got.w, want.w),
        "w (W) mismatch: {ctx} got {} want {}",
        got.w,
        want.w
    );
}

/// One candidate stream fixture: an `RNG` seed, the candidate / weight pairs
/// and a target `pdf` for `finalize_w`.
struct Stream {
    seed: u32,
    candidates: Vec<u32>,
    weights: Vec<f32>,
    target_pdf: f32,
}

/// `CPU` golden for one stream: an empty reservoir driven by one `update` per
/// candidate (drawing one uniform each) then `finalize_w`.
fn cpu_stream(s: &Stream) -> Reservoir {
    let mut rng = Rng::new(s.seed);
    let mut r = Reservoir::empty();
    for i in 0..s.candidates.len() {
        let rand = rng.next_u01();
        r.update(s.candidates[i], s.weights[i], rand);
    }
    r.finalize_w(s.target_pdf);
    r
}

/// Flattens a set of streams into the parallel arrays the kernel expects and
/// runs the `GPU` sampler.
fn run_streams(
    gpu: &GpuReservoirSample,
    ctx: &GpuContext,
    streams: &[Stream],
) -> Vec<ReservoirValue> {
    let mut seeds = Vec::new();
    let mut offsets = Vec::new();
    let mut lengths = Vec::new();
    let mut target_pdfs = Vec::new();
    let mut candidates = Vec::new();
    let mut weights = Vec::new();
    for s in streams {
        seeds.push(s.seed);
        offsets.push(candidates.len() as u32);
        lengths.push(s.candidates.len() as u32);
        target_pdfs.push(s.target_pdf);
        candidates.extend_from_slice(&s.candidates);
        weights.extend_from_slice(&s.weights);
    }
    gpu.sample_streams(
        ctx,
        &seeds,
        &offsets,
        &lengths,
        &target_pdfs,
        &candidates,
        &weights,
    )
}

/// A broad fixture of streams exercising every required shape.
fn stream_fixture() -> Vec<Stream> {
    vec![
        // Single candidate: always selected, m = 1.
        Stream {
            seed: 0xABCD_1234,
            candidates: vec![42],
            weights: vec![2.5],
            target_pdf: 1.0,
        },
        // Equal-weight multi-candidate.
        Stream {
            seed: 0x0BAD_F00D,
            candidates: vec![0, 1, 2, 3, 4, 5, 6, 7],
            weights: vec![1.0; 8],
            target_pdf: 2.0,
        },
        // Strictly increasing weights.
        Stream {
            seed: 0xDEAD_BEEF,
            candidates: vec![10, 11, 12, 13, 14],
            weights: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            target_pdf: 0.5,
        },
        // A zero-weight candidate in the middle: counted, never selected on its
        // own term.
        Stream {
            seed: 0x1357_9BDF,
            candidates: vec![100, 101, 102, 103],
            weights: vec![3.0, 0.0, 2.0, 0.0],
            target_pdf: 4.0,
        },
        // Empty stream: degenerate m = 0, W forced to zero.
        Stream {
            seed: 0x0000_0001,
            candidates: vec![],
            weights: vec![],
            target_pdf: 1.0,
        },
        // Fractional weights and a vanishing target pdf (W clamps to zero).
        Stream {
            seed: 0x9E37_79B9,
            candidates: vec![7, 8, 9],
            weights: vec![0.25, 0.5, 1.25],
            target_pdf: 1e-7,
        },
        // Longer well-mixed stream under a different seed.
        Stream {
            seed: 0x5555_AAAA,
            candidates: (0..32).collect(),
            weights: (0..32).map(|i| 1.0 + (i % 5) as f32).collect(),
            target_pdf: 3.0,
        },
    ]
}

#[test]
fn sample_streams_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuReservoirSample::new(&ctx);
    let streams = stream_fixture();
    let got = run_streams(&gpu, &ctx, &streams);
    assert_eq!(got.len(), streams.len());
    for (idx, s) in streams.iter().enumerate() {
        let want = cpu_stream(s);
        assert_reservoir(
            got[idx],
            &want,
            &format!("stream {idx} seed {:#010x}", s.seed),
        );
    }
}

#[test]
fn single_candidate_is_always_selected() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuReservoirSample::new(&ctx);
    let streams = vec![Stream {
        seed: 123,
        candidates: vec![77],
        weights: vec![9.0],
        target_pdf: 1.0,
    }];
    let got = run_streams(&gpu, &ctx, &streams);
    assert_eq!(got[0].sample, 77);
    assert_eq!(got[0].m, 1);
    assert!(close(got[0].w_sum, 9.0));
}

#[test]
fn empty_stream_is_degenerate() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuReservoirSample::new(&ctx);
    // A single empty stream: the flattened candidate buffer is empty, so the
    // dispatch exercises the zero-size padding path too.
    let streams = vec![Stream {
        seed: 42,
        candidates: vec![],
        weights: vec![],
        target_pdf: 2.0,
    }];
    let got = run_streams(&gpu, &ctx, &streams);
    assert_eq!(got[0].sample, 0);
    assert_eq!(got[0].m, 0);
    assert!(close(got[0].w_sum, 0.0));
    assert!(close(got[0].w, 0.0));
}

#[test]
fn independent_streams_do_not_interfere() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuReservoirSample::new(&ctx);
    // Many streams with distinct seeds dispatched together must each match the
    // CPU golden computed in isolation, proving per-thread independence.
    let seeds = [1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17];
    let streams: Vec<Stream> = seeds
        .iter()
        .enumerate()
        .map(|(k, &seed)| {
            let n = 1 + (k % 6);
            Stream {
                seed,
                candidates: (0..n as u32).map(|c| c + 1000 * k as u32).collect(),
                weights: (0..n).map(|c| 1.0 + c as f32 * 0.5).collect(),
                target_pdf: 1.0 + k as f32,
            }
        })
        .collect();
    let got = run_streams(&gpu, &ctx, &streams);
    assert_eq!(got.len(), streams.len());
    for (idx, s) in streams.iter().enumerate() {
        let want = cpu_stream(s);
        assert_reservoir(got[idx], &want, &format!("independent stream {idx}"));
    }
}

#[test]
fn empty_input_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuReservoirSample::new(&ctx);
    assert!(gpu
        .sample_streams(&ctx, &[], &[], &[], &[], &[], &[])
        .is_empty());
    assert!(gpu.merge_pairs(&ctx, &[], &[], &[], &[]).is_empty());
}

/// `CPU` golden for one merge: `a` adopts `b` with one `splitmix32` draw, then
/// `finalize_w`.
fn cpu_merge(a: Reservoir, b: Reservoir, seed: u32, target_pdf: f32) -> Reservoir {
    let rand = Rng::new(seed).next_u01();
    let mut merged = a;
    merged.merge(&b, rand);
    merged.finalize_w(target_pdf);
    merged
}

/// Converts a golden [`Reservoir`] into the uploadable [`ReservoirValue`].
fn value_of(r: &Reservoir) -> ReservoirValue {
    ReservoirValue {
        sample: r.sample,
        w_sum: r.w_sum,
        m: r.m,
        w: r.w,
    }
}

#[test]
fn merge_pairs_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuReservoirSample::new(&ctx);
    // Pairs of reservoirs with a spread of w_sum ratios, counts and seeds,
    // including an empty operand on either side.
    let a_in = [
        Reservoir {
            sample: 10,
            w_sum: 1.0,
            m: 1,
            w: 0.0,
        },
        Reservoir {
            sample: 9,
            w_sum: 2.0,
            m: 4,
            w: 0.0,
        },
        Reservoir {
            sample: 5,
            w_sum: 6.0,
            m: 3,
            w: 0.0,
        },
        Reservoir::empty(),
        Reservoir {
            sample: 1,
            w_sum: 0.5,
            m: 2,
            w: 0.0,
        },
    ];
    let b_in = [
        Reservoir {
            sample: 20,
            w_sum: 3.0,
            m: 1,
            w: 0.0,
        },
        Reservoir::empty(),
        Reservoir {
            sample: 6,
            w_sum: 2.0,
            m: 5,
            w: 0.0,
        },
        Reservoir {
            sample: 30,
            w_sum: 4.0,
            m: 2,
            w: 0.0,
        },
        Reservoir {
            sample: 2,
            w_sum: 0.5,
            m: 2,
            w: 0.0,
        },
    ];
    let seeds = [0xABCD_1234u32, 0x0BAD_F00D, 0xDEAD_BEEF, 7, 0x5555_AAAA];
    let target_pdfs = [2.0f32, 1.0, 0.5, 1.0, 4.0];

    let a_vals: Vec<ReservoirValue> = a_in.iter().map(value_of).collect();
    let b_vals: Vec<ReservoirValue> = b_in.iter().map(value_of).collect();
    let got = gpu.merge_pairs(&ctx, &a_vals, &b_vals, &seeds, &target_pdfs);
    assert_eq!(got.len(), a_in.len());
    for idx in 0..a_in.len() {
        let want = cpu_merge(a_in[idx], b_in[idx], seeds[idx], target_pdfs[idx]);
        assert_reservoir(got[idx], &want, &format!("merge pair {idx}"));
    }
}

#[test]
fn merge_with_empty_operand_preserves_sample() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuReservoirSample::new(&ctx);
    // Merging an empty reservoir in adds nothing: w_sum unchanged, m unchanged,
    // sample preserved (0 * w_sum < 0 is false).
    let a = Reservoir {
        sample: 9,
        w_sum: 2.0,
        m: 4,
        w: 0.0,
    };
    let empty = Reservoir::empty();
    let got = gpu.merge_pairs(&ctx, &[value_of(&a)], &[value_of(&empty)], &[12345], &[1.0]);
    let want = cpu_merge(a, empty, 12345, 1.0);
    assert_reservoir(got[0], &want, "merge with empty operand");
    assert_eq!(got[0].sample, 9);
    assert_eq!(got[0].m, 4);
}
