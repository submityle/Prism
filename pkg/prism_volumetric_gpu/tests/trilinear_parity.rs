//! Real-device parity for the trilinear corner-weights twin: [`GpuTrilinear`]
//! must reproduce the `CPU` golden
//! [`trilinear_weights`](prism_render_architecture::volumetric::multiscatter::trilinear_weights)
//! across a spread of fractions, the cube corners and a deterministic ramp.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The blend is only multiply/add on the raw fractions — no transcendental —
//! so `CPU` and `GPU` evaluate the identical closed-form algebra, differing at
//! most by a legal multiply-add contraction of a few `ULP`. Each of the eight
//! weights is asserted to within `abs_diff < 1e-6` or `rel_diff < 1e-5` — tight
//! enough to fail a wrong port (a swapped `frac`/`1 - frac`, a mis-mapped
//! corner bit). The scenes also assert the partition-of-unity sum stays within
//! `1e-6` of one and that each weight is non-negative, so a degenerate kernel
//! could not pass.
//!
//! Provenance: standard trilinear partition-of-unity interpolation; no Unreal
//! Engine source or derived code.

use prism_render_architecture::volumetric::multiscatter::trilinear_weights;
use prism_volumetric_gpu::{GpuContext, GpuTrilinear, TrilinearQuery};

/// Asserts every `gpu` weight matches the `CPU` golden to within the documented
/// tolerance, stays non-negative and sums to one.
fn assert_parity(queries: &[TrilinearQuery], gpu: &[[f32; 8]]) {
    assert_eq!(gpu.len(), queries.len(), "one weight octet per query");
    for (i, q) in queries.iter().enumerate() {
        let exp = trilinear_weights(q.fx, q.fy, q.fz);
        let got = gpu[i];
        let mut sum = 0.0f32;
        for corner in 0..8 {
            let abs_diff = (got[corner] - exp[corner]).abs();
            let rel_diff = abs_diff / exp[corner].abs().max(1e-6);
            assert!(
                abs_diff < 1e-6 || rel_diff < 1e-5,
                "trilinear mismatch for query {i} ({q:?}) corner {corner}: \
                 gpu {}, cpu {} (abs {abs_diff}, rel {rel_diff})",
                got[corner],
                exp[corner]
            );
            assert!(
                got[corner] >= -1e-6,
                "gpu trilinear weight must be non-negative: {}",
                got[corner]
            );
            sum += got[corner];
        }
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "gpu trilinear weights must form a partition of unity: sum {sum}"
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_trilinear_matches_cpu_golden_across_queries() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping trilinear parity: no wgpu adapter on this host");
        return;
    };
    let gpu_trilinear = GpuTrilinear::new(&ctx);

    // A deterministic spread: the cube origin, the far corner, the centre, and
    // several asymmetric mixes exercising every axis independently.
    let mut queries: Vec<TrilinearQuery> = vec![
        TrilinearQuery {
            fx: 0.0,
            fy: 0.0,
            fz: 0.0,
        },
        TrilinearQuery {
            fx: 1.0,
            fy: 1.0,
            fz: 1.0,
        },
        TrilinearQuery {
            fx: 0.5,
            fy: 0.5,
            fz: 0.5,
        },
        TrilinearQuery {
            fx: 0.25,
            fy: 0.75,
            fz: 0.1,
        },
        TrilinearQuery {
            fx: 0.9,
            fy: 0.05,
            fz: 0.6,
        },
        TrilinearQuery {
            fx: 0.33,
            fy: 0.66,
            fz: 0.99,
        },
    ];
    // A deterministic diagonal ramp across the cube.
    for k in 0..90 {
        let t = (k as f32) / 89.0;
        queries.push(TrilinearQuery {
            fx: t,
            fy: 1.0 - t,
            fz: (t * 2.0).fract(),
        });
    }

    let gpu = gpu_trilinear.eval(&ctx, &queries);
    assert_parity(&queries, &gpu);

    // The origin puts all weight on corner 0; the far corner on corner 7; the
    // centre splits evenly across all eight corners.
    assert!(
        (gpu[0][0] - 1.0).abs() < 1e-6 && gpu[0][1..].iter().all(|w| w.abs() < 1e-6),
        "the cube origin must weight only corner 0: {:?}",
        gpu[0]
    );
    assert!(
        (gpu[1][7] - 1.0).abs() < 1e-6 && gpu[1][..7].iter().all(|w| w.abs() < 1e-6),
        "the far corner must weight only corner 7: {:?}",
        gpu[1]
    );
    assert!(
        gpu[2].iter().all(|w| (w - 0.125).abs() < 1e-6),
        "the cube centre must split evenly across all corners: {:?}",
        gpu[2]
    );
}

#[test]
fn empty_queries_yields_no_results() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_trilinear = GpuTrilinear::new(&ctx);
    let out = gpu_trilinear.eval(&ctx, &[]);
    assert!(out.is_empty(), "an empty query slice yields no values");
}
