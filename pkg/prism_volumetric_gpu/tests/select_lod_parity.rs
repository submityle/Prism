//! Real-device parity for the cloud-LOD twin:
//! [`GpuSelectLod`] must reproduce the `CPU` golden
//! [`select_lod`](prism_render_architecture::volumetric::cloud_lod::select_lod)
//! across a deterministic sweep of view distances against fixed thresholds,
//! including the exact boundaries (`distance == threshold`, which selects the
//! coarser bucket because the classification is strict `<`).
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The classification is float comparisons against ascending thresholds, so the
//! parity test asserts every bucket matches the `CPU` golden exactly, including
//! the boundaries and the extreme inputs (negative, infinity). A degenerate
//! kernel (dropped branch, `<=` instead of `<`) could not pass.
//!
//! Provenance: standard distance-bucketed cloud LOD selection; no Unreal Engine
//! source or derived code.

use prism_render_architecture::volumetric::cloud_lod::{select_lod, CloudLod, CloudLodThresholds};
use prism_volumetric_gpu::{GpuContext, GpuSelectLod, SelectLodQuery};

/// Asserts every `gpu` bucket matches the `CPU` golden exactly.
fn assert_parity(queries: &[SelectLodQuery], gpu: &[CloudLod]) {
    assert_eq!(gpu.len(), queries.len(), "one bucket per query");
    for (i, q) in queries.iter().enumerate() {
        let exp = select_lod(q.distance, q.thresholds);
        assert_eq!(
            gpu[i], exp,
            "select-lod mismatch for query {i} (distance {}): gpu {:?}, cpu {exp:?}",
            q.distance, gpu[i]
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_select_lod_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping select-lod parity: no wgpu adapter on this host");
        return;
    };
    let gpu_kernel = GpuSelectLod::new(&ctx);

    let thresholds = CloudLodThresholds {
        mid_beyond: 1_000.0,
        far_beyond: 4_000.0,
        imposter_beyond: 12_000.0,
    };

    // Hand-picked scenes: the exact boundaries select the coarser bucket, plus
    // the extreme inputs.
    let mut queries: Vec<SelectLodQuery> = vec![
        SelectLodQuery {
            distance: -50.0,
            thresholds,
        },
        SelectLodQuery {
            distance: 0.0,
            thresholds,
        },
        // Exact boundaries: strict `<` sends these to the coarser bucket.
        SelectLodQuery {
            distance: thresholds.mid_beyond,
            thresholds,
        },
        SelectLodQuery {
            distance: thresholds.far_beyond,
            thresholds,
        },
        SelectLodQuery {
            distance: thresholds.imposter_beyond,
            thresholds,
        },
        SelectLodQuery {
            distance: f32::INFINITY,
            thresholds,
        },
    ];

    // A deterministic sweep spanning below the first threshold to well past the
    // last, crossing every boundary.
    for k in 0..=300 {
        let distance = (k as f32) * 50.0;
        queries.push(SelectLodQuery {
            distance,
            thresholds,
        });
    }

    let gpu = gpu_kernel.eval(&ctx, &queries);
    assert_parity(&queries, &gpu);

    // Spot-check the hand-picked scenes reach the expected buckets.
    assert_eq!(gpu[0], CloudLod::Near, "negative distance is Near");
    assert_eq!(gpu[1], CloudLod::Near, "zero distance is Near");
    assert_eq!(gpu[2], CloudLod::Mid, "mid boundary crosses to Mid");
    assert_eq!(gpu[3], CloudLod::Far, "far boundary crosses to Far");
    assert_eq!(
        gpu[4],
        CloudLod::Imposter,
        "imposter boundary crosses to Imposter"
    );
    assert_eq!(gpu[5], CloudLod::Imposter, "infinity is Imposter");
}

#[test]
fn empty_queries_yields_no_results() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_kernel = GpuSelectLod::new(&ctx);
    let out = gpu_kernel.eval(&ctx, &[]);
    assert!(out.is_empty(), "an empty query slice yields no values");
}
