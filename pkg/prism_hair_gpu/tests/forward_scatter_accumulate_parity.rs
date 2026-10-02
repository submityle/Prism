//! Real-device parity for the forward-scatter crossing accumulation twin:
//! [`GpuForwardScatterAccumulate`] must reproduce the `CPU` golden
//! [`accumulate_forward_scatter`](prism_render_architecture::hair::dual_scattering::accumulate_forward_scatter)
//! for a batch of light rays, each a slice of
//! [`TransmittanceSample`](prism_render_architecture::hair::deep_transmittance::TransmittanceSample)
//! plus a receiver depth, emitting one coverage-weighted crossing count per ray
//! in input order.
//!
//! The suite covers the depth-gated sum over a multi-sample ray (asserting the
//! gated count is a genuine, strictly smaller subset of the total so no
//! count-everything kernel could pass), the infinite-receiver reduction to
//! [`total_crossings`](prism_render_architecture::hair::dual_scattering::total_crossings),
//! the public-field re-sanitise (an out-of-range struct-literal opacity is
//! clamped before it is summed), ordering independence (shuffled samples yield
//! the identical count), the empty ray (zero crossings), the empty batch (no
//! dispatch), a mixed multi-ray batch, and a 100-ray batch that crosses the
//! 64-wide dispatch boundary so threads in distinct workgroups each recover
//! their own count.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The reduction is a closed-form clamped sum with no transcendental call, so
//! the `CPU` and `GPU` evaluate the same arithmetic and diverge only through
//! legal reassociation or fused-multiply-add contraction. Each crossing count is
//! asserted to within `abs_diff < 1e-4` or `rel_diff < 1e-3` — tight enough to
//! fail a genuinely wrong port (a missing depth gate, a dropped clamp, a wrong
//! sign), loose enough to admit the reassociation. All sample data are explicit
//! decimal literals (never `sin`/`cos`).
//!
//! Provenance: standard coverage-weighted depth-gated crossing sum plus `wgpu`
//! compute dispatch; no Unreal Engine source or derived code.

use prism_hair_gpu::forward_scatter_accumulate::{ForwardScatterRay, GpuForwardScatterAccumulate};
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::deep_transmittance::TransmittanceSample;
use prism_render_architecture::hair::dual_scattering::{
    accumulate_forward_scatter, total_crossings,
};

/// Acquires a headless context, or `None` (with a skip notice) when the host has
/// no `wgpu` adapter so the suite stays green off-device.
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn context_or_skip(label: &str) -> Option<GpuContext> {
    match GpuContext::try_headless() {
        Some(ctx) => Some(ctx),
        None => {
            eprintln!("skipping {label}: no wgpu adapter on this host");
            None
        }
    }
}

/// True when `got` matches `want` within the fused-multiply-add tolerance.
fn close(got: f32, want: f32) -> bool {
    let diff = (got - want).abs();
    diff < 1.0e-4 || diff <= 1.0e-3 * want.abs()
}

/// A sanitised sample from explicit decimal literals (no transcendental input).
fn s(depth: f32, opacity: f32) -> TransmittanceSample {
    TransmittanceSample::new(depth, opacity)
}

/// Runs the twin and the golden on the same `rays`, asserting one
/// tolerance-close crossing count per ray in input order.
fn assert_batch_matches(
    ctx: &GpuContext,
    twin: &GpuForwardScatterAccumulate,
    rays: &[(Vec<TransmittanceSample>, f32)],
) {
    let inputs: Vec<ForwardScatterRay<'_>> = rays
        .iter()
        .map(|(samples, receiver_depth)| ForwardScatterRay {
            samples: samples.as_slice(),
            receiver_depth: *receiver_depth,
        })
        .collect();
    let gpu = twin.eval(ctx, &inputs);
    assert_eq!(gpu.len(), rays.len(), "one count per ray");
    for (r, (samples, receiver_depth)) in rays.iter().enumerate() {
        let cpu = accumulate_forward_scatter(samples, *receiver_depth);
        assert!(close(gpu[r], cpu), "ray {r}: gpu {}, cpu {cpu}", gpu[r]);
    }
}

#[test]
fn gpu_depth_gated_crossings_match_golden() {
    let Some(ctx) = context_or_skip("gpu_depth_gated_crossings_match_golden") else {
        return;
    };
    let twin = GpuForwardScatterAccumulate::new(&ctx);

    // Four strands at increasing depth; the receiver sits between the second and
    // third, so only the two front strands' coverage counts.
    let samples = vec![s(0.5, 0.4), s(1.0, 0.3), s(2.0, 0.6), s(3.0, 0.2)];
    let receiver = 1.5_f32;

    let inputs = [ForwardScatterRay {
        samples: &samples,
        receiver_depth: receiver,
    }];
    let gpu = twin.eval(&ctx, &inputs);
    let cpu = accumulate_forward_scatter(&samples, receiver);

    assert!(close(gpu[0], cpu), "gpu {}, cpu {cpu}", gpu[0]);
    // The gate must be live: counting only the two front strands is strictly
    // fewer crossings than counting all four.
    let total = total_crossings(&samples);
    assert!(
        cpu < total - 1.0e-3,
        "receiver gate must drop back strands: gated {cpu}, total {total}"
    );
    assert!(close(gpu[0], 0.7), "front coverage 0.4 + 0.3 = 0.7");
}

#[test]
fn gpu_infinite_receiver_matches_total_crossings() {
    let Some(ctx) = context_or_skip("gpu_infinite_receiver_matches_total_crossings") else {
        return;
    };
    let twin = GpuForwardScatterAccumulate::new(&ctx);

    let samples = vec![s(0.2, 0.5), s(5.0, 0.4), s(12.0, 0.9), s(40.0, 0.3)];
    let inputs = [ForwardScatterRay {
        samples: &samples,
        receiver_depth: f32::INFINITY,
    }];
    let gpu = twin.eval(&ctx, &inputs);

    // An infinitely deep receiver counts every strand: the saturation count.
    let total = total_crossings(&samples);
    assert!(close(gpu[0], total), "gpu {}, total {total}", gpu[0]);
    assert!(close(gpu[0], 2.1), "0.5 + 0.4 + 0.9 + 0.3 = 2.1");
}

#[test]
fn gpu_out_of_range_opacity_is_clamped() {
    let Some(ctx) = context_or_skip("gpu_out_of_range_opacity_is_clamped") else {
        return;
    };
    let twin = GpuForwardScatterAccumulate::new(&ctx);

    // Hand-built struct literals bypass `TransmittanceSample::new`, so the
    // kernel must reproduce the golden's defensive `opacity.clamp(0, 1)`.
    let samples = vec![
        TransmittanceSample {
            depth: 0.3,
            opacity: 1.75,
        },
        TransmittanceSample {
            depth: 0.6,
            opacity: -0.5,
        },
        TransmittanceSample {
            depth: 0.9,
            opacity: 0.4,
        },
    ];
    let receiver = 10.0_f32;
    let inputs = [ForwardScatterRay {
        samples: &samples,
        receiver_depth: receiver,
    }];
    let gpu = twin.eval(&ctx, &inputs);
    let cpu = accumulate_forward_scatter(&samples, receiver);

    assert!(close(gpu[0], cpu), "gpu {}, cpu {cpu}", gpu[0]);
    // clamp(1.75)=1 + clamp(-0.5)=0 + 0.4 = 1.4.
    assert!(close(gpu[0], 1.4), "clamped coverage 1 + 0 + 0.4 = 1.4");
}

#[test]
fn gpu_sum_is_order_independent() {
    let Some(ctx) = context_or_skip("gpu_sum_is_order_independent") else {
        return;
    };
    let twin = GpuForwardScatterAccumulate::new(&ctx);

    let ordered = vec![s(0.5, 0.2), s(1.5, 0.3), s(2.5, 0.4), s(3.5, 0.1)];
    let shuffled = vec![s(2.5, 0.4), s(0.5, 0.2), s(3.5, 0.1), s(1.5, 0.3)];
    let receiver = 2.0_f32;

    let inputs = [
        ForwardScatterRay {
            samples: &ordered,
            receiver_depth: receiver,
        },
        ForwardScatterRay {
            samples: &shuffled,
            receiver_depth: receiver,
        },
    ];
    let gpu = twin.eval(&ctx, &inputs);

    assert!(
        close(gpu[0], gpu[1]),
        "shuffled samples must yield the same count: {} vs {}",
        gpu[0],
        gpu[1]
    );
    let cpu = accumulate_forward_scatter(&ordered, receiver);
    assert!(close(gpu[0], cpu), "gpu {}, cpu {cpu}", gpu[0]);
    assert!(close(gpu[0], 0.5), "front coverage 0.2 + 0.3 = 0.5");
}

#[test]
fn gpu_empty_ray_yields_zero() {
    let Some(ctx) = context_or_skip("gpu_empty_ray_yields_zero") else {
        return;
    };
    let twin = GpuForwardScatterAccumulate::new(&ctx);

    // A single empty ray contributes no coverage; the batch still has one
    // sampled ray so a dispatch happens.
    let occluded = vec![s(1.0, 0.5), s(2.0, 0.5)];
    let rays = [(Vec::new(), 1.0_f32), (occluded.clone(), 5.0_f32)];
    let inputs: Vec<ForwardScatterRay<'_>> = rays
        .iter()
        .map(|(samples, receiver_depth)| ForwardScatterRay {
            samples: samples.as_slice(),
            receiver_depth: *receiver_depth,
        })
        .collect();
    let gpu = twin.eval(&ctx, &inputs);

    assert!(
        close(gpu[0], 0.0),
        "empty ray has zero crossings: {}",
        gpu[0]
    );
    assert!(close(gpu[1], 1.0), "0.5 + 0.5 = 1.0: {}", gpu[1]);
}

#[test]
fn gpu_empty_batch_yields_empty() {
    let Some(ctx) = context_or_skip("gpu_empty_batch_yields_empty") else {
        return;
    };
    let twin = GpuForwardScatterAccumulate::new(&ctx);
    assert!(
        twin.eval(&ctx, &[]).is_empty(),
        "empty batch -> empty result"
    );

    // Every ray empty -> all-zero counts without a dispatch.
    let rays = [(Vec::new(), 1.0_f32), (Vec::new(), 2.0_f32)];
    let inputs: Vec<ForwardScatterRay<'_>> = rays
        .iter()
        .map(|(samples, receiver_depth)| ForwardScatterRay {
            samples: samples.as_slice(),
            receiver_depth: *receiver_depth,
        })
        .collect();
    let gpu = twin.eval(&ctx, &inputs);
    assert_eq!(gpu.len(), 2, "one count per ray");
    assert!(gpu.iter().all(|&n| close(n, 0.0)), "all rays empty -> zero");
}

#[test]
fn gpu_mixed_multi_ray_batch_matches_golden() {
    let Some(ctx) = context_or_skip("gpu_mixed_multi_ray_batch_matches_golden") else {
        return;
    };
    let twin = GpuForwardScatterAccumulate::new(&ctx);

    let rays = vec![
        (vec![s(0.5, 0.3), s(1.5, 0.4), s(2.5, 0.2)], 1.0_f32),
        (vec![s(0.1, 0.9)], 0.05_f32),
        (
            vec![s(0.2, 0.1), s(0.4, 0.2), s(0.6, 0.3), s(0.8, 0.4)],
            f32::INFINITY,
        ),
        (Vec::new(), 3.0_f32),
    ];
    assert_batch_matches(&ctx, &twin, &rays);
}

#[test]
fn gpu_many_rays_cross_workgroup_boundary() {
    let Some(ctx) = context_or_skip("gpu_many_rays_cross_workgroup_boundary") else {
        return;
    };
    let twin = GpuForwardScatterAccumulate::new(&ctx);

    // 100 rays > 64 so threads span two workgroups; each ray's sample count and
    // receiver depth vary deterministically with its index.
    let mut rays: Vec<(Vec<TransmittanceSample>, f32)> = Vec::with_capacity(100);
    for i in 0..100u32 {
        let count = (i % 5) + 1;
        let mut samples = Vec::with_capacity(count as usize);
        for k in 0..count {
            let depth = 0.25 * (k as f32 + 1.0);
            let opacity = 0.1 + 0.05 * ((i + k) % 7) as f32;
            samples.push(s(depth, opacity));
        }
        let receiver = 0.25 * (((i % 4) + 1) as f32) + 0.1;
        rays.push((samples, receiver));
    }
    assert_batch_matches(&ctx, &twin, &rays);
}
