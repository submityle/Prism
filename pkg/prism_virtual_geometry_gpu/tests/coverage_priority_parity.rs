//! Real-device parity for the virtual-geometry paging priority twin:
//! [`GpuCoveragePriority`] must reproduce the CPU golden
//! [`coverage_priority`](prism_render_architecture::virtual_geometry::coverage_priority)
//! for every query — the `center - view_origin` difference, the squared
//! distance with its `f32::EPSILON` floor, the `radius.max(0)` clamp, the focal
//! multiply and the divide by the clamped squared distance.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The value is `extent^2 / max(dx^2 + dy^2 + dz^2, EPSILON)`. `max` is exact
//! and every multiply is correctly rounded on both sides, but a GPU divide by a
//! non-power-of-two divisor is only guaranteed correctly rounded to within one
//! ULP, so the general criterion is `<= 1 ULP`. When the clamped squared
//! distance is an exact power of two the divide degenerates to exponent scaling
//! and the result is bit-for-bit equal; those cases are asserted with zero
//! tolerance. All test operands keep the squares and their partial sums exactly
//! representable so an on-device FMA contraction of the sum of squares cannot
//! diverge from the sequential CPU adds — the only inexact step is the divide.
//!
//! Provenance: standard perspective screen-coverage streaming priority for a
//! paged cluster hierarchy; no Unreal Engine source or derived code.

use prism_render_architecture::gpu_scene::SceneBounds;
use prism_render_architecture::virtual_geometry::{coverage_priority, LodProjection};
use prism_virtual_geometry_gpu::{CoveragePriorityQuery, GpuContext, GpuCoveragePriority};

/// ULP distance between two finite, equal-sign `f32` values. Every priority is
/// non-negative, so the bit pattern is monotonic and the unsigned integer
/// difference of the bit patterns is the ULP distance.
fn ulp_distance(a: f32, b: f32) -> u32 {
    let (a, b) = (a.to_bits(), b.to_bits());
    a.abs_diff(b)
}

/// Evaluates the CPU golden for one query.
fn cpu_golden(q: &CoveragePriorityQuery) -> f32 {
    let bounds = SceneBounds {
        center: q.center,
        radius: q.radius,
        half_extents: [q.radius.abs(), q.radius.abs(), q.radius.abs()],
        _padding: 0.0,
    };
    let projection = LodProjection::from_focal_length_pixels(q.focal_length_pixels);
    coverage_priority(&bounds, q.view_origin, projection)
}

/// Asserts the twin matches the golden within one ULP for every query - the
/// honest bound for a GPU divide by an arbitrary divisor.
fn assert_parity(ctx: &GpuContext, queries: &[CoveragePriorityQuery]) {
    let gpu = GpuCoveragePriority::new(ctx).evaluate(ctx, queries);
    assert_eq!(gpu.len(), queries.len(), "one priority per query");
    for (i, q) in queries.iter().enumerate() {
        let expected = cpu_golden(q);
        let ulps = ulp_distance(gpu[i], expected);
        assert!(
            ulps <= 1,
            "priority off by {ulps} ULP for query {i}: gpu {} ({:#010x}), cpu {expected} ({:#010x})",
            gpu[i],
            gpu[i].to_bits(),
            expected.to_bits(),
        );
    }
}

/// Asserts the twin matches the golden bit-for-bit - only valid when the
/// clamped squared distance is a power of two (or the result is zero), where
/// the divide is exact exponent scaling.
fn assert_bit_exact(ctx: &GpuContext, queries: &[CoveragePriorityQuery]) {
    let gpu = GpuCoveragePriority::new(ctx).evaluate(ctx, queries);
    assert_eq!(gpu.len(), queries.len(), "one priority per query");
    for (i, q) in queries.iter().enumerate() {
        let expected = cpu_golden(q);
        assert_eq!(
            gpu[i].to_bits(),
            expected.to_bits(),
            "power-of-two divisor must be bit-exact for query {i}: gpu {} ({:#010x}), cpu {expected} ({:#010x})",
            gpu[i],
            gpu[i].to_bits(),
            expected.to_bits(),
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_coverage_priority_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping coverage-priority parity: no wgpu adapter on this host");
        return;
    };
    // (dx, dy, dz) chosen so dx^2 + dy^2 + dz^2 is an exactly representable
    // integer; non-power-of-two sums (25, 169, 14) only guarantee
    // correctly-rounded-to-1-ULP division on the GPU.
    let queries = [
        CoveragePriorityQuery {
            center: [3.0, 4.0, 0.0], // 9 + 16 = 25
            radius: 2.0,
            view_origin: [0.0, 0.0, 0.0],
            focal_length_pixels: 8.0, // extent = 16, extent^2 = 256
        },
        CoveragePriorityQuery {
            center: [5.0, 12.0, 0.0], // 25 + 144 = 169
            radius: 1.5,
            view_origin: [0.0, 0.0, 0.0],
            focal_length_pixels: 4.0, // extent = 6, extent^2 = 36
        },
        CoveragePriorityQuery {
            center: [11.0, 3.0, 4.0], // (11-8)^2 + 9 + 16 = 9 + 9 + ... see origin
            radius: 3.0,
            view_origin: [8.0, 0.0, 2.0], // dx=3,dy=3,dz=2 -> 9+9+4 = 22
            focal_length_pixels: 2.0, // extent = 6, extent^2 = 36
        },
    ];
    assert_parity(&ctx, &queries);
}

#[test]
fn gpu_coverage_priority_power_of_two_divisor_is_bit_exact() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    // Every squared distance is a power of two, so the divide is exact exponent
    // scaling and the twin must match the golden to the bit.
    let queries = [
        CoveragePriorityQuery {
            center: [2.0, 0.0, 0.0], // distance_sq = 4
            radius: 2.0,
            view_origin: [0.0, 0.0, 0.0],
            focal_length_pixels: 8.0, // extent = 16, 256 / 4 = 64
        },
        CoveragePriorityQuery {
            center: [0.0, 0.0, 8.0], // distance_sq = 64
            radius: 1.0,
            view_origin: [0.0, 0.0, 0.0],
            focal_length_pixels: 16.0, // extent = 16, 256 / 64 = 4
        },
        CoveragePriorityQuery {
            center: [4.0, 0.0, 0.0], // distance_sq = 16
            radius: 4.0,
            view_origin: [0.0, 0.0, 0.0],
            focal_length_pixels: 2.0, // extent = 8, 64 / 16 = 4
        },
    ];
    assert_bit_exact(&ctx, &queries);

    let gpu = GpuCoveragePriority::new(&ctx).evaluate(&ctx, &queries);
    assert_eq!(gpu[0], 64.0, "256 / 4 must be exactly 64");
    assert_eq!(gpu[1], 4.0, "256 / 64 must be exactly 4");
    assert_eq!(gpu[2], 4.0, "64 / 16 must be exactly 4");
}

#[test]
fn gpu_coverage_priority_clamps_negative_radius_to_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    // Negative radius must clamp to 0 -> extent 0 -> priority 0, exactly as
    // `radius.max(0.0)` on the golden (result is zero, so bit-exact).
    let queries = [
        CoveragePriorityQuery {
            center: [3.0, 4.0, 0.0],
            radius: -2.0,
            view_origin: [0.0, 0.0, 0.0],
            focal_length_pixels: 8.0,
        },
        CoveragePriorityQuery {
            center: [1.0, 0.0, 0.0],
            radius: -0.001,
            view_origin: [0.0, 0.0, 0.0],
            focal_length_pixels: 100.0,
        },
        CoveragePriorityQuery {
            center: [5.0, 0.0, 0.0],
            radius: 0.0,
            view_origin: [0.0, 0.0, 0.0],
            focal_length_pixels: 64.0,
        },
    ];
    assert_bit_exact(&ctx, &queries);

    let gpu = GpuCoveragePriority::new(&ctx).evaluate(&ctx, &queries);
    assert!(
        gpu.iter().all(|&p| p == 0.0),
        "non-positive radius yields zero priority: {gpu:?}"
    );
}

#[test]
fn gpu_coverage_priority_clamps_sub_epsilon_distance() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    // A camera resting on the sphere centre collapses distance_sq to 0, which
    // must clamp up to f32::EPSILON (= 2^-23, a power of two) before the divide,
    // so the divide is exact exponent scaling and the twin is bit-exact.
    let queries = [
        CoveragePriorityQuery {
            center: [4.0, 5.0, 6.0],
            radius: 2.0,
            view_origin: [4.0, 5.0, 6.0], // distance_sq = 0 -> clamps to EPSILON
            focal_length_pixels: 8.0,
        },
        CoveragePriorityQuery {
            center: [1.0, 1.0, 1.0],
            radius: 1.0,
            view_origin: [1.0, 1.0, 1.0], // distance_sq = 0 -> clamps to EPSILON
            focal_length_pixels: 16.0,
        },
    ];
    assert_bit_exact(&ctx, &queries);

    let gpu = GpuCoveragePriority::new(&ctx).evaluate(&ctx, &queries);
    assert!(
        gpu.iter().all(|&p| p.is_finite() && p > 0.0),
        "coincident camera yields the maximum finite priority: {gpu:?}"
    );
}

#[test]
fn gpu_coverage_priority_handles_many_queries() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    // More than one workgroup (>64) to exercise the dispatch tiling and
    // per-thread independence. Distances are integer dx along one axis so the
    // squared distance is exactly representable; non-power-of-two divisors make
    // parity the <= 1 ULP bound.
    let mut queries: Vec<CoveragePriorityQuery> = Vec::new();
    for k in 0..200i32 {
        let dx = f32::from(i16::try_from(k % 40 + 1).expect("small range fits i16"));
        let radius = f32::from(i16::try_from(k % 8 + 1).expect("small range fits i16"));
        queries.push(CoveragePriorityQuery {
            center: [dx, 0.0, 0.0],
            radius,
            view_origin: [0.0, 0.0, 0.0],
            focal_length_pixels: 4.0,
        });
    }
    assert_parity(&ctx, &queries);
}

#[test]
fn empty_input_yields_empty_output() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let out = GpuCoveragePriority::new(&ctx).evaluate(&ctx, &[]);
    assert!(out.is_empty(), "no queries yields no priorities");
}
