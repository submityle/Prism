//! Real-device parity for the LOD screen-space-error projection twin:
//! [`GpuLodProjection`] must reproduce the CPU golden
//! [`LodProjection::projected_error_pixels`](prism_render_architecture::virtual_geometry::LodProjection::projected_error_pixels)
//! for every query — the `error.max(0)` clamp, the `distance.max(EPSILON)`
//! clamp, the multiply by the focal length and the divide by the clamped
//! distance.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The value is `max(error, 0) * focal / max(distance, EPSILON)`. `max` is
//! exact and the multiply is correctly rounded on both sides, but a GPU divide
//! by a non-power-of-two divisor is only guaranteed correctly rounded to within
//! one ULP (Metal implements it as a refined reciprocal), so the general
//! criterion is `<= 1 ULP`. When the clamped distance is an exact power of two
//! the divide degenerates to exact exponent scaling and the result is
//! bit-for-bit equal; those cases are asserted with zero tolerance as spot
//! checks. The boundary cases (negative error clamped to zero, sub-EPSILON
//! distance clamped up) land on power-of-two or zero divisors and are therefore
//! bit-exact.
//!
//! Provenance: standard perspective screen-space-error projection for LOD
//! selection; no Unreal Engine source or derived code.

use prism_render_architecture::virtual_geometry::LodProjection;
use prism_virtual_geometry_gpu::{GpuContext, GpuLodProjection, ProjectedErrorQuery};

/// ULP distance between two finite, equal-sign `f32` values. For non-negative
/// results (which every projection is) the bit pattern is monotonic, so the
/// unsigned integer difference of the bit patterns is the ULP distance.
fn ulp_distance(a: f32, b: f32) -> u32 {
    let (a, b) = (a.to_bits(), b.to_bits());
    a.abs_diff(b)
}

/// Asserts the twin matches the golden within one ULP for every query - the
/// honest bound for a GPU divide by an arbitrary divisor.
fn assert_parity(ctx: &GpuContext, queries: &[ProjectedErrorQuery]) {
    let gpu = GpuLodProjection::new(ctx).project(ctx, queries);
    assert_eq!(gpu.len(), queries.len(), "one pixel size per query");
    for (i, q) in queries.iter().enumerate() {
        let projection = LodProjection::from_focal_length_pixels(q.focal_length_pixels);
        let expected = projection.projected_error_pixels(q.geometric_error, q.view_distance);
        let ulps = ulp_distance(gpu[i], expected);
        assert!(
            ulps <= 1,
            "projection off by {ulps} ULP for query {i}: gpu {} ({:#010x}), cpu {expected} ({:#010x})",
            gpu[i],
            gpu[i].to_bits(),
            expected.to_bits(),
        );
    }
}

/// Asserts the twin matches the golden bit-for-bit - only valid when the
/// clamped distance is a power of two (or the result is zero), where the divide
/// is exact exponent scaling.
fn assert_bit_exact(ctx: &GpuContext, queries: &[ProjectedErrorQuery]) {
    let gpu = GpuLodProjection::new(ctx).project(ctx, queries);
    assert_eq!(gpu.len(), queries.len(), "one pixel size per query");
    for (i, q) in queries.iter().enumerate() {
        let projection = LodProjection::from_focal_length_pixels(q.focal_length_pixels);
        let expected = projection.projected_error_pixels(q.geometric_error, q.view_distance);
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
fn gpu_lod_projection_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping lod-projection parity: no wgpu adapter on this host");
        return;
    };
    // A spread of finite errors, distances and focal lengths - dyadic and
    // non-dyadic mixed. Non-power-of-two divisors (100.0, 3.0, 250.75) only
    // guarantee correctly-rounded-to-1-ULP division on the GPU.
    let queries = [
        ProjectedErrorQuery {
            geometric_error: 0.5,
            view_distance: 8.0,
            focal_length_pixels: 540.0,
        },
        ProjectedErrorQuery {
            geometric_error: 2.0,
            view_distance: 100.0,
            focal_length_pixels: 960.0,
        },
        ProjectedErrorQuery {
            geometric_error: 0.125,
            view_distance: 3.0,
            focal_length_pixels: 719.3,
        },
        ProjectedErrorQuery {
            geometric_error: 13.37,
            view_distance: 250.75,
            focal_length_pixels: 1234.5,
        },
    ];
    assert_parity(&ctx, &queries);
}

#[test]
fn gpu_lod_projection_power_of_two_divisor_is_bit_exact() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    // Every clamped distance is a power of two, so the divide is exact exponent
    // scaling and the twin must match the golden to the bit.
    let queries = [
        ProjectedErrorQuery {
            geometric_error: 0.5,
            view_distance: 8.0,
            focal_length_pixels: 540.0,
        }, // 0.5 * 540 / 8 = 33.75, exact.
        ProjectedErrorQuery {
            geometric_error: 3.0,
            view_distance: 4.0,
            focal_length_pixels: 1024.0,
        }, // 3 * 1024 / 4 = 768, exact.
        ProjectedErrorQuery {
            geometric_error: 1.5,
            view_distance: 2.0,
            focal_length_pixels: 256.0,
        }, // 1.5 * 256 / 2 = 192, exact.
    ];
    assert_bit_exact(&ctx, &queries);

    let gpu = GpuLodProjection::new(&ctx).project(&ctx, &queries);
    assert_eq!(gpu[0], 33.75, "0.5 * 540 / 8 must be exactly 33.75");
    assert_eq!(gpu[1], 768.0, "3 * 1024 / 4 must be exactly 768");
    assert_eq!(gpu[2], 192.0, "1.5 * 256 / 2 must be exactly 192");
}

#[test]
fn gpu_lod_projection_clamps_negative_error_to_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    // Negative geometric error must clamp to 0 -> projected pixels 0, exactly
    // as `error.max(0.0)` on the golden (result is zero, so bit-exact).
    let queries = [
        ProjectedErrorQuery {
            geometric_error: -1.0,
            view_distance: 10.0,
            focal_length_pixels: 512.0,
        },
        ProjectedErrorQuery {
            geometric_error: -0.001,
            view_distance: 1.0,
            focal_length_pixels: 800.0,
        },
        ProjectedErrorQuery {
            geometric_error: 0.0,
            view_distance: 4.0,
            focal_length_pixels: 640.0,
        },
    ];
    assert_bit_exact(&ctx, &queries);

    let gpu = GpuLodProjection::new(&ctx).project(&ctx, &queries);
    assert!(
        gpu.iter().all(|&p| p == 0.0),
        "non-positive error projects to zero pixels: {gpu:?}"
    );
}

#[test]
fn gpu_lod_projection_clamps_sub_epsilon_distance() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    // Zero / negative / sub-EPSILON distances must all clamp up to
    // f32::EPSILON (= 2^-23, a power of two) before the divide, so the divide
    // is exact exponent scaling and the twin is bit-exact.
    let queries = [
        ProjectedErrorQuery {
            geometric_error: 1.0,
            view_distance: 0.0,
            focal_length_pixels: 256.0,
        },
        ProjectedErrorQuery {
            geometric_error: 1.0,
            view_distance: -5.0,
            focal_length_pixels: 256.0,
        },
        ProjectedErrorQuery {
            geometric_error: 1.0,
            view_distance: f32::EPSILON * 0.5,
            focal_length_pixels: 256.0,
        },
        ProjectedErrorQuery {
            geometric_error: 1.0,
            view_distance: f32::EPSILON,
            focal_length_pixels: 256.0,
        },
    ];
    assert_bit_exact(&ctx, &queries);

    // The three sub-EPSILON distances all collapse to the same clamped value.
    let gpu = GpuLodProjection::new(&ctx).project(&ctx, &queries);
    assert_eq!(gpu[0], gpu[1], "0 and -5 both clamp to EPSILON");
    assert_eq!(gpu[0], gpu[2], "sub-EPSILON clamps to EPSILON");
    assert_eq!(gpu[0], gpu[3], "EPSILON boundary matches the clamp");
}

#[test]
fn gpu_lod_projection_handles_many_queries() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    // More than one workgroup (>64) to exercise the dispatch tiling and
    // per-thread independence. Distances 1..=32 include non-power-of-two
    // divisors, so parity is the <= 1 ULP bound.
    let mut queries: Vec<ProjectedErrorQuery> = Vec::new();
    for k in 0..200i32 {
        let error = f32::from(i16::try_from(k % 16).expect("small range fits i16")) * 0.25;
        let distance = f32::from(i16::try_from(k % 32 + 1).expect("small range fits i16"));
        queries.push(ProjectedErrorQuery {
            geometric_error: error,
            view_distance: distance,
            focal_length_pixels: 1080.0,
        });
    }
    assert_parity(&ctx, &queries);
}

#[test]
fn empty_input_yields_empty_output() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let out = GpuLodProjection::new(&ctx).project(&ctx, &[]);
    assert!(out.is_empty(), "no queries yields no pixel sizes");
}
