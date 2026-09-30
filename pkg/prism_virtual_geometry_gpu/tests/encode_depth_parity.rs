//! Real-device parity for the vis-buffer depth-key encode twin:
//! [`GpuEncodeDepth`] must reproduce the CPU golden
//! [`encode_depth`](prism_render_architecture::virtual_geometry::encode_depth)
//! for every depth — `bitcast<u32>(clamp(depth, 0, 1))`.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The encode is bit-exact: `clamp` is comparison + select (no rounding) and
//! `bitcast` copies the bit pattern, so every finite non-negative depth returns
//! unchanged, every strictly negative depth clamps to `+0.0`, and every value
//! above `1.0` (including `+inf`) clamps to `1.0`. The suite asserts exact key
//! equality (`.to_bits()`).
//!
//! `NaN`, `-0.0` and subnormals are deliberately excluded: WGSL `min`/`max`
//! NaN handling and the `max(-0.0, +0.0)` sign are backend-indeterminate, and
//! GPUs may flush subnormals to zero, whereas the rasterizer only feeds finite,
//! normal, non-negative-zero depths, so the parity domain matches the golden's
//! real contract.
//!
//! Provenance: standard reversed-Z depth-key encode for atomicMax vis-buffer
//! compositing; no Unreal Engine source or derived code.

use prism_render_architecture::virtual_geometry::encode_depth;
use prism_virtual_geometry_gpu::{GpuContext, GpuEncodeDepth};

/// Asserts the twin matches the golden bit-for-bit for every depth.
fn assert_bit_exact(ctx: &GpuContext, depths: &[f32]) {
    let gpu = GpuEncodeDepth::new(ctx).encode(ctx, depths);
    assert_eq!(gpu.len(), depths.len(), "one key per depth");
    for (i, &d) in depths.iter().enumerate() {
        let expected = encode_depth(d);
        assert_eq!(
            gpu[i], expected,
            "depth key must be bit-exact for depth {d} (query {i}): gpu {:#010x}, cpu {:#010x}",
            gpu[i], expected,
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_encode_depth_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping encode-depth parity: no wgpu adapter on this host");
        return;
    };
    // Interior [0,1] identity (dyadic and non-dyadic — clamp copies bits with
    // no rounding), strictly-negative clamp-to-+0.0, and above-1.0 (incl. +inf)
    // clamp-to-1.0. No NaN, no -0.0.
    let depths = [
        0.0f32,
        1.0,
        0.5,
        0.25,
        0.125,
        0.75,
        0.3,
        0.7,
        0.999_999_9,
        0.000_001,
        f32::from_bits(0x3f7f_ffff), // largest f32 strictly below 1.0
        -0.5,
        -1.0,
        -1.0e30,
        -f32::MAX,
        2.0,
        1.5,
        1.000_001,
        1.0e30,
        f32::MAX,
        f32::INFINITY,
    ];
    assert_bit_exact(&ctx, &depths);

    // Spot-check the exact keys the contract must produce.
    let gpu = GpuEncodeDepth::new(&ctx).encode(&ctx, &depths);
    assert_eq!(gpu[0], 0.0f32.to_bits(), "0.0 -> farthest / cleared key 0");
    assert_eq!(gpu[1], 1.0f32.to_bits(), "1.0 -> nearest key");
    assert_eq!(gpu[11], 0.0f32.to_bits(), "-0.5 clamps to +0.0");
    assert_eq!(gpu[15], 1.0f32.to_bits(), "2.0 clamps to 1.0");
    assert_eq!(gpu[20], 1.0f32.to_bits(), "+inf clamps to 1.0");
    // Monotonicity of the key over the covered [0,1] samples: 0.0 < 0.5 < 1.0.
    assert!(gpu[0] < gpu[2] && gpu[2] < gpu[1], "keys monotone in depth");
}

#[test]
fn gpu_encode_depth_handles_many_depths() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    // More than one workgroup (>64) to exercise the dispatch tiling and
    // per-thread independence. A ramp across [0,1] plus periodic out-of-range
    // values that clamp to the boundaries.
    let mut depths: Vec<f32> = Vec::new();
    for k in 0..300i32 {
        let t = f32::from(i16::try_from(k).expect("small range fits i16")) / 299.0;
        depths.push(t);
        if k % 5 == 0 {
            depths.push(-t); // strictly negative (t>0) -> +0.0; -0.0 at k==0 avoided below
        }
        if k % 7 == 0 {
            depths.push(1.0 + t); // above 1.0 -> 1.0
        }
    }
    // Drop the single k==0 `-0.0` (t==0) that the ramp produced, since -0.0 is
    // outside the parity domain; replace with a safe strictly-negative value.
    for d in &mut depths {
        if d.to_bits() == (-0.0f32).to_bits() {
            *d = -0.25;
        }
    }
    assert_bit_exact(&ctx, &depths);
}

#[test]
fn empty_input_yields_empty_output() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let out = GpuEncodeDepth::new(&ctx).encode(&ctx, &[]);
    assert!(out.is_empty(), "no depths yields no keys");
}
