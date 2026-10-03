//! Real-device parity for the compact motion-vector encode/pack kernel.
//!
//! Each test builds a batch of [`MotionSample`]s plus a per-pixel flag byte,
//! runs the CPU golden [`encode_sample`] over them, runs the `GPU`
//! [`GpuEncodeMotion`] kernel on a real adapter, and asserts the two agree
//! **bit-for-bit** by comparing whole [`EncodedMotion`] records with
//! `assert_eq!`. Exact equality is sound here because the golden quantiser
//! clamps `NaN` before any `clamp`, rounds with an explicit half-step bias and
//! truncates toward zero (never round-half-even), and its clamp bounds keep
//! every truncating cast inside the integer range -- so no rounding-mode or
//! saturation ambiguity can diverge between host and device. The suite skips
//! gracefully when no adapter is available so it still passes on a device-less
//! CI image, while running the full dispatch on a real `GPU`.

use prism_motion_gpu::context::GpuContext;
use prism_motion_gpu::encode::GpuEncodeMotion;
use prism_render_architecture::motion::MotionSample;
use prism_render_architecture::motion::encode::{
    EncodedMotion, VelocityEncoding, encode_sample, flags,
};

/// Acquires a device, or prints a skip note and returns `None` on hosts without
/// a usable adapter.
#[expect(
    clippy::print_stderr,
    reason = "test-only skip note when no GPU adapter is present"
)]
fn with_gpu() -> Option<GpuContext> {
    match GpuContext::try_headless() {
        Some(ctx) => Some(ctx),
        None => {
            eprintln!("skipping: no usable GPU adapter on this host");
            None
        }
    }
}

/// Deterministic LCG mapped to `f32`, no transcendentals.
struct Lcg {
    state: u32,
}

impl Lcg {
    fn new(seed: u32) -> Lcg {
        Lcg { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        self.state
    }

    /// Uniform-ish `f32` in `[0, 1)` from the high bits.
    fn unit(&mut self) -> f32 {
        let v = self.next_u32() >> 8; // 24 bits of entropy
        (v as f32) / (16_777_216.0_f32)
    }

    /// `f32` in `[-range, range)`.
    fn signed(&mut self, range: f32) -> f32 {
        (self.unit() * 2.0 - 1.0) * range
    }
}

/// Encodes `samples` on the device and compares to the golden element-by-element.
fn assert_encode_parity(
    ctx: &GpuContext,
    kernel: &GpuEncodeMotion,
    samples: &[MotionSample],
    encoding: VelocityEncoding,
    extra_flags: &[u8],
) -> Vec<EncodedMotion> {
    let golden: Vec<EncodedMotion> = samples
        .iter()
        .zip(extra_flags.iter())
        .map(|(&s, &f)| encode_sample(s, encoding, f))
        .collect();
    let gpu = kernel
        .encode(ctx, samples, encoding, extra_flags)
        .expect("matched lengths");
    assert_eq!(gpu.len(), golden.len(), "length mismatch");
    for (i, (g, c)) in gpu.iter().zip(golden.iter()).enumerate() {
        assert_eq!(g, c, "pixel {i}: gpu {g:?} != golden {c:?}");
    }
    gpu
}

fn sample(
    vx: f32,
    vy: f32,
    reactive: f32,
    transparency: f32,
    confidence: f32,
) -> MotionSample {
    MotionSample {
        velocity_pixels: [vx, vy],
        reprojection_confidence: confidence,
        reactive,
        transparency,
        surface_id: 0,
    }
}

#[test]
fn snorm16_endpoints_and_saturation_match_golden() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuEncodeMotion::new(&ctx);

    let encoding = VelocityEncoding::new(100.0);
    // Full-scale, beyond full-scale (saturates), zero, and a mid value.
    let samples = [
        sample(100.0, -100.0, 0.0, 0.0, 1.0),  // +/- full scale
        sample(1000.0, -1000.0, 0.0, 0.0, 1.0), // saturates to +/-32767
        sample(0.0, 0.0, 0.0, 0.0, 1.0),        // zero -> 0
        sample(25.0, -75.0, 0.0, 0.0, 1.0),     // interior
    ];
    let flags_in = [0u8; 4];
    let gpu = assert_encode_parity(&ctx, &kernel, &samples, encoding, &flags_in);

    // Anti-vacuous: full-scale maps to +/-32767 and zero stays zero.
    assert_eq!(gpu[0].velocity, [32767, -32767]);
    assert_eq!(gpu[1].velocity, [32767, -32767]);
    assert_eq!(gpu[2].velocity, [0, 0]);
}

#[test]
fn nan_velocity_resolves_to_zero_like_golden() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuEncodeMotion::new(&ctx);

    let encoding = VelocityEncoding::new(64.0);
    let samples = [
        sample(f32::NAN, 10.0, 0.0, 0.0, 1.0),
        sample(-20.0, f32::NAN, 0.0, 0.0, 1.0),
    ];
    let flags_in = [0u8; 2];
    let gpu = assert_encode_parity(&ctx, &kernel, &samples, encoding, &flags_in);

    // Anti-vacuous: the NaN axis lands on 0 while the finite axis is nonzero.
    assert_eq!(gpu[0].velocity[0], 0);
    assert_ne!(gpu[0].velocity[1], 0);
    assert_eq!(gpu[1].velocity[1], 0);
    assert_ne!(gpu[1].velocity[0], 0);
}

#[test]
fn mask_channels_and_flags_match_golden() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuEncodeMotion::new(&ctx);

    let encoding = VelocityEncoding::default();
    // Sweep the unorm8 channels across representative levels and set the
    // transparency so the golden forces the TRANSPARENT flag.
    let samples = [
        sample(0.0, 0.0, 0.0, 0.0, 0.0),
        sample(0.0, 0.0, 0.25, 0.0, 0.75),
        sample(0.0, 0.0, 0.5, 0.5, 0.5),
        sample(0.0, 0.0, 0.75, 1.0, 0.25),
        sample(0.0, 0.0, 1.0, 0.4, 0.0),
    ];
    // Caller-supplied extra flags must survive in the top byte.
    let flags_in = [
        0u8,
        flags::DISOCCLUDED,
        flags::STREAMING_REVEAL,
        0u8,
        flags::DISOCCLUDED | flags::STREAMING_REVEAL,
    ];
    let gpu = assert_encode_parity(&ctx, &kernel, &samples, encoding, &flags_in);

    // Anti-vacuous: transparency > 0 auto-sets TRANSPARENT; the opaque first
    // pixel keeps it clear; caller flags are preserved.
    assert_eq!(gpu[0].masks.flags() & flags::TRANSPARENT, 0);
    assert_ne!(gpu[2].masks.flags() & flags::TRANSPARENT, 0);
    assert_ne!(gpu[3].masks.flags() & flags::TRANSPARENT, 0);
    assert_ne!(gpu[1].masks.flags() & flags::DISOCCLUDED, 0);
    assert_ne!(gpu[4].masks.flags() & flags::STREAMING_REVEAL, 0);
}

#[test]
fn length_mismatch_returns_none_and_empty_is_empty() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuEncodeMotion::new(&ctx);

    let samples = [sample(1.0, 2.0, 0.0, 0.0, 1.0)];
    // Mismatched flag length -> None.
    assert!(
        kernel
            .encode(&ctx, &samples, VelocityEncoding::default(), &[])
            .is_none(),
        "length mismatch must return None"
    );
    // Empty input -> empty output.
    let empty = kernel
        .encode(&ctx, &[], VelocityEncoding::default(), &[])
        .expect("empty matched lengths");
    assert!(empty.is_empty(), "empty input must yield empty output");
}

#[test]
fn large_multi_workgroup_batch_matches_golden() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuEncodeMotion::new(&ctx);

    // 211 pixels spans several 64-wide workgroups (and a partial final group).
    let count = 211;
    let encoding = VelocityEncoding::new(256.0);
    let mut rng = Lcg::new(0x1234_5678);
    let samples: Vec<MotionSample> = (0..count)
        .map(|_| {
            sample(
                rng.signed(400.0), // spans saturation on both sides
                rng.signed(400.0),
                rng.unit(),
                rng.unit(),
                rng.unit(),
            )
        })
        .collect();
    let extra_flags: Vec<u8> = (0..count)
        .map(|i| if i % 7 == 0 { flags::DISOCCLUDED } else { 0 })
        .collect();

    let gpu = assert_encode_parity(&ctx, &kernel, &samples, encoding, &extra_flags);

    // Anti-vacuous: at least one nonzero velocity and at least one pixel with a
    // flag bit set in the packed top byte.
    assert!(
        gpu.iter().any(|e| e.velocity != [0, 0]),
        "all velocities were zero"
    );
    assert!(
        gpu.iter().any(|e| e.masks.flags() != 0),
        "no flag bit was ever set"
    );
}
