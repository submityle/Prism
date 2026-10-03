//! M5 tests: color spaces, deterministic PRNGs, procedural noise, and batched
//! transforms.
//!
//! Tests run with `std` available (the crate is only `no_std` for non-test
//! builds), so `f64`/`std` facilities are used purely as test scaffolding and
//! reference oracles; they never participate in the code paths under test.
//!
//! Coverage:
//! - **Color:** round-trips through every edge of the conversion graph
//!   (`srgb⇄linear`, `linear⇄xyz`, `linear⇄oklab`, `oklab⇄oklch`,
//!   `srgb⇄hsl`, `srgb⇄hsv`), white/black/primary sanity, `u8` quantization,
//!   and color-temperature neutrality.
//! - **PRNG:** per-generator golden anchors (bit-exact known-seed outputs),
//!   determinism, uniform-float/range bounds, boolean extremes, distribution
//!   moments, and geometric samplers (disk/sphere/Gaussian).
//! - **Noise:** determinism, seed sensitivity, range bounds, `C^0` continuity,
//!   and fractal-helper ranges.
//! - **Batch:** AVX2-vs-scalar parity (exact on this scalar-only host),
//!   odd-length tails, and agreement with the per-element `Mat4` reference.

use crate::prelude::*;
use crate::color::transfer::{linear_to_srgb, srgb_to_linear};

// --------------------------------------------------------------------------
// helpers
// --------------------------------------------------------------------------

#[track_caller]
fn close(a: f32, b: f32, eps: f32) {
    assert!((a - b).abs() <= eps, "{a} vs {b} (eps {eps})");
}

fn vec4_close(a: Vec4, b: Vec4, eps: f32) {
    close(a.x, b.x, eps);
    close(a.y, b.y, eps);
    close(a.z, b.z, eps);
    close(a.w, b.w, eps);
}

fn vec3_close(a: Vec3, b: Vec3, eps: f32) {
    close(a.x, b.x, eps);
    close(a.y, b.y, eps);
    close(a.z, b.z, eps);
}

// --------------------------------------------------------------------------
// color: transfer functions
// --------------------------------------------------------------------------

#[test]
fn transfer_fixed_points() {
    close(srgb_to_linear(0.0), 0.0, 1e-7);
    close(srgb_to_linear(1.0), 1.0, 1e-6);
    close(linear_to_srgb(0.0), 0.0, 1e-7);
    close(linear_to_srgb(1.0), 1.0, 1e-6);
    // Mid-gray sRGB 0.5 decodes to roughly linear 0.214.
    close(srgb_to_linear(0.5), 0.214_041_14, 1e-4);
}

#[test]
fn transfer_round_trip() {
    let mut c = 0.0f32;
    while c <= 1.0 {
        let back = linear_to_srgb(srgb_to_linear(c));
        close(back, c, 1e-5);
        c += 1.0 / 256.0;
    }
}

// --------------------------------------------------------------------------
// color: round-trips through the graph
// --------------------------------------------------------------------------

fn sample_srgbs() -> [Srgba; 7] {
    [
        Srgba::new(0.0, 0.0, 0.0, 1.0),
        Srgba::new(1.0, 1.0, 1.0, 1.0),
        Srgba::new(0.75, 0.2, 0.4, 0.5),
        Srgba::new(0.1, 0.6, 0.9, 1.0),
        Srgba::new(0.9, 0.45, 0.05, 0.25),
        Srgba::new(0.3, 0.3, 0.3, 1.0),
        Srgba::new(0.05, 0.95, 0.5, 0.8),
    ]
}

#[test]
fn srgb_linear_round_trip() {
    for c in sample_srgbs() {
        let back = c.to_linear().to_srgb();
        close(back.red, c.red, 1e-5);
        close(back.green, c.green, 1e-5);
        close(back.blue, c.blue, 1e-5);
        close(back.alpha, c.alpha, 1e-7);
    }
}

#[test]
fn linear_xyz_round_trip() {
    for c in sample_srgbs() {
        let lin = c.to_linear();
        let back = LinearRgba::from_xyz(lin.to_xyz());
        close(back.red, lin.red, 1e-4);
        close(back.green, lin.green, 1e-4);
        close(back.blue, lin.blue, 1e-4);
    }
}

#[test]
fn linear_oklab_round_trip() {
    for c in sample_srgbs() {
        let lin = c.to_linear();
        let back = lin.to_oklab().to_linear();
        close(back.red, lin.red, 1e-4);
        close(back.green, lin.green, 1e-4);
        close(back.blue, lin.blue, 1e-4);
    }
}

#[test]
fn oklab_oklch_round_trip() {
    for c in sample_srgbs() {
        let lab = c.to_linear().to_oklab();
        let back = lab.to_oklch().to_oklab();
        close(back.l, lab.l, 1e-5);
        close(back.a, lab.a, 1e-5);
        close(back.b, lab.b, 1e-5);
    }
}

#[test]
fn srgb_hsl_round_trip() {
    for c in sample_srgbs() {
        let back = Hsla::from_srgb(c).to_srgb();
        close(back.red, c.red, 1e-4);
        close(back.green, c.green, 1e-4);
        close(back.blue, c.blue, 1e-4);
        close(back.alpha, c.alpha, 1e-7);
    }
}

#[test]
fn srgb_hsv_round_trip() {
    for c in sample_srgbs() {
        let back = Hsva::from_srgb(c).to_srgb();
        close(back.red, c.red, 1e-4);
        close(back.green, c.green, 1e-4);
        close(back.blue, c.blue, 1e-4);
        close(back.alpha, c.alpha, 1e-7);
    }
}

// --------------------------------------------------------------------------
// color: anchor values
// --------------------------------------------------------------------------

#[test]
fn white_black_anchors() {
    let white_lin = Srgba::WHITE.to_linear();
    close(white_lin.red, 1.0, 1e-6);
    close(white_lin.green, 1.0, 1e-6);
    close(white_lin.blue, 1.0, 1e-6);

    let black_lin = Srgba::BLACK.to_linear();
    close(black_lin.red, 0.0, 1e-7);

    // Linear white maps to the D65 reference white in XYZ.
    let xyz = LinearRgba::WHITE.to_xyz();
    close(xyz.x, Xyza::D65_WHITE.x, 1e-3);
    close(xyz.y, Xyza::D65_WHITE.y, 1e-3);
    close(xyz.z, Xyza::D65_WHITE.z, 1e-3);

    // OkLab lightness of white is 1 and the chroma axes vanish.
    let ok = LinearRgba::WHITE.to_oklab();
    close(ok.l, 1.0, 1e-3);
    close(ok.a, 0.0, 1e-3);
    close(ok.b, 0.0, 1e-3);
}

#[test]
fn srgb_u8_round_trip() {
    for &(r, g, b, a) in &[(0u8, 0u8, 0u8, 255u8), (255, 128, 64, 32), (12, 200, 90, 255)] {
        let c = Srgba::from_u8a(r, g, b, a);
        assert_eq!(c.to_u8_array(), [r, g, b, a]);
    }
    assert_eq!(Srgba::from_u8(10, 20, 30).to_u8_array(), [10, 20, 30, 255]);
}

#[test]
fn temperature_is_neutralish() {
    // D65-ish daylight should be near-neutral: channels within ~15% of each
    // other once normalized by the max channel.
    let c = LinearRgba::from_temperature(6500.0);
    let m = c.red.max(c.green).max(c.blue);
    assert!(m > 0.0);
    assert!(c.red / m > 0.80, "r/m = {}", c.red / m);
    assert!(c.blue / m > 0.70, "b/m = {}", c.blue / m);

    // Warm (low K) biases toward red over blue; cool (high K) the opposite.
    let warm = LinearRgba::from_temperature(2000.0);
    assert!(warm.red > warm.blue);
    let cool = LinearRgba::from_temperature(12000.0);
    assert!(cool.blue > cool.red);

    // Out-of-range inputs clamp instead of exploding.
    assert!(LinearRgba::from_temperature(100.0).is_finite());
    assert!(LinearRgba::from_temperature(1.0e9).is_finite());
}

#[test]
fn linear_lerp_midpoint() {
    let a = LinearRgba::new(0.0, 0.0, 0.0, 0.0);
    let b = LinearRgba::new(1.0, 0.5, 0.25, 1.0);
    let mid = a.lerp(b, 0.5);
    close(mid.red, 0.5, 1e-7);
    close(mid.green, 0.25, 1e-7);
    close(mid.blue, 0.125, 1e-7);
    close(mid.alpha, 0.5, 1e-7);
}

// --------------------------------------------------------------------------
// rng: golden anchors (bit-exact)
// --------------------------------------------------------------------------

#[test]
fn splitmix_golden() {
    let mut r = SplitMix64::new(0);
    assert_eq!(r.next(), 0xE220_A839_7B1D_CDAF);
    assert_eq!(r.next(), 0x6E78_9E6A_A1B9_65F4);
    assert_eq!(r.next(), 0x06C4_5D18_8009_454F);
}

#[test]
fn pcg32_golden() {
    let mut r = Pcg32::new(42);
    assert_eq!(r.next(), 0x21b7_56ee);
    assert_eq!(r.next(), 0xc15e_f750);
    assert_eq!(r.next(), 0x9548_a9bd);
}

#[test]
fn xoshiro_golden() {
    let mut r = Xoshiro256StarStar::new(42);
    assert_eq!(r.next(), 0x1578_0b2e_0c2e_c716);
    assert_eq!(r.next(), 0x6104_d986_6d11_3a7e);
    assert_eq!(r.next(), 0xae17_5332_39e4_99a1);
}

// --------------------------------------------------------------------------
// rng: determinism
// --------------------------------------------------------------------------

#[test]
fn determinism_same_seed() {
    for seed in [0u64, 1, 42, 0xDEAD_BEEF, u64::MAX] {
        let (mut a, mut b) = (SplitMix64::new(seed), SplitMix64::new(seed));
        for _ in 0..64 {
            assert_eq!(a.next(), b.next());
        }
        let (mut a, mut b) = (Pcg32::new(seed), Pcg32::new(seed));
        for _ in 0..64 {
            assert_eq!(a.next(), b.next());
        }
        let (mut a, mut b) = (Xoshiro256StarStar::new(seed), Xoshiro256StarStar::new(seed));
        for _ in 0..64 {
            assert_eq!(a.next(), b.next());
        }
    }
}

#[test]
fn different_seeds_differ() {
    let mut a = Xoshiro256StarStar::new(1);
    let mut b = Xoshiro256StarStar::new(2);
    let mut any_diff = false;
    for _ in 0..16 {
        any_diff |= a.next() != b.next();
    }
    assert!(any_diff);
}

#[test]
fn xoshiro_jump_forks_stream() {
    let mut base = Xoshiro256StarStar::new(7);
    let mut jumped = base.clone();
    jumped.jump();
    // The jumped stream is 2^128 draws ahead, so its near-term outputs should
    // not coincide with the base stream's near-term outputs.
    let base_seq: Vec<u64> = (0..8).map(|_| base.next()).collect();
    let jump_seq: Vec<u64> = (0..8).map(|_| jumped.next()).collect();
    assert_ne!(base_seq, jump_seq);
}

// --------------------------------------------------------------------------
// rng: float / range bounds
// --------------------------------------------------------------------------

#[test]
fn float_bounds() {
    let mut r = Xoshiro256StarStar::new(99);
    for _ in 0..100_000 {
        let f = r.next_f32();
        assert!((0.0..1.0).contains(&f), "f32 {f}");
        let d = r.next_f64();
        assert!((0.0..1.0).contains(&d), "f64 {d}");
    }
}

#[test]
fn range_bounds() {
    let mut r = Pcg32::new(5);
    for _ in 0..50_000 {
        assert!(r.range_u64(1000) < 1000);
        assert!(r.range_u32(37) < 37);
        let i = r.range_i64(-10, 10);
        assert!((-10..10).contains(&i));
        let f = r.range_f32(-2.0, 3.0);
        assert!((-2.0..3.0).contains(&f));
        let d = r.range_f64(100.0, 200.0);
        assert!((100.0..200.0).contains(&d));
    }
    // Degenerate bounds are well-defined.
    assert_eq!(r.range_u64(0), 0);
    assert_eq!(r.range_u32(0), 0);
}

#[test]
fn bool_extremes() {
    let mut r = SplitMix64::new(3);
    for _ in 0..1000 {
        assert!(!r.gen_bool(0.0));
        assert!(r.gen_bool(1.0));
    }
}

// --------------------------------------------------------------------------
// rng: distribution moments
// --------------------------------------------------------------------------

#[test]
fn uniform_moments() {
    let mut r = Xoshiro256StarStar::new(2024);
    let n = 200_000;
    let mut sum = 0.0f64;
    let mut sumsq = 0.0f64;
    for _ in 0..n {
        let x = r.next_f32() as f64;
        sum += x;
        sumsq += x * x;
    }
    let mean = sum / n as f64;
    let var = sumsq / n as f64 - mean * mean;
    assert!((mean - 0.5).abs() < 0.01, "mean {mean}");
    assert!((var - 1.0 / 12.0).abs() < 0.01, "var {var}");
}

#[test]
fn gaussian_moments() {
    let mut r = Xoshiro256StarStar::new(555);
    let n = 200_000;
    let mut sum = 0.0f64;
    let mut sumsq = 0.0f64;
    for _ in 0..n {
        let x = r.gaussian() as f64;
        sum += x;
        sumsq += x * x;
    }
    let mean = sum / n as f64;
    let var = sumsq / n as f64 - mean * mean;
    assert!(mean.abs() < 0.02, "mean {mean}");
    assert!((var - 1.0).abs() < 0.03, "var {var}");
}

#[test]
fn geometric_samplers() {
    let mut r = Xoshiro256StarStar::new(314);
    for _ in 0..50_000 {
        close(r.unit_circle().length(), 1.0, 1e-5);
        assert!(r.in_unit_disk().length() <= 1.0 + 1e-5);
        close(r.unit_sphere().length(), 1.0, 1e-5);
        assert!(r.in_unit_sphere().length() <= 1.0 + 1e-5);
    }
}

// --------------------------------------------------------------------------
// noise: determinism / seed sensitivity
// --------------------------------------------------------------------------

#[test]
fn noise_determinism() {
    let a = Perlin::new(1234);
    let b = Perlin::new(1234);
    let sa = Simplex::new(1234);
    let sb = Simplex::new(1234);
    for i in 0..50 {
        let x = i as f32 * 0.31;
        let y = i as f32 * -0.17 + 1.0;
        let z = i as f32 * 0.07;
        assert_eq!(a.get2(x, y).to_bits(), b.get2(x, y).to_bits());
        assert_eq!(a.get3(x, y, z).to_bits(), b.get3(x, y, z).to_bits());
        assert_eq!(sa.get2(x, y).to_bits(), sb.get2(x, y).to_bits());
        assert_eq!(sa.get3(x, y, z).to_bits(), sb.get3(x, y, z).to_bits());
    }
}

#[test]
fn noise_seed_sensitivity() {
    let a = Perlin::new(1);
    let b = Perlin::new(2);
    let mut diff = false;
    for i in 0..64 {
        let x = i as f32 * 0.5 + 0.25;
        diff |= a.get2(x, x * 0.5).to_bits() != b.get2(x, x * 0.5).to_bits();
    }
    assert!(diff);
}

#[test]
fn noise_range_bounds() {
    let perlin = Perlin::new(77);
    let simplex = Simplex::new(77);
    let tol = 1.0e-3;
    for i in 0..200 {
        for j in 0..200 {
            let x = i as f32 * 0.113 - 7.0;
            let y = j as f32 * 0.091 + 4.0;
            assert!(perlin.get2(x, y).abs() <= 1.0 + tol, "perlin2 {}", perlin.get2(x, y));
            assert!(simplex.get2(x, y).abs() <= 1.0 + tol, "simplex2 {}", simplex.get2(x, y));
            let z = (i + j) as f32 * 0.037 - 3.0;
            assert!(perlin.get3(x, y, z).abs() <= 1.0 + tol, "perlin3 {}", perlin.get3(x, y, z));
            assert!(simplex.get3(x, y, z).abs() <= 1.0 + tol, "simplex3 {}", simplex.get3(x, y, z));
        }
    }
}

#[test]
fn noise_continuity() {
    // A gradient-noise field is Lipschitz, so tiny steps give tiny changes.
    let perlin = Perlin::new(5);
    let simplex = Simplex::new(5);
    let eps = 1.0e-3;
    for i in 0..100 {
        let x = i as f32 * 0.237;
        let y = i as f32 * 0.191 - 2.0;
        assert!((perlin.get2(x, y) - perlin.get2(x + eps, y)).abs() < 0.05);
        assert!((simplex.get2(x, y) - simplex.get2(x, y + eps)).abs() < 0.05);
        let z = i as f32 * 0.123;
        assert!((perlin.get3(x, y, z) - perlin.get3(x, y, z + eps)).abs() < 0.05);
        assert!((simplex.get3(x, y, z) - simplex.get3(x + eps, y, z)).abs() < 0.05);
    }
}

// --------------------------------------------------------------------------
// noise: fractal helpers
// --------------------------------------------------------------------------

#[test]
fn fractal_ranges_and_determinism() {
    let perlin = Perlin::new(9);
    let frac = Fractal::default();
    for i in 0..150 {
        for j in 0..3 {
            let x = i as f32 * 0.1 - 5.0;
            let y = j as f32 * 2.5 + i as f32 * 0.03;

            let fb = frac.fbm2(&perlin, x, y);
            assert!(fb.abs() <= 1.0 + 1e-3, "fbm2 {fb}");
            assert_eq!(fb.to_bits(), frac.fbm2(&perlin, x, y).to_bits());

            let tb = frac.turbulence2(&perlin, x, y);
            assert!((0.0..=1.0 + 1e-3).contains(&tb), "turbulence2 {tb}");

            let rg = frac.ridged2(&perlin, x, y);
            assert!((0.0..=1.0 + 1e-3).contains(&rg), "ridged2 {rg}");

            let fb3 = frac.fbm3(&perlin, x, y, 0.5);
            assert!(fb3.abs() <= 1.0 + 1e-3, "fbm3 {fb3}");
        }
    }
}

// --------------------------------------------------------------------------
// batch: parity and correctness
// --------------------------------------------------------------------------

fn sample_matrix() -> Mat4 {
    // A full affine with scale, shear, and translation (last row stays
    // [0,0,0,1] so points keep w = 1).
    Mat4::from_cols(
        Vec4::new(2.0, 0.5, -0.25, 0.0),
        Vec4::new(-0.5, 3.0, 0.75, 0.0),
        Vec4::new(0.1, -0.2, 4.0, 0.0),
        Vec4::new(1.0, -2.0, 3.0, 1.0),
    )
}

#[test]
fn batch_vec4_parity() {
    let m = sample_matrix();
    let mut rng = Xoshiro256StarStar::new(42);
    // Odd length to exercise the SIMD tail path.
    let n = 101;
    let mut src = Vec::with_capacity(n);
    for _ in 0..n {
        src.push(Vec4::new(
            rng.range_f32(-5.0, 5.0),
            rng.range_f32(-5.0, 5.0),
            rng.range_f32(-5.0, 5.0),
            rng.range_f32(-5.0, 5.0),
        ));
    }
    let mut dst = vec![Vec4::ZERO; n];
    batch::transform_vec4(&m, &src, &mut dst);
    for (i, &v) in src.iter().enumerate() {
        vec4_close(dst[i], batch::transform_vec4_one(&m, v), 1e-4);
    }
}

#[test]
fn batch_points_and_vectors() {
    let m = sample_matrix();
    let src = [Vec3::new(1.0, 1.0, 1.0), Vec3::new(-2.0, 0.5, 3.0), Vec3::new(0.0, 0.0, 0.0)];

    let mut pts = [Vec3::ZERO; 3];
    batch::transform_points3(&m, &src, &mut pts);
    for (i, &p) in src.iter().enumerate() {
        vec3_close(pts[i], m.transform_point3(p), 1e-5);
    }
    // Known value for the pure translation of the origin.
    vec3_close(pts[2], Vec3::new(1.0, -2.0, 3.0), 1e-6);

    let mut vecs = [Vec3::ZERO; 3];
    batch::transform_vectors3(&m, &src, &mut vecs);
    for (i, &v) in src.iter().enumerate() {
        vec3_close(vecs[i], m.transform_vector3(v), 1e-5);
    }
}

#[test]
fn batch_normalize() {
    let src = [
        Vec3::new(3.0, 0.0, 4.0),
        Vec3::new(1.0, 2.0, 2.0),
        Vec3::new(-5.0, 1.0, 0.0),
    ];
    let mut dst = [Vec3::ZERO; 3];
    batch::normalize3(&src, &mut dst);
    for (i, &v) in src.iter().enumerate() {
        vec3_close(dst[i], v.normalize(), 1e-6);
        close(dst[i].length(), 1.0, 1e-6);
    }
}

#[test]
#[should_panic = "length mismatch"]
fn batch_length_mismatch_panics() {
    let m = sample_matrix();
    let src = [Vec4::ZERO; 3];
    let mut dst = [Vec4::ZERO; 2];
    batch::transform_vec4(&m, &src, &mut dst);
}
