//! Cross-check tests: assert the compile-time-selected SIMD backend matches the
//! [`scalar`](super::scalar) reference within a tight tolerance over many
//! randomized inputs. On scalar-only targets this degenerates to an identity
//! check, which is still a useful smoke test of the dispatch layer.

use super::{scalar, Backend};

/// Tiny deterministic LCG so the tests need no external `rand` dependency.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    /// Next `f32` uniformly in `[-1, 1)`.
    fn next_f32(&mut self) -> f32 {
        // Numerical Recipes LCG constants.
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let bits = (self.0 >> 40) as u32; // 24 high-quality bits
        let unit = (bits as f32) / ((1u32 << 24) as f32); // [0, 1)
        unit * 2.0 - 1.0
    }
    fn vec4(&mut self) -> [f32; 4] {
        [self.next_f32(), self.next_f32(), self.next_f32(), self.next_f32()]
    }
}

const TOL: f32 = 2.0e-4;

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() <= TOL * (1.0 + a.abs().max(b.abs()))
}

fn close4(a: [f32; 4], b: [f32; 4]) -> bool {
    (0..4).all(|i| close(a[i], b[i]))
}

fn unit4(v: [f32; 4]) -> [f32; 4] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]).sqrt();
    let inv = if n > 1.0e-6 { 1.0 / n } else { 1.0 };
    [v[0] * inv, v[1] * inv, v[2] * inv, v[3] * inv]
}

#[test]
fn active_backend_matches_target() {
    let b = super::active();
    if cfg!(target_arch = "aarch64") {
        assert_eq!(b, Backend::Neon);
    } else if cfg!(target_arch = "x86_64") {
        assert!(matches!(b, Backend::Sse2 | Backend::Avx2));
    } else {
        assert_eq!(b, Backend::Scalar);
    }
}

#[test]
fn vec4_arith_matches_scalar() {
    let mut r = Lcg::new(0x1234_5678);
    for _ in 0..2000 {
        let a = r.vec4();
        let b = r.vec4();
        let s = r.next_f32();
        assert!(close4(super::vec4_add(a, b), scalar::vec4_add(a, b)));
        assert!(close4(super::vec4_sub(a, b), scalar::vec4_sub(a, b)));
        assert!(close4(super::vec4_mul(a, b), scalar::vec4_mul(a, b)));
        assert!(close4(super::vec4_scale(a, s), scalar::vec4_scale(a, s)));
        // Divisor kept away from zero to avoid inf/nan noise.
        let d = [b[0] + 1.5, b[1] + 1.5, b[2] + 1.5, b[3] + 1.5];
        assert!(close4(super::vec4_div(a, d), scalar::vec4_div(a, d)));
    }
}

#[test]
fn vec4_reductions_match_scalar() {
    let mut r = Lcg::new(0x9e37_79b9);
    for _ in 0..2000 {
        let a = r.vec4();
        let b = r.vec4();
        assert!(close(super::vec4_dot(a, b), scalar::vec4_dot(a, b)));
        assert!(close(super::vec4_length(a), scalar::vec4_length(a)));
        // Keep away from the origin so normalize is well conditioned.
        let a = [a[0] + 2.0, a[1] - 2.0, a[2] + 2.0, a[3] - 2.0];
        assert!(close4(super::vec4_normalize(a), scalar::vec4_normalize(a)));
    }
}

#[test]
fn vec3_ops_match_scalar() {
    let mut r = Lcg::new(0xdead_beef);
    for _ in 0..2000 {
        // Pack as [x, y, z, 0] like the Vec3A facade.
        let a = r.vec4();
        let a = [a[0], a[1], a[2], 0.0];
        let b = r.vec4();
        let b = [b[0], b[1], b[2], 0.0];
        assert!(close(super::vec3_dot(a, b), scalar::vec3_dot(a, b)));
        assert!(close(super::vec3_length(a), scalar::vec3_length(a)));
        let a = [a[0] + 2.0, a[1] - 2.0, a[2] + 2.0, 0.0];
        assert!(close4(super::vec3_normalize(a), scalar::vec3_normalize(a)));
    }
}

#[test]
fn vec3_dot_ignores_padding_lane() {
    // The 3-lane dot must not read lane 3 even when it is garbage.
    let mut r = Lcg::new(0x00c0_ffee);
    for _ in 0..500 {
        let a = r.vec4();
        let b = r.vec4();
        let clean = super::vec3_dot([a[0], a[1], a[2], 0.0], [b[0], b[1], b[2], 0.0]);
        let dirty = super::vec3_dot([a[0], a[1], a[2], 7.0], [b[0], b[1], b[2], -3.0]);
        assert!(close(clean, dirty));
    }
}

#[test]
fn mat4_mul_matches_scalar() {
    let mut r = Lcg::new(0x0bad_f00d);
    for _ in 0..1000 {
        let a = [r.vec4(), r.vec4(), r.vec4(), r.vec4()];
        let b = [r.vec4(), r.vec4(), r.vec4(), r.vec4()];
        let got = super::mat4_mul(&a, &b);
        let want = scalar::mat4_mul(&a, &b);
        for i in 0..4 {
            assert!(close4(got[i], want[i]));
        }
        let v = r.vec4();
        assert!(close4(super::mat4_mul_vec4(&a, v), scalar::mat4_mul_vec4(&a, v)));
    }
}

#[test]
fn quat_mul_matches_scalar() {
    let mut r = Lcg::new(0xfeed_face);
    for _ in 0..2000 {
        let a = r.vec4();
        let b = r.vec4();
        assert!(close4(super::quat_mul(a, b), scalar::quat_mul(a, b)));
    }
}

#[test]
fn quat_mul_vec3_matches_scalar() {
    let mut r = Lcg::new(0xcafe_d00d);
    for _ in 0..2000 {
        // Rotation path is only equivalent for unit quaternions.
        let q = unit4(r.vec4());
        let v = r.vec4();
        let v = [v[0], v[1], v[2], 0.0];
        assert!(close4(super::quat_mul_vec3(q, v), scalar::quat_mul_vec3(q, v)));
    }
}
