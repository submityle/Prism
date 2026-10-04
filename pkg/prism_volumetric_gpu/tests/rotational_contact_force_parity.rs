//! Real-device parity for the rotational discrete-element contact twin:
//! [`GpuRotationalContactForce`](prism_volumetric_gpu::rotational_contact_force::GpuRotationalContactForce)
//! must reproduce the `CPU` golden `rotational_contact_between` of
//! `prism_physics_core::collider::rotational_contact` for one explicit sphere
//! contact pair. The golden evaluates a normal penalty force, a Cundall–Strack
//! tangential-history friction force clamped to the Coulomb cone, and a rolling
//! resistance couple clamped to `μ_r · R_r · F_n`, advancing the two persistent
//! springs; this suite checks the ported kernel against an independent host
//! oracle on a real device.
//!
//! The oracle here is a self-contained `f32` re-implementation of that closed
//! form — its own `[f32; 3]` `dot` / `cross` / `add` / `sub` / `scale` /
//! `length`, then the golden `if`-branch structure replayed in golden operator
//! order — so the test never imports `prism_render_architecture` or
//! `prism_physics_core`.
//!
//! The fixtures cover an overlapping pair with linear and angular motion and
//! finite springs, two no-op degeneracies (coincident centres and
//! non-overlapping spheres, both echoing the incoming springs), a tangential
//! contact that reaches the Coulomb cap (sliding) and one that stays below it,
//! a rolling couple that reaches its cap and one below it, a batch of two or
//! more elements mixing valid and no-op pairs to validate the `std430` stride,
//! and an empty batch the host short-circuits with no dispatch. A `512`-sample
//! `LCG` sweep over random overlapping pairs follows.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Both the reference and the device evaluate the law in `f32`; the host oracle
//! replays the exact golden branch structure in `f32`, so the comparison is
//! against the same closed form. Each continuous quantity (the three force and
//! torque vectors, the two advanced springs, and the four scalar magnitudes) is
//! compared with `abs <= 1e-4 || rel <= 1e-3` (`REL_FLOOR = 1e-6`); the discrete
//! `valid` and `sliding` flags are compared exactly, and the sweep keeps both
//! Coulomb clamps well away from their caps so the discrete sliding / clamp
//! decisions cannot be flipped by `f32` rounding.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::rotational_contact`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::rotational_contact_force::{
    GpuRotationalContactForce, RotationalContactForceQuery, RotationalContactForceResult,
};
use prism_volumetric_gpu::GpuContext;

/// Relative-tolerance floor, so the relative test never divides by a magnitude
/// smaller than this.
const REL_FLOOR: f32 = 1.0e-6;

/// A minimal self-contained 3-vector so the oracle re-derives the golden
/// closed form without importing any engine crate or `glam`.
#[derive(Clone, Copy)]
struct V3 {
    x: f32,
    y: f32,
    z: f32,
}

impl V3 {
    const ZERO: V3 = V3 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    fn from(a: [f32; 3]) -> V3 {
        V3 {
            x: a[0],
            y: a[1],
            z: a[2],
        }
    }

    fn arr(self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }

    fn add(self, o: V3) -> V3 {
        V3 {
            x: self.x + o.x,
            y: self.y + o.y,
            z: self.z + o.z,
        }
    }

    fn sub(self, o: V3) -> V3 {
        V3 {
            x: self.x - o.x,
            y: self.y - o.y,
            z: self.z - o.z,
        }
    }

    fn scale(self, s: f32) -> V3 {
        V3 {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
    }

    fn dot(self, o: V3) -> f32 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    fn cross(self, o: V3) -> V3 {
        V3 {
            x: self.y * o.z - self.z * o.y,
            y: self.z * o.x - self.x * o.z,
            z: self.x * o.y - self.y * o.x,
        }
    }

    fn length(self) -> f32 {
        self.dot(self).sqrt()
    }
}

/// The full oracle output: the public result fields plus the four pre-clamp
/// diagnostics the sweep uses to reject samples that sit on a Coulomb knee.
struct Oracle {
    valid: bool,
    force_on_b: [f32; 3],
    torque_on_a: [f32; 3],
    torque_on_b: [f32; 3],
    spring_tangential_out: [f32; 3],
    spring_rolling_out: [f32; 3],
    overlap: f32,
    normal_magnitude: f32,
    tangential_magnitude: f32,
    rolling_magnitude: f32,
    sliding: bool,
    tmag0: f32,
    max_friction: f32,
    rmag0: f32,
    max_rolling: f32,
}

/// Independent host oracle: replays the golden `rotational_contact_between`
/// `if`-branch structure in `f32`, returning the full contact outcome plus the
/// pre-clamp cap diagnostics.
fn oracle(q: &RotationalContactForceQuery) -> Oracle {
    let pos_a = V3::from(q.pos_a);
    let pos_b = V3::from(q.pos_b);
    let vel_a = V3::from(q.vel_a);
    let vel_b = V3::from(q.vel_b);
    let omega_a = V3::from(q.omega_a);
    let omega_b = V3::from(q.omega_b);
    let spring_t = V3::from(q.spring_tangential);
    let spring_r = V3::from(q.spring_rolling);
    let k_n = q.normal_stiffness;
    let gamma_n = q.normal_damping;
    let k_t = q.tangential_stiffness;
    let gamma_t = q.tangential_damping;
    let friction = q.friction;
    let k_r = q.rolling_stiffness;
    let gamma_r = q.rolling_damping;
    let rolling_friction = q.rolling_friction;
    let rad_a = q.rad_a;
    let rad_b = q.rad_b;
    let dt = q.dt;

    let invalid = Oracle {
        valid: false,
        force_on_b: [0.0; 3],
        torque_on_a: [0.0; 3],
        torque_on_b: [0.0; 3],
        spring_tangential_out: q.spring_tangential,
        spring_rolling_out: q.spring_rolling,
        overlap: 0.0,
        normal_magnitude: 0.0,
        tangential_magnitude: 0.0,
        rolling_magnitude: 0.0,
        sliding: false,
        tmag0: 0.0,
        max_friction: 0.0,
        rmag0: 0.0,
        max_rolling: 0.0,
    };

    let delta = pos_b.sub(pos_a);
    let distance = delta.length();
    if distance <= 0.0 {
        return invalid;
    }
    let overlap = (rad_a + rad_b) - distance;
    if overlap <= 0.0 {
        return invalid;
    }
    let normal = delta.scale(1.0 / distance);

    let arm_a = rad_a - 0.5 * overlap;
    let arm_b = rad_b - 0.5 * overlap;
    let r_a = normal.scale(arm_a);
    let r_b = normal.scale(-arm_b);

    let surf_a = vel_a.add(omega_a.cross(r_a));
    let surf_b = vel_b.add(omega_b.cross(r_b));
    let rel = surf_b.sub(surf_a);

    let v_n = rel.dot(normal);
    let normal_force = (k_n * overlap - gamma_n * v_n).max(0.0);

    let v_t = rel.sub(normal.scale(v_n));
    let mut tangential = spring_t.sub(normal.scale(spring_t.dot(normal)));
    tangential = tangential.add(v_t.scale(dt));
    let mut tangential_force = tangential.scale(-k_t).sub(v_t.scale(gamma_t));
    let mut tangential_magnitude = tangential_force.length();
    let tmag0 = tangential_magnitude;
    let max_friction = friction * normal_force;
    let mut sliding = false;
    if tangential_magnitude > max_friction {
        if tangential_magnitude > 0.0 {
            let direction = tangential_force.scale(1.0 / tangential_magnitude);
            tangential_force = direction.scale(max_friction);
            tangential = tangential_force.scale(-1.0 / k_t);
        } else {
            tangential_force = V3::ZERO;
            tangential = V3::ZERO;
        }
        tangential_magnitude = max_friction;
        sliding = max_friction > 0.0;
    }

    let force_on_b = normal.scale(normal_force).add(tangential_force);
    let mut torque_on_b = r_b.cross(force_on_b);
    let mut torque_on_a = r_a.cross(force_on_b.scale(-1.0));

    let omega_rel = omega_a.sub(omega_b);
    let rolling_radius = (rad_a * rad_b) / (rad_a + rad_b);
    let mut rolling = spring_r.add(omega_rel.scale(dt));
    let mut rolling_torque = rolling.scale(-k_r).sub(omega_rel.scale(gamma_r));
    let mut rolling_magnitude = rolling_torque.length();
    let rmag0 = rolling_magnitude;
    let max_rolling = rolling_friction * rolling_radius * normal_force;
    if rolling_magnitude > max_rolling {
        if rolling_magnitude > 0.0 {
            let direction = rolling_torque.scale(1.0 / rolling_magnitude);
            rolling_torque = direction.scale(max_rolling);
            rolling = rolling_torque.scale(-1.0 / k_r);
        } else {
            rolling_torque = V3::ZERO;
            rolling = V3::ZERO;
        }
        rolling_magnitude = max_rolling;
    }

    torque_on_a = torque_on_a.add(rolling_torque);
    torque_on_b = torque_on_b.sub(rolling_torque);

    Oracle {
        valid: true,
        force_on_b: force_on_b.arr(),
        torque_on_a: torque_on_a.arr(),
        torque_on_b: torque_on_b.arr(),
        spring_tangential_out: tangential.arr(),
        spring_rolling_out: rolling.arr(),
        overlap,
        normal_magnitude: normal_force,
        tangential_magnitude,
        rolling_magnitude,
        sliding,
        tmag0,
        max_friction,
        rmag0,
        max_rolling,
    }
}

/// Mixed absolute-or-relative closeness for a continuous quantity.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= 1.0e-4 {
        return true;
    }
    diff <= 1.0e-3 * a.abs().max(b.abs()).max(REL_FLOOR)
}

/// Component-wise closeness for a 3-vector.
fn close3(a: [f32; 3], b: [f32; 3]) -> bool {
    close(a[0], b[0]) && close(a[1], b[1]) && close(a[2], b[2])
}

/// Asserts a single `GPU` result matches the independent oracle: the discrete
/// `valid` and `sliding` flags exactly, and every continuous quantity to
/// tolerance when valid; a no-op pair zeroes the forces and echoes the springs.
fn assert_parity(gpu: &RotationalContactForceResult, q: &RotationalContactForceQuery, label: &str) {
    let o = oracle(q);
    assert_eq!(gpu.valid, o.valid, "{label}: valid flag mismatch");
    if o.valid {
        assert_eq!(gpu.sliding, o.sliding, "{label}: sliding flag mismatch");
        assert!(
            close3(gpu.force_on_b, o.force_on_b),
            "{label}: force_on_b mismatch gpu={:?} oracle={:?}",
            gpu.force_on_b,
            o.force_on_b
        );
        assert!(
            close3(gpu.torque_on_a, o.torque_on_a),
            "{label}: torque_on_a mismatch gpu={:?} oracle={:?}",
            gpu.torque_on_a,
            o.torque_on_a
        );
        assert!(
            close3(gpu.torque_on_b, o.torque_on_b),
            "{label}: torque_on_b mismatch gpu={:?} oracle={:?}",
            gpu.torque_on_b,
            o.torque_on_b
        );
        assert!(
            close3(gpu.spring_tangential_out, o.spring_tangential_out),
            "{label}: spring_tangential_out mismatch gpu={:?} oracle={:?}",
            gpu.spring_tangential_out,
            o.spring_tangential_out
        );
        assert!(
            close3(gpu.spring_rolling_out, o.spring_rolling_out),
            "{label}: spring_rolling_out mismatch gpu={:?} oracle={:?}",
            gpu.spring_rolling_out,
            o.spring_rolling_out
        );
        assert!(
            close(gpu.overlap, o.overlap),
            "{label}: overlap mismatch gpu={} oracle={}",
            gpu.overlap,
            o.overlap
        );
        assert!(
            close(gpu.normal_magnitude, o.normal_magnitude),
            "{label}: normal_magnitude mismatch gpu={} oracle={}",
            gpu.normal_magnitude,
            o.normal_magnitude
        );
        assert!(
            close(gpu.tangential_magnitude, o.tangential_magnitude),
            "{label}: tangential_magnitude mismatch gpu={} oracle={}",
            gpu.tangential_magnitude,
            o.tangential_magnitude
        );
        assert!(
            close(gpu.rolling_magnitude, o.rolling_magnitude),
            "{label}: rolling_magnitude mismatch gpu={} oracle={}",
            gpu.rolling_magnitude,
            o.rolling_magnitude
        );
    } else {
        assert!(!gpu.sliding, "{label}: a no-op pair must not slide");
        assert_eq!(gpu.force_on_b, [0.0; 3], "{label}: no-op force");
        assert_eq!(gpu.torque_on_a, [0.0; 3], "{label}: no-op torque_on_a");
        assert_eq!(gpu.torque_on_b, [0.0; 3], "{label}: no-op torque_on_b");
        assert_eq!(gpu.overlap, 0.0, "{label}: no-op overlap");
        assert_eq!(gpu.normal_magnitude, 0.0, "{label}: no-op normal");
        assert_eq!(gpu.tangential_magnitude, 0.0, "{label}: no-op tangential");
        assert_eq!(gpu.rolling_magnitude, 0.0, "{label}: no-op rolling");
        assert_eq!(
            gpu.spring_tangential_out, q.spring_tangential,
            "{label}: no-op must echo tangential spring"
        );
        assert_eq!(
            gpu.spring_rolling_out, q.spring_rolling,
            "{label}: no-op must echo rolling spring"
        );
    }
}

/// Model parameter array `[kₙ, γₙ, k_t, γ_t, μ, k_r, γ_r, μ_r]`.
fn model(
    k_n: f32,
    gamma_n: f32,
    k_t: f32,
    gamma_t: f32,
    friction: f32,
    k_r: f32,
    gamma_r: f32,
    rolling_friction: f32,
) -> [f32; 8] {
    [
        k_n,
        gamma_n,
        k_t,
        gamma_t,
        friction,
        k_r,
        gamma_r,
        rolling_friction,
    ]
}

#[test]
fn overlapping_pair_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRotationalContactForce::new(&ctx);
    // Centres 1.5 apart, radii 1 each: overlap 0.5. High friction so the
    // tangential force sticks (no sliding), finite springs and spins.
    let q = RotationalContactForceQuery::new(
        model(1.0e3, 5.0, 1.0e3, 2.0, 1.0, 1.0e3, 1.0, 0.5),
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        1.0,
        1.0,
        [0.1, 0.2, 0.0],
        [-0.1, 0.0, 0.3],
        [0.0, 0.0, 0.5],
        [0.0, 0.4, 0.0],
        [0.01, -0.02, 0.0],
        [0.0, 0.01, -0.01],
        1.0 / 120.0,
    );
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(out[0].valid);
    assert_parity(&out[0], &q, "overlapping");
}

#[test]
fn coincident_centres_is_noop() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRotationalContactForce::new(&ctx);
    // Identical centres -> distance 0 -> no-op, springs echoed.
    let q = RotationalContactForceQuery::new(
        model(1.0e3, 0.0, 1.0e3, 0.0, 0.5, 1.0e3, 0.0, 0.3),
        [0.4, -0.2, 0.1],
        [0.4, -0.2, 0.1],
        1.0,
        1.0,
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.03, 0.02, 0.01],
        [-0.01, 0.02, 0.0],
        1.0 / 120.0,
    );
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(!out[0].valid);
    assert_parity(&out[0], &q, "coincident");
}

#[test]
fn non_overlapping_is_noop() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRotationalContactForce::new(&ctx);
    // Centres 3 apart, radii 1 each: overlap -1 -> no-op, springs echoed.
    let q = RotationalContactForceQuery::new(
        model(1.0e3, 0.0, 1.0e3, 0.0, 0.5, 1.0e3, 0.0, 0.3),
        [0.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        1.0,
        1.0,
        [0.5, 0.5, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.02, -0.03, 0.04],
        [0.01, 0.0, -0.02],
        1.0 / 120.0,
    );
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(!out[0].valid);
    assert_parity(&out[0], &q, "non_overlapping");
}

#[test]
fn tangential_reaches_coulomb_cap() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRotationalContactForce::new(&ctx);
    // Normal force 1e3*0.5=500, max_friction=0.1*500=50. A pure tangential
    // relative velocity drives tf0=100 > 50, so the cone clamp bites (sliding).
    let q = RotationalContactForceQuery::new(
        model(1.0e3, 0.0, 1.0e3, 0.0, 0.1, 1.0e3, 0.0, 0.3),
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        1.0,
        1.0,
        [0.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        0.01,
    );
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(out[0].valid);
    assert!(out[0].sliding, "the tangential force should reach the cone");
    assert_parity(&out[0], &q, "tangential_cap");
}

#[test]
fn tangential_below_cap_sticks() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRotationalContactForce::new(&ctx);
    // Same drive but friction=1.0 -> max_friction=500 > tf0=100, so it sticks.
    let q = RotationalContactForceQuery::new(
        model(1.0e3, 0.0, 1.0e3, 0.0, 1.0, 1.0e3, 0.0, 0.3),
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        1.0,
        1.0,
        [0.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        0.01,
    );
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(out[0].valid);
    assert!(
        !out[0].sliding,
        "the tangential force should stay inside the cone"
    );
    assert_parity(&out[0], &q, "tangential_stick");
}

#[test]
fn rolling_reaches_cap() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRotationalContactForce::new(&ctx);
    // Reduced radius 0.5, normal force 500, rolling_friction 0.01 ->
    // max_rolling 2.5; a spin of omega_a.z=5 drives rt0=50 >> 2.5 (capped).
    let q = RotationalContactForceQuery::new(
        model(1.0e3, 0.0, 1.0e3, 0.0, 1.0, 1.0e3, 0.0, 0.01),
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        1.0,
        1.0,
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 5.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        0.01,
    );
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(out[0].valid);
    let o = oracle(&q);
    assert!(
        o.rmag0 > o.max_rolling,
        "fixture must drive the rolling cap"
    );
    assert_parity(&out[0], &q, "rolling_cap");
}

#[test]
fn rolling_below_cap() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRotationalContactForce::new(&ctx);
    // Same spin but a large rolling_friction keeps the couple below its cap.
    let q = RotationalContactForceQuery::new(
        model(1.0e3, 0.0, 1.0e3, 0.0, 1.0, 1.0e3, 0.0, 1.0),
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        1.0,
        1.0,
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.5],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        0.01,
    );
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(out[0].valid);
    let o = oracle(&q);
    assert!(
        o.rmag0 <= o.max_rolling,
        "fixture must stay below the rolling cap"
    );
    assert_parity(&out[0], &q, "rolling_below");
}

#[test]
fn batch_mixes_valid_and_noop_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRotationalContactForce::new(&ctx);
    let overlapping = RotationalContactForceQuery::new(
        model(1.0e3, 5.0, 1.0e3, 2.0, 0.5, 1.0e3, 1.0, 0.3),
        [0.0, 0.0, 0.0],
        [1.4, 0.3, 0.0],
        1.0,
        1.0,
        [0.2, 0.0, 0.1],
        [0.0, -0.2, 0.0],
        [0.0, 0.0, 1.0],
        [0.1, 0.0, 0.0],
        [0.01, 0.0, -0.02],
        [0.0, 0.02, 0.0],
        1.0 / 90.0,
    );
    let coincident = RotationalContactForceQuery::new(
        model(1.0e3, 0.0, 1.0e3, 0.0, 0.5, 1.0e3, 0.0, 0.3),
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        1.0,
        1.0,
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.05, -0.05, 0.0],
        [0.0, 0.03, 0.0],
        1.0 / 90.0,
    );
    let far = RotationalContactForceQuery::new(
        model(1.0e3, 0.0, 1.0e3, 0.0, 0.5, 1.0e3, 0.0, 0.3),
        [0.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
        1.0,
        1.0,
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.01],
        [0.02, 0.0, 0.0],
        1.0 / 90.0,
    );
    let queries = vec![overlapping, coincident, far];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    assert!(out[0].valid);
    assert!(!out[1].valid);
    assert!(!out[2].valid);
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_parity(res, q, &format!("batch[{i}]"));
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRotationalContactForce::new(&ctx);
    let out = gpu.evaluate(&ctx, &[]);
    assert!(out.is_empty());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRotationalContactForce::new(&ctx);
    let mut lcg = Lcg::new(0x51D3_C0DE);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        // Random unit contact normal, rejecting near-zero directions.
        let nx = lcg.next_range(-1.0, 1.0);
        let ny = lcg.next_range(-1.0, 1.0);
        let nz = lcg.next_range(-1.0, 1.0);
        let nlen = (nx * nx + ny * ny + nz * nz).sqrt();
        if nlen < 0.25 {
            continue;
        }
        let normal = [nx / nlen, ny / nlen, nz / nlen];
        // Radii ~1, centres separated so overlap sits in (0.2, 1.8).
        let distance = lcg.next_range(0.2, 1.8);
        let pos_a = [
            lcg.next_range(-0.5, 0.5),
            lcg.next_range(-0.5, 0.5),
            lcg.next_range(-0.5, 0.5),
        ];
        let pos_b = [
            pos_a[0] + normal[0] * distance,
            pos_a[1] + normal[1] * distance,
            pos_a[2] + normal[2] * distance,
        ];
        let candidate = RotationalContactForceQuery::new(
            model(
                lcg.next_range(5.0e2, 2.0e3),
                lcg.next_range(0.0, 5.0),
                lcg.next_range(5.0e2, 2.0e3),
                lcg.next_range(0.0, 5.0),
                lcg.next_range(0.1, 0.9),
                lcg.next_range(5.0e2, 2.0e3),
                lcg.next_range(0.0, 5.0),
                lcg.next_range(0.05, 0.5),
            ),
            pos_a,
            pos_b,
            1.0,
            1.0,
            [
                lcg.next_range(-1.0, 1.0),
                lcg.next_range(-1.0, 1.0),
                lcg.next_range(-1.0, 1.0),
            ],
            [
                lcg.next_range(-1.0, 1.0),
                lcg.next_range(-1.0, 1.0),
                lcg.next_range(-1.0, 1.0),
            ],
            [
                lcg.next_range(-2.0, 2.0),
                lcg.next_range(-2.0, 2.0),
                lcg.next_range(-2.0, 2.0),
            ],
            [
                lcg.next_range(-2.0, 2.0),
                lcg.next_range(-2.0, 2.0),
                lcg.next_range(-2.0, 2.0),
            ],
            [
                lcg.next_range(-0.05, 0.05),
                lcg.next_range(-0.05, 0.05),
                lcg.next_range(-0.05, 0.05),
            ],
            [
                lcg.next_range(-0.05, 0.05),
                lcg.next_range(-0.05, 0.05),
                lcg.next_range(-0.05, 0.05),
            ],
            lcg.next_range(1.0 / 240.0, 1.0 / 30.0),
        );
        // Reject samples sitting on either Coulomb knee, so the f32 rounding
        // gap cannot flip the discrete sliding / clamp decisions.
        let o = oracle(&candidate);
        let t_margin = (o.tmag0 - o.max_friction).abs();
        let t_scale = o.tmag0.max(o.max_friction).max(1.0);
        if t_margin < 0.05 * t_scale {
            continue;
        }
        let r_margin = (o.rmag0 - o.max_rolling).abs();
        let r_scale = o.rmag0.max(o.max_rolling).max(1.0);
        if r_margin < 0.05 * r_scale {
            continue;
        }
        queries.push(candidate);
    }
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_parity(res, q, &format!("sweep[{i}]"));
    }
}

/// A small deterministic linear-congruential generator; the fixture carries no
/// external randomness. Constants are the Numerical Recipes values.
struct Lcg {
    state: u32,
}

impl Lcg {
    fn new(seed: u32) -> Self {
        Lcg { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        self.state
    }

    /// A `[0, 1)` fraction built from the top bits, keeping the fixture pure
    /// integer host-side with no transcendental call.
    fn next_unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }

    /// A `[lo, hi)` fraction.
    fn next_range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_unit()
    }
}
