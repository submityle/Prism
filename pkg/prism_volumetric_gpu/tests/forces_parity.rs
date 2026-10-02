//! Real-device parity for the built-in force-library twin:
//! [`GpuForces`](prism_volumetric_gpu::forces::GpuForces) must reproduce the
//! `CPU` golden
//! [`forces`](prism_render_architecture::particle::forces) for every one of the
//! ten single-force evaluations it twins — point / line attractors, signed
//! radial blasts (`radial`, `explosion`, `implosion`), orbital drive, softened
//! gravity wells, quadratic drag, layered curl-noise turbulence and
//! spring-damper anchors — across each of the four
//! [`Falloff`](prism_render_architecture::particle::forces::Falloff) shaping
//! curves, and across a large randomized batch that spans every force code and
//! several workgroups.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, adds and
//! divides, so `CPU` and `GPU` evaluate the same closed form in the same order.
//! They are not bit-exact: a `GPU` may fuse a multiply-add the scalar reference
//! leaves separate, perturbing the low mantissa bits by a few units in the last
//! place. The comparison therefore allows `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3` on every acceleration channel.
//!
//! # Conditioning
//!
//! Every fixture is deliberately well away from a branch tie and from the
//! degeneracy cracks: attractor samples sit clearly inside (or clearly with an
//! unbounded `radius` past) the influence radius so the falloff gate never
//! splits across devices; orbital samples keep the particle clearly off the
//! axis so the orbital plane is well defined; drag samples keep the speed and
//! coefficient clearly positive; direction and axis vectors are kept clearly
//! non-zero so `normalize_or_zero` takes the same branch on both devices.
//! Deliberately degenerate fixtures (a zero line direction, a zero drag
//! coefficient) use *exact* zeros that both devices evaluate identically.
//! Turbulence uses a central-difference `epsilon` of `1e-2` and at most four
//! octaves so the finite-difference amplification of a few units in the last
//! place stays comfortably inside the parity bound.
//!
//! Provenance: twinned from this repository's
//! [`forces`](prism_render_architecture::particle::forces); no third-party
//! engine source or derived code.

use prism_render_architecture::particle::forces::{
    explosion, gravity_well, implosion, line_attractor, orbital_force, point_attractor,
    quadratic_drag, radial_force, spring_damper, turbulence, Falloff,
};
use prism_render_architecture::particle::Vec3;
use prism_volumetric_gpu::forces::{
    GpuForces, GpuForcesQuery, GpuForcesResult, FALLOFF_CONSTANT, FALLOFF_INVERSE_SQUARE,
    FALLOFF_LINEAR, FALLOFF_SMOOTHSTEP, FORCE_EXPLOSION, FORCE_GRAVITY_WELL, FORCE_IMPLOSION,
    FORCE_LINE_ATTRACTOR, FORCE_ORBITAL, FORCE_POINT_ATTRACTOR, FORCE_QUADRATIC_DRAG, FORCE_RADIAL,
    FORCE_SPRING_DAMPER, FORCE_TURBULENCE,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound. A `GPU` may fuse a multiply-add the scalar reference
/// leaves separate, perturbing the low mantissa bits by a few units in the last
/// place; `1e-4` admits that legal slack while still failing a wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in
/// the last place exceed the absolute floor.
const REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
}

/// Asserts two vectors agree channel-for-channel within the parity bound.
fn close_vec(label: &str, idx: usize, got: Vec3, want: Vec3) {
    assert!(
        close(got.x, want.x) && close(got.y, want.y) && close(got.z, want.z),
        "query {idx} {label}: gpu ({}, {}, {}) vs cpu ({}, {}, {})",
        got.x,
        got.y,
        got.z,
        want.x,
        want.y,
        want.z
    );
}

/// A tiny integer linear-congruential generator; only integer and divide work,
/// so no transcendental appears. Returns a value in `[0, 1)`.
fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = (*state >> 40) as u32;
    (bits & 0x00ff_ffff) as f32 / 16_777_216.0
}

/// A raw `u32` draw from `state`, used for noise seeds.
fn rand_u32(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 32) as u32
}

/// A pseudo-random value in `[-span, span)` drawn from `state`.
fn signed(state: &mut u64, span: f32) -> f32 {
    (lcg(state) * 2.0 - 1.0) * span
}

/// A pseudo-random vector with each component in `[-span, span)`.
fn rand_vec(state: &mut u64, span: f32) -> Vec3 {
    Vec3::new(
        signed(state, span),
        signed(state, span),
        signed(state, span),
    )
}

/// Draws a vector whose length is clearly at least `min_len`, by rejection, so
/// direction / axis inputs stay well away from the `normalize_or_zero` cutoff.
fn rand_offset(state: &mut u64, min_len: f32, span: f32) -> Vec3 {
    loop {
        let v = rand_vec(state, span);
        if v.length_squared() >= min_len * min_len {
            return v;
        }
    }
}

/// A neutral query: every parameter zeroed except the turbulence ladder, which
/// carries sensible defaults so turbulence fixtures only override what matters.
fn base() -> GpuForcesQuery {
    GpuForcesQuery {
        position: Vec3::ZERO,
        velocity: Vec3::ZERO,
        center: Vec3::ZERO,
        direction: Vec3::new(0.0, 1.0, 0.0),
        force_code: FORCE_POINT_ATTRACTOR,
        falloff_code: FALLOFF_CONSTANT,
        seed: 0,
        octaves: 0,
        strength: 0.0,
        radius: 0.0,
        softening: 0.0,
        coefficient: 0.0,
        target_radius: 0.0,
        tangential_strength: 0.0,
        radial_stiffness: 0.0,
        stiffness: 0.0,
        damping: 0.0,
        frequency: 1.0,
        amplitude: 1.0,
        frequency_multiplier: 2.0,
        amplitude_multiplier: 0.5,
        epsilon: 1.0e-2,
    }
}

/// Maps a public falloff classification code back to the golden
/// [`Falloff`](prism_render_architecture::particle::forces::Falloff) variant.
fn golden_falloff(code: u32) -> Falloff {
    match code {
        FALLOFF_LINEAR => Falloff::Linear,
        FALLOFF_SMOOTHSTEP => Falloff::Smoothstep,
        FALLOFF_INVERSE_SQUARE => Falloff::InverseSquare,
        // `FALLOFF_CONSTANT` and any out-of-range code: flat full strength.
        _ => Falloff::Constant,
    }
}

/// Evaluates the golden reference acceleration for one query by dispatching on
/// its force classification code to the matching free function.
fn golden_accel(q: &GpuForcesQuery) -> Vec3 {
    let falloff = golden_falloff(q.falloff_code);
    match q.force_code {
        FORCE_POINT_ATTRACTOR => point_attractor(
            q.position,
            q.center,
            q.strength,
            q.radius,
            q.softening,
            falloff,
        ),
        FORCE_LINE_ATTRACTOR => line_attractor(
            q.position,
            q.center,
            q.direction,
            q.strength,
            q.radius,
            q.softening,
            falloff,
        ),
        FORCE_RADIAL => radial_force(
            q.position,
            q.center,
            q.strength,
            q.radius,
            q.softening,
            falloff,
        ),
        FORCE_EXPLOSION => explosion(
            q.position,
            q.center,
            q.strength,
            q.radius,
            q.softening,
            falloff,
        ),
        FORCE_IMPLOSION => implosion(
            q.position,
            q.center,
            q.strength,
            q.radius,
            q.softening,
            falloff,
        ),
        FORCE_ORBITAL => orbital_force(
            q.position,
            q.center,
            q.direction,
            q.target_radius,
            q.tangential_strength,
            q.radial_stiffness,
        ),
        FORCE_GRAVITY_WELL => gravity_well(q.position, q.center, q.strength, q.softening),
        FORCE_QUADRATIC_DRAG => quadratic_drag(q.velocity, q.coefficient),
        FORCE_TURBULENCE => turbulence(
            q.position,
            q.seed,
            q.frequency,
            q.amplitude,
            q.octaves,
            q.frequency_multiplier,
            q.amplitude_multiplier,
            q.epsilon,
        ),
        FORCE_SPRING_DAMPER => {
            spring_damper(q.position, q.velocity, q.center, q.stiffness, q.damping)
        }
        // No other classification code exists; the twin leaves it at zero.
        _ => Vec3::ZERO,
    }
}

/// Runs the whole batch on the `GPU` and pins every result against the golden
/// reference acceleration for the same query.
fn check(ctx: &GpuContext, gpu: &GpuForces, queries: &[GpuForcesQuery]) {
    let got: Vec<GpuForcesResult> = gpu.eval(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (query, result)) in queries.iter().zip(got.iter()).enumerate() {
        close_vec(
            "acceleration",
            idx,
            result.acceleration,
            golden_accel(query),
        );
    }
}

/// Builds a well-conditioned attractor-style query (`point`, `radial`,
/// `explosion`, `implosion`) with the particle clearly inside — or clearly with
/// an unbounded radius past — the influence gate.
fn attractor_query(state: &mut u64, force_code: u32) -> GpuForcesQuery {
    let center = rand_vec(state, 3.0);
    let offset = rand_offset(state, 0.5, 2.5);
    let position = center.add(offset);
    let dist = offset.length();
    // Either unbounded (radius 0, clearly <= EPS) or clearly larger than the
    // sample distance, so the hard radius gate never splits across devices.
    let radius = if lcg(state) < 0.5 {
        0.0
    } else {
        dist + 1.5 + lcg(state)
    };
    let falloff_code = (lcg(state) * 4.0) as u32;
    GpuForcesQuery {
        position,
        center,
        force_code,
        falloff_code,
        strength: signed(state, 3.0),
        radius,
        softening: 0.05 + lcg(state) * 0.4,
        ..base()
    }
}

/// Builds a line-attractor query with a clearly non-zero (unit-sized) axis and
/// an unbounded radius, so the perpendicular-distance gate never splits.
fn line_query(state: &mut u64) -> GpuForcesQuery {
    let line_point = rand_vec(state, 3.0);
    let direction = rand_offset(state, 1.0, 3.0);
    let position = line_point.add(rand_vec(state, 3.0));
    let falloff_code = (lcg(state) * 4.0) as u32;
    GpuForcesQuery {
        position,
        center: line_point,
        direction,
        force_code: FORCE_LINE_ATTRACTOR,
        falloff_code,
        strength: signed(state, 3.0),
        radius: 0.0,
        softening: 0.05 + lcg(state) * 0.4,
        ..base()
    }
}

/// Builds an orbital query, rejection-sampling the particle so its distance to
/// the axis is clearly non-zero and the orbital plane is well defined.
fn orbital_query(state: &mut u64) -> GpuForcesQuery {
    let center = rand_vec(state, 3.0);
    let axis = rand_offset(state, 1.0, 2.0);
    let unit = axis.normalize_or_zero();
    loop {
        let position = center.add(rand_vec(state, 3.0));
        let radial = position.sub(center);
        let axial = unit.scale(radial.dot(unit));
        let perp = radial.sub(axial);
        if perp.length_squared() >= 0.3 * 0.3 {
            return GpuForcesQuery {
                position,
                center,
                direction: axis,
                force_code: FORCE_ORBITAL,
                target_radius: 0.5 + lcg(state) * 2.0,
                tangential_strength: signed(state, 3.0),
                radial_stiffness: 0.5 + lcg(state) * 2.0,
                ..base()
            };
        }
    }
}

/// Builds a gravity-well query with the particle clearly off the center so the
/// pull direction is well defined.
fn gravity_query(state: &mut u64) -> GpuForcesQuery {
    let center = rand_vec(state, 3.0);
    let position = center.add(rand_offset(state, 0.5, 3.0));
    GpuForcesQuery {
        position,
        center,
        force_code: FORCE_GRAVITY_WELL,
        strength: signed(state, 4.0),
        softening: 0.1 + lcg(state) * 0.5,
        ..base()
    }
}

/// Builds a quadratic-drag query with a clearly positive speed and coefficient.
fn drag_query(state: &mut u64) -> GpuForcesQuery {
    GpuForcesQuery {
        velocity: rand_offset(state, 0.5, 3.0),
        force_code: FORCE_QUADRATIC_DRAG,
        coefficient: 0.2 + lcg(state) * 1.0,
        ..base()
    }
}

/// Builds a spring-damper query (no branch gates to avoid).
fn spring_query(state: &mut u64) -> GpuForcesQuery {
    GpuForcesQuery {
        position: rand_vec(state, 3.0),
        velocity: rand_vec(state, 3.0),
        center: rand_vec(state, 3.0),
        force_code: FORCE_SPRING_DAMPER,
        stiffness: 0.5 + lcg(state) * 4.0,
        damping: 0.1 + lcg(state) * 2.0,
        ..base()
    }
}

/// Builds a turbulence query with a moderate position, one to four octaves, and
/// a safe central-difference step so the finite-difference stays well inside
/// the parity bound.
fn turbulence_query(state: &mut u64) -> GpuForcesQuery {
    let octaves = 1 + (lcg(state) * 4.0) as u32;
    GpuForcesQuery {
        position: rand_vec(state, 2.5),
        force_code: FORCE_TURBULENCE,
        seed: rand_u32(state),
        octaves,
        strength: 0.0,
        frequency: 0.8 + lcg(state) * 0.8,
        amplitude: 0.8 + lcg(state) * 0.8,
        frequency_multiplier: 2.0,
        amplitude_multiplier: 0.5,
        epsilon: 1.0e-2,
        ..base()
    }
}

#[test]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuForces::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn point_attractor_all_falloffs_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuForces::new(&ctx);
    // Particle at distance 3 inside a radius-4 influence (ratio 0.75, clearly
    // interior) so each falloff shapes a non-trivial, non-boundary multiplier.
    let shaped: Vec<GpuForcesQuery> = [
        FALLOFF_CONSTANT,
        FALLOFF_LINEAR,
        FALLOFF_SMOOTHSTEP,
        FALLOFF_INVERSE_SQUARE,
    ]
    .into_iter()
    .map(|falloff_code| GpuForcesQuery {
        position: Vec3::new(3.0, 0.0, 0.0),
        center: Vec3::ZERO,
        force_code: FORCE_POINT_ATTRACTOR,
        falloff_code,
        strength: 2.0,
        radius: 4.0,
        softening: 0.1,
        ..base()
    })
    .collect();
    // Plus an unbounded (radius 0) pull where every falloff is full strength.
    let unbounded = GpuForcesQuery {
        position: Vec3::new(-1.0, 2.0, 0.5),
        center: Vec3::new(0.5, 0.0, -1.0),
        force_code: FORCE_POINT_ATTRACTOR,
        falloff_code: FALLOFF_SMOOTHSTEP,
        strength: -1.5,
        radius: 0.0,
        softening: 0.2,
        ..base()
    };
    let mut queries = shaped;
    queries.push(unbounded);
    check(&ctx, &gpu, &queries);
}

#[test]
fn line_attractor_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuForces::new(&ctx);
    // A line through the origin along +y with the particle clearly off the line.
    let on_line = GpuForcesQuery {
        position: Vec3::new(2.0, 1.0, 0.0),
        center: Vec3::ZERO,
        direction: Vec3::new(0.0, 2.0, 0.0),
        force_code: FORCE_LINE_ATTRACTOR,
        falloff_code: FALLOFF_LINEAR,
        strength: 3.0,
        radius: 5.0,
        softening: 0.1,
        ..base()
    };
    // A degenerate (exactly zero) direction collapses to a point attractor
    // about `center`; the exact zero takes the same branch on both devices.
    let degenerate = GpuForcesQuery {
        position: Vec3::new(1.0, -2.0, 0.5),
        center: Vec3::new(0.0, 0.0, 0.0),
        direction: Vec3::ZERO,
        force_code: FORCE_LINE_ATTRACTOR,
        falloff_code: FALLOFF_CONSTANT,
        strength: 2.0,
        radius: 0.0,
        softening: 0.15,
        ..base()
    };
    check(&ctx, &gpu, &[on_line, degenerate]);
}

#[test]
fn radial_explosion_implosion_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuForces::new(&ctx);
    let position = Vec3::new(2.0, -1.0, 1.0);
    let center = Vec3::new(-0.5, 0.5, 0.0);
    let radial = GpuForcesQuery {
        position,
        center,
        force_code: FORCE_RADIAL,
        falloff_code: FALLOFF_LINEAR,
        strength: 2.5,
        radius: 6.0,
        softening: 0.1,
        ..base()
    };
    let blast = GpuForcesQuery {
        position,
        center,
        force_code: FORCE_EXPLOSION,
        falloff_code: FALLOFF_INVERSE_SQUARE,
        // A negative author value still blasts outward (|strength| is used).
        strength: -3.0,
        radius: 0.0,
        softening: 0.2,
        ..base()
    };
    let collapse = GpuForcesQuery {
        position,
        center,
        force_code: FORCE_IMPLOSION,
        falloff_code: FALLOFF_SMOOTHSTEP,
        strength: 3.0,
        radius: 6.0,
        softening: 0.2,
        ..base()
    };
    check(&ctx, &gpu, &[radial, blast, collapse]);
}

#[test]
fn orbital_force_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuForces::new(&ctx);
    // Particle clearly off the +y axis, so the orbital plane is well defined.
    let query = GpuForcesQuery {
        position: Vec3::new(2.0, 1.0, 0.0),
        center: Vec3::ZERO,
        direction: Vec3::new(0.0, 1.0, 0.0),
        force_code: FORCE_ORBITAL,
        target_radius: 1.0,
        tangential_strength: 2.0,
        radial_stiffness: 1.5,
        ..base()
    };
    check(&ctx, &gpu, &[query]);
}

#[test]
fn gravity_well_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuForces::new(&ctx);
    let query = GpuForcesQuery {
        position: Vec3::new(2.0, 0.0, 0.0),
        center: Vec3::ZERO,
        force_code: FORCE_GRAVITY_WELL,
        strength: 8.0,
        softening: 0.25,
        ..base()
    };
    check(&ctx, &gpu, &[query]);
}

#[test]
fn quadratic_drag_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuForces::new(&ctx);
    // A live drag plus two exactly-degenerate cases (zero coefficient, zero
    // velocity) that both devices resolve to an exact zero.
    let live = GpuForcesQuery {
        velocity: Vec3::new(3.0, -1.0, 2.0),
        force_code: FORCE_QUADRATIC_DRAG,
        coefficient: 0.5,
        ..base()
    };
    let zero_coeff = GpuForcesQuery {
        velocity: Vec3::new(3.0, -1.0, 2.0),
        force_code: FORCE_QUADRATIC_DRAG,
        coefficient: 0.0,
        ..base()
    };
    let zero_speed = GpuForcesQuery {
        velocity: Vec3::ZERO,
        force_code: FORCE_QUADRATIC_DRAG,
        coefficient: 0.5,
        ..base()
    };
    check(&ctx, &gpu, &[live, zero_coeff, zero_speed]);
}

#[test]
fn spring_damper_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuForces::new(&ctx);
    let query = GpuForcesQuery {
        position: Vec3::new(2.0, 0.0, 0.0),
        velocity: Vec3::new(1.0, 0.0, 0.0),
        center: Vec3::ZERO,
        force_code: FORCE_SPRING_DAMPER,
        stiffness: 4.0,
        damping: 0.5,
        ..base()
    };
    check(&ctx, &gpu, &[query]);
}

#[test]
fn turbulence_octaves_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuForces::new(&ctx);
    // One to four octaves of the same base field; the bounded loop in the twin
    // trips the identical number of times (all <= the MAX_OCTAVES bound).
    let queries: Vec<GpuForcesQuery> = (1..=4u32)
        .map(|octaves| GpuForcesQuery {
            position: Vec3::new(1.3, -2.1, 0.7),
            force_code: FORCE_TURBULENCE,
            seed: 0x00c0_ffee,
            octaves,
            frequency: 1.0,
            amplitude: 1.0,
            frequency_multiplier: 2.0,
            amplitude_multiplier: 0.5,
            epsilon: 1.0e-2,
            ..base()
        })
        .collect();
    check(&ctx, &gpu, &queries);
}

#[test]
fn mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuForces::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    // Many rounds, each contributing one well-conditioned query per force code,
    // so the batch spans every classification and several workgroups (one
    // thread per query, flattened to a 1-D dispatch) in a single submission.
    let mut queries = Vec::new();
    for _ in 0..9 {
        queries.push(attractor_query(&mut state, FORCE_POINT_ATTRACTOR));
        queries.push(attractor_query(&mut state, FORCE_RADIAL));
        queries.push(attractor_query(&mut state, FORCE_EXPLOSION));
        queries.push(attractor_query(&mut state, FORCE_IMPLOSION));
        queries.push(line_query(&mut state));
        queries.push(orbital_query(&mut state));
        queries.push(gravity_query(&mut state));
        queries.push(drag_query(&mut state));
        queries.push(spring_query(&mut state));
        queries.push(turbulence_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}
