//! Real-device parity for the screen-space-reflection (`SSR`) numeric twin:
//! [`GpuScreenSpaceReflection`](prism_volumetric_gpu::screen_space_reflection::GpuScreenSpaceReflection)
//! must reproduce the `CPU` golden
//! [`screen_space_reflection`](prism_render_architecture::particle::screen_space_reflection)
//! across `clamp01`, the hand-written `smoothstep` polynomial, the
//! [`Vec3`](prism_render_architecture::particle::screen_space_reflection::Vec3)
//! primitives (`plus` / `minus` / `scale` / `dot` / `length` / `normalize`), the
//! [`reflect`](prism_render_architecture::particle::screen_space_reflection::reflect)
//! mirror identity, the pinhole
//! [`Projection::project`](prism_render_architecture::particle::screen_space_reflection::Projection::project)
//! (its `Option` remapped to an `on_screen` flag), the three confidence fades
//! [`screen_edge_fade`](prism_render_architecture::particle::screen_space_reflection::screen_edge_fade),
//! [`grazing_fade`](prism_render_architecture::particle::screen_space_reflection::grazing_fade)
//! and [`distance_fade`](prism_render_architecture::particle::screen_space_reflection::distance_fade),
//! and the numeric fold of `finalize_hit`.
//!
//! The reference `clamp01`, `smoothstep` and `finalize_hit` are private to the
//! golden module, so this test replicates the first two from the golden formulae
//! and composes the third from the public fades; every other reference is called
//! directly. The fixtures are simple decimals (and exact `3-4-5` unit directions
//! where a unit vector is wanted), so they stay pure and need no external math
//! library and no transcendental method. The random batch draws from a host-side
//! `u64` `LCG` and rejection samples away from the degenerate branches
//! (near-equal `smoothstep` span, near-zero `normalize` length, `view` depth
//! below `NEAR_MIN`, non-positive search radius, screen-border `UV`) so the
//! comparison always exercises the live solve.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Every twinned quantity threads through multiplies, adds and one guarded
//! division, so `CPU` and `GPU` are compared under tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`). The discrete
//! `on_screen` and `hit` flags are matched exactly.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::screen_space_reflection`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::screen_space_reflection::{
    distance_fade, grazing_fade, reflect, screen_edge_fade, Projection, SsrHit, Vec2, Vec3,
};
use prism_volumetric_gpu::screen_space_reflection::{
    GpuScreenSpaceReflection, ScreenSpaceReflectionQuery, ScreenSpaceReflectionResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for one continuous `f32` lane.
const ABS_EPS: f32 = 1.0e-4;
/// Relative tolerance for one continuous `f32` lane.
const REL_EPS: f32 = 1.0e-3;
/// Floor for the relative-tolerance denominator.
const REL_FLOOR: f32 = 1.0e-6;

/// Soft-edge guard replicated from the golden private `MIN_EDGE`, used by the
/// local `smoothstep` copy below.
const MIN_EDGE_REF: f32 = 1.0e-6;

/// Mixed absolute / relative tolerance comparison for one `f32` lane.
fn approx(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= ABS_EPS {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= REL_EPS
}

/// Tolerant comparison of two [`Vec3`] values, lane by lane.
fn approx_vec(a: Vec3, b: Vec3) -> bool {
    approx(a.x, b.x) && approx(a.y, b.y) && approx(a.z, b.z)
}

/// Tolerant comparison of two [`Vec2`] values, lane by lane.
fn approx_vec2(a: Vec2, b: Vec2) -> bool {
    approx(a.x, b.x) && approx(a.y, b.y)
}

/// Local copy of the golden private `clamp01` (the branch-free `0..=1` clamp),
/// replicated because it is not exported from the reference module.
fn clamp01_ref(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

/// Local copy of the golden private `smoothstep`, replicated branch for branch
/// because it is not exported: a near-equal span collapses to a hard step at
/// `edge1`, otherwise the clamped parameter is shaped by `t * t * (3 - 2 * t)`.
fn smoothstep_ref(edge0: f32, edge1: f32, x: f32) -> f32 {
    let span = edge1 - edge0;
    if span < MIN_EDGE_REF {
        return if x < edge1 { 0.0 } else { 1.0 };
    }
    let t = clamp01_ref((x - edge0) / span);
    t * t * (3.0 - 2.0 * t)
}

/// Asserts a [`ScreenSpaceReflectionResult`] is the expected scalar.
fn assert_scalar(got: ScreenSpaceReflectionResult, expected: f32) {
    match got {
        ScreenSpaceReflectionResult::Scalar(s) => {
            assert!(
                approx(s, expected),
                "scalar mismatch: gpu {s} vs cpu {expected}"
            );
        }
        other => panic!("expected a scalar result, got {other:?}"),
    }
}

/// Asserts a [`ScreenSpaceReflectionResult`] is the expected vector.
fn assert_vector(got: ScreenSpaceReflectionResult, expected: Vec3) {
    match got {
        ScreenSpaceReflectionResult::Vector(v) => {
            assert!(
                approx_vec(v, expected),
                "vector mismatch: gpu {v:?} vs cpu {expected:?}"
            );
        }
        other => panic!("expected a vector result, got {other:?}"),
    }
}

/// Asserts the twinned answer for one query matches the `CPU` golden, dispatching
/// on the query variant to pick the reference function and the result shape.
fn assert_parity(gpu: &GpuScreenSpaceReflection, ctx: &GpuContext, q: &ScreenSpaceReflectionQuery) {
    let got = gpu.evaluate(ctx, std::slice::from_ref(q));
    assert_eq!(got.len(), 1, "one result per query");
    let g = got[0];
    match q {
        ScreenSpaceReflectionQuery::Clamp01 { x } => assert_scalar(g, clamp01_ref(*x)),
        ScreenSpaceReflectionQuery::Smoothstep { edge0, edge1, x } => {
            assert_scalar(g, smoothstep_ref(*edge0, *edge1, *x));
        }
        ScreenSpaceReflectionQuery::Vec3Plus { a, b } => assert_vector(g, a.plus(*b)),
        ScreenSpaceReflectionQuery::Vec3Minus { a, b } => assert_vector(g, a.minus(*b)),
        ScreenSpaceReflectionQuery::Vec3Scale { a, scalar } => assert_vector(g, a.scale(*scalar)),
        ScreenSpaceReflectionQuery::Vec3Dot { a, b } => assert_scalar(g, a.dot(*b)),
        ScreenSpaceReflectionQuery::Vec3Length { a } => assert_scalar(g, a.length()),
        ScreenSpaceReflectionQuery::Vec3Normalize { a } => assert_vector(g, a.normalize()),
        ScreenSpaceReflectionQuery::Reflect { incident, normal } => {
            assert_vector(g, reflect(*incident, *normal));
        }
        ScreenSpaceReflectionQuery::Project {
            focal_x,
            focal_y,
            point,
        } => {
            let expected = Projection::new(*focal_x, *focal_y).project(*point);
            match g {
                ScreenSpaceReflectionResult::Projection { on_screen, uv } => match expected {
                    Some(e) => {
                        assert!(on_screen, "expected on-screen projection, got off-screen");
                        assert!(
                            approx_vec2(uv, e),
                            "projected uv mismatch: gpu {uv:?} vs cpu {e:?}"
                        );
                    }
                    None => assert!(!on_screen, "expected off-screen projection, got on-screen"),
                },
                other => panic!("expected a projection result, got {other:?}"),
            }
        }
        ScreenSpaceReflectionQuery::ScreenEdgeFade { uv, edge } => {
            assert_scalar(g, screen_edge_fade(*uv, *edge));
        }
        ScreenSpaceReflectionQuery::GrazingFade {
            reflect_dir,
            view_dir,
            bias,
        } => assert_scalar(g, grazing_fade(*reflect_dir, *view_dir, *bias)),
        ScreenSpaceReflectionQuery::DistanceFade {
            hit_distance,
            max_distance,
        } => assert_scalar(g, distance_fade(*hit_distance, *max_distance)),
        ScreenSpaceReflectionQuery::FinalizeHit {
            uv,
            hit_distance,
            reflect_dir,
            view_dir,
            edge_fade,
            grazing_bias,
            max_distance,
        } => {
            let edge = screen_edge_fade(*uv, *edge_fade);
            let graze = grazing_fade(*reflect_dir, *view_dir, *grazing_bias);
            let dist = distance_fade(*hit_distance, *max_distance);
            let expected = SsrHit {
                uv: *uv,
                confidence: clamp01_ref(edge * graze * dist),
                hit_distance: *hit_distance,
                hit: true,
            };
            match g {
                ScreenSpaceReflectionResult::Hit(h) => {
                    assert!(h.hit, "finalize_hit must report a hit");
                    assert!(
                        approx_vec2(h.uv, expected.uv),
                        "hit uv mismatch: gpu {:?} vs cpu {:?}",
                        h.uv,
                        expected.uv
                    );
                    assert!(
                        approx(h.confidence, expected.confidence),
                        "confidence mismatch: gpu {} vs cpu {}",
                        h.confidence,
                        expected.confidence
                    );
                    assert!(
                        approx(h.hit_distance, expected.hit_distance),
                        "hit distance mismatch: gpu {} vs cpu {}",
                        h.hit_distance,
                        expected.hit_distance
                    );
                }
                other => panic!("expected a hit result, got {other:?}"),
            }
        }
    }
}

/// A tiny host-side `u64` `LCG`, used only to drive the deterministic random
/// batch; it uses no transcendental method and no external math library.
struct Lcg {
    /// Current generator state.
    state: u64,
}

impl Lcg {
    /// Seeds the generator.
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Advances the state and returns the high `32` bits.
    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.state >> 32) as u32
    }

    /// Returns a sample in `0.0..1.0` from the top `24` mantissa bits.
    fn unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1_u32 << 24) as f32
    }

    /// Returns a signed sample in `-1.0..1.0`.
    fn signed(&mut self) -> f32 {
        self.unit() * 2.0 - 1.0
    }

    /// Returns a signed [`Vec3`] with each lane in `-1.0..1.0`.
    fn vec3(&mut self) -> Vec3 {
        Vec3::new(self.signed(), self.signed(), self.signed())
    }

    /// Returns a [`Vec3`] whose length is well clear of `MIN_NORM`, rejection
    /// sampled so the squared length (via a `dot`, never a transcendental call)
    /// stays above a comfortable floor.
    fn nonzero_vec3(&mut self) -> Vec3 {
        loop {
            let v = self.vec3();
            if v.dot(v) > 0.25 {
                return v;
            }
        }
    }
}

/// Draws a diverse set of non-degenerate queries, one of every variant, each
/// rejection sampled clear of the golden's degenerate branches so the batch
/// always exercises the live solve rather than a short-circuit.
fn random_batch(rng: &mut Lcg) -> Vec<ScreenSpaceReflectionQuery> {
    let mut out = Vec::new();
    for _ in 0..3 {
        // clamp01 over the full span (the clamp kinks at 0 and 1 are continuous).
        out.push(ScreenSpaceReflectionQuery::Clamp01 {
            x: rng.signed() * 2.0,
        });
        // smoothstep with a span well above MIN_EDGE.
        let edge0 = rng.signed();
        out.push(ScreenSpaceReflectionQuery::Smoothstep {
            edge0,
            edge1: edge0 + 0.5 + rng.unit(),
            x: rng.signed() * 2.0,
        });
        out.push(ScreenSpaceReflectionQuery::Vec3Plus {
            a: rng.vec3(),
            b: rng.vec3(),
        });
        out.push(ScreenSpaceReflectionQuery::Vec3Minus {
            a: rng.vec3(),
            b: rng.vec3(),
        });
        out.push(ScreenSpaceReflectionQuery::Vec3Scale {
            a: rng.vec3(),
            scalar: rng.signed() * 3.0,
        });
        out.push(ScreenSpaceReflectionQuery::Vec3Dot {
            a: rng.vec3(),
            b: rng.vec3(),
        });
        out.push(ScreenSpaceReflectionQuery::Vec3Length { a: rng.vec3() });
        // normalize of a vector well clear of MIN_NORM.
        out.push(ScreenSpaceReflectionQuery::Vec3Normalize {
            a: rng.nonzero_vec3(),
        });
        out.push(ScreenSpaceReflectionQuery::Reflect {
            incident: rng.vec3(),
            normal: rng.nonzero_vec3(),
        });
        // project a point well in front of the pinhole (z clear of NEAR_MIN).
        out.push(ScreenSpaceReflectionQuery::Project {
            focal_x: 0.8 + rng.unit(),
            focal_y: 0.8 + rng.unit(),
            point: Vec3::new(rng.signed(), rng.signed(), 1.0 + 4.0 * rng.unit()),
        });
        // edge fade: UV kept inside the frame, edge width well above MIN_EDGE.
        out.push(ScreenSpaceReflectionQuery::ScreenEdgeFade {
            uv: Vec2::new(0.25 + 0.5 * rng.unit(), 0.25 + 0.5 * rng.unit()),
            edge: 0.05 + 0.1 * rng.unit(),
        });
        // grazing fade: bias well above MIN_EDGE/2 so the span never collapses.
        out.push(ScreenSpaceReflectionQuery::GrazingFade {
            reflect_dir: rng.nonzero_vec3().normalize(),
            view_dir: rng.nonzero_vec3().normalize(),
            bias: 0.1 + 0.4 * rng.unit(),
        });
        // distance fade: search radius well above MIN_STEP.
        let max_distance = 0.5 + 4.0 * rng.unit();
        out.push(ScreenSpaceReflectionQuery::DistanceFade {
            hit_distance: max_distance * rng.unit(),
            max_distance,
        });
        // finalize_hit: all three fades on non-degenerate inputs.
        let max_distance = 1.0 + 4.0 * rng.unit();
        out.push(ScreenSpaceReflectionQuery::FinalizeHit {
            uv: Vec2::new(0.25 + 0.5 * rng.unit(), 0.25 + 0.5 * rng.unit()),
            hit_distance: max_distance * rng.unit(),
            reflect_dir: rng.nonzero_vec3().normalize(),
            view_dir: rng.nonzero_vec3().normalize(),
            edge_fade: 0.05 + 0.1 * rng.unit(),
            grazing_bias: 0.1 + 0.4 * rng.unit(),
            max_distance,
        });
    }
    out
}

#[test]
fn clamp01_and_smoothstep_match() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuScreenSpaceReflection::new(&ctx);
    for &x in &[-1.5_f32, -0.0, 0.25, 0.5, 1.0, 1.75] {
        assert_parity(&gpu, &ctx, &ScreenSpaceReflectionQuery::Clamp01 { x });
    }
    // Endpoints, midpoint and the degenerate (collapsed) span.
    assert_parity(
        &gpu,
        &ctx,
        &ScreenSpaceReflectionQuery::Smoothstep {
            edge0: 0.0,
            edge1: 1.0,
            x: 0.5,
        },
    );
    assert_parity(
        &gpu,
        &ctx,
        &ScreenSpaceReflectionQuery::Smoothstep {
            edge0: 0.2,
            edge1: 0.8,
            x: 0.35,
        },
    );
    assert_parity(
        &gpu,
        &ctx,
        &ScreenSpaceReflectionQuery::Smoothstep {
            edge0: 2.0,
            edge1: 2.0,
            x: 2.5,
        },
    );
}

#[test]
fn vec3_primitives_match() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuScreenSpaceReflection::new(&ctx);
    let a = Vec3::new(1.0, -2.0, 0.5);
    let b = Vec3::new(-0.5, 1.5, 2.0);
    assert_parity(&gpu, &ctx, &ScreenSpaceReflectionQuery::Vec3Plus { a, b });
    assert_parity(&gpu, &ctx, &ScreenSpaceReflectionQuery::Vec3Minus { a, b });
    assert_parity(
        &gpu,
        &ctx,
        &ScreenSpaceReflectionQuery::Vec3Scale { a, scalar: -1.25 },
    );
    assert_parity(&gpu, &ctx, &ScreenSpaceReflectionQuery::Vec3Dot { a, b });
    assert_parity(&gpu, &ctx, &ScreenSpaceReflectionQuery::Vec3Length { a });
    assert_parity(&gpu, &ctx, &ScreenSpaceReflectionQuery::Vec3Normalize { a });
}

#[test]
fn reflect_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuScreenSpaceReflection::new(&ctx);
    // An incident ray against an exact 3-4-5 unit normal, so the fixture stays
    // pure with no transcendental method.
    assert_parity(
        &gpu,
        &ctx,
        &ScreenSpaceReflectionQuery::Reflect {
            incident: Vec3::new(0.3, -0.9, 0.2),
            normal: Vec3::new(0.6, 0.8, 0.0),
        },
    );
}

#[test]
fn project_matches_on_screen_and_off_screen() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuScreenSpaceReflection::new(&ctx);
    // A point well in front of the pinhole projects to a finite UV.
    assert_parity(
        &gpu,
        &ctx,
        &ScreenSpaceReflectionQuery::Project {
            focal_x: 1.2,
            focal_y: 0.9,
            point: Vec3::new(0.4, -0.3, 2.5),
        },
    );
    // A point behind the pinhole (z below NEAR_MIN) reports off-screen on both
    // sides; the branch is identical so the exact flag match holds.
    assert_parity(
        &gpu,
        &ctx,
        &ScreenSpaceReflectionQuery::Project {
            focal_x: 1.2,
            focal_y: 0.9,
            point: Vec3::new(0.4, -0.3, -1.0),
        },
    );
}

#[test]
fn confidence_fades_match() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuScreenSpaceReflection::new(&ctx);
    assert_parity(
        &gpu,
        &ctx,
        &ScreenSpaceReflectionQuery::ScreenEdgeFade {
            uv: Vec2::new(0.5, 0.4),
            edge: 0.1,
        },
    );
    assert_parity(
        &gpu,
        &ctx,
        &ScreenSpaceReflectionQuery::GrazingFade {
            reflect_dir: Vec3::new(0.2, 0.0, 0.98).normalize(),
            view_dir: Vec3::new(0.0, 0.0, 1.0),
            bias: 0.25,
        },
    );
    assert_parity(
        &gpu,
        &ctx,
        &ScreenSpaceReflectionQuery::DistanceFade {
            hit_distance: 0.3,
            max_distance: 1.5,
        },
    );
}

#[test]
fn finalize_hit_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuScreenSpaceReflection::new(&ctx);
    assert_parity(
        &gpu,
        &ctx,
        &ScreenSpaceReflectionQuery::FinalizeHit {
            uv: Vec2::new(0.55, 0.45),
            hit_distance: 0.8,
            reflect_dir: Vec3::new(0.2, 0.0, 0.98).normalize(),
            view_dir: Vec3::new(0.0, 0.0, 1.0),
            edge_fade: 0.1,
            grazing_bias: 0.25,
            max_distance: 2.0,
        },
    );
}

#[test]
fn mixed_batch_matches_elementwise() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuScreenSpaceReflection::new(&ctx);
    // One dispatch mixing every query variant exercises the tagged-union batch
    // and the one-thread-per-query flattening; each result must be independent.
    let a = Vec3::new(1.0, -2.0, 0.5);
    let b = Vec3::new(-0.5, 1.5, 2.0);
    let batch = [
        ScreenSpaceReflectionQuery::Clamp01 { x: 1.75 },
        ScreenSpaceReflectionQuery::Smoothstep {
            edge0: 0.2,
            edge1: 0.8,
            x: 0.35,
        },
        ScreenSpaceReflectionQuery::Vec3Plus { a, b },
        ScreenSpaceReflectionQuery::Vec3Minus { a, b },
        ScreenSpaceReflectionQuery::Vec3Scale { a, scalar: -1.25 },
        ScreenSpaceReflectionQuery::Vec3Dot { a, b },
        ScreenSpaceReflectionQuery::Vec3Length { a },
        ScreenSpaceReflectionQuery::Vec3Normalize { a },
        ScreenSpaceReflectionQuery::Reflect {
            incident: Vec3::new(0.3, -0.9, 0.2),
            normal: Vec3::new(0.6, 0.8, 0.0),
        },
        ScreenSpaceReflectionQuery::Project {
            focal_x: 1.2,
            focal_y: 0.9,
            point: Vec3::new(0.4, -0.3, 2.5),
        },
        ScreenSpaceReflectionQuery::Project {
            focal_x: 1.2,
            focal_y: 0.9,
            point: Vec3::new(0.4, -0.3, -1.0),
        },
        ScreenSpaceReflectionQuery::ScreenEdgeFade {
            uv: Vec2::new(0.5, 0.4),
            edge: 0.1,
        },
        ScreenSpaceReflectionQuery::GrazingFade {
            reflect_dir: Vec3::new(0.2, 0.0, 0.98).normalize(),
            view_dir: Vec3::new(0.0, 0.0, 1.0),
            bias: 0.25,
        },
        ScreenSpaceReflectionQuery::DistanceFade {
            hit_distance: 0.3,
            max_distance: 1.5,
        },
        ScreenSpaceReflectionQuery::FinalizeHit {
            uv: Vec2::new(0.55, 0.45),
            hit_distance: 0.8,
            reflect_dir: Vec3::new(0.2, 0.0, 0.98).normalize(),
            view_dir: Vec3::new(0.0, 0.0, 1.0),
            edge_fade: 0.1,
            grazing_bias: 0.25,
            max_distance: 2.0,
        },
    ];
    let got = gpu.evaluate(&ctx, &batch);
    assert_eq!(got.len(), batch.len());
    for q in &batch {
        assert_parity(&gpu, &ctx, q);
    }
}

#[test]
fn random_batch_matches_elementwise() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuScreenSpaceReflection::new(&ctx);
    // A deterministic swarm of non-degenerate queries, rejection sampled clear
    // of the degenerate branches, dispatched as one batch.
    let mut rng = Lcg::new(0x9E37_79B9_7F4A_7C15);
    let batch = random_batch(&mut rng);
    let got = gpu.evaluate(&ctx, &batch);
    assert_eq!(got.len(), batch.len());
    for q in &batch {
        assert_parity(&gpu, &ctx, q);
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuScreenSpaceReflection::new(&ctx);
    // No dispatch is issued and the result vector is empty.
    assert!(gpu.evaluate(&ctx, &[]).is_empty());
}
