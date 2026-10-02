//! Real-device parity for the effect-instancing twin:
//! [`GpuInstancing`](prism_volumetric_gpu::instancing::GpuInstancing) must
//! reproduce the `CPU` golden
//! [`instancing`](prism_render_architecture::particle::instancing) across the
//! hand-rolled `Vec3` algebra
//! ([`Vec3::length`](prism_render_architecture::particle::instancing::Vec3::length),
//! [`Vec3::scaled`](prism_render_architecture::particle::instancing::Vec3::scaled),
//! [`Vec3::add`](prism_render_architecture::particle::instancing::Vec3::add) and
//! [`Vec3::approx_eq`](prism_render_architecture::particle::instancing::Vec3::approx_eq)),
//! the uniform-scale localization
//! ([`InstanceTransform::local_to_world_scale`](prism_render_architecture::particle::instancing::InstanceTransform::local_to_world_scale)
//! and
//! [`InstanceTransform::apply`](prism_render_architecture::particle::instancing::InstanceTransform::apply)),
//! and the saturating per-template particle budget
//! ([`EffectTemplate::template_particle_capacity`](prism_render_architecture::particle::instancing::EffectTemplate::template_particle_capacity)
//! and
//! [`instance_particle_upper_bound`](prism_render_architecture::particle::instancing::instance_particle_upper_bound)).
//!
//! The fixtures use non-zero scales and radii, vectors written as integers or
//! simple decimals, and capacity products that sit comfortably below
//! [`u32::MAX`], so every continuous quantity stays far from any degeneracy. One
//! dedicated fixture maxes out the emitter count so the `u32` saturating product
//! overflows and must clamp to [`u32::MAX`], and one `apply` fixture zeroes the
//! uniform scale so the localization collapses to the pure translation. All
//! fixtures stay pure and need no external math library and no transcendental
//! math (`sqrt` on the device aside).
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The continuous answers (length, scaled, add, localized radius and position)
//! thread through multiplies, adds and one `sqrt`, so they are compared under
//! tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`). The
//! discrete answers (the `approx_eq` verdict and the saturating capacities) are
//! compared exactly.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::instancing`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::instancing::{
    instance_particle_upper_bound, EffectTemplate, InstanceTransform, Vec3,
};
use prism_volumetric_gpu::instancing::{GpuInstancing, InstancingQuery, InstancingResult};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous quantities.
const ABS_EPS: f32 = 1.0e-4;
/// Relative tolerance for the continuous quantities.
const REL_EPS: f32 = 1.0e-3;
/// Floor for the relative-tolerance denominator.
const REL_FLOOR: f32 = 1.0e-6;

/// Mixed absolute / relative tolerance comparison for one `f32` lane.
fn approx(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= ABS_EPS {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= REL_EPS
}

/// Tolerant comparison of a `Vec3` answer against a `[f32; 3]` lane triple.
fn approx_vec3(a: Vec3, b: [f32; 3]) -> bool {
    approx(a.x, b[0]) && approx(a.y, b[1]) && approx(a.z, b[2])
}

/// Flattens a `Vec3` into its `[f32; 3]` lane triple.
fn arr(v: Vec3) -> [f32; 3] {
    [v.x, v.y, v.z]
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_instancing_vec3_length_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping instancing parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuInstancing::new(&ctx);
    let v = Vec3::new(3.0, 4.0, 12.0);
    let q = InstancingQuery::Vec3Length { vector: arr(v) };
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(got.len(), 1, "one result per query");
    let cpu = v.length();
    let InstancingResult::Vec3Length { length } = got[0] else {
        panic!("expected a Vec3Length result, got {:?}", got[0]);
    };
    assert!(
        approx(length, cpu),
        "length mismatch: gpu {length} vs cpu {cpu}"
    );
    // The fixture must have a genuinely non-trivial length so the `sqrt` ran.
    assert!(cpu > 1.0, "fixture should exercise a non-trivial length");
}

#[test]
fn gpu_instancing_vec3_scaled_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuInstancing::new(&ctx);
    let v = Vec3::new(1.0, -2.0, 3.0);
    let s = 2.0;
    let q = InstancingQuery::Vec3Scaled {
        vector: arr(v),
        scale: s,
    };
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(got.len(), 1, "one result per query");
    let cpu = v.scaled(s);
    let InstancingResult::Vec3Scaled { vector } = got[0] else {
        panic!("expected a Vec3Scaled result, got {:?}", got[0]);
    };
    assert!(
        approx_vec3(cpu, vector),
        "scaled mismatch: gpu {vector:?} vs cpu {cpu:?}"
    );
}

#[test]
fn gpu_instancing_vec3_add_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuInstancing::new(&ctx);
    let a = Vec3::new(1.0, 2.0, 3.0);
    let b = Vec3::new(4.0, 5.0, 6.0);
    let q = InstancingQuery::Vec3Add {
        lhs: arr(a),
        rhs: arr(b),
    };
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(got.len(), 1, "one result per query");
    let cpu = a.add(b);
    let InstancingResult::Vec3Add { vector } = got[0] else {
        panic!("expected a Vec3Add result, got {:?}", got[0]);
    };
    assert!(
        approx_vec3(cpu, vector),
        "add mismatch: gpu {vector:?} vs cpu {cpu:?}"
    );
}

#[test]
fn gpu_instancing_vec3_approx_eq_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuInstancing::new(&ctx);
    // An equal pair (within `CMP_EPS`) and a clearly unequal pair.
    let a = Vec3::new(1.0, -2.0, 3.5);
    let a_same = Vec3::new(1.0, -2.0, 3.5);
    let b = Vec3::new(1.0, -2.0, 4.5);
    let batch = [
        InstancingQuery::Vec3ApproxEq {
            lhs: arr(a),
            rhs: arr(a_same),
        },
        InstancingQuery::Vec3ApproxEq {
            lhs: arr(a),
            rhs: arr(b),
        },
    ];
    let got = gpu.evaluate(&ctx, &batch);
    assert_eq!(got.len(), 2, "one result per query");

    let cpu_equal = a.approx_eq(a_same);
    let InstancingResult::Vec3ApproxEq { equal } = got[0] else {
        panic!("expected a Vec3ApproxEq result, got {:?}", got[0]);
    };
    assert_eq!(equal, cpu_equal, "approx_eq (equal case) mismatch");
    assert!(cpu_equal, "the equal fixture should report equal");

    let cpu_unequal = a.approx_eq(b);
    let InstancingResult::Vec3ApproxEq { equal } = got[1] else {
        panic!("expected a Vec3ApproxEq result, got {:?}", got[1]);
    };
    assert_eq!(equal, cpu_unequal, "approx_eq (unequal case) mismatch");
    assert!(!cpu_unequal, "the unequal fixture should report unequal");
}

#[test]
fn gpu_instancing_local_to_world_scale_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuInstancing::new(&ctx);
    let transform = InstanceTransform {
        translation: Vec3::new(10.0, 0.0, 0.0),
        uniform_scale: 2.5,
    };
    let local_radius = 4.0;
    let q = InstancingQuery::LocalToWorldScale {
        uniform_scale: transform.uniform_scale,
        local_radius,
    };
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(got.len(), 1, "one result per query");
    let cpu = transform.local_to_world_scale(local_radius);
    let InstancingResult::LocalToWorldScale { scale } = got[0] else {
        panic!("expected a LocalToWorldScale result, got {:?}", got[0]);
    };
    assert!(
        approx(scale, cpu),
        "local_to_world_scale mismatch: gpu {scale} vs cpu {cpu}"
    );
}

#[test]
fn gpu_instancing_apply_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuInstancing::new(&ctx);
    let transform = InstanceTransform {
        translation: Vec3::new(5.0, -3.0, 2.0),
        uniform_scale: 2.0,
    };
    let local = Vec3::new(1.0, 1.0, 1.0);
    let q = InstancingQuery::Apply {
        translation: arr(transform.translation),
        uniform_scale: transform.uniform_scale,
        local: arr(local),
    };
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(got.len(), 1, "one result per query");
    let cpu = transform.apply(local);
    let InstancingResult::Apply { world } = got[0] else {
        panic!("expected an Apply result, got {:?}", got[0]);
    };
    assert!(
        approx_vec3(cpu, world),
        "apply mismatch: gpu {world:?} vs cpu {cpu:?}"
    );
    // A real scale-then-translate must move the local point off the origin.
    assert!(
        cpu.approx_eq(Vec3::new(7.0, -1.0, 4.0)),
        "fixture should exercise a non-trivial scale-then-translate"
    );
}

#[test]
fn gpu_instancing_apply_with_zero_scale_collapses_to_translation() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuInstancing::new(&ctx);
    let transform = InstanceTransform {
        translation: Vec3::new(1.0, 2.0, 3.0),
        uniform_scale: 0.0,
    };
    let local = Vec3::new(9.0, 9.0, 9.0);
    let q = InstancingQuery::Apply {
        translation: arr(transform.translation),
        uniform_scale: transform.uniform_scale,
        local: arr(local),
    };
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(got.len(), 1, "one result per query");
    let cpu = transform.apply(local);
    let InstancingResult::Apply { world } = got[0] else {
        panic!("expected an Apply result, got {:?}", got[0]);
    };
    assert!(
        approx_vec3(cpu, world),
        "zero-scale apply mismatch: gpu {world:?} vs cpu {cpu:?}"
    );
    // A zero uniform scale must collapse the localization to the translation.
    assert!(
        approx_vec3(transform.translation, world),
        "zero-scale apply should equal the pure translation, got {world:?}"
    );
}

#[test]
fn gpu_instancing_template_capacity_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuInstancing::new(&ctx);
    let template = EffectTemplate {
        emitter_count: 3,
        particle_capacity_per_emitter: 64,
    };
    let q = InstancingQuery::TemplateParticleCapacity {
        emitter_count: template.emitter_count,
        particle_capacity_per_emitter: template.particle_capacity_per_emitter,
    };
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(got.len(), 1, "one result per query");
    let cpu = template.template_particle_capacity();
    let InstancingResult::TemplateParticleCapacity { capacity } = got[0] else {
        panic!(
            "expected a TemplateParticleCapacity result, got {:?}",
            got[0]
        );
    };
    assert_eq!(capacity, cpu, "template capacity mismatch");
    assert_eq!(capacity, 192, "fixture capacity should be 3 * 64");
}

#[test]
fn gpu_instancing_instance_upper_bound_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuInstancing::new(&ctx);
    let template = EffectTemplate {
        emitter_count: 5,
        particle_capacity_per_emitter: 100,
    };
    let q = InstancingQuery::InstanceParticleUpperBound {
        emitter_count: template.emitter_count,
        particle_capacity_per_emitter: template.particle_capacity_per_emitter,
    };
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(got.len(), 1, "one result per query");
    let cpu = instance_particle_upper_bound(&template);
    let InstancingResult::InstanceParticleUpperBound { upper_bound } = got[0] else {
        panic!(
            "expected an InstanceParticleUpperBound result, got {:?}",
            got[0]
        );
    };
    assert_eq!(upper_bound, cpu, "instance upper bound mismatch");
    assert_eq!(upper_bound, 500, "fixture upper bound should be 5 * 100");
}

#[test]
fn gpu_instancing_capacity_saturates_in_u32() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuInstancing::new(&ctx);
    // A maxed-out emitter count makes the product overflow a `u32`, which must
    // clamp to `u32::MAX` rather than wrapping to a small, under-allocated pool.
    let template = EffectTemplate {
        emitter_count: u32::MAX,
        particle_capacity_per_emitter: 2,
    };
    let batch = [
        InstancingQuery::TemplateParticleCapacity {
            emitter_count: template.emitter_count,
            particle_capacity_per_emitter: template.particle_capacity_per_emitter,
        },
        InstancingQuery::InstanceParticleUpperBound {
            emitter_count: template.emitter_count,
            particle_capacity_per_emitter: template.particle_capacity_per_emitter,
        },
    ];
    let got = gpu.evaluate(&ctx, &batch);
    assert_eq!(got.len(), 2, "one result per query");

    let cpu_cap = template.template_particle_capacity();
    let InstancingResult::TemplateParticleCapacity { capacity } = got[0] else {
        panic!(
            "expected a TemplateParticleCapacity result, got {:?}",
            got[0]
        );
    };
    assert_eq!(capacity, cpu_cap, "saturating capacity mismatch");
    assert_eq!(capacity, u32::MAX, "overflowing product must clamp");

    let cpu_bound = instance_particle_upper_bound(&template);
    let InstancingResult::InstanceParticleUpperBound { upper_bound } = got[1] else {
        panic!(
            "expected an InstanceParticleUpperBound result, got {:?}",
            got[1]
        );
    };
    assert_eq!(upper_bound, cpu_bound, "saturating upper bound mismatch");
    assert_eq!(upper_bound, u32::MAX, "overflowing product must clamp");
}

#[test]
fn gpu_instancing_batch_matches_elementwise() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuInstancing::new(&ctx);
    // A mixed batch exercises the one-thread-per-query flattening; each result
    // must be independent of its neighbours.
    let v = Vec3::new(2.0, -1.0, 2.0);
    let a = Vec3::new(1.0, 1.0, 1.0);
    let b = Vec3::new(-2.0, 3.0, 0.5);
    let transform = InstanceTransform {
        translation: Vec3::new(4.0, -1.0, 2.0),
        uniform_scale: 1.5,
    };
    let local = Vec3::new(2.0, 0.0, -2.0);
    let template = EffectTemplate {
        emitter_count: 7,
        particle_capacity_per_emitter: 48,
    };
    let batch = [
        InstancingQuery::Vec3Length { vector: arr(v) },
        InstancingQuery::Vec3Scaled {
            vector: arr(v),
            scale: 3.0,
        },
        InstancingQuery::Vec3Add {
            lhs: arr(a),
            rhs: arr(b),
        },
        InstancingQuery::LocalToWorldScale {
            uniform_scale: transform.uniform_scale,
            local_radius: 2.0,
        },
        InstancingQuery::Apply {
            translation: arr(transform.translation),
            uniform_scale: transform.uniform_scale,
            local: arr(local),
        },
        InstancingQuery::TemplateParticleCapacity {
            emitter_count: template.emitter_count,
            particle_capacity_per_emitter: template.particle_capacity_per_emitter,
        },
    ];
    let got = gpu.evaluate(&ctx, &batch);
    assert_eq!(got.len(), batch.len(), "one result per query");

    let cpu_len = v.length();
    let InstancingResult::Vec3Length { length } = got[0] else {
        panic!("expected a Vec3Length result, got {:?}", got[0]);
    };
    assert!(
        approx(length, cpu_len),
        "batch length mismatch: gpu {length} vs cpu {cpu_len}"
    );

    let cpu_scaled = v.scaled(3.0);
    let InstancingResult::Vec3Scaled { vector } = got[1] else {
        panic!("expected a Vec3Scaled result, got {:?}", got[1]);
    };
    assert!(
        approx_vec3(cpu_scaled, vector),
        "batch scaled mismatch: gpu {vector:?} vs cpu {cpu_scaled:?}"
    );

    let cpu_add = a.add(b);
    let InstancingResult::Vec3Add { vector } = got[2] else {
        panic!("expected a Vec3Add result, got {:?}", got[2]);
    };
    assert!(
        approx_vec3(cpu_add, vector),
        "batch add mismatch: gpu {vector:?} vs cpu {cpu_add:?}"
    );

    let cpu_scale = transform.local_to_world_scale(2.0);
    let InstancingResult::LocalToWorldScale { scale } = got[3] else {
        panic!("expected a LocalToWorldScale result, got {:?}", got[3]);
    };
    assert!(
        approx(scale, cpu_scale),
        "batch local_to_world_scale mismatch: gpu {scale} vs cpu {cpu_scale}"
    );

    let cpu_world = transform.apply(local);
    let InstancingResult::Apply { world } = got[4] else {
        panic!("expected an Apply result, got {:?}", got[4]);
    };
    assert!(
        approx_vec3(cpu_world, world),
        "batch apply mismatch: gpu {world:?} vs cpu {cpu_world:?}"
    );

    let cpu_cap = template.template_particle_capacity();
    let InstancingResult::TemplateParticleCapacity { capacity } = got[5] else {
        panic!(
            "expected a TemplateParticleCapacity result, got {:?}",
            got[5]
        );
    };
    assert_eq!(capacity, cpu_cap, "batch capacity mismatch");
}

#[test]
fn gpu_instancing_empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuInstancing::new(&ctx);
    // No dispatch is issued and the result vector is empty.
    assert!(gpu.evaluate(&ctx, &[]).is_empty());
}
