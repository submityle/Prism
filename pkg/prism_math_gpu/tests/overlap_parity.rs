//! Real-device parity for the §24.1 / §24.7 broad-phase overlap shader mirror.
//!
//! Each test evaluates a batch of primitive-pair overlap predicates on a real
//! `GPU` from the single-sourced
//! [`WGSL_OVERLAP`](prism_math::shader_mirror::WGSL_OVERLAP) fragment and diffs
//! the booleans against the CPU references
//! [`prism_math::intersect::aabb_aabb`] /
//! [`sphere_sphere`](prism_math::intersect::sphere_sphere) /
//! [`sphere_aabb`](prism_math::intersect::sphere_aabb). The predicates are pure
//! comparisons and dot products, so for pairs with a comfortable margin from
//! exact tangency the discrete boolean agrees exactly (the only fast-math
//! sensitivity is a pair poised within rounding of touching, excluded here by
//! construction). The suite skips gracefully when no adapter is available.

use prism_math::geom::aabb::Aabb3;
use prism_math::geom::sphere::BoundingSphere;
use prism_math::intersect::{aabb_aabb, sphere_aabb, sphere_sphere};
use prism_math::Vec3;
use prism_math_gpu::GpuContext;
use prism_math_gpu::GpuOverlap;

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

/// A small deterministic linear-congruential sequence for pseudo-random floats.
fn lcg(seed: &mut u32) -> f32 {
    *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    // Map to roughly [-10, 10); continuous values stay clear of exact tangency.
    ((*seed >> 8) as f32 / (1u32 << 24) as f32) * 20.0 - 10.0
}

#[test]
fn aabb_aabb_matches_cpu_reference() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuOverlap::new(&ctx);

    let mut pairs: Vec<[[f32; 4]; 4]> = Vec::new();
    let mut refs: Vec<(Aabb3, Aabb3)> = Vec::new();
    let mut seed = 0x51ed_1234u32;
    for _ in 0..600 {
        let ac = Vec3::new(lcg(&mut seed), lcg(&mut seed), lcg(&mut seed));
        let ah = Vec3::new(
            lcg(&mut seed).abs() + 0.5,
            lcg(&mut seed).abs() + 0.5,
            lcg(&mut seed).abs() + 0.5,
        );
        let bc = Vec3::new(lcg(&mut seed), lcg(&mut seed), lcg(&mut seed));
        let bh = Vec3::new(
            lcg(&mut seed).abs() + 0.5,
            lcg(&mut seed).abs() + 0.5,
            lcg(&mut seed).abs() + 0.5,
        );
        let a = Aabb3::new(ac - ah, ac + ah);
        let b = Aabb3::new(bc - bh, bc + bh);
        pairs.push([
            [a.min.x, a.min.y, a.min.z, 0.0],
            [a.max.x, a.max.y, a.max.z, 0.0],
            [b.min.x, b.min.y, b.min.z, 0.0],
            [b.max.x, b.max.y, b.max.z, 0.0],
        ]);
        refs.push((a, b));
    }

    let gpu = kernel.aabb_aabb(&ctx, &pairs);
    assert_eq!(gpu.len(), refs.len());
    let mut hits = 0usize;
    for (&g, &(a, b)) in gpu.iter().zip(refs.iter()) {
        let cpu = aabb_aabb(a, b);
        assert_eq!(g, cpu, "aabb_aabb drift for {a:?} vs {b:?}");
        hits += usize::from(cpu);
    }
    // Sanity: the batch exercises both outcomes, not a degenerate all-one side.
    assert!(
        hits > 0 && hits < refs.len(),
        "degenerate batch: {hits} hits"
    );
}

#[test]
fn sphere_sphere_matches_cpu_reference() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuOverlap::new(&ctx);

    let mut pairs: Vec<[[f32; 4]; 2]> = Vec::new();
    let mut refs: Vec<(BoundingSphere, BoundingSphere)> = Vec::new();
    let mut seed = 0x0bad_cafeu32;
    for _ in 0..600 {
        let a = BoundingSphere::new(
            Vec3::new(lcg(&mut seed), lcg(&mut seed), lcg(&mut seed)),
            lcg(&mut seed).abs() + 0.5,
        );
        let b = BoundingSphere::new(
            Vec3::new(lcg(&mut seed), lcg(&mut seed), lcg(&mut seed)),
            lcg(&mut seed).abs() + 0.5,
        );
        pairs.push([
            [a.center.x, a.center.y, a.center.z, a.radius],
            [b.center.x, b.center.y, b.center.z, b.radius],
        ]);
        refs.push((a, b));
    }

    let gpu = kernel.sphere_sphere(&ctx, &pairs);
    assert_eq!(gpu.len(), refs.len());
    let mut hits = 0usize;
    for (&g, &(a, b)) in gpu.iter().zip(refs.iter()) {
        let cpu = sphere_sphere(a, b);
        assert_eq!(g, cpu, "sphere_sphere drift for {a:?} vs {b:?}");
        hits += usize::from(cpu);
    }
    assert!(
        hits > 0 && hits < refs.len(),
        "degenerate batch: {hits} hits"
    );
}

#[test]
fn sphere_aabb_matches_cpu_reference() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuOverlap::new(&ctx);

    let mut pairs: Vec<[[f32; 4]; 3]> = Vec::new();
    let mut refs: Vec<(BoundingSphere, Aabb3)> = Vec::new();
    let mut seed = 0x1337_0042u32;
    for _ in 0..600 {
        let s = BoundingSphere::new(
            Vec3::new(lcg(&mut seed), lcg(&mut seed), lcg(&mut seed)),
            lcg(&mut seed).abs() + 0.5,
        );
        let bc = Vec3::new(lcg(&mut seed), lcg(&mut seed), lcg(&mut seed));
        let bh = Vec3::new(
            lcg(&mut seed).abs() + 0.5,
            lcg(&mut seed).abs() + 0.5,
            lcg(&mut seed).abs() + 0.5,
        );
        let b = Aabb3::new(bc - bh, bc + bh);
        pairs.push([
            [s.center.x, s.center.y, s.center.z, s.radius],
            [b.min.x, b.min.y, b.min.z, 0.0],
            [b.max.x, b.max.y, b.max.z, 0.0],
        ]);
        refs.push((s, b));
    }

    let gpu = kernel.sphere_aabb(&ctx, &pairs);
    assert_eq!(gpu.len(), refs.len());
    let mut hits = 0usize;
    for (&g, &(s, b)) in gpu.iter().zip(refs.iter()) {
        let cpu = sphere_aabb(s, b);
        assert_eq!(g, cpu, "sphere_aabb drift for {s:?} vs {b:?}");
        hits += usize::from(cpu);
    }
    assert!(
        hits > 0 && hits < refs.len(),
        "degenerate batch: {hits} hits"
    );
}

#[test]
fn empty_batch_is_empty() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuOverlap::new(&ctx);
    assert!(kernel.aabb_aabb(&ctx, &[]).is_empty());
    assert!(kernel.sphere_sphere(&ctx, &[]).is_empty());
    assert!(kernel.sphere_aabb(&ctx, &[]).is_empty());
}
