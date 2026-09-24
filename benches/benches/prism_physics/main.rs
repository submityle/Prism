//! Benchmarks for the Prism physics M0 crates.
//!
//! These cover the two performance-critical M0 paths:
//!
//! - `prism_physics_geometry`: dynamic BVH build, overlap query, ray cast, and
//!   broad-phase pair generation throughput.
//! - `prism_physics_core`: semi-implicit Euler integration and a full CPU
//!   backend world step over many dynamic bodies.
//!
//! A tiny deterministic xorshift RNG keeps the scenes reproducible without
//! pulling in an external RNG dependency.

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use std::hint::black_box;

use glam::Vec3;
use prism_physics_core::{BodyDesc, BodyStorage, CpuBackend, Integrator, PhysicsBackend, PhysicsWorld};
use prism_physics_geometry::{generate_pairs, Aabb, DynamicBvh, Ray};

/// Deterministic xorshift64* RNG, used only to lay out reproducible scenes.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_f32()
    }
}

/// Builds `count` random boxes spread through a cube of the given `spread`.
fn random_boxes(count: usize, spread: f32) -> Vec<Aabb> {
    let mut rng = Rng(0x1234_5678_9abc_def0);
    (0..count)
        .map(|_| {
            let center = Vec3::new(
                rng.range(-spread, spread),
                rng.range(-spread, spread),
                rng.range(-spread, spread),
            );
            let half = Vec3::new(
                rng.range(0.25, 2.0),
                rng.range(0.25, 2.0),
                rng.range(0.25, 2.0),
            );
            Aabb::from_center_half_extents(center, half)
        })
        .collect()
}

/// Builds a fully populated BVH from the given boxes.
fn build_bvh(boxes: &[Aabb]) -> DynamicBvh {
    let mut bvh = DynamicBvh::with_capacity(boxes.len() * 2);
    for (i, b) in boxes.iter().enumerate() {
        bvh.insert(*b, i as u64);
    }
    bvh
}

fn bench_bvh_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("prism_physics/bvh_build");
    for &count in &[1_000usize, 10_000] {
        let boxes = random_boxes(count, 100.0);
        group.bench_with_input(BenchmarkId::from_parameter(count), &boxes, |b, boxes| {
            b.iter(|| black_box(build_bvh(black_box(boxes))));
        });
    }
    group.finish();
}

fn bench_bvh_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("prism_physics/bvh_query");
    for &count in &[1_000usize, 10_000] {
        let boxes = random_boxes(count, 100.0);
        let bvh = build_bvh(&boxes);
        let queries = random_boxes(256, 100.0);
        group.bench_with_input(BenchmarkId::from_parameter(count), &bvh, |b, bvh| {
            b.iter(|| {
                let mut hits = 0usize;
                for q in &queries {
                    bvh.query_aabb(*q, &mut |_| hits += 1);
                }
                black_box(hits)
            });
        });
    }
    group.finish();
}

fn bench_bvh_ray_cast(c: &mut Criterion) {
    let mut group = c.benchmark_group("prism_physics/bvh_ray_cast");
    for &count in &[1_000usize, 10_000] {
        let boxes = random_boxes(count, 100.0);
        let bvh = build_bvh(&boxes);
        let mut rng = Rng(0xfeed_face_cafe_babe);
        let rays: Vec<Ray> = (0..256)
            .map(|_| {
                let origin = Vec3::new(
                    rng.range(-120.0, 120.0),
                    rng.range(-120.0, 120.0),
                    rng.range(-120.0, 120.0),
                );
                let dir = Vec3::new(
                    rng.range(-1.0, 1.0),
                    rng.range(-1.0, 1.0),
                    rng.range(-1.0, 1.0),
                );
                Ray::with_tmax(origin, dir, 500.0)
            })
            .collect();
        group.bench_with_input(BenchmarkId::from_parameter(count), &bvh, |b, bvh| {
            b.iter(|| {
                let mut hits = 0usize;
                for ray in &rays {
                    bvh.ray_cast(ray, &mut |_, _| hits += 1);
                }
                black_box(hits)
            });
        });
    }
    group.finish();
}

fn bench_broadphase_pairs(c: &mut Criterion) {
    let mut group = c.benchmark_group("prism_physics/broadphase_pairs");
    for &count in &[1_000usize, 5_000] {
        // A tighter spread makes overlaps common, exercising the pair path.
        let boxes = random_boxes(count, 40.0);
        let bvh = build_bvh(&boxes);
        group.bench_with_input(BenchmarkId::from_parameter(count), &bvh, |b, bvh| {
            b.iter(|| black_box(generate_pairs(black_box(bvh))));
        });
    }
    group.finish();
}

/// Fills a `BodyStorage` with `count` dynamic bodies at random positions.
fn dynamic_storage(count: usize) -> BodyStorage {
    let mut rng = Rng(0x0bad_c0de_dead_10cc);
    let mut bodies = BodyStorage::new();
    for _ in 0..count {
        let p = Vec3::new(
            rng.range(-50.0, 50.0),
            rng.range(-50.0, 50.0),
            rng.range(-50.0, 50.0),
        );
        bodies.insert(BodyDesc::dynamic_at(p));
    }
    bodies
}

fn bench_integrator_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("prism_physics/integrator_step");
    let gravity = Vec3::new(0.0, -9.81, 0.0);
    for &count in &[10_000usize, 100_000] {
        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, &count| {
            b.iter_batched_ref(
                || dynamic_storage(count),
                |bodies| Integrator::integrate(bodies, gravity, 1.0 / 60.0),
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

fn bench_world_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("prism_physics/world_step");
    for &count in &[10_000usize, 100_000] {
        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, &count| {
            b.iter_batched_ref(
                || {
                    let mut world = PhysicsWorld::with_gravity(Vec3::new(0.0, -9.81, 0.0));
                    world.bodies = dynamic_storage(count);
                    (world, CpuBackend::with_defaults())
                },
                |(world, backend)| backend.step(world, 1.0 / 60.0),
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_bvh_build,
    bench_bvh_query,
    bench_bvh_ray_cast,
    bench_broadphase_pairs,
    bench_integrator_step,
    bench_world_step,
);
criterion_main!(benches);
