//! Real-device parity for the mesh-bounds reduction twin: [`GpuMeshAabb`] must
//! reproduce the `CPU` golden
//! [`Mesh::aabb`](prism_render_architecture::particle::mesh_emission::Mesh::aabb)
//! across a single vertex, the empty mesh, and large random vertex clouds under
//! several block sizes.
//!
//! The tests skip (with a printed notice on the first) when the host has no
//! `wgpu` adapter, so the suite stays green everywhere while still exercising
//! the full dispatch-and-readback on any real device. The kernel is portable
//! core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The reduction is a component-wise `min` / `max` fold, and both `min` and
//! `max` *select* one of their inputs rather than compute a new value, so the
//! `CPU` and `GPU` results are bit-identical regardless of fold order. The tests
//! therefore assert an exact match on the raw `f32` bit patterns (an integer
//! `u32` comparison of `f32::to_bits`), with no tolerance — the strongest
//! possible parity statement.
//!
//! # Fixtures
//!
//! The deterministic `LCG` fixtures generate vertex coordinates in a range that
//! stays clear of zero so no component is a signed zero whose `min` / `max`
//! selection could be ambiguous. Vertex counts span a single vertex, the empty
//! mesh and several large clouds, and the large clouds are reduced under block
//! sizes `1`, `7`, `64` and `256` to exercise the host final aggregation across
//! one, a few and many blocks. The fixtures use no external math library and no
//! transcendental method (pure integer `LCG` plus a scale and shift).
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::mesh_emission::Mesh::aabb` 的真机 parity；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::mesh_emission::{Mesh, MeshVertex};
use prism_render_architecture::particle::Vec3;
use prism_volumetric_gpu::mesh_aabb::{GpuMeshAabb, GpuMeshAabbQuery};
use prism_volumetric_gpu::GpuContext;

/// A tiny deterministic linear-congruential generator so the "random" fixtures
/// are reproducible run to run without pulling in an external crate. The
/// constants are the Numerical Recipes `LCG` multiplier and increment.
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

    /// A reproducible `f32` in `[0, 1)`.
    fn next_unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }

    /// A reproducible `f32` in `[lo, hi)`.
    fn next_range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.next_unit() * (hi - lo)
    }
}

/// Builds a reproducible cloud of `count` vertices with coordinates in a range
/// that stays clear of zero (so no component is a signed zero).
fn random_positions(seed: u32, count: usize) -> Vec<Vec3> {
    let mut rng = Lcg::new(seed);
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        // Two disjoint sub-ranges, both bounded away from zero, chosen per
        // component so the box is non-degenerate and no coordinate is zero.
        let x = rng.next_range(10.0, 500.0);
        let y = rng.next_range(-500.0, -10.0);
        let z = rng.next_range(10.0, 500.0);
        out.push(Vec3::new(x, y, z));
    }
    out
}

/// Builds the golden mesh bounds for a set of positions. Indices are irrelevant
/// to [`Mesh::aabb`], which folds every vertex position, so the mesh is built
/// with no triangles.
fn golden_aabb(positions: &[Vec3]) -> (Vec3, Vec3) {
    let vertices: Vec<MeshVertex> = positions.iter().map(|&p| MeshVertex::at(p)).collect();
    Mesh::new(vertices, Vec::new()).aabb()
}

/// Asserts two [`Vec3`] match bit for bit (an integer comparison of each
/// component's `f32::to_bits`).
fn assert_bits_eq(label: &str, cpu: Vec3, gpu: Vec3) {
    assert_eq!(
        cpu.x.to_bits(),
        gpu.x.to_bits(),
        "{label}.x: cpu {cpu:?}, gpu {gpu:?}"
    );
    assert_eq!(
        cpu.y.to_bits(),
        gpu.y.to_bits(),
        "{label}.y: cpu {cpu:?}, gpu {gpu:?}"
    );
    assert_eq!(
        cpu.z.to_bits(),
        gpu.z.to_bits(),
        "{label}.z: cpu {cpu:?}, gpu {gpu:?}"
    );
}

/// Runs one batch on both sides and asserts bit-exact bounds parity.
fn check_batch(
    label: &str,
    engine: &GpuMeshAabb,
    ctx: &GpuContext,
    positions: &[Vec3],
    block: u32,
) {
    let (cpu_min, cpu_max) = golden_aabb(positions);
    let q = GpuMeshAabbQuery {
        positions: positions.to_vec(),
        block_size: block,
    };
    let gpu = engine.evaluate(ctx, &q);
    assert_bits_eq(&format!("{label}.min"), cpu_min, gpu.min);
    assert_bits_eq(&format!("{label}.max"), cpu_max, gpu.max);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_matches_cpu_across_large_clouds() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping mesh-aabb parity: no wgpu adapter on this host");
        return;
    };
    let engine = GpuMeshAabb::new(&ctx);
    for (seed, count) in [257usize, 1000, 4096, 63].into_iter().enumerate() {
        let positions = random_positions(0x6EED_u32.wrapping_add(seed as u32), count);
        for block in [1u32, 7, 64, 256] {
            check_batch(
                &format!("cloud {seed} (count={count}, block={block})"),
                &engine,
                &ctx,
                &positions,
                block,
            );
        }
    }
}

#[test]
fn gpu_matches_cpu_on_single_vertex() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuMeshAabb::new(&ctx);
    let positions = vec![Vec3::new(-3.5, 12.25, 0.75)];
    // For a single vertex min == max == that position; try several block sizes.
    for block in [0u32, 1, 64] {
        check_batch("single vertex", &engine, &ctx, &positions, block);
    }
    // Structural check: the box collapses to the vertex exactly.
    let gpu = engine.evaluate(
        &ctx,
        &GpuMeshAabbQuery {
            positions: positions.clone(),
            block_size: 0,
        },
    );
    assert_bits_eq("single vertex min", positions[0], gpu.min);
    assert_bits_eq("single vertex max", positions[0], gpu.max);
}

#[test]
fn gpu_matches_cpu_on_small_counts() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuMeshAabb::new(&ctx);
    // A spread of small counts, including a block size larger than the count
    // (one block), equal to it, and smaller than it (several blocks).
    for count in [2usize, 3, 5, 8, 65] {
        let positions = random_positions(0x1234_u32.wrapping_add(count as u32), count);
        for block in [0u32, 1, 3, 64] {
            check_batch(
                &format!("small count={count}, block={block}"),
                &engine,
                &ctx,
                &positions,
                block,
            );
        }
    }
}

#[test]
fn gpu_matches_cpu_on_empty_mesh() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuMeshAabb::new(&ctx);
    let gpu = engine.evaluate(
        &ctx,
        &GpuMeshAabbQuery {
            positions: Vec::new(),
            block_size: 0,
        },
    );
    // The golden returns (ZERO, ZERO) for an empty mesh; the twin short-circuits
    // to the same, with no dispatch issued.
    let (cpu_min, cpu_max) = golden_aabb(&[]);
    assert_bits_eq("empty min", cpu_min, gpu.min);
    assert_bits_eq("empty max", cpu_max, gpu.max);
    assert_bits_eq("empty min is zero", Vec3::ZERO, gpu.min);
    assert_bits_eq("empty max is zero", Vec3::ZERO, gpu.max);
}
