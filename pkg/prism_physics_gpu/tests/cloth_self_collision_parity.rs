//! Real-device parity for the `GPU` virtual-particle cloth self-collision pass
//! against its `CPU` golden twin.
//!
//! Every test acquires a headless device with `GpuContext::try_headless()` and
//! skips cleanly when no adapter is available (for example inside a sandbox),
//! so the suite is a no-op rather than a failure off a real `GPU`.
//!
//! The pass is a single parallel-safe Jacobi tier of the `NvCloth`
//! virtual-particle method. Cell assignment and bucketing are built on the host
//! and are integer-exact, so the only divergence between device and twin is the
//! separating-push arithmetic (`GPU` fused multiply-add and division/sqrt
//! rounding); parity is therefore checked within a tight tolerance rather than
//! bit-for-bit, the same model the `XPBD` solver uses.
//!
//! Provenance: the virtual-particle technique is the published `NvCloth`
//! method; uniform spatial hashing is the classical Teschner et al. 2003
//! scheme. No Unreal Engine source or derived code.

use glam::Vec3;
use prism_physics_core::{generate_virtual_particles, VirtualParticle, VirtualParticlePattern};
use prism_physics_gpu::context::GpuContext;
use prism_physics_gpu::{
    cpu_cloth_self_collision_jacobi, ClothSelfCollisionScope, GpuClothSelfCollision,
};

/// A tiny deterministic `xorshift64*` generator so the tests need no external
/// crate for reproducible jitter on the cloth layers.
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Rng {
        Rng { state: seed | 1 }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// A float in `[lo, hi)`.
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        let unit = (self.next_u64() >> 40) as f32 / (1_u64 << 24) as f32;
        lo + (hi - lo) * unit
    }
}

/// Relative per-component tolerance: the scatter accumulation is exact on the
/// bit patterns and only the per-pair normalise (division and square root, with
/// `GPU` fused multiply-add in the dot product) can perturb the low bits.
const TOL: f32 = 1.0e-4;

fn close(a: Vec3, b: Vec3) -> bool {
    let scale = a.abs().max(b.abs()).max(Vec3::ONE);
    (a.x - b.x).abs() <= TOL * scale.x
        && (a.y - b.y).abs() <= TOL * scale.y
        && (a.z - b.z).abs() <= TOL * scale.z
}

/// Two stacked `rows x cols` triangle sheets separated by `gap` along `z`, with
/// a little in-plane jitter so the layers are not perfectly aligned. The sheets
/// are closer than the collision thickness, so the pass has real penetrations
/// to resolve. Returns `(positions, inverse_masses, triangles)`.
fn stacked_sheets(
    rows: usize,
    cols: usize,
    spacing: f32,
    gap: f32,
    rng: &mut Rng,
) -> (Vec<Vec3>, Vec<f32>, Vec<[u32; 3]>) {
    let per_layer = rows * cols;
    let mut positions = Vec::with_capacity(2 * per_layer);
    let mut inverse_masses = Vec::with_capacity(2 * per_layer);
    let mut triangles = Vec::new();

    for layer in 0..2 {
        let z = layer as f32 * gap;
        for r in 0..rows {
            for c in 0..cols {
                positions.push(Vec3::new(
                    c as f32 * spacing + rng.range(-0.02, 0.02),
                    r as f32 * spacing + rng.range(-0.02, 0.02),
                    z + rng.range(-0.01, 0.01),
                ));
                // A spread of finite inverse masses, plus the occasional pinned
                // (infinite-mass) vertex to exercise the `w_sum` guard.
                let im = if (r + c) % 7 == 0 {
                    0.0
                } else {
                    rng.range(0.5, 1.5)
                };
                inverse_masses.push(im);
            }
        }
        let base = (layer * per_layer) as u32;
        for r in 0..rows - 1 {
            for c in 0..cols - 1 {
                let v00 = base + (r * cols + c) as u32;
                let v01 = base + (r * cols + c + 1) as u32;
                let v10 = base + ((r + 1) * cols + c) as u32;
                let v11 = base + ((r + 1) * cols + c + 1) as u32;
                triangles.push([v00, v01, v11]);
                triangles.push([v00, v11, v10]);
            }
        }
    }

    (positions, inverse_masses, triangles)
}

fn assert_parity(
    label: &str,
    ctx: &GpuContext,
    kernel: &GpuClothSelfCollision,
    positions: &[Vec3],
    inverse_masses: &[f32],
    virtuals: &[VirtualParticle],
    cell_size: f32,
    thickness: f32,
    scope: ClothSelfCollisionScope,
) {
    let cpu = cpu_cloth_self_collision_jacobi(
        positions,
        inverse_masses,
        virtuals,
        cell_size,
        thickness,
        scope,
    );
    let gpu = kernel.solve(
        ctx,
        positions,
        inverse_masses,
        virtuals,
        cell_size,
        thickness,
        scope,
    );
    assert_eq!(cpu.len(), gpu.len(), "{label}: length mismatch");
    for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        assert!(close(*c, *g), "{label}: vertex {i} cpu {c:?} vs gpu {g:?}");
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log"
)]
fn stacked_cloth_matches_cpu_full_scope() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("no GPU adapter; skipping stacked_cloth_matches_cpu_full_scope");
        return;
    };
    let kernel = GpuClothSelfCollision::new(&ctx);

    let mut rng = Rng::new(0x5e1f_c011);
    let thickness = 0.12_f32;
    let (positions, inverse_masses, triangles) = stacked_sheets(5, 5, 0.1, 0.05, &mut rng);
    let virtuals = generate_virtual_particles(&triangles, &VirtualParticlePattern::nvcloth_default());

    assert_parity(
        "full",
        &ctx,
        &kernel,
        &positions,
        &inverse_masses,
        &virtuals,
        thickness,
        thickness,
        ClothSelfCollisionScope::All,
    );
    eprintln!(
        "cloth self-collision (All): {} verts, {} virtuals",
        positions.len(),
        virtuals.len()
    );
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log"
)]
fn stacked_cloth_matches_cpu_virtual_only_scope() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("no GPU adapter; skipping stacked_cloth_matches_cpu_virtual_only_scope");
        return;
    };
    let kernel = GpuClothSelfCollision::new(&ctx);

    let mut rng = Rng::new(0xa11c_e5ee);
    let thickness = 0.15_f32;
    let (positions, inverse_masses, triangles) = stacked_sheets(6, 4, 0.1, 0.04, &mut rng);
    let virtuals = generate_virtual_particles(&triangles, &VirtualParticlePattern::nvcloth_default());

    // A cell size strictly larger than the thickness still captures the same
    // penetrating pairs but exercises a different host bucketing.
    assert_parity(
        "virtual-only",
        &ctx,
        &kernel,
        &positions,
        &inverse_masses,
        &virtuals,
        thickness * 1.5,
        thickness,
        ClothSelfCollisionScope::VirtualOnly,
    );
    eprintln!(
        "cloth self-collision (VirtualOnly): {} verts, {} virtuals",
        positions.len(),
        virtuals.len()
    );
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log"
)]
fn no_op_inputs_pass_through_unchanged() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("no GPU adapter; skipping no_op_inputs_pass_through_unchanged");
        return;
    };
    let kernel = GpuClothSelfCollision::new(&ctx);

    // Empty scene.
    let empty = kernel.solve(
        &ctx,
        &[],
        &[],
        &[],
        0.1,
        0.1,
        ClothSelfCollisionScope::All,
    );
    assert!(empty.is_empty(), "empty scene should stay empty");

    // A single particle has fewer than two samples: nothing to resolve.
    let one_pos = [Vec3::new(1.0, 2.0, 3.0)];
    let one = kernel.solve(
        &ctx,
        &one_pos,
        &[1.0],
        &[],
        0.1,
        0.1,
        ClothSelfCollisionScope::All,
    );
    assert_eq!(one.len(), 1, "single particle keeps its slot");
    assert!(close(one[0], one_pos[0]), "single particle must not move");

    // Two particles already further apart than the thickness: no penetration,
    // so the pass is a no-op even though it dispatches.
    let apart = [Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0)];
    let resolved = kernel.solve(
        &ctx,
        &apart,
        &[1.0, 1.0],
        &[],
        0.1,
        0.1,
        ClothSelfCollisionScope::All,
    );
    assert!(close(resolved[0], apart[0]), "far pair must not move (0)");
    assert!(close(resolved[1], apart[1]), "far pair must not move (1)");
}
