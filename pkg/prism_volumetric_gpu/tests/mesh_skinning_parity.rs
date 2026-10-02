//! Real-device parity for the linear-blend skinning twin:
//! [`GpuMeshSkinning`](prism_volumetric_gpu::mesh_skinning::GpuMeshSkinning) must
//! reproduce the `CPU` golden
//! [`mesh_emission`](prism_render_architecture::particle::mesh_emission) skinning
//! pure functions value for value — the skinned position and the skinned,
//! renormalized normal.
//!
//! The fixtures cover the degenerate and interior shapes the golden unit tests
//! exercise: an empty vertex batch (no dispatch), rigid single-bone vertices
//! under non-identity bones, a multi-bone blend with weights clearly above the
//! skip threshold, an out-of-range bone index (that influence skipped), a
//! near-zero weight (that influence skipped), a degenerate zero normal that
//! renormalizes to zero, an empty bone palette (every influence skipped) and a
//! larger deterministic sweep. Weights are kept clearly above or below the skip
//! threshold `EPS` and the non-degenerate normals keep a blended length well
//! away from the renormalize floor, so no fixture lands on a branch-critical
//! point.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each output is a fixed sequence of multiply-adds plus the normal's
//! `sqrt`-based renormalize, so `CPU` and `GPU` evaluate the same closed form but
//! need not be bit-exact (a `GPU` may contract a multiply-add). The comparison is
//! per component with `abs_diff <= 1e-4 || rel_diff <= 1e-3`
//! (`REL_FLOOR = 1e-6`), tight enough to catch a genuinely wrong port yet loose
//! enough to admit legal fused multiply-add contraction.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::mesh_emission` 的
//! 线性混合蒙皮纯函数；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::mesh_emission::{
    skin_normal, skin_position, BoneTransform, SkinnedVertex,
};
use prism_render_architecture::particle::Vec3;
use prism_volumetric_gpu::mesh_skinning::{GpuMeshSkinning, GpuSkinResult};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous-quantity parity comparison.
const ABS_TOL: f32 = 1.0e-4;
/// Relative tolerance for the continuous-quantity parity comparison.
const REL_TOL: f32 = 1.0e-3;
/// Floor on the relative-tolerance denominator so values near zero still use a
/// meaningful scale.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns `true` when two continuous values agree within the module tolerance.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= ABS_TOL {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= REL_TOL
}

/// Asserts two vectors agree component by component within tolerance.
fn assert_vec3_close(gpu: Vec3, cpu: Vec3, label: &str) {
    assert!(
        close(gpu.x, cpu.x) && close(gpu.y, cpu.y) && close(gpu.z, cpu.z),
        "{label}: GPU ({}, {}, {}) vs CPU ({}, {}, {})",
        gpu.x,
        gpu.y,
        gpu.z,
        cpu.x,
        cpu.y,
        cpu.z
    );
}

/// A small integer `LCG` used to synthesize deterministic vertices and bones
/// without any external math library or transcendental call.
struct Lcg {
    /// The current `64`-bit state.
    state: u64,
}

impl Lcg {
    /// Seeds the generator, forcing an odd state so the stream never degenerates.
    fn new(seed: u64) -> Self {
        Lcg { state: seed | 1 }
    }

    /// Advances the state and returns the high `32` bits.
    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.state >> 32) as u32
    }

    /// A deterministic value in roughly `[-1, 1)`, pure integer math scaled to
    /// `f32` with no transcendental call.
    fn unit(&mut self) -> f32 {
        let n = (self.next_u32() % 2000) as f32;
        n * 0.001 - 1.0
    }

    /// A deterministic value in roughly `[-5, 5)`, pure integer math scaled to
    /// `f32` with no transcendental call.
    fn signed(&mut self) -> f32 {
        let n = (self.next_u32() % 1000) as f32;
        n * 0.01 - 5.0
    }
}

/// Builds a deterministic, well-conditioned bone palette: diagonally dominant
/// basis columns (so a blend never collapses the normal) plus a random
/// translation.
fn make_bones(count: usize, seed: u64) -> Vec<BoneTransform> {
    let mut rng = Lcg::new(seed);
    let mut bones = Vec::with_capacity(count);
    for _ in 0..count {
        bones.push(BoneTransform::new(
            Vec3::new(1.0 + 0.2 * rng.unit(), 0.1 * rng.unit(), 0.1 * rng.unit()),
            Vec3::new(0.1 * rng.unit(), 1.0 + 0.2 * rng.unit(), 0.1 * rng.unit()),
            Vec3::new(0.1 * rng.unit(), 0.1 * rng.unit(), 1.0 + 0.2 * rng.unit()),
            Vec3::new(rng.signed(), rng.signed(), rng.signed()),
        ));
    }
    bones
}

/// Compares the full `GPU` batch against the golden functions, vertex by vertex.
fn assert_parity(
    gpu: &[GpuSkinResult],
    vertices: &[SkinnedVertex],
    bones: &[BoneTransform],
    label: &str,
) {
    assert_eq!(gpu.len(), vertices.len(), "{label}: result count");
    for (i, g) in gpu.iter().enumerate() {
        let cpu_pos = skin_position(&vertices[i], bones);
        let cpu_nrm = skin_normal(&vertices[i], bones);
        assert_vec3_close(g.position, cpu_pos, &format!("{label} position[{i}]"));
        assert_vec3_close(g.normal, cpu_nrm, &format!("{label} normal[{i}]"));
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let skinner = GpuMeshSkinning::new(&ctx);
    let bones = make_bones(3, 0x5EED_0001);
    let out = skinner.skin(&ctx, &[], &bones);
    assert!(
        out.is_empty(),
        "empty vertex batch must return an empty vector"
    );
}

#[test]
fn rigid_single_bone_parity() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let skinner = GpuMeshSkinning::new(&ctx);
    let bones = make_bones(4, 0x5EED_0002);
    let mut rng = Lcg::new(0xABCD_0002);
    let mut vertices = Vec::new();
    for k in 0..16u16 {
        let bone = k % 4;
        vertices.push(SkinnedVertex::rigid(
            Vec3::new(rng.signed(), rng.signed(), rng.signed()),
            // Non-zero normal (components near 1) so renormalize stays well away
            // from the degenerate floor.
            Vec3::new(
                1.0 + 0.2 * rng.unit(),
                1.0 + 0.2 * rng.unit(),
                1.0 + 0.2 * rng.unit(),
            ),
            [0.0, 0.0],
            bone,
        ));
    }
    let out = skinner.skin(&ctx, &vertices, &bones);
    assert_parity(&out, &vertices, &bones, "rigid");
}

#[test]
fn multi_bone_blend_parity() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let skinner = GpuMeshSkinning::new(&ctx);
    let bones = make_bones(6, 0x5EED_0003);
    let mut rng = Lcg::new(0xABCD_0003);
    let mut vertices = Vec::new();
    for _ in 0..24 {
        // Four equal weights, each clearly above the skip threshold.
        vertices.push(SkinnedVertex {
            position: Vec3::new(rng.signed(), rng.signed(), rng.signed()),
            normal: Vec3::new(
                0.5 + 0.3 * rng.unit(),
                0.5 + 0.3 * rng.unit(),
                0.5 + 0.3 * rng.unit(),
            ),
            uv: [0.0, 0.0],
            weights: [0.25, 0.25, 0.25, 0.25],
            bones: [0, 1, 2, 3],
        });
    }
    let out = skinner.skin(&ctx, &vertices, &bones);
    assert_parity(&out, &vertices, &bones, "multi-bone");
}

#[test]
fn out_of_range_bone_index_skipped() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let skinner = GpuMeshSkinning::new(&ctx);
    // Palette of three bones; index 99 is deliberately out of range.
    let bones = make_bones(3, 0x5EED_0004);
    let mut rng = Lcg::new(0xABCD_0004);
    let mut vertices = Vec::new();
    for _ in 0..12 {
        vertices.push(SkinnedVertex {
            position: Vec3::new(rng.signed(), rng.signed(), rng.signed()),
            normal: Vec3::new(
                1.0 + 0.2 * rng.unit(),
                1.0 + 0.2 * rng.unit(),
                1.0 + 0.2 * rng.unit(),
            ),
            uv: [0.0, 0.0],
            // The second influence (weight clearly above threshold) points past
            // the palette and must be skipped on both sides.
            weights: [0.6, 0.4, 0.0, 0.0],
            bones: [1, 99, 0, 2],
        });
    }
    let out = skinner.skin(&ctx, &vertices, &bones);
    assert_parity(&out, &vertices, &bones, "out-of-range");
}

#[test]
fn near_zero_weight_skipped() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let skinner = GpuMeshSkinning::new(&ctx);
    let bones = make_bones(4, 0x5EED_0005);
    let mut rng = Lcg::new(0xABCD_0005);
    let mut vertices = Vec::new();
    for _ in 0..12 {
        vertices.push(SkinnedVertex {
            position: Vec3::new(rng.signed(), rng.signed(), rng.signed()),
            normal: Vec3::new(
                1.0 + 0.2 * rng.unit(),
                1.0 + 0.2 * rng.unit(),
                1.0 + 0.2 * rng.unit(),
            ),
            uv: [0.0, 0.0],
            // The third weight is far below the skip threshold (`1e-9 < EPS`) and
            // must be skipped on both sides; the valid weights stay well above it.
            weights: [0.5, 0.5, 1.0e-9, 0.0],
            bones: [0, 1, 2, 3],
        });
    }
    let out = skinner.skin(&ctx, &vertices, &bones);
    assert_parity(&out, &vertices, &bones, "near-zero-weight");
}

#[test]
fn degenerate_zero_normal_renormalizes_to_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let skinner = GpuMeshSkinning::new(&ctx);
    let bones = make_bones(4, 0x5EED_0006);
    let mut rng = Lcg::new(0xABCD_0006);
    let mut vertices = Vec::new();
    for _ in 0..8 {
        // A zero authored normal: the blended normal stays zero and renormalizes
        // to zero, exercising the degenerate branch; the position is unaffected.
        vertices.push(SkinnedVertex {
            position: Vec3::new(rng.signed(), rng.signed(), rng.signed()),
            normal: Vec3::ZERO,
            uv: [0.0, 0.0],
            weights: [0.25, 0.25, 0.25, 0.25],
            bones: [0, 1, 2, 3],
        });
    }
    let out = skinner.skin(&ctx, &vertices, &bones);
    assert_parity(&out, &vertices, &bones, "degenerate-normal");
    for (i, g) in out.iter().enumerate() {
        assert_vec3_close(
            g.normal,
            Vec3::ZERO,
            &format!("degenerate normal[{i}] must be zero"),
        );
    }
}

#[test]
fn empty_bone_palette_skips_all() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let skinner = GpuMeshSkinning::new(&ctx);
    let bones: Vec<BoneTransform> = Vec::new();
    let mut rng = Lcg::new(0xABCD_0007);
    let mut vertices = Vec::new();
    for _ in 0..8 {
        vertices.push(SkinnedVertex {
            position: Vec3::new(rng.signed(), rng.signed(), rng.signed()),
            normal: Vec3::new(1.0, 2.0, 3.0),
            uv: [0.0, 0.0],
            weights: [0.25, 0.25, 0.25, 0.25],
            bones: [0, 1, 2, 3],
        });
    }
    let out = skinner.skin(&ctx, &vertices, &bones);
    // With no bones every influence is skipped, so position and normal are zero.
    assert_parity(&out, &vertices, &bones, "empty-palette");
    for (i, g) in out.iter().enumerate() {
        assert_vec3_close(
            g.position,
            Vec3::ZERO,
            &format!("empty palette position[{i}]"),
        );
        assert_vec3_close(g.normal, Vec3::ZERO, &format!("empty palette normal[{i}]"));
    }
}

#[test]
fn large_deterministic_sweep_parity() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let skinner = GpuMeshSkinning::new(&ctx);
    let bone_count = 12usize;
    let bones = make_bones(bone_count, 0x5EED_0008);
    let mut rng = Lcg::new(0xABCD_0008);
    let mut vertices = Vec::new();
    for _ in 0..257 {
        // Random weights clearly above the skip threshold (each in [0.1, 0.5)).
        let w0 = 0.1 + 0.4 * (rng.unit() * 0.5 + 0.5);
        let w1 = 0.1 + 0.4 * (rng.unit() * 0.5 + 0.5);
        let w2 = 0.1 + 0.4 * (rng.unit() * 0.5 + 0.5);
        let w3 = 0.1 + 0.4 * (rng.unit() * 0.5 + 0.5);
        // Random in-range bone indices.
        let b0 = (rng.next_u32() as usize % bone_count) as u16;
        let b1 = (rng.next_u32() as usize % bone_count) as u16;
        let b2 = (rng.next_u32() as usize % bone_count) as u16;
        let b3 = (rng.next_u32() as usize % bone_count) as u16;
        // Non-degenerate normal (components near 1) so renormalize stays away
        // from the floor.
        vertices.push(SkinnedVertex {
            position: Vec3::new(rng.signed(), rng.signed(), rng.signed()),
            normal: Vec3::new(
                1.0 + 0.3 * rng.unit(),
                1.0 + 0.3 * rng.unit(),
                1.0 + 0.3 * rng.unit(),
            ),
            uv: [0.0, 0.0],
            weights: [w0, w1, w2, w3],
            bones: [b0, b1, b2, b3],
        });
    }
    let out = skinner.skin(&ctx, &vertices, &bones);
    assert_parity(&out, &vertices, &bones, "sweep");
}
