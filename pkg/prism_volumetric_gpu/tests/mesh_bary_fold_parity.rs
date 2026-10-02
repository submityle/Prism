//! Real-device parity for the barycentric-fold twin: [`GpuMeshBaryFold`] must
//! reproduce the `CPU` golden unit-square fold plus a triangle barycentric
//! interpolation across random draws on both sides of the reflection line, the
//! structural triangle corners and centroid, and the degenerate empty batch.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable
//! core-`WGSL`, so it needs no optional device feature.
//!
//! # Private golden mirror
//!
//! The reference `fold_barycentric` is a *private* `fn` in
//! [`mesh_emission`](prism_render_architecture::particle::mesh_emission) and
//! cannot be imported. This file therefore carries `golden_fold_barycentric`, a
//! line-for-line transcription of that private helper, annotated as a mirror;
//! the interpolation reference is formed by hand from the same mirrored weights
//! in the fixed order `w0*p0 + w1*p1 + w2*p2`.
//!
//! # Parity criterion
//!
//! The fold and interpolation are comparison plus `+ − ×` only, accumulated in
//! the identical order the mirror uses, so `CPU` and `GPU` evaluate the same
//! algebra. Values are asserted to within `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3` — loose enough to admit a `GPU` fused multiply-add, yet
//! tight enough to fail a wrong port (a dropped reflection, a swapped weight, a
//! reordered interpolation).
//!
//! # Fixtures
//!
//! The deterministic `LCG` fixtures reject-sample so each `(u, v)` sits a margin
//! clear of the reflection line `u + v = 1`, exercising both branches without
//! letting a rounding difference flip one; dedicated structural cases pin the
//! three corners and the centroid, where the weights are exact and sum to one.
//! The fixtures use no external math library and no transcendental method.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::mesh_emission`
//! 的私有重心折叠真机 parity；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::Vec3;
use prism_volumetric_gpu::mesh_bary_fold::{GpuMeshBaryFold, GpuMeshBaryFoldQuery};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity tolerance. Chosen a decade above the single multiply-add
/// rounding so a legal `GPU` fused multiply-add stays inside it while a
/// genuinely wrong port falls outside.
const ABS_EPS: f32 = 1.0e-4;

/// Relative parity tolerance, applied for samples large enough that the
/// absolute floor is pessimistic.
const REL_EPS: f32 = 1.0e-3;

/// Relative-tolerance floor so near-zero references do not divide by a tiny
/// magnitude.
const REL_FLOOR: f32 = 1.0e-6;

// MIRROR of mesh_emission.rs::fold_barycentric (private, exact transcription).
// Folds two unit draws into uniform barycentric weights `(w0, w1, w2)` via the
// standard reflection `if u + v > 1 { u = 1 - u; v = 1 - v }`. Returned as a
// `[f32; 3]` so this test needs no access to the private golden symbol.
//
// Provenance: 逐行精确转抄本仓 prism_render_architecture 的私有
// `particle::mesh_emission::fold_barycentric`；无第三方引擎源码或衍生代码。
fn golden_fold_barycentric(u: f32, v: f32) -> [f32; 3] {
    let (su, sv) = if u + v > 1.0 {
        (1.0 - u, 1.0 - v)
    } else {
        (u, v)
    };
    [1.0 - su - sv, su, sv]
}

/// Hand interpolation reference `w0*p0 + w1*p1 + w2*p2` in the fixed
/// left-to-right order the kernel uses.
fn golden_interpolate(w: [f32; 3], tri: [Vec3; 3]) -> Vec3 {
    tri[0]
        .scale(w[0])
        .add(tri[1].scale(w[1]))
        .add(tri[2].scale(w[2]))
}

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

/// Asserts two scalars match within the documented tolerance.
fn assert_close(label: &str, cpu: f32, gpu: f32) {
    let abs_diff = (cpu - gpu).abs();
    let rel_diff = abs_diff / cpu.abs().max(REL_FLOOR);
    assert!(
        abs_diff <= ABS_EPS || rel_diff <= REL_EPS,
        "{label}: mismatch cpu {cpu}, gpu {gpu} (abs {abs_diff}, rel {rel_diff})"
    );
}

/// Asserts two [`Vec3`] match within the documented tolerance.
fn assert_vec3(label: &str, cpu: Vec3, gpu: Vec3) {
    assert_close(&format!("{label}.x"), cpu.x, gpu.x);
    assert_close(&format!("{label}.y"), cpu.y, gpu.y);
    assert_close(&format!("{label}.z"), cpu.z, gpu.z);
}

/// Runs one batch on the device and asserts every sample matches the mirrored
/// fold and the hand interpolation, and that the weights sum to one.
fn check_batch(label: &str, engine: &GpuMeshBaryFold, ctx: &GpuContext, q: &GpuMeshBaryFoldQuery) {
    let gpu = engine.fold(ctx, q);
    assert_eq!(gpu.weights.len(), q.uvs.len(), "{label}: weight count");
    assert_eq!(gpu.positions.len(), q.uvs.len(), "{label}: position count");
    for i in 0..q.uvs.len() {
        let [u, v] = q.uvs[i];
        let w = golden_fold_barycentric(u, v);
        let cpu_w = Vec3::new(w[0], w[1], w[2]);
        let cpu_pos = golden_interpolate(w, q.triangles[i]);
        assert_vec3(&format!("{label}[{i}].weight"), cpu_w, gpu.weights[i]);
        assert_vec3(&format!("{label}[{i}].position"), cpu_pos, gpu.positions[i]);
        // The folded weights are a convex combination, so they must sum to one.
        let sum = gpu.weights[i].x + gpu.weights[i].y + gpu.weights[i].z;
        assert_close(&format!("{label}[{i}].weight_sum"), 1.0, sum);
    }
}

/// A fixed sample triangle used across the fixtures.
fn tri() -> [Vec3; 3] {
    [
        Vec3::new(-1.0, 0.0, 2.0),
        Vec3::new(3.0, 1.0, -0.5),
        Vec3::new(0.5, 4.0, 1.5),
    ]
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_matches_cpu_across_random_draws() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping mesh-bary-fold parity: no wgpu adapter on this host");
        return;
    };
    let engine = GpuMeshBaryFold::new(&ctx);
    let triangle = tri();
    // Margin keeping each draw clear of the reflection line u + v = 1.
    let margin = 0.05f32;
    for (seed, count) in [41usize, 150, 256, 7].into_iter().enumerate() {
        let mut rng = Lcg::new(0x51ED_u32.wrapping_add(seed as u32));
        let mut uvs = Vec::with_capacity(count);
        let mut triangles = Vec::with_capacity(count);
        for i in 0..count {
            // Alternate below / above the reflection line so both branches run.
            let below = i % 2 == 0;
            let (u, v) = loop {
                let u = rng.next_range(0.0, 1.0);
                let v = rng.next_range(0.0, 1.0);
                let s = u + v;
                if below && s < 1.0 - margin {
                    break (u, v);
                }
                if !below && s > 1.0 + margin {
                    break (u, v);
                }
            };
            uvs.push([u, v]);
            triangles.push(triangle);
        }
        let q = GpuMeshBaryFoldQuery { uvs, triangles };
        check_batch(
            &format!("random batch {seed} (count={count})"),
            &engine,
            &ctx,
            &q,
        );
    }
}

#[test]
fn gpu_matches_cpu_on_triangle_corners_and_centroid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuMeshBaryFold::new(&ctx);
    let triangle = tri();
    // The three corners of the (u, v) domain plus the centroid. (0,0) and the
    // off-diagonal corners land exactly on triangle vertices; (1/3, 1/3) is the
    // centroid. None sits on the reflection line u + v = 1 (the corners (1,0)
    // and (0,1) have u + v = 1, which is not strictly greater than 1, so they
    // take the identity branch deterministically).
    let uvs = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0 / 3.0, 1.0 / 3.0]];
    let triangles = vec![triangle; uvs.len()];
    let q = GpuMeshBaryFoldQuery { uvs, triangles };
    check_batch("corners+centroid", &engine, &ctx, &q);

    // Structural checks on the exact weights the fold must produce.
    let gpu = engine.fold(&ctx, &q);
    assert_vec3("corner (0,0)", Vec3::new(1.0, 0.0, 0.0), gpu.weights[0]);
    assert_vec3("corner (1,0)", Vec3::new(0.0, 1.0, 0.0), gpu.weights[1]);
    assert_vec3("corner (0,1)", Vec3::new(0.0, 0.0, 1.0), gpu.weights[2]);
    assert_vec3(
        "centroid",
        Vec3::new(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
        gpu.weights[3],
    );
    // The corner weights must select the matching triangle vertex position.
    assert_vec3("corner (0,0) pos", triangle[0], gpu.positions[0]);
    assert_vec3("corner (1,0) pos", triangle[1], gpu.positions[1]);
    assert_vec3("corner (0,1) pos", triangle[2], gpu.positions[2]);
}

#[test]
fn gpu_matches_cpu_above_reflection_line() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuMeshBaryFold::new(&ctx);
    let triangle = tri();
    // Draws strictly above u + v = 1, forcing the reflection branch.
    let uvs = vec![[0.8, 0.7], [0.95, 0.9], [0.6, 0.55], [0.99, 0.99]];
    let triangles = vec![triangle; uvs.len()];
    let q = GpuMeshBaryFoldQuery { uvs, triangles };
    check_batch("above reflection", &engine, &ctx, &q);
}

#[test]
fn gpu_matches_cpu_on_single_sample() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuMeshBaryFold::new(&ctx);
    let q = GpuMeshBaryFoldQuery {
        uvs: vec![[0.2, 0.3]],
        triangles: vec![tri()],
    };
    check_batch("single sample", &engine, &ctx, &q);
}

#[test]
fn gpu_handles_empty_batch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuMeshBaryFold::new(&ctx);
    let q = GpuMeshBaryFoldQuery {
        uvs: Vec::new(),
        triangles: Vec::new(),
    };
    let out = engine.fold(&ctx, &q);
    assert!(out.weights.is_empty(), "empty batch must return no weights");
    assert!(
        out.positions.is_empty(),
        "empty batch must return no positions"
    );
}
