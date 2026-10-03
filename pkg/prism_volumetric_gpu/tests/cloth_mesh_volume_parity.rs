//! Real-device parity for the cloth mesh-volume twin:
//! [`GpuClothMeshVolume`](prism_volumetric_gpu::cloth_mesh_volume::GpuClothMeshVolume)
//! must reproduce the `CPU` golden
//! [`mesh_volume`](prism_render_architecture::cloth::pressure::mesh_volume)
//! across a single triangle, a closed tetrahedron, a closed unit cube, meshes
//! with out-of-range triangle indices, the empty-mesh short-circuit, and a
//! randomized sweep.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The expected volume is produced by calling the golden `mesh_volume` directly
//! with the hand-rolled render [`Vec3`](prism_render_architecture::cloth::Vec3)
//! and the same `[u32; 3]` triangle list, so the test pins `GPU == golden`, not
//! merely that the shader compiles.
//!
//! # Parity criterion
//!
//! The signed volume is a continuous `f32` quantity threaded through a long
//! additive reduction, so it is compared with the widened tolerance
//! `abs <= 2e-4` or `rel <= 2e-3` (relative floor `1e-6`) to absorb the
//! reduction's accumulated rounding. The index-range skip is an integer-exact
//! branch that both sides evaluate identically, so no reject sampling is
//! required.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::cloth::pressure`；无第三方引擎源码或衍生代码。

use prism_render_architecture::cloth::pressure::mesh_volume;
use prism_render_architecture::cloth::Vec3;
use prism_volumetric_gpu::cloth_mesh_volume::{
    ClothMeshVolumeQuery, ClothMeshVolumeResult, GpuClothMeshVolume,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous volume parity comparison, widened to
/// absorb the additive reduction's accumulated rounding.
const EPS: f32 = 2e-4;
/// Relative tolerance for the continuous volume parity comparison.
const REL: f32 = 2e-3;
/// Floor on the relative-tolerance denominator so tiny magnitudes stay stable.
const REL_FLOOR: f32 = 1e-6;

/// Returns `true` when `a` and `b` agree within the widened continuous
/// tolerance (`abs <= 2e-4` or `rel <= 2e-3`, floor `1e-6`).
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
}

/// Computes the in-host golden volume for one query by calling the reference
/// `mesh_volume` directly with render-space vectors.
fn oracle(q: &ClothMeshVolumeQuery) -> f32 {
    let positions: Vec<Vec3> = q
        .positions
        .iter()
        .map(|p| Vec3::new(p[0], p[1], p[2]))
        .collect();
    mesh_volume(&positions, &q.triangles)
}

/// Pins one `GPU` result against the in-host oracle.
fn check_query(idx: usize, q: &ClothMeshVolumeQuery, got: &ClothMeshVolumeResult) {
    let want = oracle(q);
    assert!(
        close(got.volume, want),
        "mesh {idx}: gpu {} golden {want}",
        got.volume,
    );
}

/// The eight corners of the unit cube `[0, 1]^3`.
fn unit_cube_vertices() -> Vec<[f32; 3]> {
    vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ]
}

/// The twelve outward-wound triangles of the unit cube.
fn unit_cube_triangles() -> Vec<[u32; 3]> {
    vec![
        // bottom (z = 0), facing -Z
        [0, 2, 1],
        [0, 3, 2],
        // top (z = 1), facing +Z
        [4, 5, 6],
        [4, 6, 7],
        // front (y = 0), facing -Y
        [0, 1, 5],
        [0, 5, 4],
        // back (y = 1), facing +Y
        [3, 7, 6],
        [3, 6, 2],
        // left (x = 0), facing -X
        [0, 4, 7],
        [0, 7, 3],
        // right (x = 1), facing +X
        [1, 2, 6],
        [1, 6, 5],
    ]
}

/// A tiny host-side `LCG` producing a stream of `u32` words; used only to drive
/// the randomized fixture sweep (no `GPU` state depends on it).
struct Lcg {
    state: u64,
}

impl Lcg {
    /// Seeds the generator.
    fn new(seed: u64) -> Lcg {
        Lcg { state: seed }
    }

    /// Advances the state and returns the next `u32` word.
    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.state >> 33) as u32
    }

    /// Returns the next `f32` uniformly in `[lo, hi)`.
    fn next_f32(&mut self, lo: f32, hi: f32) -> f32 {
        let unit = (self.next_u32() >> 8) as f32 * (1.0 / 16_777_216.0);
        lo + (hi - lo) * unit
    }

    /// Returns the next `u32` uniformly in `[lo, hi)` (half-open).
    fn next_range(&mut self, lo: u32, hi: u32) -> u32 {
        lo + self.next_u32() % (hi - lo)
    }
}

#[test]
fn empty_mesh_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let twin = GpuClothMeshVolume::new(&ctx);
    let out = twin.evaluate(&ctx, &ClothMeshVolumeQuery::default());
    assert!(
        close(out.volume, 0.0),
        "empty mesh must enclose zero volume, got {}",
        out.volume,
    );
}

#[test]
fn single_triangle_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let twin = GpuClothMeshVolume::new(&ctx);

    // A single tetra spanned from the origin; the signed contribution is
    // (1/6) * p0 . (p1 x p2) over this one triangle.
    let query = ClothMeshVolumeQuery {
        positions: vec![[0.3, -0.4, 0.2], [1.7, 0.1, -0.5], [-0.6, 1.3, 0.9]],
        triangles: vec![[0, 1, 2]],
    };
    let got = twin.evaluate(&ctx, &query);
    check_query(0, &query, &got);
}

#[test]
fn closed_tetrahedron_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let twin = GpuClothMeshVolume::new(&ctx);

    // Unit corner tetrahedron with the four outward-wound faces; the enclosed
    // volume is 1/6, but parity only compares the twin against the golden.
    let query = ClothMeshVolumeQuery {
        positions: vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        triangles: vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
    };
    let got = twin.evaluate(&ctx, &query);
    check_query(0, &query, &got);
}

#[test]
fn closed_unit_cube_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let twin = GpuClothMeshVolume::new(&ctx);

    let query = ClothMeshVolumeQuery {
        positions: unit_cube_vertices(),
        triangles: unit_cube_triangles(),
    };
    let got = twin.evaluate(&ctx, &query);
    check_query(0, &query, &got);
}

#[test]
fn out_of_range_triangles_are_skipped() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let twin = GpuClothMeshVolume::new(&ctx);

    // A valid tetrahedron plus several triangles whose indices fall at or past
    // the vertex count; the twin and golden must both skip them.
    let mut triangles = vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]];
    triangles.push([4, 0, 1]); // first index out of range
    triangles.push([0, 9, 2]); // middle index out of range
    triangles.push([0, 1, 100]); // last index out of range
    triangles.push([50, 60, 70]); // all out of range

    let query = ClothMeshVolumeQuery {
        positions: vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        triangles,
    };
    let got = twin.evaluate(&ctx, &query);
    check_query(0, &query, &got);
}

#[test]
fn randomized_sweep_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let twin = GpuClothMeshVolume::new(&ctx);

    let mut rng = Lcg::new(0x1234_5678_9ABC_DEF0);
    let mut meshes: Vec<ClothMeshVolumeQuery> = Vec::with_capacity(256);
    while meshes.len() < 256 {
        // Small meshes keep the additive reduction's accumulated error within
        // the widened tolerance.
        let vert_count = rng.next_range(3, 13);
        let mut positions: Vec<[f32; 3]> = Vec::with_capacity(vert_count as usize);
        for _ in 0..vert_count {
            positions.push([
                rng.next_f32(-2.0, 2.0),
                rng.next_f32(-2.0, 2.0),
                rng.next_f32(-2.0, 2.0),
            ]);
        }

        let tri_count = rng.next_range(1, 21);
        let mut triangles: Vec<[u32; 3]> = Vec::with_capacity(tri_count as usize);
        for _ in 0..tri_count {
            // Roughly one in six triangles deliberately references an
            // out-of-range vertex so the index-skip branch is exercised; the
            // skip is integer-exact on both sides.
            let mut idx = [0u32; 3];
            for slot in &mut idx {
                *slot = if rng.next_range(0, 6) == 0 {
                    vert_count + rng.next_range(0, 8)
                } else {
                    rng.next_range(0, vert_count)
                };
            }
            triangles.push(idx);
        }

        meshes.push(ClothMeshVolumeQuery {
            positions,
            triangles,
        });
    }

    for (idx, query) in meshes.iter().enumerate() {
        let got = twin.evaluate(&ctx, query);
        check_query(idx, query, &got);
    }
}
