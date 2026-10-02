//! Real-device parity for the mesh-emission twin:
//! [`GpuMeshEmission`](prism_volumetric_gpu::mesh_emission::GpuMeshEmission)
//! must reproduce the `CPU` golden
//! [`mesh_emission`](prism_render_architecture::particle::mesh_emission) across
//! linear-blend skinning of a position
//! ([`skin_position`](prism_render_architecture::particle::mesh_emission::skin_position)),
//! linear-blend skinning of a renormalized normal
//! ([`skin_normal`](prism_render_architecture::particle::mesh_emission::skin_normal)),
//! the triangle surface area
//! ([`MeshTriangle::area`](prism_render_architecture::particle::mesh_emission::MeshTriangle::area)),
//! and the barycentric sample of a surface point and shading normal
//! ([`MeshTriangle::position_at`](prism_render_architecture::particle::mesh_emission::MeshTriangle::position_at)
//! and
//! [`MeshTriangle::normal_at`](prism_render_architecture::particle::mesh_emission::MeshTriangle::normal_at)).
//!
//! The fixtures use orthonormal-basis-plus-translation bones, blend weights that
//! sum to `1`, and triangles whose area is comfortably positive with
//! non-parallel shading normals, so every quantity sits far from the degeneracy
//! and fallback thresholds. One dedicated fixture zeroes every shading normal to
//! exercise the geometric-normal fallback the reference `normal_at` performs.
//! All positions, normals and weights are written as integers or simple
//! decimals, so the fixtures stay pure and need no external math library and no
//! transcendental math; the orthonormal bases are built from `3`-`4`-`5`
//! rational rotations rather than any `sin` / `cos` call.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The skinned positions, skinned normals, areas and samples thread through
//! multiplies, adds, a cross product and one guarded reciprocal `sqrt`, so they
//! are compared under tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`,
//! `REL_FLOOR = 1e-6`).
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::mesh_emission`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::mesh_emission::{
    skin_normal, skin_position, BoneTransform, MeshTriangle, MeshVertex, SkinnedVertex,
    MAX_BONE_INFLUENCES,
};
use prism_render_architecture::particle::Vec3;
use prism_volumetric_gpu::mesh_emission::{GpuMeshEmission, MeshEmissionQuery, MeshEmissionResult};
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

/// Four orthonormal-basis-plus-translation bones, each basis built from a
/// `3`-`4`-`5` rational rotation so the fixture needs no transcendental math.
fn palette() -> [BoneTransform; 4] {
    [
        // Rotation about `+Z` plus a translation.
        BoneTransform::new(
            Vec3::new(0.6, 0.8, 0.0),
            Vec3::new(-0.8, 0.6, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, -2.0, 0.5),
        ),
        // Rotation about `+X` plus a translation.
        BoneTransform::new(
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.6, 0.8),
            Vec3::new(0.0, -0.8, 0.6),
            Vec3::new(-1.0, 0.0, 2.0),
        ),
        // Rotation about `+Y` plus a translation.
        BoneTransform::new(
            Vec3::new(0.6, 0.0, -0.8),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.8, 0.0, 0.6),
            Vec3::new(2.0, 0.5, 0.0),
        ),
        // A pure translation (identity basis).
        BoneTransform::from_translation(Vec3::new(0.0, 1.0, -1.0)),
    ]
}

/// A four-bone skinned vertex whose `weights` sum to `1` and whose `bones`
/// index the full `palette`.
fn skinned_vertex() -> SkinnedVertex {
    SkinnedVertex {
        position: Vec3::new(3.0, -2.0, 5.0),
        normal: Vec3::new(0.0, 0.6, 0.8),
        uv: [0.0, 0.0],
        weights: [0.4, 0.3, 0.2, 0.1],
        bones: [0, 1, 2, 3],
    }
}

/// Resolves `vertex` against the bone palette the way the host does before a
/// dispatch: each in-range influence takes its `BoneTransform` from `bones`, and
/// an out-of-range influence has its weight zeroed, matching the reference
/// `continue`.
fn resolve_skin(
    vertex: &SkinnedVertex,
    bones: &[BoneTransform],
) -> (
    [f32; MAX_BONE_INFLUENCES],
    [BoneTransform; MAX_BONE_INFLUENCES],
) {
    let mut weights = vertex.weights;
    let mut resolved = [BoneTransform::IDENTITY; MAX_BONE_INFLUENCES];
    for (i, &bone_idx) in vertex.bones.iter().enumerate() {
        match bones.get(usize::from(bone_idx)) {
            Some(bone) => resolved[i] = *bone,
            None => weights[i] = 0.0,
        }
    }
    (weights, resolved)
}

/// A slanted triangle with distinct, non-parallel shading normals; its area is
/// comfortably positive.
fn slanted() -> MeshTriangle {
    MeshTriangle {
        a: MeshVertex::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            [0.0, 0.0],
        ),
        b: MeshVertex::new(
            Vec3::new(4.0, 0.0, 1.0),
            Vec3::new(1.0, 0.0, 2.0),
            [1.0, 0.0],
        ),
        c: MeshVertex::new(
            Vec3::new(1.0, 3.0, 2.0),
            Vec3::new(0.0, 1.0, 3.0),
            [0.0, 1.0],
        ),
    }
}

/// A right triangle in the `z = 0` plane with every shading normal zeroed, so
/// the blended normal is numerically zero and `normal_at` must fall back to the
/// geometric normal.
fn unauthored() -> MeshTriangle {
    MeshTriangle {
        a: MeshVertex::new(Vec3::new(0.0, 0.0, 0.0), Vec3::ZERO, [0.0, 0.0]),
        b: MeshVertex::new(Vec3::new(2.0, 0.0, 0.0), Vec3::ZERO, [1.0, 0.0]),
        c: MeshVertex::new(Vec3::new(0.0, 2.0, 0.0), Vec3::ZERO, [0.0, 1.0]),
    }
}

/// The three corner positions of `tri` in `a`, `b`, `c` order.
fn tri_positions(tri: &MeshTriangle) -> [[f32; 3]; 3] {
    [
        arr(tri.a.position),
        arr(tri.b.position),
        arr(tri.c.position),
    ]
}

/// The three corner shading normals of `tri` in `a`, `b`, `c` order.
fn tri_normals(tri: &MeshTriangle) -> [[f32; 3]; 3] {
    [arr(tri.a.normal), arr(tri.b.normal), arr(tri.c.normal)]
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_mesh_emission_skin_position_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping mesh_emission parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuMeshEmission::new(&ctx);
    let palette = palette();
    let vertex = skinned_vertex();
    let (weights, bones) = resolve_skin(&vertex, &palette);
    let q = MeshEmissionQuery::SkinPosition {
        position: arr(vertex.position),
        weights,
        bones,
    };
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(got.len(), 1, "one result per query");
    let cpu = skin_position(&vertex, &palette);
    let MeshEmissionResult::SkinPosition { position } = got[0] else {
        panic!("expected a SkinPosition result, got {:?}", got[0]);
    };
    assert!(
        approx_vec3(cpu, position),
        "skin_position mismatch: gpu {position:?} vs cpu {cpu:?}"
    );
    // A multi-bone blend must actually move the bind-pose position, proving a
    // real skinning transform ran rather than a pass-through.
    assert!(
        cpu.distance(vertex.position) > 1.0e-3,
        "fixture should exercise a non-trivial skinning transform"
    );
}

#[test]
fn gpu_mesh_emission_skin_normal_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMeshEmission::new(&ctx);
    let palette = palette();
    let vertex = skinned_vertex();
    let (weights, bones) = resolve_skin(&vertex, &palette);
    let q = MeshEmissionQuery::SkinNormal {
        normal: arr(vertex.normal),
        weights,
        bones,
    };
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(got.len(), 1, "one result per query");
    let cpu = skin_normal(&vertex, &palette);
    let MeshEmissionResult::SkinNormal { normal } = got[0] else {
        panic!("expected a SkinNormal result, got {:?}", got[0]);
    };
    assert!(
        approx_vec3(cpu, normal),
        "skin_normal mismatch: gpu {normal:?} vs cpu {cpu:?}"
    );
    // The renormalized blend must be a unit vector (the guarded division ran).
    let len = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
    assert!(
        approx(len, 1.0),
        "skinned normal should be renormalized to unit length, got {len}"
    );
}

#[test]
fn gpu_mesh_emission_triangle_area_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMeshEmission::new(&ctx);
    let tri = slanted();
    let q = MeshEmissionQuery::TriangleArea {
        positions: tri_positions(&tri),
    };
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(got.len(), 1, "one result per query");
    let cpu = tri.area();
    let MeshEmissionResult::TriangleArea { area } = got[0] else {
        panic!("expected a TriangleArea result, got {:?}", got[0]);
    };
    assert!(
        approx(area, cpu),
        "triangle area mismatch: gpu {area} vs cpu {cpu}"
    );
    // The fixture is non-degenerate, so the area sits far from the zero floor.
    assert!(cpu > 0.5, "fixture triangle should be non-degenerate");
}

#[test]
fn gpu_mesh_emission_barycentric_sample_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMeshEmission::new(&ctx);
    let tri = slanted();
    // An interior barycentric sample kept off any tie.
    let bary = [0.2, 0.3, 0.5];
    let q = MeshEmissionQuery::BarycentricSample {
        positions: tri_positions(&tri),
        normals: tri_normals(&tri),
        barycentric: bary,
    };
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(got.len(), 1, "one result per query");
    let bary_v = Vec3::new(bary[0], bary[1], bary[2]);
    let cpu_pos = tri.position_at(bary_v);
    let cpu_nrm = tri.normal_at(bary_v);
    let MeshEmissionResult::BarycentricSample { position, normal } = got[0] else {
        panic!("expected a BarycentricSample result, got {:?}", got[0]);
    };
    assert!(
        approx_vec3(cpu_pos, position),
        "sample position mismatch: gpu {position:?} vs cpu {cpu_pos:?}"
    );
    assert!(
        approx_vec3(cpu_nrm, normal),
        "sample normal mismatch: gpu {normal:?} vs cpu {cpu_nrm:?}"
    );
    // The shading normal must differ from the geometric normal on this fixture,
    // proving a real corner blend ran rather than the fallback.
    let geo = tri.geometric_normal();
    assert!(
        geo.distance(cpu_nrm) > 1.0e-3,
        "shading normal should differ from the geometric normal on this fixture"
    );
}

#[test]
fn gpu_mesh_emission_barycentric_sample_falls_back_to_geometric_normal() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMeshEmission::new(&ctx);
    let tri = unauthored();
    let bary = [0.25, 0.25, 0.5];
    let q = MeshEmissionQuery::BarycentricSample {
        positions: tri_positions(&tri),
        normals: tri_normals(&tri),
        barycentric: bary,
    };
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(got.len(), 1, "one result per query");
    let bary_v = Vec3::new(bary[0], bary[1], bary[2]);
    let cpu_nrm = tri.normal_at(bary_v);
    let MeshEmissionResult::BarycentricSample { normal, .. } = got[0] else {
        panic!("expected a BarycentricSample result, got {:?}", got[0]);
    };
    assert!(
        approx_vec3(cpu_nrm, normal),
        "fallback normal mismatch: gpu {normal:?} vs cpu {cpu_nrm:?}"
    );
    // The geometric normal of this `z = 0` triangle points along `+Z`.
    assert!(
        approx_vec3(Vec3::new(0.0, 0.0, 1.0), normal),
        "unauthored normals must fall back to the geometric normal, got {normal:?}"
    );
}

#[test]
fn gpu_mesh_emission_batch_matches_elementwise() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMeshEmission::new(&ctx);
    // A mixed batch exercises the one-thread-per-query flattening; each result
    // must be independent of its neighbours.
    let palette = palette();
    let vertex = skinned_vertex();
    let (weights, bones) = resolve_skin(&vertex, &palette);
    let tri = slanted();
    let bary = [0.25, 0.35, 0.4];
    let batch = [
        MeshEmissionQuery::SkinPosition {
            position: arr(vertex.position),
            weights,
            bones,
        },
        MeshEmissionQuery::SkinNormal {
            normal: arr(vertex.normal),
            weights,
            bones,
        },
        MeshEmissionQuery::TriangleArea {
            positions: tri_positions(&tri),
        },
        MeshEmissionQuery::BarycentricSample {
            positions: tri_positions(&tri),
            normals: tri_normals(&tri),
            barycentric: bary,
        },
    ];
    let got = gpu.evaluate(&ctx, &batch);
    assert_eq!(got.len(), batch.len(), "one result per query");

    let cpu_pos = skin_position(&vertex, &palette);
    let MeshEmissionResult::SkinPosition { position } = got[0] else {
        panic!("expected a SkinPosition result, got {:?}", got[0]);
    };
    assert!(
        approx_vec3(cpu_pos, position),
        "batch skin_position mismatch: gpu {position:?} vs cpu {cpu_pos:?}"
    );

    let cpu_nrm = skin_normal(&vertex, &palette);
    let MeshEmissionResult::SkinNormal { normal } = got[1] else {
        panic!("expected a SkinNormal result, got {:?}", got[1]);
    };
    assert!(
        approx_vec3(cpu_nrm, normal),
        "batch skin_normal mismatch: gpu {normal:?} vs cpu {cpu_nrm:?}"
    );

    let cpu_area = tri.area();
    let MeshEmissionResult::TriangleArea { area } = got[2] else {
        panic!("expected a TriangleArea result, got {:?}", got[2]);
    };
    assert!(
        approx(area, cpu_area),
        "batch area mismatch: gpu {area} vs cpu {cpu_area}"
    );

    let bary_v = Vec3::new(bary[0], bary[1], bary[2]);
    let cpu_sample_pos = tri.position_at(bary_v);
    let cpu_sample_nrm = tri.normal_at(bary_v);
    let MeshEmissionResult::BarycentricSample { position, normal } = got[3] else {
        panic!("expected a BarycentricSample result, got {:?}", got[3]);
    };
    assert!(
        approx_vec3(cpu_sample_pos, position),
        "batch sample position mismatch: gpu {position:?} vs cpu {cpu_sample_pos:?}"
    );
    assert!(
        approx_vec3(cpu_sample_nrm, normal),
        "batch sample normal mismatch: gpu {normal:?} vs cpu {cpu_sample_nrm:?}"
    );
}

#[test]
fn gpu_mesh_emission_empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMeshEmission::new(&ctx);
    // No dispatch is issued and the result vector is empty.
    assert!(gpu.evaluate(&ctx, &[]).is_empty());
}
