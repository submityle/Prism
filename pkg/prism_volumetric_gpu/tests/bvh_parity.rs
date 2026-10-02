//! Real-device parity for the particle bounding-volume-hierarchy (`BVH`) twin:
//! [`GpuBvh`](prism_volumetric_gpu::bvh::GpuBvh) must reproduce the `CPU` golden
//! [`bvh`](prism_render_architecture::particle::bvh) across the hand-rolled
//! three-component vector algebra
//! ([`Vec3::add`](prism_render_architecture::particle::bvh::Vec3::add),
//! [`Vec3::sub`](prism_render_architecture::particle::bvh::Vec3::sub),
//! [`Vec3::scale`](prism_render_architecture::particle::bvh::Vec3::scale),
//! [`Vec3::min`](prism_render_architecture::particle::bvh::Vec3::min),
//! [`Vec3::max`](prism_render_architecture::particle::bvh::Vec3::max),
//! [`Vec3::length_squared`](prism_render_architecture::particle::bvh::Vec3::length_squared),
//! [`Vec3::length`](prism_render_architecture::particle::bvh::Vec3::length) and
//! [`Vec3::component`](prism_render_architecture::particle::bvh::Vec3::component)),
//! the axis-aligned box operators
//! ([`Aabb::union`](prism_render_architecture::particle::bvh::Aabb::union),
//! [`Aabb::center`](prism_render_architecture::particle::bvh::Aabb::center),
//! [`Aabb::half_extent`](prism_render_architecture::particle::bvh::Aabb::half_extent),
//! [`Aabb::surface_area`](prism_render_architecture::particle::bvh::Aabb::surface_area),
//! [`Aabb::longest_axis`](prism_render_architecture::particle::bvh::Aabb::longest_axis),
//! [`Aabb::contains`](prism_render_architecture::particle::bvh::Aabb::contains),
//! [`Aabb::is_empty`](prism_render_architecture::particle::bvh::Aabb::is_empty)
//! and
//! [`Aabb::expand_point`](prism_render_architecture::particle::bvh::Aabb::expand_point)),
//! the grid quantization and `Morton` interleave
//! ([`quantize_to_grid`](prism_render_architecture::particle::bvh::quantize_to_grid),
//! [`expand_bits`](prism_render_architecture::particle::bvh::expand_bits) and
//! [`morton_code_3d`](prism_render_architecture::particle::bvh::morton_code_3d)),
//! the integer tree-sizing counts
//! ([`BvhTopology::internal_node_count`](prism_render_architecture::particle::bvh::BvhTopology::internal_node_count),
//! [`BvhTopology::total_node_count`](prism_render_architecture::particle::bvh::BvhTopology::total_node_count)
//! and
//! [`BvhTopology::max_stack_depth`](prism_render_architecture::particle::bvh::BvhTopology::max_stack_depth)),
//! and the surface-area-heuristic split cost
//! ([`sah_cost`](prism_render_architecture::particle::bvh::sah_cost)).
//!
//! The fixtures use non-degenerate boxes (every axis extent well away from
//! zero), vectors written as integers or simple decimals, grid points strictly
//! inside their domain, leaf counts that are exact powers of two, and split
//! costs whose parent area is comfortably positive, so every continuous
//! quantity stays far from any degeneracy. Dedicated fixtures still exercise the
//! guarded paths: an empty box (`min > max`) that must report empty, a
//! longest-axis tie that must resolve toward the lower index, a zero-resolution
//! and a collapsed-domain quantization that must fall back to cell `0`, and a
//! non-positive parent-area split cost that must collapse to the bare traversal
//! cost. All fixtures stay pure, need no external math library and use no
//! transcendental math (`sqrt` on the device aside).
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The continuous answers (vector algebra, box center / half-extent / surface
//! area, length and split cost) thread through multiplies, adds and one guarded
//! divide, so they are compared under tolerance (`abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`). The discrete answers (longest-axis
//! index, containment / emptiness verdicts, quantized cells, `Morton` lanes and
//! codes, and the node counts / stack depth) are compared exactly.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::bvh`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::bvh::{
    expand_bits, morton_code_3d, quantize_to_grid, sah_cost, Aabb, BvhTopology, Vec3,
};
use prism_volumetric_gpu::bvh::{BvhQuery, BvhResult, GpuBvh};
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
fn gpu_bvh_vec3_add_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping bvh parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuBvh::new(&ctx);
    let lhs = Vec3::new(1.0, 2.0, 3.0);
    let rhs = Vec3::new(4.0, 5.0, 6.0);
    let q = BvhQuery::Vec3Add {
        lhs: arr(lhs),
        rhs: arr(rhs),
    };
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(got.len(), 1, "one result per query");
    let cpu = lhs.add(rhs);
    let BvhResult::Vec3Add { vector } = got[0] else {
        panic!("expected a Vec3Add result, got {:?}", got[0]);
    };
    assert!(
        approx_vec3(cpu, vector),
        "add mismatch: gpu {vector:?} vs cpu {cpu:?}"
    );
}

#[test]
fn gpu_bvh_vec3_sub_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBvh::new(&ctx);
    let lhs = Vec3::new(5.0, 7.0, 9.0);
    let rhs = Vec3::new(1.0, 2.0, 3.0);
    let q = BvhQuery::Vec3Sub {
        lhs: arr(lhs),
        rhs: arr(rhs),
    };
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    let cpu = lhs.sub(rhs);
    let BvhResult::Vec3Sub { vector } = got[0] else {
        panic!("expected a Vec3Sub result, got {:?}", got[0]);
    };
    assert!(
        approx_vec3(cpu, vector),
        "sub mismatch: gpu {vector:?} vs cpu {cpu:?}"
    );
}

#[test]
fn gpu_bvh_vec3_scale_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBvh::new(&ctx);
    let v = Vec3::new(1.0, -2.0, 3.0);
    let s = 2.5;
    let q = BvhQuery::Vec3Scale {
        vector: arr(v),
        scale: s,
    };
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    let cpu = v.scale(s);
    let BvhResult::Vec3Scale { vector } = got[0] else {
        panic!("expected a Vec3Scale result, got {:?}", got[0]);
    };
    assert!(
        approx_vec3(cpu, vector),
        "scale mismatch: gpu {vector:?} vs cpu {cpu:?}"
    );
}

#[test]
fn gpu_bvh_vec3_min_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBvh::new(&ctx);
    let lhs = Vec3::new(1.0, 5.0, -2.0);
    let rhs = Vec3::new(3.0, 2.0, 4.0);
    let q = BvhQuery::Vec3Min {
        lhs: arr(lhs),
        rhs: arr(rhs),
    };
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    let cpu = lhs.min(rhs);
    let BvhResult::Vec3Min { vector } = got[0] else {
        panic!("expected a Vec3Min result, got {:?}", got[0]);
    };
    assert!(
        approx_vec3(cpu, vector),
        "min mismatch: gpu {vector:?} vs cpu {cpu:?}"
    );
}

#[test]
fn gpu_bvh_vec3_max_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBvh::new(&ctx);
    let lhs = Vec3::new(1.0, 5.0, -2.0);
    let rhs = Vec3::new(3.0, 2.0, 4.0);
    let q = BvhQuery::Vec3Max {
        lhs: arr(lhs),
        rhs: arr(rhs),
    };
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    let cpu = lhs.max(rhs);
    let BvhResult::Vec3Max { vector } = got[0] else {
        panic!("expected a Vec3Max result, got {:?}", got[0]);
    };
    assert!(
        approx_vec3(cpu, vector),
        "max mismatch: gpu {vector:?} vs cpu {cpu:?}"
    );
}

#[test]
fn gpu_bvh_vec3_length_squared_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBvh::new(&ctx);
    let v = Vec3::new(3.0, 4.0, 12.0);
    let q = BvhQuery::Vec3LengthSquared { vector: arr(v) };
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    let cpu = v.length_squared();
    let BvhResult::Vec3LengthSquared { value } = got[0] else {
        panic!("expected a Vec3LengthSquared result, got {:?}", got[0]);
    };
    assert!(
        approx(value, cpu),
        "length_squared mismatch: gpu {value} vs cpu {cpu}"
    );
    assert!(cpu > 1.0, "fixture should exercise a non-trivial magnitude");
}

#[test]
fn gpu_bvh_vec3_length_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBvh::new(&ctx);
    let v = Vec3::new(3.0, 4.0, 12.0);
    let q = BvhQuery::Vec3Length { vector: arr(v) };
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    let cpu = v.length();
    let BvhResult::Vec3Length { length } = got[0] else {
        panic!("expected a Vec3Length result, got {:?}", got[0]);
    };
    assert!(
        approx(length, cpu),
        "length mismatch: gpu {length} vs cpu {cpu}"
    );
    // 3-4-12 is a Pythagorean quadruple, so the length is exactly 13.
    assert!(approx(cpu, 13.0), "fixture length should be 13");
}

#[test]
fn gpu_bvh_vec3_component_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBvh::new(&ctx);
    let v = Vec3::new(7.0, 8.0, 9.0);
    // Exercise all three axis selectors, including the catch-all z branch.
    let batch = [
        BvhQuery::Vec3Component {
            vector: arr(v),
            axis: 0,
        },
        BvhQuery::Vec3Component {
            vector: arr(v),
            axis: 1,
        },
        BvhQuery::Vec3Component {
            vector: arr(v),
            axis: 2,
        },
    ];
    let got = gpu.evaluate(&ctx, &batch);
    assert_eq!(got.len(), 3, "one result per query");
    for (i, axis) in [0u8, 1u8, 2u8].into_iter().enumerate() {
        let cpu = v.component(axis);
        let BvhResult::Vec3Component { value } = got[i] else {
            panic!("expected a Vec3Component result, got {:?}", got[i]);
        };
        assert!(
            approx(value, cpu),
            "component[{axis}] mismatch: gpu {value} vs cpu {cpu}"
        );
    }
}

#[test]
fn gpu_bvh_aabb_union_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBvh::new(&ctx);
    let a = Aabb::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
    let b = Aabb::new(Vec3::new(-1.0, 2.0, 0.5), Vec3::new(2.0, 3.0, 4.0));
    let q = BvhQuery::AabbUnion {
        a_min: arr(a.min),
        a_max: arr(a.max),
        b_min: arr(b.min),
        b_max: arr(b.max),
    };
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    let cpu = a.union(&b);
    let BvhResult::AabbUnion { min, max } = got[0] else {
        panic!("expected an AabbUnion result, got {:?}", got[0]);
    };
    assert!(
        approx_vec3(cpu.min, min) && approx_vec3(cpu.max, max),
        "union mismatch: gpu min {min:?} max {max:?} vs cpu {cpu:?}"
    );
}

#[test]
fn gpu_bvh_aabb_center_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBvh::new(&ctx);
    let bx = Aabb::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 4.0, 6.0));
    let q = BvhQuery::AabbCenter {
        min: arr(bx.min),
        max: arr(bx.max),
    };
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    let cpu = bx.center();
    let BvhResult::AabbCenter { center } = got[0] else {
        panic!("expected an AabbCenter result, got {:?}", got[0]);
    };
    assert!(
        approx_vec3(cpu, center),
        "center mismatch: gpu {center:?} vs cpu {cpu:?}"
    );
}

#[test]
fn gpu_bvh_aabb_half_extent_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBvh::new(&ctx);
    let bx = Aabb::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 4.0, 6.0));
    let q = BvhQuery::AabbHalfExtent {
        min: arr(bx.min),
        max: arr(bx.max),
    };
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    let cpu = bx.half_extent();
    let BvhResult::AabbHalfExtent { half_extent } = got[0] else {
        panic!("expected an AabbHalfExtent result, got {:?}", got[0]);
    };
    assert!(
        approx_vec3(cpu, half_extent),
        "half_extent mismatch: gpu {half_extent:?} vs cpu {cpu:?}"
    );
}

#[test]
fn gpu_bvh_aabb_surface_area_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBvh::new(&ctx);
    // A unit cube has surface area 6; a 1x2x3 box has 2*(2+6+3)=22.
    let fixtures = [
        Aabb::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0)),
        Aabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(0.0, 1.0, 2.0)),
    ];
    for bx in fixtures {
        let q = BvhQuery::AabbSurfaceArea {
            min: arr(bx.min),
            max: arr(bx.max),
        };
        let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
        let cpu = bx.surface_area();
        let BvhResult::AabbSurfaceArea { area } = got[0] else {
            panic!("expected an AabbSurfaceArea result, got {:?}", got[0]);
        };
        assert!(
            approx(area, cpu),
            "surface_area mismatch: gpu {area} vs cpu {cpu}"
        );
    }
}

#[test]
fn gpu_bvh_aabb_longest_axis_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBvh::new(&ctx);
    // Distinct extents pick each axis, and a dx==dy==2, dz==1 box exercises the
    // tie resolving toward the lower index (axis 0).
    let fixtures = [
        Aabb::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(5.0, 1.0, 1.0)),
        Aabb::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 5.0, 1.0)),
        Aabb::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 5.0)),
        Aabb::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 2.0, 1.0)),
    ];
    for bx in fixtures {
        let q = BvhQuery::AabbLongestAxis {
            min: arr(bx.min),
            max: arr(bx.max),
        };
        let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
        let cpu = u32::from(bx.longest_axis());
        let BvhResult::AabbLongestAxis { axis } = got[0] else {
            panic!("expected an AabbLongestAxis result, got {:?}", got[0]);
        };
        assert_eq!(axis, cpu, "longest_axis mismatch for {bx:?}");
    }
}

#[test]
fn gpu_bvh_aabb_contains_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBvh::new(&ctx);
    let bx = Aabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
    let inside = Vec3::new(0.25, -0.5, 0.75);
    let outside = Vec3::new(2.0, 0.0, 0.0);
    let batch = [
        BvhQuery::AabbContains {
            min: arr(bx.min),
            max: arr(bx.max),
            point: arr(inside),
        },
        BvhQuery::AabbContains {
            min: arr(bx.min),
            max: arr(bx.max),
            point: arr(outside),
        },
    ];
    let got = gpu.evaluate(&ctx, &batch);
    assert_eq!(got.len(), 2, "one result per query");
    for (i, p) in [inside, outside].into_iter().enumerate() {
        let cpu = bx.contains(p);
        let BvhResult::AabbContains { contains } = got[i] else {
            panic!("expected an AabbContains result, got {:?}", got[i]);
        };
        assert_eq!(contains, cpu, "contains mismatch for {p:?}");
    }
}

#[test]
fn gpu_bvh_aabb_is_empty_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBvh::new(&ctx);
    let valid = Aabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
    // An inverted box (min > max on every axis) is the canonical empty box.
    let empty = Aabb::new(Vec3::new(1.0, 1.0, 1.0), Vec3::new(0.0, 0.0, 0.0));
    let batch = [
        BvhQuery::AabbIsEmpty {
            min: arr(valid.min),
            max: arr(valid.max),
        },
        BvhQuery::AabbIsEmpty {
            min: arr(empty.min),
            max: arr(empty.max),
        },
    ];
    let got = gpu.evaluate(&ctx, &batch);
    assert_eq!(got.len(), 2, "one result per query");
    for (i, bx) in [valid, empty].into_iter().enumerate() {
        let cpu = bx.is_empty();
        let BvhResult::AabbIsEmpty { is_empty } = got[i] else {
            panic!("expected an AabbIsEmpty result, got {:?}", got[i]);
        };
        assert_eq!(is_empty, cpu, "is_empty mismatch for {bx:?}");
    }
}

#[test]
fn gpu_bvh_aabb_expand_point_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBvh::new(&ctx);
    let base = Aabb::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
    let point = Vec3::new(-2.0, 0.5, 3.0);
    let q = BvhQuery::AabbExpandPoint {
        min: arr(base.min),
        max: arr(base.max),
        point: arr(point),
    };
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    let mut cpu = base;
    cpu.expand_point(point);
    let BvhResult::AabbExpandPoint { min, max } = got[0] else {
        panic!("expected an AabbExpandPoint result, got {:?}", got[0]);
    };
    assert!(
        approx_vec3(cpu.min, min) && approx_vec3(cpu.max, max),
        "expand_point mismatch: gpu min {min:?} max {max:?} vs cpu {cpu:?}"
    );
}

#[test]
fn gpu_bvh_quantize_to_grid_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBvh::new(&ctx);
    let domain_min = Vec3::new(0.0, 0.0, 0.0);
    let domain_max = Vec3::new(10.0, 10.0, 10.0);
    // Interior point, below-domain clamp to 0, above-domain clamp to the last
    // cell.
    let fixtures = [
        Vec3::new(5.0, 2.5, 7.5),
        Vec3::new(-5.0, -5.0, -5.0),
        Vec3::new(99.0, 99.0, 99.0),
    ];
    let resolution = 10u32;
    for p in fixtures {
        let q = BvhQuery::QuantizeToGrid {
            point: arr(p),
            domain_min: arr(domain_min),
            domain_max: arr(domain_max),
            resolution,
        };
        let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
        let (cx, cy, cz) = quantize_to_grid(p, domain_min, domain_max, resolution);
        let BvhResult::QuantizeToGrid { cell } = got[0] else {
            panic!("expected a QuantizeToGrid result, got {:?}", got[0]);
        };
        assert_eq!(cell, [cx, cy, cz], "quantize mismatch for {p:?}");
    }
}

#[test]
fn gpu_bvh_quantize_to_grid_degenerate_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBvh::new(&ctx);
    let p = Vec3::new(5.0, 5.0, 5.0);
    // A collapsed domain (max <= min) and a zero resolution both fall back to
    // cell 0 on every axis.
    let collapsed = BvhQuery::QuantizeToGrid {
        point: arr(p),
        domain_min: arr(Vec3::new(10.0, 10.0, 10.0)),
        domain_max: arr(Vec3::new(0.0, 0.0, 0.0)),
        resolution: 10,
    };
    let zero_res = BvhQuery::QuantizeToGrid {
        point: arr(p),
        domain_min: arr(Vec3::new(0.0, 0.0, 0.0)),
        domain_max: arr(Vec3::new(10.0, 10.0, 10.0)),
        resolution: 0,
    };
    let batch = [collapsed, zero_res];
    let got = gpu.evaluate(&ctx, &batch);
    assert_eq!(got.len(), 2, "one result per query");

    let (cx0, cy0, cz0) =
        quantize_to_grid(p, Vec3::new(10.0, 10.0, 10.0), Vec3::new(0.0, 0.0, 0.0), 10);
    let BvhResult::QuantizeToGrid { cell } = got[0] else {
        panic!("expected a QuantizeToGrid result, got {:?}", got[0]);
    };
    assert_eq!(cell, [cx0, cy0, cz0], "collapsed-domain quantize mismatch");
    assert_eq!(cell, [0, 0, 0], "collapsed domain must map to cell 0");

    let (cx1, cy1, cz1) =
        quantize_to_grid(p, Vec3::new(0.0, 0.0, 0.0), Vec3::new(10.0, 10.0, 10.0), 0);
    let BvhResult::QuantizeToGrid { cell } = got[1] else {
        panic!("expected a QuantizeToGrid result, got {:?}", got[1]);
    };
    assert_eq!(cell, [cx1, cy1, cz1], "zero-resolution quantize mismatch");
}

#[test]
fn gpu_bvh_expand_bits_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBvh::new(&ctx);
    // Low 10 bits set, a sparse pattern, and the all-ones lane.
    let fixtures = [0u32, 1, 0b10_1010_0101, 0x3FF];
    for value in fixtures {
        let q = BvhQuery::ExpandBits { value };
        let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
        let cpu = expand_bits(value);
        let BvhResult::ExpandBits { bits } = got[0] else {
            panic!("expected an ExpandBits result, got {:?}", got[0]);
        };
        // Golden holds the spread lane in a u64 that fits the low 30 bits.
        assert_eq!(u64::from(bits), cpu, "expand_bits mismatch for {value:#x}");
    }
}

#[test]
fn gpu_bvh_morton_code_3d_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBvh::new(&ctx);
    let fixtures = [
        (0u32, 0u32, 0u32),
        (1, 2, 3),
        (1023, 1023, 1023),
        (512, 256, 128),
    ];
    for (x, y, z) in fixtures {
        let q = BvhQuery::MortonCode3d { x, y, z };
        let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
        let cpu = morton_code_3d(x, y, z);
        let BvhResult::MortonCode3d { code } = got[0] else {
            panic!("expected a MortonCode3d result, got {:?}", got[0]);
        };
        // The 30-bit interleave fits a u32; compare against the golden u64.
        assert_eq!(u64::from(code), cpu, "morton mismatch for ({x}, {y}, {z})");
    }
}

#[test]
fn gpu_bvh_internal_node_count_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBvh::new(&ctx);
    // Empty tree saturates to 0; a power-of-two leaf count gives n - 1.
    let fixtures = [0u32, 1, 8, 1024];
    for leaf_count in fixtures {
        let q = BvhQuery::InternalNodeCount { leaf_count };
        let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
        let cpu = BvhTopology::new(leaf_count).internal_node_count();
        let BvhResult::InternalNodeCount { count } = got[0] else {
            panic!("expected an InternalNodeCount result, got {:?}", got[0]);
        };
        assert_eq!(count, cpu, "internal_node_count mismatch for {leaf_count}");
    }
}

#[test]
fn gpu_bvh_total_node_count_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBvh::new(&ctx);
    // Empty tree has no nodes; otherwise 2n - 1.
    let fixtures = [0u32, 1, 8, 1024];
    for leaf_count in fixtures {
        let q = BvhQuery::TotalNodeCount { leaf_count };
        let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
        let cpu = BvhTopology::new(leaf_count).total_node_count();
        let BvhResult::TotalNodeCount { count } = got[0] else {
            panic!("expected a TotalNodeCount result, got {:?}", got[0]);
        };
        assert_eq!(count, cpu, "total_node_count mismatch for {leaf_count}");
    }
}

#[test]
fn gpu_bvh_max_stack_depth_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBvh::new(&ctx);
    // ceil(log2(n)) + margin(2): 1 -> 2, 8 -> 5, 1024 -> 12, and the n <= 1
    // short-circuit at 0 -> 2.
    let fixtures = [0u32, 1, 8, 1024];
    for leaf_count in fixtures {
        let q = BvhQuery::MaxStackDepth { leaf_count };
        let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
        let cpu = BvhTopology::new(leaf_count).max_stack_depth();
        let BvhResult::MaxStackDepth { depth } = got[0] else {
            panic!("expected a MaxStackDepth result, got {:?}", got[0]);
        };
        assert_eq!(depth, cpu, "max_stack_depth mismatch for {leaf_count}");
    }
}

#[test]
fn gpu_bvh_sah_cost_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBvh::new(&ctx);
    // A balanced split with positive parent area, and a degenerate parent
    // (total_area <= 0) that must collapse to the bare traversal cost 1.0.
    let fixtures = [(6.0f32, 6.0f32, 12.0f32, 4u32, 4u32), (0.0, 0.0, 0.0, 4, 4)];
    for (left_area, right_area, total_area, left_n, right_n) in fixtures {
        let q = BvhQuery::SahCost {
            left_area,
            right_area,
            total_area,
            left_n,
            right_n,
        };
        let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
        let cpu = sah_cost(left_area, right_area, total_area, left_n, right_n);
        let BvhResult::SahCost { cost } = got[0] else {
            panic!("expected a SahCost result, got {:?}", got[0]);
        };
        assert!(
            approx(cost, cpu),
            "sah_cost mismatch: gpu {cost} vs cpu {cpu}"
        );
    }
}

#[test]
fn gpu_bvh_batch_matches_elementwise() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBvh::new(&ctx);
    // A mixed batch exercises the one-thread-per-query flattening; each result
    // must be independent of its neighbours.
    let v = Vec3::new(3.0, 4.0, 12.0);
    let a = Aabb::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 4.0, 6.0));
    let batch = [
        BvhQuery::Vec3Length { vector: arr(v) },
        BvhQuery::AabbCenter {
            min: arr(a.min),
            max: arr(a.max),
        },
        BvhQuery::AabbSurfaceArea {
            min: arr(a.min),
            max: arr(a.max),
        },
        BvhQuery::MortonCode3d { x: 1, y: 2, z: 3 },
        BvhQuery::TotalNodeCount { leaf_count: 8 },
        BvhQuery::SahCost {
            left_area: 6.0,
            right_area: 6.0,
            total_area: 12.0,
            left_n: 4,
            right_n: 4,
        },
    ];
    let got = gpu.evaluate(&ctx, &batch);
    assert_eq!(got.len(), batch.len(), "one result per query");

    let cpu_len = v.length();
    let BvhResult::Vec3Length { length } = got[0] else {
        panic!("expected a Vec3Length result, got {:?}", got[0]);
    };
    assert!(
        approx(length, cpu_len),
        "batch length mismatch: gpu {length} vs cpu {cpu_len}"
    );

    let cpu_center = a.center();
    let BvhResult::AabbCenter { center } = got[1] else {
        panic!("expected an AabbCenter result, got {:?}", got[1]);
    };
    assert!(
        approx_vec3(cpu_center, center),
        "batch center mismatch: gpu {center:?} vs cpu {cpu_center:?}"
    );

    let cpu_area = a.surface_area();
    let BvhResult::AabbSurfaceArea { area } = got[2] else {
        panic!("expected an AabbSurfaceArea result, got {:?}", got[2]);
    };
    assert!(
        approx(area, cpu_area),
        "batch surface_area mismatch: gpu {area} vs cpu {cpu_area}"
    );

    let cpu_morton = morton_code_3d(1, 2, 3);
    let BvhResult::MortonCode3d { code } = got[3] else {
        panic!("expected a MortonCode3d result, got {:?}", got[3]);
    };
    assert_eq!(u64::from(code), cpu_morton, "batch morton mismatch");

    let cpu_total = BvhTopology::new(8).total_node_count();
    let BvhResult::TotalNodeCount { count } = got[4] else {
        panic!("expected a TotalNodeCount result, got {:?}", got[4]);
    };
    assert_eq!(count, cpu_total, "batch total_node_count mismatch");

    let cpu_cost = sah_cost(6.0, 6.0, 12.0, 4, 4);
    let BvhResult::SahCost { cost } = got[5] else {
        panic!("expected a SahCost result, got {:?}", got[5]);
    };
    assert!(
        approx(cost, cpu_cost),
        "batch sah_cost mismatch: gpu {cost} vs cpu {cpu_cost}"
    );
}

#[test]
fn gpu_bvh_empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBvh::new(&ctx);
    // No dispatch is issued and the result vector is empty.
    assert!(gpu.evaluate(&ctx, &[]).is_empty());
}
