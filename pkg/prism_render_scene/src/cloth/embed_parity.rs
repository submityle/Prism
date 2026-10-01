#![cfg(test)]
//! `cloth_embed.wesl` 的 `cloth_skin_embed` 内核的 **逐位** CPU 转写，对齐
//! 架构层黄金 [`embed_render_vertex`]（渲染网格重心嵌入 pass）。
//!
//! GPU 内核把一条渲染网格顶点冻结进一个仿真三角形：读取该顶点的三个宿主
//! 三角形索引、重心权重与带符号法向偏移，取当前仿真位置重建面内点
//! `p0*w0 + p1*w1 + p2*w2`，再沿 **重新求值** 的面法线按偏移推出以恢复
//! 服装厚度。越界索引退化为零输出；退化（零面积）三角形贡献零法线。
//!
//! 本模块逐词把 WESL 算术搬到 CPU（原生 `f32`，不复用黄金的 `Vec3` 方法），
//! 再用 `f32::to_bits` 逐分量比对黄金，证明 WESL 内核与黄金 **逐位** 一致。
//! WESL 现用显式 `1.0 / sqrt(len_sq)`（已从 `inverseSqrt` 这一设备级近似改掉）；
//! `sqrt`/`fdiv` 在 GPU 与 CPU 上均为 IEEE-754 正确舍入，故 `embed_gpu_tests` 的
//! 真机输出与黄金也逐位吻合（原 `rsqrt` vs `1/sqrt` 的设备级分歧已消除）。

use prism_render_architecture::cloth::embed::{
    bind_render_vertex, embed_render_vertex, BarycentricBinding,
};
use prism_render_architecture::cloth::{Vec3, EPS_LEN_SQ};

/// WESL `cloth_embed_normalize_or_zero` 的逐位转写：平方长度超过 `EPS_LEN_SQ`
/// 时乘以 `1/sqrt(len_sq)`，否则返回零向量。乘子只算一次，镜像黄金
/// `Vec3::normalize_or_zero` 的 `scale(1.0 / len_sq.sqrt())`。
fn wesl_normalize_or_zero(v: [f32; 3]) -> [f32; 3] {
    let len_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    if len_sq > EPS_LEN_SQ {
        let inv = 1.0_f32 / len_sq.sqrt();
        [v[0] * inv, v[1] * inv, v[2] * inv]
    } else {
        [0.0, 0.0, 0.0]
    }
}

/// WESL `cloth_embed_face_normal` 的逐位转写：
/// `normalize_or_zero(cross(p1 - p0, p2 - p0))`，叉乘分量顺序镜像黄金
/// `Vec3::cross`。
fn wesl_face_normal(p0: [f32; 3], p1: [f32; 3], p2: [f32; 3]) -> [f32; 3] {
    let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
    let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
    let cross = [
        e1[1] * e2[2] - e1[2] * e2[1],
        e1[2] * e2[0] - e1[0] * e2[2],
        e1[0] * e2[1] - e1[1] * e2[0],
    ];
    wesl_normalize_or_zero(cross)
}

/// WESL `cloth_skin_embed` 单次调用的逐位转写。越界索引返回零（镜像内核的
/// `render_positions[i] = vec4(0,0,0, w)` 的 `xyz` 与黄金的 `Vec3::ZERO`），
/// 否则重建面内重心点再沿重新求值的面法线按偏移推出。
fn wesl_skin_embed(binding: &BarycentricBinding, positions: &[Vec3]) -> [f32; 3] {
    let count = positions.len();
    let t0 = binding.tri[0] as usize;
    let t1 = binding.tri[1] as usize;
    let t2 = binding.tri[2] as usize;
    if t0 >= count || t1 >= count || t2 >= count {
        return [0.0, 0.0, 0.0];
    }
    let p0 = [positions[t0].x, positions[t0].y, positions[t0].z];
    let p1 = [positions[t1].x, positions[t1].y, positions[t1].z];
    let p2 = [positions[t2].x, positions[t2].y, positions[t2].z];
    let (w0, w1, w2) = binding.bary;

    let base = [
        p0[0] * w0 + p1[0] * w1 + p2[0] * w2,
        p0[1] * w0 + p1[1] * w1 + p2[1] * w2,
        p0[2] * w0 + p1[2] * w1 + p2[2] * w2,
    ];
    let normal = wesl_face_normal(p0, p1, p2);
    let off = binding.normal_offset;
    [
        base[0] + normal[0] * off,
        base[1] + normal[1] * off,
        base[2] + normal[2] * off,
    ]
}

/// 逐位断言：WESL 转写与黄金 `embed_render_vertex` 的 `xyz` 比特完全相同。
#[track_caller]
fn assert_embed_bit_exact(binding: &BarycentricBinding, positions: &[Vec3]) {
    let wesl = wesl_skin_embed(binding, positions);
    let golden = embed_render_vertex(binding, positions);
    assert_eq!(
        wesl[0].to_bits(),
        golden.x.to_bits(),
        "x bits diverge: wesl={} golden={}",
        wesl[0],
        golden.x
    );
    assert_eq!(
        wesl[1].to_bits(),
        golden.y.to_bits(),
        "y bits diverge: wesl={} golden={}",
        wesl[1],
        golden.y
    );
    assert_eq!(
        wesl[2].to_bits(),
        golden.z.to_bits(),
        "z bits diverge: wesl={} golden={}",
        wesl[2],
        golden.z
    );
}

/// 构造一个非退化的 XY 平面宿主三角形（面法线沿 +Z）。
fn planar_triangle() -> [Vec3; 3] {
    [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(2.0, 0.0, 0.0),
        Vec3::new(0.0, 3.0, 0.0),
    ]
}

#[test]
fn in_plane_rest_point_matches_golden_bit_for_bit() {
    let positions = planar_triangle();
    let p = Vec3::new(0.6, 0.9, 0.0);
    let binding = bind_render_vertex(p, [0, 1, 2], &positions).expect("in range");
    assert_embed_bit_exact(&binding, &positions);
}

#[test]
fn centroid_binding_matches_golden_bit_for_bit() {
    let positions = planar_triangle();
    let centroid = Vec3::new(
        (positions[0].x + positions[1].x + positions[2].x) / 3.0,
        (positions[0].y + positions[1].y + positions[2].y) / 3.0,
        (positions[0].z + positions[1].z + positions[2].z) / 3.0,
    );
    let binding = bind_render_vertex(centroid, [0, 1, 2], &positions).expect("in range");
    assert_embed_bit_exact(&binding, &positions);
}

#[test]
fn translated_sim_mesh_matches_golden_bit_for_bit() {
    let rest = planar_triangle();
    let p = Vec3::new(0.6, 0.9, 0.0);
    let binding = bind_render_vertex(p, [0, 1, 2], &rest).expect("in range");
    let shift = Vec3::new(10.0, -4.0, 2.5);
    let moved = [
        rest[0].add(shift),
        rest[1].add(shift),
        rest[2].add(shift),
    ];
    assert_embed_bit_exact(&binding, &moved);
}

#[test]
fn normal_offset_thickness_matches_golden_bit_for_bit() {
    let positions = planar_triangle();
    let p = Vec3::new(0.5, 0.5, 1.25);
    let binding = bind_render_vertex(p, [0, 1, 2], &positions).expect("in range");
    assert_embed_bit_exact(&binding, &positions);
}

#[test]
fn negative_offset_pushes_opposite_matches_golden_bit_for_bit() {
    let positions = planar_triangle();
    let p = Vec3::new(0.5, 0.5, -0.75);
    let binding = bind_render_vertex(p, [0, 1, 2], &positions).expect("in range");
    assert_embed_bit_exact(&binding, &positions);
}

#[test]
fn rotated_host_triangle_tracks_normal_bit_for_bit() {
    let rest = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
    ];
    let p = Vec3::new(0.25, 0.25, 2.0);
    let binding = bind_render_vertex(p, [0, 1, 2], &rest).expect("in range");
    let rotated = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
    ];
    assert_embed_bit_exact(&binding, &rotated);
}

#[test]
fn out_of_range_tri0_is_zero_bit_for_bit() {
    let positions = [Vec3::new(1.0, 2.0, 3.0), Vec3::new(4.0, 5.0, 6.0)];
    let binding = BarycentricBinding::new([9, 0, 1], (0.5, 0.25, 0.25), 1.0);
    assert_embed_bit_exact(&binding, &positions);
}

#[test]
fn out_of_range_tri1_is_zero_bit_for_bit() {
    let positions = [Vec3::new(1.0, 2.0, 3.0), Vec3::new(4.0, 5.0, 6.0)];
    let binding = BarycentricBinding::new([0, 9, 1], (0.5, 0.25, 0.25), 1.0);
    assert_embed_bit_exact(&binding, &positions);
}

#[test]
fn out_of_range_tri2_is_zero_bit_for_bit() {
    let positions = [Vec3::new(1.0, 2.0, 3.0), Vec3::new(4.0, 5.0, 6.0)];
    let binding = BarycentricBinding::new([0, 1, 7], (0.5, 0.25, 0.25), 1.0);
    assert_embed_bit_exact(&binding, &positions);
}

#[test]
fn degenerate_triangle_zero_normal_matches_golden_bit_for_bit() {
    // 三点共点 → 零面积 → 零法线 → 偏移项消失，只剩重心点。
    let a = Vec3::new(1.0, 1.0, 1.0);
    let positions = [a, a, a];
    let binding = BarycentricBinding::new([0, 1, 2], (0.2, 0.3, 0.5), 2.0);
    assert_embed_bit_exact(&binding, &positions);
}

#[test]
fn collinear_triangle_zero_normal_matches_golden_bit_for_bit() {
    // 三点共线 → 叉乘为零 → 零法线。
    let positions = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 1.0, 1.0),
        Vec3::new(2.0, 2.0, 2.0),
    ];
    let binding = BarycentricBinding::new([0, 1, 2], (0.3, 0.3, 0.4), 1.5);
    assert_embed_bit_exact(&binding, &positions);
}

#[test]
fn zero_offset_is_pure_barycentric_bit_for_bit() {
    let positions = planar_triangle();
    let binding = BarycentricBinding::new([0, 1, 2], (0.25, 0.5, 0.25), 0.0);
    assert_embed_bit_exact(&binding, &positions);
}

#[test]
fn arbitrary_weights_not_summing_to_one_match_golden_bit_for_bit() {
    // 内核不重归一化权重；任意权重也必须逐位一致。
    let positions = [
        Vec3::new(-1.0, 0.5, 2.0),
        Vec3::new(3.0, -2.0, 0.0),
        Vec3::new(0.0, 4.0, -1.0),
    ];
    let binding = BarycentricBinding::new([0, 1, 2], (0.1, 0.7, 0.9), -0.3);
    assert_embed_bit_exact(&binding, &positions);
}

#[test]
fn large_coordinates_match_golden_bit_for_bit() {
    let positions = [
        Vec3::new(1.0e5, -2.0e5, 3.0e5),
        Vec3::new(-4.0e5, 5.0e5, 6.0e5),
        Vec3::new(7.0e5, 8.0e5, -9.0e5),
    ];
    let binding = BarycentricBinding::new([0, 1, 2], (0.33, 0.33, 0.34), 12.5);
    assert_embed_bit_exact(&binding, &positions);
}

#[test]
fn mixed_grid_across_workgroups_matches_golden_bit_for_bit() {
    // >64 条渲染顶点，跨 64-lane workgroup 边界，混入越界 binding。
    let mut positions = Vec::new();
    for i in 0..40_u32 {
        let f = i as f32;
        positions.push(Vec3::new(f * 0.5, f.sin(), f.cos() * 2.0 - 1.0));
    }
    let count = positions.len() as u32;
    let mut bindings = Vec::new();
    for i in 0..100_u32 {
        let a = i % count;
        let b = (i * 7 + 3) % count;
        let c = (i * 13 + 5) % count;
        // 每 11 个塞一个越界。
        let tri = if i % 11 == 10 {
            [a, count + 2, c]
        } else {
            [a, b, c]
        };
        let w0 = 0.2 + (i % 5) as f32 * 0.1;
        let w1 = 0.3 + (i % 3) as f32 * 0.1;
        let w2 = 1.0 - w0 - w1;
        let off = (i as f32 - 50.0) * 0.05;
        bindings.push(BarycentricBinding::new(tri, (w0, w1, w2), off));
    }
    for binding in &bindings {
        assert_embed_bit_exact(binding, &positions);
    }
}

#[test]
fn jittered_bindings_match_golden_bit_for_bit() {
    // 伪随机抖动的位置/权重/偏移，确保非平凡数值下仍逐位一致。
    let mut state = 0x1234_5678_u32;
    let mut next = || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (state >> 8) as f32 / (1_u32 << 24) as f32
    };
    for _ in 0..64 {
        let positions = [
            Vec3::new(next() * 4.0 - 2.0, next() * 4.0 - 2.0, next() * 4.0 - 2.0),
            Vec3::new(next() * 4.0 - 2.0, next() * 4.0 - 2.0, next() * 4.0 - 2.0),
            Vec3::new(next() * 4.0 - 2.0, next() * 4.0 - 2.0, next() * 4.0 - 2.0),
        ];
        let w0 = next();
        let w1 = next();
        let w2 = 1.0 - w0 - w1;
        let off = next() * 2.0 - 1.0;
        let binding = BarycentricBinding::new([0, 1, 2], (w0, w1, w2), off);
        assert_embed_bit_exact(&binding, &positions);
    }
}

#[test]
fn empty_positions_yields_zero_bit_for_bit() {
    let positions: [Vec3; 0] = [];
    let binding = BarycentricBinding::new([0, 1, 2], (0.3, 0.3, 0.4), 1.0);
    assert_embed_bit_exact(&binding, &positions);
}
