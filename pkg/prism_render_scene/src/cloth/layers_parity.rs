//! `cloth_layers.wesl` 内核的**逐位** CPU parity，直接对齐架构层黄金
//! [`resolve_layer_coupling_jacobi`]（`pub`，可跨 crate 调用）。
//!
//! `shader_tests` 只做 naga 编译门禁，证明不了 `cloth_layers.wesl` 的层间耦合
//! Jacobi pass 逐位复刻了 CPU 黄金的算术（设计 §9：不造假 parity）。本模块把该
//! 内核的逐条算术独立转写成 CPU 版本（原生 `[f32; 3]` 数组，不复用黄金 `Vec3`
//! 方法），对多种层叠接触配置断言其**修正后位置**与黄金
//! [`resolve_layer_coupling_jacobi`] **逐位（`to_bits`）一致**。
//!
//! ## 为什么层间耦合可以做到真正逐位
//! 层间 Jacobi 耦合全程只有「倒数乘」与「两侧同为除法」的除法：
//! * 法线归一化镜像黄金 `glam::Vec3::normalize_or_zero`：倒数长度有限且为正即
//!   归一化，微小非零法线仍为可用方向，只有零/非有限长度回退径向，三者同为
//!   **先取标量倒数再乘**，逐位一致（对齐既有 `inverseSqrt` 转写惯例）。
//! * 径向回退方向 `dir = delta * (1.0 / dist)` 同为倒数乘。
//! * 质量权重分配 `w_inner / w_sum` / `w_outer / w_sum` 在 WESL 与黄金**两侧
//!   同为除法**（非倒数乘），逐位一致。
//! * 其余 `dot`、`thickness - signed`、分量 × 标量后相加全是乘加。
//!
//! ## 多邻居求和顺序也逐位
//! 黄金 `accumulate_layer_jacobi_corrections` 把每个粒子按 `cell_of` 投进
//! 升序 [`BTreeMap`] 空间哈希，按 `dx,dy,dz ∈ -1..=1` 的 27 邻格、每格桶内升序
//! 索引遍历并累加「本粒子那一半」的 own-slot 修正。该顺序**完全确定且可复刻**：
//! 本 harness 用同一 `BTreeMap` 键序（元组字典序）+ 相同的三重 `dx/dy/dz` 嵌套
//! + 相同桶内升序，重建逐位相同的邻居求和序，故多接触场景也逐位一致（区别于
//! `virtual_particles` / 点对点自碰撞那种 `atomicExchange` 桶序不定的内核）。

use std::collections::BTreeMap;

use prism_render_architecture::cloth::layers::LayerParams;
use prism_render_architecture::cloth::layers_jacobi::resolve_layer_coupling_jacobi;
use prism_render_architecture::cloth::{ClothParticle, Vec3};

/// 平方长度地板，低于此视作零向量，镜像黄金 `EPS_LEN_SQ` 与 WESL
/// `CLOTH_LAYERS_EPS_LEN_SQ`（同值 `1.0e-12`），使归一化与径向重合判定一致。
const EPS_LEN_SQ: f32 = 1.0e-12;

/// 分量点积 `a · b`，按 WESL `dot` 的 `x*rx + y*ry + z*rz` 同序。
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// 分量差 `a - b`。
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// 分量 × 标量 `v * s`。
fn scale(v: [f32; 3], s: f32) -> [f32; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

/// `cell_of` 的逐位转写：`inv = 1.0 / cell_size`，每轴 `(coord * inv).floor() as i32`
/// （`as` 饱和转换，与黄金一致）。`cell_size` 必须为已 sanitized 的正值。
fn wesl_cell_of(p: [f32; 3], cell_size: f32) -> (i32, i32, i32) {
    let inv = 1.0 / cell_size;
    let cx = (p[0] * inv).floor() as i32;
    let cy = (p[1] * inv).floor() as i32;
    let cz = (p[2] * inv).floor() as i32;
    (cx, cy, cz)
}

/// `glam::Vec3::normalize_or_zero` 的逐位转写，镜像 WESL
/// `cloth_layers_normalize_or_zero` 与物理引擎 WGSL 孪生：倒数长度有限且为正时
/// 归一化，否则返回零向量。只有恰为零或非有限长度的法线才回退径向，微小非零法线
/// 仍保留为可用方向。
fn wesl_normalize_or_zero(v: [f32; 3]) -> [f32; 3] {
    let len = dot(v, v).sqrt();
    let rcp = 1.0 / len;
    if rcp.is_finite() && rcp > 0.0 {
        scale(v, rcp)
    } else {
        [0.0; 3]
    }
}

/// WESL `half_layer_correction` 的逐位转写：返回粒子 `a` 对抗跨层邻居 `b` 的
/// 「本粒子那一半」分离修正。较低层号为内层（其外法线定向接触），外层沿法线被
/// 推到至少 `thickness`；法线退化时回退到对称径向最小距离推开。
fn wesl_half_layer_correction(
    pos: &[[f32; 3]],
    inv_mass: &[f32],
    layer_of: &[u32],
    normals: &[[f32; 3]],
    a: usize,
    b: usize,
    thickness: f32,
    thickness_sq: f32,
) -> [f32; 3] {
    // WESL：默认 inner=b, outer=a；若 layer_of[a] < layer_of[b] 则 inner=a, outer=b。
    let (inner, outer) = if layer_of[a] < layer_of[b] {
        (a, b)
    } else {
        (b, a)
    };

    let w_inner = inv_mass[inner].max(0.0);
    let w_outer = inv_mass[outer].max(0.0);
    let w_sum = w_inner + w_outer;
    if w_sum <= 0.0 {
        return [0.0; 3];
    }

    let p_inner = pos[inner];
    let p_outer = pos[outer];
    let normal = normals[inner];

    // normalize_or_zero：镜像物理引擎单一真源 glam::Vec3::normalize_or_zero——
    // 倒数长度有限且为正即归一化，否则零向量；微小但非零的法线仍视作可用方向。
    let unit = wesl_normalize_or_zero(normal);

    if dot(unit, unit) > EPS_LEN_SQ {
        // 定向平面接触：沿内层外法线把外层推到至少 thickness。
        let signed = dot(sub(p_outer, p_inner), unit);
        if signed >= thickness {
            return [0.0; 3];
        }
        let penetration = thickness - signed;
        if a == inner {
            return scale(unit, -penetration * (w_inner / w_sum));
        }
        return scale(unit, penetration * (w_outer / w_sum));
    }

    // 无可用法线：对称径向最小距离分离。
    let delta = sub(p_outer, p_inner);
    let dist_sq = dot(delta, delta);
    if dist_sq >= thickness_sq {
        return [0.0; 3];
    }
    let mut dir = [1.0_f32, 0.0, 0.0];
    let mut penetration = thickness;
    if dist_sq > EPS_LEN_SQ {
        let dist = dist_sq.sqrt();
        dir = scale(delta, 1.0 / dist);
        penetration = thickness - dist;
    }
    if a == inner {
        scale(dir, -penetration * (w_inner / w_sum))
    } else {
        scale(dir, penetration * (w_outer / w_sum))
    }
}

/// 以 WESL 内核语义（host 预建 CSR + `cloth_layer_coupling` own-slot 累加）跑一遍
/// 层间耦合，返回修正后的位置。sanitize/guard/网格遍历顺序均逐位复刻黄金。
fn run_wesl_layers(
    particles: &[ClothParticle],
    layer_of: &[u32],
    normals: &[Vec3],
    params: LayerParams,
) -> Vec<[f32; 3]> {
    let pos: Vec<[f32; 3]> = particles
        .iter()
        .map(|p| [p.position.x, p.position.y, p.position.z])
        .collect();
    let inv_mass: Vec<f32> = particles.iter().map(|p| p.inverse_mass).collect();
    let nrm: Vec<[f32; 3]> = normals.iter().map(|n| [n.x, n.y, n.z]).collect();

    // 用黄金自身的 sanitize 以保证 thickness / cell_size 与黄金完全同值。
    let sane = params.sanitized();
    let thickness = sane.thickness;
    let cell_size = sane.cell_size;

    let mut corrected = pos.clone();
    if thickness <= 0.0 || cell_size <= 0.0 || particles.len() < 2 {
        return corrected;
    }

    // 复刻黄金空间哈希：仅 index < layer_of.len() 的粒子入格，升序入桶。
    let mut grid: BTreeMap<(i32, i32, i32), Vec<u32>> = BTreeMap::new();
    for (index, p) in pos.iter().enumerate() {
        if index < layer_of.len() {
            let cell = wesl_cell_of(*p, cell_size);
            grid.entry(cell).or_default().push(index as u32);
        }
    }

    let thickness_sq = thickness * thickness;
    // 每个粒子 own-slot：按黄金的 27 邻格 (dx,dy,dz) + 桶内升序求和自己那一半。
    // out[a] 相互独立，故外层按索引遍历与黄金按格遍历结果逐位相同。
    for a in 0..particles.len() {
        if a >= layer_of.len() {
            continue;
        }
        let cell = wesl_cell_of(pos[a], cell_size);
        let mut accum = [0.0_f32; 3];
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let neighbor = (cell.0 + dx, cell.1 + dy, cell.2 + dz);
                    let Some(nbucket) = grid.get(&neighbor) else {
                        continue;
                    };
                    for &b in nbucket {
                        let bi = b as usize;
                        if bi == a {
                            continue;
                        }
                        if layer_of[a] == layer_of[bi] {
                            continue;
                        }
                        let half = wesl_half_layer_correction(
                            &pos,
                            &inv_mass,
                            layer_of,
                            &nrm,
                            a,
                            bi,
                            thickness,
                            thickness_sq,
                        );
                        accum[0] += half[0];
                        accum[1] += half[1];
                        accum[2] += half[2];
                    }
                }
            }
        }
        corrected[a][0] = pos[a][0] + accum[0];
        corrected[a][1] = pos[a][1] + accum[1];
        corrected[a][2] = pos[a][2] + accum[2];
    }
    corrected
}

/// 逐位断言：黄金修正后的 `ClothParticle` 位置与 WESL 转写位置的每个分量
/// `to_bits` 完全相等。
fn assert_bit_exact(golden: &[ClothParticle], wesl: &[[f32; 3]]) {
    assert_eq!(golden.len(), wesl.len(), "length mismatch");
    for (i, (g, w)) in golden.iter().zip(wesl.iter()).enumerate() {
        assert_eq!(
            g.position.x.to_bits(),
            w[0].to_bits(),
            "x bits differ at {i}"
        );
        assert_eq!(
            g.position.y.to_bits(),
            w[1].to_bits(),
            "y bits differ at {i}"
        );
        assert_eq!(
            g.position.z.to_bits(),
            w[2].to_bits(),
            "z bits differ at {i}"
        );
    }
}

/// 跑黄金 + WESL 转写并断言逐位一致，返回黄金结果供额外断言。
fn run_both(
    particles: &[ClothParticle],
    layer_of: &[u32],
    normals: &[Vec3],
    params: LayerParams,
) -> Vec<ClothParticle> {
    let mut golden = particles.to_vec();
    resolve_layer_coupling_jacobi(&mut golden, layer_of, normals, params);
    let wesl = run_wesl_layers(particles, layer_of, normals, params);
    assert_bit_exact(&golden, &wesl);
    golden
}

fn free(x: f32, y: f32, z: f32, inv_mass: f32) -> ClothParticle {
    ClothParticle::new(Vec3::new(x, y, z), inv_mass)
}

fn params(thickness: f32, cell_size: f32) -> LayerParams {
    LayerParams {
        thickness,
        cell_size,
    }
}

#[test]
fn single_oriented_contact_bit_for_bit() {
    let base = [free(0.0, 0.0, 0.0, 1.0), free(0.0, -0.05, 0.0, 1.0)];
    let layer_of = [0u32, 1u32];
    let normals = [Vec3::new(0.0, 1.0, 0.0), Vec3::ZERO];
    let out = run_both(&base, &layer_of, &normals, params(0.1, 0.2));
    // 外层被抬到法线 + 侧至少 thickness。
    let signed = out[1].position.y - out[0].position.y;
    assert!(signed >= 0.1 - 1.0e-6, "signed separation {signed}");
}

#[test]
fn single_radial_contact_bit_for_bit() {
    let base = [free(0.0, 0.0, 0.0, 1.0), free(0.02, 0.0, 0.0, 1.0)];
    let layer_of = [0u32, 1u32];
    let normals = [Vec3::ZERO, Vec3::ZERO];
    run_both(&base, &layer_of, &normals, params(0.1, 0.2));
}

#[test]
fn coincident_particles_separate_along_x_bit_for_bit() {
    let base = [free(0.3, 0.3, 0.3, 1.0), free(0.3, 0.3, 0.3, 1.0)];
    let layer_of = [0u32, 1u32];
    let normals = [Vec3::ZERO, Vec3::ZERO];
    run_both(&base, &layer_of, &normals, params(0.1, 0.2));
}

#[test]
fn pinned_inner_takes_whole_push_bit_for_bit() {
    let base = [free(0.0, 0.0, 0.0, 0.0), free(0.0, -0.05, 0.0, 1.0)];
    let layer_of = [0u32, 1u32];
    let normals = [Vec3::new(0.0, 1.0, 0.0), Vec3::ZERO];
    let out = run_both(&base, &layer_of, &normals, params(0.1, 0.2));
    assert!(out[0].position.y.abs() < 1.0e-9, "pinned inner moved");
}

#[test]
fn pinned_outer_keeps_inner_taking_push_bit_for_bit() {
    // 外层钉死（inv_mass 0），w_outer=0：外层保持原位，内层独自吃下整条修正。
    let base = [free(0.0, 0.0, 0.0, 1.0), free(0.0, -0.05, 0.0, 0.0)];
    let layer_of = [0u32, 1u32];
    let normals = [Vec3::new(0.0, 1.0, 0.0), Vec3::ZERO];
    let out = run_both(&base, &layer_of, &normals, params(0.1, 0.2));
    // 钉死的外层逐位不动。
    assert_eq!(
        out[1].position.y.to_bits(),
        (-0.05_f32).to_bits(),
        "pinned outer moved"
    );
    // 自由内层被沿法线反向推下，发生位移。
    assert!(
        out[0].position.y < -1.0e-6,
        "free inner did not take the push"
    );
}

#[test]
fn both_pinned_is_a_noop_bit_for_bit() {
    let base = [free(0.0, 0.0, 0.0, 0.0), free(0.0, -0.05, 0.0, 0.0)];
    let layer_of = [0u32, 1u32];
    let normals = [Vec3::new(0.0, 1.0, 0.0), Vec3::ZERO];
    let out = run_both(&base, &layer_of, &normals, params(0.1, 0.2));
    assert_eq!(out[0].position.y.to_bits(), 0.0_f32.to_bits());
    assert_eq!(out[1].position.y.to_bits(), (-0.05_f32).to_bits());
}

#[test]
fn same_layer_pair_is_untouched_bit_for_bit() {
    let base = [free(0.0, 0.0, 0.0, 1.0), free(0.0, 0.01, 0.0, 1.0)];
    let layer_of = [2u32, 2u32];
    let normals = [Vec3::new(0.0, 1.0, 0.0), Vec3::ZERO];
    run_both(&base, &layer_of, &normals, params(0.1, 0.2));
}

#[test]
fn separated_pair_is_untouched_bit_for_bit() {
    let base = [free(0.0, 0.0, 0.0, 1.0), free(0.0, 0.5, 0.0, 1.0)];
    let layer_of = [0u32, 1u32];
    let normals = [Vec3::new(0.0, 1.0, 0.0), Vec3::ZERO];
    run_both(&base, &layer_of, &normals, params(0.1, 0.2));
}

#[test]
fn mixed_inverse_masses_split_bit_for_bit() {
    // 不等逆质量：触发 w_inner / w_sum、w_outer / w_sum 的非平凡除法分配。
    let base = [free(0.01, 0.0, 0.0, 2.0), free(-0.01, 0.0, 0.0, 0.5)];
    let layer_of = [1u32, 0u32];
    let normals = [Vec3::ZERO, Vec3::new(-1.0, 0.0, 0.0)];
    run_both(&base, &layer_of, &normals, params(0.1, 0.2));
}

#[test]
fn non_unit_normal_is_normalized_bit_for_bit() {
    // 非单位法线：触发 normalize_or_zero 的倒数乘 reciprocal-sqrt。
    let base = [free(0.0, 0.0, 0.0, 1.0), free(0.03, 0.04, 0.0, 1.0)];
    let layer_of = [0u32, 1u32];
    let normals = [Vec3::new(0.0, 3.0, 4.0), Vec3::ZERO];
    run_both(&base, &layer_of, &normals, params(0.1, 0.2));
}

#[test]
fn degenerate_normal_falls_back_to_radial_bit_for_bit() {
    // 法线平方长度 (3e-7)^2 ≈ 9e-14 < EPS，走径向回退分支。
    let base = [free(0.0, 0.0, 0.0, 1.0), free(0.0, -0.05, 0.0, 1.0)];
    let layer_of = [0u32, 1u32];
    let normals = [Vec3::new(0.0, 3.0e-7, 0.0), Vec3::ZERO];
    run_both(&base, &layer_of, &normals, params(0.1, 0.2));
}

#[test]
fn three_layer_stack_multi_neighbor_bit_for_bit() {
    // 中层粒子同时被内层与外层穿透（分处不同格），测跨格多邻居 own-slot 累加序。
    let base = [
        free(0.0, 0.0, 0.0, 1.0),  // 内层 0
        free(0.0, 0.06, 0.0, 1.0), // 中层 1（被 0 与 2 夹）
        free(0.0, 0.12, 0.0, 1.0), // 外层 2
    ];
    let layer_of = [0u32, 1u32, 2u32];
    let normals = [
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::ZERO,
    ];
    run_both(&base, &layer_of, &normals, params(0.1, 0.2));
}

#[test]
fn disabled_thickness_is_zero_corrections_bit_for_bit() {
    let base = [free(0.0, 0.0, 0.0, 1.0), free(0.0, -0.05, 0.0, 1.0)];
    let layer_of = [0u32, 1u32];
    let normals = [Vec3::new(0.0, 1.0, 0.0), Vec3::ZERO];
    run_both(&base, &layer_of, &normals, params(0.0, 0.2));
}

#[test]
fn disabled_cell_size_is_zero_corrections_bit_for_bit() {
    let base = [free(0.0, 0.0, 0.0, 1.0), free(0.0, -0.05, 0.0, 1.0)];
    let layer_of = [0u32, 1u32];
    let normals = [Vec3::new(0.0, 1.0, 0.0), Vec3::ZERO];
    run_both(&base, &layer_of, &normals, params(0.1, -1.0));
}

#[test]
fn fewer_than_two_particles_is_a_noop_bit_for_bit() {
    let base = [free(0.0, 0.0, 0.0, 1.0)];
    let layer_of = [0u32];
    let normals = [Vec3::new(0.0, 1.0, 0.0)];
    run_both(&base, &layer_of, &normals, params(0.1, 0.2));
}

#[test]
fn missing_layer_entry_is_skipped_bit_for_bit() {
    // 末粒子无 layer_of 条目：黄金不将其入格、也不作为邻居，逐位应保持原位。
    let base = [
        free(0.0, 0.0, 0.0, 1.0),
        free(0.0, -0.05, 0.0, 1.0),
        free(0.0, -0.04, 0.0, 1.0),
    ];
    let layer_of = [0u32, 1u32];
    let normals = [Vec3::new(0.0, 1.0, 0.0), Vec3::ZERO, Vec3::ZERO];
    let out = run_both(&base, &layer_of, &normals, params(0.1, 0.2));
    assert_eq!(out[2].position.y.to_bits(), (-0.04_f32).to_bits());
}

#[test]
fn dense_jittered_grid_bit_for_bit() {
    // 跨多格的抖动密集两层场景：全程遍历序 + 多邻居累加逐位可复。
    let mut base = Vec::new();
    let mut layer_of = Vec::new();
    let mut normals = Vec::new();
    let mut seed: u32 = 0x9e37_79b9;
    let mut rng = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        (seed as f32 / u32::MAX as f32) - 0.5
    };
    for gx in 0..5 {
        for gz in 0..5 {
            let bx = gx as f32 * 0.07;
            let bz = gz as f32 * 0.07;
            // 内层
            base.push(free(
                bx + 0.01 * rng(),
                0.0 + 0.01 * rng(),
                bz + 0.01 * rng(),
                1.0,
            ));
            layer_of.push(0u32);
            normals.push(Vec3::new(0.0, 1.0, 0.0));
            // 外层（略高于内层但在 thickness 内，制造穿透）
            base.push(free(
                bx + 0.01 * rng(),
                0.03 + 0.01 * rng(),
                bz + 0.01 * rng(),
                1.0,
            ));
            layer_of.push(1u32);
            normals.push(Vec3::ZERO);
        }
    }
    run_both(&base, &layer_of, &normals, params(0.08, 0.1));
}
