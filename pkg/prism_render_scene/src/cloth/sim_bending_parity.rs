//! `cloth_project_bending_batch` 内核的**逐位** CPU parity，直接对齐架构层
//! 黄金 [`project_bending`]（`pub`，可跨 crate 调用）。
//!
//! `shader_tests` 只做 naga 编译门禁，证明不了 `cloth_sim.wesl` 的二面角弯曲
//! 投影逐位复刻了 CPU 黄金的算术（设计 §9：不造假 parity）。本模块把该内核的
//! 逐条算术独立转写成 CPU 版本（原生数组，不复用黄金 `Vec3` 方法），对多种
//! 四顶点铰链配置断言其输出位置与黄金 [`project_bending`] **逐位（`to_bits`）
//! 一致**。
//!
//! ## 为什么弯曲投影可以做到真正逐位（区别于 distance 投影）
//! 距离投影 `cloth_project_distance_batch` 的方向归一化 `direction = delta / dist`
//! 是**逐分量除法**，而黄金 `project_distance` 用 `delta.scale(1.0 / dist)`
//! （先取**倒数**再乘），除法与倒数乘在 IEEE 下不逐位，故距离/长程投影只能由
//! `sim_gpu_tests` 带容差覆盖。弯曲投影则**全程无归一化除法**：
//! * 弯曲向量 `S = Σ wᵢ·xᵢ`（分量 × 标量后累加，与黄金 [`BendingConstraint::bend_vector`]
//!   同序）。
//! * 能量 `E = 0.5 · scale · |S|²`、分母 `Σ wmassᵢ·(scale·wᵢ)²·|S|²`、
//!   `α̃ = max(compliance,0) / dt_sub²` 全是乘加。
//! * 唯一的除法 `Δλ = -E / denom` 两侧**同为除法**（非倒数乘），逐位一致。
//! * 修正 `Δxᵢ = S · (wmassᵢ·Δλ·scale·wᵢ)` 为分量 × 标量后相加，同序。
//! 因此黄金与 `WESL` 转写在**全部**算术上逐位同构，位置可做 `to_bits` 严格比对。
//!
//! ## 逐位一致的关键约定
//! * `cloth_inverse_mass(w) = max(w, 0.0)` 与黄金 `effective_inverse_mass`
//!   （pinned → 0，否则 `inverse_mass`，即 `max(inverse_mass, 0)`）一致。
//! * `CLOTH_EPS_LEN_SQ = 1.0e-12` 与黄金 `EPS_LEN_SQ = 1e-12` 同值；`|S|² ≤ eps`
//!   的 flat-rest 早退两侧一致（位置不变）。
//! * 黄金 `project_bending` 对 `dt_sub <= 0.0` 直接早退不改位置；`WESL` 内核靠 host
//!   提供正 `dt_sub`。本 harness 的转写保留 `dt_sub <= 0.0` 守卫并只在 `dt_sub > 0.0`
//!   上做 parity。
//! * 多铰链按**相同顺序**在两侧依次原位应用（铰链内 `S` 在应用循环前算定一次，
//!   修正只读该定值，故逐铰链顺序投影两侧严格一致）。

#![cfg(test)]

use prism_render_architecture::cloth::bending::{project_bending, BendingConstraint};
use prism_render_architecture::cloth::{ClothParticle, Compliance, Vec3};

/// `CLOTH_EPS_LEN_SQ`（= 黄金 `EPS_LEN_SQ`）：`|S|²` 退化阈值。
const EPS_LEN_SQ: f32 = 1.0e-12;

/// 一个四顶点弯曲铰链输入：`[edge0, edge1, apex_a, apex_b]` 索引 + 权重 +
/// 面积尺度 + 原始 compliance（未 clamp，由两侧各自 `max(.,0)`）。
#[derive(Clone, Copy)]
struct Hinge {
    vertices: [u32; 4],
    weights: [f32; 4],
    scale: f32,
    compliance: f32,
}

/// `cloth_sim.wesl` `cloth_project_bending_batch` 的逐位 CPU 转写（原生数组）。
/// 就地更新 `positions`，`inv_masses` 为每粒子已 clamp 的逆质量快照。
fn wesl_project_bending(
    positions: &mut [[f32; 3]],
    inv_masses: &[f32],
    hinge: Hinge,
    dt_sub: f32,
) {
    if dt_sub <= 0.0 {
        // 黄金 project_bending 对 dt_sub <= 0 早退不改位置。
        return;
    }
    let idx = [
        hinge.vertices[0] as usize,
        hinge.vertices[1] as usize,
        hinge.vertices[2] as usize,
        hinge.vertices[3] as usize,
    ];
    let wgt = hinge.weights;

    // 弯曲向量 S = Σ wᵢ·xᵢ（分量 × 标量后累加，与黄金 bend_vector 同序）。
    let mut s = [0.0f32, 0.0, 0.0];
    for i in 0..4 {
        let pos = positions[idx[i]];
        s[0] += pos[0] * wgt[i];
        s[1] += pos[1] * wgt[i];
        s[2] += pos[2] * wgt[i];
    }
    let s_len_sq = s[0] * s[0] + s[1] * s[1] + s[2] * s[2];
    if s_len_sq <= EPS_LEN_SQ {
        return;
    }
    let energy = 0.5 * hinge.scale * s_len_sq;

    let mut sum_w_grad = 0.0f32;
    for i in 0..4 {
        let inv_mass = inv_masses[idx[i]].max(0.0);
        if inv_mass <= 0.0 {
            continue;
        }
        let grad_scalar = hinge.scale * wgt[i];
        sum_w_grad += inv_mass * grad_scalar * grad_scalar * s_len_sq;
    }
    let alpha_tilde = hinge.compliance.max(0.0) / (dt_sub * dt_sub);
    let denom = sum_w_grad + alpha_tilde;
    if denom <= 0.0 {
        return;
    }
    let d_lambda = -energy / denom;

    for i in 0..4 {
        let inv_mass = inv_masses[idx[i]].max(0.0);
        if inv_mass <= 0.0 {
            continue;
        }
        let factor = inv_mass * d_lambda * hinge.scale * wgt[i];
        positions[idx[i]][0] += s[0] * factor;
        positions[idx[i]][1] += s[1] * factor;
        positions[idx[i]][2] += s[2] * factor;
    }
}

/// 单条粒子输入：`(position, inverse_mass)`（弯曲投影不读速度）。
type ParticleInput = ([f32; 3], f32);

/// 跑黄金 [`project_bending`] 与 `WESL` 转写（相同铰链顺序），逐分量断言
/// 位置 `to_bits` 一致。
fn assert_bending_bit_exact(particles: &[ParticleInput], hinges: &[Hinge], dt_sub: f32) {
    assert!(dt_sub > 0.0, "harness 只在 dt_sub > 0.0 上做 parity");

    let mut golden: Vec<ClothParticle> = particles
        .iter()
        .map(|&(pos, inv_mass)| {
            if inv_mass <= 0.0 {
                ClothParticle::pinned(Vec3::new(pos[0], pos[1], pos[2]))
            } else {
                ClothParticle::new(Vec3::new(pos[0], pos[1], pos[2]), inv_mass)
            }
        })
        .collect();

    // 从黄金起始状态快照 WESL 侧位置与逆质量，确保两侧初态逐位相同。
    let mut wpos: Vec<[f32; 3]> = golden
        .iter()
        .map(|c| [c.position.x, c.position.y, c.position.z])
        .collect();
    let inv_masses: Vec<f32> = golden.iter().map(|c| c.inverse_mass).collect();

    for &hinge in hinges {
        let constraint = BendingConstraint {
            vertices: hinge.vertices,
            weights: hinge.weights,
            scale: hinge.scale,
            compliance: Compliance(hinge.compliance),
        };
        project_bending(&mut golden, constraint, dt_sub);
        wesl_project_bending(&mut wpos, &inv_masses, hinge, dt_sub);
    }

    for (i, cp) in golden.iter().enumerate() {
        let got = [cp.position.x, cp.position.y, cp.position.z];
        for k in 0..3 {
            assert_eq!(
                got[k].to_bits(),
                wpos[i][k].to_bits(),
                "粒子 {i} 分量 {k} 位置不逐位：黄金 {} ({:#010x}) vs WESL {} ({:#010x})",
                got[k],
                got[k].to_bits(),
                wpos[i][k],
                wpos[i][k].to_bits(),
            );
        }
    }
}

/// 单铰链、折叠非平面、均匀质量：rigid（compliance == 0）。
#[test]
fn single_hinge_fold_rigid() {
    assert_bending_bit_exact(
        &[
            ([0.0, 0.0, 0.0], 1.0),
            ([1.0, 0.0, 0.0], 1.0),
            ([0.5, 0.8, 0.2], 1.0),
            ([0.5, -0.6, -0.3], 1.0),
        ],
        &[Hinge {
            vertices: [0, 1, 2, 3],
            weights: [1.0, 1.0, -1.0, -1.0],
            scale: 2.0,
            compliance: 0.0,
        }],
        1.0 / 60.0,
    );
}

/// 单铰链、软（compliance > 0）。
#[test]
fn single_hinge_compliant() {
    assert_bending_bit_exact(
        &[
            ([-0.3, 0.1, 0.0], 1.0),
            ([1.2, 0.0, 0.1], 1.0),
            ([0.4, 0.9, -0.2], 1.0),
            ([0.6, -0.7, 0.5], 1.0),
        ],
        &[Hinge {
            vertices: [0, 1, 2, 3],
            weights: [0.5, 0.5, -0.5, -0.5],
            scale: 1.5,
            compliance: 1.0e-4,
        }],
        1.0 / 90.0,
    );
}

/// 一个 apex pinned（`inverse_mass == 0`）：该顶点跳过累加与修正。
#[test]
fn single_hinge_one_apex_pinned() {
    assert_bending_bit_exact(
        &[
            ([0.0, 0.0, 0.0], 1.0),
            ([1.0, 0.0, 0.0], 1.0),
            ([0.5, 0.7, 0.0], 0.0),
            ([0.5, -0.5, 0.4], 2.0),
        ],
        &[Hinge {
            vertices: [0, 1, 2, 3],
            weights: [1.0, 1.0, -1.0, -1.0],
            scale: 3.0,
            compliance: 0.0,
        }],
        1.0 / 60.0,
    );
}

/// 平坦铰链（`S == 0`）：`|S|² ≤ eps` 早退，两侧位置不变（no-op）。
#[test]
fn flat_stencil_is_a_noop() {
    // weights [1,-1,1,-1] 且 p0==p1、p2==p3 ⇒ S = (p0-p1)+(p2-p3) = 0。
    assert_bending_bit_exact(
        &[
            ([0.3, 0.4, 0.5], 1.0),
            ([0.3, 0.4, 0.5], 1.0),
            ([-0.2, 0.1, 0.9], 1.0),
            ([-0.2, 0.1, 0.9], 1.0),
        ],
        &[Hinge {
            vertices: [0, 1, 2, 3],
            weights: [1.0, -1.0, 1.0, -1.0],
            scale: 2.0,
            compliance: 0.0,
        }],
        1.0 / 60.0,
    );
}

/// 负权重（cotangent Laplacian 可为负）+ 非均匀质量。
#[test]
fn negative_weights_mixed_masses() {
    assert_bending_bit_exact(
        &[
            ([0.1, -0.2, 0.3], 0.5),
            ([1.1, 0.2, -0.1], 1.5),
            ([0.5, 1.0, 0.4], 2.0),
            ([0.4, -0.9, -0.6], 0.75),
        ],
        &[Hinge {
            vertices: [0, 1, 2, 3],
            weights: [-1.3, 0.7, 1.1, -0.5],
            scale: 1.25,
            compliance: 5.0e-5,
        }],
        1.0 / 72.0,
    );
}

/// 负 compliance：两侧都 `max(.,0)` clamp 到 0（等同 rigid）。
#[test]
fn negative_compliance_is_clamped() {
    assert_bending_bit_exact(
        &[
            ([0.0, 0.0, 0.0], 1.0),
            ([1.0, 0.2, 0.0], 1.0),
            ([0.5, 0.8, 0.3], 1.0),
            ([0.5, -0.6, -0.2], 1.0),
        ],
        &[Hinge {
            vertices: [0, 1, 2, 3],
            weights: [1.0, 1.0, -1.0, -1.0],
            scale: 2.0,
            compliance: -0.5,
        }],
        1.0 / 60.0,
    );
}

/// 全部顶点 pinned：`sum_w_grad == 0` 且 `compliance == 0` ⇒ `denom <= 0`
/// 早退（no-op）。
#[test]
fn all_pinned_is_a_noop() {
    assert_bending_bit_exact(
        &[
            ([0.0, 0.0, 0.0], 0.0),
            ([1.0, 0.0, 0.0], 0.0),
            ([0.5, 0.9, 0.1], 0.0),
            ([0.5, -0.7, -0.4], 0.0),
        ],
        &[Hinge {
            vertices: [0, 1, 2, 3],
            weights: [1.0, 1.0, -1.0, -1.0],
            scale: 2.0,
            compliance: 0.0,
        }],
        1.0 / 60.0,
    );
}

/// 大 `dt_sub`（30fps）检验 `α̃` 缩放。
#[test]
fn larger_timestep_compliant() {
    assert_bending_bit_exact(
        &[
            ([0.2, 0.1, -0.3], 1.0),
            ([1.3, -0.2, 0.2], 1.0),
            ([0.7, 0.95, 0.1], 1.0),
            ([0.6, -0.85, 0.55], 1.0),
        ],
        &[Hinge {
            vertices: [0, 1, 2, 3],
            weights: [0.9, 0.9, -0.9, -0.9],
            scale: 1.0,
            compliance: 2.0e-4,
        }],
        1.0 / 30.0,
    );
}

/// 多铰链共享顶点，按相同顺序串行应用（Gauss-Seidel 跨铰链）。
#[test]
fn multiple_hinges_shared_vertices() {
    assert_bending_bit_exact(
        &[
            ([0.0, 0.0, 0.0], 1.0),
            ([1.0, 0.0, 0.0], 1.0),
            ([0.5, 0.8, 0.2], 1.0),
            ([0.5, -0.6, -0.3], 1.0),
            ([1.5, 0.7, 0.1], 1.0),
            ([2.0, -0.5, 0.4], 1.0),
        ],
        &[
            Hinge {
                vertices: [0, 1, 2, 3],
                weights: [1.0, 1.0, -1.0, -1.0],
                scale: 2.0,
                compliance: 0.0,
            },
            Hinge {
                vertices: [1, 4, 2, 5],
                weights: [0.8, -0.8, 0.6, -0.6],
                scale: 1.5,
                compliance: 1.0e-4,
            },
        ],
        1.0 / 60.0,
    );
}

/// 抖动稠密：多铰链、不同质量/权重/尺度，检验数值鲁棒性。
#[test]
fn jittered_dense_hinges() {
    let particles: Vec<ParticleInput> = (0..10)
        .map(|i| {
            let f = i as f32;
            (
                [f.mul_add(0.21, -1.0), (f * 0.5).sin(), (f * 0.3).cos() * 0.4],
                0.5 + f * 0.1,
            )
        })
        .collect();
    let hinges: Vec<Hinge> = (0..6)
        .map(|i| {
            let f = i as f32;
            Hinge {
                vertices: [i, i + 1, i + 2, i + 3],
                weights: [
                    1.0 + f * 0.1,
                    -(1.0 + f * 0.05),
                    0.7 - f * 0.03,
                    -(0.6 + f * 0.04),
                ],
                scale: 1.0 + f * 0.2,
                compliance: f * 1.0e-5,
            }
        })
        .collect();
    assert_bending_bit_exact(&particles, &hinges, 1.0 / 120.0);
}
