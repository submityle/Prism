#![cfg(test)]
//! `cloth_vbd.wesl` 的 Vertex Block Descent 内核的 **逐位** CPU 转写，对齐架构层
//! 黄金 [`solve_cloth_vbd_colored`]（color-major 调度：同色 Jacobi、跨色
//! Gauss-Seidel，正是 GPU 每色一次 dispatch 的 schedule）。
//!
//! `vbd_gpu_tests` 已覆盖「真机 GPU vs 黄金 **带容差**(`VBD_PARITY_EPS = 2e-3`)」，
//! 其容差源于设备级 fused-multiply-add 收缩（非 op-order bug）。本模块补上**与设备
//! 无关**的另一条闭环：把 `cloth_vbd.wesl` 的 predict / sweep / velocity 三段算术
//! **独立**转写成原生 `f32`（不复用黄金的 `Vec3` / `Mat3` 方法，而是照 `WESL` 源码
//! 逐词重写），按同一 color-major schedule 驱动，再对多种网格断言其 position 与
//! velocity 与黄金 **逐位(`to_bits`)一致**。无 FMA、纯 `mul`/`add`，故构造正确即
//! 必然逐位吻合，与 `vbd_gpu_tests` 的真机带容差互补。
//!
//! ## 逐位一致的关键约定（已逐行核对 `WESL` vs 黄金 op-order 全同序）
//! * predict：`(x + (vel*retain)*dt_sub) + gravity_step`，左结合；`gravity_step =
//!   gravity * dt_sub_sq`、`retain = 1 - damping`。
//! * inertia：`mass = 1/inverse_mass`、`inertia = mass / dt_sub_sq`；grad 初值
//!   `(x - target) * inertia`，hess 初值 `scaled_identity(inertia)`。
//! * accumulate：`n = d * (1/len)`（先 `sqrt` 再取倒数相乘）；`grad += n * (k *
//!   (len - rest))`（标量系数先算）；`tangential = max(1 - rest/len, 0)`（**直接除**
//!   `rest/len`）；先 `hess += scaled_identity(k * tangential)` 再 `hess +=
//!   scaled_outer(n, k * (1 - tangential))`，`scaled_outer` 内 `(s * ni) * nj` 左结合。
//! * `Mat3::solve`：cofactor `c00 = m4*m8 - m5*m7` … `det = m0*c00 + m1*c01 +
//!   m2*c02`；`|det| < EPS_DET` 守卫；`inv_det = 1/det`；`dx = (c00*gx + c10*gy +
//!   c20*gz) * inv_det`，非有限结果视作不移动该顶点。
//! * velocity：`(x - x_prev) * inv_dt_sub`，pinned 顶点冻结为零。
//! * one-sided（LRA / tether）：`len <= rest` 的松弛约束不贡献；stiffness
//!   `if alpha > 0 { min(1/(alpha*dt_sub_sq), 1e9) } else { 1e9 }`。
//! * 常量全等：`EPS_LEN_SQ = 1e-12`、`EPS_DET = 1e-20`、`MAX_STIFFNESS = 1e9`。

use prism_render_architecture::cloth::vbd::{solve_cloth_vbd_colored, VbdParams};
use prism_render_architecture::cloth::vbd_coloring::color_cloth_vertices;
use prism_render_architecture::cloth::{
    ClothParticle, Compliance, Constraint, ConstraintKind, Vec3, EPS_LEN_SQ,
};

/// 对称 3x3 行列式低于此阈值时跳过求解，避免除以约零的奇异 Hessian。
/// 与黄金私有 `vbd::EPS_DET` 全等（在此本地重定义，因其未公开）。
const EPS_DET: f32 = 1.0e-20;
/// 每约束刚度上限；刚性约束(`compliance == 0`)或极小 compliance 否则会推出无穷大
/// 刚度。与黄金私有 `vbd::MAX_STIFFNESS` 全等。
const MAX_STIFFNESS: f32 = 1.0e9;

/// `cloth_vbd.wesl` 里把 XPBD `compliance` 折算成 VBD 能量刚度 `k` 的逐位转写：
/// `alpha > 0` 时 `min(1/(alpha*dt_sub_sq), 1e9)`，否则 `1e9`。
fn wesl_stiffness(alpha: f32, dt_sub_sq: f32) -> f32 {
    if alpha > 0.0 {
        (1.0 / (alpha * dt_sub_sq)).min(MAX_STIFFNESS)
    } else {
        MAX_STIFFNESS
    }
}

/// `cloth_vbd_predict` 的逐位转写：`target = (x + (vel*retain)*dt_sub) +
/// gravity_step`（左结合），并快照 pre-solve 位置。
fn wesl_predict(
    position: [f32; 3],
    velocity: [f32; 3],
    retain: f32,
    dt_sub: f32,
    gravity_step: [f32; 3],
) -> [f32; 3] {
    let v = [velocity[0] * retain, velocity[1] * retain, velocity[2] * retain];
    let stepped = [
        position[0] + v[0] * dt_sub,
        position[1] + v[1] * dt_sub,
        position[2] + v[2] * dt_sub,
    ];
    [
        stepped[0] + gravity_step[0],
        stepped[1] + gravity_step[1],
        stepped[2] + gravity_step[2],
    ]
}

/// 累加一条距离约束对某顶点 grad(3) 与 Hessian(9，行主序) 的贡献，逐位复刻黄金
/// `accumulate_constraint`：PSD 投影的 `k·nnᵀ + k·max(0,1-rest/len)·(I-nnᵀ)`，压缩
/// (`len < rest`) 时丢弃不定部分；one-sided 且 `len <= rest` 时不贡献。
fn wesl_accumulate(
    grad: &mut [f32; 3],
    hess: &mut [f32; 9],
    x: [f32; 3],
    other: [f32; 3],
    rest: f32,
    k: f32,
    one_sided: bool,
) {
    let d = [x[0] - other[0], x[1] - other[1], x[2] - other[2]];
    let len_sq = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    if len_sq < EPS_LEN_SQ {
        return;
    }
    let len = len_sq.sqrt();
    if one_sided && len <= rest {
        return;
    }
    let inv_len = 1.0 / len;
    let n = [d[0] * inv_len, d[1] * inv_len, d[2] * inv_len];
    let coef = k * (len - rest);
    grad[0] += n[0] * coef;
    grad[1] += n[1] * coef;
    grad[2] += n[2] * coef;
    let tangential = (1.0 - rest / len).max(0.0);
    // 先加 k·tangential·I（仅对角），再加 k·(1-tangential)·nnᵀ（全 9 元）。
    let diag = k * tangential;
    hess[0] += diag;
    hess[4] += diag;
    hess[8] += diag;
    let s = k * (1.0 - tangential);
    hess[0] += s * n[0] * n[0];
    hess[1] += s * n[0] * n[1];
    hess[2] += s * n[0] * n[2];
    hess[3] += s * n[1] * n[0];
    hess[4] += s * n[1] * n[1];
    hess[5] += s * n[1] * n[2];
    hess[6] += s * n[2] * n[0];
    hess[7] += s * n[2] * n[1];
    hess[8] += s * n[2] * n[2];
}

/// `Mat3::solve` 的逐位转写：显式 cofactor 求逆，`|det| < EPS_DET` 或非有限结果
/// 时返回 `None`（该 sweep 不移动顶点）。
fn wesl_mat3_solve(m: &[f32; 9], rhs: [f32; 3]) -> Option<[f32; 3]> {
    let c00 = m[4] * m[8] - m[5] * m[7];
    let c01 = m[5] * m[6] - m[3] * m[8];
    let c02 = m[3] * m[7] - m[4] * m[6];
    let det = m[0] * c00 + m[1] * c01 + m[2] * c02;
    if det.abs() < EPS_DET {
        return None;
    }
    let inv_det = 1.0 / det;
    let c10 = m[2] * m[7] - m[1] * m[8];
    let c11 = m[0] * m[8] - m[2] * m[6];
    let c12 = m[1] * m[6] - m[0] * m[7];
    let c20 = m[1] * m[5] - m[2] * m[4];
    let c21 = m[2] * m[3] - m[0] * m[5];
    let c22 = m[0] * m[4] - m[1] * m[3];
    let x = (c00 * rhs[0] + c10 * rhs[1] + c20 * rhs[2]) * inv_det;
    let y = (c01 * rhs[0] + c11 * rhs[1] + c21 * rhs[2]) * inv_det;
    let z = (c02 * rhs[0] + c12 * rhs[1] + c22 * rhs[2]) * inv_det;
    if x.is_finite() && y.is_finite() && z.is_finite() {
        Some([x, y, z])
    } else {
        None
    }
}

/// 顶点 CSR 邻接：`out[i]` 按约束迭代序列出所有触及顶点 `i` 的约束 index（升序，
/// 即 GPU CSR 上传序）；`a == b` 或越界端点跳过，逐位复刻黄金 `build_adjacency`。
fn build_adjacency(constraints: &[Constraint], count: usize) -> Vec<Vec<u32>> {
    let mut adjacency: Vec<Vec<u32>> = vec![Vec::new(); count];
    for (index, constraint) in constraints.iter().enumerate() {
        let a = constraint.a as usize;
        let b = constraint.b as usize;
        if a == b || a >= count || b >= count {
            continue;
        }
        adjacency[a].push(index as u32);
        adjacency[b].push(index as u32);
    }
    adjacency
}

/// 对顶点 `i` 取一次精确 per-vertex Newton 步（原位），逐位复刻黄金 `relax_vertex`；
/// pinned 顶点原样不动。约束累加序遵循 `adjacency[i]`（升序约束 index）。
fn wesl_relax_vertex(
    i: usize,
    positions: &mut [[f32; 3]],
    inverse_mass: &[f32],
    constraints: &[Constraint],
    adjacency: &[Vec<u32>],
    target: [f32; 3],
    dt_sub_sq: f32,
) {
    if inverse_mass[i] <= 0.0 {
        return;
    }
    let x = positions[i];
    let mass = 1.0 / inverse_mass[i];
    let inertia = mass / dt_sub_sq;

    let mut grad = [
        (x[0] - target[0]) * inertia,
        (x[1] - target[1]) * inertia,
        (x[2] - target[2]) * inertia,
    ];
    let mut hess = [0.0f32; 9];
    hess[0] = inertia;
    hess[4] = inertia;
    hess[8] = inertia;

    for &c_index in &adjacency[i] {
        let constraint = constraints[c_index as usize];
        let other_index = if constraint.a as usize == i {
            constraint.b as usize
        } else {
            constraint.a as usize
        };
        let k = wesl_stiffness(constraint.compliance.value(), dt_sub_sq);
        wesl_accumulate(
            &mut grad,
            &mut hess,
            x,
            positions[other_index],
            constraint.rest_length,
            k,
            constraint.kind.is_one_sided(),
        );
    }

    if let Some(delta) = wesl_mat3_solve(&hess, grad) {
        positions[i] = [x[0] - delta[0], x[1] - delta[1], x[2] - delta[2]];
    }
}

/// 完整 color-major VBD 调度的逐位转写，返回最终 `(positions, velocities)`，逐位复刻
/// 黄金 `solve_cloth_vbd_colored`：空数组 / `dt <= 0` / 非有限 `dt` 直接 no-op；否则
/// 每 substep：predict 全部 → 每 iteration 遍历 `coloring.order()` relax → recover。
fn wesl_solve_colored(
    particles: &[ClothParticle],
    constraints: &[Constraint],
    params: VbdParams,
    dt: f32,
    order: &[u32],
) -> (Vec<[f32; 3]>, Vec<[f32; 3]>) {
    let count = particles.len();
    let mut positions: Vec<[f32; 3]> = particles
        .iter()
        .map(|p| [p.position.x, p.position.y, p.position.z])
        .collect();
    let mut velocities: Vec<[f32; 3]> = particles
        .iter()
        .map(|p| [p.velocity.x, p.velocity.y, p.velocity.z])
        .collect();
    let inverse_mass: Vec<f32> = particles.iter().map(|p| p.inverse_mass).collect();

    if count == 0 || dt <= 0.0 || !dt.is_finite() {
        return (positions, velocities);
    }

    let params = params.sanitized();
    let dt_sub = dt / params.substeps as f32;
    let dt_sub_sq = dt_sub * dt_sub;
    let inv_dt_sub = 1.0 / dt_sub;
    let retain = 1.0 - params.damping;
    let gravity_step = [
        params.gravity.x * dt_sub_sq,
        params.gravity.y * dt_sub_sq,
        params.gravity.z * dt_sub_sq,
    ];

    let adjacency = build_adjacency(constraints, count);
    let mut targets: Vec<[f32; 3]> = vec![[0.0; 3]; count];
    let mut previous: Vec<[f32; 3]> = vec![[0.0; 3]; count];

    for _ in 0..params.substeps {
        for i in 0..count {
            previous[i] = positions[i];
            targets[i] = wesl_predict(positions[i], velocities[i], retain, dt_sub, gravity_step);
        }

        for _ in 0..params.iterations {
            for &v in order {
                let i = v as usize;
                if i < count {
                    wesl_relax_vertex(
                        i,
                        &mut positions,
                        &inverse_mass,
                        constraints,
                        &adjacency,
                        targets[i],
                        dt_sub_sq,
                    );
                }
            }
        }

        for i in 0..count {
            if inverse_mass[i] <= 0.0 {
                velocities[i] = [0.0, 0.0, 0.0];
            } else {
                velocities[i] = [
                    (positions[i][0] - previous[i][0]) * inv_dt_sub,
                    (positions[i][1] - previous[i][1]) * inv_dt_sub,
                    (positions[i][2] - previous[i][2]) * inv_dt_sub,
                ];
            }
        }
    }

    (positions, velocities)
}

/// 把转写结果与黄金 `solve_cloth_vbd_colored` 的 position + velocity 逐位
/// (`to_bits`) 比对。coloring 由黄金 `color_cloth_vertices` 生成，两路同序。
#[track_caller]
fn assert_vbd_bit_exact(
    particles: &[ClothParticle],
    constraints: &[Constraint],
    params: VbdParams,
    dt: f32,
) {
    let count = particles.len();
    let coloring = color_cloth_vertices(constraints, count);
    let (pos, vel) = wesl_solve_colored(particles, constraints, params, dt, coloring.order());

    let mut golden = particles.to_vec();
    solve_cloth_vbd_colored(&mut golden, constraints, params, dt, &coloring);

    for (i, g) in golden.iter().enumerate() {
        assert_eq!(
            pos[i][0].to_bits(),
            g.position.x.to_bits(),
            "particle {i} position.x bits differ (wesl {} vs golden {})",
            pos[i][0],
            g.position.x
        );
        assert_eq!(
            pos[i][1].to_bits(),
            g.position.y.to_bits(),
            "particle {i} position.y bits differ (wesl {} vs golden {})",
            pos[i][1],
            g.position.y
        );
        assert_eq!(
            pos[i][2].to_bits(),
            g.position.z.to_bits(),
            "particle {i} position.z bits differ (wesl {} vs golden {})",
            pos[i][2],
            g.position.z
        );
        assert_eq!(
            vel[i][0].to_bits(),
            g.velocity.x.to_bits(),
            "particle {i} velocity.x bits differ (wesl {} vs golden {})",
            vel[i][0],
            g.velocity.x
        );
        assert_eq!(
            vel[i][1].to_bits(),
            g.velocity.y.to_bits(),
            "particle {i} velocity.y bits differ (wesl {} vs golden {})",
            vel[i][1],
            g.velocity.y
        );
        assert_eq!(
            vel[i][2].to_bits(),
            g.velocity.z.to_bits(),
            "particle {i} velocity.z bits differ (wesl {} vs golden {})",
            vel[i][2],
            g.velocity.z
        );
    }
}

/// 自由粒子（单位 inverse mass）。
fn free_particle(position: Vec3) -> ClothParticle {
    ClothParticle::new(position, 1.0)
}

/// 双向拉伸边。
fn stretch(a: u32, b: u32, rest: f32, compliance: f32) -> Constraint {
    Constraint::new(a, b, rest, Compliance(compliance), ConstraintKind::Stretch)
}

/// 指定类别的边（用于 one-sided LRA / tether）。
fn edge(a: u32, b: u32, rest: f32, compliance: f32, kind: ConstraintKind) -> Constraint {
    Constraint::new(a, b, rest, Compliance(compliance), kind)
}

/// 常用参数构造。
fn params(substeps: u32, iterations: u32, damping: f32) -> VbdParams {
    VbdParams {
        substeps,
        iterations,
        gravity: Vec3::new(0.0, -9.81, 0.0),
        damping,
    }
}

/// 轻量确定性 PRNG（线性同余，避免 `.sin()/.cos()`）→ `[-spread, spread]`。
fn lcg_next(state: &mut u32) -> f32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    // 取高 24 位映射到 [0,1)，再平移到 [-0.5, 0.5)。
    let unit = (*state >> 8) as f32 / (1u32 << 24) as f32;
    unit - 0.5
}

#[test]
fn single_stretch_edge_matches_golden() {
    // 一端 pin、一端自由、初始被拉伸的单边。
    let particles = vec![
        ClothParticle::pinned(Vec3::new(0.0, 0.0, 0.0)),
        free_particle(Vec3::new(1.5, 0.0, 0.0)),
    ];
    let constraints = vec![stretch(0, 1, 1.0, 1.0e-5)];
    assert_vbd_bit_exact(&particles, &constraints, params(4, 6, 0.02), 1.0 / 60.0);
}

#[test]
fn compressed_edge_drops_indefinite_part_like_golden() {
    // len < rest（压缩）：tangential 被 max(_,0) 夹为 0，Hessian 退化为 k·nnᵀ+惯性。
    let particles = vec![
        ClothParticle::pinned(Vec3::new(0.0, 0.0, 0.0)),
        free_particle(Vec3::new(0.4, 0.0, 0.0)),
    ];
    let constraints = vec![stretch(0, 1, 1.0, 1.0e-4)];
    assert_vbd_bit_exact(&particles, &constraints, params(3, 5, 0.05), 1.0 / 90.0);
}

#[test]
fn rigid_zero_compliance_saturates_max_stiffness() {
    // compliance == 0 → k = MAX_STIFFNESS（1e9），刚性边主导惯性。
    let particles = vec![
        ClothParticle::pinned(Vec3::new(0.0, 0.0, 0.0)),
        free_particle(Vec3::new(1.3, 0.1, 0.0)),
    ];
    let constraints = vec![stretch(0, 1, 1.0, 0.0)];
    assert_vbd_bit_exact(&particles, &constraints, params(2, 4, 0.0), 1.0 / 60.0);
}

#[test]
fn chain_of_free_particles_matches_golden() {
    // 顶端 pin 的 5 节链条，纯重力下垂，多约束/多顶点累加序闭环。
    let mut particles = vec![ClothParticle::pinned(Vec3::new(0.0, 0.0, 0.0))];
    for i in 1..5u32 {
        particles.push(free_particle(Vec3::new(i as f32, 0.0, 0.0)));
    }
    let mut constraints = Vec::new();
    for i in 0..4u32 {
        constraints.push(stretch(i, i + 1, 1.0, 2.0e-5));
    }
    assert_vbd_bit_exact(&particles, &constraints, params(4, 8, 0.03), 1.0 / 120.0);
}

#[test]
fn one_sided_lra_slack_gate_matches_golden() {
    // LRA 到 pinned anchor：松弛(len<=rest)不贡献；此处 len>rest 应收紧。
    let particles = vec![
        ClothParticle::pinned(Vec3::new(0.0, 0.0, 0.0)),
        free_particle(Vec3::new(0.0, -2.0, 0.0)),
    ];
    // rest=1.0，当前距离 2.0 > rest → 过伸，LRA 生效。
    let constraints = vec![edge(0, 1, 1.0, 1.0e-6, ConstraintKind::Lra)];
    assert_vbd_bit_exact(&particles, &constraints, params(3, 6, 0.01), 1.0 / 60.0);
}

#[test]
fn one_sided_tether_slack_is_noop_like_golden() {
    // Tether 松弛（len=0.5 <= rest=1.0）→ 不贡献，顶点仅受惯性+重力。
    let particles = vec![
        ClothParticle::pinned(Vec3::new(0.0, 0.0, 0.0)),
        free_particle(Vec3::new(0.0, -0.5, 0.0)),
    ];
    let constraints = vec![edge(0, 1, 1.0, 1.0e-6, ConstraintKind::Tether)];
    assert_vbd_bit_exact(&particles, &constraints, params(2, 4, 0.0), 1.0 / 60.0);
}

#[test]
fn all_pinned_is_frozen_like_golden() {
    // 全 pin：relax 跳过、recover 冻结为零速度。
    let particles = vec![
        ClothParticle::pinned(Vec3::new(0.0, 0.0, 0.0)),
        ClothParticle::pinned(Vec3::new(1.0, 0.0, 0.0)),
    ];
    let constraints = vec![stretch(0, 1, 1.0, 1.0e-5)];
    assert_vbd_bit_exact(&particles, &constraints, params(3, 3, 0.1), 1.0 / 60.0);
}

#[test]
fn non_positive_dt_is_a_noop_on_both_paths() {
    let particles = vec![
        ClothParticle::pinned(Vec3::new(0.0, 0.0, 0.0)),
        free_particle(Vec3::new(1.5, 0.0, 0.0)),
    ];
    let constraints = vec![stretch(0, 1, 1.0, 1.0e-5)];
    assert_vbd_bit_exact(&particles, &constraints, params(4, 4, 0.02), 0.0);
    assert_vbd_bit_exact(&particles, &constraints, params(4, 4, 0.02), -0.016);
}

#[test]
fn empty_mesh_is_a_noop_on_both_paths() {
    let particles: Vec<ClothParticle> = Vec::new();
    let constraints: Vec<Constraint> = Vec::new();
    assert_vbd_bit_exact(&particles, &constraints, params(4, 4, 0.02), 1.0 / 60.0);
}

#[test]
fn multi_substep_iteration_matches_golden() {
    // 高 substep × iteration：验证 recover→predict 跨 substep 的速度链逐位闭环。
    let mut particles = vec![ClothParticle::pinned(Vec3::new(0.0, 0.0, 0.0))];
    for i in 1..4u32 {
        particles.push(free_particle(Vec3::new(i as f32 * 0.9, -0.2, 0.1)));
    }
    let constraints = vec![
        stretch(0, 1, 1.0, 1.0e-5),
        stretch(1, 2, 1.0, 1.0e-5),
        stretch(2, 3, 1.0, 1.0e-5),
    ];
    assert_vbd_bit_exact(&particles, &constraints, params(8, 8, 0.02), 1.0 / 60.0);
}

#[test]
fn pseudo_random_grid_sweep_matches_golden() {
    // 4x4 悬挂网格 + 结构边，顶点位置用 LCG 扰动，驱动多色多约束的综合闭环。
    let rows = 4u32;
    let cols = 4u32;
    let mut state = 0x1234_5678u32;
    let mut particles = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            let base = Vec3::new(c as f32, -(r as f32), 0.0);
            let jitter = Vec3::new(
                lcg_next(&mut state) * 0.3,
                lcg_next(&mut state) * 0.3,
                lcg_next(&mut state) * 0.3,
            );
            let pos = Vec3::new(base.x + jitter.x, base.y + jitter.y, base.z + jitter.z);
            if r == 0 {
                particles.push(ClothParticle::pinned(pos));
            } else {
                let mut p = free_particle(pos);
                p.velocity = Vec3::new(
                    lcg_next(&mut state) * 0.5,
                    lcg_next(&mut state) * 0.5,
                    lcg_next(&mut state) * 0.5,
                );
                particles.push(p);
            }
        }
    }
    let idx = |r: u32, c: u32| r * cols + c;
    let mut constraints = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            if c + 1 < cols {
                constraints.push(stretch(idx(r, c), idx(r, c + 1), 1.0, 3.0e-5));
            }
            if r + 1 < rows {
                constraints.push(stretch(idx(r, c), idx(r + 1, c), 1.0, 3.0e-5));
            }
        }
    }
    assert_vbd_bit_exact(&particles, &constraints, params(5, 6, 0.04), 1.0 / 60.0);
}

#[test]
fn mixed_kinds_and_degenerate_edges_match_golden() {
    // 混合 stretch/LRA/tether + 一条退化自约束(a==b 被跳过) + 一条越界端点(被跳过)。
    let particles = vec![
        ClothParticle::pinned(Vec3::new(0.0, 0.0, 0.0)),
        free_particle(Vec3::new(1.2, -0.3, 0.0)),
        free_particle(Vec3::new(2.1, -0.1, 0.2)),
    ];
    let constraints = vec![
        stretch(0, 1, 1.0, 1.0e-5),
        edge(0, 1, 1.0, 1.0e-6, ConstraintKind::Lra),
        stretch(1, 2, 1.0, 2.0e-5),
        edge(0, 2, 1.5, 1.0e-6, ConstraintKind::Tether),
        stretch(1, 1, 1.0, 1.0e-5), // 退化自约束：两路都跳过
        stretch(2, 9, 1.0, 1.0e-5), // 越界端点：两路都跳过
    ];
    assert_vbd_bit_exact(&particles, &constraints, params(4, 5, 0.02), 1.0 / 60.0);
}
