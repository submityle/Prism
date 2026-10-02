//! `cloth_sim.wesl` 的四个投影/回收内核的**逐位** CPU parity，对齐架构层黄金
//! 整帧求解器 [`solve_cloth_with_collision`]。
//!
//! `shader_tests` 只做 naga 编译门禁，证明 `cloth_sim.wesl` 解析通过，却
//! **证明不了**其 `cloth_project_distance_batch` / `cloth_project_long_range_batch`
//! / `cloth_strain_limit` / `cloth_velocity_update` 逐位复刻了 CPU 黄金的算术
//! （设计 §9：不造假 parity）。姊妹模块 `sim_predict_parity` 已覆盖 `cloth_predict`、
//! `sim_bending_parity` 已直接覆盖 pub `project_bending`（黄金 `solve_cloth` 的
//! 距离图里并不含二面角弯曲，弯曲是独立 pub 路径），本模块补齐剩下四个内核：
//!
//! * `cloth_project_distance_batch` — 两侧（stretch / shear / bend）距离投影；
//! * `cloth_project_long_range_batch` — 单侧（LRA / tether）长程投影，仅过伸拉回；
//! * `cloth_strain_limit` — 结构（stretch）边过伸硬夹；
//! * `cloth_velocity_update` — 由位置增量回收速度（pinned 清零）。
//!
//! 做法与 `sim_predict_parity` 同款：把每个内核的逐条算术**独立**转写成原生
//! `[f32; 3]` 数组版本（不复用黄金 `Vec3` 方法），再按黄金 host 调度顺序
//! （substep × iteration × 每 color batch）驱动，对多种夹具断言其最终**位置与
//! 速度**都与黄金 [`solve_cloth_with_collision`]（no-op 碰撞钩子）**逐位
//! （`to_bits`）一致**。
//!
//! ## 为何能逐位（已逐行核对 `cloth_sim.wesl` ↔ 黄金 `dynamics.rs`）
//! 黄金 `solve_cloth_with_collision` 每 substep 依次做：① predict ② 对每个
//! color batch 的每条约束调 `project_distance`（内部按 `is_one_sided` 分流）
//! ③ `apply_strain_limit`（仅 stretch）④ 碰撞钩子（本测用 no-op `|p| p.position`）
//! ⑤ 速度回收。WESL 把步骤 ② 拆成 `distance` / `long_range` 两个 dispatch（host 按
//! 单侧性把约束放进对应 dispatch），步骤 ③/⑤ 各一个 dispatch。逐内核算术：
//! * **距离/长程**：现已统一委托物理引擎单一真源
//!   [`prism_physics_core::soft::constraint::project_distance_constraint`]：
//!   `d_lambda = -error / denom`（**直接除**，两侧同序）、分离方向
//!   `direction = delta / dist`（**逐分量除**，与物理 `normal = delta / length`
//!   及更新后的 `cloth_sim.wesl` 距离/长程内核逐位同序；黄金 `project_distance`
//!   亦已改为委托，三侧共用这一步算术）、`alpha_tilde =
//!   max(compliance, 0.0) / (dt_sub * dt_sub)`（黄金 `Compliance::value()` 本身即
//!   `self.0.max(0.0)`，与物理自由函数的 raw compliance、WESL `max(.,0.0)` clamp
//!   口径一致）。单侧内核多一道 `error <= 0.0` 早退门，恰为黄金 `project_distance`
//!   里 `is_one_sided() && error <= 0.0` 的同序分支（该门仍留在 render 侧，门后
//!   才调委托）。
//! * **应变限制**：现已统一委托物理引擎单一真源
//!   [`prism_physics_core::soft::constraint::project_strain_limit`]
//!   （`min_scale = 0`，最大拉伸硬夹）：分离方向 `direction = delta / dist`
//!   （**逐分量除**，与物理 `direction = delta / length` 及更新后的
//!   `cloth_sim.wesl` 应变内核逐位同序）、`correction = direction * excess`、
//!   `pos_a -= correction * (wa / w_sum)`、`pos_b += correction * (wb / w_sum)`，
//!   与黄金 `apply_strain_limit`（亦已改为委托）完全同序。黄金侧保留
//!   `dist_sq <= max_sq || dist_sq <= EPS_LEN_SQ` 的 band / 退化门，门后才调委托，
//!   故物理自由函数内更紧的 `length < f32::EPSILON` 退化门永不改变已过门边的结果。
//! * **速度回收**：`delta * (1.0 / dt_sub)`（倒数乘），与黄金
//!   `position.sub(prev).scale(1.0 / dt_sub)` 同序；pinned 清零亦一致。
//!   （`cloth_sim.wesl` 现已把回收从早期的逐分量除法改为倒数乘，故速度现亦可
//!   逐位——这纠正了 `sim_predict_parity` 早期「只比位置不比速度」的过时边界。）
//!
//! ## 诚实边界：应变限制的夹具保持粒子互斥
//! 黄金 `apply_strain_limit` 按 color batch 顺序串行扫描（Gauss-Seidel），而
//! WESL `cloth_strain_limit` 以单次 dispatch 扫过整条约束缓冲（gid 覆盖
//! `constraint_count`，color 内 Jacobi）。只要被限制器触及的结构边**两两不共享
//! 粒子**，串行与并行结果就逐位一致（既对齐黄金，也对齐真机单 dispatch）。本模块
//! 的应变夹具据此保持结构边粒子互斥；跨 batch 共享粒子的 Gauss-Seidel 次序敏感
//! 情形不在本 CPU parity 承诺内（另由 `sim_gpu_tests` 带语义覆盖）。
//! 距离/长程投影在 color batch 内天然粒子互斥（图着色保证），故 Jacobi 等于
//! 串行，真机与黄金一致，无此约束。
//!
//! ## 索引定义域
//! WESL 投影内核只有 `a == b` 自约束守卫，越界靠 host 保证合法；黄金另有
//! `a >= len || b >= len` 越界跳过。本模块夹具的约束索引**恒在 `particles`
//! 范围内**，故两侧越界分支永不触发差异；`a == b` 自约束两侧同样跳过，一并覆盖。

#![cfg(test)]

use prism_render_architecture::cloth::dynamics::{solve_cloth_with_collision, SolverParams};
use prism_render_architecture::cloth::{
    ClothParticle, ColorBatch, Compliance, Constraint, ConstraintGraph, ConstraintKind, Vec3,
};

/// 与黄金 `EPS_LEN_SQ` / WESL `CLOTH_EPS_LEN_SQ` 对齐的平方长度地板。
const EPS_LEN_SQ: f32 = 1.0e-12;

/// 原生粒子状态：`pos` 世界位置、`vel` 速度、`w` 逆质量（`<= 0` 为 pinned）。
#[derive(Clone, Copy)]
struct WeslParticle {
    pos: [f32; 3],
    vel: [f32; 3],
    w: f32,
}

/// WESL `cloth_inverse_mass`：`<= 0`（pinned）读作 0。
fn cloth_inverse_mass(w: f32) -> f32 {
    w.max(0.0)
}

/// `cloth_predict` 的逐位转写：pinned（`w <= 0`）位置/速度不变；否则
/// `vel = vel * retain + gravity * dt_sub`，再 `pos = pos + vel * dt_sub`。
fn wesl_predict(p: &mut WeslParticle, gravity: [f32; 3], retain: f32, dt_sub: f32) {
    if p.w <= 0.0 {
        return;
    }
    let v = [
        p.vel[0] * retain + gravity[0] * dt_sub,
        p.vel[1] * retain + gravity[1] * dt_sub,
        p.vel[2] * retain + gravity[2] * dt_sub,
    ];
    p.vel = v;
    p.pos = [
        p.pos[0] + v[0] * dt_sub,
        p.pos[1] + v[1] * dt_sub,
        p.pos[2] + v[2] * dt_sub,
    ];
}

/// `cloth_project_distance_batch`（两侧）与 `cloth_project_long_range_batch`
/// （单侧）共用的核心：`one_sided` 为真时加一道 `error <= 0.0` 早退门。
fn wesl_project(
    particles: &mut [WeslParticle],
    a: usize,
    b: usize,
    rest_length: f32,
    compliance: f32,
    dt_sub: f32,
    one_sided: bool,
) {
    if a == b {
        return;
    }
    let wa = cloth_inverse_mass(particles[a].w);
    let wb = cloth_inverse_mass(particles[b].w);
    let w_sum = wa + wb;
    if w_sum <= 0.0 {
        return;
    }
    let delta = [
        particles[a].pos[0] - particles[b].pos[0],
        particles[a].pos[1] - particles[b].pos[1],
        particles[a].pos[2] - particles[b].pos[2],
    ];
    let dist_sq = delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2];
    if dist_sq <= EPS_LEN_SQ {
        return;
    }
    let dist = dist_sq.sqrt();
    let error = dist - rest_length;
    if one_sided && error <= 0.0 {
        return;
    }
    let alpha_tilde = compliance.max(0.0) / (dt_sub * dt_sub);
    let denom = w_sum + alpha_tilde;
    if denom <= 0.0 {
        return;
    }
    let d_lambda = -error / denom;
    // Separation direction is a component-wise divide (`delta / dist`), matching
    // the physics `project_distance_constraint` single source and the updated
    // `cloth_sim.wesl` distance / long-range kernels.
    let direction = [delta[0] / dist, delta[1] / dist, delta[2] / dist];
    let correction = [
        direction[0] * d_lambda,
        direction[1] * d_lambda,
        direction[2] * d_lambda,
    ];
    particles[a].pos = [
        particles[a].pos[0] + correction[0] * wa,
        particles[a].pos[1] + correction[1] * wa,
        particles[a].pos[2] + correction[2] * wa,
    ];
    particles[b].pos = [
        particles[b].pos[0] - correction[0] * wb,
        particles[b].pos[1] - correction[1] * wb,
        particles[b].pos[2] - correction[2] * wb,
    ];
}

/// `cloth_strain_limit` 的逐位转写：仅 `Stretch`，过伸硬夹到 `rest * max_scale`。
fn wesl_strain_limit(
    particles: &mut [WeslParticle],
    a: usize,
    b: usize,
    rest_length: f32,
    is_stretch: bool,
    max_scale: f32,
) {
    if a == b {
        return;
    }
    if !is_stretch {
        return;
    }
    let wa = cloth_inverse_mass(particles[a].w);
    let wb = cloth_inverse_mass(particles[b].w);
    let w_sum = wa + wb;
    if w_sum <= 0.0 {
        return;
    }
    let max_len = rest_length * max_scale;
    let delta = [
        particles[a].pos[0] - particles[b].pos[0],
        particles[a].pos[1] - particles[b].pos[1],
        particles[a].pos[2] - particles[b].pos[2],
    ];
    let dist_sq = delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2];
    let max_sq = max_len * max_len;
    if dist_sq <= max_sq || dist_sq <= EPS_LEN_SQ {
        return;
    }
    let dist = dist_sq.sqrt();
    let excess = dist - max_len;
    // Separation direction is a component-wise divide (`delta / dist`),
    // matching the physics `project_strain_limit` single source and the
    // updated `cloth_sim.wesl` strain-limit kernel.
    let direction = [delta[0] / dist, delta[1] / dist, delta[2] / dist];
    let correction = [
        direction[0] * excess,
        direction[1] * excess,
        direction[2] * excess,
    ];
    let ka = wa / w_sum;
    let kb = wb / w_sum;
    particles[a].pos = [
        particles[a].pos[0] - correction[0] * ka,
        particles[a].pos[1] - correction[1] * ka,
        particles[a].pos[2] - correction[2] * ka,
    ];
    particles[b].pos = [
        particles[b].pos[0] + correction[0] * kb,
        particles[b].pos[1] + correction[1] * kb,
        particles[b].pos[2] + correction[2] * kb,
    ];
}

/// `cloth_velocity_update` 的逐位转写：pinned（`w <= 0`）清零；否则
/// `vel = (pos - prev) * (1.0 / dt_sub)`。
fn wesl_velocity_update(p: &mut WeslParticle, prev: [f32; 3], dt_sub: f32) {
    if p.w <= 0.0 {
        p.vel = [0.0, 0.0, 0.0];
        return;
    }
    if dt_sub <= 0.0 {
        return;
    }
    let inv_dt = 1.0 / dt_sub;
    p.vel = [
        (p.pos[0] - prev[0]) * inv_dt,
        (p.pos[1] - prev[1]) * inv_dt,
        (p.pos[2] - prev[2]) * inv_dt,
    ];
}

/// 整帧 WESL 调度转写：严格复刻黄金 `solve_cloth_with_collision` 的 host 循环
/// （substep → predict → iteration × 每 batch 投影 → 应变限制 → no-op 碰撞 →
/// 速度回收），逐内核调用上面的原生转写。遍历 `graph.batches` / `graph.batch()`
/// 与黄金同序，故 color batch 的 Gauss-Seidel 次序逐位一致。
fn wesl_solve(
    particles: &mut [WeslParticle],
    graph: &ConstraintGraph,
    params: SolverParams,
    dt: f32,
) {
    if dt <= 0.0 || particles.is_empty() {
        return;
    }
    let substeps = params.substeps.max(1);
    let iterations = params.iterations.max(1);
    let dt_sub = dt / substeps as f32;
    let damping = params.damping.clamp(0.0, 1.0);
    let retain = 1.0 - damping;
    let gravity = [params.gravity.x, params.gravity.y, params.gravity.z];

    for _ in 0..substeps {
        let previous: Vec<[f32; 3]> = particles.iter().map(|p| p.pos).collect();

        for p in particles.iter_mut() {
            wesl_predict(p, gravity, retain, dt_sub);
        }

        for _ in 0..iterations {
            for batch in &graph.batches {
                for constraint in graph.batch(*batch) {
                    wesl_project(
                        particles,
                        constraint.a as usize,
                        constraint.b as usize,
                        constraint.rest_length,
                        constraint.compliance.value(),
                        dt_sub,
                        constraint.kind.is_one_sided(),
                    );
                }
            }
        }

        if params.strain_limit > 0.0 {
            let max_scale = 1.0 + params.strain_limit;
            for batch in &graph.batches {
                for constraint in graph.batch(*batch) {
                    wesl_strain_limit(
                        particles,
                        constraint.a as usize,
                        constraint.b as usize,
                        constraint.rest_length,
                        constraint.kind == ConstraintKind::Stretch,
                        max_scale,
                    );
                }
            }
        }

        // 步骤 ④ 碰撞钩子为 no-op（`|p| p.position`），位置恒等，略去。

        for (p, &prev) in particles.iter_mut().zip(previous.iter()) {
            wesl_velocity_update(p, prev, dt_sub);
        }
    }
}

/// 单条粒子输入：`(position, velocity, inverse_mass)`。
type ParticleInput = ([f32; 3], [f32; 3], f32);

/// 构造黄金粒子批：pinned 走 [`ClothParticle::pinned`]，自由粒子走
/// [`ClothParticle::new`] 后显式写入初速度。
fn build_golden(particles: &[ParticleInput]) -> Vec<ClothParticle> {
    particles
        .iter()
        .map(|&(p, v, inv_mass)| {
            let mut cp = if inv_mass <= 0.0 {
                ClothParticle::pinned(Vec3::new(p[0], p[1], p[2]))
            } else {
                ClothParticle::new(Vec3::new(p[0], p[1], p[2]), inv_mass)
            };
            cp.velocity = Vec3::new(v[0], v[1], v[2]);
            cp
        })
        .collect()
}

/// 构造原生 WESL 粒子批（与黄金同一输入）。
fn build_wesl(particles: &[ParticleInput]) -> Vec<WeslParticle> {
    particles
        .iter()
        .map(|&(pos, vel, w)| WeslParticle { pos, vel, w })
        .collect()
}

/// 把一串约束按「每条自成一个 color batch」铺成图（严格 Gauss-Seidel 串行）。
/// 用于需要稳定跨约束次序的夹具（如链条、跨批共享粒子的距离投影）。
fn graph_serial(constraints: &[Constraint]) -> ConstraintGraph {
    let batches = (0..constraints.len())
        .map(|i| ColorBatch {
            start: i as u32,
            len: 1,
        })
        .collect();
    ConstraintGraph {
        constraints: constraints.to_vec(),
        batches,
    }
}

/// 把一串**粒子互斥**的约束铺成单个 color batch（color 内 Jacobi 安全）。
fn graph_single_batch(constraints: &[Constraint]) -> ConstraintGraph {
    ConstraintGraph {
        constraints: constraints.to_vec(),
        batches: vec![ColorBatch {
            start: 0,
            len: constraints.len() as u32,
        }],
    }
}

/// 跑黄金整帧求解与 WESL 调度转写，逐分量断言**位置与速度** `to_bits` 一致。
#[track_caller]
fn assert_solve_bit_exact(
    particles: &[ParticleInput],
    graph: &ConstraintGraph,
    params: SolverParams,
    dt: f32,
) {
    let mut golden = build_golden(particles);
    let mut wesl = build_wesl(particles);

    solve_cloth_with_collision(&mut golden, graph, params, dt, |p| p.position);
    wesl_solve(&mut wesl, graph, params, dt);

    assert_eq!(golden.len(), wesl.len(), "粒子数一致");
    for (i, (g, w)) in golden.iter().zip(wesl.iter()).enumerate() {
        let gp = [g.position.x, g.position.y, g.position.z];
        let gv = [g.velocity.x, g.velocity.y, g.velocity.z];
        for axis in 0..3 {
            assert_eq!(
                gp[axis].to_bits(),
                w.pos[axis].to_bits(),
                "粒子 {i} 位置分量 {axis} 不逐位一致：黄金 {} vs WESL {}",
                gp[axis],
                w.pos[axis],
            );
            assert_eq!(
                gv[axis].to_bits(),
                w.vel[axis].to_bits(),
                "粒子 {i} 速度分量 {axis} 不逐位一致：黄金 {} vs WESL {}",
                gv[axis],
                w.vel[axis],
            );
        }
    }
}

/// 关闭应变限制的默认参数（单 substep/iteration，便于隔离各内核）。
fn params_no_strain(substeps: u32, iterations: u32, damping: f32) -> SolverParams {
    SolverParams {
        substeps,
        iterations,
        gravity: Vec3::new(0.0, -9.81, 0.0),
        damping,
        strain_limit: 0.0,
    }
}

#[test]
fn rigid_stretch_under_gravity_matches_golden() {
    // 一端 pinned、一端自由，刚性 stretch 链；predict 下坠后距离投影拉回。
    let particles = [
        ([0.0f32, 0.0, 0.0], [0.0f32, 0.0, 0.0], 0.0),
        ([0.0, -1.2, 0.0], [0.3, 0.0, -0.1], 1.0),
    ];
    let graph = graph_single_batch(&[Constraint::new(
        0,
        1,
        1.0,
        Compliance::RIGID,
        ConstraintKind::Stretch,
    )]);
    assert_solve_bit_exact(&particles, &graph, params_no_strain(1, 1, 0.02), 1.0 / 60.0);
}

#[test]
fn compliant_stretch_matches_golden() {
    // 软 compliance：alpha_tilde 非零，denom 更大，d_lambda 更小。
    let particles = [
        ([0.0f32, 0.0, 0.0], [0.0f32, 0.0, 0.0], 0.0),
        ([1.4, 0.1, 0.0], [0.0, -0.2, 0.05], 1.0),
    ];
    let graph = graph_single_batch(&[Constraint::new(
        0,
        1,
        1.0,
        Compliance(0.005),
        ConstraintKind::Stretch,
    )]);
    assert_solve_bit_exact(&particles, &graph, params_no_strain(1, 1, 0.0), 1.0 / 90.0);
}

#[test]
fn shear_and_bend_are_two_sided_like_golden() {
    // Shear 与 Bend 均两侧（is_one_sided == false），走 distance 内核；两条约束
    // 粒子互斥，单 batch Jacobi 安全。
    let particles = [
        ([0.0f32, 0.0, 0.0], [0.1, 0.0, 0.0], 1.0),
        ([1.3, -0.2, 0.0], [0.0, 0.0, 0.2], 1.0),
        ([0.0, 0.0, 1.0], [0.0, 0.1, 0.0], 1.0),
        ([1.1, 0.0, 1.2], [-0.1, 0.0, 0.0], 1.0),
    ];
    let graph = graph_single_batch(&[
        Constraint::new(0, 1, 1.0, Compliance(0.001), ConstraintKind::Shear),
        Constraint::new(2, 3, 1.0, Compliance(0.01), ConstraintKind::Bend),
    ]);
    assert_solve_bit_exact(&particles, &graph, params_no_strain(1, 1, 0.01), 1.0 / 60.0);
}

#[test]
fn long_range_pulls_only_when_overextended() {
    // LRA：过伸（dist > rest）→ 投影拉回。单侧门 error > 0 命中。
    let particles = [
        ([0.0f32, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0),
        ([2.5, 0.0, 0.0], [0.1, -0.3, 0.0], 1.0),
    ];
    let graph = graph_single_batch(&[Constraint::new(
        0,
        1,
        1.0,
        Compliance::RIGID,
        ConstraintKind::Lra,
    )]);
    assert_solve_bit_exact(&particles, &graph, params_no_strain(1, 1, 0.0), 1.0 / 60.0);
}

#[test]
fn long_range_is_noop_when_slack() {
    // LRA：松弛（dist < rest）→ error <= 0 早退，投影 no-op，仅剩 predict+回收。
    let particles = [
        ([0.0f32, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0),
        ([0.3, -0.1, 0.0], [0.2, 0.0, 0.1], 1.0),
    ];
    let graph = graph_single_batch(&[Constraint::new(
        0,
        1,
        2.0,
        Compliance::RIGID,
        ConstraintKind::Lra,
    )]);
    assert_solve_bit_exact(&particles, &graph, params_no_strain(1, 1, 0.0), 1.0 / 60.0);
}

#[test]
fn tether_overextended_matches_golden() {
    // Tether 亦单侧；过伸拉回。
    let particles = [
        ([0.0f32, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0),
        ([0.0, -3.0, 0.4], [0.0, -0.5, 0.0], 1.0),
    ];
    let graph = graph_single_batch(&[Constraint::new(
        0,
        1,
        1.5,
        Compliance(0.002),
        ConstraintKind::Tether,
    )]);
    assert_solve_bit_exact(&particles, &graph, params_no_strain(1, 1, 0.0), 1.0 / 60.0);
}

#[test]
fn chain_across_color_batches_preserves_gauss_seidel_order() {
    // 三粒子链 0-1-2，两条 stretch 共享粒子 1，故放不同 batch 串行；验证跨 batch
    // 的 Gauss-Seidel 次序逐位一致。
    let particles = [
        ([0.0f32, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0),
        ([1.3, -0.2, 0.0], [0.1, -0.1, 0.0], 1.0),
        ([2.7, -0.3, 0.0], [0.0, -0.2, 0.05], 1.0),
    ];
    let graph = graph_serial(&[
        Constraint::new(0, 1, 1.0, Compliance::RIGID, ConstraintKind::Stretch),
        Constraint::new(1, 2, 1.0, Compliance::RIGID, ConstraintKind::Stretch),
    ]);
    assert_solve_bit_exact(&particles, &graph, params_no_strain(1, 2, 0.01), 1.0 / 60.0);
}

#[test]
fn strain_limit_clamps_overstretch() {
    // 过伸的 stretch 边：投影后仍超 1+limit，应变限制硬夹。单条互斥，Jacobi 安全。
    let particles = [
        ([0.0f32, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0),
        ([1.6, 0.0, 0.0], [0.4, -0.2, 0.0], 1.0),
    ];
    let graph = graph_single_batch(&[Constraint::new(
        0,
        1,
        1.0,
        Compliance(0.5),
        ConstraintKind::Stretch,
    )]);
    let params = SolverParams {
        substeps: 1,
        iterations: 1,
        gravity: Vec3::new(0.0, -9.81, 0.0),
        damping: 0.0,
        strain_limit: 0.1,
    };
    assert_solve_bit_exact(&particles, &graph, params, 1.0 / 60.0);
}

#[test]
fn strain_limit_noop_within_tolerance() {
    // 边长在 1+limit 内 → 应变限制 no-op。
    let particles = [
        ([0.0f32, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0),
        ([1.02, 0.0, 0.0], [0.0, -0.1, 0.0], 1.0),
    ];
    let graph = graph_single_batch(&[Constraint::new(
        0,
        1,
        1.0,
        Compliance::RIGID,
        ConstraintKind::Stretch,
    )]);
    let params = SolverParams {
        substeps: 1,
        iterations: 1,
        gravity: Vec3::new(0.0, 0.0, 0.0),
        damping: 0.0,
        strain_limit: 0.1,
    };
    assert_solve_bit_exact(&particles, &graph, params, 1.0 / 60.0);
}

#[test]
fn strain_limit_skips_non_stretch_kinds() {
    // 过伸的 Shear 边：应变限制只认 Stretch，Shear 不被夹（但仍走 distance 投影）。
    let particles = [
        ([0.0f32, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0),
        ([1.9, 0.0, 0.0], [0.0, -0.3, 0.0], 1.0),
    ];
    let graph = graph_single_batch(&[Constraint::new(
        0,
        1,
        1.0,
        Compliance(0.3),
        ConstraintKind::Shear,
    )]);
    let params = SolverParams {
        substeps: 1,
        iterations: 1,
        gravity: Vec3::new(0.0, -9.81, 0.0),
        damping: 0.0,
        strain_limit: 0.05,
    };
    assert_solve_bit_exact(&particles, &graph, params, 1.0 / 60.0);
}

#[test]
fn degenerate_constraints_are_skipped_both_sides() {
    // 自约束 a==b 跳过；重合粒子 dist_sq <= EPS 跳过；两约束粒子互斥。
    let particles = [
        ([0.0f32, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0),
        ([0.5, -0.4, 0.0], [0.1, 0.0, 0.0], 1.0),
        ([1.0, 1.0, 1.0], [0.0, -0.2, 0.0], 1.0),
        ([1.0, 1.0, 1.0], [0.0, -0.2, 0.0], 1.0),
    ];
    let graph = graph_single_batch(&[
        // 自约束：a == b，两侧跳过。
        Constraint::new(1, 1, 1.0, Compliance::RIGID, ConstraintKind::Stretch),
        // 重合粒子：dist_sq <= EPS，两侧跳过。
        Constraint::new(2, 3, 1.0, Compliance::RIGID, ConstraintKind::Stretch),
    ]);
    assert_solve_bit_exact(&particles, &graph, params_no_strain(1, 1, 0.0), 1.0 / 60.0);
}

#[test]
fn both_pinned_endpoints_skip_projection() {
    // 两端皆 pinned → w_sum <= 0 跳过投影；pinned 位置不动、速度清零。
    let particles = [
        ([0.0f32, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0),
        ([3.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0),
    ];
    let graph = graph_single_batch(&[Constraint::new(
        0,
        1,
        1.0,
        Compliance::RIGID,
        ConstraintKind::Stretch,
    )]);
    assert_solve_bit_exact(&particles, &graph, params_no_strain(1, 1, 0.0), 1.0 / 60.0);
}

#[test]
fn non_positive_dt_is_a_noop_on_both_paths() {
    // dt <= 0：黄金与转写均早退，位置/速度不变。
    let particles = [
        ([0.0f32, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0),
        ([1.5, 0.0, 0.0], [0.2, -0.3, 0.1], 1.0),
    ];
    let graph = graph_single_batch(&[Constraint::new(
        0,
        1,
        1.0,
        Compliance::RIGID,
        ConstraintKind::Stretch,
    )]);
    assert_solve_bit_exact(&particles, &graph, params_no_strain(1, 1, 0.0), 0.0);
    assert_solve_bit_exact(&particles, &graph, params_no_strain(1, 1, 0.0), -0.5);
}

#[test]
fn high_substep_iteration_pseudo_random_grid_matches_golden() {
    // 4×4 网格，LCG 伪随机初速度，多 substep × iteration，混合约束类别；验证整帧
    // 调度在重循环下仍逐位一致。结构 stretch 边按行/列铺色串行以稳住 Gauss-Seidel。
    let cols = 4usize;
    let rows = 4usize;
    let mut state: u32 = 0x1234_5678;
    let mut next = || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        // 取高位映射到 [-0.5, 0.5)。
        ((state >> 8) as f32 / (1u32 << 24) as f32) - 0.5
    };
    let mut particles: Vec<ParticleInput> = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            let pinned = r == 0 && (c == 0 || c == cols - 1);
            let pos = [c as f32, -(r as f32), next() * 0.1];
            let vel = if pinned {
                [0.0, 0.0, 0.0]
            } else {
                [next(), next(), next()]
            };
            let w = if pinned { 0.0 } else { 1.0 };
            particles.push((pos, vel, w));
        }
    }
    let idx = |r: usize, c: usize| (r * cols + c) as u32;
    // 结构横/纵边 → 各自成 batch（串行，稳住跨 batch 次序）；对角 shear、长程 LRA
    // 追加其后。所有边统一 1 单位静长，compliance 混合。
    let mut constraints = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            if c + 1 < cols {
                constraints.push(Constraint::new(
                    idx(r, c),
                    idx(r, c + 1),
                    1.0,
                    Compliance(0.001),
                    ConstraintKind::Stretch,
                ));
            }
            if r + 1 < rows {
                constraints.push(Constraint::new(
                    idx(r, c),
                    idx(r + 1, c),
                    1.0,
                    Compliance(0.001),
                    ConstraintKind::Stretch,
                ));
            }
            if r + 1 < rows && c + 1 < cols {
                constraints.push(Constraint::new(
                    idx(r, c),
                    idx(r + 1, c + 1),
                    1.4,
                    Compliance(0.02),
                    ConstraintKind::Shear,
                ));
            }
        }
    }
    // 自由角到对侧 pinned 锚的长程约束（过伸时拉回）。
    constraints.push(Constraint::new(
        idx(0, 0),
        idx(rows - 1, cols - 1),
        2.0,
        Compliance::RIGID,
        ConstraintKind::Lra,
    ));
    let graph = graph_serial(&constraints);
    let params = SolverParams {
        substeps: 4,
        iterations: 3,
        gravity: Vec3::new(0.0, -9.81, 0.3),
        damping: 0.03,
        strain_limit: 0.0,
    };
    assert_solve_bit_exact(&particles, &graph, params, 1.0 / 60.0);
}

#[test]
fn negative_compliance_clamps_to_zero_both_sides() {
    // 负 compliance：黄金 `Compliance::value()` 与 WESL `max(_, 0)` 都夹到 0。
    let particles = [
        ([0.0f32, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0),
        ([1.5, 0.0, 0.0], [0.0, -0.4, 0.2], 1.0),
    ];
    let graph = graph_single_batch(&[Constraint::new(
        0,
        1,
        1.0,
        Compliance(-7.0),
        ConstraintKind::Stretch,
    )]);
    assert_solve_bit_exact(&particles, &graph, params_no_strain(1, 1, 0.0), 1.0 / 60.0);
}
