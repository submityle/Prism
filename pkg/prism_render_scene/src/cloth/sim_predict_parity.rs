//! `cloth_predict` 内核的**逐位** CPU parity，对齐架构层黄金求解器
//! [`solve_cloth_with_collision`] 的位置积分。
//!
//! `shader_tests` 只做 naga 编译门禁，证明 `cloth_sim.wesl` 解析通过，却
//! **证明不了**其 `cloth_predict` 逐位复刻了 CPU 黄金的算术（设计 §9：不造假
//! parity）。本模块补上这条闭环：把 `cloth_predict` 的逐条算术独立转写成 CPU
//! 版本（不复用黄金 `Vec3` 方法，而是照 `WESL` 源码用原生数组重写），再用
//! **空约束图 + 隔离参数**把黄金 `solve_cloth_with_collision` 塌缩为纯 predict，
//! 对多种粒子批断言其输出位置与黄金**逐位（`to_bits`）一致**。
//!
//! ## 隔离法：把整帧求解塌缩为纯 predict
//! 黄金 `solve_cloth_with_collision` 每个 substep 依次做：1 预测、2 约束投影、
//! 3 应变限制、4 碰撞钩子、5 速度回收。用以下参数可让 2/3/4 全部变为 no-op，
//! 从而最终位置**完全由 predict 决定**：
//! * [`ConstraintGraph::default()`]：空 `constraints`/`batches`，步骤 2 不动位置。
//! * `SolverParams { substeps: 1, iterations: 1, strain_limit: 0.0, .. }`：
//!   单 substep（`dt_sub == dt / 1.0 == dt`，除 `1.0` 为 IEEE 精确）、
//!   `strain_limit <= 0.0` 关闭步骤 3。
//! * no-op 碰撞闭包 `|p| p.position`：步骤 4 把位置映射为自身（恒等），不动位置。
//! * 显式 `dt > 0.0`：黄金对 `dt <= 0.0` 直接早退（位置不变），本 harness 的
//!   转写无此守卫，故**只在 `dt > 0.0` 上做 parity**（`dt <= 0.0` 的 no-op 语义
//!   另由 `sim_gpu_tests` 覆盖）。
//!
//! ## 只比位置，不比速度（honest 边界）
//! 步骤 5 的速度回收，`WESL` `cloth_velocity_update` 用 `delta / dt_sub`
//! （逐分量**除法**），黄金用 `delta.scale(1.0 / dt_sub)`（先取**倒数**再乘）。
//! 除法与倒数乘在 IEEE 下**不逐位**（真实发现，类比 `body_collision` 摩擦段），
//! 故速度由 `sim_gpu_tests` 带容差覆盖，本 CPU parity **只认位置**。
//!
//! ## predict 算术逐位同构（已逐行核对 `WESL` ↔ 黄金 `dynamics.rs`）
//! * `retain = 1.0 - damping.clamp(0.0, 1.0)`（两侧同序）。
//! * `gravity_step` 预乘 `gravity * dt_sub`（分量 × 标量），与黄金
//!   `gravity.scale(dt_sub)` 同序。
//! * `new_vel = vel * retain + gravity_step`（`WESL` `vel*retain + gravity*dt_sub`
//!   与黄金 `vel.scale(retain).add(gravity_step)` 同分量同序）。
//! * `position = pos + new_vel * dt_sub`（`WESL` `pos + velocity*dt_sub` 与黄金
//!   `position.add(velocity.scale(dt_sub))` 同序）。
//! * pinned（`inverse_mass <= 0.0`）位置全程不变，两侧一致。
//!
//! `cloth_sim.wesl` 的其余内核（`cloth_project_distance_batch` /
//! `cloth_project_bending_batch` / `cloth_project_long_range_batch` /
//! `cloth_strain_limit`）为后续 lane，不在本模块范畴。

#![cfg(test)]

use prism_render_architecture::cloth::dynamics::{solve_cloth_with_collision, SolverParams};
use prism_render_architecture::cloth::{ClothParticle, ConstraintGraph, Vec3};

/// `cloth_sim.wesl` `cloth_predict` 的逐位 CPU 转写（原生数组，不复用黄金
/// `Vec3` 方法）。pinned（`inv_mass <= 0.0`）位置原样返回；否则按
/// `vel*retain + gravity*dt_sub` 更新速度，再 `pos + new_vel*dt_sub` 更新位置。
fn wesl_predict_position(
    pos: [f32; 3],
    vel: [f32; 3],
    inv_mass: f32,
    gravity: [f32; 3],
    retain: f32,
    dt_sub: f32,
) -> [f32; 3] {
    if inv_mass <= 0.0 {
        // pinned：`cloth_predict` 在 `inv_mass <= 0.0` 分支早退，位置不变。
        return pos;
    }
    let new_vel = [
        vel[0] * retain + gravity[0] * dt_sub,
        vel[1] * retain + gravity[1] * dt_sub,
        vel[2] * retain + gravity[2] * dt_sub,
    ];
    [
        pos[0] + new_vel[0] * dt_sub,
        pos[1] + new_vel[1] * dt_sub,
        pos[2] + new_vel[2] * dt_sub,
    ]
}

/// 单条粒子输入：`(position, velocity, inverse_mass)`。
type ParticleInput = ([f32; 3], [f32; 3], f32);

/// 构造黄金粒子批：pinned 走 [`ClothParticle::pinned`]，自由粒子走
/// [`ClothParticle::new`] 后显式写入初速度（无带初速的构造器）。
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

/// 跑黄金隔离 predict 与 `WESL` 转写，逐分量断言位置 `to_bits` 一致。
fn assert_predict_positions_bit_exact(
    particles: &[ParticleInput],
    gravity: [f32; 3],
    damping: f32,
    dt: f32,
) {
    assert!(dt > 0.0, "harness 只在 dt > 0.0 上做 parity");
    assert!(!particles.is_empty(), "至少一个粒子");

    let mut golden = build_golden(particles);
    let params = SolverParams {
        substeps: 1,
        iterations: 1,
        gravity: Vec3::new(gravity[0], gravity[1], gravity[2]),
        damping,
        strain_limit: 0.0,
    };
    let graph = ConstraintGraph::default();
    // no-op 碰撞钩子：位置映射为自身，步骤 4 不改变位置。
    solve_cloth_with_collision(&mut golden, &graph, params, dt, |p| p.position);

    // substeps == 1 ⇒ dt_sub == dt / 1.0 == dt（除 1.0 为 IEEE 精确）。
    let dt_sub = dt;
    let retain = 1.0 - damping.clamp(0.0, 1.0);

    for (i, &(p, v, inv_mass)) in particles.iter().enumerate() {
        let expected = wesl_predict_position(p, v, inv_mass, gravity, retain, dt_sub);
        let got = [
            golden[i].position.x,
            golden[i].position.y,
            golden[i].position.z,
        ];
        for k in 0..3 {
            assert_eq!(
                got[k].to_bits(),
                expected[k].to_bits(),
                "粒子 {i} 分量 {k} 位置不逐位：黄金 {} ({:#010x}) vs WESL {} ({:#010x})",
                got[k],
                got[k].to_bits(),
                expected[k],
                expected[k].to_bits(),
            );
        }
    }
}

/// 单个自由粒子零初速、标准重力下自由下落。
#[test]
fn free_particle_falls_under_gravity() {
    assert_predict_positions_bit_exact(
        &[([0.0, 10.0, 0.0], [0.0, 0.0, 0.0], 1.0)],
        [0.0, -9.81, 0.0],
        0.01,
        1.0 / 60.0,
    );
}

/// 单个自由粒子带三维初速度 + 重力。
#[test]
fn free_particle_with_initial_velocity() {
    assert_predict_positions_bit_exact(
        &[([1.0, 2.0, -3.0], [0.5, -0.25, 0.75], 2.0)],
        [0.0, -9.81, 0.0],
        0.05,
        1.0 / 120.0,
    );
}

/// pinned 粒子位置全程不变（`inverse_mass == 0`）。
#[test]
fn pinned_particle_does_not_move() {
    assert_predict_positions_bit_exact(
        &[([4.0, 5.0, 6.0], [10.0, 10.0, 10.0], 0.0)],
        [0.0, -9.81, 0.0],
        0.01,
        1.0 / 60.0,
    );
}

/// 零重力零初速：predict 为 no-op，位置不变。
#[test]
fn zero_gravity_zero_velocity_is_noop() {
    assert_predict_positions_bit_exact(
        &[([7.0, -2.0, 3.0], [0.0, 0.0, 0.0], 1.5)],
        [0.0, 0.0, 0.0],
        0.0,
        1.0 / 60.0,
    );
}

/// 零重力带初速：匀速平移（无外力，retain 衰减后仍前进）。
#[test]
fn zero_gravity_with_velocity_drifts() {
    assert_predict_positions_bit_exact(
        &[([0.0, 0.0, 0.0], [1.0, 2.0, 3.0], 1.0)],
        [0.0, 0.0, 0.0],
        0.0,
        1.0 / 90.0,
    );
}

/// 大重力加速度（强风/强力场），检验大数积分。
#[test]
fn large_gravity_magnitude() {
    assert_predict_positions_bit_exact(
        &[([0.0, 100.0, 0.0], [0.0, -5.0, 0.0], 1.0)],
        [0.0, -500.0, 0.0],
        0.02,
        1.0 / 60.0,
    );
}

/// damping == 0：`retain == 1.0`，速度不衰减。
#[test]
fn damping_zero_keeps_all_velocity() {
    assert_predict_positions_bit_exact(
        &[([0.0, 0.0, 0.0], [3.0, 4.0, 5.0], 1.0)],
        [0.0, -9.81, 0.0],
        0.0,
        1.0 / 60.0,
    );
}

/// damping == 1：`retain == 0.0`，初速完全被抹除，只剩本步重力增量。
#[test]
fn damping_one_kills_velocity() {
    assert_predict_positions_bit_exact(
        &[([0.0, 0.0, 0.0], [100.0, 100.0, 100.0], 1.0)],
        [0.0, -9.81, 0.0],
        1.0,
        1.0 / 60.0,
    );
}

/// damping > 1：两侧都 `clamp` 到 1.0，`retain == 0.0`。
#[test]
fn damping_above_one_is_clamped() {
    assert_predict_positions_bit_exact(
        &[([2.0, 2.0, 2.0], [7.0, -7.0, 7.0], 1.0)],
        [0.0, -9.81, 0.0],
        2.5,
        1.0 / 60.0,
    );
}

/// damping < 0：两侧都 `clamp` 到 0.0，`retain == 1.0`。
#[test]
fn damping_below_zero_is_clamped() {
    assert_predict_positions_bit_exact(
        &[([2.0, 2.0, 2.0], [7.0, -7.0, 7.0], 1.0)],
        [0.0, -9.81, 0.0],
        -0.5,
        1.0 / 60.0,
    );
}

/// 负坐标 + 负重力分量三轴混合。
#[test]
fn negative_coordinates_and_gravity() {
    assert_predict_positions_bit_exact(
        &[([-5.0, -10.0, -15.0], [-1.5, -2.5, -3.5], 0.75)],
        [-1.0, -9.81, -2.0],
        0.1,
        1.0 / 72.0,
    );
}

/// 三维斜向重力 + 斜向初速，非轴对齐。
#[test]
fn three_dimensional_oblique_motion() {
    assert_predict_positions_bit_exact(
        &[([0.3, 0.7, -0.2], [0.11, -0.22, 0.33], 1.25)],
        [1.7, -9.81, 0.6],
        0.03,
        1.0 / 144.0,
    );
}

/// 不同帧 `dt`：较大步长（30fps）检验 `dt_sub` 缩放。
#[test]
fn larger_timestep() {
    assert_predict_positions_bit_exact(
        &[([0.0, 50.0, 0.0], [2.0, 0.0, -2.0], 1.0)],
        [0.0, -9.81, 0.0],
        0.01,
        1.0 / 30.0,
    );
}

/// 混合批：pinned + 多个自由粒子同帧（检验按索引逐粒子对齐）。
#[test]
fn mixed_pinned_and_free_batch() {
    assert_predict_positions_bit_exact(
        &[
            ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0),
            ([1.0, 1.0, 1.0], [0.5, 0.5, 0.5], 1.0),
            ([2.0, -2.0, 2.0], [-1.0, 3.0, -4.0], 2.0),
            ([3.0, 3.0, -3.0], [0.0, 0.0, 0.0], 0.0),
        ],
        [0.0, -9.81, 0.0],
        0.02,
        1.0 / 60.0,
    );
}

/// 抖动混合：多粒子不同质量/初速/坐标尺度，检验数值鲁棒性。
#[test]
fn jittered_dense_batch() {
    let particles: Vec<ParticleInput> = (0..12)
        .map(|i| {
            let f = i as f32;
            (
                [f * 0.37 - 2.0, 10.0 - f * 0.51, f.mul_add(-0.29, 1.0)],
                [f.sin() * 0.5, f.cos() * 0.5, (f * 0.3).sin()],
                0.5 + f * 0.1,
            )
        })
        .collect();
    assert_predict_positions_bit_exact(&particles, [0.2, -9.81, -0.4], 0.015, 1.0 / 90.0);
}
