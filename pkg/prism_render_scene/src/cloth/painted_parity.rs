//! 画师绘制逐顶点约束四内核（`cloth_painted.wesl` 的 `painted_anim_drive` /
//! `painted_clamp_max_distance` / `painted_backstop` / `painted_blend_to_skin`）
//! 的 **逐位** CPU 转写，对齐架构层黄金 [`drive_toward_anim`] /
//! [`clamp_max_distance`] / [`apply_painted_backstop`] / [`blend_to_skin`]。
//!
//! `painted_gpu_tests` 只在**有** wgpu 适配器时才真机派发，无头环境整组
//! `skipping` 通过，证明不了内核逐位复刻了黄金的四段投影（设计 §9：不造假
//! parity）。本模块补上与设备无关的闭环：把每个 pass 的逐顶点算术独立转写成 CPU
//! 版本（不复用 host 打包，而是照 `WESL` 源码重写权重 clamp 与投影），按 host
//! 上传布局打包位置/速度/锚点/法线/权重，再对多种顶点断言每个 pass 后的位置
//! （`anim_drive` 另含速度）与黄金 **逐位（`to_bits`）一致**。
//!
//! ## 逐位一致的关键约定
//! * 每个 pass 都是逐顶点无状态投影：只读顶点 `i` 自身与其锚点/权重、只写顶点
//!   `i`，无跨顶点写冲突，故并行派发与黄金顺序遍历逐位一致。
//! * `is_pinned`（`inverse_mass <= 0`，存于位置 `.w`）、越界下标（`idx >=
//!   vertex_count`，`vertex_count` 取位置/锚点/权重三者最短长度）两类顶点所有
//!   pass 一律跳过。
//! * 权重 clamp 逐词镜像 [`PaintedConstraint::clamped`]：`backstop` 取 `max(0)`、
//!   `blend_weight`/`anim_drive` 夹到 `[0, 1]`、`max_distance` 的 `+inf`（未设
//!   上限）原样保留并跳过 clamp pass。本 parity 只喂 host 实际上传的**已清洗
//!   值域**（有限非负 `max_distance` 或 `+inf`、非负 `backstop`、`[0, 1]` 的
//!   `blend`/`anim`、有限 `gain`），该值域内内核与黄金逐位相同；`NaN` 由 host
//!   在上传前清洗，不会抵达内核，故不在本范畴。
//! * `normalize_or_zero` 用 `v * (1 / sqrt(len_sq))`（非 `inverseSqrt`），
//!   `distance`/投影用 `sqrt(dist_sq)`，逐词镜像黄金，CPU `f32::sqrt` 与黄金逐位
//!   一致；GPU 硬件 `sqrt` 精度差属设备级，由 `painted_gpu_tests` 真机覆盖。
//! * `EPS_LEN_SQ = 1e-12` 的退化守卫（归一化、投影分母）与黄金一致。

#![cfg(test)]

use prism_render_architecture::cloth::asset::PaintedConstraint;
use prism_render_architecture::cloth::painted::{
    apply_painted_backstop, blend_to_skin, clamp_max_distance, drive_toward_anim, AnimDriveParams,
    SkinnedAnchor,
};
use prism_render_architecture::cloth::{ClothParticle, Vec3};

/// `cloth_painted.wesl` 里 `CLOTH_PAINTED_EPS_LEN_SQ`：退化平方长度地板，镜像
/// 黄金 `EPS_LEN_SQ`。
const CLOTH_PAINTED_EPS_LEN_SQ: f32 = 1.0e-12;

/// 正无穷的 IEEE-754 位型，`cloth_painted.wesl` 用它判别未设上限的
/// `max_distance`（`bitcast<u32>(max_distance) == 0x7f800000u`）。
const F32_POS_INF_BITS: u32 = 0x7f80_0000;

// ---- 轻量 [f32; 3] 向量算术，逐词镜像 WESL 的 vec3 运算 ----

fn v_sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn v_add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn v_scale(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn v_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// `painted_normalize_or_zero` 的转写：`v * (1 / sqrt(len_sq))`，否则零。
fn normalize_or_zero(v: [f32; 3]) -> [f32; 3] {
    let len_sq = v_dot(v, v);
    if len_sq > CLOTH_PAINTED_EPS_LEN_SQ {
        v_scale(v, 1.0 / len_sq.sqrt())
    } else {
        [0.0, 0.0, 0.0]
    }
}

fn xyz(v: [f32; 4]) -> [f32; 3] {
    [v[0], v[1], v[2]]
}

// ---- host 上传打包 ----

/// 顶点状态：`xyz` 世界位置，`.w` inverse mass（`<= 0` 为 pinned）。
fn pack_positions(particles: &[ClothParticle]) -> Vec<[f32; 4]> {
    particles
        .iter()
        .map(|p| [p.position.x, p.position.y, p.position.z, p.inverse_mass])
        .collect()
}

/// 顶点速度：`xyz` 用于 anim-drive 的速度更新，`.w` 载荷位原样保留。
fn pack_velocities(particles: &[ClothParticle]) -> Vec<[f32; 4]> {
    particles
        .iter()
        .map(|p| [p.velocity.x, p.velocity.y, p.velocity.z, 0.0])
        .collect()
}

fn pack_anchor_positions(anchors: &[SkinnedAnchor]) -> Vec<[f32; 4]> {
    anchors
        .iter()
        .map(|a| [a.position.x, a.position.y, a.position.z, 0.0])
        .collect()
}

fn pack_anchor_normals(anchors: &[SkinnedAnchor]) -> Vec<[f32; 4]> {
    anchors
        .iter()
        .map(|a| [a.normal.x, a.normal.y, a.normal.z, 0.0])
        .collect()
}

/// 原始绘制权重：`x = max_distance, y = backstop, z = blend_weight,
/// w = anim_drive`，与 `painted_gpu_tests` 上传同款。
fn pack_weights(painted: &[PaintedConstraint]) -> Vec<[f32; 4]> {
    painted
        .iter()
        .map(|p| [p.max_distance, p.backstop, p.blend_weight, p.anim_drive])
        .collect()
}

/// `vertex_count`：位置/锚点/权重三者最短长度，镜像内核 uniform 与黄金 `zip`。
fn vertex_count(particles: &[ClothParticle], anchors: &[SkinnedAnchor], painted: &[PaintedConstraint]) -> u32 {
    particles.len().min(anchors.len()).min(painted.len()) as u32
}

// ---- 四个 pass 的逐顶点转写（原地改 positions / velocities）----

/// `painted_anim_drive` 转写：预解算软拉 + 速度更新。
fn wesl_anim_drive(
    positions: &mut [[f32; 4]],
    velocities: &mut [[f32; 4]],
    anchor_positions: &[[f32; 4]],
    weights: &[[f32; 4]],
    anim_gain: f32,
    inv_dt: f32,
    count: u32,
) {
    for idx in 0..count as usize {
        let state = positions[idx];
        if state[3] <= 0.0 {
            continue;
        }
        let anim_drive = weights[idx][3].clamp(0.0, 1.0);
        let follow = (anim_drive * anim_gain).clamp(0.0, 1.0);
        if follow <= 0.0 {
            continue;
        }
        let position = xyz(state);
        let anchor = xyz(anchor_positions[idx]);
        let delta = v_scale(v_sub(anchor, position), follow);
        let moved = v_add(position, delta);
        positions[idx] = [moved[0], moved[1], moved[2], state[3]];
        let velocity = v_add(xyz(velocities[idx]), v_scale(delta, inv_dt));
        velocities[idx] = [velocity[0], velocity[1], velocity[2], velocities[idx][3]];
    }
}

/// `painted_clamp_max_distance` 转写：越限顶点投回 max-distance 球面。
fn wesl_clamp_max_distance(
    positions: &mut [[f32; 4]],
    anchor_positions: &[[f32; 4]],
    weights: &[[f32; 4]],
    count: u32,
) {
    for idx in 0..count as usize {
        let state = positions[idx];
        if state[3] <= 0.0 {
            continue;
        }
        let raw_md = weights[idx][0];
        let max_distance = raw_md.max(0.0);
        let raw_is_nan = raw_md.is_nan();
        let md_is_pos_inf = max_distance.to_bits() == F32_POS_INF_BITS;
        if raw_is_nan || md_is_pos_inf {
            continue;
        }
        let position = xyz(state);
        let anchor = xyz(anchor_positions[idx]);
        let drift = v_sub(position, anchor);
        let dist_sq = v_dot(drift, drift);
        let max_sq = max_distance * max_distance;
        if dist_sq <= max_sq {
            continue;
        }
        if dist_sq > CLOTH_PAINTED_EPS_LEN_SQ {
            let dist = dist_sq.sqrt();
            let projected = v_add(anchor, v_scale(drift, max_distance / dist));
            positions[idx] = [projected[0], projected[1], projected[2], state[3]];
        } else {
            positions[idx] = [anchor[0], anchor[1], anchor[2], state[3]];
        }
    }
}

/// `painted_backstop` 转写：把陷入垫球内的顶点径向推出。
fn wesl_backstop(
    positions: &mut [[f32; 4]],
    anchor_positions: &[[f32; 4]],
    anchor_normals: &[[f32; 4]],
    weights: &[[f32; 4]],
    count: u32,
) {
    for idx in 0..count as usize {
        let state = positions[idx];
        if state[3] <= 0.0 {
            continue;
        }
        let backstop = weights[idx][1].max(0.0);
        if backstop <= 0.0 {
            continue;
        }
        let normal = normalize_or_zero(xyz(anchor_normals[idx]));
        if v_dot(normal, normal) <= CLOTH_PAINTED_EPS_LEN_SQ {
            continue;
        }
        let position = xyz(state);
        let anchor = xyz(anchor_positions[idx]);
        let center = v_sub(anchor, v_scale(normal, backstop));
        let to_vertex = v_sub(position, center);
        let dist_sq = v_dot(to_vertex, to_vertex);
        let radius = backstop;
        if dist_sq >= radius * radius {
            continue;
        }
        if dist_sq > CLOTH_PAINTED_EPS_LEN_SQ {
            let dist = dist_sq.sqrt();
            let pushed = v_add(center, v_scale(to_vertex, radius / dist));
            positions[idx] = [pushed[0], pushed[1], pushed[2], state[3]];
        } else {
            let pushed = v_add(center, v_scale(normal, radius));
            positions[idx] = [pushed[0], pushed[1], pushed[2], state[3]];
        }
    }
}

/// `painted_blend_to_skin` 转写：按 `blend_weight` 把仿真位置混向锚点。
fn wesl_blend_to_skin(
    positions: &mut [[f32; 4]],
    anchor_positions: &[[f32; 4]],
    weights: &[[f32; 4]],
    count: u32,
) {
    for idx in 0..count as usize {
        let state = positions[idx];
        if state[3] <= 0.0 {
            continue;
        }
        let blend_weight = weights[idx][2].clamp(0.0, 1.0);
        let anchor = xyz(anchor_positions[idx]);
        let offset = v_sub(xyz(state), anchor);
        let blended = v_add(anchor, v_scale(offset, blend_weight));
        positions[idx] = [blended[0], blended[1], blended[2], state[3]];
    }
}

// ---- 比对 ----

/// 断言转写后的 positions（及可选 velocities）与黄金粒子逐位一致。
fn assert_positions_bit_exact(
    golden: &[ClothParticle],
    positions: &[[f32; 4]],
    velocities: Option<&[[f32; 4]]>,
) {
    for (idx, g) in golden.iter().enumerate() {
        let p = positions[idx];
        assert_eq!(
            p[0].to_bits(),
            g.position.x.to_bits(),
            "position.x diverged at vertex {idx}"
        );
        assert_eq!(
            p[1].to_bits(),
            g.position.y.to_bits(),
            "position.y diverged at vertex {idx}"
        );
        assert_eq!(
            p[2].to_bits(),
            g.position.z.to_bits(),
            "position.z diverged at vertex {idx}"
        );
        if let Some(vels) = velocities {
            let v = vels[idx];
            assert_eq!(
                v[0].to_bits(),
                g.velocity.x.to_bits(),
                "velocity.x diverged at vertex {idx}"
            );
            assert_eq!(
                v[1].to_bits(),
                g.velocity.y.to_bits(),
                "velocity.y diverged at vertex {idx}"
            );
            assert_eq!(
                v[2].to_bits(),
                g.velocity.z.to_bits(),
                "velocity.z diverged at vertex {idx}"
            );
        }
    }
}

/// 跑黄金 [`drive_toward_anim`] 与转写，断言位置+速度逐位一致。
fn assert_anim_drive_bit_exact(
    particles: &[ClothParticle],
    anchors: &[SkinnedAnchor],
    painted: &[PaintedConstraint],
    params: AnimDriveParams,
    dt: f32,
) {
    let mut golden = particles.to_vec();
    drive_toward_anim(&mut golden, anchors, painted, params, dt);

    let count = vertex_count(particles, anchors, painted);
    let mut positions = pack_positions(particles);
    let mut velocities = pack_velocities(particles);
    let anchor_positions = pack_anchor_positions(anchors);
    let weights = pack_weights(painted);

    // host 侧与黄金同款的 gain 清洗与 inv_dt 预计算；禁用 / 非正 dt 下内核整体不派发。
    let sanitized = params.sanitized();
    if sanitized.enabled && sanitized.gain > 0.0 && dt > 0.0 {
        let inv_dt = 1.0 / dt;
        wesl_anim_drive(
            &mut positions,
            &mut velocities,
            &anchor_positions,
            &weights,
            sanitized.gain,
            inv_dt,
            count,
        );
    }
    assert_positions_bit_exact(&golden, &positions, Some(&velocities));
}

/// 跑黄金 [`clamp_max_distance`] 与转写，断言位置逐位一致。
fn assert_clamp_bit_exact(
    particles: &[ClothParticle],
    anchors: &[SkinnedAnchor],
    painted: &[PaintedConstraint],
) {
    let mut golden = particles.to_vec();
    clamp_max_distance(&mut golden, anchors, painted);

    let count = vertex_count(particles, anchors, painted);
    let mut positions = pack_positions(particles);
    let anchor_positions = pack_anchor_positions(anchors);
    let weights = pack_weights(painted);
    wesl_clamp_max_distance(&mut positions, &anchor_positions, &weights, count);
    assert_positions_bit_exact(&golden, &positions, None);
}

/// 跑黄金 [`apply_painted_backstop`] 与转写，断言位置逐位一致。
fn assert_backstop_bit_exact(
    particles: &[ClothParticle],
    anchors: &[SkinnedAnchor],
    painted: &[PaintedConstraint],
) {
    let mut golden = particles.to_vec();
    apply_painted_backstop(&mut golden, anchors, painted);

    let count = vertex_count(particles, anchors, painted);
    let mut positions = pack_positions(particles);
    let anchor_positions = pack_anchor_positions(anchors);
    let anchor_normals = pack_anchor_normals(anchors);
    let weights = pack_weights(painted);
    wesl_backstop(&mut positions, &anchor_positions, &anchor_normals, &weights, count);
    assert_positions_bit_exact(&golden, &positions, None);
}

/// 跑黄金 [`blend_to_skin`] 与转写，断言位置逐位一致。
fn assert_blend_bit_exact(
    particles: &[ClothParticle],
    anchors: &[SkinnedAnchor],
    painted: &[PaintedConstraint],
) {
    let mut golden = particles.to_vec();
    blend_to_skin(&mut golden, anchors, painted);

    let count = vertex_count(particles, anchors, painted);
    let mut positions = pack_positions(particles);
    let anchor_positions = pack_anchor_positions(anchors);
    let weights = pack_weights(painted);
    wesl_blend_to_skin(&mut positions, &anchor_positions, &weights, count);
    assert_positions_bit_exact(&golden, &positions, None);
}

/// 顺序跑四个 pass（drive → clamp → backstop → blend），断言最终位置+速度一致。
fn assert_all_passes_bit_exact(
    particles: &[ClothParticle],
    anchors: &[SkinnedAnchor],
    painted: &[PaintedConstraint],
    params: AnimDriveParams,
    dt: f32,
) {
    let mut golden = particles.to_vec();
    drive_toward_anim(&mut golden, anchors, painted, params, dt);
    clamp_max_distance(&mut golden, anchors, painted);
    apply_painted_backstop(&mut golden, anchors, painted);
    blend_to_skin(&mut golden, anchors, painted);

    let count = vertex_count(particles, anchors, painted);
    let mut positions = pack_positions(particles);
    let mut velocities = pack_velocities(particles);
    let anchor_positions = pack_anchor_positions(anchors);
    let anchor_normals = pack_anchor_normals(anchors);
    let weights = pack_weights(painted);

    let sanitized = params.sanitized();
    if sanitized.enabled && sanitized.gain > 0.0 && dt > 0.0 {
        let inv_dt = 1.0 / dt;
        wesl_anim_drive(
            &mut positions,
            &mut velocities,
            &anchor_positions,
            &weights,
            sanitized.gain,
            inv_dt,
            count,
        );
    }
    wesl_clamp_max_distance(&mut positions, &anchor_positions, &weights, count);
    wesl_backstop(&mut positions, &anchor_positions, &anchor_normals, &weights, count);
    wesl_blend_to_skin(&mut positions, &anchor_positions, &weights, count);
    assert_positions_bit_exact(&golden, &positions, Some(&velocities));
}

// ---- 构造工具 ----

fn free_particle(position: Vec3) -> ClothParticle {
    ClothParticle::new(position, 1.0)
}

fn moving_particle(position: Vec3, velocity: Vec3) -> ClothParticle {
    ClothParticle {
        position,
        velocity,
        inverse_mass: 1.0,
    }
}

fn anchor(position: Vec3, normal: Vec3) -> SkinnedAnchor {
    SkinnedAnchor::new(position, normal)
}

/// 启用、半增益的 anim-drive 全局参数。
fn drive_params() -> AnimDriveParams {
    AnimDriveParams {
        gain: 0.5,
        enabled: true,
    }
}

const DT: f32 = 1.0 / 60.0;

// ============================ anim_drive ============================

#[test]
fn anim_drive_pulls_free_vertex_and_updates_velocity_bit_for_bit() {
    let particles = [moving_particle(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.1, -0.2, 0.3))];
    let anchors = [anchor(Vec3::new(2.0, 1.0, -1.0), Vec3::ZERO)];
    let painted = [PaintedConstraint::new(f32::INFINITY, 0.0, 1.0, 0.8)];
    assert_anim_drive_bit_exact(&particles, &anchors, &painted, drive_params(), DT);
}

#[test]
fn anim_drive_disabled_is_noop_bit_for_bit() {
    let particles = [moving_particle(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.1, 0.2, 0.3))];
    let anchors = [anchor(Vec3::new(2.0, 1.0, -1.0), Vec3::ZERO)];
    let painted = [PaintedConstraint::new(f32::INFINITY, 0.0, 1.0, 1.0)];
    let params = AnimDriveParams {
        gain: 1.0,
        enabled: false,
    };
    assert_anim_drive_bit_exact(&particles, &anchors, &painted, params, DT);
}

#[test]
fn anim_drive_zero_follow_is_noop_bit_for_bit() {
    let particles = [moving_particle(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.1, 0.2, 0.3))];
    let anchors = [anchor(Vec3::new(2.0, 1.0, -1.0), Vec3::ZERO)];
    // anim_drive = 0 → follow = 0，整顶点跳过。
    let painted = [PaintedConstraint::new(f32::INFINITY, 0.0, 1.0, 0.0)];
    assert_anim_drive_bit_exact(&particles, &anchors, &painted, drive_params(), DT);
}

#[test]
fn anim_drive_non_positive_dt_is_noop_bit_for_bit() {
    let particles = [moving_particle(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.1, 0.2, 0.3))];
    let anchors = [anchor(Vec3::new(2.0, 1.0, -1.0), Vec3::ZERO)];
    let painted = [PaintedConstraint::new(f32::INFINITY, 0.0, 1.0, 1.0)];
    assert_anim_drive_bit_exact(&particles, &anchors, &painted, drive_params(), 0.0);
}

#[test]
fn anim_drive_skips_pinned_bit_for_bit() {
    let particles = [ClothParticle::pinned(Vec3::new(9.0, 9.0, 9.0))];
    let anchors = [anchor(Vec3::ZERO, Vec3::ZERO)];
    let painted = [PaintedConstraint::new(f32::INFINITY, 0.0, 1.0, 1.0)];
    let params = AnimDriveParams {
        gain: 1.0,
        enabled: true,
    };
    assert_anim_drive_bit_exact(&particles, &anchors, &painted, params, DT);
}

// ======================= clamp_max_distance =======================

#[test]
fn clamp_projects_over_limit_onto_sphere_bit_for_bit() {
    let particles = [free_particle(Vec3::new(3.0, 0.0, 0.0))];
    let anchors = [anchor(Vec3::ZERO, Vec3::ZERO)];
    let painted = [PaintedConstraint::new(1.0, 0.0, 1.0, 0.0)];
    assert_clamp_bit_exact(&particles, &anchors, &painted);
}

#[test]
fn clamp_within_distance_is_untouched_bit_for_bit() {
    let particles = [free_particle(Vec3::new(0.5, 0.0, 0.0))];
    let anchors = [anchor(Vec3::ZERO, Vec3::ZERO)];
    let painted = [PaintedConstraint::new(1.0, 0.0, 1.0, 0.0)];
    assert_clamp_bit_exact(&particles, &anchors, &painted);
}

#[test]
fn clamp_zero_distance_welds_to_anchor_bit_for_bit() {
    let particles = [free_particle(Vec3::new(4.0, 5.0, 6.0))];
    let anchors = [anchor(Vec3::new(1.0, 2.0, 3.0), Vec3::ZERO)];
    let painted = [PaintedConstraint::new(0.0, 0.0, 1.0, 0.0)];
    assert_clamp_bit_exact(&particles, &anchors, &painted);
}

#[test]
fn clamp_uncapped_infinite_is_noop_bit_for_bit() {
    let particles = [free_particle(Vec3::new(50.0, 0.0, 0.0))];
    let anchors = [anchor(Vec3::ZERO, Vec3::ZERO)];
    let painted = [PaintedConstraint::new(f32::INFINITY, 0.0, 1.0, 0.0)];
    assert_clamp_bit_exact(&particles, &anchors, &painted);
}

#[test]
fn clamp_degenerate_drift_snaps_to_anchor_bit_for_bit() {
    // max_distance = 0 → max_sq = 0；漂移极小（< sqrt(EPS)）落入退化分支，直接贴锚。
    let particles = [free_particle(Vec3::new(1.0e-7, 0.0, 0.0))];
    let anchors = [anchor(Vec3::ZERO, Vec3::ZERO)];
    let painted = [PaintedConstraint::new(0.0, 0.0, 1.0, 0.0)];
    assert_clamp_bit_exact(&particles, &anchors, &painted);
}

// ============================ backstop ============================

#[test]
fn backstop_pushes_vertex_out_of_cushion_bit_for_bit() {
    let particles = [free_particle(Vec3::new(0.0, -0.5, 0.0))];
    let anchors = [anchor(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0))];
    let painted = [PaintedConstraint::new(f32::INFINITY, 1.0, 1.0, 0.0)];
    assert_backstop_bit_exact(&particles, &anchors, &painted);
}

#[test]
fn backstop_outside_cushion_is_untouched_bit_for_bit() {
    let particles = [free_particle(Vec3::new(0.0, 0.5, 0.0))];
    let anchors = [anchor(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0))];
    let painted = [PaintedConstraint::new(f32::INFINITY, 1.0, 1.0, 0.0)];
    assert_backstop_bit_exact(&particles, &anchors, &painted);
}

#[test]
fn backstop_zero_distance_disabled_bit_for_bit() {
    let particles = [free_particle(Vec3::new(0.0, -0.5, 0.0))];
    let anchors = [anchor(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0))];
    let painted = [PaintedConstraint::new(f32::INFINITY, 0.0, 1.0, 0.0)];
    assert_backstop_bit_exact(&particles, &anchors, &painted);
}

#[test]
fn backstop_zero_normal_disabled_bit_for_bit() {
    let particles = [free_particle(Vec3::new(0.0, -0.5, 0.0))];
    let anchors = [anchor(Vec3::ZERO, Vec3::ZERO)];
    let painted = [PaintedConstraint::new(f32::INFINITY, 1.0, 1.0, 0.0)];
    assert_backstop_bit_exact(&particles, &anchors, &painted);
}

#[test]
fn backstop_at_cushion_center_pushes_along_normal_bit_for_bit() {
    // 顶点恰在垫球心（anchor - normal*backstop = (0,-1,0)），落入退化分支，
    // 沿法线推到最近表面点（锚点 (0,0,0)）。
    let particles = [free_particle(Vec3::new(0.0, -1.0, 0.0))];
    let anchors = [anchor(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0))];
    let painted = [PaintedConstraint::new(f32::INFINITY, 1.0, 1.0, 0.0)];
    assert_backstop_bit_exact(&particles, &anchors, &painted);
}

// ========================== blend_to_skin ==========================

#[test]
fn blend_mixes_sim_and_skin_bit_for_bit() {
    let particles = [free_particle(Vec3::new(0.0, 4.0, 0.0))];
    let anchors = [anchor(Vec3::ZERO, Vec3::ZERO)];
    let painted = [PaintedConstraint::new(f32::INFINITY, 0.0, 0.25, 0.0)];
    assert_blend_bit_exact(&particles, &anchors, &painted);
}

#[test]
fn blend_full_weight_keeps_sim_bit_for_bit() {
    let particles = [free_particle(Vec3::new(1.0, 2.0, 3.0))];
    let anchors = [anchor(Vec3::new(-1.0, -2.0, -3.0), Vec3::ZERO)];
    let painted = [PaintedConstraint::new(f32::INFINITY, 0.0, 1.0, 0.0)];
    assert_blend_bit_exact(&particles, &anchors, &painted);
}

#[test]
fn blend_zero_weight_snaps_to_skin_bit_for_bit() {
    let particles = [free_particle(Vec3::new(1.0, 2.0, 3.0))];
    let anchors = [anchor(Vec3::new(-1.0, -2.0, -3.0), Vec3::ZERO)];
    let painted = [PaintedConstraint::new(f32::INFINITY, 0.0, 0.0, 0.0)];
    assert_blend_bit_exact(&particles, &anchors, &painted);
}

// =========================== 综合 / 边界 ===========================

#[test]
fn out_of_range_indices_are_skipped_bit_for_bit() {
    // 2 粒子、1 锚点、1 权重 → vertex_count = 1，仅下标 0 被处理。
    let particles = [
        free_particle(Vec3::new(5.0, 0.0, 0.0)),
        free_particle(Vec3::new(9.0, 0.0, 0.0)),
    ];
    let anchors = [anchor(Vec3::ZERO, Vec3::ZERO)];
    let painted = [PaintedConstraint::new(1.0, 0.0, 1.0, 0.0)];
    assert_clamp_bit_exact(&particles, &anchors, &painted);
}

#[test]
fn pinned_vertices_never_move_across_all_passes_bit_for_bit() {
    let particles = [ClothParticle::pinned(Vec3::new(9.0, 9.0, 9.0))];
    let anchors = [anchor(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0))];
    let painted = [PaintedConstraint::new(0.0, 1.0, 0.0, 1.0)];
    let params = AnimDriveParams {
        gain: 1.0,
        enabled: true,
    };
    assert_all_passes_bit_exact(&particles, &anchors, &painted, params, DT);
}

#[test]
fn mixed_grid_across_workgroups_matches_golden_bit_for_bit() {
    // 跨越 64 线程 workgroup 边界的混合顶点：自由 / pinned / 各种权重交错，
    // 四 pass 顺序跑完仍逐位一致。
    let mut particles = Vec::new();
    let mut anchors = Vec::new();
    let mut painted = Vec::new();
    for i in 0..96u32 {
        let f = i as f32;
        let position = Vec3::new((f * 0.21).sin() * 2.0, (f * 0.13).cos() * 2.0, (f * 0.07).sin());
        if i % 11 == 0 {
            particles.push(ClothParticle::pinned(position));
        } else {
            particles.push(moving_particle(position, Vec3::new((f * 0.05).sin() * 0.3, 0.1, -0.1)));
        }
        let anchor_pos = Vec3::new((f * 0.17).cos(), (f * 0.09).sin(), (f * 0.11).cos());
        let normal = if i % 7 == 0 {
            Vec3::ZERO
        } else {
            Vec3::new((f * 0.3).sin(), 1.0, (f * 0.2).cos())
        };
        anchors.push(anchor(anchor_pos, normal));
        let max_distance = if i % 5 == 0 { f32::INFINITY } else { 0.1 + (i % 4) as f32 * 0.25 };
        let backstop = (i % 3) as f32 * 0.4;
        let blend = ((i % 6) as f32) / 5.0;
        let anim = ((i % 4) as f32) / 3.0;
        painted.push(PaintedConstraint::new(max_distance, backstop, blend, anim));
    }
    let params = AnimDriveParams {
        gain: 0.75,
        enabled: true,
    };
    assert_all_passes_bit_exact(&particles, &anchors, &painted, params, DT);
}

#[test]
fn jittered_positions_match_golden_bit_for_bit() {
    // 非轴对齐 3D 抖动位型，验证 distance / normalize 的平方和开方逐位复刻黄金。
    let particles = [
        moving_particle(Vec3::new(0.013, -0.047, 0.021), Vec3::new(0.03, -0.02, 0.05)),
        moving_particle(Vec3::new(1.737, 0.902, -0.613), Vec3::new(-0.01, 0.04, 0.02)),
        moving_particle(Vec3::new(-0.411, 1.228, 0.774), Vec3::new(0.02, 0.01, -0.03)),
    ];
    let anchors = [
        anchor(Vec3::new(0.1, 0.2, -0.1), Vec3::new(0.3, 0.9, 0.2)),
        anchor(Vec3::new(1.5, 1.0, -0.5), Vec3::new(-0.4, 0.7, 0.6)),
        anchor(Vec3::new(-0.3, 1.1, 0.8), Vec3::new(0.2, -0.5, 0.8)),
    ];
    let painted = [
        PaintedConstraint::new(0.5, 0.3, 0.8, 0.6),
        PaintedConstraint::new(2.0, 0.1, 0.4, 0.9),
        PaintedConstraint::new(f32::INFINITY, 0.5, 0.9, 0.2),
    ];
    let params = drive_params();
    assert_all_passes_bit_exact(&particles, &anchors, &painted, params, DT);
}
