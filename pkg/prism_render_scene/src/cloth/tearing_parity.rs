//! 布料撕裂（约束断裂）检测内核 `cloth_tearing_flag`（`cloth_tearing.wesl`）的
//! **逐位** CPU 转写，对齐架构层黄金 [`tear_flags`]。
//!
//! `tearing_gpu_tests` 只在**有** wgpu 适配器时才真机派发，无头环境整组
//! `skipping` 通过，证明不了内核的逐边判定逐位复刻了 CPU 黄金（设计 §9：不造假
//! parity）。本模块补上与设备无关的闭环：把内核的逐条判定独立转写成 CPU 版本
//! （不复用 host 的 `pack_constraint`，而是照 `WESL` 源码重写 kind→u32 映射与
//! 判定），按 host 上传路径打包位置与约束，再对多种边断言逐边 flag 与
//! [`tear_flags`] **逐位一致**。
//!
//! ## 逐位一致的关键约定
//! * 拉伸应变 `e = (len - rest) / rest`，其中 `len` 取 `sqrt(dx*dx + dy*dy +
//!   dz*dz)`，逐词镜像黄金 `Vec3::distance == sub().length() ==
//!   length_squared().sqrt()`，减法方向 `a - b` 与黄金 `a.distance(b)` 一致。
//! * 断裂判定为**严格** `e > break_strain`：恰好等于阈值不断裂，与黄金一致。
//! * 跳过三类边并保 flag 0：单边约束（`Lra`/`Tether`，kind u32 为 3/4）、退化
//!   rest（`<= EPS_REST = 1e-9`）、越界端点（`a >= count || b >= count`），逐条
//!   镜像黄金 `edge_strain` 的 `None` 守卫。
//! * `break_strain` 的 sanitize（`NaN` 或 `< 0` → `+∞`）与黄金
//!   [`TearingParams::sanitized`] 一致：阈值被设成 `+∞` 时任何有限应变都
//!   `> +∞ == false`，故一条都不撕——压缩（负应变）同理永不撕。
//! * `sqrt` 在 CPU 上按黄金同款 `f32::sqrt` 求值，二者逐位一致；GPU `distance`
//!   的硬件 `sqrt` 精度差属设备级，由 `tearing_gpu_tests` 真机覆盖，不在本范畴。

#![cfg(test)]

use prism_render_architecture::cloth::tearing::{tear_flags, TearingParams};
use prism_render_architecture::cloth::{ClothParticle, Constraint, ConstraintKind, Vec3};
use prism_render_architecture::cloth::Compliance;

/// `cloth_tearing.wesl` 里 `CLOTH_TEARING_KIND_LRA`：单边长程约束 kind 的 u32
/// 码，撕裂恒跳过。
const CLOTH_TEARING_KIND_LRA: u32 = 3;
/// `cloth_tearing.wesl` 里 `CLOTH_TEARING_KIND_TETHER`：单边系绳 kind 的 u32 码。
const CLOTH_TEARING_KIND_TETHER: u32 = 4;
/// `cloth_tearing.wesl` 里 `CLOTH_TEARING_EPS_REST`：退化 rest 的数值地板。
const CLOTH_TEARING_EPS_REST: f32 = 1.0e-9;

/// 把一组粒子按 host 上传布局打包成位置缓冲：`xyz` 为世界位置，`.w` 为 inverse
/// mass（撕裂不读，仅凑 16 字节步长），与 `tearing_gpu_tests::upload_positions`
/// 同款。
fn upload_positions(particles: &[ClothParticle]) -> Vec<[f32; 4]> {
    particles
        .iter()
        .map(|p| [p.position.x, p.position.y, p.position.z, p.inverse_mass])
        .collect()
}

/// 把 [`ConstraintKind`] 映射到 host 上传的 u32 kind 码，逐一镜像
/// `cloth_sim` ABI 常量（`Stretch=0, Bend=1, Shear=2, Lra=3, Tether=4`）。
fn gpu_constraint_kind(kind: ConstraintKind) -> u32 {
    match kind {
        ConstraintKind::Stretch => 0,
        ConstraintKind::Bend => 1,
        ConstraintKind::Shear => 2,
        ConstraintKind::Lra => CLOTH_TEARING_KIND_LRA,
        ConstraintKind::Tether => CLOTH_TEARING_KIND_TETHER,
    }
}

/// `cloth_tearing.wesl` 里 host 上传前对 `break_strain` 的 sanitize，逐位镜像
/// [`TearingParams::sanitized`]：`NaN` 或 `< 0` → `+∞`，否则原样。
fn sanitize_break_strain(break_strain: f32) -> f32 {
    if break_strain.is_nan() || break_strain < 0.0 {
        f32::INFINITY
    } else {
        break_strain
    }
}

/// `cloth_tearing_flag` 的单边 CPU 转写：对一条约束回传 `1`（撕）或 `0`（留）。
fn wesl_tear_flag(
    positions: &[[f32; 4]],
    a: u32,
    b: u32,
    rest_length: f32,
    kind: u32,
    break_strain: f32,
    particle_count: u32,
) -> u32 {
    let one_sided = kind == CLOTH_TEARING_KIND_LRA || kind == CLOTH_TEARING_KIND_TETHER;
    let degenerate = rest_length <= CLOTH_TEARING_EPS_REST;
    let out_of_range = a >= particle_count || b >= particle_count;
    if !one_sided && !degenerate && !out_of_range {
        let pa = positions[a as usize];
        let pb = positions[b as usize];
        let dx = pa[0] - pb[0];
        let dy = pa[1] - pb[1];
        let dz = pa[2] - pb[2];
        let len = (dx * dx + dy * dy + dz * dz).sqrt();
        let strain = (len - rest_length) / rest_length;
        if strain > break_strain {
            return 1;
        }
    }
    0
}

/// 对一组约束断言：WESL 转写的逐边 flag 与黄金 [`tear_flags`] **逐位一致**。
fn assert_bit_exact(
    particles: &[ClothParticle],
    constraints: &[Constraint],
    break_strain: f32,
) {
    let params = TearingParams { break_strain };
    let golden = tear_flags(constraints, particles, params);

    let positions = upload_positions(particles);
    let sanitized = sanitize_break_strain(break_strain);
    let particle_count = particles.len() as u32;

    assert_eq!(golden.len(), constraints.len());
    for (edge, con) in constraints.iter().enumerate() {
        let twin = wesl_tear_flag(
            &positions,
            con.a,
            con.b,
            con.rest_length,
            gpu_constraint_kind(con.kind),
            sanitized,
            particle_count,
        );
        let golden_flag = u32::from(golden[edge]);
        assert_eq!(
            twin, golden_flag,
            "tear flag diverged at edge {edge}: twin={twin} golden={golden_flag}"
        );
    }
}

/// 原点与给定位置的两粒子，便于按欧氏距离控制应变。
fn particle_at(position: Vec3) -> ClothParticle {
    ClothParticle::new(position, 1.0)
}

/// 两端点 `(a, b)`、rest 为 `rest` 的两边织物边（`Stretch`）。
fn stretch(a: u32, b: u32, rest: f32) -> Constraint {
    Constraint::new(a, b, rest, Compliance::RIGID, ConstraintKind::Stretch)
}

#[test]
fn stretched_two_sided_edge_matches_golden_bit_for_bit() {
    // 端点相距 2.0、rest 1.0 → 应变 1.0 > 0.5，应撕。
    let particles = [
        particle_at(Vec3::new(0.0, 0.0, 0.0)),
        particle_at(Vec3::new(2.0, 0.0, 0.0)),
    ];
    assert_bit_exact(&particles, &[stretch(0, 1, 1.0)], 0.5);
}

#[test]
fn slack_two_sided_edge_matches_golden_bit_for_bit() {
    // 端点相距 0.4、rest 1.0 → 应变 -0.6 < 0.5，应留。
    let particles = [
        particle_at(Vec3::new(0.0, 0.0, 0.0)),
        particle_at(Vec3::new(0.4, 0.0, 0.0)),
    ];
    assert_bit_exact(&particles, &[stretch(0, 1, 1.0)], 0.5);
}

#[test]
fn near_threshold_edge_matches_golden_bit_for_bit() {
    // 应变 ~0.5，贴阈值：twin 与 golden 用同一算术，结果必然一致。
    let particles = [
        particle_at(Vec3::new(0.0, 0.0, 0.0)),
        particle_at(Vec3::new(1.5, 0.0, 0.0)),
    ];
    assert_bit_exact(&particles, &[stretch(0, 1, 1.0)], 0.5);
}

#[test]
fn lra_leash_never_tears_bit_for_bit() {
    // 单边长程约束即便拉伸 10 倍也不撕。
    let particles = [
        particle_at(Vec3::new(0.0, 0.0, 0.0)),
        particle_at(Vec3::new(10.0, 0.0, 0.0)),
    ];
    let con = Constraint::new(0, 1, 1.0, Compliance::RIGID, ConstraintKind::Lra);
    assert_bit_exact(&particles, &[con], 0.5);
}

#[test]
fn tether_leash_never_tears_bit_for_bit() {
    let particles = [
        particle_at(Vec3::new(0.0, 0.0, 0.0)),
        particle_at(Vec3::new(8.0, 0.0, 0.0)),
    ];
    let con = Constraint::new(0, 1, 1.0, Compliance::RIGID, ConstraintKind::Tether);
    assert_bit_exact(&particles, &[con], 0.5);
}

#[test]
fn degenerate_rest_length_never_tears_bit_for_bit() {
    // rest <= EPS_REST：退化边恒留。
    let particles = [
        particle_at(Vec3::new(0.0, 0.0, 0.0)),
        particle_at(Vec3::new(5.0, 0.0, 0.0)),
    ];
    assert_bit_exact(&particles, &[stretch(0, 1, 1.0e-12)], 0.5);
}

#[test]
fn out_of_range_endpoint_never_tears_bit_for_bit() {
    // 端点 b=5 越界（只有 2 个粒子）：恒留。
    let particles = [
        particle_at(Vec3::new(0.0, 0.0, 0.0)),
        particle_at(Vec3::new(3.0, 0.0, 0.0)),
    ];
    assert_bit_exact(&particles, &[stretch(0, 5, 1.0)], 0.5);
}

#[test]
fn nan_break_strain_tears_nothing_bit_for_bit() {
    // sanitize 把 NaN 阈值抬成 +∞，任何有限应变都不 > +∞。
    let particles = [
        particle_at(Vec3::new(0.0, 0.0, 0.0)),
        particle_at(Vec3::new(100.0, 0.0, 0.0)),
    ];
    assert_bit_exact(&particles, &[stretch(0, 1, 1.0)], f32::NAN);
}

#[test]
fn negative_break_strain_tears_nothing_bit_for_bit() {
    let particles = [
        particle_at(Vec3::new(0.0, 0.0, 0.0)),
        particle_at(Vec3::new(100.0, 0.0, 0.0)),
    ];
    assert_bit_exact(&particles, &[stretch(0, 1, 1.0)], -2.0);
}

#[test]
fn zero_break_strain_tears_any_tension_bit_for_bit() {
    // 阈值 0：任何正应变都撕，零/负应变留。
    let particles = [
        particle_at(Vec3::new(0.0, 0.0, 0.0)),
        particle_at(Vec3::new(1.000_01, 0.0, 0.0)),
        particle_at(Vec3::new(2.0, 0.0, 0.0)),
    ];
    let constraints = [stretch(0, 1, 1.0), stretch(1, 2, 2.0)];
    assert_bit_exact(&particles, &constraints, 0.0);
}

#[test]
fn mixed_graph_across_workgroups_matches_golden_bit_for_bit() {
    // 跨多个 64 宽工作组的混合图：不同 kind、不同应变、含退化/越界/单边边。
    let mut particles = Vec::new();
    for i in 0..160u32 {
        let f = i as f32;
        particles.push(particle_at(Vec3::new(f * 0.5, (f * 0.1).sin(), f * 0.05)));
    }
    let mut constraints = Vec::new();
    for i in 0..150u32 {
        let a = i;
        let b = i + 1;
        let rest = 0.3 + (i as f32) * 0.001;
        let kind = match i % 5 {
            0 => ConstraintKind::Stretch,
            1 => ConstraintKind::Shear,
            2 => ConstraintKind::Bend,
            3 => ConstraintKind::Lra,
            _ => ConstraintKind::Tether,
        };
        constraints.push(Constraint::new(a, b, rest, Compliance::RIGID, kind));
    }
    // 掺入一条退化 rest 与一条越界端点。
    constraints.push(stretch(0, 2, 1.0e-11));
    constraints.push(stretch(10, 999, 0.5));
    assert_bit_exact(&particles, &constraints, 0.4);
}

#[test]
fn jittered_positions_match_golden_bit_for_bit() {
    // 非规整坐标，确保 sqrt/除法的尾数位在两侧逐位吻合。
    let mut particles = Vec::new();
    let seeds = [0.137_f32, 1.919, 2.718, 0.577, 3.141, 1.414];
    for (i, s) in seeds.iter().cycle().take(80).enumerate() {
        let j = i as f32;
        particles.push(particle_at(Vec3::new(
            s * 0.7 + j * 0.013,
            s * -0.3 + j * 0.007,
            s * 1.1 - j * 0.004,
        )));
    }
    let mut constraints = Vec::new();
    for i in 0..79u32 {
        constraints.push(stretch(i, i + 1, 0.2 + (i as f32) * 0.0007));
    }
    assert_bit_exact(&particles, &constraints, 0.35);
}
