//! 布料塑性蠕变内核 `cloth_apply_plasticity`（`cloth_plasticity.wesl`）的 **逐位**
//! CPU 转写，对齐架构层黄金 [`apply_plasticity`]。
//!
//! `plasticity_gpu_tests` 只在**有** wgpu 适配器时才真机派发，无头环境整组
//! `skipping` 通过，证明不了内核把黄金的逐边永久形变逐位复刻了（设计 §9：不造假
//! parity）。本模块补上与设备无关的闭环：把内核的逐条判定独立转写成 CPU 版本
//! （不复用 host 的 `pack_constraint`，而是照 `WESL` 源码重写 kind→u32 映射、
//! 参数 sanitize 与蠕变算术），按 host 上传路径打包位置与约束，再对多种边断言
//! 蠕变后的 rest 长度与 [`apply_plasticity`] **逐位（`to_bits`）一致**。
//!
//! ## 逐位一致的关键约定
//! * 拉伸应变 `e = (len - rest) / rest`，`len` 取 `sqrt(dx*dx + dy*dy + dz*dz)`，
//!   逐词镜像黄金 `Vec3::distance == sub().length()`，减法方向 `a - b` 一致。
//! * 屈服带内（`abs(e) <= yield_strain`）、单边约束（`Lra`/`Tether`，kind u32 为
//!   3/4）、退化 rest（`<= EPS_REST = 1e-9`）、越界端点（`a >= count || b >=
//!   count`）四类边 rest 原样不动，逐条镜像黄金 `edge_strain` 的 `None` 守卫与
//!   屈服带早退。
//! * 蠕变：`excess = e - sign(e) * yield_strain`，`new_rest = rest * (1 + creep *
//!   excess)`，先按 `EPS_REST` 下取整；再按残余弹性应变 `residual = (len -
//!   new_rest) / new_rest` 夹持到 `max_strain`：越界时 `new_rest = len / (1 +
//!   sign(residual) * max_strain)`；最终仅当 `new_rest > EPS_REST` 才写回，否则
//!   保留原 rest——与黄金 `if new_rest > EPS_REST` 写回守卫逐位一致。
//! * `sign`/`residual_sign` 默认 `-1.0`，`>= 0.0` 时取 `1.0`，逐词镜像黄金的
//!   `if x >= 0.0 { 1.0 } else { -1.0 }`。
//! * 参数 sanitize（`yield_strain`/`max_strain` 的 `NaN` 或 `< 0` → `0`，
//!   `creep` 的 `NaN` → `0` 否则 `clamp(0, 1)`）与黄金 [`PlasticParams::sanitized`]
//!   一致。
//! * `sqrt` 在 CPU 上按黄金同款 `f32::sqrt` 求值，二者逐位一致；GPU `distance` 的
//!   硬件 `sqrt` 精度差属设备级，由 `plasticity_gpu_tests` 真机覆盖，不在本范畴。

#![cfg(test)]

use prism_render_architecture::cloth::tearing::{apply_plasticity, PlasticParams};
use prism_render_architecture::cloth::Compliance;
use prism_render_architecture::cloth::{ClothParticle, Constraint, ConstraintKind, Vec3};

/// `cloth_plasticity.wesl` 里 `CLOTH_PLASTICITY_KIND_LRA`：单边长程约束 kind 的
/// u32 码，蠕变恒跳过。
const CLOTH_PLASTICITY_KIND_LRA: u32 = 3;
/// `cloth_plasticity.wesl` 里 `CLOTH_PLASTICITY_KIND_TETHER`：单边系绳 kind 的
/// u32 码。
const CLOTH_PLASTICITY_KIND_TETHER: u32 = 4;
/// `cloth_plasticity.wesl` 里 `CLOTH_PLASTICITY_EPS_REST`：退化 rest 的数值地板。
const CLOTH_PLASTICITY_EPS_REST: f32 = 1.0e-9;

/// `cloth_plasticity.wesl` 的 `ClothPlasticityParams` 中被内核读到的三个标量，
/// 经与黄金同款 sanitize 后的值。
#[derive(Clone, Copy)]
struct SanitizedPlastic {
    /// 屈服应变（`>= 0`）。
    yield_strain: f32,
    /// 蠕变分数（`[0, 1]`）。
    creep: f32,
    /// 残余弹性应变上限（`>= 0`）。
    max_strain: f32,
}

/// 把一组粒子按 host 上传布局打包成位置缓冲：`xyz` 为世界位置，`.w` 为 inverse
/// mass（塑性不读，仅凑 16 字节步长），与 `plasticity_gpu_tests` 上传同款。
fn upload_positions(particles: &[ClothParticle]) -> Vec<[f32; 4]> {
    particles
        .iter()
        .map(|p| [p.position.x, p.position.y, p.position.z, p.inverse_mass])
        .collect()
}

/// 把 [`ConstraintKind`] 映射到 host 上传的 u32 kind 码，逐一镜像 `cloth_sim` ABI
/// 常量（`Stretch=0, Bend=1, Shear=2, Lra=3, Tether=4`）。
fn gpu_constraint_kind(kind: ConstraintKind) -> u32 {
    match kind {
        ConstraintKind::Stretch => 0,
        ConstraintKind::Bend => 1,
        ConstraintKind::Shear => 2,
        ConstraintKind::Lra => CLOTH_PLASTICITY_KIND_LRA,
        ConstraintKind::Tether => CLOTH_PLASTICITY_KIND_TETHER,
    }
}

/// `cloth_plasticity.wesl` 里 host 上传前对参数的 sanitize，逐位镜像
/// [`PlasticParams::sanitized`]。
fn sanitize_params(params: PlasticParams) -> SanitizedPlastic {
    let clamp_nonneg = |v: f32| if v.is_nan() || v < 0.0 { 0.0 } else { v };
    SanitizedPlastic {
        yield_strain: clamp_nonneg(params.yield_strain),
        creep: if params.creep.is_nan() {
            0.0
        } else {
            params.creep.clamp(0.0, 1.0)
        },
        max_strain: clamp_nonneg(params.max_strain),
    }
}

/// `cloth_apply_plasticity` 的单边 CPU 转写：回传这条约束蠕变后应存回的 rest 长度
/// （未命中蠕变条件时原样回传入参 `rest_length`）。
fn wesl_apply_plasticity(
    positions: &[[f32; 4]],
    a: u32,
    b: u32,
    rest_length: f32,
    kind: u32,
    params: SanitizedPlastic,
    particle_count: u32,
) -> f32 {
    if kind == CLOTH_PLASTICITY_KIND_LRA || kind == CLOTH_PLASTICITY_KIND_TETHER {
        return rest_length;
    }
    if rest_length <= CLOTH_PLASTICITY_EPS_REST {
        return rest_length;
    }
    if a >= particle_count || b >= particle_count {
        return rest_length;
    }

    let pa = positions[a as usize];
    let pb = positions[b as usize];
    let dx = pa[0] - pb[0];
    let dy = pa[1] - pb[1];
    let dz = pa[2] - pb[2];
    let len = (dx * dx + dy * dy + dz * dz).sqrt();
    let strain = (len - rest_length) / rest_length;
    let magnitude = strain.abs();
    if magnitude <= params.yield_strain {
        return rest_length;
    }

    let sign = if strain >= 0.0 { 1.0 } else { -1.0 };
    let excess = strain - sign * params.yield_strain;

    let mut new_rest = rest_length * (1.0 + params.creep * excess);
    if new_rest <= CLOTH_PLASTICITY_EPS_REST {
        new_rest = CLOTH_PLASTICITY_EPS_REST;
    }

    let residual = (len - new_rest) / new_rest;
    if residual.abs() > params.max_strain {
        let residual_sign = if residual >= 0.0 { 1.0 } else { -1.0 };
        new_rest = len / (1.0 + residual_sign * params.max_strain);
    }

    if new_rest > CLOTH_PLASTICITY_EPS_REST {
        new_rest
    } else {
        rest_length
    }
}

/// 对一组约束断言：WESL 转写蠕变后的逐边 rest 长度与黄金 [`apply_plasticity`]
/// **逐位一致**。
fn assert_bit_exact(particles: &[ClothParticle], constraints: &[Constraint], params: PlasticParams) {
    let mut golden = constraints.to_vec();
    apply_plasticity(&mut golden, particles, params);

    let positions = upload_positions(particles);
    let sanitized = sanitize_params(params);
    let particle_count = particles.len() as u32;

    assert_eq!(golden.len(), constraints.len());
    for (edge, con) in constraints.iter().enumerate() {
        let twin = wesl_apply_plasticity(
            &positions,
            con.a,
            con.b,
            con.rest_length,
            gpu_constraint_kind(con.kind),
            sanitized,
            particle_count,
        );
        let golden_rest = golden[edge].rest_length;
        assert_eq!(
            twin.to_bits(),
            golden_rest.to_bits(),
            "plastic rest diverged at edge {edge}: twin={twin} golden={golden_rest}"
        );
    }
}

/// 自由粒子（单位 inverse mass），便于按欧氏距离控制应变。
fn particle_at(position: Vec3) -> ClothParticle {
    ClothParticle::new(position, 1.0)
}

/// 两端点 `(a, b)`、rest 为 `rest` 的两边织物边（`Stretch`）。
fn stretch(a: u32, b: u32, rest: f32) -> Constraint {
    Constraint::new(a, b, rest, Compliance::RIGID, ConstraintKind::Stretch)
}

/// 一条给定 kind、端点 `(a, b)`、rest 为 `rest` 的约束。
fn edge(a: u32, b: u32, rest: f32, kind: ConstraintKind) -> Constraint {
    Constraint::new(a, b, rest, Compliance::RIGID, kind)
}

/// 中等蠕变、带残余夹持的参数，足以触发全部蠕变与夹持分支。
fn moderate() -> PlasticParams {
    PlasticParams {
        yield_strain: 0.1,
        creep: 0.5,
        max_strain: 0.3,
    }
}

#[test]
fn stretched_edge_creeps_rest_bit_for_bit() {
    // 相距 2.0、rest 1.0 → 应变 1.0 远超 0.1 屈服，正向蠕变加长 rest。
    let particles = [
        particle_at(Vec3::new(0.0, 0.0, 0.0)),
        particle_at(Vec3::new(2.0, 0.0, 0.0)),
    ];
    let constraints = [stretch(0, 1, 1.0)];
    assert_bit_exact(&particles, &constraints, moderate());
}

#[test]
fn compressed_edge_creeps_rest_bit_for_bit() {
    // 相距 0.4、rest 1.0 → 应变 -0.6 超屈服，负向蠕变缩短 rest。
    let particles = [
        particle_at(Vec3::new(0.0, 0.0, 0.0)),
        particle_at(Vec3::new(0.4, 0.0, 0.0)),
    ];
    let constraints = [stretch(0, 1, 1.0)];
    assert_bit_exact(&particles, &constraints, moderate());
}

#[test]
fn within_yield_band_leaves_rest_untouched_bit_for_bit() {
    // 相距 1.05、rest 1.0 → 应变 0.05 <= 0.1 屈服，rest 不动。
    let particles = [
        particle_at(Vec3::new(0.0, 0.0, 0.0)),
        particle_at(Vec3::new(1.05, 0.0, 0.0)),
    ];
    let constraints = [stretch(0, 1, 1.0)];
    assert_bit_exact(&particles, &constraints, moderate());
}

#[test]
fn lra_leash_never_creeps_bit_for_bit() {
    // 单边 LRA：即便应变巨大也恒跳过。
    let particles = [
        particle_at(Vec3::new(0.0, 0.0, 0.0)),
        particle_at(Vec3::new(5.0, 0.0, 0.0)),
    ];
    let constraints = [edge(0, 1, 1.0, ConstraintKind::Lra)];
    assert_bit_exact(&particles, &constraints, moderate());
}

#[test]
fn tether_leash_never_creeps_bit_for_bit() {
    let particles = [
        particle_at(Vec3::new(0.0, 0.0, 0.0)),
        particle_at(Vec3::new(5.0, 0.0, 0.0)),
    ];
    let constraints = [edge(0, 1, 1.0, ConstraintKind::Tether)];
    assert_bit_exact(&particles, &constraints, moderate());
}

#[test]
fn degenerate_rest_length_leaves_rest_untouched_bit_for_bit() {
    // rest <= EPS_REST 退化，跳过以免除零。
    let particles = [
        particle_at(Vec3::new(0.0, 0.0, 0.0)),
        particle_at(Vec3::new(1.0, 0.0, 0.0)),
    ];
    let constraints = [stretch(0, 1, 1.0e-12)];
    assert_bit_exact(&particles, &constraints, moderate());
}

#[test]
fn out_of_range_endpoint_leaves_rest_untouched_bit_for_bit() {
    // 端点 b=5 越界（只有 2 个粒子），跳过。
    let particles = [
        particle_at(Vec3::new(0.0, 0.0, 0.0)),
        particle_at(Vec3::new(3.0, 0.0, 0.0)),
    ];
    let constraints = [stretch(0, 5, 1.0)];
    assert_bit_exact(&particles, &constraints, moderate());
}

#[test]
fn aggressive_creep_residual_clamped_to_max_strain_bit_for_bit() {
    // creep=1、max_strain=0.1：蠕变会大幅放松，残余应变被夹持到 0.1。
    let particles = [
        particle_at(Vec3::new(0.0, 0.0, 0.0)),
        particle_at(Vec3::new(3.0, 0.0, 0.0)),
    ];
    let constraints = [stretch(0, 1, 1.0)];
    let params = PlasticParams {
        yield_strain: 0.1,
        creep: 1.0,
        max_strain: 0.1,
    };
    assert_bit_exact(&particles, &constraints, params);
}

#[test]
fn zero_creep_is_noop_bit_for_bit() {
    // creep=0：excess 乘 0，new_rest == rest，残余应变等于原应变；
    // 若原应变超 max_strain 仍会被夹持——与黄金逐位一致即可。
    let particles = [
        particle_at(Vec3::new(0.0, 0.0, 0.0)),
        particle_at(Vec3::new(2.0, 0.0, 0.0)),
    ];
    let constraints = [stretch(0, 1, 1.0)];
    let params = PlasticParams {
        yield_strain: 0.1,
        creep: 0.0,
        max_strain: 10.0,
    };
    assert_bit_exact(&particles, &constraints, params);
}

#[test]
fn nan_params_are_sanitized_bit_for_bit() {
    // yield/creep/max 全 NaN → sanitize 到 0/0/0：max_strain=0 会把残余夹到 0。
    let particles = [
        particle_at(Vec3::new(0.0, 0.0, 0.0)),
        particle_at(Vec3::new(2.0, 0.0, 0.0)),
    ];
    let constraints = [stretch(0, 1, 1.0)];
    let params = PlasticParams {
        yield_strain: f32::NAN,
        creep: f32::NAN,
        max_strain: f32::NAN,
    };
    assert_bit_exact(&particles, &constraints, params);
}

#[test]
fn negative_params_are_sanitized_bit_for_bit() {
    // 负 yield/creep/max → sanitize：yield/max → 0，creep clamp 到 0。
    let particles = [
        particle_at(Vec3::new(0.0, 0.0, 0.0)),
        particle_at(Vec3::new(2.0, 0.0, 0.0)),
    ];
    let constraints = [stretch(0, 1, 1.0)];
    let params = PlasticParams {
        yield_strain: -0.5,
        creep: -2.0,
        max_strain: -1.0,
    };
    assert_bit_exact(&particles, &constraints, params);
}

#[test]
fn default_params_match_golden_bit_for_bit() {
    let particles = [
        particle_at(Vec3::new(0.0, 0.0, 0.0)),
        particle_at(Vec3::new(1.5, 0.0, 0.0)),
    ];
    let constraints = [stretch(0, 1, 1.0)];
    assert_bit_exact(&particles, &constraints, PlasticParams::default());
}

#[test]
fn mixed_graph_across_workgroups_matches_golden_bit_for_bit() {
    // 跨越 64 线程 workgroup 边界的混合图：拉伸/压缩/屈服带内/单边/退化/越界
    // 交错，验证逐边自洽（每条只改自身 rest，无写冲突）仍逐位一致。
    let mut particles = Vec::new();
    for i in 0..80u32 {
        let x = i as f32 * 0.37;
        let y = (i as f32 * 0.11).sin() * 0.5;
        let z = (i as f32 * 0.07).cos() * 0.5;
        particles.push(particle_at(Vec3::new(x, y, z)));
    }
    let mut constraints = Vec::new();
    for i in 0..79u32 {
        let kind = match i % 5 {
            0 => ConstraintKind::Stretch,
            1 => ConstraintKind::Bend,
            2 => ConstraintKind::Shear,
            3 => ConstraintKind::Lra,
            _ => ConstraintKind::Tether,
        };
        // rest 刻意偏离实际距离以制造超屈服应变；个别设成退化/越界。
        let rest = if i % 13 == 0 { 1.0e-12 } else { 0.2 + (i % 3) as f32 * 0.15 };
        let b = if i % 17 == 0 { 999 } else { i + 1 };
        constraints.push(edge(i, b, rest, kind));
    }
    assert_bit_exact(&particles, &constraints, moderate());
}

#[test]
fn jittered_positions_match_golden_bit_for_bit() {
    // 非轴对齐的 3D 抖动位型，验证 distance 的三轴平方和开方逐位复刻黄金。
    let particles = [
        particle_at(Vec3::new(0.013, -0.047, 0.021)),
        particle_at(Vec3::new(1.737, 0.902, -0.613)),
        particle_at(Vec3::new(-0.411, 1.228, 0.774)),
    ];
    let constraints = [
        stretch(0, 1, 0.5),
        stretch(1, 2, 0.7),
        stretch(2, 0, 1.3),
    ];
    assert_bit_exact(&particles, &constraints, moderate());
}
