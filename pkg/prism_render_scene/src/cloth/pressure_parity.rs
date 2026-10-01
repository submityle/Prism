//! 闭合网格压力（体积保持）约束两段式 `WESL` 内核的**逐位** CPU 转写，对齐
//! 架构层黄金 [`project_pressure`]。
//!
//! `shader_tests` 只做 naga 编译门禁，能证明 `cloth_pressure.wesl` 解析通过，却
//! **证明不了**它逐位复刻了 CPU 黄金的算术（设计 §9：不造假 parity）。本模块补上
//! 这条闭环：把 `cloth_pressure_solve` / `cloth_pressure_apply` 两个内核的逐条
//! 算术**独立**转写成 CPU 版本（不复用 arch 私有逻辑，而是照 `WESL` 源码重写一
//! 遍），按 host 上传路径打包输入（`vec4` 位置的 `.w` 承载 inverse mass、扁平三
//! 角），再对多种闭合网格断言其结果与 [`project_pressure`] **逐位（`to_bits`）
//! 一致**。它与真机 `pressure_gpu_tests` 互补：后者证明设备上的真实 GPU parity，
//! 本模块证明算术转写本身与黄金逐位吻合。
//!
//! ## 逐位一致的关键约定
//! * 体积、梯度、分母的累加顺序**严格复刻黄金**：三角按索引序单遍累加，梯度按
//!   `i0 -> i1 -> i2` 的三角内顺序 scatter 到共享顶点，分母按粒子索引序求和。浮点
//!   加法不满足结合律，顺序必须逐字对齐。
//! * `alpha_tilde = compliance / dt^2`、`target_volume = overpressure * rest_volume`
//!   由 host 从**已 sanitize**的黄金 [`PressureParams`] 预计算，乘除结合序与黄金
//!   内部的 `sanitized()` / `target_volume()` 完全一致。
//! * `dt <= 0`、空粒子、空三角三种 no-op 由 host gate 决定是否 dispatch：黄金提前
//!   返回、host 侧不 dispatch，两条路径都保持位置不变。本模块镜像该 gate。
//! * 退化网格（分母 `< EPS_LEN_SQ`）时黄金提前返回（不写回），`WESL` 则解出
//!   `d_lambda = 0` 后 apply 写回 `pos + grad * (w * 0)`。两者在**非 `-0.0`** 的有限
//!   位置分量上逐位一致（`x + 0.0 == x`），故退化 fixture 的坐标一律避开 `-0.0`。

#![cfg(test)]

use prism_render_architecture::cloth::pressure::{project_pressure, PressureParams};
use prism_render_architecture::cloth::{ClothParticle, Compliance, Vec3};

/// 六分之一，tetra 体积与梯度公式里的常数因子，对齐 CPU 黄金 `INV_SIX`。
const INV_SIX: f32 = 1.0 / 6.0;

/// 退化网格分母下限：分母 `<= EPS_LEN_SQ`（塌缩壳体）时 `d_lambda` 取零，更新永不
/// 除零或产生 `NaN`。对齐 CPU 黄金 `EPS_LEN_SQ` 与 `WESL` `CLOTH_PRESSURE_EPS_LEN_SQ`。
const EPS_LEN_SQ: f32 = 1.0e-12;

// ------- [f32; 3] 上的标量算术，逐位镜像 arch `Vec3` 的成员方法 -------

/// 分量和 `a + b`，镜像 `Vec3::add`。
fn v_add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// 均匀缩放 `a * s`，镜像 `Vec3::scale`。
fn v_scale(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// 点积，镜像 `Vec3::dot`（`x*x' + y*y' + z*z'` 的固定结合序）。
fn v_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// 叉积 `a × b`，镜像 `Vec3::cross`。
fn v_cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// 取 `vec4` 的 `xyz`，丢弃承载 inverse mass 的 `.w`。
fn xyz(v: [f32; 4]) -> [f32; 3] {
    [v[0], v[1], v[2]]
}

/// 有效 inverse mass：pinned（`inverse_mass <= 0`）时为零，否则为存储值。镜像黄金
/// `particle_weight` 与 `WESL` `cloth_pressure_weight`。
fn cloth_pressure_weight(inv_mass: f32) -> f32 {
    if inv_mass <= 0.0 {
        0.0
    } else {
        inv_mass
    }
}

/// `cloth_pressure.wesl` 里 `ClothPressureParams` uniform 的 CPU 镜像（扁平 16 字节
/// 块）：两个计数 + host 从已 sanitize 的黄金 [`PressureParams`] 与子步预计算的两个
/// 派生标量。
#[derive(Clone, Copy)]
struct WeslPressureParams {
    /// 粒子数，约束 solve 归约与 apply dispatch 的范围。
    particle_count: u32,
    /// 闭合壳体的三角数。
    triangle_count: u32,
    /// 目标封闭体积 `overpressure * rest_volume`（已 sanitize）。
    target_volume: f32,
    /// XPBD compliance 除以 `dt^2`，即 `compliance / dt^2`。
    alpha_tilde: f32,
}

// --------------------- `cloth_pressure.wesl` 转写 ---------------------

/// `cloth_pressure_solve` 转写：按黄金的固定顺序累加带符号体积与逐顶点梯度，再解出
/// 单一 compliant `d_lambda`。返回逐顶点梯度与 `d_lambda`。
fn pressure_solve_cpu(
    positions: &[[f32; 4]],
    flat_triangles: &[u32],
    params: &WeslPressureParams,
) -> (Vec<[f32; 3]>, f32) {
    let count = params.particle_count as usize;

    // 清零梯度累加器，等价黄金的 `vec![Vec3::ZERO; count]`。
    let mut gradients = vec![[0.0f32; 3]; count];

    // 单遍三角累加带符号体积与逐顶点梯度（越界三角跳过，与黄金一致）。
    let mut volume = 0.0f32;
    let tri_count = params.triangle_count as usize;
    let mut t = 0usize;
    while t < tri_count {
        let base = t * 3;
        let i0 = flat_triangles[base];
        let i1 = flat_triangles[base + 1];
        let i2 = flat_triangles[base + 2];
        t += 1;
        if i0 as usize >= count || i1 as usize >= count || i2 as usize >= count {
            continue;
        }
        let p0 = xyz(positions[i0 as usize]);
        let p1 = xyz(positions[i1 as usize]);
        let p2 = xyz(positions[i2 as usize]);
        volume += v_dot(p0, v_cross(p1, p2));
        gradients[i0 as usize] =
            v_add(gradients[i0 as usize], v_scale(v_cross(p1, p2), INV_SIX));
        gradients[i1 as usize] =
            v_add(gradients[i1 as usize], v_scale(v_cross(p2, p0), INV_SIX));
        gradients[i2 as usize] =
            v_add(gradients[i2 as usize], v_scale(v_cross(p0, p1), INV_SIX));
    }
    volume *= INV_SIX;

    let error = volume - params.target_volume;

    // 分母：`Σ_i w_i |grad_i|^2 + alpha_tilde`（pinned 粒子 w <= 0，贡献为零）。
    let mut denom = 0.0f32;
    let mut k = 0usize;
    while k < count {
        let w = cloth_pressure_weight(positions[k][3]);
        if w > 0.0 {
            let g = gradients[k];
            denom += w * v_dot(g, g);
        }
        k += 1;
    }
    denom += params.alpha_tilde;

    let mut d_lambda = 0.0f32;
    if denom >= EPS_LEN_SQ {
        d_lambda = -error / denom;
    }
    (gradients, d_lambda)
}

/// `cloth_pressure_apply` 转写：把共享的 `d_lambda` 折进每个自由粒子位置，pinned 粒
/// 子（`inverse_mass <= 0`）跳过。`.w`（inverse mass 载荷）原样保留。
fn pressure_apply_cpu(
    positions: &mut [[f32; 4]],
    gradients: &[[f32; 3]],
    d_lambda: f32,
    particle_count: u32,
) {
    let count = particle_count as usize;
    let mut i = 0usize;
    while i < count {
        let w = cloth_pressure_weight(positions[i][3]);
        if w > 0.0 {
            let delta = v_scale(gradients[i], w * d_lambda);
            positions[i][0] += delta[0];
            positions[i][1] += delta[1];
            positions[i][2] += delta[2];
        }
        i += 1;
    }
}

// ------------------------- parity 测试脚手架 -------------------------

/// 一块压力 parity 用例的输入：闭合网格的粒子（位置 + inverse mass）与三角拓扑。
struct Case {
    particles: Vec<ClothParticle>,
    triangles: Vec<[u32; 3]>,
}

/// 跑黄金 [`project_pressure`] 与 `WESL` 转写两条路径，断言结果位置**逐位**一致
/// （并校验 `.w` 承载的 inverse mass 原样保留）。
///
/// host gate 镜像黄金的提前返回：`dt <= 0`、空粒子、空三角时两条路径都不写回。
fn assert_bit_exact(case: &Case, params: PressureParams, dt: f32) {
    let count = case.particles.len();

    // 黄金路径：在 ClothParticle 副本上原地投影。
    let mut golden = case.particles.clone();
    project_pressure(&mut golden, &case.triangles, params, dt);

    // WESL 路径：按 host 上传口径打包 vec4 位置（.w = inverse mass），扁平三角。
    let mut positions: Vec<[f32; 4]> = case
        .particles
        .iter()
        .map(|p| [p.position.x, p.position.y, p.position.z, p.inverse_mass])
        .collect();
    let flat: Vec<u32> = case
        .triangles
        .iter()
        .flat_map(|t| t.iter().copied())
        .collect();

    // host gate：与黄金 `project_pressure` 的提前返回条件逐条对齐。
    let should_dispatch = dt > 0.0 && !case.particles.is_empty() && !case.triangles.is_empty();
    if should_dispatch {
        // 参数按黄金口径 sanitize，派生标量乘除序与黄金 `target_volume()` /
        // `alpha_tilde` 一致。
        let s = params.sanitized();
        let wesl = WeslPressureParams {
            particle_count: count as u32,
            triangle_count: case.triangles.len() as u32,
            target_volume: s.overpressure * s.rest_volume,
            alpha_tilde: s.compliance.value() / (dt * dt),
        };
        let (gradients, d_lambda) = pressure_solve_cpu(&positions, &flat, &wesl);
        pressure_apply_cpu(&mut positions, &gradients, d_lambda, wesl.particle_count);
    }

    for (i, g) in golden.iter().enumerate() {
        let w = positions[i];
        assert_eq!(
            w[0].to_bits(),
            g.position.x.to_bits(),
            "vertex {i}: x position drifted from golden"
        );
        assert_eq!(
            w[1].to_bits(),
            g.position.y.to_bits(),
            "vertex {i}: y position drifted from golden"
        );
        assert_eq!(
            w[2].to_bits(),
            g.position.z.to_bits(),
            "vertex {i}: z position drifted from golden"
        );
        // `.w` 承载的 inverse mass 必须原样保留。
        assert_eq!(
            w[3].to_bits(),
            case.particles[i].inverse_mass.to_bits(),
            "vertex {i}: inverse-mass payload .w was mutated"
        );
    }
}

/// 一个自由（可动）粒子。
fn free(pos: [f32; 3], inv_mass: f32) -> ClothParticle {
    ClothParticle::new(Vec3::new(pos[0], pos[1], pos[2]), inv_mass)
}

/// 一个 pinned（固定）粒子，`inverse_mass == 0`。
fn pinned(pos: [f32; 3]) -> ClothParticle {
    ClothParticle::pinned(Vec3::new(pos[0], pos[1], pos[2]))
}

/// 一个闭合正四面体（4 顶点、4 外向三角），坐标全为非负（无 `-0.0`）。
fn tetra_positions() -> [[f32; 3]; 4] {
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
}

/// 正四面体的 4 个外向三角。
fn tetra_triangles() -> Vec<[u32; 3]> {
    vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 3, 2]]
}

/// 轴对齐单位立方体 `[0,1]^3` 的 8 个角（无 `-0.0`）。
fn cube_positions() -> [[f32; 3]; 8] {
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ]
}

/// 单位立方体的 12 个外向三角。
fn cube_triangles() -> Vec<[u32; 3]> {
    vec![
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [3, 6, 2],
        [3, 7, 6],
        [0, 4, 7],
        [0, 7, 3],
        [1, 2, 6],
        [1, 6, 5],
    ]
}

/// 用确定性标量从位置数组构造一组自由粒子。
fn free_particles(positions: &[[f32; 3]], inv_mass: f32) -> Vec<ClothParticle> {
    positions.iter().map(|&p| free(p, inv_mass)).collect()
}

#[test]
fn tetra_inflation_matches_golden_bit_for_bit() {
    let case = Case {
        particles: free_particles(&tetra_positions(), 1.0),
        triangles: tetra_triangles(),
    };
    // overpressure > 1 充气：解出非零 d_lambda，沿梯度推动顶点。
    let params = PressureParams::new(0.1, 2.0, Compliance::RIGID);
    assert_bit_exact(&case, params, 1.0 / 60.0);
}

#[test]
fn cube_inflation_matches_golden_bit_for_bit() {
    let case = Case {
        particles: free_particles(&cube_positions(), 1.0),
        triangles: cube_triangles(),
    };
    let params = PressureParams::new(1.0, 2.5, Compliance::RIGID);
    assert_bit_exact(&case, params, 1.0 / 60.0);
}

#[test]
fn cube_deflation_matches_golden_bit_for_bit() {
    let case = Case {
        particles: free_particles(&cube_positions(), 1.0),
        triangles: cube_triangles(),
    };
    // overpressure < 1 放气：误差反号，d_lambda 反向推动顶点。
    let params = PressureParams::new(1.0, 0.4, Compliance::RIGID);
    assert_bit_exact(&case, params, 1.0 / 60.0);
}

#[test]
fn pinned_vertices_match_golden_bit_for_bit() {
    // 立方体混合 pinned/free 顶点：pinned 的 w = 0，梯度仍累加但位置不动。
    let positions = cube_positions();
    let mut particles = free_particles(&positions, 1.0);
    particles[0] = pinned(positions[0]);
    particles[3] = pinned(positions[3]);
    let case = Case {
        particles,
        triangles: cube_triangles(),
    };
    let params = PressureParams::new(1.0, 3.0, Compliance::RIGID);
    assert_bit_exact(&case, params, 1.0 / 60.0);
}

#[test]
fn varying_inverse_mass_matches_golden_bit_for_bit() {
    // 每个顶点不同 inverse mass，压满分母的加权求和与 apply 的逐粒子缩放。
    let positions = cube_positions();
    let inv_masses = [0.25f32, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0];
    let particles: Vec<ClothParticle> = positions
        .iter()
        .zip(inv_masses.iter())
        .map(|(&p, &m)| free(p, m))
        .collect();
    let case = Case {
        particles,
        triangles: cube_triangles(),
    };
    let params = PressureParams::new(1.0, 1.8, Compliance(0.01));
    assert_bit_exact(&case, params, 1.0 / 90.0);
}

#[test]
fn varying_compliance_matches_golden_bit_for_bit() {
    let case = Case {
        particles: free_particles(&cube_positions(), 1.0),
        triangles: cube_triangles(),
    };
    for compliance in [0.0f32, 1.0e-6, 1.0e-3, 0.1, 10.0] {
        let params = PressureParams::new(1.0, 2.0, Compliance(compliance));
        assert_bit_exact(&case, params, 1.0 / 60.0);
    }
}

#[test]
fn out_of_range_triangle_matches_golden_bit_for_bit() {
    // 混入越界三角索引：两条路径都应跳过该三角、只累加合法三角。
    let mut triangles = cube_triangles();
    triangles.push([50, 60, 70]);
    triangles.push([0, 99, 1]);
    let case = Case {
        particles: free_particles(&cube_positions(), 1.0),
        triangles,
    };
    let params = PressureParams::new(1.0, 2.0, Compliance::RIGID);
    assert_bit_exact(&case, params, 1.0 / 60.0);
}

#[test]
fn collapsed_mesh_is_a_noop_on_both_paths() {
    // 全部顶点塌缩到同一点：梯度全零、分母 = alpha_tilde。RIGID 下 alpha_tilde = 0
    // → 分母 < EPS_LEN_SQ，黄金提前返回；WESL 解出 d_lambda = 0 后 apply 加 0.0。
    // 坐标取正值避开 -0.0，故两条路径逐位一致。
    let collapsed = [[0.5f32, 0.5, 0.5]; 8];
    let case = Case {
        particles: free_particles(&collapsed, 1.0),
        triangles: cube_triangles(),
    };
    let params = PressureParams::new(1.0, 2.0, Compliance::RIGID);
    assert_bit_exact(&case, params, 1.0 / 60.0);
}

#[test]
fn non_positive_dt_is_a_noop_on_both_paths() {
    let case = Case {
        particles: free_particles(&cube_positions(), 1.0),
        triangles: cube_triangles(),
    };
    let params = PressureParams::new(1.0, 2.0, Compliance::RIGID);
    // dt == 0 与 dt < 0 两侧都应保持位置不变。
    assert_bit_exact(&case, params, 0.0);
    assert_bit_exact(&case, params, -1.0 / 60.0);
}

#[test]
fn empty_triangles_is_a_noop_on_both_paths() {
    let case = Case {
        particles: free_particles(&cube_positions(), 1.0),
        triangles: Vec::new(),
    };
    let params = PressureParams::new(1.0, 2.0, Compliance::RIGID);
    assert_bit_exact(&case, params, 1.0 / 60.0);
}

#[test]
fn empty_particles_is_a_noop_on_both_paths() {
    let case = Case {
        particles: Vec::new(),
        triangles: Vec::new(),
    };
    let params = PressureParams::new(1.0, 2.0, Compliance::RIGID);
    assert_bit_exact(&case, params, 1.0 / 60.0);
}

#[test]
fn raw_unsanitized_params_match_golden_bit_for_bit() {
    // 原始未 sanitize 参数（负 rest_volume、负 overpressure、负 compliance）：黄金内
    // 部 sanitize，host 侧同样 sanitize 后预计算派生标量，两条路径逐位一致。
    let case = Case {
        particles: free_particles(&cube_positions(), 1.0),
        triangles: cube_triangles(),
    };
    let params = PressureParams::new(-2.0, -1.5, Compliance(-3.0));
    assert_bit_exact(&case, params, 1.0 / 60.0);
}

#[test]
fn jittered_cube_matches_golden_bit_for_bit() {
    // 用确定性伪随机抖动立方体顶点，压满共享顶点的多面求和顺序与浮点舍入。
    let base = cube_positions();
    let mut seed: u32 = 0x1234_5678;
    let mut next = || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (seed >> 8) as f32 * (1.0 / 16_777_216.0) * 2.0 - 1.0
    };
    let particles: Vec<ClothParticle> = base
        .iter()
        .map(|&p| {
            // 偏移幅度小，确保坐标恒为正、避开 -0.0。
            let pos = [
                p[0] + 0.2 + 0.1 * next(),
                p[1] + 0.2 + 0.1 * next(),
                p[2] + 0.2 + 0.1 * next(),
            ];
            free(pos, 0.5 + 0.5 * (next() + 1.0))
        })
        .collect();
    let case = Case {
        particles,
        triangles: cube_triangles(),
    };
    let params = PressureParams::new(1.3, 1.6, Compliance(0.005));
    assert_bit_exact(&case, params, 1.0 / 60.0);
}
