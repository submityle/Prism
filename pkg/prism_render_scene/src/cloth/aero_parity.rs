//! 气动力两段式 `WESL` 内核的**逐位** CPU 转写，对齐架构层黄金
//! [`accumulate_aero_gather`]。
//!
//! `shader_tests` 只做 naga 编译门禁，能证明 `cloth_aerodynamics.wesl` /
//! `cloth_aerodynamics_snapshot.wesl` 解析通过，却**证明不了**它们逐位复刻了
//! CPU 黄金的算术（设计 §9：不造假 parity）。本模块补上这条闭环：把两个内核的
//! 逐条算术**独立**转写成 CPU 版本（不复用 arch 私有的 `turbulence_offset`，
//! 而是照 `WESL` 源码重写一遍），按 host 上传路径打包输入（`vec4` 位置的 `.w`
//! 承载 inverse mass、扁平三角、由 [`VertexTriangleAdjacency`] 生成的 `CSR`），
//! 再对多种网格断言其结果与 [`accumulate_aero_gather`] **逐位（`to_bits`）一致**。
//! 这与虚拟粒子 tier 的 `virtual_particles_jacobi` GPU-twin 金标准同款。
//!
//! ## 逐位一致的关键约定
//! * `WESL` 的 `normal = cross * (1.0 / sqrt(cross_len_sq))` 与黄金
//!   `Vec3::normalize_or_zero` 的 `self.scale(1.0 / len_sq.sqrt())` 逐位一致：
//!   二者都是「先 `sqrt` 再取倒数相除」，`sqrt`/`fdiv` 在 GPU 与 CPU 上均为
//!   IEEE-754 正确舍入，故 GPU 实机输出与黄金也逐位吻合（`cloth_aerodynamics.wesl`
//!   已从 `inverseSqrt` 这一设备级近似改为显式 `1.0 / sqrt(..)`，消除原设备级分歧）。
//! * 速度增量按 `accum * (inv_mass * dt)`（先算标量积）缩放，与黄金
//!   `accum.scale(inverse_mass * dt)` 的结合序一致——`cloth_aerodynamics.wesl`
//!   已同步修正为先算 `inv_mass * dt` 再缩放（浮点乘法不满足结合律）。
//! * `dt` 定义域取有限正值（帧 dt 恒有限正）：黄金对 `!dt.is_finite()` 直接
//!   跳过、`WESL` 靠 host 提供有限 dt，二者在有限正 dt 上逐位吻合；`dt <= 0`
//!   的 no-op 分支两侧同样吻合，故一并覆盖。

#![cfg(test)]

use prism_render_architecture::cloth::aero_gather::{
    accumulate_aero_gather, VertexTriangleAdjacency,
};
use prism_render_architecture::cloth::wind::{AeroParams, WindField};
use prism_render_architecture::cloth::{ClothParticle, Vec3};

/// `cloth_aerodynamics.wesl` 里 `GpuClothAeroParams` uniform 的 CPU 镜像。
///
/// 字段承载的是**已 sanitize**的标量（与 `solve_plan` 打包进
/// `GpuClothAeroParams` 的口径一致：风分量有限化、湍流夹到 `0..=1`、drag/lift
/// 夹非负），因此内核里再做的 `max(_, 0)` 是幂等的。
#[derive(Clone, Copy)]
struct WeslAeroParams {
    wind_velocity: [f32; 3],
    turbulence: f32,
    drag: f32,
    lift: f32,
    dt: f32,
    vertex_count: u32,
    /// 流体（空气）密度：`<= 0` 选线性模型，`> 0` 选二次（airspeed²）模型。
    air_density: f32,
}

// ------- [f32; 3] 上的标量算术，逐位镜像 arch `Vec3` 的成员方法 -------

fn v_add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn v_sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn v_scale(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn v_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn v_cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn xyz(v: [f32; 4]) -> [f32; 3] {
    [v[0], v[1], v[2]]
}

// ------------------- `cloth_aerodynamics.wesl` 转写 -------------------

/// `WESL` `hash_to_unit`：三轮「乘常数 + 右移异或」整数雪崩，取高 24 位映射到
/// `[-1, 1]`。`u32` 乘法为 wrapping，与 shader 一致。
fn hash_to_unit(seed: u32) -> f32 {
    let mut h = seed.wrapping_mul(0x9E37_79B1);
    h ^= h >> 15;
    h = h.wrapping_mul(0x85EB_CA77);
    h ^= h >> 13;
    h = h.wrapping_mul(0xC2B2_AE3D);
    h ^= h >> 16;
    let unit = (h >> 8) as f32 * (1.0 / 16_777_216.0);
    unit * 2.0 - 1.0
}

/// `WESL` `turbulence_offset`：`turb <= 0` 返回零；否则用三个大素数把顶点索引
/// 混成基种子，再逐轴异或掩码哈希出 `[-1, 1]` 分量整体乘 `turb`。
fn turbulence_offset(i0: u32, i1: u32, i2: u32, turb: f32) -> [f32; 3] {
    if turb <= 0.0 {
        return [0.0; 3];
    }
    let base =
        i0.wrapping_mul(73_856_093) ^ i1.wrapping_mul(19_349_663) ^ i2.wrapping_mul(83_492_791);
    v_scale(
        [
            hash_to_unit(base ^ 0x00A5_5A00),
            hash_to_unit(base ^ 0x5A00_00A5),
            hash_to_unit(base ^ 0x00FF_00FF),
        ],
        turb,
    )
}

/// `WESL` `triangle_wind_force`：叉积求法向与面积，退化三角返回零；相对风速拆
/// 法向（乘 drag）+ 切向（乘 lift）再乘面积。归一化用显式 `1.0 / sqrt(..)`。
fn triangle_wind_force(
    p0: [f32; 3],
    p1: [f32; 3],
    p2: [f32; 3],
    v0: [f32; 3],
    v1: [f32; 3],
    v2: [f32; 3],
    wind: [f32; 3],
    drag: f32,
    lift: f32,
    air_density: f32,
) -> [f32; 3] {
    const CLOTH_EPS_LEN_SQ: f32 = 1.0e-12;
    let cross_vec = v_cross(v_sub(p1, p0), v_sub(p2, p0));
    let cross_len_sq = v_dot(cross_vec, cross_vec);
    if cross_len_sq <= CLOTH_EPS_LEN_SQ {
        return [0.0; 3];
    }
    let area = 0.5 * cross_len_sq.sqrt();
    let normal = v_scale(cross_vec, 1.0 / cross_len_sq.sqrt());
    let face_vel = v_scale(v_add(v_add(v0, v1), v2), 1.0 / 3.0);
    let relative = v_sub(wind, face_vel);
    let normal_comp = v_scale(normal, v_dot(relative, normal));
    let tangent = v_sub(relative, normal_comp);
    let directional = v_add(v_scale(normal_comp, drag), v_scale(tangent, lift));
    // 镜像 `WESL`：线性模型（`air_density <= 0`）用面积；二次模型（`> 0`）再乘
    // 动压因子 `0.5 * air_density * sqrt(dot(relative, relative))`，乘序逐位一致。
    let pressure = if air_density > 0.0 {
        area * (0.5 * air_density * v_dot(relative, relative).sqrt())
    } else {
        area
    };
    v_scale(directional, pressure)
}

/// `cloth_aerodynamics_snapshot.wesl` 转写：整槽（含 `.w` 载荷位）逐位拷贝。
fn aero_snapshot(velocities: &[[f32; 4]]) -> Vec<[f32; 4]> {
    velocities.to_vec()
}

/// `cloth_aerodynamics.wesl` gather 主内核转写：一 invocation 一顶点，越界 / pin
/// (`inv_mass <= 0`) / `!(dt > 0)` 跳过；沿 `CSR` 遍历一环邻接三角，摊 `1/3`
/// 累加风力，最后按 `accum * (inv_mass * dt)` 写回自身速度、`.w` 不变。
fn aero_gather(
    positions: &[[f32; 4]],
    velocities: &mut [[f32; 4]],
    snapshot: &[[f32; 4]],
    triangles: &[u32],
    csr_offsets: &[u32],
    csr_entries: &[u32],
    params: &WeslAeroParams,
) {
    for v in 0..params.vertex_count as usize {
        let inv_mass = positions[v][3];
        if inv_mass <= 0.0 {
            continue;
        }
        let dt = params.dt;
        #[expect(
            clippy::neg_cmp_op_on_partial_ord,
            reason = "逐位镜像 WESL 内核的 `!(dt > 0.0)`：NaN dt 必须命中提前返回，改用 partial_cmp 会改变 NaN 语义、破坏与 GPU 的一致性。"
        )]
        let dt_is_non_positive = !(dt > 0.0);
        if dt_is_non_positive {
            continue;
        }
        let drag = params.drag.max(0.0);
        let lift = params.lift.max(0.0);
        let air_density = params.air_density.max(0.0);
        let mut accum = [0.0f32; 3];
        let start = csr_offsets[v];
        let end = csr_offsets[v + 1];
        let mut e = start;
        while e < end {
            let t = csr_entries[e as usize] as usize;
            let b = t * 3;
            let i0 = triangles[b];
            let i1 = triangles[b + 1];
            let i2 = triangles[b + 2];
            let toff = turbulence_offset(i0, i1, i2, params.turbulence);
            let wind_vec = v_add(params.wind_velocity, toff);
            let force = triangle_wind_force(
                xyz(positions[i0 as usize]),
                xyz(positions[i1 as usize]),
                xyz(positions[i2 as usize]),
                xyz(snapshot[i0 as usize]),
                xyz(snapshot[i1 as usize]),
                xyz(snapshot[i2 as usize]),
                wind_vec,
                drag,
                lift,
                air_density,
            );
            accum = v_add(accum, v_scale(force, 1.0 / 3.0));
            e += 1;
        }
        let mass_dt = inv_mass * dt;
        let delta = v_scale(accum, mass_dt);
        velocities[v][0] += delta[0];
        velocities[v][1] += delta[1];
        velocities[v][2] += delta[2];
    }
}

// ------------------------- parity 测试脚手架 -------------------------

/// 一块布料 parity 用例的输入：粒子（位置 + 速度 + inverse mass）与三角拓扑。
struct Case {
    particles: Vec<ClothParticle>,
    triangles: Vec<[u32; 3]>,
}

/// 跑黄金 [`accumulate_aero_gather`] 与 `WESL` 转写两条路径，断言结果速度**逐位**
/// 一致（并校验 `.w` 载荷位原样保留）。
fn assert_bit_exact(case: &Case, wind: WindField, aero: AeroParams, dt: f32) {
    let count = case.particles.len();
    let adjacency = VertexTriangleAdjacency::build(count, &case.triangles);

    // 黄金路径：在 ClothParticle 副本上原地累加。
    let mut golden = case.particles.clone();
    accumulate_aero_gather(&mut golden, &case.triangles, &adjacency, &wind, aero, dt);

    // WESL 路径：按 host 上传口径打包 vec4 位置（.w = inverse mass）与 vec4 速度
    // （.w 塞入可辨识的载荷位以验证内核不动它），扁平三角与 CSR。
    let positions: Vec<[f32; 4]> = case
        .particles
        .iter()
        .map(|p| [p.position.x, p.position.y, p.position.z, p.inverse_mass])
        .collect();
    let mut velocities: Vec<[f32; 4]> = case
        .particles
        .iter()
        .enumerate()
        .map(|(i, p)| [p.velocity.x, p.velocity.y, p.velocity.z, (i as f32) + 0.5])
        .collect();
    let flat: Vec<u32> = case
        .triangles
        .iter()
        .flat_map(|t| t.iter().copied())
        .collect();

    // 参数按 solve_plan 口径 sanitize（WindField/AeroParams 的黄金 sanitizer）。
    let field = wind.sanitized();
    let clean = aero.sanitized();
    let params = WeslAeroParams {
        wind_velocity: [field.velocity.x, field.velocity.y, field.velocity.z],
        turbulence: field.turbulence,
        drag: clean.drag,
        lift: clean.lift,
        dt,
        vertex_count: count as u32,
        air_density: clean.air_density,
    };

    let snapshot = aero_snapshot(&velocities);
    aero_gather(
        &positions,
        &mut velocities,
        &snapshot,
        &flat,
        adjacency.offsets(),
        adjacency.entries(),
        &params,
    );

    for (i, g) in golden.iter().enumerate() {
        let w = velocities[i];
        assert_eq!(
            w[0].to_bits(),
            g.velocity.x.to_bits(),
            "vertex {i}: x velocity drifted from golden"
        );
        assert_eq!(
            w[1].to_bits(),
            g.velocity.y.to_bits(),
            "vertex {i}: y velocity drifted from golden"
        );
        assert_eq!(
            w[2].to_bits(),
            g.velocity.z.to_bits(),
            "vertex {i}: z velocity drifted from golden"
        );
        // `.w` 载荷位必须原样保留。
        assert_eq!(
            w[3].to_bits(),
            ((i as f32) + 0.5).to_bits(),
            "vertex {i}: payload .w was mutated"
        );
    }
}

/// 一个带自定义速度的自由粒子。
fn moving(pos: [f32; 3], vel: [f32; 3], inv_mass: f32) -> ClothParticle {
    let mut p = ClothParticle::new(Vec3::new(pos[0], pos[1], pos[2]), inv_mass);
    p.velocity = Vec3::new(vel[0], vel[1], vel[2]);
    p
}

/// 一块两三角的斜置四边形（四粒子、两面共享对角边），速度/质量各异。
fn quad_case() -> Case {
    Case {
        particles: vec![
            moving([0.0, 0.0, 0.0], [0.1, -0.2, 0.05], 1.0),
            moving([1.0, 0.2, 0.0], [-0.05, 0.1, 0.2], 0.75),
            moving([0.0, 0.1, 1.0], [0.2, 0.0, -0.1], 1.5),
            moving([1.0, -0.1, 1.0], [-0.1, 0.15, 0.0], 0.5),
        ],
        triangles: vec![[0, 1, 2], [2, 1, 3]],
    }
}

#[test]
fn single_triangle_matches_golden() {
    let case = Case {
        particles: vec![
            moving([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1.0),
            moving([2.0, 0.0, 0.0], [0.3, 0.0, 0.0], 1.0),
            moving([0.0, 0.0, 2.0], [0.0, 0.1, 0.0], 1.0),
        ],
        triangles: vec![[0, 1, 2]],
    };
    let wind = WindField::new(Vec3::new(4.0, 1.0, -2.0), 0.0);
    let aero = AeroParams::new(1.25, 0.6);
    assert_bit_exact(&case, wind, aero, 1.0 / 60.0);
}

#[test]
fn shared_vertex_quad_matches_golden() {
    let case = quad_case();
    let wind = WindField::new(Vec3::new(3.0, -1.5, 2.0), 0.0);
    let aero = AeroParams::new(0.9, 0.35);
    assert_bit_exact(&case, wind, aero, 1.0 / 90.0);
}

#[test]
fn turbulence_only_matches_golden() {
    let case = quad_case();
    // 无稳态风，只有湍流——仍是驱动力，逐三角确定性抖动。
    let wind = WindField::new(Vec3::ZERO, 0.7);
    let aero = AeroParams::new(1.0, 0.5);
    assert_bit_exact(&case, wind, aero, 1.0 / 60.0);
}

#[test]
fn wind_and_turbulence_matches_golden() {
    let case = quad_case();
    let wind = WindField::new(Vec3::new(2.5, 0.5, -3.0), 0.4);
    let aero = AeroParams::new(1.1, 0.8);
    assert_bit_exact(&case, wind, aero, 1.0 / 120.0);
}

#[test]
fn pinned_vertices_are_skipped_like_golden() {
    let mut case = quad_case();
    // 把 0 号与 3 号 pin 住（inverse_mass = 0）。
    case.particles[0] = ClothParticle::pinned(Vec3::new(0.0, 0.0, 0.0));
    case.particles[3] = ClothParticle::pinned(Vec3::new(1.0, -0.1, 1.0));
    let wind = WindField::new(Vec3::new(3.0, 2.0, 1.0), 0.3);
    let aero = AeroParams::new(1.0, 0.5);
    assert_bit_exact(&case, wind, aero, 1.0 / 60.0);
}

#[test]
fn multiple_wind_directions_match_golden() {
    let case = quad_case();
    let aero = AeroParams::new(1.4, 0.65);
    for w in [
        [5.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [0.0, 0.0, 5.0],
        [-2.0, 3.0, -4.0],
        [1.5, -1.5, 2.5],
    ] {
        let wind = WindField::new(Vec3::new(w[0], w[1], w[2]), 0.2);
        assert_bit_exact(&case, wind, aero, 1.0 / 60.0);
    }
}

#[test]
fn non_positive_dt_is_a_noop_on_both_paths() {
    let case = quad_case();
    let wind = WindField::new(Vec3::new(3.0, 1.0, -2.0), 0.5);
    let aero = AeroParams::new(1.0, 0.5);
    // dt == 0 与 dt < 0 两侧都应保持速度不变。
    assert_bit_exact(&case, wind, aero, 0.0);
    assert_bit_exact(&case, wind, aero, -1.0 / 60.0);
}

#[test]
fn dense_grid_matches_golden_bit_for_bit() {
    // 一块 6×6 顶点的三角化网格，位置/速度/质量用确定性伪随机填充，压满
    // gather 的多面求和顺序，验证在较大共享度下仍逐位一致。
    const W: usize = 6;
    const H: usize = 6;
    let mut particles = Vec::with_capacity(W * H);
    let mut seed: u32 = 0x1234_5678;
    let mut next = || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (seed >> 8) as f32 * (1.0 / 16_777_216.0) * 2.0 - 1.0
    };
    for y in 0..H {
        for x in 0..W {
            let pos = [
                x as f32 + 0.05 * next(),
                0.2 * next(),
                y as f32 + 0.05 * next(),
            ];
            let vel = [0.3 * next(), 0.3 * next(), 0.3 * next()];
            let inv_mass = if (x + y) % 7 == 0 {
                0.0
            } else {
                0.5 + 0.5 * (next() + 1.0)
            };
            particles.push(moving(pos, vel, inv_mass));
        }
    }
    let mut triangles = Vec::new();
    for y in 0..H - 1 {
        for x in 0..W - 1 {
            let i = (y * W + x) as u32;
            let r = i + 1;
            let d = i + W as u32;
            let dr = d + 1;
            triangles.push([i, r, d]);
            triangles.push([d, r, dr]);
        }
    }
    let case = Case {
        particles,
        triangles,
    };
    let wind = WindField::new(Vec3::new(2.0, -1.0, 3.5), 0.45);
    let aero = AeroParams::new(1.2, 0.7);
    assert_bit_exact(&case, wind, aero, 1.0 / 60.0);
}

#[test]
fn quadratic_single_triangle_matches_golden() {
    let case = Case {
        particles: vec![
            moving([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1.0),
            moving([2.0, 0.0, 0.0], [0.3, 0.0, 0.0], 1.0),
            moving([0.0, 0.0, 2.0], [0.0, 0.1, 0.0], 1.0),
        ],
        triangles: vec![[0, 1, 2]],
    };
    let wind = WindField::new(Vec3::new(4.0, 1.0, -2.0), 0.0);
    // 正空气密度选二次模型；黄金侧同样开启，两条路径逐位一致。
    let aero = AeroParams::new(1.25, 0.6).with_air_density(1.225);
    assert_bit_exact(&case, wind, aero, 1.0 / 60.0);
}

#[test]
fn quadratic_shared_vertex_quad_matches_golden() {
    let case = quad_case();
    let wind = WindField::new(Vec3::new(3.0, -1.5, 2.0), 0.4);
    let aero = AeroParams::new(0.9, 0.35).with_air_density(2.5);
    assert_bit_exact(&case, wind, aero, 1.0 / 90.0);
}

#[test]
fn quadratic_dense_grid_matches_golden_bit_for_bit() {
    const W: usize = 6;
    const H: usize = 6;
    let mut particles = Vec::with_capacity(W * H);
    let mut seed: u32 = 0x1234_5678;
    let mut next = || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (seed >> 8) as f32 * (1.0 / 16_777_216.0) * 2.0 - 1.0
    };
    for y in 0..H {
        for x in 0..W {
            let pos = [
                x as f32 + 0.05 * next(),
                0.2 * next(),
                y as f32 + 0.05 * next(),
            ];
            let vel = [0.3 * next(), 0.3 * next(), 0.3 * next()];
            let inv_mass = if (x + y) % 7 == 0 {
                0.0
            } else {
                0.5 + 0.5 * (next() + 1.0)
            };
            particles.push(moving(pos, vel, inv_mass));
        }
    }
    let mut triangles = Vec::new();
    for y in 0..H - 1 {
        for x in 0..W - 1 {
            let i = (y * W + x) as u32;
            let r = i + 1;
            let d = i + W as u32;
            let dr = d + 1;
            triangles.push([i, r, d]);
            triangles.push([d, r, dr]);
        }
    }
    let case = Case {
        particles,
        triangles,
    };
    let wind = WindField::new(Vec3::new(2.0, -1.0, 3.5), 0.45);
    let aero = AeroParams::new(1.2, 0.7).with_air_density(1.8);
    assert_bit_exact(&case, wind, aero, 1.0 / 60.0);
}

#[test]
fn non_positive_density_falls_back_to_linear_on_both_paths() {
    // 负密度必须命中线性分支：与显式线性系数逐位一致（黄金/转写同源）。
    let case = quad_case();
    let wind = WindField::new(Vec3::new(2.5, 0.5, -3.0), 0.4);
    let aero = AeroParams::new(1.1, 0.8).with_air_density(-4.0);
    assert_bit_exact(&case, wind, aero, 1.0 / 120.0);
}
