#![cfg(test)]
//! `cloth_collision.wesl` 的 `cloth_body_collision` 内核的 **逐位** CPU 转写，覆盖
//! **frictionless** 与 **Coulomb 摩擦** 两条路径，分别对齐架构层黄金
//! [`resolve_body_collisions`] 与 [`resolve_body_collisions_with_friction`]（把每个
//! 自由粒子依序投影出每个解析碰撞体，摩擦路径再对帧首切向滑移施加 Coulomb 阻尼）。
//!
//! ## frictionless 路径
//! `mu == 0` 时内核的 `cloth_damp_tangential_slip` 对每个碰撞体早退、原样返回投影
//! 位置，整趟塌缩为「逐碰撞体顺序投影」，与黄金 `resolve_body_collisions` 逐位一致。
//!
//! ## 摩擦路径（`mu > 0`）现已逐位对齐
//! WESL `cloth_damp_tangential_slip` 已把尾段从早期的
//! `tan_dir * min(||t||, mu*depth)` 倒数乘重构，改为 `scale = min(mu*depth/||t||, 1)`
//! 再 `projected - t*scale`，与黄金 `apply_coulomb_friction` 的
//! `scale = (mu*normal_push/||t||).min(1)` + `pos - t*scale` **逐位同序**（同为一次
//! fmul、一次 fdiv、一次 min、一次 fmul）。EPS 阈值亦全等
//! (`CLOTH_COL_EPS_LEN_SQ` = `EPS_FRICTION` = `EPS_LEN_SQ` = `1e-12`)，故摩擦段现与
//! 黄金 CPU 逐位吻合，不再是「仅带容差覆盖」。本模块据此对摩擦路径也建立与设备无关
//! 的逐位闭环，与 `body_collision_gpu_tests` 的真机带容差（`PARITY_EPS`）比对互补。
//!
//! 本模块逐词把 WESL 算术搬到 CPU（原生 `f32`，不复用黄金的 `Vec3` 方法），再用
//! `f32::to_bits` 逐分量比对黄金。WESL 球面投影现用显式 `1.0 / sqrt(dist_sq)`
//! （已从 `inverseSqrt` 这一设备级近似改掉），与黄金 `Vec3::normalize_or_zero`
//! 的 `scale(1.0 / len_sq.sqrt())` 完全同序；`sqrt`/`fdiv` 在 GPU 与 CPU 上均为
//! IEEE-754 正确舍入，故 `body_collision_gpu_tests` 的真机输出与黄金也逐位吻合。

use prism_render_architecture::cloth::collision::{
    resolve_body_collisions, resolve_body_collisions_with_friction, BodyCollider,
};
use prism_render_architecture::cloth::{ClothParticle, Vec3, EPS_LEN_SQ};

/// WESL `cloth_project_out_of_sphere` 的逐位转写：非正半径或球外点原样返回；与球心
/// 重合（`dist_sq <= EPS_LEN_SQ`）时沿 `+Y` 推出一个确定、非 `NaN` 的结果；否则按
/// `1/sqrt(dist_sq)` 单位化径向后推到球面。
fn wesl_project_out_of_sphere(pos: [f32; 3], center: [f32; 3], radius: f32) -> [f32; 3] {
    if radius <= 0.0 {
        return pos;
    }
    let delta = [pos[0] - center[0], pos[1] - center[1], pos[2] - center[2]];
    let dist_sq = delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2];
    if dist_sq >= radius * radius {
        return pos;
    }
    if dist_sq <= EPS_LEN_SQ {
        return [center[0], center[1] + radius, center[2]];
    }
    let inv = 1.0_f32 / dist_sq.sqrt();
    let dir = [delta[0] * inv, delta[1] * inv, delta[2] * inv];
    [
        center[0] + dir[0] * radius,
        center[1] + dir[1] * radius,
        center[2] + dir[2] * radius,
    ]
}

/// WESL `cloth_project_out_of_half_space` 的逐位转写：（近）零法线无定义平面，原样
/// 返回；可行侧（`signed >= 0`）原样返回；否则沿法线移动 `-signed / len_sq` 落到平面。
fn wesl_project_out_of_half_space(pos: [f32; 3], normal: [f32; 3], offset: f32) -> [f32; 3] {
    let len_sq = normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2];
    if len_sq <= EPS_LEN_SQ {
        return pos;
    }
    let signed = (normal[0] * pos[0] + normal[1] * pos[1] + normal[2] * pos[2]) - offset;
    if signed >= 0.0 {
        return pos;
    }
    let t = -signed / len_sq;
    [pos[0] + normal[0] * t, pos[1] + normal[1] * t, pos[2] + normal[2] * t]
}

/// WESL `cloth_closest_point_on_segment` 的逐位转写：零长线段退化为 `p0`；否则投影
/// 参数 `clamp` 到 `[0, 1]`（把胶囊端帽变成半球）。
fn wesl_closest_point_on_segment(p0: [f32; 3], p1: [f32; 3], pos: [f32; 3]) -> [f32; 3] {
    let axis = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
    let len_sq = axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2];
    if len_sq <= EPS_LEN_SQ {
        return p0;
    }
    let dp = [pos[0] - p0[0], pos[1] - p0[1], pos[2] - p0[2]];
    let t = (dp[0] * axis[0] + dp[1] * axis[1] + dp[2] * axis[2]) / len_sq;
    let t = t.clamp(0.0, 1.0);
    [p0[0] + axis[0] * t, p0[1] + axis[1] * t, p0[2] + axis[2] * t]
}

/// WESL `cloth_project_collider` 的逐位转写：镜像 `BodyCollider::project` 的变体分派，
/// 胶囊先取轴上最近点再按球面投影。
fn wesl_project_collider(col: &BodyCollider, pos: [f32; 3]) -> [f32; 3] {
    match *col {
        BodyCollider::Sphere { center, radius } => {
            wesl_project_out_of_sphere(pos, [center.x, center.y, center.z], radius)
        }
        BodyCollider::Capsule { p0, p1, radius } => {
            let closest = wesl_closest_point_on_segment(
                [p0.x, p0.y, p0.z],
                [p1.x, p1.y, p1.z],
                pos,
            );
            wesl_project_out_of_sphere(pos, closest, radius)
        }
        BodyCollider::HalfSpace { normal, offset } => {
            wesl_project_out_of_half_space(pos, [normal.x, normal.y, normal.z], offset)
        }
    }
}

/// WESL `cloth_body_collision` 在 `friction == 0`（shipping）下整趟派发的逐位转写：
/// 逐粒子（派发顺序无关、无跨粒子依赖）跳过逆质量 `<= 0` 的钉住粒子，其余把当前位置
/// 依序投影出每个碰撞体（后一碰撞体吃前一碰撞体的投影结果）；`mu == 0` 时内核的摩擦段
/// 对每个碰撞体早退，故塌缩为纯顺序投影。
fn wesl_resolve_body(particles: &[ClothParticle], colliders: &[BodyCollider]) -> Vec<[f32; 3]> {
    particles
        .iter()
        .map(|particle| {
            let mut pos = [particle.position.x, particle.position.y, particle.position.z];
            if particle.inverse_mass <= 0.0 {
                return pos;
            }
            for collider in colliders {
                pos = wesl_project_collider(collider, pos);
            }
            pos
        })
        .collect()
}

/// 逐位断言：WESL frictionless 转写整趟与黄金 [`resolve_body_collisions`] 的每个粒子
/// `xyz` 比特完全相同，且逆质量（内核直通字段）绝不被改动。
#[track_caller]
fn assert_body_bit_exact(particles: &[ClothParticle], colliders: &[BodyCollider]) {
    let mut golden = particles.to_vec();
    resolve_body_collisions(&mut golden, colliders);
    let wesl = wesl_resolve_body(particles, colliders);
    assert_eq!(wesl.len(), golden.len());
    for (i, (w, g)) in wesl.iter().zip(golden.iter()).enumerate() {
        assert_eq!(
            w[0].to_bits(),
            g.position.x.to_bits(),
            "particle {i}: x bits diverge: wesl={} golden={}",
            w[0],
            g.position.x
        );
        assert_eq!(
            w[1].to_bits(),
            g.position.y.to_bits(),
            "particle {i}: y bits diverge: wesl={} golden={}",
            w[1],
            g.position.y
        );
        assert_eq!(
            w[2].to_bits(),
            g.position.z.to_bits(),
            "particle {i}: z bits diverge: wesl={} golden={}",
            w[2],
            g.position.z
        );
        assert_eq!(
            g.inverse_mass.to_bits(),
            particles[i].inverse_mass.to_bits(),
            "particle {i}: inverse mass 被投影 pass 改动了"
        );
    }
}

/// 自由粒子（逆质量 `1`）。
fn free(x: f32, y: f32, z: f32) -> ClothParticle {
    ClothParticle::new(Vec3::new(x, y, z), 1.0)
}

#[test]
fn sphere_interior_point_pushed_to_surface_bit_for_bit() {
    let particles = [free(0.3, 0.1, -0.2)];
    let colliders = [BodyCollider::Sphere {
        center: Vec3::new(0.0, 0.0, 0.0),
        radius: 1.0,
    }];
    assert_body_bit_exact(&particles, &colliders);
}

#[test]
fn sphere_exterior_point_untouched_bit_for_bit() {
    let particles = [free(3.0, 0.0, 0.0)];
    let colliders = [BodyCollider::Sphere {
        center: Vec3::new(0.0, 0.0, 0.0),
        radius: 1.0,
    }];
    assert_body_bit_exact(&particles, &colliders);
}

#[test]
fn sphere_center_coincident_escapes_along_plus_y_bit_for_bit() {
    let particles = [free(0.0, 0.0, 0.0)];
    let colliders = [BodyCollider::Sphere {
        center: Vec3::new(0.0, 0.0, 0.0),
        radius: 0.75,
    }];
    assert_body_bit_exact(&particles, &colliders);
}

#[test]
fn sphere_non_positive_radius_is_inert_bit_for_bit() {
    let particles = [free(0.1, 0.0, 0.0)];
    let colliders = [BodyCollider::Sphere {
        center: Vec3::new(0.0, 0.0, 0.0),
        radius: 0.0,
    }];
    assert_body_bit_exact(&particles, &colliders);
}

#[test]
fn half_space_infeasible_point_projected_bit_for_bit() {
    // 法线 +Y、offset 0：y < 0 被推到平面。
    let particles = [free(0.4, -0.9, 0.3)];
    let colliders = [BodyCollider::HalfSpace {
        normal: Vec3::new(0.0, 1.0, 0.0),
        offset: 0.0,
    }];
    assert_body_bit_exact(&particles, &colliders);
}

#[test]
fn half_space_feasible_point_untouched_bit_for_bit() {
    let particles = [free(0.4, 1.5, 0.3)];
    let colliders = [BodyCollider::HalfSpace {
        normal: Vec3::new(0.0, 1.0, 0.0),
        offset: 0.0,
    }];
    assert_body_bit_exact(&particles, &colliders);
}

#[test]
fn half_space_non_unit_normal_bit_for_bit() {
    // 非单位斜法线，correction 按 len_sq 归一。
    let particles = [free(-0.5, -0.5, 0.2)];
    let colliders = [BodyCollider::HalfSpace {
        normal: Vec3::new(0.3, 0.6, -0.2),
        offset: 0.1,
    }];
    assert_body_bit_exact(&particles, &colliders);
}

#[test]
fn half_space_zero_normal_is_inert_bit_for_bit() {
    let particles = [free(0.4, -9.0, 0.3)];
    let colliders = [BodyCollider::HalfSpace {
        normal: Vec3::ZERO,
        offset: 0.0,
    }];
    assert_body_bit_exact(&particles, &colliders);
}

#[test]
fn capsule_cylinder_side_pushed_bit_for_bit() {
    // 轴沿 X，点在圆柱侧内部。
    let particles = [free(0.0, 0.2, 0.1)];
    let colliders = [BodyCollider::Capsule {
        p0: Vec3::new(-1.0, 0.0, 0.0),
        p1: Vec3::new(1.0, 0.0, 0.0),
        radius: 0.5,
    }];
    assert_body_bit_exact(&particles, &colliders);
}

#[test]
fn capsule_end_cap_pushed_bit_for_bit() {
    // 点超出轴端，最近点钳到 p1，端帽作半球投影。
    let particles = [free(1.3, 0.1, 0.05)];
    let colliders = [BodyCollider::Capsule {
        p0: Vec3::new(-1.0, 0.0, 0.0),
        p1: Vec3::new(1.0, 0.0, 0.0),
        radius: 0.5,
    }];
    assert_body_bit_exact(&particles, &colliders);
}

#[test]
fn degenerate_capsule_acts_as_sphere_bit_for_bit() {
    // 零长线段：退化为绕 p0 的球。
    let particles = [free(0.1, 0.1, 0.0)];
    let colliders = [BodyCollider::Capsule {
        p0: Vec3::new(0.0, 0.0, 0.0),
        p1: Vec3::new(0.0, 0.0, 0.0),
        radius: 0.6,
    }];
    assert_body_bit_exact(&particles, &colliders);
}

#[test]
fn pinned_particle_untouched_bit_for_bit() {
    let particles = [ClothParticle::pinned(Vec3::new(0.0, 0.0, 0.0))];
    let colliders = [BodyCollider::Sphere {
        center: Vec3::new(0.0, 0.0, 0.0),
        radius: 1.0,
    }];
    assert_body_bit_exact(&particles, &colliders);
}

#[test]
fn empty_colliders_is_noop_bit_for_bit() {
    let particles = [free(0.2, -0.3, 0.4), free(1.0, 1.0, 1.0)];
    let colliders: [BodyCollider; 0] = [];
    assert_body_bit_exact(&particles, &colliders);
}

#[test]
fn multiple_colliders_apply_in_order_bit_for_bit() {
    // 多碰撞体顺序投影：后者吃前者结果（最后推的赢）。
    let particles = [free(0.0, 0.0, 0.0)];
    let colliders = [
        BodyCollider::Sphere {
            center: Vec3::new(0.0, 0.0, 0.0),
            radius: 1.0,
        },
        BodyCollider::HalfSpace {
            normal: Vec3::new(0.0, 1.0, 0.0),
            offset: 1.5,
        },
        BodyCollider::Capsule {
            p0: Vec3::new(-2.0, 2.0, 0.0),
            p1: Vec3::new(2.0, 2.0, 0.0),
            radius: 0.4,
        },
    ];
    assert_body_bit_exact(&particles, &colliders);
}

#[test]
fn dense_grid_across_workgroups_bit_for_bit() {
    // 跨 workgroup（> 64）逐粒子一致性；混合自由/钉住 + 多碰撞体。
    let colliders = [
        BodyCollider::Sphere {
            center: Vec3::new(0.0, 0.0, 0.0),
            radius: 1.0,
        },
        BodyCollider::HalfSpace {
            normal: Vec3::new(0.1, 1.0, 0.05),
            offset: -0.8,
        },
        BodyCollider::Capsule {
            p0: Vec3::new(-1.5, -0.5, 0.0),
            p1: Vec3::new(1.5, -0.5, 0.2),
            radius: 0.45,
        },
    ];
    let mut particles = Vec::new();
    for i in 0..100u32 {
        let f = i as f32;
        let x = (f * 0.023).sin() * 1.3;
        let y = (f * 0.041).cos() * 1.1;
        let z = (f * 0.017).sin() * 0.8;
        if i % 9 == 0 {
            particles.push(ClothParticle::pinned(Vec3::new(x, y, z)));
        } else {
            particles.push(free(x, y, z));
        }
    }
    assert_body_bit_exact(&particles, &colliders);
}

#[test]
fn jittered_particles_and_colliders_bit_for_bit() {
    let mut state = 0x1234_5678_u32;
    let mut next = || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((state >> 8) as f32 / 16_777_216.0) * 2.0 - 1.0
    };
    for _ in 0..48 {
        let colliders = [
            BodyCollider::Sphere {
                center: Vec3::new(next() * 0.5, next() * 0.5, next() * 0.5),
                radius: 0.3 + next().abs(),
            },
            BodyCollider::HalfSpace {
                normal: Vec3::new(next(), 0.5 + next().abs(), next()),
                offset: next() * 0.5,
            },
            BodyCollider::Capsule {
                p0: Vec3::new(next(), next(), next()),
                p1: Vec3::new(next(), next(), next()),
                radius: 0.2 + next().abs() * 0.5,
            },
        ];
        let mut particles = Vec::new();
        for _ in 0..17 {
            particles.push(free(next() * 2.0, next() * 2.0, next() * 2.0));
        }
        assert_body_bit_exact(&particles, &colliders);
    }
}

// ---------------------------------------------------------------------------
// Coulomb 摩擦路径（`mu > 0`）逐位转写
// ---------------------------------------------------------------------------

/// WESL `cloth_damp_tangential_slip` 的逐位转写：给定帧首位置 `prev`、投影后位置
/// `projected` 与本碰撞体 correction（`projected - before`，模长为推出深度、方向为
/// 外向接触法线），把帧位移 `projected - prev` 分解到法线/切向，并按
/// `scale = min(mu*depth/||t||, 1)` 收缩切向分量：静摩擦锥内 (`mu*depth >= ||t||`)
/// 饱和到 1 全消，动态区按 `mu*depth` 线性收缩。非正 `friction`、(近)零深度或
/// (近)零切向滑移原样返回 `projected`。op 顺序与黄金 `apply_coulomb_friction` 逐位
/// 相同（一次 fmul、一次 fdiv、一次 min、一次 fmul）。
fn wesl_damp_tangential_slip(
    prev: [f32; 3],
    projected: [f32; 3],
    correction: [f32; 3],
    friction: f32,
) -> [f32; 3] {
    if friction <= 0.0 {
        return projected;
    }
    let depth_sq = correction[0] * correction[0]
        + correction[1] * correction[1]
        + correction[2] * correction[2];
    if depth_sq <= EPS_LEN_SQ {
        return projected;
    }
    let depth = depth_sq.sqrt();
    let inv = 1.0_f32 / depth;
    let normal = [correction[0] * inv, correction[1] * inv, correction[2] * inv];
    let disp = [
        projected[0] - prev[0],
        projected[1] - prev[1],
        projected[2] - prev[2],
    ];
    let normal_amount = disp[0] * normal[0] + disp[1] * normal[1] + disp[2] * normal[2];
    let tangential = [
        disp[0] - normal[0] * normal_amount,
        disp[1] - normal[1] * normal_amount,
        disp[2] - normal[2] * normal_amount,
    ];
    let tan_sq = tangential[0] * tangential[0]
        + tangential[1] * tangential[1]
        + tangential[2] * tangential[2];
    if tan_sq <= EPS_LEN_SQ {
        return projected;
    }
    let tan_len = tan_sq.sqrt();
    let scale = (friction * depth / tan_len).min(1.0);
    [
        projected[0] - tangential[0] * scale,
        projected[1] - tangential[1] * scale,
        projected[2] - tangential[2] * scale,
    ]
}

/// WESL `cloth_body_collision` 在 `friction > 0` 下整趟派发的逐位转写：逐粒子（派发
/// 顺序无关、无跨粒子依赖）跳过逆质量 `<= 0` 的钉住粒子，其余把当前位置依序投影出每
/// 个碰撞体（后一碰撞体吃前一碰撞体的结果），并在每个碰撞体后对帧首切向滑移施加
/// Coulomb 摩擦（切向始终相对帧首 `prev` 测量，而非 running 位置）。镜像黄金
/// `resolve_body_collisions_with_friction` 的逐碰撞体施加顺序；`friction` 在读入时
/// `clamp` 到 `0..=1`，`0` 恰好退化为 frictionless 顺序投影。
fn wesl_resolve_body_with_friction(
    particles: &[ClothParticle],
    prev_positions: &[[f32; 3]],
    colliders: &[BodyCollider],
    friction: f32,
) -> Vec<[f32; 3]> {
    let mu = friction.clamp(0.0, 1.0);
    particles
        .iter()
        .enumerate()
        .map(|(i, particle)| {
            let mut pos = [particle.position.x, particle.position.y, particle.position.z];
            if particle.inverse_mass <= 0.0 {
                return pos;
            }
            let prev = prev_positions[i];
            for collider in colliders {
                let before = pos;
                let projected = wesl_project_collider(collider, before);
                let correction = [
                    projected[0] - before[0],
                    projected[1] - before[1],
                    projected[2] - before[2],
                ];
                pos = wesl_damp_tangential_slip(prev, projected, correction, mu);
            }
            pos
        })
        .collect()
}

/// 逐位断言：WESL 摩擦路径转写与黄金 [`resolve_body_collisions_with_friction`] 的每个
/// 粒子 `xyz` 比特完全相同，且逆质量（内核直通字段）绝不被摩擦 pass 改动。`prev` 的
/// 长度必须等于 `particles`（host 恒上传等长 `prev_positions`，无 fallback 分支）。
#[track_caller]
fn assert_body_friction_bit_exact(
    particles: &[ClothParticle],
    prev: &[Vec3],
    colliders: &[BodyCollider],
    friction: f32,
) {
    assert_eq!(prev.len(), particles.len(), "测试须为每个粒子提供帧首位置");
    let mut golden = particles.to_vec();
    resolve_body_collisions_with_friction(&mut golden, prev, colliders, friction);
    let prev_native: Vec<[f32; 3]> = prev.iter().map(|v| [v.x, v.y, v.z]).collect();
    let wesl = wesl_resolve_body_with_friction(particles, &prev_native, colliders, friction);
    assert_eq!(wesl.len(), golden.len());
    for (i, (w, g)) in wesl.iter().zip(golden.iter()).enumerate() {
        assert_eq!(
            w[0].to_bits(),
            g.position.x.to_bits(),
            "particle {i}: x bits diverge: wesl={} golden={}",
            w[0],
            g.position.x
        );
        assert_eq!(
            w[1].to_bits(),
            g.position.y.to_bits(),
            "particle {i}: y bits diverge: wesl={} golden={}",
            w[1],
            g.position.y
        );
        assert_eq!(
            w[2].to_bits(),
            g.position.z.to_bits(),
            "particle {i}: z bits diverge: wesl={} golden={}",
            w[2],
            g.position.z
        );
        assert_eq!(
            g.inverse_mass.to_bits(),
            particles[i].inverse_mass.to_bits(),
            "particle {i}: inverse mass 被摩擦 pass 改动了"
        );
    }
}

#[test]
fn friction_dynamic_slide_shrinks_bit_for_bit() {
    // 球内点被径向推出，帧首到投影存在明显切向滑移；中等 mu 走动态区（scale < 1）。
    let particles = [free(0.3, 0.3, 0.0)];
    let prev = [Vec3::new(0.5, 0.0, 0.0)];
    let colliders = [BodyCollider::Sphere {
        center: Vec3::new(0.0, 0.0, 0.0),
        radius: 1.0,
    }];
    assert_body_friction_bit_exact(&particles, &prev, &colliders, 0.5);
}

#[test]
fn friction_static_cone_locks_slide_bit_for_bit() {
    // 深穿透 + 微小切向滑移 + mu=1：mu*depth >= ||t||，scale 饱和到 1，切向全消。
    let particles = [free(0.0, 0.02, 0.0)];
    let prev = [Vec3::new(0.01, 0.0, 0.0)];
    let colliders = [BodyCollider::Sphere {
        center: Vec3::new(0.0, 0.0, 0.0),
        radius: 1.0,
    }];
    assert_body_friction_bit_exact(&particles, &prev, &colliders, 1.0);
}

#[test]
fn friction_zero_collapses_to_frictionless_bit_for_bit() {
    // mu == 0：黄金走 resolve_body_collisions，WESL damp 对每碰撞体早退，两者同为纯投影。
    let particles = [free(0.3, 0.3, 0.1)];
    let prev = [Vec3::new(0.5, -0.2, 0.0)];
    let colliders = [BodyCollider::Sphere {
        center: Vec3::new(0.0, 0.0, 0.0),
        radius: 1.0,
    }];
    assert_body_friction_bit_exact(&particles, &prev, &colliders, 0.0);
}

#[test]
fn friction_half_space_slide_bit_for_bit() {
    // 半空间推出 + 平面内切向滑移。
    let particles = [free(0.4, -0.9, 0.3)];
    let prev = [Vec3::new(0.1, -0.4, 0.1)];
    let colliders = [BodyCollider::HalfSpace {
        normal: Vec3::new(0.0, 1.0, 0.0),
        offset: 0.0,
    }];
    assert_body_friction_bit_exact(&particles, &prev, &colliders, 0.7);
}

#[test]
fn friction_capsule_slide_bit_for_bit() {
    // 胶囊侧面推出 + 切向滑移。
    let particles = [free(0.0, 0.2, 0.1)];
    let prev = [Vec3::new(0.2, 0.1, 0.0)];
    let colliders = [BodyCollider::Capsule {
        p0: Vec3::new(-1.0, 0.0, 0.0),
        p1: Vec3::new(1.0, 0.0, 0.0),
        radius: 0.5,
    }];
    assert_body_friction_bit_exact(&particles, &prev, &colliders, 0.6);
}

#[test]
fn friction_pinned_particle_untouched_bit_for_bit() {
    let particles = [ClothParticle::pinned(Vec3::new(0.0, 0.0, 0.0))];
    let prev = [Vec3::new(0.3, 0.1, 0.0)];
    let colliders = [BodyCollider::Sphere {
        center: Vec3::new(0.0, 0.0, 0.0),
        radius: 1.0,
    }];
    assert_body_friction_bit_exact(&particles, &prev, &colliders, 0.8);
}

#[test]
fn friction_no_contact_is_noop_bit_for_bit() {
    // 球外点：correction≈0（depth_sq <= EPS），damp 早退，摩擦不生效。
    let particles = [free(3.0, 0.0, 0.0)];
    let prev = [Vec3::new(2.5, 0.4, 0.0)];
    let colliders = [BodyCollider::Sphere {
        center: Vec3::new(0.0, 0.0, 0.0),
        radius: 1.0,
    }];
    assert_body_friction_bit_exact(&particles, &prev, &colliders, 0.9);
}

#[test]
fn friction_multi_collider_per_contact_bit_for_bit() {
    // 多碰撞体：每个碰撞体后施加一次摩擦，切向始终相对帧首 prev 测量。
    let particles = [free(0.0, 0.0, 0.0)];
    let prev = [Vec3::new(0.2, -0.3, 0.1)];
    let colliders = [
        BodyCollider::Sphere {
            center: Vec3::new(0.0, 0.0, 0.0),
            radius: 1.0,
        },
        BodyCollider::HalfSpace {
            normal: Vec3::new(0.0, 1.0, 0.0),
            offset: 1.5,
        },
        BodyCollider::Capsule {
            p0: Vec3::new(-2.0, 2.0, 0.0),
            p1: Vec3::new(2.0, 2.0, 0.0),
            radius: 0.4,
        },
    ];
    assert_body_friction_bit_exact(&particles, &prev, &colliders, 0.5);
}

#[test]
fn friction_jittered_particles_and_colliders_bit_for_bit() {
    // 伪随机扫动：混合自由/钉住粒子、三类碰撞体、多档 mu，逐位闭合摩擦路径。
    let mut state = 0x0bad_f00d_u32;
    let mut next = || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((state >> 8) as f32 / 16_777_216.0) * 2.0 - 1.0
    };
    for round in 0..48 {
        let colliders = [
            BodyCollider::Sphere {
                center: Vec3::new(next() * 0.5, next() * 0.5, next() * 0.5),
                radius: 0.3 + next().abs(),
            },
            BodyCollider::HalfSpace {
                normal: Vec3::new(next(), 0.5 + next().abs(), next()),
                offset: next() * 0.5,
            },
            BodyCollider::Capsule {
                p0: Vec3::new(next(), next(), next()),
                p1: Vec3::new(next(), next(), next()),
                radius: 0.2 + next().abs() * 0.5,
            },
        ];
        let mut particles = Vec::new();
        let mut prev = Vec::new();
        for i in 0..17 {
            let pos = Vec3::new(next() * 2.0, next() * 2.0, next() * 2.0);
            if i % 6 == 0 {
                particles.push(ClothParticle::pinned(pos));
            } else {
                particles.push(free(pos.x, pos.y, pos.z));
            }
            prev.push(Vec3::new(next() * 2.0, next() * 2.0, next() * 2.0));
        }
        let mu = (round as f32 / 47.0).clamp(0.0, 1.0);
        assert_body_friction_bit_exact(&particles, &prev, &colliders, mu);
    }
}
