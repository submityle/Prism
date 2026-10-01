#![cfg(test)]
//! `cloth_collision.wesl` 的 `cloth_body_collision` 内核 **frictionless（shipping）
//! 路径** 的 **逐位** CPU 转写，对齐架构层黄金 [`resolve_body_collisions`]（把每个
//! 自由粒子依序投影出每个解析碰撞体）。
//!
//! ## 为何只覆盖 frictionless 路径
//! 场景层上传口径恒把 body 摩擦 uniform 置零（`GpuClothBodyParams` 无 `friction`
//! 字段），故 shipping 恒走 `mu == 0`：此时内核的 `cloth_damp_tangential_slip`
//! 对每个碰撞体早退、原样返回投影位置，整趟塌缩为「逐碰撞体顺序投影」，与黄金
//! `resolve_body_collisions` 逐位一致。摩擦段（`mu > 0`）的 WESL
//! `cloth_damp_tangential_slip` 与黄金 `apply_coulomb_friction` 采用 **不同的浮点
//! 结合顺序**（WESL：`tan_dir * min(||t||, mu*depth)`；黄金：`t * min(mu*depth/||t||,
//! 1)`），静摩擦锥内 `(t/||t||)*||t||` 与 `t*1` 不逐位相等，故摩擦段 **本就无法**
//! 逐位对齐——这由 `body_collision_gpu_tests` 的带容差（`PARITY_EPS`）比对如实覆盖，
//! 不在此 CPU parity 的射程内。本模块因此只对 shipping 的 frictionless 路径建立
//! 与设备无关的逐位闭环。
//!
//! 本模块逐词把 WESL 算术搬到 CPU（原生 `f32`，不复用黄金的 `Vec3` 方法），再用
//! `f32::to_bits` 逐分量比对黄金。WESL 球面投影的 `inverseSqrt(dist_sq)` 按
//! `1.0 / dist_sq.sqrt()` 转写，对应黄金 `Vec3::normalize_or_zero` 的
//! `scale(1.0 / len_sq.sqrt())`；真机 `rsqrt` 的硬件差异由 `body_collision_gpu_tests`
//! 的带容差比对覆盖。

use prism_render_architecture::cloth::collision::{resolve_body_collisions, BodyCollider};
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
