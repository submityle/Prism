#![cfg(test)]
//! `cloth_collision.wesl` 的 `cloth_backstop` 内核的 **逐位** CPU 转写，对齐
//! 架构层黄金 [`resolve_backstops`] / [`apply_backstop`](prism_render_architecture::cloth::collision::apply_backstop)（逐顶点绘制式背板约束
//! pass）。
//!
//! GPU 内核对每个粒子单独求值一块绘制背板平面：读取锚点 `origin`、（无需单位化
//! 的）外向法线 `normal` 与允许的后撤距离 `distance`，把法线归一化后算带符号
//! 距离 `s = dot(n, pos - origin)`；当 `s < -distance`（粒子陷到限制面之后）时
//! 沿单位法线把它推回限制面上，否则原样返回。被钉住的粒子（逆质量 `<= 0`）整段
//! 跳过。
//!
//! 本模块逐词把 WESL 算术搬到 CPU（原生 `f32`，不复用黄金的 `Vec3` 方法），再用
//! `f32::to_bits` 逐分量比对黄金，证明 WESL 内核与黄金 **逐位** 一致。WESL 的
//! `inverseSqrt(len_sq)` 在此按 `1.0 / len_sq.sqrt()` 转写——与黄金
//! `Vec3::normalize_or_zero` 的 `scale(1.0 / len_sq.sqrt())` 对应，建立与设备
//! 无关的「算法等价」闭环；真机 GPU 的 `rsqrt` 与 `1/sqrt` 的硬件差异由
//! `backstop_gpu_tests` 的带容差比对覆盖。

use prism_render_architecture::cloth::collision::{resolve_backstops, Backstop};
use prism_render_architecture::cloth::{ClothParticle, Vec3, EPS_LEN_SQ};

/// WESL `cloth_apply_backstop` 的逐位转写。平方长度不超过 `EPS_LEN_SQ` 的（近）零
/// 法线无定义平面，原样返回；否则把法线按 `1/sqrt(len_sq)` 归一化（镜像黄金
/// `Vec3::normalize_or_zero`），当带符号距离落到 `-distance` 之后时沿单位法线推回
/// 限制面。
fn wesl_apply_backstop(pos: [f32; 3], origin: [f32; 3], normal: [f32; 3], distance: f32) -> [f32; 3] {
    let len_sq = normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2];
    if len_sq <= EPS_LEN_SQ {
        return pos;
    }
    let inv = 1.0_f32 / len_sq.sqrt();
    let n = [normal[0] * inv, normal[1] * inv, normal[2] * inv];
    let d = [pos[0] - origin[0], pos[1] - origin[1], pos[2] - origin[2]];
    let s = n[0] * d[0] + n[1] * d[1] + n[2] * d[2];
    let min_s = -distance;
    if s < min_s {
        let k = min_s - s;
        [pos[0] + n[0] * k, pos[1] + n[1] * k, pos[2] + n[2] * k]
    } else {
        pos
    }
}

/// WESL `cloth_backstop` 整趟派发的逐位转写：逐粒子（派发顺序无关、无跨粒子依赖）
/// 跳过逆质量 `<= 0` 的钉住粒子，其余按配对的背板记录推回限制面。返回每个粒子的
/// 结算后位置。
fn wesl_resolve_backstops(particles: &[ClothParticle], backstops: &[Backstop]) -> Vec<[f32; 3]> {
    particles
        .iter()
        .zip(backstops.iter())
        .map(|(particle, backstop)| {
            let pos = [particle.position.x, particle.position.y, particle.position.z];
            if particle.inverse_mass <= 0.0 {
                return pos;
            }
            let origin = [backstop.origin.x, backstop.origin.y, backstop.origin.z];
            let normal = [backstop.normal.x, backstop.normal.y, backstop.normal.z];
            wesl_apply_backstop(pos, origin, normal, backstop.distance)
        })
        .collect()
}

/// 逐位断言：WESL 转写整趟与黄金 [`resolve_backstops`] 的每个粒子 `xyz` 比特完全
/// 相同，且逆质量（内核直通字段）绝不被改动。
#[track_caller]
fn assert_backstop_bit_exact(particles: &[ClothParticle], backstops: &[Backstop]) {
    assert_eq!(
        particles.len(),
        backstops.len(),
        "parity harness 要求粒子与背板等长（GPU 逐粒子索引 backstops[p]）"
    );
    let mut golden = particles.to_vec();
    resolve_backstops(&mut golden, backstops);
    let wesl = wesl_resolve_backstops(particles, backstops);
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
            "particle {i}: inverse mass 被背板 pass 改动了"
        );
    }
}

/// 构造一个自由粒子（逆质量 `1`）。
fn free(x: f32, y: f32, z: f32) -> ClothParticle {
    ClothParticle::new(Vec3::new(x, y, z), 1.0)
}

/// 一块锚在原点、法线沿 `+Y`、允许后撤 `0.5` 的背板。
fn plane_up(distance: f32) -> Backstop {
    Backstop {
        origin: Vec3::new(0.0, 0.0, 0.0),
        normal: Vec3::new(0.0, 1.0, 0.0),
        distance,
    }
}

#[test]
fn in_front_of_plane_is_untouched_bit_for_bit() {
    // s = +2 >= -0.5 ⇒ 不动。
    let particles = [free(0.3, 2.0, -0.7)];
    let backstops = [plane_up(0.5)];
    assert_backstop_bit_exact(&particles, &backstops);
}

#[test]
fn exactly_at_limit_is_untouched_bit_for_bit() {
    // s = -0.5 == min_s ⇒ 条件是严格 `<`，故不动。
    let particles = [free(1.0, -0.5, 2.0)];
    let backstops = [plane_up(0.5)];
    assert_backstop_bit_exact(&particles, &backstops);
}

#[test]
fn behind_plane_is_pushed_forward_bit_for_bit() {
    // s = -1.3 < -0.5 ⇒ 沿 +Y 推回到 y = -0.5。
    let particles = [free(0.4, -1.3, 0.9)];
    let backstops = [plane_up(0.5)];
    assert_backstop_bit_exact(&particles, &backstops);
}

#[test]
fn zero_normal_is_inert_bit_for_bit() {
    let particles = [free(1.0, -5.0, 2.0)];
    let backstops = [Backstop {
        origin: Vec3::new(0.0, 0.0, 0.0),
        normal: Vec3::ZERO,
        distance: 0.25,
    }];
    assert_backstop_bit_exact(&particles, &backstops);
}

#[test]
fn near_zero_normal_below_eps_is_inert_bit_for_bit() {
    // len_sq = 3e-13 <= EPS_LEN_SQ(1e-12) ⇒ 无定义平面，原样返回。
    let tiny = 3.162_277_6e-7_f32; // tiny^2 ≈ 1e-13，三分量合 ~3e-13
    let particles = [free(0.5, -9.0, 0.5)];
    let backstops = [Backstop {
        origin: Vec3::new(0.0, 0.0, 0.0),
        normal: Vec3::new(tiny, tiny, tiny),
        distance: 0.1,
    }];
    assert_backstop_bit_exact(&particles, &backstops);
}

#[test]
fn non_unit_normal_is_normalized_bit_for_bit() {
    // 法线长度 2（非单位）：平面几何按 1/sqrt(len_sq) 恢复。
    let particles = [free(0.2, -1.0, 0.3)];
    let backstops = [Backstop {
        origin: Vec3::new(0.0, 0.0, 0.0),
        normal: Vec3::new(0.0, 2.0, 0.0),
        distance: 0.5,
    }];
    assert_backstop_bit_exact(&particles, &backstops);
}

#[test]
fn tilted_non_unit_normal_pushes_bit_for_bit() {
    // 斜置且非单位的法线，粒子陷在后方 ⇒ 沿单位法线推回。
    let particles = [free(-0.8, -0.9, 0.4)];
    let backstops = [Backstop {
        origin: Vec3::new(0.1, -0.2, 0.05),
        normal: Vec3::new(0.3, 0.7, -0.4),
        distance: 0.2,
    }];
    assert_backstop_bit_exact(&particles, &backstops);
}

#[test]
fn negative_distance_pins_to_front_side_bit_for_bit() {
    // distance = -0.3 ⇒ min_s = +0.3：可行域 s >= 0.3，s = 0.1 < 0.3 ⇒ 推到 s = 0.3。
    let particles = [free(0.6, 0.1, -0.2)];
    let backstops = [plane_up(-0.3)];
    assert_backstop_bit_exact(&particles, &backstops);
}

#[test]
fn pinned_particle_is_untouched_bit_for_bit() {
    // 逆质量 0（钉住）：即便深陷背板之后也绝不移动。
    let particles = [ClothParticle::pinned(Vec3::new(1.0, -4.0, 2.0))];
    let backstops = [plane_up(0.5)];
    assert_backstop_bit_exact(&particles, &backstops);
}

#[test]
fn deep_penetration_large_push_bit_for_bit() {
    let particles = [free(2.5, -37.5, -4.25)];
    let backstops = [plane_up(0.75)];
    assert_backstop_bit_exact(&particles, &backstops);
}

#[test]
fn multiple_particles_each_own_backstop_bit_for_bit() {
    let particles = [
        free(0.0, 3.0, 0.0),            // 前方，不动
        free(0.0, -2.0, 0.0),           // 后方，推回
        ClothParticle::pinned(Vec3::new(0.0, -9.0, 0.0)), // 钉住，不动
        free(-1.0, -0.6, 0.7),          // 斜法线推回
    ];
    let backstops = [
        plane_up(0.5),
        plane_up(0.5),
        plane_up(0.5),
        Backstop {
            origin: Vec3::new(0.0, 0.0, 0.0),
            normal: Vec3::new(0.2, 0.9, -0.1),
            distance: 0.3,
        },
    ];
    assert_backstop_bit_exact(&particles, &backstops);
}

#[test]
fn dense_grid_across_workgroups_bit_for_bit() {
    // 跨 workgroup（> 64）的逐粒子一致性；混合前方/后方/钉住与非单位法线。
    let mut particles = Vec::new();
    let mut backstops = Vec::new();
    for i in 0..100u32 {
        let f = i as f32;
        let x = (f * 0.013).sin() * 1.7;
        let y = (f * 0.07).cos() * 2.5 - 0.4;
        let z = (f * 0.021).sin() * 0.9;
        if i % 7 == 0 {
            particles.push(ClothParticle::pinned(Vec3::new(x, y, z)));
        } else {
            particles.push(free(x, y, z));
        }
        let nx = (f * 0.03).cos() * 0.5;
        let ny = 1.0 + (f * 0.017).sin() * 0.3;
        let nz = (f * 0.05).sin() * 0.4;
        backstops.push(Backstop {
            origin: Vec3::new((f * 0.01).sin() * 0.2, (f * 0.02).cos() * 0.15, 0.0),
            normal: Vec3::new(nx, ny, nz),
            distance: 0.1 + (f * 0.009).sin().abs() * 0.6,
        });
    }
    assert_backstop_bit_exact(&particles, &backstops);
}

#[test]
fn jittered_particles_and_backstops_bit_for_bit() {
    // 伪随机抖动多轮：覆盖前方/后方/退化法线/负距离的组合，逐位必然收敛。
    let mut state = 0x9e37_79b9_u32;
    let mut next = || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((state >> 8) as f32 / 16_777_216.0) * 2.0 - 1.0
    };
    for _ in 0..64 {
        let mut particles = Vec::new();
        let mut backstops = Vec::new();
        for _ in 0..17 {
            let p = free(next() * 3.0, next() * 3.0, next() * 3.0);
            particles.push(p);
            // 偶尔塞一个（近）零法线触发惰性分支。
            let scale = if next() > 0.8 { 1e-7 } else { 1.0 };
            backstops.push(Backstop {
                origin: Vec3::new(next() * 0.5, next() * 0.5, next() * 0.5),
                normal: Vec3::new(next() * scale, (0.5 + next().abs()) * scale, next() * scale),
                distance: next() * 0.8,
            });
        }
        assert_backstop_bit_exact(&particles, &backstops);
    }
}
