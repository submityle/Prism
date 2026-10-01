#![cfg(test)]
//! `cloth_ccd.wesl` 的 `cloth_resolve_ccd` 内核的 **逐位** CPU 转写，对齐架构层
//! 黄金 [`resolve_ccd`]（逐粒子连续碰撞 TOI 扫掠 + 恢复 + 位置级库仑摩擦）。
//!
//! CCD 内核每条调用只读写自己的 prev/curr/vel、以只读方式遍历碰撞体集合，
//! 无归约、无跨粒子依赖，故逐调用内核在任意派发顺序下都与黄金（按索引顺序）
//! 逐位一致。本模块把 WESL 的三类闭式 TOI 求解器（sphere=一次二次方程、
//! half-space=线性、capsule=无限圆柱板 ∪ 两端帽球）、最早命中选择、恢复反射与
//! 库仑摩擦逐词搬到 CPU（原生 `f32`，不复用黄金方法），再用 `f32::to_bits`
//! 逐分量比对黄金，证明 WESL 内核与黄金 **逐位** 一致。
//!
//! ⚠️ 只喂 host 实际上传的「已清洗值域」：有限 `skin`/`restitution`/`dt`/
//! `friction`。黄金 `CcdParams::sanitized` 会把 NaN 映射为安全值，但 host 在
//! 上传前已清洗，NaN 不会抵达内核，故本 parity 刻意不测 NaN（测了反而偏离
//! 真实派发契约）。真机 GPU 的 `rsqrt` 硬件差异由 `ccd_gpu_tests` 带容差覆盖；
//! 本 CPU parity 建立的是与设备无关的算法等价闭环。

use prism_render_architecture::cloth::ccd::{resolve_ccd, CcdParams};
use prism_render_architecture::cloth::collision::BodyCollider;
use prism_render_architecture::cloth::{ClothParticle, Vec3, EPS_LEN_SQ};

/// 标量系数判零下限，镜像黄金 `ccd::EPS_COEF`。
const EPS_COEF: f32 = 1e-12;
/// 切向滑移判零下限，镜像黄金 `collision::EPS_FRICTION`。
const EPS_FRICTION: f32 = 1e-12;
/// 无界区间端点的有限替身，镜像 WESL `CLOTH_CCD_INF`（任意有限 TOI 都落在其内，
/// 故与黄金的 ±∞ 产生逐位相同的 min/max 交集结果）。
const CCD_INF: f32 = 3.4e38;

/// 碰撞体变体标签，镜像 WESL `CLOTH_COLLIDER_*`。
const KIND_SPHERE: u32 = 0;
const KIND_CAPSULE: u32 = 1;
const KIND_HALF_SPACE: u32 = 2;

/// WESL `ClothCollider` 的 CPU 镜像（kind + 两个点 + 半径/偏移）。
#[derive(Clone, Copy)]
struct WeslCollider {
    kind: u32,
    a: [f32; 3],
    b: [f32; 3],
    radius: f32,
}

impl WeslCollider {
    fn from_body(c: BodyCollider) -> Self {
        match c {
            BodyCollider::Sphere { center, radius } => Self {
                kind: KIND_SPHERE,
                a: v(center),
                b: [0.0; 3],
                radius,
            },
            BodyCollider::Capsule { p0, p1, radius } => Self {
                kind: KIND_CAPSULE,
                a: v(p0),
                b: v(p1),
                radius,
            },
            BodyCollider::HalfSpace { normal, offset } => Self {
                kind: KIND_HALF_SPACE,
                a: v(normal),
                b: [0.0; 3],
                radius: offset,
            },
        }
    }
}

/// 最早命中结果，镜像 WESL `ClothCcdToi`。
#[derive(Clone, Copy)]
struct Toi {
    hit: bool,
    t: f32,
}

const MISS: Toi = Toi { hit: false, t: 0.0 };

fn v(p: Vec3) -> [f32; 3] {
    [p.x, p.y, p.z]
}
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn scale(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// WESL `cloth_ccd_normalize_or_zero`：`1/sqrt(len_sq)` 形式（非硬件 rsqrt）。
fn normalize_or_zero(v: [f32; 3]) -> [f32; 3] {
    let len_sq = dot(v, v);
    if len_sq > EPS_LEN_SQ {
        scale(v, 1.0 / len_sq.sqrt())
    } else {
        [0.0, 0.0, 0.0]
    }
}

/// WESL `cloth_ccd_project_out_of_sphere`。
fn project_out_of_sphere(pos: [f32; 3], center: [f32; 3], radius: f32) -> [f32; 3] {
    if radius <= 0.0 {
        return pos;
    }
    let delta = sub(pos, center);
    let dist_sq = dot(delta, delta);
    if dist_sq >= radius * radius {
        return pos;
    }
    if dist_sq <= EPS_LEN_SQ {
        return add(center, [0.0, radius, 0.0]);
    }
    let dir = scale(delta, 1.0 / dist_sq.sqrt());
    add(center, scale(dir, radius))
}

/// WESL `cloth_ccd_project_out_of_half_space`。
fn project_out_of_half_space(pos: [f32; 3], normal: [f32; 3], offset: f32) -> [f32; 3] {
    let len_sq = dot(normal, normal);
    if len_sq <= EPS_LEN_SQ {
        return pos;
    }
    let signed = dot(normal, pos) - offset;
    if signed >= 0.0 {
        return pos;
    }
    let t = -signed / len_sq;
    add(pos, scale(normal, t))
}

/// WESL `cloth_ccd_closest_point_on_segment`。
fn closest_point_on_segment(p0: [f32; 3], p1: [f32; 3], pos: [f32; 3]) -> [f32; 3] {
    let axis = sub(p1, p0);
    let len_sq = dot(axis, axis);
    if len_sq <= EPS_LEN_SQ {
        return p0;
    }
    let t = (dot(sub(pos, p0), axis) / len_sq).clamp(0.0, 1.0);
    add(p0, scale(axis, t))
}

/// WESL `cloth_ccd_project_collider`。
fn project_collider(col: WeslCollider, pos: [f32; 3]) -> [f32; 3] {
    if col.kind == KIND_SPHERE {
        return project_out_of_sphere(pos, col.a, col.radius);
    }
    if col.kind == KIND_CAPSULE {
        let closest = closest_point_on_segment(col.a, col.b, pos);
        return project_out_of_sphere(pos, closest, col.radius);
    }
    project_out_of_half_space(pos, col.a, col.radius)
}

/// WESL `cloth_ccd_first_entry_time`。
fn first_entry_time(a: f32, b: f32, c: f32) -> Toi {
    if c <= 0.0 {
        return Toi { hit: true, t: 0.0 };
    }
    if a <= EPS_COEF {
        if b >= -EPS_COEF {
            return MISS;
        }
        let t = -c / b;
        if t <= 1.0 {
            return Toi {
                hit: true,
                t: t.max(0.0),
            };
        }
        return MISS;
    }
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return MISS;
    }
    let root = disc.sqrt();
    let t = ((-b) - root) / (2.0 * a);
    if (0.0..=1.0).contains(&t) {
        Toi { hit: true, t }
    } else {
        MISS
    }
}

/// WESL `cloth_ccd_sphere_toi`。
fn sphere_toi(prev: [f32; 3], curr: [f32; 3], center: [f32; 3], radius: f32) -> Toi {
    if radius <= 0.0 {
        return MISS;
    }
    let m = sub(curr, prev);
    let e = sub(prev, center);
    let a = dot(m, m);
    let b = 2.0 * dot(e, m);
    let c = dot(e, e) - radius * radius;
    first_entry_time(a, b, c)
}

/// WESL `cloth_ccd_half_space_toi`。
fn half_space_toi(prev: [f32; 3], curr: [f32; 3], normal: [f32; 3], offset: f32) -> Toi {
    if dot(normal, normal) <= EPS_LEN_SQ {
        return MISS;
    }
    let s0 = dot(normal, prev) - offset;
    if s0 <= 0.0 {
        return Toi { hit: true, t: 0.0 };
    }
    let ds = dot(normal, sub(curr, prev));
    if ds >= -EPS_COEF {
        return MISS;
    }
    let t = -s0 / ds;
    if t <= 1.0 {
        Toi {
            hit: true,
            t: t.max(0.0),
        }
    } else {
        MISS
    }
}

/// WESL `cloth_ccd_cylinder_slab_toi`。
fn cylinder_slab_toi(
    prev: [f32; 3],
    curr: [f32; 3],
    p0: [f32; 3],
    axis: [f32; 3],
    radius: f32,
) -> Toi {
    let len = dot(axis, axis).sqrt();
    if len <= EPS_COEF {
        return MISS;
    }
    let u = scale(axis, 1.0 / len);
    let e0 = sub(prev, p0);
    let m = sub(curr, prev);
    let m_u = dot(m, u);
    let e0u = dot(e0, u);

    let a = dot(m, m) - m_u * m_u;
    let b = 2.0 * (dot(e0, m) - e0u * m_u);
    let c = dot(e0, e0) - e0u * e0u - radius * radius;
    let (rad_lo, rad_hi) = if a > EPS_COEF {
        let disc = b * b - 4.0 * a * c;
        if disc < 0.0 {
            return MISS;
        }
        let root = disc.sqrt();
        (((-b) - root) / (2.0 * a), ((-b) + root) / (2.0 * a))
    } else if c <= 0.0 {
        (-CCD_INF, CCD_INF)
    } else {
        return MISS;
    };

    let (ax_lo, ax_hi) = if m_u.abs() > EPS_COEF {
        let t_at_zero = -e0u / m_u;
        let t_at_len = (len - e0u) / m_u;
        (t_at_zero.min(t_at_len), t_at_zero.max(t_at_len))
    } else if (0.0..=len).contains(&e0u) {
        (-CCD_INF, CCD_INF)
    } else {
        return MISS;
    };

    let lo = rad_lo.max(ax_lo).max(0.0);
    let hi = rad_hi.min(ax_hi).min(1.0);
    if lo <= hi {
        Toi { hit: true, t: lo }
    } else {
        MISS
    }
}

/// WESL `cloth_ccd_earliest`。
fn earliest(lhs: Toi, rhs: Toi) -> Toi {
    if lhs.hit && rhs.hit {
        return Toi {
            hit: true,
            t: lhs.t.min(rhs.t),
        };
    }
    if lhs.hit {
        return lhs;
    }
    rhs
}

/// WESL `cloth_ccd_capsule_toi`。
fn capsule_toi(prev: [f32; 3], curr: [f32; 3], p0: [f32; 3], p1: [f32; 3], radius: f32) -> Toi {
    if radius <= 0.0 {
        return MISS;
    }
    let axis = sub(p1, p0);
    let len_sq = dot(axis, axis);
    if len_sq <= EPS_LEN_SQ {
        return sphere_toi(prev, curr, p0, radius);
    }
    let mut best = cylinder_slab_toi(prev, curr, p0, axis, radius);
    best = earliest(best, sphere_toi(prev, curr, p0, radius));
    best = earliest(best, sphere_toi(prev, curr, p1, radius));
    best
}

/// WESL `cloth_ccd_collider_toi`。
fn collider_toi(col: WeslCollider, prev: [f32; 3], curr: [f32; 3]) -> Toi {
    if col.kind == KIND_SPHERE {
        return sphere_toi(prev, curr, col.a, col.radius);
    }
    if col.kind == KIND_CAPSULE {
        return capsule_toi(prev, curr, col.a, col.b, col.radius);
    }
    half_space_toi(prev, curr, col.a, col.radius)
}

/// WESL `cloth_ccd_outward_normal`。返回 `(has, n)`。
fn outward_normal(col: WeslCollider, surf: [f32; 3]) -> (bool, [f32; 3]) {
    let n = if col.kind == KIND_SPHERE {
        normalize_or_zero(sub(surf, col.a))
    } else if col.kind == KIND_CAPSULE {
        let closest = closest_point_on_segment(col.a, col.b, surf);
        normalize_or_zero(sub(surf, closest))
    } else {
        normalize_or_zero(col.a)
    };
    if dot(n, n) <= EPS_LEN_SQ {
        (false, n)
    } else {
        (true, n)
    }
}

/// WESL `cloth_ccd_apply_coulomb_friction`。
fn apply_coulomb_friction(
    pos: [f32; 3],
    prev: [f32; 3],
    normal: [f32; 3],
    normal_push: f32,
    mu: f32,
) -> [f32; 3] {
    if mu <= 0.0 || normal_push <= 0.0 {
        return pos;
    }
    let delta = sub(pos, prev);
    let normal_amount = dot(delta, normal);
    let tangent = sub(delta, scale(normal, normal_amount));
    let tan_len_sq = dot(tangent, tangent);
    if tan_len_sq <= EPS_FRICTION {
        return pos;
    }
    let tan_len = tan_len_sq.sqrt();
    let s = (mu * normal_push / tan_len).min(1.0);
    sub(pos, scale(tangent, s))
}

/// WESL `cloth_resolve_ccd` 单条调用的逐位转写。返回 `(新位置, 新速度)`。
#[expect(
    clippy::too_many_arguments,
    reason = "逐词镜像 WESL 内核的扁平 uniform/storage 入参，保持与着色器一一对应便于复查。"
)]
fn wesl_resolve_ccd(
    prev: [f32; 3],
    curr: [f32; 3],
    inv_mass: f32,
    vel_in: [f32; 3],
    colliders: &[WeslCollider],
    skin_raw: f32,
    restitution_raw: f32,
    dt: f32,
    friction_raw: f32,
) -> ([f32; 3], [f32; 3]) {
    if inv_mass <= 0.0 {
        return (curr, vel_in);
    }
    let motion = sub(curr, prev);
    if dot(motion, motion) <= EPS_LEN_SQ {
        return (curr, vel_in);
    }
    let skin = skin_raw.max(0.0);
    let restitution = restitution_raw.clamp(0.0, 1.0);
    let mu = friction_raw.clamp(0.0, 1.0);
    let inv_dt = if dt.abs() > EPS_COEF { 1.0 / dt } else { 0.0 };

    let mut have = false;
    let mut best_t = 0.0_f32;
    let mut best = colliders[0];
    for &col in colliders {
        let toi = collider_toi(col, prev, curr);
        if toi.hit {
            let take = !have || toi.t < best_t;
            if take {
                best_t = toi.t;
                best = col;
                have = true;
            }
        }
    }
    if !have {
        return (curr, vel_in);
    }

    let contact = add(prev, scale(motion, best_t));
    let surface = project_collider(best, contact);
    let (has, n) = outward_normal(best, surface);
    if has {
        let placed = add(surface, scale(n, skin));
        let vel = scale(sub(placed, prev), inv_dt);
        let vn = dot(vel, n);
        let mut vel_out = vel_in;
        if vn < 0.0 {
            vel_out = sub(vel, scale(n, (1.0 + restitution) * vn));
        }
        let push = dot(sub(placed, curr), n);
        let final_pos = apply_coulomb_friction(placed, prev, n, push, mu);
        (final_pos, vel_out)
    } else {
        (surface, vel_in)
    }
}

/// 单条粒子的初始态：位置、逆质量、速度。
type Init = (Vec3, f32, Vec3);

/// 用黄金 `resolve_ccd` 与 WESL 逐位转写各跑一遍，逐分量（to_bits）比对位置与速度。
#[track_caller]
fn assert_ccd_bit_exact(
    initial: &[Init],
    prev: &[Vec3],
    colliders: &[BodyCollider],
    skin: f32,
    restitution: f32,
    dt: f32,
    friction: f32,
) {
    // 黄金。
    let mut golden: Vec<ClothParticle> = initial
        .iter()
        .map(|&(pos, im, vel)| {
            let mut p = ClothParticle::new(pos, im);
            p.position = pos;
            p.velocity = vel;
            p
        })
        .collect();
    resolve_ccd(
        &mut golden,
        prev,
        colliders,
        CcdParams {
            skin,
            restitution,
            enabled: true,
        },
        dt,
        friction,
    );

    // WESL 镜像。
    let wcol: Vec<WeslCollider> = colliders.iter().map(|&c| WeslCollider::from_body(c)).collect();
    let count = initial.len().min(prev.len());
    for i in 0..count {
        let (pos, im, vel) = initial[i];
        let (wpos, wvel) = wesl_resolve_ccd(
            v(prev[i]),
            v(pos),
            im,
            v(vel),
            &wcol,
            skin,
            restitution,
            dt,
            friction,
        );
        let gp = golden[i].position;
        let gv = golden[i].velocity;
        for (k, (w, g)) in [(wpos[0], gp.x), (wpos[1], gp.y), (wpos[2], gp.z)]
            .into_iter()
            .enumerate()
        {
            assert_eq!(
                w.to_bits(),
                g.to_bits(),
                "particle {i} position comp {k} diverges: wesl={w} golden={g}"
            );
        }
        for (k, (w, g)) in [(wvel[0], gv.x), (wvel[1], gv.y), (wvel[2], gv.z)]
            .into_iter()
            .enumerate()
        {
            assert_eq!(
                w.to_bits(),
                g.to_bits(),
                "particle {i} velocity comp {k} diverges: wesl={w} golden={g}"
            );
        }
    }
}

fn free(pos: Vec3, vel: Vec3) -> Init {
    (pos, 1.0, vel)
}

#[test]
fn sphere_sweep_hit_with_restitution_and_friction_bit_for_bit() {
    let colliders = [BodyCollider::Sphere {
        center: Vec3::new(0.0, 0.0, 0.0),
        radius: 1.0,
    }];
    let prev = [Vec3::new(0.0, 2.0, 0.0)];
    let initial = [free(Vec3::new(0.3, -1.5, 0.1), Vec3::new(0.1, -8.0, 0.05))];
    assert_ccd_bit_exact(&initial, &prev, &colliders, 1e-3, 0.4, 1.0 / 60.0, 0.5);
}

#[test]
fn sphere_moving_away_is_miss_bit_for_bit() {
    let colliders = [BodyCollider::Sphere {
        center: Vec3::new(0.0, 0.0, 0.0),
        radius: 1.0,
    }];
    let prev = [Vec3::new(0.0, 3.0, 0.0)];
    let initial = [free(Vec3::new(0.0, 5.0, 0.0), Vec3::new(0.0, 2.0, 0.0))];
    assert_ccd_bit_exact(&initial, &prev, &colliders, 1e-3, 0.0, 1.0 / 60.0, 0.0);
}

#[test]
fn sphere_center_coincident_fallback_bit_for_bit() {
    // curr 恰在球心内部 → project_out_of_sphere 的 dist_sq<=EPS 走 +Y 退化分支。
    let colliders = [BodyCollider::Sphere {
        center: Vec3::new(0.0, 0.0, 0.0),
        radius: 0.5,
    }];
    let prev = [Vec3::new(0.0, 1.0, 0.0)];
    let initial = [free(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, -3.0, 0.0))];
    assert_ccd_bit_exact(&initial, &prev, &colliders, 2e-3, 0.2, 1.0 / 120.0, 0.3);
}

#[test]
fn half_space_crossing_hit_bit_for_bit() {
    let colliders = [BodyCollider::HalfSpace {
        normal: Vec3::new(0.0, 1.0, 0.0),
        offset: 0.0,
    }];
    let prev = [Vec3::new(0.5, 1.0, -0.2)];
    let initial = [free(Vec3::new(0.6, -0.8, -0.1), Vec3::new(0.3, -9.0, 0.2))];
    assert_ccd_bit_exact(&initial, &prev, &colliders, 1e-3, 0.6, 1.0 / 60.0, 0.4);
}

#[test]
fn half_space_starts_behind_reports_zero_bit_for_bit() {
    let colliders = [BodyCollider::HalfSpace {
        normal: Vec3::new(0.0, 1.0, 0.0),
        offset: 0.0,
    }];
    let prev = [Vec3::new(0.0, -0.5, 0.0)];
    let initial = [free(Vec3::new(0.1, -1.0, 0.0), Vec3::new(0.0, -2.0, 0.0))];
    assert_ccd_bit_exact(&initial, &prev, &colliders, 1e-3, 0.5, 1.0 / 60.0, 0.2);
}

#[test]
fn half_space_moving_away_is_miss_bit_for_bit() {
    let colliders = [BodyCollider::HalfSpace {
        normal: Vec3::new(0.0, 1.0, 0.0),
        offset: 0.0,
    }];
    let prev = [Vec3::new(0.0, 1.0, 0.0)];
    let initial = [free(Vec3::new(0.0, 2.0, 0.0), Vec3::new(0.0, 1.0, 0.0))];
    assert_ccd_bit_exact(&initial, &prev, &colliders, 1e-3, 0.0, 1.0 / 60.0, 0.0);
}

#[test]
fn capsule_cylinder_side_hit_bit_for_bit() {
    let colliders = [BodyCollider::Capsule {
        p0: Vec3::new(-2.0, 0.0, 0.0),
        p1: Vec3::new(2.0, 0.0, 0.0),
        radius: 0.5,
    }];
    let prev = [Vec3::new(0.0, 1.5, 0.0)];
    let initial = [free(Vec3::new(0.1, -0.8, 0.05), Vec3::new(0.0, -7.0, 0.0))];
    assert_ccd_bit_exact(&initial, &prev, &colliders, 1e-3, 0.3, 1.0 / 60.0, 0.6);
}

#[test]
fn capsule_end_cap_hit_bit_for_bit() {
    let colliders = [BodyCollider::Capsule {
        p0: Vec3::new(-2.0, 0.0, 0.0),
        p1: Vec3::new(2.0, 0.0, 0.0),
        radius: 0.5,
    }];
    // 从端帽外侧沿 -X 扫入右端帽球。
    let prev = [Vec3::new(3.5, 0.1, 0.0)];
    let initial = [free(Vec3::new(1.9, 0.05, 0.0), Vec3::new(-9.0, 0.0, 0.0))];
    assert_ccd_bit_exact(&initial, &prev, &colliders, 1e-3, 0.5, 1.0 / 60.0, 0.4);
}

#[test]
fn degenerate_capsule_acts_as_sphere_bit_for_bit() {
    let colliders = [BodyCollider::Capsule {
        p0: Vec3::new(0.0, 0.0, 0.0),
        p1: Vec3::new(0.0, 0.0, 0.0),
        radius: 0.75,
    }];
    let prev = [Vec3::new(0.0, 2.0, 0.0)];
    let initial = [free(Vec3::new(0.2, -1.0, 0.1), Vec3::new(0.0, -8.0, 0.0))];
    assert_ccd_bit_exact(&initial, &prev, &colliders, 1e-3, 0.2, 1.0 / 60.0, 0.5);
}

#[test]
fn pinned_particle_untouched_bit_for_bit() {
    let colliders = [BodyCollider::Sphere {
        center: Vec3::new(0.0, 0.0, 0.0),
        radius: 2.0,
    }];
    let prev = [Vec3::new(0.0, 3.0, 0.0)];
    // inv_mass = 0 → pinned，CCD 必须跳过。
    let initial = [(Vec3::new(0.0, 0.0, 0.0), 0.0, Vec3::new(0.0, -5.0, 0.0))];
    assert_ccd_bit_exact(&initial, &prev, &colliders, 1e-3, 0.5, 1.0 / 60.0, 0.5);
}

#[test]
fn stationary_particle_untouched_bit_for_bit() {
    let colliders = [BodyCollider::Sphere {
        center: Vec3::new(0.0, 0.0, 0.0),
        radius: 2.0,
    }];
    // prev == curr（运动低于 EPS）→ 跳过。
    let p = Vec3::new(0.0, 0.5, 0.0);
    let prev = [p];
    let initial = [free(p, Vec3::new(0.0, 0.0, 0.0))];
    assert_ccd_bit_exact(&initial, &prev, &colliders, 1e-3, 0.5, 1.0 / 60.0, 0.5);
}

#[test]
fn multiple_colliders_earliest_wins_bit_for_bit() {
    let colliders = [
        BodyCollider::Sphere {
            center: Vec3::new(0.0, -3.0, 0.0),
            radius: 1.0,
        },
        BodyCollider::HalfSpace {
            normal: Vec3::new(0.0, 1.0, 0.0),
            offset: 0.0,
        },
        BodyCollider::Sphere {
            center: Vec3::new(0.0, 0.0, 0.0),
            radius: 0.8,
        },
    ];
    let prev = [Vec3::new(0.1, 3.0, 0.0)];
    let initial = [free(Vec3::new(0.0, -4.0, 0.0), Vec3::new(0.1, -9.0, 0.0))];
    assert_ccd_bit_exact(&initial, &prev, &colliders, 1e-3, 0.4, 1.0 / 60.0, 0.5);
}

#[test]
fn grazing_sphere_miss_bit_for_bit() {
    // 水平掠过球体上方，判别式 < 0 → miss。
    let colliders = [BodyCollider::Sphere {
        center: Vec3::new(0.0, 0.0, 0.0),
        radius: 0.5,
    }];
    let prev = [Vec3::new(-3.0, 2.0, 0.0)];
    let initial = [free(Vec3::new(3.0, 2.0, 0.0), Vec3::new(6.0, 0.0, 0.0))];
    assert_ccd_bit_exact(&initial, &prev, &colliders, 1e-3, 0.0, 1.0 / 60.0, 0.0);
}

#[test]
fn perfect_bounce_restitution_one_bit_for_bit() {
    let colliders = [BodyCollider::HalfSpace {
        normal: Vec3::new(0.0, 1.0, 0.0),
        offset: 0.0,
    }];
    let prev = [Vec3::new(0.0, 1.0, 0.0)];
    let initial = [free(Vec3::new(0.0, -0.5, 0.0), Vec3::new(0.0, -6.0, 0.0))];
    assert_ccd_bit_exact(&initial, &prev, &colliders, 1e-3, 1.0, 1.0 / 60.0, 0.0);
}

#[test]
fn friction_static_cone_full_cancel_bit_for_bit() {
    // 高摩擦 + 浅切向滑移 → 静摩擦锥内完全抵消切向。
    let colliders = [BodyCollider::HalfSpace {
        normal: Vec3::new(0.0, 1.0, 0.0),
        offset: 0.0,
    }];
    let prev = [Vec3::new(0.0, 0.5, 0.0)];
    let initial = [free(Vec3::new(0.05, -0.6, 0.03), Vec3::new(0.5, -5.0, 0.3))];
    assert_ccd_bit_exact(&initial, &prev, &colliders, 0.2, 0.2, 1.0 / 60.0, 1.0);
}

#[test]
fn friction_dynamic_regime_bit_for_bit() {
    // 低摩擦 + 大切向滑移 → 动摩擦区仅收缩切向。
    let colliders = [BodyCollider::HalfSpace {
        normal: Vec3::new(0.0, 1.0, 0.0),
        offset: 0.0,
    }];
    let prev = [Vec3::new(0.0, 0.3, 0.0)];
    let initial = [free(Vec3::new(2.0, -0.4, 1.5), Vec3::new(9.0, -4.0, 7.0))];
    assert_ccd_bit_exact(&initial, &prev, &colliders, 1e-3, 0.1, 1.0 / 60.0, 0.15);
}

#[test]
fn zero_dt_leaves_velocity_untouched_bit_for_bit() {
    // dt≈0 → inv_dt=0 → 反射速度为零向量方向项消失（但位置仍 snap）。
    let colliders = [BodyCollider::Sphere {
        center: Vec3::new(0.0, 0.0, 0.0),
        radius: 1.0,
    }];
    let prev = [Vec3::new(0.0, 2.0, 0.0)];
    let initial = [free(Vec3::new(0.0, -0.5, 0.0), Vec3::new(0.0, -8.0, 0.0))];
    assert_ccd_bit_exact(&initial, &prev, &colliders, 1e-3, 0.5, 0.0, 0.5);
}

#[test]
fn tilted_half_space_non_unit_normal_bit_for_bit() {
    // 非单位、倾斜法线：验证按 len_sq 归一的平面几何与法线归一。
    let colliders = [BodyCollider::HalfSpace {
        normal: Vec3::new(0.0, 2.0, 2.0),
        offset: 0.5,
    }];
    let prev = [Vec3::new(0.0, 1.5, 1.5)];
    let initial = [free(Vec3::new(0.1, -0.5, -0.5), Vec3::new(0.3, -6.0, -6.0))];
    assert_ccd_bit_exact(&initial, &prev, &colliders, 2e-3, 0.3, 1.0 / 90.0, 0.4);
}

#[test]
fn mixed_grid_across_workgroups_bit_for_bit() {
    // >64 条粒子跨 64-lane workgroup 边界，混入 pinned / 静止 / 命中 / 未命中。
    let colliders = [
        BodyCollider::Sphere {
            center: Vec3::new(0.0, 0.0, 0.0),
            radius: 1.2,
        },
        BodyCollider::Capsule {
            p0: Vec3::new(-1.0, -2.0, 0.0),
            p1: Vec3::new(1.0, -2.0, 0.0),
            radius: 0.4,
        },
        BodyCollider::HalfSpace {
            normal: Vec3::new(0.0, 1.0, 0.0),
            offset: -3.0,
        },
    ];
    let mut prev = Vec::new();
    let mut initial = Vec::new();
    for i in 0..100_u32 {
        let f = i as f32;
        let px = (f * 0.37).sin() * 3.0;
        let py = 3.0 - (f * 0.11);
        let pz = (f * 0.53).cos() * 2.0;
        let cx = px + (f * 0.19).cos() * 0.5;
        let cy = py - 4.5;
        let cz = pz + (f * 0.23).sin() * 0.5;
        prev.push(Vec3::new(px, py, pz));
        let inv_mass = if i % 13 == 7 { 0.0 } else { 1.0 };
        initial.push((
            Vec3::new(cx, cy, cz),
            inv_mass,
            Vec3::new(cx - px, cy - py, cz - pz),
        ));
    }
    assert_ccd_bit_exact(&initial, &prev, &colliders, 1.5e-3, 0.35, 1.0 / 60.0, 0.45);
}

#[test]
fn jittered_particles_and_colliders_bit_for_bit() {
    let mut state = 0x9e37_79b9_u32;
    let mut next = || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (state >> 8) as f32 / (1_u32 << 24) as f32
    };
    for _ in 0..48 {
        let colliders = [
            BodyCollider::Sphere {
                center: Vec3::new(next() * 2.0 - 1.0, next() * 2.0 - 1.0, next() * 2.0 - 1.0),
                radius: 0.5 + next(),
            },
            BodyCollider::Capsule {
                p0: Vec3::new(next() * 2.0 - 1.0, next() * 2.0 - 1.0, next() * 2.0 - 1.0),
                p1: Vec3::new(next() * 2.0 - 1.0, next() * 2.0 - 1.0, next() * 2.0 - 1.0),
                radius: 0.3 + next() * 0.6,
            },
            BodyCollider::HalfSpace {
                normal: Vec3::new(next() * 2.0 - 1.0, 1.0 + next(), next() * 2.0 - 1.0),
                offset: next() * 2.0 - 1.0,
            },
        ];
        let prev = [Vec3::new(next() * 6.0 - 3.0, next() * 6.0 - 3.0, next() * 6.0 - 3.0)];
        let curr = Vec3::new(next() * 6.0 - 3.0, next() * 6.0 - 3.0, next() * 6.0 - 3.0);
        let vel = Vec3::new(next() * 10.0 - 5.0, next() * 10.0 - 5.0, next() * 10.0 - 5.0);
        let initial = [free(curr, vel)];
        let skin = next() * 3e-3;
        let restitution = next();
        let friction = next();
        assert_ccd_bit_exact(&initial, &prev, &colliders, skin, restitution, 1.0 / 60.0, friction);
    }
}
