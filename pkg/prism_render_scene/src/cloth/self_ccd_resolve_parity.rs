#![cfg(test)]
//! `cloth_self_ccd.wesl` 的 `cloth_ccd_resolve` + `cloth_ccd_apply` 两相位
//! 「接触解算」的 **逐位** CPU 转写，对齐架构层黄金 [`resolve_self_ccd`] 的窄相解算
//! 核心（已命中 TOI 的一对粒子如何被推开、如何交换法向速度冲量）。
//!
//! ## 这里补的是哪块缺口
//! 姊妹文件 `self_ccd_parity` 只逐位覆盖 `cloth_ccd_swept_toi`（TOI 求根数学）。
//! 命中 TOI 之后的 **解算相位** —— 把两粒子吸附到 TOI 快照位置、对法向做
//! `normalize_or_zero` 并在退化（重合）时回退到固定 `+X`、按穿透深度与逆质量对称
//! 推开、再按恢复系数交换入射法向相对速度冲量 —— 此前只由带容差的真机
//! `self_ccd_gpu_tests` 对拍，缺一份与设备无关的逐位黄金对齐。本模块补上它。
//!
//! ## 为何能做逐位 parity（含对称半边论证）
//! GPU 内核是 **每粒子各算自己那半** 的 scatter 布局：`cloth_ccd_resolve` 为粒子
//! `i` 累加 `pos_delta = pos_a - curr_a` 与 `vel_delta = new_va - velocities[i]`，
//! 写入 `ccd_pos_delta[i]`/`ccd_vel_delta[i]`；随后 `cloth_ccd_apply` 对自由粒子做
//! `pos += delta`，对 pinned（`w <= 0`）粒子 **跳过写入**。黄金则在一次 Gauss-Seidel
//! 访问里就地写 `position = pos_a` / `position = pos_b`。
//!
//! 注意：GPU 的 `curr + (pos_a - curr)` 与黄金就地 `position = pos_a` 的 **最终位置**
//! 一般相差亚 ULP（回环加法舍入），**不逐位相等**，故本模块不伪装比对最终位置。
//! 我们逐位比对的是 **内核写出的增量**：对自由粒子，断言 `wesl_resolve_one` 返回的
//! `pos_delta` 恰等于黄金的净变化 `golden_final - curr_in`（因黄金 `golden_final`
//! 是 `pos_a` 的精确拷贝，`fl(pos_a - curr)` 与内核 `pos_delta = fl(pos_a - curr)`
//! 同序同运算 → 逐位）；对 pinned 粒子，`cloth_ccd_apply` 与黄金 `is_pinned()` 守卫
//! 两侧净变化恒为 0。速度同理（黄金 **覆盖** `velocity = new_va`，故净变化
//! `new_va - velocity_in` 与内核 `vel_delta = new_va - velocities[i]` 同构）。
//!
//! ### 隔离纪律
//! 每个场景只放 **两枚粒子**，坐标全部落在 `[0.30, 0.70)` 且 `cell_size = 1.0`，
//! 保证广相把两枚粒子分到同一 cell `(0, 0, 0)`、形成 **唯一** 候选对。单对下
//! Gauss-Seidel 无跨对耦合、与访问序无关，GPU 的每粒子半边与黄金读到的是同一份
//! 未改动 `curr` 快照，输入完全一致。
//!
//! ### 两半边逐位对称（非退化分支）
//! 粒子 `a` 的半边以 `(a 为 self, b 为 other)` 调 [`wesl_resolve_one`]，粒子 `b` 的
//! 半边以 `(b 为 self, a 为 other)` 调一次。记 `delta = a_c - b_c`，则 b 的半边里
//! `delta' = b_c - a_c = -delta`。关键 IEEE 恒等式：
//! `normalize(-x)` 的每个分量 `== -(normalize(x) 的分量)`（`len_sq` 对负向量逐位
//! 相同，`1/sqrt(len_sq)` 相同，`(-v) * s == -(v * s)` 精确）；`dot(-x, -y) == dot(x, y)`
//! （`(-xi) * (-yi) == xi * yi` 精确）。于是 b 半边的 `normal' = -normal`、`vrel_n' = vrel_n`
//! （同号分支）、`penetration' = penetration`、`impulse' = impulse`，而黄金对 b 用的是
//! `b_c - normal * (...)`，与半边的 `b_c + normal' * (...) = b_c + (-normal) * (...)`
//! 逐位相等（`c - n*s` 与 `c + (-n)*s` 在 IEEE 下同值）。故非退化场景两半边均逐位。
//!
//! ### 诚实边界
//! **退化重合**（TOI 处 `a_c == b_c`，`len_sq <= EPS_LEN_SQ`）两侧都回退 `+X`，于是
//! GPU 每粒子半边会让 **双方都朝 +X** 推，而黄金对 b 用 `-normal`（即 `-X`）——这是
//! 零测度退化角上 GPU scatter 设计与黄金就地设计的 **真实分歧**，不可逐位。本模块因此
//! 在退化重合场景里把伙伴设为 pinned（其净变化恒 0），只对自由粒子逐位比对 `+X` 回退
//! 分支；自由-自由重合这一分歧交由真机 `self_ccd_gpu_tests` 的容差对拍互补覆盖。

use prism_render_architecture::cloth::self_ccd::{resolve_self_ccd, SelfCcdParams};
use prism_render_architecture::cloth::{ClothParticle, Vec3};

/// WESL `CLOTH_CCD_EPS_LEN_SQ` 的镜像：法向归一化的平方长度地板。与黄金私有
/// `EPS_LEN_SQ`（`Vec3::normalize_or_zero` 用）同值 `1e-12`。
const WESL_EPS_LEN_SQ: f32 = 1.0e-12;

/// WESL `CLOTH_CCD_EPS_REL_MOTION` 的镜像：相对运动平方地板，也用于 `inv_dt` 守卫。
/// 与黄金私有 `EPS_REL_MOTION` 同值 `1e-12`。
const WESL_EPS_REL_MOTION: f32 = 1.0e-12;

/// 原生 `[f32; 3]` 逐分量差 `a - b`，运算序对齐黄金 `Vec3::sub`。
fn v_sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// 原生 `[f32; 3]` 逐分量和 `a + b`，运算序对齐黄金 `Vec3::add`。
fn v_add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// 原生 `[f32; 3]` 标量缩放 `v * s`，运算序对齐黄金 `Vec3::scale`。
fn v_scale(v: [f32; 3], s: f32) -> [f32; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

/// 原生 `[f32; 3]` 点积，左结合求和序对齐黄金 `Vec3::dot`。
fn v_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// WESL `cloth_ccd_swept_toi` 的自包含逐位转写（与 `self_ccd_parity` 中同构，这里
/// 内联以保持本模块对解算相位的完整自描述）。粒子 `a` 沿 `prev_a -> curr_a`、`b` 沿
/// `prev_b -> curr_b` 在单位帧内线性移动，返回二次方程 `|d0 + t*dv|^2 = thickness^2`
/// 的入射根；`c <= 0` 立即接触返回 `Some(0.0)`，无相对运动 / 判别式为负 / 根越界返回
/// `None`。仅用 `f32::sqrt`，全程比较式早退、确定且无 `NaN`。
fn wesl_swept_toi(
    prev_a: [f32; 3],
    curr_a: [f32; 3],
    prev_b: [f32; 3],
    curr_b: [f32; 3],
    thickness: f32,
) -> Option<f32> {
    let d0 = v_sub(prev_a, prev_b);
    let d1 = v_sub(curr_a, curr_b);
    let dv = v_sub(d1, d0);
    let c = v_dot(d0, d0) - thickness * thickness;
    if c <= 0.0 {
        return Some(0.0);
    }
    let a = v_dot(dv, dv);
    if a <= WESL_EPS_REL_MOTION {
        return None;
    }
    let b = 2.0 * v_dot(d0, dv);
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return None;
    }
    let t = (-b - disc.sqrt()) / (2.0 * a);
    if (0.0..=1.0).contains(&t) {
        Some(t)
    } else {
        None
    }
}

/// 黄金 `inv_dt` 守卫的逐位镜像：`|dt| <= EPS` 时取 `0`，否则 `1/dt`。
fn guarded_inv_dt(dt: f32) -> f32 {
    if dt.abs() <= WESL_EPS_REL_MOTION {
        0.0
    } else {
        1.0 / dt
    }
}

/// `cloth_ccd_resolve` 内核中 **单个粒子（self）对单个伙伴（other）那半** 的逐词镜像。
///
/// 返回 `(pos_delta, vel_delta)`，即内核写入 `ccd_pos_delta[i]` / `ccd_vel_delta[i]`
/// 的增量。`self_w` / `other_w` 为逆质量（内核在读取时 `max(.w, 0.0)`，这里内联同样
/// 的夹取）。TOI 未命中、或 `wsum <= 0`（两枚都 pinned）时返回零增量，与内核 `continue`
/// 一致。法向 `normal` 走 `normalize_or_zero` + `dot(unit, unit) > 0` 检查 + 退化 `+X`
/// 回退，`penetration = max(thickness - sqrt(len_sq), 0)`，位置增量
/// `pos_a - self_curr`；速度冲量仅在入射法向相对速度 `vrel_n < 0` 时施加，
/// `vel_delta = new_va - self_vel`。
#[expect(
    clippy::too_many_arguments,
    reason = "逐词镜像 cloth_ccd_resolve 内核单半边的全部输入（self/other 各自的 prev/curr/逆质量、self 速度、thickness、restitution、inv_dt）；拆成结构体会掩盖与内核的一一对应，反而降低可审计性。"
)]
fn wesl_resolve_one(
    self_prev: [f32; 3],
    self_curr: [f32; 3],
    self_vel: [f32; 3],
    self_w: f32,
    other_prev: [f32; 3],
    other_curr: [f32; 3],
    other_w: f32,
    thickness: f32,
    restitution: f32,
    inv_dt: f32,
) -> ([f32; 3], [f32; 3]) {
    let zero = [0.0, 0.0, 0.0];
    let wa = self_w.max(0.0);
    let wb = other_w.max(0.0);
    let wsum = wa + wb;
    if wsum <= 0.0 {
        return (zero, zero);
    }
    let Some(t) = wesl_swept_toi(self_prev, self_curr, other_prev, other_curr, thickness) else {
        return (zero, zero);
    };

    // TOI 快照位置：a_c = prev + (curr - prev) * t。
    let a_c = v_add(self_prev, v_scale(v_sub(self_curr, self_prev), t));
    let b_c = v_add(other_prev, v_scale(v_sub(other_curr, other_prev), t));
    let delta = v_sub(a_c, b_c);

    // normalize_or_zero：len_sq > EPS 时按 1/sqrt(len_sq) 缩放（倒数乘，非逐分量除），
    // 并保留 dot(unit, unit) > 0 的二次检查；否则回退固定 +X。逐词镜像内核分支。
    let len_sq = v_dot(delta, delta);
    let mut normal = [1.0, 0.0, 0.0];
    if len_sq > WESL_EPS_LEN_SQ {
        let unit = v_scale(delta, 1.0 / len_sq.sqrt());
        if v_dot(unit, unit) > 0.0 {
            normal = unit;
        }
    }
    let penetration = (thickness - len_sq.sqrt()).max(0.0);
    let inv_wsum = 1.0 / wsum;

    // self 吸附到 TOI 后按逆质量份额推开；增量相对当前位置，供 apply 相位相加。
    let pos_self = v_add(a_c, v_scale(normal, wa * inv_wsum * penetration));
    let pos_delta = v_sub(pos_self, self_curr);

    // 从 TOI 运动恢复入射速度，交换法向相对速度冲量。
    let va_in = v_scale(v_sub(a_c, self_prev), inv_dt);
    let vb_in = v_scale(v_sub(b_c, other_prev), inv_dt);
    let vrel_n = v_dot(v_sub(va_in, vb_in), normal);
    let mut vel_delta = zero;
    if vrel_n < 0.0 {
        let impulse = -(1.0 + restitution) * vrel_n * inv_wsum;
        let new_va = v_add(va_in, v_scale(normal, wa * impulse));
        vel_delta = v_sub(new_va, self_vel);
    }
    (pos_delta, vel_delta)
}

/// `Vec3` -> `[f32; 3]`。
fn arr(v: Vec3) -> [f32; 3] {
    [v.x, v.y, v.z]
}

/// `[f32; 3]` -> `Vec3`。
fn vec3(a: [f32; 3]) -> Vec3 {
    Vec3::new(a[0], a[1], a[2])
}

/// 逐分量 `f32::to_bits` 比对，`#[track_caller]` 把失败定位到调用场景。
#[track_caller]
fn assert_vec_bit_exact(label: &str, got: [f32; 3], want: [f32; 3]) {
    for axis in 0..3 {
        assert_eq!(
            got[axis].to_bits(),
            want[axis].to_bits(),
            "{label} 分量 {axis} 不逐位相等: got={} (0x{:08x}) want={} (0x{:08x})",
            got[axis],
            got[axis].to_bits(),
            want[axis],
            want[axis].to_bits(),
        );
    }
}

/// 一对粒子的隔离解算场景描述。坐标须落在 `[0.30, 0.70)` 并配 `cell_size = 1.0`，
/// 以保证两枚粒子共享唯一 cell、形成唯一候选对。逆质量为 `0` 表示 pinned。
#[derive(Clone, Copy)]
struct Pair {
    a_prev: [f32; 3],
    a_curr: [f32; 3],
    a_vel: [f32; 3],
    a_w: f32,
    b_prev: [f32; 3],
    b_curr: [f32; 3],
    b_vel: [f32; 3],
    b_w: f32,
    cell_size: f32,
    thickness: f32,
    restitution: f32,
    dt: f32,
}

/// 核心断言：构造一对黄金粒子跑 [`resolve_self_ccd`]，再用 [`wesl_resolve_one`] 跑两枚
/// 粒子各自的半边，逐位比对 **四组净变化**（a/b 的位置、a/b 的速度）。pinned 粒子的
/// 「有效应用增量」取零（`cloth_ccd_apply` 跳过写入），与黄金 `is_pinned()` 守卫一致。
fn assert_pair_resolution_bit_exact(p: Pair) {
    // 黄金侧。
    let mut golden = [
        ClothParticle {
            position: vec3(p.a_curr),
            velocity: vec3(p.a_vel),
            inverse_mass: p.a_w,
        },
        ClothParticle {
            position: vec3(p.b_curr),
            velocity: vec3(p.b_vel),
            inverse_mass: p.b_w,
        },
    ];
    let prevs = [vec3(p.a_prev), vec3(p.b_prev)];
    let mut params = SelfCcdParams::new(p.cell_size, p.thickness);
    params.restitution = p.restitution;
    resolve_self_ccd(&mut golden, &prevs, params, p.dt);

    // 原生侧：取黄金 sanitized 后的 thickness/restitution，并用黄金 inv_dt 守卫。
    let sp = params.sanitized();
    let inv_dt = guarded_inv_dt(p.dt);

    let a_pinned = p.a_w <= 0.0;
    let b_pinned = p.b_w <= 0.0;

    let (pos_delta_a, vel_delta_a) = wesl_resolve_one(
        p.a_prev,
        p.a_curr,
        p.a_vel,
        p.a_w,
        p.b_prev,
        p.b_curr,
        p.b_w,
        sp.thickness,
        sp.restitution,
        inv_dt,
    );
    let (pos_delta_b, vel_delta_b) = wesl_resolve_one(
        p.b_prev,
        p.b_curr,
        p.b_vel,
        p.b_w,
        p.a_prev,
        p.a_curr,
        p.a_w,
        sp.thickness,
        sp.restitution,
        inv_dt,
    );

    // apply 相位对 pinned 粒子跳过写入 => 有效增量取零。
    let eff_pos_a = if a_pinned { [0.0, 0.0, 0.0] } else { pos_delta_a };
    let eff_vel_a = if a_pinned { [0.0, 0.0, 0.0] } else { vel_delta_a };
    let eff_pos_b = if b_pinned { [0.0, 0.0, 0.0] } else { pos_delta_b };
    let eff_vel_b = if b_pinned { [0.0, 0.0, 0.0] } else { vel_delta_b };

    // 黄金净变化：final - 输入快照（pinned 未写 => 恒为零）。
    let golden_net_pos_a = v_sub(arr(golden[0].position), p.a_curr);
    let golden_net_vel_a = v_sub(arr(golden[0].velocity), p.a_vel);
    let golden_net_pos_b = v_sub(arr(golden[1].position), p.b_curr);
    let golden_net_vel_b = v_sub(arr(golden[1].velocity), p.b_vel);

    assert_vec_bit_exact("a 位置净变化", eff_pos_a, golden_net_pos_a);
    assert_vec_bit_exact("a 速度净变化", eff_vel_a, golden_net_vel_a);
    assert_vec_bit_exact("b 位置净变化", eff_pos_b, golden_net_pos_b);
    assert_vec_bit_exact("b 速度净变化", eff_vel_b, golden_net_vel_b);
}

/// 迎面闭合（自由-自由）：两枚相向而行的自由粒子穿过彼此，TOI 命中后对称推开。
#[test]
fn head_on_free_free() {
    assert_pair_resolution_bit_exact(Pair {
        a_prev: [0.40, 0.50, 0.50],
        a_curr: [0.60, 0.50, 0.50],
        a_vel: [0.3, 0.0, 0.0],
        a_w: 1.0,
        b_prev: [0.60, 0.50, 0.50],
        b_curr: [0.40, 0.50, 0.50],
        b_vel: [-0.3, 0.0, 0.0],
        b_w: 1.0,
        cell_size: 1.0,
        thickness: 0.05,
        restitution: 0.0,
        dt: 1.0 / 60.0,
    });
}

/// 自由 vs pinned（高索引 pinned）：仅自由粒子承担全部推开量，pinned 净变化为零。
#[test]
fn free_vs_pinned_high_index() {
    assert_pair_resolution_bit_exact(Pair {
        a_prev: [0.42, 0.50, 0.50],
        a_curr: [0.58, 0.50, 0.50],
        a_vel: [0.2, 0.1, 0.0],
        a_w: 1.0,
        b_prev: [0.55, 0.50, 0.50],
        b_curr: [0.55, 0.50, 0.50],
        b_vel: [0.0, 0.0, 0.0],
        b_w: 0.0,
        cell_size: 1.0,
        thickness: 0.06,
        restitution: 0.0,
        dt: 1.0 / 60.0,
    });
}

/// pinned vs 自由（低索引 pinned）：验证推开份额与写入守卫不随索引次序改变。
#[test]
fn pinned_low_index_vs_free() {
    assert_pair_resolution_bit_exact(Pair {
        a_prev: [0.45, 0.50, 0.50],
        a_curr: [0.45, 0.50, 0.50],
        a_vel: [0.0, 0.0, 0.0],
        a_w: 0.0,
        b_prev: [0.58, 0.50, 0.50],
        b_curr: [0.42, 0.50, 0.50],
        b_vel: [-0.2, 0.0, 0.1],
        b_w: 1.0,
        cell_size: 1.0,
        thickness: 0.06,
        restitution: 0.0,
        dt: 1.0 / 60.0,
    });
}

/// 恢复系数 0.5：半弹性回弹，冲量按 `-(1 + 0.5) * vrel_n / wsum`。
#[test]
fn restitution_half() {
    assert_pair_resolution_bit_exact(Pair {
        a_prev: [0.40, 0.50, 0.50],
        a_curr: [0.56, 0.50, 0.50],
        a_vel: [0.0, 0.0, 0.0],
        a_w: 1.0,
        b_prev: [0.60, 0.50, 0.50],
        b_curr: [0.44, 0.50, 0.50],
        b_vel: [0.0, 0.0, 0.0],
        b_w: 1.0,
        cell_size: 1.0,
        thickness: 0.05,
        restitution: 0.5,
        dt: 1.0 / 90.0,
    });
}

/// 恢复系数 1.0：完全弹性回弹（入射法向速度被镜像）。
#[test]
fn restitution_full_bounce() {
    assert_pair_resolution_bit_exact(Pair {
        a_prev: [0.41, 0.50, 0.50],
        a_curr: [0.57, 0.50, 0.50],
        a_vel: [0.1, 0.0, 0.0],
        a_w: 1.0,
        b_prev: [0.59, 0.50, 0.50],
        b_curr: [0.43, 0.50, 0.50],
        b_vel: [-0.1, 0.0, 0.0],
        b_w: 1.0,
        cell_size: 1.0,
        thickness: 0.05,
        restitution: 1.0,
        dt: 1.0 / 120.0,
    });
}

/// 恢复系数越界（2.0）：`sanitized` 夹到 1.0；黄金与原生取同一夹取值。
#[test]
fn restitution_clamped_out_of_range() {
    assert_pair_resolution_bit_exact(Pair {
        a_prev: [0.40, 0.50, 0.50],
        a_curr: [0.58, 0.50, 0.50],
        a_vel: [0.0, 0.0, 0.0],
        a_w: 1.0,
        b_prev: [0.60, 0.50, 0.50],
        b_curr: [0.42, 0.50, 0.50],
        b_vel: [0.0, 0.0, 0.0],
        b_w: 1.0,
        cell_size: 1.0,
        thickness: 0.05,
        restitution: 2.0,
        dt: 1.0 / 60.0,
    });
}

/// 三维斜向闭合：法向在三个轴都非零，验证逐分量归一与推开。
#[test]
fn diagonal_3d() {
    assert_pair_resolution_bit_exact(Pair {
        a_prev: [0.40, 0.42, 0.44],
        a_curr: [0.58, 0.56, 0.54],
        a_vel: [0.2, 0.15, 0.1],
        a_w: 1.0,
        b_prev: [0.60, 0.58, 0.56],
        b_curr: [0.42, 0.44, 0.46],
        b_vel: [-0.2, -0.15, -0.1],
        b_w: 1.0,
        cell_size: 1.0,
        thickness: 0.07,
        restitution: 0.25,
        dt: 1.0 / 75.0,
    });
}

/// 非对称逆质量（wa=1, wb=4）：轻粒子承担更大推开份额，验证 `inv_wsum` 加权。
#[test]
fn asymmetric_inverse_masses() {
    assert_pair_resolution_bit_exact(Pair {
        a_prev: [0.40, 0.50, 0.50],
        a_curr: [0.58, 0.50, 0.50],
        a_vel: [0.1, 0.0, 0.0],
        a_w: 1.0,
        b_prev: [0.60, 0.50, 0.50],
        b_curr: [0.42, 0.50, 0.50],
        b_vel: [-0.4, 0.0, 0.0],
        b_w: 4.0,
        cell_size: 1.0,
        thickness: 0.05,
        restitution: 0.3,
        dt: 1.0 / 60.0,
    });
}

/// 退化重合 + `+X` 回退：两枚粒子起点完全重合（`d0 = 0` => `c <= 0` => `t = 0`），
/// TOI 快照处 `delta = 0`、`len_sq = 0`，法向回退固定 `+X`，穿透取满 `thickness`。
/// 伙伴设为 pinned，故仅对自由粒子逐位比对（自由-自由重合的半边符号分歧见模块 doc）。
#[test]
fn degenerate_coincident_toi_plus_x_fallback() {
    assert_pair_resolution_bit_exact(Pair {
        a_prev: [0.50, 0.50, 0.50],
        a_curr: [0.50, 0.50, 0.50],
        a_vel: [0.0, 0.0, 0.0],
        a_w: 1.0,
        b_prev: [0.50, 0.50, 0.50],
        b_curr: [0.50, 0.50, 0.50],
        b_vel: [0.0, 0.0, 0.0],
        b_w: 0.0,
        cell_size: 1.0,
        thickness: 1.0e-7,
        restitution: 0.0,
        dt: 1.0 / 60.0,
    });
}

/// 帧初即重叠（`c <= 0` => `t = 0`，但非重合）：法向良态，立即接触推开。
#[test]
fn already_overlapping_immediate_contact() {
    assert_pair_resolution_bit_exact(Pair {
        a_prev: [0.50, 0.50, 0.50],
        a_curr: [0.50, 0.50, 0.50],
        a_vel: [0.0, 0.0, 0.0],
        a_w: 1.0,
        b_prev: [0.52, 0.50, 0.50],
        b_curr: [0.52, 0.50, 0.50],
        b_vel: [0.0, 0.0, 0.0],
        b_w: 1.0,
        cell_size: 1.0,
        thickness: 0.05,
        restitution: 0.0,
        dt: 1.0 / 60.0,
    });
}

/// 静止且分离（TOI 为 None）：共享 cell 但无相对运动、起始分离 => 无解算、净变化全零。
#[test]
fn stationary_separated_noop() {
    assert_pair_resolution_bit_exact(Pair {
        a_prev: [0.32, 0.32, 0.32],
        a_curr: [0.32, 0.32, 0.32],
        a_vel: [0.0, 0.0, 0.0],
        a_w: 1.0,
        b_prev: [0.66, 0.66, 0.66],
        b_curr: [0.66, 0.66, 0.66],
        b_vel: [0.0, 0.0, 0.0],
        b_w: 1.0,
        cell_size: 1.0,
        thickness: 0.05,
        restitution: 0.0,
        dt: 1.0 / 60.0,
    });
}

/// 两枚都 pinned（`wsum <= 0`）：无可动自由度 => 两侧净变化全零。
#[test]
fn both_pinned_noop() {
    assert_pair_resolution_bit_exact(Pair {
        a_prev: [0.48, 0.50, 0.50],
        a_curr: [0.48, 0.50, 0.50],
        a_vel: [0.0, 0.0, 0.0],
        a_w: 0.0,
        b_prev: [0.52, 0.50, 0.50],
        b_curr: [0.52, 0.50, 0.50],
        b_vel: [0.0, 0.0, 0.0],
        b_w: 0.0,
        cell_size: 1.0,
        thickness: 0.05,
        restitution: 0.0,
        dt: 1.0 / 60.0,
    });
}

/// 近零 `dt`（`1e-15`）：`inv_dt` 守卫取 0 => 入射速度为零 => 无速度冲量，但位置仍解算。
#[test]
fn near_zero_dt_zeros_velocity_impulse() {
    assert_pair_resolution_bit_exact(Pair {
        a_prev: [0.40, 0.50, 0.50],
        a_curr: [0.58, 0.50, 0.50],
        a_vel: [0.5, 0.0, 0.0],
        a_w: 1.0,
        b_prev: [0.60, 0.50, 0.50],
        b_curr: [0.42, 0.50, 0.50],
        b_vel: [-0.5, 0.0, 0.0],
        b_w: 1.0,
        cell_size: 1.0,
        thickness: 0.05,
        restitution: 0.5,
        dt: 1.0e-15,
    });
}

/// 确定性模糊：64 轮 LCG 生成的混合单对（坐标映射到 `[0.30, 0.70)` 保证单 cell 共享，
/// 逐轮在 自由-自由 / 自由-pinned / pinned-自由 间轮换），每轮都走全套逐位断言。
#[test]
fn jittered_mixed_single_pairs() {
    let mut state: u32 = 0x9E37_79B9;
    // 单一 LCG 步进闭包，返回 [0, 1)，避免对 state 的多重可变借用。
    let mut step = || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((state >> 8) as f32) / ((1_u32 << 24) as f32)
    };
    // 坐标映射到 [0.30, 0.70) 保证单 cell 共享；速度映射到 [-0.4, 0.4)。
    let mut next = move || -> f32 { 0.30 + step() * 0.40 };

    for round in 0..64_u32 {
        let a_prev = [next(), next(), next()];
        let a_curr = [next(), next(), next()];
        let b_prev = [next(), next(), next()];
        let b_curr = [next(), next(), next()];
        let a_vel = [(next() - 0.50) * 2.0, (next() - 0.50) * 2.0, (next() - 0.50) * 2.0];
        let b_vel = [(next() - 0.50) * 2.0, (next() - 0.50) * 2.0, (next() - 0.50) * 2.0];
        // 质量/pinned 布局在三种组合间轮换。
        let (a_w, b_w) = match round % 3 {
            0 => (1.0, 1.0),
            1 => (1.0, 0.0),
            _ => (0.0, 1.0),
        };
        assert_pair_resolution_bit_exact(Pair {
            a_prev,
            a_curr,
            a_vel,
            a_w,
            b_prev,
            b_curr,
            b_vel,
            b_w,
            cell_size: 1.0,
            thickness: 0.05,
            restitution: if round % 2 == 0 { 0.0 } else { 0.4 },
            dt: 1.0 / 60.0,
        });
    }
}
