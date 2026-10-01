#![cfg(test)]
//! `cloth_self_ccd.wesl` 的 `cloth_ccd_swept_toi` 函数的 **逐位** CPU 转写，对齐
//! 架构层黄金 [`swept_pair_toi`]（求解一对粒子在单位帧区间 `t in [0, 1]` 内扫掠
//! 到达 `thickness` 间距的最早命中时间）。
//!
//! ## 为何可做逐位 parity
//! WESL `cloth_ccd_swept_toi` 与黄金 `swept_pair_toi` 逐词同构：二者都把问题化为
//! 二次方程 `|d0 + t*dv|^2 = thickness^2`（系数 `a = dv·dv`、`b = 2·(d0·dv)`、
//! `c = d0·d0 - thickness^2`），用同一套比较式早退（`c <= 0` 立即命中、`a` 太小
//! 判无相对运动、判别式为负判永不相交、入射根落在 `[0, 1]` 外判错过），且只用
//! `f32::sqrt`。`sqrt` 是 IEEE 正确舍入运算，`rsqrt` 的硬件 ULP 差异在此 **不出现**，
//! 故本 parity 在无头 CPU 与真机 GPU 上都逐位成立——不同于含 `inverseSqrt` 的内核
//! 只能建立「与设备无关的算法等价」闭环。
//!
//! 本模块逐词把 WESL 算术搬到 CPU（原生 `f32` 数组，不复用黄金的 `Vec3` 方法），
//! 再用 `f32::to_bits` 比对黄金返回的 `Option<f32>`：要么同为 `None`，要么同为
//! `Some` 且载荷逐位相等。`WESL_EPS_REL_MOTION` 取 `1.0e-12`，与 WESL 常量
//! `CLOTH_CCD_EPS_REL_MOTION` 及黄金私有常量 `EPS_REL_MOTION`（`1e-12`）同值。

use prism_render_architecture::cloth::self_ccd::swept_pair_toi;
use prism_render_architecture::cloth::Vec3;

/// WESL `CLOTH_CCD_EPS_REL_MOTION` 的镜像：相对运动的平方低于该阈值即视为无运动。
/// 与黄金私有 `EPS_REL_MOTION`（`1e-12`）同值。
const WESL_EPS_REL_MOTION: f32 = 1.0e-12;

/// WESL `cloth_ccd_swept_toi` 的逐位转写。
///
/// 粒子 `a` 沿 `prev_a -> curr_a`、粒子 `b` 沿 `prev_b -> curr_b` 在单位帧内线性
/// 移动。相对偏移 `d(t) = d0 + t*dv`（`d0 = prev_a - prev_b`、
/// `dv = (curr_a - curr_b) - d0`），首次 `|d(t)| = thickness` 为二次方程较小根。
/// 起始已在接触带内（`c <= 0`）返回 `Some(0.0)`；无相对运动（`a <= eps`）、判别式
/// 为负、或入射根落在 `[0, 1]` 外均返回 `None`；否则返回入射根 `Some(t)`。全程仅用
/// `f32::sqrt`，所有早退均为比较式，结果确定且无 `NaN`。
fn wesl_swept_toi(
    prev_a: [f32; 3],
    curr_a: [f32; 3],
    prev_b: [f32; 3],
    curr_b: [f32; 3],
    thickness: f32,
) -> Option<f32> {
    let d0 = [
        prev_a[0] - prev_b[0],
        prev_a[1] - prev_b[1],
        prev_a[2] - prev_b[2],
    ];
    let d1 = [
        curr_a[0] - curr_b[0],
        curr_a[1] - curr_b[1],
        curr_a[2] - curr_b[2],
    ];
    let dv = [d1[0] - d0[0], d1[1] - d0[1], d1[2] - d0[2]];
    let c = (d0[0] * d0[0] + d0[1] * d0[1] + d0[2] * d0[2]) - thickness * thickness;
    // 帧初即落在接触带内：接触立即发生。
    if c <= 0.0 {
        return Some(0.0);
    }
    let a = dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2];
    // 没有可观的相对运动且未重叠：间距永不闭合。
    if a <= WESL_EPS_REL_MOTION {
        return None;
    }
    let b = 2.0 * (d0[0] * dv[0] + d0[1] * dv[1] + d0[2] * dv[2]);
    let disc = b * b - 4.0 * a * c;
    // 扫掠间距从未到达 `thickness`。
    if disc < 0.0 {
        return None;
    }
    // 入射根（较小根）：间距首次闭合到 `thickness` 的时刻。
    let t = (-b - disc.sqrt()) / (2.0 * a);
    if t >= 0.0 && t <= 1.0 {
        Some(t)
    } else {
        None
    }
}

/// 跑黄金 `swept_pair_toi` 与 WESL 转写，逐位比对二者的 `Option<f32>` 结果：要么同为
/// `None`，要么同为 `Some` 且载荷 `to_bits` 相等。
fn assert_toi_bit_exact(
    prev_a: [f32; 3],
    curr_a: [f32; 3],
    prev_b: [f32; 3],
    curr_b: [f32; 3],
    thickness: f32,
) {
    let golden = swept_pair_toi(
        Vec3::new(prev_a[0], prev_a[1], prev_a[2]),
        Vec3::new(curr_a[0], curr_a[1], curr_a[2]),
        Vec3::new(prev_b[0], prev_b[1], prev_b[2]),
        Vec3::new(curr_b[0], curr_b[1], curr_b[2]),
        thickness,
    );
    let wesl = wesl_swept_toi(prev_a, curr_a, prev_b, curr_b, thickness);
    match (golden, wesl) {
        (None, None) => {}
        (Some(g), Some(w)) => assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "TOI 载荷逐位不一致：golden={g} wesl={w}"
        ),
        _ => panic!("TOI 存在性不一致：golden={golden:?} wesl={wesl:?}"),
    }
}

/// 帧初即在接触带内（`c <= 0`）立即命中，返回 `Some(0.0)`。
#[test]
fn already_overlapping_returns_zero_bit_for_bit() {
    assert_toi_bit_exact(
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.01, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        0.05,
    );
}

/// 起始间距恰等于 `thickness`（`c == 0`）走 `c <= 0` 分支，返回 `Some(0.0)`。
#[test]
fn exactly_at_thickness_start_returns_zero_bit_for_bit() {
    assert_toi_bit_exact(
        [0.1, 0.0, 0.0],
        [0.2, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        0.1,
    );
}

/// 两粒子同步平移（无相对运动，`a <= eps`）且起始分离：返回 `None`。
#[test]
fn no_relative_motion_returns_none_bit_for_bit() {
    assert_toi_bit_exact(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        0.1,
    );
}

/// 完全静止且分离：`a == 0` 早退，返回 `None`。
#[test]
fn stationary_separated_pair_returns_none_bit_for_bit() {
    assert_toi_bit_exact(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        0.1,
    );
}

/// 平行掠过，最近距离远大于 `thickness`（判别式为负）：返回 `None`。
#[test]
fn discriminant_negative_returns_none_bit_for_bit() {
    assert_toi_bit_exact(
        [-1.0, 0.5, 0.0],
        [1.0, 0.5, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        0.1,
    );
}

/// 正面接近静止目标，入射根落在 `[0, 1]` 内：返回 `Some(t)` 并逐位一致。
#[test]
fn head_on_collision_returns_entry_root_bit_for_bit() {
    assert_toi_bit_exact(
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        0.1,
    );
}

/// 两粒子相向而行、在原点附近相遇，入射根在 `[0, 1]` 内。
#[test]
fn both_moving_converging_bit_for_bit() {
    assert_toi_bit_exact(
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        0.1,
    );
}

/// 交叉早已发生在过去（较小根为负）：返回 `None`。
#[test]
fn entry_root_before_frame_returns_none_bit_for_bit() {
    assert_toi_bit_exact(
        [0.2, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        0.1,
    );
}

/// 缓慢接近，入射根 `> 1`（本帧内尚未到达 `thickness`）：返回 `None`。
#[test]
fn entry_root_after_frame_returns_none_bit_for_bit() {
    assert_toi_bit_exact(
        [-10.0, 0.0, 0.0],
        [-8.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        0.1,
    );
}

/// 三维斜向接近命中：全分量非平凡，验证 `dot` 的乘加结合顺序逐位一致。
#[test]
fn diagonal_3d_collision_bit_for_bit() {
    assert_toi_bit_exact(
        [-0.7, -0.4, 0.3],
        [0.6, 0.5, -0.2],
        [0.3, 0.2, -0.1],
        [0.1, -0.1, 0.05],
        0.08,
    );
}

/// `thickness = 0` 且起始重合：`c = 0 <= 0` 立即命中，返回 `Some(0.0)`。
#[test]
fn zero_thickness_coincident_start_bit_for_bit() {
    assert_toi_bit_exact(
        [0.5, -0.5, 0.25],
        [1.0, 0.0, 0.0],
        [0.5, -0.5, 0.25],
        [-1.0, 0.0, 0.0],
        0.0,
    );
}

/// `thickness = 0` 且分离接近：二次式仍可解，验证零厚度下的入射根。
#[test]
fn zero_thickness_separated_approach_bit_for_bit() {
    assert_toi_bit_exact(
        [-1.0, 0.001, 0.0],
        [1.0, 0.001, 0.0],
        [0.0, -0.001, 0.0],
        [0.0, -0.001, 0.0],
        0.0,
    );
}

/// 增大 `thickness` 把原本错过的掠过变成命中：同一几何、两种厚度分别判定。
#[test]
fn thickness_opens_contact_window_bit_for_bit() {
    let (pa, ca, pb, cb) = (
        [-1.0, 0.3, 0.0],
        [1.0, 0.3, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    );
    // 薄：掠过（判别式为负）→ None。
    assert_toi_bit_exact(pa, ca, pb, cb, 0.1);
    // 厚：进入接触带 → Some(t)。
    assert_toi_bit_exact(pa, ca, pb, cb, 0.4);
}

/// 负坐标象限的正面接近：验证符号不影响逐位结合顺序。
#[test]
fn negative_quadrant_collision_bit_for_bit() {
    assert_toi_bit_exact(
        [-3.0, -2.0, -1.0],
        [-1.5, -2.0, -1.0],
        [-1.0, -2.0, -1.0],
        [-1.0, -2.0, -1.0],
        0.2,
    );
}

/// 近似切向掠过（判别式小正）：入射根仍在 `[0, 1]` 内，逐位比对。
#[test]
fn grazing_tangent_small_discriminant_bit_for_bit() {
    assert_toi_bit_exact(
        [-1.0, 0.0999, 0.0],
        [1.0, 0.0999, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        0.1,
    );
}

/// 一组抖动的 `prev/curr/thickness` 混合用例，批量断言逐位一致（命中与错过皆覆盖）。
#[test]
fn jittered_mixed_cases_bit_for_bit() {
    let cases: [([f32; 3], [f32; 3], [f32; 3], [f32; 3], f32); 6] = [
        (
            [0.13, -0.27, 0.44],
            [0.51, 0.19, -0.08],
            [-0.22, 0.31, -0.17],
            [0.09, -0.12, 0.26],
            0.07,
        ),
        (
            [-0.9, 0.05, 0.6],
            [0.8, -0.05, -0.6],
            [0.85, 0.0, 0.0],
            [-0.85, 0.0, 0.0],
            0.12,
        ),
        (
            [2.3, 1.1, -0.4],
            [2.31, 1.09, -0.41],
            [-1.7, -0.8, 0.9],
            [-1.69, -0.81, 0.91],
            0.05,
        ),
        (
            [0.0, 0.0, 0.0],
            [0.33, 0.33, 0.33],
            [0.02, 0.0, -0.01],
            [0.35, 0.33, 0.32],
            0.03,
        ),
        (
            [-5.0, 3.0, -2.0],
            [-4.0, 3.0, -2.0],
            [4.0, 3.0, -2.0],
            [3.0, 3.0, -2.0],
            0.25,
        ),
        (
            [0.4, 0.4, 0.4],
            [0.4, 0.4, 0.4],
            [-0.4, -0.4, -0.4],
            [0.4, 0.4, 0.4],
            0.6,
        ),
    ];
    for (pa, ca, pb, cb, th) in cases {
        assert_toi_bit_exact(pa, ca, pb, cb, th);
    }
}
