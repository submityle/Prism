//! 组构主轴重定向角（颗粒各向异性演化诊断，零耦合原语）。
//!
//! 颗粒材料的接触网络各向异性由**组构张量** `Φ` 描述。随加载推进，`Φ`
//! 的主轴会逐步转向加载方向——这一“组构重定向”是各向异性塑性与
//! 临界状态演化的关键特征。本原语给定前后两个时刻的对称组构张量，
//! 取各自的**主（最大特征值）方向**，返回二者间的锐角：
//!
//! ```text
//!   β = arccos( |n0 · n1| ),   β ∈ [0, π/2]
//! ```
//!
//! 使用绝对值点积消除特征向量符号歧义——比较的是主轴（无向直线）。
//! `β = 0` 表示组构主轴未转动，`β = π/2` 表示主轴旋转了 90°。
//!
//! 纯函数式、零耦合：构建于 [`crate::mpm::symmetric_eigen`]（Jacobi 对称
//! 特征分解）之上，不触碰主帧循环、不依赖渲染引擎。输入先取对称部
//! `½(A + Aᵀ)`，对轻微非对称数值输入稳健。
//!
//! 与 `stress_strainrate_coaxiality`（比较应力与应变率两种不同量的主轴）
//! 不同：本模块比较**同一量（组构）在两个时刻**的主轴，刻画时间演化。

use crate::mpm::symmetric_eigen;
use glam::{Mat3, Vec3};

/// 低于该特征值间隔视为主轴病态（近各向同性），主方向定义不良。
const DEGENERACY_EPS: f32 = 1e-6;

/// 组构主轴在两个时刻之间的重定向分析结果。
///
/// 由 [`FabricReorientation::from_tensors`] 构造。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FabricReorientation {
    /// 初始组构主（最大特征值）方向，符号规范化的单位向量。
    initial_axis: Vec3,
    /// 终末组构主（最大特征值）方向，符号规范化的单位向量。
    final_axis: Vec3,
    /// 两主轴夹角余弦的绝对值 `|n0 · n1| ∈ [0, 1]`。
    cos_angle: f32,
    /// 初始组构最大与次大特征值之差（判定主轴是否病态）。
    initial_spectral_gap: f32,
    /// 终末组构最大与次大特征值之差（判定主轴是否病态）。
    final_spectral_gap: f32,
}

impl FabricReorientation {
    /// 由前后两个对称组构张量（行主序 `3×3`）构造重定向分析。
    ///
    /// 构造时各自取对称部，故行/列主序不影响结果。任一元素非有限时返回 [`None`]。
    #[must_use]
    pub fn from_tensors(initial: [[f32; 3]; 3], final_fabric: [[f32; 3]; 3]) -> Option<Self> {
        let phi0 = symmetric_part(initial)?;
        let phi1 = symmetric_part(final_fabric)?;

        let (v0, e0) = symmetric_eigen(phi0);
        let (v1, e1) = symmetric_eigen(phi1);

        // symmetric_eigen 特征值降序，列向量为对应单位特征向量。
        let initial_axis = canonical_sign(v0.x_axis);
        let final_axis = canonical_sign(v1.x_axis);
        if !initial_axis.is_finite() || !final_axis.is_finite() {
            return None;
        }

        let cos_angle = initial_axis.dot(final_axis).abs().clamp(0.0, 1.0);

        Some(Self {
            initial_axis,
            final_axis,
            cos_angle,
            initial_spectral_gap: e0.x - e0.y,
            final_spectral_gap: e1.x - e1.y,
        })
    }

    /// 初始组构主方向，符号规范化后的单位向量。
    #[must_use]
    pub fn initial_major_direction(&self) -> Vec3 {
        self.initial_axis
    }

    /// 终末组构主方向，符号规范化后的单位向量。
    #[must_use]
    pub fn final_major_direction(&self) -> Vec3 {
        self.final_axis
    }

    /// 两主轴夹角余弦的绝对值 `|n0 · n1| ∈ [0, 1]`。
    #[must_use]
    pub fn cos_reorientation(&self) -> f32 {
        self.cos_angle
    }

    /// 重定向角 `β ∈ [0, π/2]`（弧度）。`0` 为无转动。
    #[must_use]
    pub fn reorientation_angle(&self) -> f32 {
        // arccos 走 f64 以规避 f32 反三角禁用项。
        f64::from(self.cos_angle).clamp(-1.0, 1.0).acos() as f32
    }

    /// 重定向角（度）。
    #[must_use]
    pub fn reorientation_angle_degrees(&self) -> f32 {
        self.reorientation_angle().to_degrees()
    }

    /// 是否在给定角容差（弧度）内近似未转动。
    #[must_use]
    pub fn is_stationary(&self, tolerance_radians: f32) -> bool {
        self.reorientation_angle() <= tolerance_radians.max(0.0)
    }

    /// 初始组构最大与次大特征值之差。接近 `0` 表示初始组构近各向同性、主轴病态。
    #[must_use]
    pub fn initial_spectral_gap(&self) -> f32 {
        self.initial_spectral_gap
    }

    /// 终末组构最大与次大特征值之差。接近 `0` 表示终末组构近各向同性、主轴病态。
    #[must_use]
    pub fn final_spectral_gap(&self) -> f32 {
        self.final_spectral_gap
    }

    /// 两个主轴是否都定义良好（谱间隔均超过阈值）。病态时重定向角数值不可靠。
    #[must_use]
    pub fn is_well_defined(&self) -> bool {
        self.initial_spectral_gap > DEGENERACY_EPS && self.final_spectral_gap > DEGENERACY_EPS
    }
}

/// 取行主序数组的对称部 `½(A + Aᵀ)`，并校验全部元素有限。
fn symmetric_part(m: [[f32; 3]; 3]) -> Option<Mat3> {
    for row in &m {
        for &v in row {
            if !v.is_finite() {
                return None;
            }
        }
    }
    let sym = |i: usize, j: usize| 0.5 * (m[i][j] + m[j][i]);
    Some(Mat3::from_cols(
        Vec3::new(sym(0, 0), sym(1, 0), sym(2, 0)),
        Vec3::new(sym(0, 1), sym(1, 1), sym(2, 1)),
        Vec3::new(sym(0, 2), sym(1, 2), sym(2, 2)),
    ))
}

/// 规范化特征向量符号：令绝对值最大的分量为正，保证确定性输出。
fn canonical_sign(v: Vec3) -> Vec3 {
    let ax = v.x.abs();
    let ay = v.y.abs();
    let az = v.z.abs();
    let dominant = if ax >= ay && ax >= az {
        v.x
    } else if ay >= az {
        v.y
    } else {
        v.z
    };
    if dominant < 0.0 {
        -v
    } else {
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::{FRAC_PI_2, FRAC_PI_4};

    fn diag(a: f32, b: f32, c: f32) -> [[f32; 3]; 3] {
        [[a, 0.0, 0.0], [0.0, b, 0.0], [0.0, 0.0, c]]
    }

    #[test]
    fn identical_fabric_has_no_reorientation() {
        let phi = diag(0.5, 0.3, 0.2);
        let r = FabricReorientation::from_tensors(phi, phi).unwrap();
        assert!(r.reorientation_angle().abs() < 1e-4);
        assert!(r.cos_reorientation() > 1.0 - 1e-6);
        assert!(r.is_stationary(1e-3));
        assert!(r.is_well_defined());
    }

    #[test]
    fn ninety_degree_axis_swap_is_detected() {
        // 主轴由 x 转到 y。
        let phi0 = diag(0.6, 0.2, 0.2);
        let phi1 = diag(0.2, 0.6, 0.2);
        let r = FabricReorientation::from_tensors(phi0, phi1).unwrap();
        assert!((r.reorientation_angle() - FRAC_PI_2).abs() < 1e-4);
        assert!(r.cos_reorientation() < 1e-5);
        assert!(!r.is_stationary(1e-2));
    }

    #[test]
    fn forty_five_degree_rotation_in_xy_plane() {
        // 初始主轴沿 x；终末组构在 xy 平面内主轴转 45°。
        let phi0 = diag(0.6, 0.2, 0.2);
        // 在 xy 平面内将 (0.6,0.2) 特征结构旋转 45°：对称张量
        // [[0.4,0.2,0],[0.2,0.4,0],[0,0,0.2]] 的主轴为 (1,1,0)/√2。
        let phi1 = [[0.4, 0.2, 0.0], [0.2, 0.4, 0.0], [0.0, 0.0, 0.2]];
        let r = FabricReorientation::from_tensors(phi0, phi1).unwrap();
        assert!((r.reorientation_angle() - FRAC_PI_4).abs() < 1e-3);
    }

    #[test]
    fn non_finite_input_is_rejected() {
        let good = diag(0.5, 0.3, 0.2);
        let bad = [[f32::NAN, 0.0, 0.0], [0.0, 0.3, 0.0], [0.0, 0.0, 0.2]];
        assert!(FabricReorientation::from_tensors(bad, good).is_none());
        assert!(FabricReorientation::from_tensors(good, bad).is_none());
    }

    #[test]
    fn isotropic_fabric_is_ill_defined() {
        // 各向同性组构无确定主轴 => 谱间隔为零 => 不良定义。
        let iso = diag(0.3333, 0.3333, 0.3333);
        let anis = diag(0.6, 0.2, 0.2);
        let r = FabricReorientation::from_tensors(iso, anis).unwrap();
        assert!(!r.is_well_defined());
        assert!(r.initial_spectral_gap() < DEGENERACY_EPS);
    }

    #[test]
    fn result_is_symmetric_in_order() {
        // 交换前后快照，重定向角应不变（无向主轴）。
        let phi0 = diag(0.6, 0.2, 0.2);
        let phi1 = [[0.4, 0.2, 0.0], [0.2, 0.4, 0.0], [0.0, 0.0, 0.2]];
        let a = FabricReorientation::from_tensors(phi0, phi1).unwrap();
        let b = FabricReorientation::from_tensors(phi1, phi0).unwrap();
        assert!((a.reorientation_angle() - b.reorientation_angle()).abs() < 1e-4);
    }

    #[test]
    fn row_major_asymmetric_input_is_symmetrized() {
        // 轻微非对称输入取对称部后仍给出稳定结果。
        let phi0 = diag(0.6, 0.2, 0.2);
        let phi1 = [[0.4, 0.21, 0.0], [0.19, 0.4, 0.0], [0.0, 0.0, 0.2]];
        let r = FabricReorientation::from_tensors(phi0, phi1).unwrap();
        assert!((r.reorientation_angle() - FRAC_PI_4).abs() < 1e-2);
    }
}
