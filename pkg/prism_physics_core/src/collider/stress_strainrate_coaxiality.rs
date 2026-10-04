//! 应力—应变率非共轴角（颗粒塑性诊断，零耦合原语）。
//!
//! 经典塑性理论假设应力主轴与塑性应变率主轴重合（“共轴”，coaxial）。
//! 但颗粒材料在单剪等路径中普遍呈现**非共轴**：应力主方向与应变率主方向之间
//! 存在一个夹角 `α`，是区分颗粒塑性与经典 `J2` 塑性的关键特征之一
//! （参见 Gutierrez & Ishihara 等关于非共轴流动的研究）。
//!
//! 本原语给定对称应力张量 `σ` 与对称应变率（或应变增量）张量 `D`，取各自的
//! **主（最大特征值）方向**，返回二者之间的锐角：
//!
//! ```text
//!   α = arccos( |n_σ · n_D| ),   α ∈ [0, π/2]
//! ```
//!
//! 使用绝对值点积消除特征向量的符号歧义——比较的是主轴（无向直线）而非方向。
//! `α = 0` 为完全共轴，`α = π/2` 为完全正交。
//!
//! 纯函数式、零耦合：在 [`crate::mpm::symmetric_eigen`]（Jacobi 对称特征分解）
//! 之上计算，不触碰主帧循环、不依赖渲染引擎。输入会先取对称部
//! `½(A + Aᵀ)`，因此对轻微非对称的数值输入也稳健。

use crate::mpm::symmetric_eigen;
use glam::{Mat3, Vec3};

/// 低于该特征值间隔视为主轴病态（近各向同性/轴对称），主方向定义不良。
const DEGENERACY_EPS: f32 = 1e-6;

/// 应力主轴与应变率主轴之间的非共轴角分析结果。
///
/// 由 [`StressStrainRateCoaxiality::from_tensors`] 构造。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StressStrainRateCoaxiality {
    /// 应力张量的主（最大特征值）方向，已做符号规范化的单位向量。
    stress_axis: Vec3,
    /// 应变率张量的主（最大特征值）方向，已做符号规范化的单位向量。
    strain_rate_axis: Vec3,
    /// 两主轴夹角余弦的绝对值 `|n_σ · n_D| ∈ [0, 1]`。
    cos_angle: f32,
    /// 应力张量最大与次大特征值之差（判定主轴是否病态）。
    stress_spectral_gap: f32,
    /// 应变率张量最大与次大特征值之差（判定主轴是否病态）。
    strain_rate_spectral_gap: f32,
}

impl StressStrainRateCoaxiality {
    /// 由对称应力张量与对称应变率张量构造非共轴角分析。
    ///
    /// 两个入参均为行主序 `3×3` 数组；构造时各自取对称部，因此行/列主序不影响
    /// 结果。任一元素非有限时返回 [`None`]。
    #[must_use]
    pub fn from_tensors(stress: [[f32; 3]; 3], strain_rate: [[f32; 3]; 3]) -> Option<Self> {
        let sigma = symmetric_part(stress)?;
        let d = symmetric_part(strain_rate)?;

        let (vs, es) = symmetric_eigen(sigma);
        let (vd, ed) = symmetric_eigen(d);

        // symmetric_eigen 的特征值按降序排列，列向量为对应单位特征向量。
        let stress_axis = canonical_sign(vs.x_axis);
        let strain_rate_axis = canonical_sign(vd.x_axis);
        if !stress_axis.is_finite() || !strain_rate_axis.is_finite() {
            return None;
        }

        let cos_angle = stress_axis.dot(strain_rate_axis).abs().clamp(0.0, 1.0);

        Some(Self {
            stress_axis,
            strain_rate_axis,
            cos_angle,
            stress_spectral_gap: es.x - es.y,
            strain_rate_spectral_gap: ed.x - ed.y,
        })
    }

    /// 应力主（最大特征值）方向，符号规范化后的单位向量。
    #[must_use]
    pub fn principal_stress_direction(&self) -> Vec3 {
        self.stress_axis
    }

    /// 应变率主（最大特征值）方向，符号规范化后的单位向量。
    #[must_use]
    pub fn principal_strain_rate_direction(&self) -> Vec3 {
        self.strain_rate_axis
    }

    /// 两主轴夹角余弦的绝对值 `|n_σ · n_D| ∈ [0, 1]`。
    #[must_use]
    pub fn cos_coaxiality(&self) -> f32 {
        self.cos_angle
    }

    /// 非共轴角 `α ∈ [0, π/2]`（弧度）。`0` 为完全共轴。
    #[must_use]
    pub fn coaxiality_angle(&self) -> f32 {
        // arccos 走 f64 以规避 f32 反三角禁用项。
        f64::from(self.cos_angle).clamp(-1.0, 1.0).acos() as f32
    }

    /// 非共轴角（度）。
    #[must_use]
    pub fn coaxiality_angle_degrees(&self) -> f32 {
        self.coaxiality_angle().to_degrees()
    }

    /// 是否在给定角容差（弧度）内近似共轴。
    #[must_use]
    pub fn is_coaxial(&self, tolerance_radians: f32) -> bool {
        self.coaxiality_angle() <= tolerance_radians.max(0.0)
    }

    /// 应力张量最大与次大特征值之差。接近 `0` 表示应力近各向同性、主轴病态。
    #[must_use]
    pub fn stress_spectral_gap(&self) -> f32 {
        self.stress_spectral_gap
    }

    /// 应变率张量最大与次大特征值之差。接近 `0` 表示应变率近轴对称、主轴病态。
    #[must_use]
    pub fn strain_rate_spectral_gap(&self) -> f32 {
        self.strain_rate_spectral_gap
    }

    /// 两个主轴是否都定义良好（谱间隔均超过阈值）。病态时非共轴角数值不可靠。
    #[must_use]
    pub fn is_well_defined(&self) -> bool {
        self.stress_spectral_gap > DEGENERACY_EPS && self.strain_rate_spectral_gap > DEGENERACY_EPS
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
    // Mat3::from_cols 以列构造；对称矩阵行列主序一致，这里直接按列填。
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
    fn identical_tensors_are_coaxial() {
        let s = diag(3.0, 1.0, 1.0);
        let c = StressStrainRateCoaxiality::from_tensors(s, s).unwrap();
        assert!(c.coaxiality_angle().abs() < 1e-4);
        assert!(c.cos_coaxiality() > 1.0 - 1e-6);
        assert!(c.is_coaxial(1e-3));
        assert!(c.is_well_defined());
    }

    #[test]
    fn orthogonal_major_axes_give_ninety_degrees() {
        // 应力主轴沿 x，应变率主轴沿 y。
        let stress = diag(3.0, 1.0, 1.0);
        let strain_rate = diag(1.0, 3.0, 1.0);
        let c = StressStrainRateCoaxiality::from_tensors(stress, strain_rate).unwrap();
        assert!((c.coaxiality_angle() - FRAC_PI_2).abs() < 1e-4);
        assert!(c.cos_coaxiality() < 1e-5);
        assert!(!c.is_coaxial(1e-2));
    }

    #[test]
    fn forty_five_degree_rotation_is_detected() {
        // 应力主轴沿 x；应变率主轴沿 xy 面内 45°：对称阵 [[2,1,0],[1,2,0],[0,0,1]]
        // 的最大特征向量为 (1,1,0)/√2，与 x 轴夹角 45°。
        let stress = diag(3.0, 1.0, 1.0);
        let strain_rate = [[2.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 1.0]];
        let c = StressStrainRateCoaxiality::from_tensors(stress, strain_rate).unwrap();
        assert!((c.coaxiality_angle() - FRAC_PI_4).abs() < 1e-3);
        assert!((c.cos_coaxiality() - (0.5_f32).sqrt()).abs() < 1e-3);
        assert!((c.coaxiality_angle_degrees() - 45.0).abs() < 0.1);
    }

    #[test]
    fn input_is_symmetrized() {
        // 非对称输入应等价于其对称部：½([[2,2,0],[0,2,0],...]) → [[2,1,0],[1,2,0],...]
        let stress = diag(3.0, 1.0, 1.0);
        let skewed = [[2.0, 2.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]];
        let c = StressStrainRateCoaxiality::from_tensors(stress, skewed).unwrap();
        assert!((c.coaxiality_angle() - FRAC_PI_4).abs() < 1e-3);
    }

    #[test]
    fn directions_are_sign_canonicalized_units() {
        let c = StressStrainRateCoaxiality::from_tensors(diag(5.0, 1.0, 1.0), diag(5.0, 1.0, 1.0))
            .unwrap();
        let n = c.principal_stress_direction();
        assert!((n.length() - 1.0).abs() < 1e-5);
        // 主导分量（x）应为正。
        assert!(n.x > 0.0);
        assert_eq!(n, c.principal_strain_rate_direction());
    }

    #[test]
    fn isotropic_tensor_is_flagged_ill_defined() {
        let iso = diag(2.0, 2.0, 2.0);
        let c = StressStrainRateCoaxiality::from_tensors(iso, diag(3.0, 1.0, 1.0)).unwrap();
        assert!(!c.is_well_defined());
    }

    #[test]
    fn rejects_non_finite() {
        let bad = [[f32::NAN, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert!(StressStrainRateCoaxiality::from_tensors(bad, diag(1.0, 2.0, 3.0)).is_none());
        assert!(StressStrainRateCoaxiality::from_tensors(diag(1.0, 2.0, 3.0), bad).is_none());
    }
}
