//! Rowe 应力-剪胀关系（stress-dilatancy，颗粒最小能耗诊断，零耦合原语）。
//!
//! Rowe (1962) 从颗粒接触最小能耗比导出三轴条件下主应力比与剪胀率的
//! 线性关系：
//! ```text
//! R = K · D
//! ```
//! 其中：
//! - `R = σ₁'/σ₃'` 为有效主应力比；
//! - `D = 1 − dε_v/dε₁` 为**剪胀率**（`dε_v` 体应变增量、`dε₁` 轴向应变
//!   增量，压缩为正；纯剪缩 `D<1`，强剪胀 `D>1`）；
//! - `K = tan²(π/4 + φ_f/2)` 为 Rowe 系数，`φ_f` 为颗粒间/临界状态摩擦角。
//!
//! 由此可在 `R`、`D`、`K`（或 `φ_f`）三者中已知其二求第三者，并由
//! `K` 反演摩擦角 `φ_f = 2·atan(√K) − π/2`。本关系区别于剪胀角原语
//! （后者仅由应变增量给出 `ψ`），这里刻画应力比与剪胀的耦合定律。
//!
//! 本模块是**纯只读诊断**：不构造本构、不参与碰撞/求解管线，完全 0 耦合。

use std::f32::consts::FRAC_PI_2;
use std::f32::consts::FRAC_PI_4;

/// 低量级保护阈值。
const EPS: f32 = 1e-9;

/// Rowe 应力-剪胀关系诊断（以 Rowe 系数 `K` 为核心状态）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RoweStressDilatancy {
    coefficient: f32,
}

impl RoweStressDilatancy {
    /// 由 Rowe 系数 `K ≥ 1` 直接构造。
    pub fn from_coefficient(coefficient: f32) -> Option<Self> {
        if !coefficient.is_finite() || coefficient < 1.0 - EPS {
            return None;
        }
        Some(Self {
            coefficient: coefficient.max(1.0),
        })
    }

    /// 由摩擦角 `φ_f ∈ [0, π/2)` 构造：`K = tan²(π/4 + φ_f/2)`。
    pub fn from_friction_angle(friction_angle_rad: f32) -> Option<Self> {
        if !friction_angle_rad.is_finite() || !(0.0..FRAC_PI_2).contains(&friction_angle_rad) {
            return None;
        }
        // tan 为被禁用的 f32 超越函数，改在 f64 下计算后降精度。
        let half = f64::from(FRAC_PI_4) + f64::from(friction_angle_rad) / 2.0;
        let t = half.tan();
        Some(Self {
            coefficient: (t * t) as f32,
        })
    }

    /// 由实测应力比与剪胀率反演 `K = R/D`，要求 `R > 0`、`D > 0`。
    pub fn from_measurements(stress_ratio: f32, dilatancy_rate: f32) -> Option<Self> {
        if !stress_ratio.is_finite() || !dilatancy_rate.is_finite() {
            return None;
        }
        if stress_ratio <= EPS || dilatancy_rate <= EPS {
            return None;
        }
        Self::from_coefficient(stress_ratio / dilatancy_rate)
    }

    /// Rowe 系数 `K`。
    pub fn coefficient(&self) -> f32 {
        self.coefficient
    }

    /// 由 `K` 反演摩擦角 `φ_f = 2·atan(√K) − π/2`（弧度）。
    pub fn friction_angle(&self) -> f32 {
        // atan 为被禁用的 f32 超越函数，改在 f64 下计算后降精度。
        let root_k = f64::from(self.coefficient.sqrt());
        (2.0 * root_k.atan() - f64::from(FRAC_PI_2)) as f32
    }

    /// 摩擦角（度）。
    pub fn friction_angle_degrees(&self) -> f32 {
        self.friction_angle().to_degrees()
    }

    /// 预测应力比 `R = K·D`。
    pub fn predict_stress_ratio(&self, dilatancy_rate: f32) -> f32 {
        self.coefficient * dilatancy_rate
    }

    /// 由应力比预测剪胀率 `D = R/K`，要求 `R` 有限。
    pub fn predict_dilatancy_rate(&self, stress_ratio: f32) -> f32 {
        stress_ratio / self.coefficient
    }

    /// 由应变增量计算剪胀率 `D = 1 − dε_v/dε₁`，要求 `dε₁ ≠ 0`。
    pub fn dilatancy_rate_from_strain_increments(
        volumetric_increment: f32,
        axial_increment: f32,
    ) -> Option<f32> {
        if !volumetric_increment.is_finite() || !axial_increment.is_finite() {
            return None;
        }
        if axial_increment.abs() <= EPS {
            return None;
        }
        Some(1.0 - volumetric_increment / axial_increment)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-3;

    #[test]
    fn coefficient_from_zero_friction_is_one() {
        // φ=0 => K = tan²(45°) = 1。
        let r = RoweStressDilatancy::from_friction_angle(0.0).unwrap();
        assert!((r.coefficient() - 1.0).abs() < TOL);
    }

    #[test]
    fn coefficient_from_thirty_degrees() {
        // φ=30° => K = tan²(60°) = 3。
        let r = RoweStressDilatancy::from_friction_angle(std::f32::consts::FRAC_PI_6).unwrap();
        assert!((r.coefficient() - 3.0).abs() < TOL);
    }

    #[test]
    fn friction_angle_roundtrips_through_coefficient() {
        let phi = std::f32::consts::FRAC_PI_6; // 30°
        let r = RoweStressDilatancy::from_friction_angle(phi).unwrap();
        assert!((r.friction_angle() - phi).abs() < TOL);
        assert!((r.friction_angle_degrees() - 30.0).abs() < 0.1);
    }

    #[test]
    fn rowe_relation_r_equals_k_times_d() {
        let r = RoweStressDilatancy::from_coefficient(3.0).unwrap();
        // D=1.5 => R = 4.5。
        assert!((r.predict_stress_ratio(1.5) - 4.5).abs() < TOL);
        // R=4.5 => D = 1.5。
        assert!((r.predict_dilatancy_rate(4.5) - 1.5).abs() < TOL);
    }

    #[test]
    fn from_measurements_recovers_coefficient() {
        // R=6, D=2 => K=3。
        let r = RoweStressDilatancy::from_measurements(6.0, 2.0).unwrap();
        assert!((r.coefficient() - 3.0).abs() < TOL);
    }

    #[test]
    fn from_measurements_rejects_nonpositive() {
        assert!(RoweStressDilatancy::from_measurements(0.0, 2.0).is_none());
        assert!(RoweStressDilatancy::from_measurements(6.0, 0.0).is_none());
        assert!(RoweStressDilatancy::from_measurements(-6.0, 2.0).is_none());
    }

    #[test]
    fn from_coefficient_rejects_below_one() {
        assert!(RoweStressDilatancy::from_coefficient(0.5).is_none());
        assert!(RoweStressDilatancy::from_coefficient(f32::NAN).is_none());
    }

    #[test]
    fn dilatancy_rate_isochoric_is_one() {
        // dε_v=0 => D = 1。
        let d = RoweStressDilatancy::dilatancy_rate_from_strain_increments(0.0, 0.01).unwrap();
        assert!((d - 1.0).abs() < TOL);
    }

    #[test]
    fn dilatancy_rate_dilation_exceeds_one() {
        // 剪胀: dε_v<0 (膨胀) => D>1。
        let d = RoweStressDilatancy::dilatancy_rate_from_strain_increments(-0.005, 0.01).unwrap();
        assert!(d > 1.0);
    }

    #[test]
    fn dilatancy_rate_contraction_below_one() {
        // 剪缩: dε_v>0 => D<1。
        let d = RoweStressDilatancy::dilatancy_rate_from_strain_increments(0.004, 0.01).unwrap();
        assert!(d < 1.0);
    }

    #[test]
    fn dilatancy_rate_rejects_zero_axial() {
        assert!(RoweStressDilatancy::dilatancy_rate_from_strain_increments(0.001, 0.0).is_none());
    }
}
