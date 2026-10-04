//! Lode 角偏平面形状因子 `g(θ)`（屈服/破坏面偏量横截面形状诊断，零耦合原语）。
//!
//! 真实土/岩破坏面在偏平面（π 平面）上的横截面并非圆形，而是三轴压缩
//! 半径 `r_c` 大于三轴拉伸半径 `r_e` 的近三角形。Gudehus–Argyris (1973)
//! 用一条光滑插值函数描述该形状：
//! ```text
//! g(θ) = 2k / [ (1 + k) − (1 − k)·sin(3θ) ]
//! ```
//! 其中：
//! - `θ ∈ [−π/6, +π/6]` 为 Lode 角（本模块约定 `θ = +π/6` 为三轴压缩、
//!   `θ = −π/6` 为三轴拉伸）；
//! - `k = r_e / r_c ∈ (0, 1]` 为**拉压半径比**。
//!
//! 端点性质：`g(+π/6) = 1`（压缩参照），`g(−π/6) = k`（拉伸）。
//! 当 `k ∈ [0.5, 1]` 时横截面保持外凸（Drucker 稳定性要求）。
//! 由 Mohr–Coulomb 摩擦角可取 `k = (3 − sinφ)/(3 + sinφ)`。
//!
//! 本模块是**纯只读诊断**：不构造本构、不参与碰撞/求解管线，完全 0 耦合。

use std::f32::consts::FRAC_PI_6;

/// 低量级保护阈值。
const EPS: f32 = 1e-9;

/// Lode 角有效区间的角度容差（略放宽以容纳浮点端点）。
const SEXTANT_TOL: f32 = 1e-5;

/// 外凸性所需的最小拉压比。
const CONVEXITY_MIN_RATIO: f32 = 0.5;

/// Lode 角偏平面形状因子诊断。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LodeShapeFactor {
    ratio: f32,
}

impl LodeShapeFactor {
    /// 由拉压半径比 `k = r_e/r_c` 构造，要求 `0 < k ≤ 1`。
    pub fn from_ratio(ratio: f32) -> Option<Self> {
        if !ratio.is_finite() || ratio <= EPS || ratio > 1.0 + EPS {
            return None;
        }
        Some(Self {
            ratio: ratio.min(1.0),
        })
    }

    /// 由 Mohr–Coulomb 摩擦角 `φ ∈ (0, π/2)` 构造：
    /// `k = (3 − sinφ)/(3 + sinφ)`。
    pub fn from_friction_angle(friction_angle_rad: f32) -> Option<Self> {
        if !friction_angle_rad.is_finite()
            || friction_angle_rad <= 0.0
            || friction_angle_rad >= std::f32::consts::FRAC_PI_2
        {
            return None;
        }
        // sin 为被禁用的 f32 超越函数，改在 f64 下计算后降精度。
        let sin_phi = f64::from(friction_angle_rad).sin();
        let k = ((3.0 - sin_phi) / (3.0 + sin_phi)) as f32;
        Self::from_ratio(k)
    }

    /// 拉压半径比 `k`。
    pub fn ratio(&self) -> f32 {
        self.ratio
    }

    /// 偏平面形状因子 `g(θ) = 2k/[(1+k)−(1−k)sin(3θ)]`。
    ///
    /// `lode_angle_rad` 必须落在 `[−π/6, +π/6]`，否则返回 `None`。
    pub fn shape_factor(&self, lode_angle_rad: f32) -> Option<f32> {
        if !lode_angle_rad.is_finite() || lode_angle_rad.abs() > FRAC_PI_6 + SEXTANT_TOL {
            return None;
        }
        let theta = lode_angle_rad.clamp(-FRAC_PI_6, FRAC_PI_6);
        // sin(3θ) 为被禁用的 f32 超越函数，改在 f64 下计算后降精度。
        let sin3 = (3.0 * f64::from(theta)).sin();
        let k = f64::from(self.ratio);
        let denom = (1.0 + k) - (1.0 - k) * sin3;
        Some((2.0 * k / denom) as f32)
    }

    /// 三轴压缩端 `g(+π/6) = 1`（参照半径）。
    pub fn compression_shape_factor(&self) -> f32 {
        1.0
    }

    /// 三轴拉伸端 `g(−π/6) = k`。
    pub fn extension_shape_factor(&self) -> f32 {
        self.ratio
    }

    /// 偏平面横截面是否外凸（`k ≥ 0.5`，Drucker 稳定性）。
    pub fn is_convex(&self) -> bool {
        self.ratio >= CONVEXITY_MIN_RATIO - EPS
    }

    /// 横截面是否近似为圆（`k ≈ 1`，von Mises / Drucker–Prager）。
    pub fn is_circular(&self, tolerance: f32) -> bool {
        (self.ratio - 1.0).abs() <= tolerance.max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_6;

    const TOL: f32 = 1e-4;

    #[test]
    fn compression_endpoint_is_unity() {
        let g = LodeShapeFactor::from_ratio(0.8).unwrap();
        assert!((g.shape_factor(FRAC_PI_6).unwrap() - 1.0).abs() < TOL);
        assert!((g.compression_shape_factor() - 1.0).abs() < TOL);
    }

    #[test]
    fn extension_endpoint_is_ratio() {
        let g = LodeShapeFactor::from_ratio(0.75).unwrap();
        assert!((g.shape_factor(-FRAC_PI_6).unwrap() - 0.75).abs() < TOL);
        assert!((g.extension_shape_factor() - 0.75).abs() < TOL);
    }

    #[test]
    fn circular_ratio_is_constant_one() {
        let g = LodeShapeFactor::from_ratio(1.0).unwrap();
        assert!((g.shape_factor(0.0).unwrap() - 1.0).abs() < TOL);
        assert!((g.shape_factor(FRAC_PI_6).unwrap() - 1.0).abs() < TOL);
        assert!((g.shape_factor(-FRAC_PI_6).unwrap() - 1.0).abs() < TOL);
        assert!(g.is_circular(1e-6));
    }

    #[test]
    fn midpoint_lies_between_endpoints() {
        let g = LodeShapeFactor::from_ratio(0.7).unwrap();
        let mid = g.shape_factor(0.0).unwrap();
        // θ=0 => sin0=0 => g = 2k/(1+k) = 1.4/1.7 ≈ 0.8235。
        assert!((mid - (2.0 * 0.7 / 1.7)).abs() < TOL);
        assert!(mid > 0.7 && mid < 1.0);
    }

    #[test]
    fn monotonic_from_extension_to_compression() {
        let g = LodeShapeFactor::from_ratio(0.65).unwrap();
        let e = g.shape_factor(-FRAC_PI_6).unwrap();
        let m = g.shape_factor(0.0).unwrap();
        let c = g.shape_factor(FRAC_PI_6).unwrap();
        assert!(e < m && m < c);
    }

    #[test]
    fn out_of_sextant_is_rejected() {
        let g = LodeShapeFactor::from_ratio(0.8).unwrap();
        assert!(g.shape_factor(FRAC_PI_6 * 1.5).is_none());
        assert!(g.shape_factor(-FRAC_PI_6 * 2.0).is_none());
    }

    #[test]
    fn from_friction_angle_matches_mohr_coulomb_ratio() {
        // φ=30° => sinφ=0.5 => k=(3−0.5)/(3+0.5)=2.5/3.5≈0.7143。
        let phi = std::f32::consts::FRAC_PI_6; // 30°
        let g = LodeShapeFactor::from_friction_angle(phi).unwrap();
        assert!((g.ratio() - (2.5 / 3.5)).abs() < 1e-3);
    }

    #[test]
    fn from_friction_angle_rejects_out_of_range() {
        assert!(LodeShapeFactor::from_friction_angle(0.0).is_none());
        assert!(LodeShapeFactor::from_friction_angle(std::f32::consts::FRAC_PI_2).is_none());
    }

    #[test]
    fn convexity_threshold() {
        assert!(LodeShapeFactor::from_ratio(0.5).unwrap().is_convex());
        assert!(LodeShapeFactor::from_ratio(0.9).unwrap().is_convex());
        assert!(!LodeShapeFactor::from_ratio(0.4).unwrap().is_convex());
    }

    #[test]
    fn from_ratio_rejects_bad_inputs() {
        assert!(LodeShapeFactor::from_ratio(0.0).is_none());
        assert!(LodeShapeFactor::from_ratio(-0.3).is_none());
        assert!(LodeShapeFactor::from_ratio(1.5).is_none());
        assert!(LodeShapeFactor::from_ratio(f32::NAN).is_none());
    }

    #[test]
    fn denominator_stays_positive_across_sextant() {
        // 极端小 k 下仍不应出现除零/非有限。
        let g = LodeShapeFactor::from_ratio(0.01).unwrap();
        for i in -6..=6 {
            let theta = FRAC_PI_6 * (i as f32) / 6.0;
            let v = g.shape_factor(theta).unwrap();
            assert!(v.is_finite() && v > 0.0);
        }
    }
}
