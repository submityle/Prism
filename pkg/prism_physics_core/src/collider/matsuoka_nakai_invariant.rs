//! Matsuoka–Nakai 应力比不变量（Spatially Mobilized Plane, SMP）。
//!
//! Matsuoka–Nakai 准则以主应力不变量的组合 `κ = I1·I2/I3` 刻画一个
//! 各向同性受压应力态偏离静水态的程度，并据此反推“已动员摩擦角”
//! （mobilized friction angle）。它比 Mohr–Coulomb 更光滑（在偏平面上
//! 为外接三轴压缩/三轴拉伸的凸曲线），常用于砂土/颗粒材料的本构诊断。
//!
//! 关键关系（压缩为正约定下）：
//! ```text
//! κ = I1·I2 / I3,
//! κ = (9 − sin²φ) / (1 − sin²φ)  =>  sin²φ = (κ − 9) / (κ − 1)
//! ```
//! 其中：
//! - 静水态 σ1=σ2=σ3 时 κ=9、φ=0；
//! - 剪切越强 κ 越大、φ 越大。
//!
//! 本模块是**纯不变量诊断**，不构造屈服面、不参与碰撞/求解管线，
//! 完全 0 耦合，仅做只读分析。输入沿用本仓库的**拉伸为正**约定，
//! 内部转换为压缩为正后计算。

use std::f32::consts::FRAC_PI_2;

/// 低于该行列式幅值视为退化（含零主应力的受压态）。
const DETERMINANT_EPS: f32 = 1e-9;

/// 静水态对应的 κ 基准值（κ = 9 时无动员摩擦）。
pub const HYDROSTATIC_KAPPA: f32 = 9.0;

/// Matsuoka–Nakai 应力比不变量。
///
/// 由三个主应力（拉伸为正）构造；要求应力态为**全受压**
/// （三个主应力在压缩为正约定下均为正），否则 SMP 概念不适用，
/// 返回 `None`。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MatsuokaNakaiInvariant {
    /// 第一主不变量 I1 = c1 + c2 + c3（压缩为正）。
    i1: f32,
    /// 第二主不变量 I2 = c1·c2 + c2·c3 + c3·c1（压缩为正）。
    i2: f32,
    /// 第三主不变量 I3 = c1·c2·c3（压缩为正）。
    i3: f32,
}

impl MatsuokaNakaiInvariant {
    /// 由主应力（**拉伸为正**）构造。
    ///
    /// 内部转换为压缩为正 `c_i = −σ_i`。当任一压缩主应力 `≤ 0`
    /// （即存在拉伸分量），或行列式 `I3` 过小（退化受压态）时返回 `None`。
    pub fn from_principal_stresses(principal_tension_positive: [f32; 3]) -> Option<Self> {
        let c = [
            -principal_tension_positive[0],
            -principal_tension_positive[1],
            -principal_tension_positive[2],
        ];

        // SMP 概念要求全受压：压缩为正下三个主值都必须为正。
        if c[0] <= 0.0 || c[1] <= 0.0 || c[2] <= 0.0 {
            return None;
        }

        let i1 = c[0] + c[1] + c[2];
        let i2 = c[0] * c[1] + c[1] * c[2] + c[2] * c[0];
        let i3 = c[0] * c[1] * c[2];

        if i3 <= DETERMINANT_EPS {
            return None;
        }

        Some(Self { i1, i2, i3 })
    }

    /// 第一主不变量 I1（压缩为正）。
    pub fn first_invariant(&self) -> f32 {
        self.i1
    }

    /// 第二主不变量 I2（压缩为正）。
    pub fn second_invariant(&self) -> f32 {
        self.i2
    }

    /// 第三主不变量 I3（压缩为正）。
    pub fn third_invariant(&self) -> f32 {
        self.i3
    }

    /// Matsuoka–Nakai 比值 `κ = I1·I2 / I3`。
    ///
    /// 静水态时 `κ = 9`，剪切越强 `κ` 越大。κ 恒 `≥ 9`（算术/几何
    /// 均值不等式的推论），数值上以 `HYDROSTATIC_KAPPA` 作为下界夹取。
    pub fn kappa(&self) -> f32 {
        (self.i1 * self.i2 / self.i3).max(HYDROSTATIC_KAPPA)
    }

    /// 已动员摩擦角的正弦 `sin φ = √((κ − 9)/(κ − 1))`。
    ///
    /// 结果夹取到 `[0, 1]`。静水态返回 `0`。
    pub fn sin_mobilized_friction(&self) -> f32 {
        let kappa = self.kappa();
        // κ ≥ 9 时分子 ≥ 0；分母 κ − 1 ≥ 8 > 0。
        let sin_sq = ((kappa - HYDROSTATIC_KAPPA) / (kappa - 1.0)).clamp(0.0, 1.0);
        sin_sq.sqrt()
    }

    /// 已动员摩擦角（弧度），范围 `[0, π/2]`。
    pub fn mobilized_friction_angle(&self) -> f32 {
        let sin_phi = self.sin_mobilized_friction();
        // asin 属于被禁用的 f32 超越函数，改在 f64 下计算后降精度。
        let angle = f64::from(sin_phi).clamp(-1.0, 1.0).asin() as f32;
        angle.clamp(0.0, FRAC_PI_2)
    }

    /// 已动员摩擦角（度）。
    pub fn mobilized_friction_angle_degrees(&self) -> f32 {
        self.mobilized_friction_angle().to_degrees()
    }

    /// 当前 κ 是否达到/超过给定临界 κ（例如由破坏摩擦角换算而来）。
    pub fn is_at_or_beyond(&self, critical_kappa: f32) -> bool {
        self.kappa() >= critical_kappa
    }

    /// 由摩擦角（弧度）换算 Matsuoka–Nakai 临界 κ：
    /// `κ_f = (9 − sin²φ)/(1 − sin²φ)`。
    ///
    /// `phi` 夹取到 `[0, π/2)`；接近 π/2 时分母趋零，结果以 `f32::MAX` 封顶。
    pub fn critical_kappa_for_friction_angle(phi_radians: f32) -> f32 {
        let phi = phi_radians.clamp(0.0, FRAC_PI_2);
        // sin 属于被禁用的 f32 超越函数，改在 f64 下计算后降精度。
        let sin_phi = f64::from(phi).sin() as f32;
        let sin_sq = (sin_phi * sin_phi).clamp(0.0, 1.0);
        let denom = 1.0 - sin_sq;
        if denom <= DETERMINANT_EPS {
            return f32::MAX;
        }
        (HYDROSTATIC_KAPPA - sin_sq) / denom
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-4;

    #[test]
    fn hydrostatic_gives_kappa_nine_and_zero_friction() {
        // 拉伸为正下的受压静水态：三个主应力都为 −5。
        let mn = MatsuokaNakaiInvariant::from_principal_stresses([-5.0, -5.0, -5.0]).unwrap();
        assert!((mn.kappa() - HYDROSTATIC_KAPPA).abs() < TOL);
        assert!(mn.sin_mobilized_friction().abs() < TOL);
        assert!(mn.mobilized_friction_angle().abs() < TOL);
    }

    #[test]
    fn known_state_matches_hand_computed_kappa() {
        // 压缩为正主应力 (4,2,1)，拉伸为正输入即其相反数。
        // I1=7, I2=4*2+2*1+1*4=14, I3=8 => κ = 7*14/8 = 12.25。
        let mn = MatsuokaNakaiInvariant::from_principal_stresses([-4.0, -2.0, -1.0]).unwrap();
        assert!((mn.first_invariant() - 7.0).abs() < TOL);
        assert!((mn.second_invariant() - 14.0).abs() < TOL);
        assert!((mn.third_invariant() - 8.0).abs() < TOL);
        assert!((mn.kappa() - 12.25).abs() < TOL);

        // sin²φ = (12.25-9)/(12.25-1) = 3.25/11.25 = 0.288889。
        let expected_sin = (3.25_f32 / 11.25).sqrt();
        assert!((mn.sin_mobilized_friction() - expected_sin).abs() < TOL);
    }

    #[test]
    fn critical_kappa_round_trips_with_friction_angle() {
        // φ = 30° => sin²φ = 0.25 => κ = (9-0.25)/(1-0.25) = 8.75/0.75 = 11.6667。
        let phi = 30.0_f32.to_radians();
        let kappa = MatsuokaNakaiInvariant::critical_kappa_for_friction_angle(phi);
        assert!((kappa - (8.75_f32 / 0.75)).abs() < 1e-3);
    }

    #[test]
    fn mobilized_angle_recovers_input_friction() {
        // 构造 φ=30° 的临界 κ，再反推摩擦角应约为 30°。
        // 取压缩主应力 (a, b, b) 并选 a/b 使 κ=11.6667。
        // 对 (c1,c2,c3)=(c1,1,1): I1=c1+2, I2=2c1+1, I3=c1。
        // κ=(c1+2)(2c1+1)/c1=11.6667 => 2c1²+5c1+2 = 11.6667 c1
        //   => 2c1² -6.6667c1 +2 =0 => c1 = (6.6667±√(44.444-16))/4。
        // √28.444=5.3333 => c1=(6.6667+5.3333)/4=3.0。
        let mn = MatsuokaNakaiInvariant::from_principal_stresses([-3.0, -1.0, -1.0]).unwrap();
        assert!((mn.kappa() - (8.75_f32 / 0.75)).abs() < 1e-3);
        assert!((mn.mobilized_friction_angle_degrees() - 30.0).abs() < 0.05);
    }

    #[test]
    fn tensile_component_is_rejected() {
        // 拉伸为正下存在正主应力（拉伸）=> 非全受压 => None。
        assert!(MatsuokaNakaiInvariant::from_principal_stresses([1.0, -2.0, -3.0]).is_none());
        assert!(MatsuokaNakaiInvariant::from_principal_stresses([0.0, -2.0, -3.0]).is_none());
    }

    #[test]
    fn degenerate_determinant_is_rejected() {
        // 一个主应力为零 => 压缩主值含 0 => 被全受压判据拦截。
        assert!(MatsuokaNakaiInvariant::from_principal_stresses([-4.0, -2.0, 0.0]).is_none());
    }

    #[test]
    fn ordering_is_permutation_invariant() {
        let a = MatsuokaNakaiInvariant::from_principal_stresses([-4.0, -2.0, -1.0]).unwrap();
        let b = MatsuokaNakaiInvariant::from_principal_stresses([-1.0, -4.0, -2.0]).unwrap();
        let c = MatsuokaNakaiInvariant::from_principal_stresses([-2.0, -1.0, -4.0]).unwrap();
        assert!((a.kappa() - b.kappa()).abs() < TOL);
        assert!((b.kappa() - c.kappa()).abs() < TOL);
    }

    #[test]
    fn more_shear_increases_kappa() {
        // 近静水 (3,3,2.9) vs 强剪切 (6,2,1)。
        let mild = MatsuokaNakaiInvariant::from_principal_stresses([-3.0, -3.0, -2.9]).unwrap();
        let strong = MatsuokaNakaiInvariant::from_principal_stresses([-6.0, -2.0, -1.0]).unwrap();
        assert!(strong.kappa() > mild.kappa());
        assert!(strong.mobilized_friction_angle() > mild.mobilized_friction_angle());
    }

    #[test]
    fn is_at_or_beyond_threshold_behaves() {
        let mn = MatsuokaNakaiInvariant::from_principal_stresses([-4.0, -2.0, -1.0]).unwrap();
        // κ = 12.25。
        assert!(mn.is_at_or_beyond(12.0));
        assert!(!mn.is_at_or_beyond(13.0));
    }
}
