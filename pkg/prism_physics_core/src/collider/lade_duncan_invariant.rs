//! Lade–Duncan 应力比不变量。
//!
//! Lade–Duncan 准则以主应力不变量组合 `κ = I1³ / I3` 刻画受压应力态
//! 偏离静水态的程度。它在偏平面上为比 Matsuoka–Nakai 更“外凸”的曲线
//! （三轴拉伸角更鼓），常用于密实砂土的峰值强度拟合。
//!
//! 关键关系（压缩为正约定下）：
//! ```text
//! κ = I1³ / I3,   静水态 σ1=σ2=σ3 时 κ = 27。
//! ```
//! 剪切越强，`κ` 越大（`κ ≥ 27`，由 AM–GM 不等式保证）。
//!
//! 本模块是**纯不变量诊断**：不构造屈服面、不参与碰撞/求解管线，
//! 完全 0 耦合。输入沿用本仓库**拉伸为正**约定，内部转为压缩为正。
//!
//! 与 `matsuoka_nakai_invariant`（κ=I1·I2/I3）互补：二者分别对应不同
//! 的强度包络拟合，可同时用于诊断同一受压应力态。

/// 低于该行列式幅值视为退化（含零主应力的受压态）。
const DETERMINANT_EPS: f32 = 1e-9;

/// 静水态对应的 Lade–Duncan κ 基准值。
pub const HYDROSTATIC_KAPPA: f32 = 27.0;

/// Lade–Duncan 应力比不变量。
///
/// 由三个主应力（拉伸为正）构造，要求应力态为**全受压**，否则返回 `None`。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LadeDuncanInvariant {
    /// 第一主不变量 I1 = c1 + c2 + c3（压缩为正）。
    i1: f32,
    /// 第三主不变量 I3 = c1·c2·c3（压缩为正）。
    i3: f32,
}

impl LadeDuncanInvariant {
    /// 由主应力（**拉伸为正**）构造；内部转为压缩为正 `c_i = −σ_i`。
    ///
    /// 任一压缩主应力 `≤ 0`（存在拉伸）或 `I3` 过小（退化）时返回 `None`。
    pub fn from_principal_stresses(principal_tension_positive: [f32; 3]) -> Option<Self> {
        let c = [
            -principal_tension_positive[0],
            -principal_tension_positive[1],
            -principal_tension_positive[2],
        ];

        if c[0] <= 0.0 || c[1] <= 0.0 || c[2] <= 0.0 {
            return None;
        }

        let i1 = c[0] + c[1] + c[2];
        let i3 = c[0] * c[1] * c[2];

        if i3 <= DETERMINANT_EPS {
            return None;
        }

        Some(Self { i1, i3 })
    }

    /// 第一主不变量 I1（压缩为正）。
    pub fn first_invariant(&self) -> f32 {
        self.i1
    }

    /// 第三主不变量 I3（压缩为正）。
    pub fn third_invariant(&self) -> f32 {
        self.i3
    }

    /// Lade–Duncan 比值 `κ = I1³ / I3`。
    ///
    /// 静水态时 `κ = 27`，剪切越强越大；数值以 `HYDROSTATIC_KAPPA` 作下界夹取。
    pub fn kappa(&self) -> f32 {
        let i1 = self.i1;
        (i1 * i1 * i1 / self.i3).max(HYDROSTATIC_KAPPA)
    }

    /// 相对静水基准的剪切超量 `κ − 27`（`≥ 0`，静水态为 0）。
    pub fn shear_excess(&self) -> f32 {
        self.kappa() - HYDROSTATIC_KAPPA
    }

    /// 当前 κ 是否达到/超过给定临界 κ。
    pub fn is_at_or_beyond(&self, critical_kappa: f32) -> bool {
        self.kappa() >= critical_kappa
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-3;

    #[test]
    fn hydrostatic_gives_kappa_twenty_seven() {
        let ld = LadeDuncanInvariant::from_principal_stresses([-4.0, -4.0, -4.0]).unwrap();
        assert!((ld.kappa() - HYDROSTATIC_KAPPA).abs() < TOL);
        assert!(ld.shear_excess().abs() < TOL);
    }

    #[test]
    fn known_state_matches_hand_computed_kappa() {
        // 压缩主应力 (4,2,1): I1=7, I3=8 => κ = 343/8 = 42.875。
        let ld = LadeDuncanInvariant::from_principal_stresses([-4.0, -2.0, -1.0]).unwrap();
        assert!((ld.first_invariant() - 7.0).abs() < TOL);
        assert!((ld.third_invariant() - 8.0).abs() < TOL);
        assert!((ld.kappa() - 343.0 / 8.0).abs() < TOL);
        assert!((ld.shear_excess() - (343.0 / 8.0 - 27.0)).abs() < TOL);
    }

    #[test]
    fn tensile_component_is_rejected() {
        assert!(LadeDuncanInvariant::from_principal_stresses([1.0, -2.0, -3.0]).is_none());
        assert!(LadeDuncanInvariant::from_principal_stresses([0.0, -2.0, -3.0]).is_none());
    }

    #[test]
    fn degenerate_determinant_is_rejected() {
        assert!(LadeDuncanInvariant::from_principal_stresses([-4.0, -2.0, 0.0]).is_none());
    }

    #[test]
    fn ordering_is_permutation_invariant() {
        let a = LadeDuncanInvariant::from_principal_stresses([-4.0, -2.0, -1.0]).unwrap();
        let b = LadeDuncanInvariant::from_principal_stresses([-1.0, -4.0, -2.0]).unwrap();
        let c = LadeDuncanInvariant::from_principal_stresses([-2.0, -1.0, -4.0]).unwrap();
        assert!((a.kappa() - b.kappa()).abs() < TOL);
        assert!((b.kappa() - c.kappa()).abs() < TOL);
    }

    #[test]
    fn more_shear_increases_kappa() {
        let mild = LadeDuncanInvariant::from_principal_stresses([-3.0, -3.0, -2.9]).unwrap();
        let strong = LadeDuncanInvariant::from_principal_stresses([-6.0, -2.0, -1.0]).unwrap();
        assert!(strong.kappa() > mild.kappa());
        assert!(strong.shear_excess() > mild.shear_excess());
    }

    #[test]
    fn is_at_or_beyond_threshold_behaves() {
        // κ = 42.875。
        let ld = LadeDuncanInvariant::from_principal_stresses([-4.0, -2.0, -1.0]).unwrap();
        assert!(ld.is_at_or_beyond(42.0));
        assert!(!ld.is_at_or_beyond(43.0));
    }
}
