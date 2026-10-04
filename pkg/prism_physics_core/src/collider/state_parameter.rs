//! 状态参数 ψ（Been–Jefferies 临界状态诊断，零耦合原语）。
//!
//! 现代临界状态土力学以**状态参数**
//! ```text
//! ψ = e − e_cs
//! ```
//! 刻画当前孔隙比 `e` 相对于**同一平均有效应力** `p'` 下临界状态孔隙比
//! `e_cs` 的偏离（Been & Jefferies, 1985）。它统一了相对密实度与应力水平：
//! - `ψ < 0`：位于 CSL 之下，密实、剪切趋于**剪胀**；
//! - `ψ > 0`：位于 CSL 之上，疏松、剪切趋于**剪缩**；
//! - `ψ ≈ 0`：接近临界状态。
//!
//! 临界状态线（CSL）采用半对数形式：
//! ```text
//! e_cs = Γ − λ · ln(p' / p_ref)
//! ```
//! 其中 `Γ` 为 `p' = p_ref` 处的临界孔隙比，`λ > 0` 为 CSL 斜率。
//!
//! 本模块是**纯只读诊断**：不构造本构、不参与碰撞/求解管线，完全 0 耦合。
//! 它是 NorSand/CASM 等状态相关本构的输入量，但这里只做标量计算。

/// 低于该量级的平均应力/参考应力视为非法（对数退化保护）。
const PRESSURE_EPS: f32 = 1e-12;

/// 状态参数 ψ 的诊断结果。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StateParameter {
    void_ratio: f32,
    critical_void_ratio: f32,
    psi: f32,
}

impl StateParameter {
    /// 由当前孔隙比与临界孔隙比直接构造，`ψ = e − e_cs`。
    pub fn from_void_ratios(void_ratio: f32, critical_void_ratio: f32) -> Self {
        Self {
            void_ratio,
            critical_void_ratio,
            psi: void_ratio - critical_void_ratio,
        }
    }

    /// 由半对数 CSL 计算临界孔隙比后构造：
    /// `e_cs = Γ − λ·ln(p'/p_ref)`，`ψ = e − e_cs`。
    ///
    /// `p_mean` 与 `p_ref` 必须为正，`lambda` 必须非负，否则返回 `None`。
    pub fn from_critical_state_line(
        void_ratio: f32,
        p_mean: f32,
        gamma: f32,
        lambda: f32,
        p_ref: f32,
    ) -> Option<Self> {
        if p_mean <= PRESSURE_EPS || p_ref <= PRESSURE_EPS || lambda < 0.0 {
            return None;
        }
        if !void_ratio.is_finite() || !gamma.is_finite() {
            return None;
        }
        // ln 属被禁用的 f32 超越函数，改在 f64 下计算后降精度。
        let ln_ratio = (f64::from(p_mean) / f64::from(p_ref)).ln() as f32;
        let critical_void_ratio = gamma - lambda * ln_ratio;
        Some(Self::from_void_ratios(void_ratio, critical_void_ratio))
    }

    /// 当前孔隙比 `e`。
    pub fn void_ratio(&self) -> f32 {
        self.void_ratio
    }

    /// 临界状态孔隙比 `e_cs`。
    pub fn critical_void_ratio(&self) -> f32 {
        self.critical_void_ratio
    }

    /// 状态参数 `ψ = e − e_cs`。
    pub fn state_parameter(&self) -> f32 {
        self.psi
    }

    /// 是否密实（`ψ < 0`，趋于剪胀）。
    pub fn is_dense(&self) -> bool {
        self.psi < 0.0
    }

    /// 是否疏松（`ψ > 0`，趋于剪缩）。
    pub fn is_loose(&self) -> bool {
        self.psi > 0.0
    }

    /// 是否在给定容差内接近临界状态（`|ψ| ≤ tol`）。
    pub fn is_at_critical(&self, tolerance: f32) -> bool {
        self.psi.abs() <= tolerance.max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-4;

    #[test]
    fn direct_void_ratio_difference() {
        let s = StateParameter::from_void_ratios(0.70, 0.85);
        assert!((s.state_parameter() - (-0.15)).abs() < TOL);
        assert!(s.is_dense());
        assert!(!s.is_loose());
    }

    #[test]
    fn loose_state_is_positive() {
        let s = StateParameter::from_void_ratios(0.95, 0.85);
        assert!((s.state_parameter() - 0.10).abs() < TOL);
        assert!(s.is_loose());
        assert!(!s.is_dense());
    }

    #[test]
    fn csl_at_reference_pressure_gives_gamma() {
        // p' = p_ref => ln(1)=0 => e_cs = Γ。
        let s = StateParameter::from_critical_state_line(0.80, 100.0, 0.90, 0.05, 100.0).unwrap();
        assert!((s.critical_void_ratio() - 0.90).abs() < TOL);
        assert!((s.state_parameter() - (-0.10)).abs() < TOL);
    }

    #[test]
    fn csl_higher_pressure_lowers_critical_void_ratio() {
        // p' = e·p_ref => ln=1 => e_cs = Γ − λ。
        let p = std::f32::consts::E * 100.0;
        let s = StateParameter::from_critical_state_line(0.80, p, 0.90, 0.05, 100.0).unwrap();
        assert!((s.critical_void_ratio() - (0.90 - 0.05)).abs() < 1e-3);
    }

    #[test]
    fn csl_is_monotonic_decreasing_in_pressure() {
        let low = StateParameter::from_critical_state_line(0.80, 50.0, 0.90, 0.05, 100.0).unwrap();
        let high =
            StateParameter::from_critical_state_line(0.80, 400.0, 0.90, 0.05, 100.0).unwrap();
        assert!(high.critical_void_ratio() < low.critical_void_ratio());
    }

    #[test]
    fn csl_rejects_nonpositive_pressure() {
        assert!(StateParameter::from_critical_state_line(0.8, 0.0, 0.9, 0.05, 100.0).is_none());
        assert!(StateParameter::from_critical_state_line(0.8, 100.0, 0.9, 0.05, 0.0).is_none());
        assert!(StateParameter::from_critical_state_line(0.8, -10.0, 0.9, 0.05, 100.0).is_none());
    }

    #[test]
    fn csl_rejects_negative_lambda() {
        assert!(StateParameter::from_critical_state_line(0.8, 100.0, 0.9, -0.05, 100.0).is_none());
    }

    #[test]
    fn at_critical_detects_zero_psi() {
        let s = StateParameter::from_void_ratios(0.85, 0.85);
        assert!(s.is_at_critical(1e-6));
        assert!(!s.is_dense());
        assert!(!s.is_loose());
    }
}
