//! 韦伯数 `We`（惯性—表面张力竞争，零耦合原语）。
//!
//! 韦伯数刻画流体**惯性**与**表面张力**之比，常用于液滴碰撞、喷溅、湿颗粒
//! 冲击等场景：
//!
//! ```text
//!   We = ρ · v² · L / σ
//! ```
//!
//! 其中 `ρ` 为密度、`v` 为特征相对速度、`L` 为特征长度（液滴/颗粒直径）、
//! `σ` 为气–液表面张力。
//!
//! - `We ≪ 1`：表面张力主导，液滴/液膜倾向保持完整并回弹。
//! - `We ≫ 1`：惯性主导，液面被显著拉伸、易破碎飞溅。
//!
//! 本原语与 [`super::capillary_number`]（黏性/σ）、[`super::granular_bond_number`]
//! （重力/σ）构成表面张力三件套，三者分母同为表面张力、分子分别为惯性/黏性/重力。
//!
//! 纯函数、零耦合：只做无量纲比值与阈值判定，不触碰主帧循环、不依赖渲染引擎。

/// 区分表面张力主导与惯性主导的临界韦伯数。
pub const WEBER_CRITICAL: f32 = 1.0;

/// 由韦伯数判定的流动状态。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeberRegime {
    /// `We < 1`：表面张力主导，液面倾向保持完整。
    SurfaceTensionDominated,
    /// `We ≥ 1`：惯性主导，液面易被拉伸破碎。
    InertiaDominated,
}

/// 韦伯数 `We = ρ v² L / σ` 及其状态分类。
///
/// 由 [`WeberNumber::from_flow`] 构造。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeberNumber {
    value: f32,
}

impl WeberNumber {
    /// 由密度、特征相对速度、特征长度与表面张力构造：`We = ρ · v² · L / σ`。
    ///
    /// 要求 `σ > 0`、`ρ ≥ 0`、`L ≥ 0`，且全部有限；否则返回 [`None`]。速度取平方
    /// （与运动方向无关）。
    #[must_use]
    pub fn from_flow(
        density: f32,
        velocity: f32,
        length: f32,
        surface_tension: f32,
    ) -> Option<Self> {
        if !density.is_finite()
            || !velocity.is_finite()
            || !length.is_finite()
            || !surface_tension.is_finite()
        {
            return None;
        }
        if density < 0.0 || length < 0.0 || surface_tension <= 0.0 {
            return None;
        }
        Some(Self {
            value: density * velocity * velocity * length / surface_tension,
        })
    }

    /// 韦伯数的无量纲值 `We ≥ 0`。
    #[must_use]
    pub fn value(&self) -> f32 {
        self.value
    }

    /// 按临界值 [`WEBER_CRITICAL`] 判定的流动状态。
    #[must_use]
    pub fn regime(&self) -> WeberRegime {
        if self.value < WEBER_CRITICAL {
            WeberRegime::SurfaceTensionDominated
        } else {
            WeberRegime::InertiaDominated
        }
    }

    /// 是否表面张力主导（`We < 1`）。
    #[must_use]
    pub fn is_surface_tension_dominated(&self) -> bool {
        matches!(self.regime(), WeberRegime::SurfaceTensionDominated)
    }

    /// 是否惯性主导（`We ≥ 1`）。
    #[must_use]
    pub fn is_inertia_dominated(&self) -> bool {
        matches!(self.regime(), WeberRegime::InertiaDominated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_is_analytic() {
        // ρ=1000, v=2, L=0.001, σ=0.072 → We = 1000·4·0.001/0.072 ≈ 55.56
        let we = WeberNumber::from_flow(1000.0, 2.0, 0.001, 0.072).unwrap();
        assert!((we.value() - (4.0 / 0.072)).abs() < 1e-2);
        assert!(we.is_inertia_dominated());
        assert_eq!(we.regime(), WeberRegime::InertiaDominated);
    }

    #[test]
    fn velocity_sign_is_irrelevant() {
        let a = WeberNumber::from_flow(1.0, 3.0, 2.0, 0.5).unwrap();
        let b = WeberNumber::from_flow(1.0, -3.0, 2.0, 0.5).unwrap();
        assert_eq!(a.value(), b.value());
    }

    #[test]
    fn surface_tension_dominated_below_unity() {
        // ρ=1, v=0.1, L=0.01, σ=1 → We = 1·0.01·0.01/1 = 1e-4
        let we = WeberNumber::from_flow(1.0, 0.1, 0.01, 1.0).unwrap();
        assert!(we.value() < WEBER_CRITICAL);
        assert!(we.is_surface_tension_dominated());
    }

    #[test]
    fn critical_value_is_inertia_dominated() {
        // ρ=1, v=1, L=1, σ=1 → We=1 恰好归入惯性主导。
        let we = WeberNumber::from_flow(1.0, 1.0, 1.0, 1.0).unwrap();
        assert!((we.value() - 1.0).abs() < 1e-6);
        assert!(we.is_inertia_dominated());
    }

    #[test]
    fn zero_velocity_is_zero_weber() {
        let we = WeberNumber::from_flow(1000.0, 0.0, 0.01, 0.072).unwrap();
        assert_eq!(we.value(), 0.0);
        assert!(we.is_surface_tension_dominated());
    }

    #[test]
    fn rejects_bad_inputs() {
        assert!(WeberNumber::from_flow(1.0, 1.0, 1.0, 0.0).is_none());
        assert!(WeberNumber::from_flow(1.0, 1.0, 1.0, -0.1).is_none());
        assert!(WeberNumber::from_flow(-1.0, 1.0, 1.0, 1.0).is_none());
        assert!(WeberNumber::from_flow(1.0, 1.0, -1.0, 1.0).is_none());
        assert!(WeberNumber::from_flow(f32::NAN, 1.0, 1.0, 1.0).is_none());
        assert!(WeberNumber::from_flow(1.0, f32::INFINITY, 1.0, 1.0).is_none());
    }
}
