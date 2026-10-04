//! 毛细数 `Ca`（湿颗粒/悬浮液黏性—表面张力竞争，零耦合原语）。
//!
//! 毛细数刻画间隙液体的**黏性拖曳**与**表面张力**之比：
//!
//! ```text
//!   Ca = μ · v / σ
//! ```
//!
//! 其中 `μ` 为液体动力黏度、`v` 为特征相对速度、`σ` 为气–液表面张力。
//! 对以剪切率 `γ̇` 与特征长度 `L` 描述的流动，等价地写作
//! `Ca = μ · γ̇ · L / σ`。
//!
//! - `Ca ≪ 1`：表面张力主导，毛细桥能维持、液面近似准静态（湿颗粒团聚显著）。
//! - `Ca ≫ 1`：黏性拖曳主导，毛细桥被剪断、液面被显著拉伸变形。
//!
//! 纯函数、零耦合：只做无量纲比值与阈值判定，不触碰主帧循环、不依赖渲染引擎。
//! 与 [`super::granular_bond_number`]（毛细力/重力）互补——此处衡量的是黏性而非重力。

/// 区分表面张力主导与黏性主导的临界毛细数。
pub const CAPILLARY_CRITICAL: f32 = 1.0;

/// 由毛细数判定的流动状态。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapillaryRegime {
    /// `Ca < 1`：表面张力主导，毛细桥/液面近似准静态。
    SurfaceTensionDominated,
    /// `Ca ≥ 1`：黏性拖曳主导，毛细桥被剪断、液面显著变形。
    ViscousDominated,
}

/// 毛细数 `Ca = μ v / σ` 及其状态分类。
///
/// 由 [`CapillaryNumber::from_viscous_drag`] 或 [`CapillaryNumber::from_shear`] 构造。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CapillaryNumber {
    value: f32,
}

impl CapillaryNumber {
    /// 由黏度、特征相对速度与表面张力构造：`Ca = μ · |v| / σ`。
    ///
    /// 要求 `σ > 0`、`μ ≥ 0`，且三者均为有限值；否则返回 [`None`]。速度取绝对值
    /// （毛细数与相对运动方向无关）。
    #[must_use]
    pub fn from_viscous_drag(viscosity: f32, velocity: f32, surface_tension: f32) -> Option<Self> {
        if !viscosity.is_finite() || !velocity.is_finite() || !surface_tension.is_finite() {
            return None;
        }
        if viscosity < 0.0 || surface_tension <= 0.0 {
            return None;
        }
        Some(Self {
            value: viscosity * velocity.abs() / surface_tension,
        })
    }

    /// 由黏度、剪切率、特征长度与表面张力构造：`Ca = μ · |γ̇| · L / σ`。
    ///
    /// 要求 `σ > 0`、`μ ≥ 0`、`L ≥ 0`，且全部有限；否则返回 [`None`]。
    #[must_use]
    pub fn from_shear(
        viscosity: f32,
        shear_rate: f32,
        length: f32,
        surface_tension: f32,
    ) -> Option<Self> {
        if !length.is_finite() || length < 0.0 || !shear_rate.is_finite() {
            return None;
        }
        Self::from_viscous_drag(viscosity, shear_rate.abs() * length, surface_tension)
    }

    /// 毛细数的无量纲值 `Ca ≥ 0`。
    #[must_use]
    pub fn value(&self) -> f32 {
        self.value
    }

    /// 按临界值 [`CAPILLARY_CRITICAL`] 判定的流动状态。
    #[must_use]
    pub fn regime(&self) -> CapillaryRegime {
        if self.value < CAPILLARY_CRITICAL {
            CapillaryRegime::SurfaceTensionDominated
        } else {
            CapillaryRegime::ViscousDominated
        }
    }

    /// 是否表面张力主导（`Ca < 1`）。
    #[must_use]
    pub fn is_surface_tension_dominated(&self) -> bool {
        matches!(self.regime(), CapillaryRegime::SurfaceTensionDominated)
    }

    /// 是否黏性拖曳主导（`Ca ≥ 1`）。
    #[must_use]
    pub fn is_viscous_dominated(&self) -> bool {
        matches!(self.regime(), CapillaryRegime::ViscousDominated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn viscous_drag_value_is_analytic() {
        // μ=0.1, v=2, σ=0.05 → Ca = 0.1·2/0.05 = 4.0
        let ca = CapillaryNumber::from_viscous_drag(0.1, 2.0, 0.05).unwrap();
        assert!((ca.value() - 4.0).abs() < 1e-5);
        assert!(ca.is_viscous_dominated());
        assert_eq!(ca.regime(), CapillaryRegime::ViscousDominated);
    }

    #[test]
    fn shear_form_matches_viscous_form() {
        // Ca = μ·γ̇·L/σ = 0.02·3·5/0.3 = 1.0
        let ca = CapillaryNumber::from_shear(0.02, 3.0, 5.0, 0.3).unwrap();
        assert!((ca.value() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn velocity_sign_is_irrelevant() {
        let a = CapillaryNumber::from_viscous_drag(0.1, 2.0, 0.05).unwrap();
        let b = CapillaryNumber::from_viscous_drag(0.1, -2.0, 0.05).unwrap();
        assert_eq!(a.value(), b.value());
    }

    #[test]
    fn surface_tension_dominated_below_unity() {
        // μ=0.01, v=1, σ=1 → Ca=0.01
        let ca = CapillaryNumber::from_viscous_drag(0.01, 1.0, 1.0).unwrap();
        assert!(ca.value() < CAPILLARY_CRITICAL);
        assert!(ca.is_surface_tension_dominated());
        assert_eq!(ca.regime(), CapillaryRegime::SurfaceTensionDominated);
    }

    #[test]
    fn critical_value_is_viscous_dominated() {
        // 恰好 Ca=1 归入黏性主导（≥ 临界）。
        let ca = CapillaryNumber::from_viscous_drag(0.5, 2.0, 1.0).unwrap();
        assert!((ca.value() - 1.0).abs() < 1e-6);
        assert!(ca.is_viscous_dominated());
    }

    #[test]
    fn zero_velocity_is_zero_capillary() {
        let ca = CapillaryNumber::from_viscous_drag(0.1, 0.0, 0.05).unwrap();
        assert_eq!(ca.value(), 0.0);
        assert!(ca.is_surface_tension_dominated());
    }

    #[test]
    fn rejects_bad_inputs() {
        // 非正表面张力
        assert!(CapillaryNumber::from_viscous_drag(0.1, 1.0, 0.0).is_none());
        assert!(CapillaryNumber::from_viscous_drag(0.1, 1.0, -0.2).is_none());
        // 负黏度
        assert!(CapillaryNumber::from_viscous_drag(-0.1, 1.0, 1.0).is_none());
        // 负长度
        assert!(CapillaryNumber::from_shear(0.1, 1.0, -1.0, 1.0).is_none());
        // 非有限
        assert!(CapillaryNumber::from_viscous_drag(f32::NAN, 1.0, 1.0).is_none());
        assert!(CapillaryNumber::from_viscous_drag(0.1, f32::INFINITY, 1.0).is_none());
    }
}
