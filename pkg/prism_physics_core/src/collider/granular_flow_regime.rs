//! 颗粒-流体悬浮流的流态分类（Bagnold 数 / Stokes 数）。
//!
//! 当颗粒在粘性间隙流体中剪切流动时，其宏观应力由两种机制竞争主导：
//! 流体粘性应力与颗粒碰撞（惯性）应力。Bagnold（1954）用无量纲 Bagnold 数
//! 刻画二者之比，并据此把流态划分为宏观粘性区、过渡区与颗粒惯性区。
//!
//! 本模块采用广泛引用的简化形式（省略线性浓度 λ 的约定项）：
//!
//! ```text
//! Ba = rho_s * d^2 * shear_rate / mu_f          （Bagnold 数）
//! St = rho_s * d^2 * shear_rate / (18 * mu_f)   （Stokes 数 = Ba / 18）
//! ```
//!
//! 其中 `rho_s` 为颗粒材料密度，`d` 为颗粒直径，`shear_rate` 为剪切率 `γ̇`，
//! `mu_f` 为间隙流体动力粘度。Bagnold 的经典实验阈值：
//!
//! * **宏观粘性区**（粘性主导）：`Ba < 40`
//! * **过渡区**：`40 ≤ Ba ≤ 450`
//! * **颗粒惯性区**（碰撞主导）：`Ba > 450`
//!
//! 该分类与干颗粒的惯性数 `I`（见 [`crate::collider::granular_rheology`]）互补：
//! 惯性数描述无间隙流体的干颗粒流，而 Bagnold/Stokes 数显式引入流体粘度，
//! 刻画湿颗粒/悬浮流的流态。本模块仅做纯无量纲数计算，不触碰主帧循环、
//! 不依赖渲染引擎，可独立使用。

/// Bagnold 数宏观粘性区上界。
const MACRO_VISCOUS_UPPER: f32 = 40.0;
/// Bagnold 数过渡区上界（含）。
const TRANSITIONAL_UPPER: f32 = 450.0;
/// Stokes 数相对 Bagnold 数的约化系数。
const STOKES_DIVISOR: f32 = 18.0;

/// 颗粒-流体悬浮流的流态类别。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlowRegime {
    /// 宏观粘性区：流体粘性应力主导（`Ba < 40`）。
    MacroViscous,
    /// 过渡区：粘性与惯性应力相当（`40 ≤ Ba ≤ 450`）。
    Transitional,
    /// 颗粒惯性区：颗粒碰撞应力主导（`Ba > 450`）。
    GrainInertia,
}

/// 基于 Bagnold/Stokes 数的颗粒悬浮流流态分类器。
///
/// 由 [`GranularFlowRegime::from_state`] 构造，缓存 Bagnold 数并据此提供
/// Stokes 数与流态判别。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GranularFlowRegime {
    bagnold_number: f32,
}

impl GranularFlowRegime {
    /// 由颗粒-流体状态构造流态分类器。
    ///
    /// 要求 `rho_s > 0`、`d > 0`、`shear_rate >= 0`、`mu_f > 0`，且全部有限；
    /// 否则返回 `None`。
    #[must_use]
    pub fn from_state(rho_s: f32, d: f32, shear_rate: f32, mu_f: f32) -> Option<Self> {
        if !rho_s.is_finite()
            || !d.is_finite()
            || !shear_rate.is_finite()
            || !mu_f.is_finite()
            || rho_s <= 0.0
            || d <= 0.0
            || shear_rate < 0.0
            || mu_f <= 0.0
        {
            return None;
        }
        let bagnold_number = rho_s * d * d * shear_rate / mu_f;
        Some(Self { bagnold_number })
    }

    /// Bagnold 数 `Ba = rho_s · d² · γ̇ / mu_f`。
    #[must_use]
    pub fn bagnold_number(&self) -> f32 {
        self.bagnold_number
    }

    /// Stokes 数 `St = Ba / 18`。
    #[must_use]
    pub fn stokes_number(&self) -> f32 {
        self.bagnold_number / STOKES_DIVISOR
    }

    /// 依 Bagnold 经典阈值判定流态。
    #[must_use]
    pub fn regime(&self) -> FlowRegime {
        let ba = self.bagnold_number;
        if ba < MACRO_VISCOUS_UPPER {
            FlowRegime::MacroViscous
        } else if ba <= TRANSITIONAL_UPPER {
            FlowRegime::Transitional
        } else {
            FlowRegime::GrainInertia
        }
    }

    /// 是否处于宏观粘性区。
    #[must_use]
    pub fn is_macro_viscous(&self) -> bool {
        self.regime() == FlowRegime::MacroViscous
    }

    /// 是否处于颗粒惯性区。
    #[must_use]
    pub fn is_grain_inertia(&self) -> bool {
        self.regime() == FlowRegime::GrainInertia
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_state() {
        assert!(GranularFlowRegime::from_state(0.0, 0.001, 100.0, 0.001).is_none());
        assert!(GranularFlowRegime::from_state(2000.0, 0.0, 100.0, 0.001).is_none());
        assert!(GranularFlowRegime::from_state(2000.0, 0.001, -1.0, 0.001).is_none());
        assert!(GranularFlowRegime::from_state(2000.0, 0.001, 100.0, 0.0).is_none());
        assert!(GranularFlowRegime::from_state(f32::NAN, 0.001, 100.0, 0.001).is_none());
    }

    #[test]
    fn transitional_hand_value() {
        // rho_s=2000, d=0.001, γ̇=100, mu_f=0.001 → Ba = 200。
        let r = GranularFlowRegime::from_state(2000.0, 0.001, 100.0, 0.001).unwrap();
        assert!(
            (r.bagnold_number() - 200.0).abs() < 1e-3,
            "Ba = {}",
            r.bagnold_number()
        );
        // St = 200 / 18 ≈ 11.111
        assert!((r.stokes_number() - 200.0 / 18.0).abs() < 1e-4);
        assert_eq!(r.regime(), FlowRegime::Transitional);
    }

    #[test]
    fn macro_viscous_regime() {
        // 极细颗粒、低剪切 → Ba = 0.02 ≪ 40。
        let r = GranularFlowRegime::from_state(2000.0, 0.0001, 1.0, 0.001).unwrap();
        assert!((r.bagnold_number() - 0.02).abs() < 1e-4);
        assert_eq!(r.regime(), FlowRegime::MacroViscous);
        assert!(r.is_macro_viscous());
        assert!(!r.is_grain_inertia());
    }

    #[test]
    fn grain_inertia_regime() {
        // 粗颗粒、高剪切 → Ba = 20000 ≫ 450。
        let r = GranularFlowRegime::from_state(2000.0, 0.01, 100.0, 0.001).unwrap();
        assert!((r.bagnold_number() - 20000.0).abs() < 1.0);
        assert_eq!(r.regime(), FlowRegime::GrainInertia);
        assert!(r.is_grain_inertia());
        assert!(!r.is_macro_viscous());
    }

    #[test]
    fn boundary_values_classify_consistently() {
        // 恰为 40（宏观粘性上界，非含）→ 过渡区。
        let at_40 = GranularFlowRegime::from_state(40.0, 1.0, 1.0, 1.0).unwrap();
        assert!((at_40.bagnold_number() - 40.0).abs() < 1e-5);
        assert_eq!(at_40.regime(), FlowRegime::Transitional);
        // 恰为 450（过渡区上界，含）→ 过渡区。
        let at_450 = GranularFlowRegime::from_state(450.0, 1.0, 1.0, 1.0).unwrap();
        assert_eq!(at_450.regime(), FlowRegime::Transitional);
        // 略高于 450 → 颗粒惯性区。
        let above = GranularFlowRegime::from_state(451.0, 1.0, 1.0, 1.0).unwrap();
        assert_eq!(above.regime(), FlowRegime::GrainInertia);
    }

    #[test]
    fn bagnold_scales_linearly_with_shear_rate() {
        let base = GranularFlowRegime::from_state(2000.0, 0.001, 50.0, 0.001).unwrap();
        let doubled = GranularFlowRegime::from_state(2000.0, 0.001, 100.0, 0.001).unwrap();
        assert!((doubled.bagnold_number() / base.bagnold_number() - 2.0).abs() < 1e-4);
    }

    #[test]
    fn zero_shear_rate_is_macro_viscous() {
        let r = GranularFlowRegime::from_state(2000.0, 0.01, 0.0, 0.001).unwrap();
        assert_eq!(r.bagnold_number(), 0.0);
        assert_eq!(r.stokes_number(), 0.0);
        assert_eq!(r.regime(), FlowRegime::MacroViscous);
    }
}
