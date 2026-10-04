//! 理查森数 `Ri`（密度分层—剪切稳定性，零耦合原语）。
//!
//! 理查森数刻画**浮力（密度分层）**与**剪切惯性**之比，判定分层流动的稳定性。
//! 常用两种等价形式：
//!
//! ```text
//!   Ri = g' · L / U²          （整体形式，g' 为约化重力）
//!   Ri = N² / (du/dz)²        （梯度形式，N² 为浮力频率平方）
//! ```
//!
//! 其中约化重力 `g' = g · Δρ / ρ_ref`。经典 Miles–Howard 判据给出临界值
//! `Ri_c = 0.25`：
//!
//! - `Ri < 0`：顶重底轻，**对流失稳**。
//! - `0 ≤ Ri < 0.25`：剪切主导，**动力失稳**（易湍动/掺混）。
//! - `Ri ≥ 0.25`：浮力稳定分层，剪切不足以破坏分层。
//!
//! 对分层颗粒流（粗细分离、密度偏析）同样适用。纯函数、零耦合：只做无量纲比值与
//! 阈值判定，不触碰主帧循环、不依赖渲染引擎。

/// Miles–Howard 稳定性临界理查森数。
pub const RICHARDSON_CRITICAL: f32 = 0.25;

/// 低于该绝对值的分母（速度/剪切率）视为病态，无法定义理查森数。
const DENOMINATOR_EPS: f32 = 1e-12;

/// 由理查森数判定的分层流动稳定性状态。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RichardsonRegime {
    /// `Ri < 0`：顶重底轻，对流失稳。
    ConvectivelyUnstable,
    /// `0 ≤ Ri < 0.25`：剪切主导，动力失稳（掺混）。
    ShearDominated,
    /// `Ri ≥ 0.25`：浮力稳定分层。
    BuoyancyStabilized,
}

/// 理查森数 `Ri` 及其稳定性分类。
///
/// 由 [`RichardsonNumber::from_bulk`]、[`RichardsonNumber::from_gradient`] 或
/// [`RichardsonNumber::from_density`] 构造。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RichardsonNumber {
    value: f32,
}

impl RichardsonNumber {
    /// 整体形式：`Ri = g' · L / U²`。
    ///
    /// `g'` 可为负（不稳定分层）。要求 `L ≥ 0`、`|U| > 0`，且三者均有限；否则
    /// 返回 [`None`]。
    #[must_use]
    pub fn from_bulk(reduced_gravity: f32, length: f32, velocity: f32) -> Option<Self> {
        if !reduced_gravity.is_finite() || !length.is_finite() || !velocity.is_finite() {
            return None;
        }
        if length < 0.0 || velocity.abs() <= DENOMINATOR_EPS {
            return None;
        }
        Some(Self {
            value: reduced_gravity * length / (velocity * velocity),
        })
    }

    /// 梯度形式：`Ri = N² / (du/dz)²`。
    ///
    /// `N²`（浮力频率平方）可为负（不稳定分层）。要求 `|du/dz| > 0`，且两者有限；
    /// 否则返回 [`None`]。
    #[must_use]
    pub fn from_gradient(buoyancy_frequency_sq: f32, shear_rate: f32) -> Option<Self> {
        if !buoyancy_frequency_sq.is_finite() || !shear_rate.is_finite() {
            return None;
        }
        if shear_rate.abs() <= DENOMINATOR_EPS {
            return None;
        }
        Some(Self {
            value: buoyancy_frequency_sq / (shear_rate * shear_rate),
        })
    }

    /// 由重力、密度差、参考密度、长度与速度构造：先得约化重力
    /// `g' = g · Δρ / ρ_ref`，再套整体形式。
    ///
    /// 要求 `ρ_ref > 0`、`L ≥ 0`、`|U| > 0`，且全部有限；否则返回 [`None`]。
    #[must_use]
    pub fn from_density(
        gravity: f32,
        delta_density: f32,
        reference_density: f32,
        length: f32,
        velocity: f32,
    ) -> Option<Self> {
        if !gravity.is_finite() || !delta_density.is_finite() || !reference_density.is_finite() {
            return None;
        }
        if reference_density <= 0.0 {
            return None;
        }
        let reduced_gravity = gravity * delta_density / reference_density;
        Self::from_bulk(reduced_gravity, length, velocity)
    }

    /// 理查森数的无量纲值（可为负）。
    #[must_use]
    pub fn value(&self) -> f32 {
        self.value
    }

    /// 按 Miles–Howard 临界值判定的稳定性状态。
    #[must_use]
    pub fn regime(&self) -> RichardsonRegime {
        if self.value < 0.0 {
            RichardsonRegime::ConvectivelyUnstable
        } else if self.value < RICHARDSON_CRITICAL {
            RichardsonRegime::ShearDominated
        } else {
            RichardsonRegime::BuoyancyStabilized
        }
    }

    /// 是否浮力稳定分层（`Ri ≥ 0.25`）。
    #[must_use]
    pub fn is_stably_stratified(&self) -> bool {
        matches!(self.regime(), RichardsonRegime::BuoyancyStabilized)
    }

    /// 是否动力失稳（`Ri < 0.25`，含对流失稳）。
    #[must_use]
    pub fn is_dynamically_unstable(&self) -> bool {
        self.value < RICHARDSON_CRITICAL
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bulk_value_is_analytic() {
        // g'=10, L=2, U=4 → Ri = 20/16 = 1.25 稳定分层
        let ri = RichardsonNumber::from_bulk(10.0, 2.0, 4.0).unwrap();
        assert!((ri.value() - 1.25).abs() < 1e-6);
        assert!(ri.is_stably_stratified());
        assert_eq!(ri.regime(), RichardsonRegime::BuoyancyStabilized);
    }

    #[test]
    fn shear_dominated_below_quarter() {
        // g'=1, L=1, U=4 → Ri = 1/16 = 0.0625
        let ri = RichardsonNumber::from_bulk(1.0, 1.0, 4.0).unwrap();
        assert!((ri.value() - 0.0625).abs() < 1e-6);
        assert_eq!(ri.regime(), RichardsonRegime::ShearDominated);
        assert!(ri.is_dynamically_unstable());
        assert!(!ri.is_stably_stratified());
    }

    #[test]
    fn negative_reduced_gravity_is_convectively_unstable() {
        // g'=-5, L=1, U=2 → Ri=-1.25
        let ri = RichardsonNumber::from_bulk(-5.0, 1.0, 2.0).unwrap();
        assert!((ri.value() + 1.25).abs() < 1e-6);
        assert_eq!(ri.regime(), RichardsonRegime::ConvectivelyUnstable);
        assert!(ri.is_dynamically_unstable());
    }

    #[test]
    fn critical_quarter_is_stably_stratified() {
        // g'=1, L=1, U=2 → Ri=0.25 恰好归入稳定分层。
        let ri = RichardsonNumber::from_bulk(1.0, 1.0, 2.0).unwrap();
        assert!((ri.value() - 0.25).abs() < 1e-6);
        assert!(ri.is_stably_stratified());
    }

    #[test]
    fn gradient_form_is_analytic() {
        // N²=0.09, du/dz=0.3 → Ri = 0.09/0.09 = 1.0
        let ri = RichardsonNumber::from_gradient(0.09, 0.3).unwrap();
        assert!((ri.value() - 1.0).abs() < 1e-6);
        assert!(ri.is_stably_stratified());
    }

    #[test]
    fn density_form_matches_reduced_gravity() {
        // g=9.81, Δρ=100, ρ=1000 → g'=0.981; L=2, U=4 → Ri=0.981·2/16=0.122625
        let ri = RichardsonNumber::from_density(9.81, 100.0, 1000.0, 2.0, 4.0).unwrap();
        assert!((ri.value() - 0.122_625).abs() < 1e-5);
        assert_eq!(ri.regime(), RichardsonRegime::ShearDominated);
    }

    #[test]
    fn rejects_bad_inputs() {
        assert!(RichardsonNumber::from_bulk(1.0, 1.0, 0.0).is_none());
        assert!(RichardsonNumber::from_bulk(1.0, -1.0, 2.0).is_none());
        assert!(RichardsonNumber::from_gradient(1.0, 0.0).is_none());
        assert!(RichardsonNumber::from_density(9.81, 100.0, 0.0, 2.0, 4.0).is_none());
        assert!(RichardsonNumber::from_bulk(f32::NAN, 1.0, 2.0).is_none());
        assert!(RichardsonNumber::from_gradient(f32::INFINITY, 1.0).is_none());
    }
}
