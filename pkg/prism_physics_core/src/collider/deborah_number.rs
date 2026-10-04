//! 德博拉数 `De`（材料弛豫时间与观测/流动时间之比，零耦合原语）。
//!
//! 德博拉数刻画材料**本构弛豫时间** `λ` 与**特征观测/流动时间** `t_c` 之比，
//! 判定在给定时间尺度上材料偏向固态还是流态：
//!
//! ```text
//!   De = λ / t_c           （时间尺度形式）
//!   De = λ · γ̇            （剪切率形式，t_c = 1/γ̇）
//! ```
//!
//! - `De ≪ 1`：观测时间远长于弛豫，材料来得及松弛，表现为**流态**（黏性流动）。
//! - `De ≫ 1`：观测时间远短于弛豫，材料来不及松弛，表现为**固态**（弹性）。
//!
//! 对湿颗粒/黏弹性泥浆/密堆积颗粒，`De` 与惯性数 `I`、黏性数等共同界定本构区。
//!
//! 纯函数、零耦合：只做无量纲比值与阈值判定，不触碰主帧循环、不依赖渲染引擎。

/// 区分流态与固态的临界德博拉数。
pub const DEBORAH_CRITICAL: f32 = 1.0;

/// 由德博拉数判定的材料响应状态。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeborahRegime {
    /// `De < 1`：弛豫快于观测，材料表现为流态（黏性流动）。
    FluidLike,
    /// `De ≥ 1`：弛豫慢于观测，材料表现为固态（弹性）。
    SolidLike,
}

/// 德博拉数 `De = λ / t_c` 及其状态分类。
///
/// 由 [`DeborahNumber::from_times`] 或 [`DeborahNumber::from_relaxation_and_rate`] 构造。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeborahNumber {
    value: f32,
}

impl DeborahNumber {
    /// 由弛豫时间与观测/流动时间构造：`De = λ / t_c`。
    ///
    /// 要求 `t_c > 0`、`λ ≥ 0`，且均为有限值；否则返回 [`None`]。
    #[must_use]
    pub fn from_times(relaxation_time: f32, observation_time: f32) -> Option<Self> {
        if !relaxation_time.is_finite() || !observation_time.is_finite() {
            return None;
        }
        if relaxation_time < 0.0 || observation_time <= 0.0 {
            return None;
        }
        Some(Self {
            value: relaxation_time / observation_time,
        })
    }

    /// 由弛豫时间与剪切率构造：`De = λ · |γ̇|`（取 `t_c = 1/γ̇`）。
    ///
    /// 要求 `λ ≥ 0`，且 `λ` 与 `γ̇` 均为有限值；否则返回 [`None`]。
    #[must_use]
    pub fn from_relaxation_and_rate(relaxation_time: f32, shear_rate: f32) -> Option<Self> {
        if !relaxation_time.is_finite() || !shear_rate.is_finite() {
            return None;
        }
        if relaxation_time < 0.0 {
            return None;
        }
        Some(Self {
            value: relaxation_time * shear_rate.abs(),
        })
    }

    /// 德博拉数的无量纲值 `De ≥ 0`。
    #[must_use]
    pub fn value(&self) -> f32 {
        self.value
    }

    /// 按临界值 [`DEBORAH_CRITICAL`] 判定的材料响应状态。
    #[must_use]
    pub fn regime(&self) -> DeborahRegime {
        if self.value < DEBORAH_CRITICAL {
            DeborahRegime::FluidLike
        } else {
            DeborahRegime::SolidLike
        }
    }

    /// 是否流态（`De < 1`）。
    #[must_use]
    pub fn is_fluid_like(&self) -> bool {
        matches!(self.regime(), DeborahRegime::FluidLike)
    }

    /// 是否固态（`De ≥ 1`）。
    #[must_use]
    pub fn is_solid_like(&self) -> bool {
        matches!(self.regime(), DeborahRegime::SolidLike)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn time_ratio_is_analytic() {
        // λ=0.5, t_c=0.1 → De=5 固态
        let de = DeborahNumber::from_times(0.5, 0.1).unwrap();
        assert!((de.value() - 5.0).abs() < 1e-6);
        assert!(de.is_solid_like());
        assert_eq!(de.regime(), DeborahRegime::SolidLike);
    }

    #[test]
    fn shear_rate_form_matches_time_form() {
        // De = λ·γ̇ = 0.2·0.5 = 0.1，等价于 λ/t_c，t_c=1/γ̇=2
        let via_rate = DeborahNumber::from_relaxation_and_rate(0.2, 0.5).unwrap();
        let via_time = DeborahNumber::from_times(0.2, 2.0).unwrap();
        assert!((via_rate.value() - 0.1).abs() < 1e-6);
        assert!((via_rate.value() - via_time.value()).abs() < 1e-6);
        assert!(via_rate.is_fluid_like());
    }

    #[test]
    fn shear_rate_sign_is_irrelevant() {
        let a = DeborahNumber::from_relaxation_and_rate(0.3, 2.0).unwrap();
        let b = DeborahNumber::from_relaxation_and_rate(0.3, -2.0).unwrap();
        assert_eq!(a.value(), b.value());
    }

    #[test]
    fn critical_value_is_solid_like() {
        // λ=1, t_c=1 → De=1 恰好归入固态。
        let de = DeborahNumber::from_times(1.0, 1.0).unwrap();
        assert!((de.value() - 1.0).abs() < 1e-6);
        assert!(de.is_solid_like());
    }

    #[test]
    fn zero_relaxation_is_fluid() {
        let de = DeborahNumber::from_times(0.0, 0.5).unwrap();
        assert_eq!(de.value(), 0.0);
        assert!(de.is_fluid_like());
    }

    #[test]
    fn rejects_bad_inputs() {
        assert!(DeborahNumber::from_times(1.0, 0.0).is_none());
        assert!(DeborahNumber::from_times(1.0, -0.1).is_none());
        assert!(DeborahNumber::from_times(-1.0, 1.0).is_none());
        assert!(DeborahNumber::from_relaxation_and_rate(-1.0, 1.0).is_none());
        assert!(DeborahNumber::from_times(f32::NAN, 1.0).is_none());
        assert!(DeborahNumber::from_relaxation_and_rate(1.0, f32::INFINITY).is_none());
    }
}
