//! 奥内佐格数 `Oh`（黏性相对惯性与表面张力的综合，零耦合原语）。
//!
//! 奥内佐格数把黏性、惯性、表面张力三者联系起来，是液滴破碎/喷溅图谱中与韦伯
//! 数并用的标准判据：
//!
//! ```text
//!   Oh = μ / √(ρ · σ · L)
//! ```
//!
//! 其中 `μ` 为动力黏度、`ρ` 为密度、`σ` 为表面张力、`L` 为特征长度（直径）。
//! 它等价于毛细数与韦伯数之比 `Oh = Ca / √We`：黏性越强、尺度越小，`Oh` 越大，
//! 黏性耗散越能抑制表面张力驱动的毛细振荡与破碎。
//!
//! - `Oh ≪ 0.1`：黏性可忽略，液滴动力学由惯性与表面张力主导。
//! - `Oh ≳ 1`：黏性显著，毛细振荡被强烈阻尼。
//!
//! 本原语与 [`super::capillary_number`]、[`super::weber_number`]、
//! [`super::granular_bond_number`] 同属表面张力无量纲族。
//!
//! 纯函数、零耦合：只做无量纲比值与阈值判定，不触碰主帧循环、不依赖渲染引擎。

/// 常用工程经验阈值：低于该值通常可忽略黏性影响。
pub const OHNESORGE_INVISCID_THRESHOLD: f32 = 0.1;

/// 奥内佐格数 `Oh = μ / √(ρ σ L)`。
///
/// 由 [`OhnesorgeNumber::from_properties`] 或
/// [`OhnesorgeNumber::from_capillary_and_weber`] 构造。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OhnesorgeNumber {
    value: f32,
}

impl OhnesorgeNumber {
    /// 由黏度、密度、表面张力与特征长度构造：`Oh = μ / √(ρ σ L)`。
    ///
    /// 要求 `ρ > 0`、`σ > 0`、`L > 0`、`μ ≥ 0`，且全部有限；否则返回 [`None`]。
    #[must_use]
    pub fn from_properties(
        viscosity: f32,
        density: f32,
        surface_tension: f32,
        length: f32,
    ) -> Option<Self> {
        if !viscosity.is_finite()
            || !density.is_finite()
            || !surface_tension.is_finite()
            || !length.is_finite()
        {
            return None;
        }
        if viscosity < 0.0 || density <= 0.0 || surface_tension <= 0.0 || length <= 0.0 {
            return None;
        }
        Some(Self {
            value: viscosity / (density * surface_tension * length).sqrt(),
        })
    }

    /// 由毛细数与韦伯数构造：`Oh = Ca / √We`。
    ///
    /// 要求 `We > 0`、`Ca ≥ 0`，且均为有限值；否则返回 [`None`]。
    #[must_use]
    pub fn from_capillary_and_weber(capillary: f32, weber: f32) -> Option<Self> {
        if !capillary.is_finite() || !weber.is_finite() {
            return None;
        }
        if capillary < 0.0 || weber <= 0.0 {
            return None;
        }
        Some(Self {
            value: capillary / weber.sqrt(),
        })
    }

    /// 奥内佐格数的无量纲值 `Oh ≥ 0`。
    #[must_use]
    pub fn value(&self) -> f32 {
        self.value
    }

    /// 按经验阈值 [`OHNESORGE_INVISCID_THRESHOLD`] 判定黏性是否可忽略（`Oh < 0.1`）。
    #[must_use]
    pub fn is_viscosity_negligible(&self) -> bool {
        self.value < OHNESORGE_INVISCID_THRESHOLD
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_is_analytic() {
        // μ=1e-3, ρ=1000, σ=0.072, L=1e-3 → Oh = 1e-3/√0.072 ≈ 0.0037268
        let oh = OhnesorgeNumber::from_properties(1.0e-3, 1000.0, 0.072, 1.0e-3).unwrap();
        let expected = 1.0e-3 / (0.072_f32).sqrt();
        assert!((oh.value() - expected).abs() < 1e-6);
        assert!(oh.is_viscosity_negligible());
    }

    #[test]
    fn matches_capillary_over_sqrt_weber() {
        // 取同一物性下的 Ca 与 We，验证 Oh = Ca/√We 与直接公式一致。
        // μ=0.01, ρ=1, σ=0.5, L=2, v=3 →
        //   Ca = μv/σ = 0.06;  We = ρv²L/σ = 1·9·2/0.5 = 36
        //   Oh = Ca/√We = 0.06/6 = 0.01;  直接式 = 0.01/√(1·0.5·2)=0.01/1=0.01
        let via_props = OhnesorgeNumber::from_properties(0.01, 1.0, 0.5, 2.0).unwrap();
        let via_ratio = OhnesorgeNumber::from_capillary_and_weber(0.06, 36.0).unwrap();
        assert!((via_props.value() - 0.01).abs() < 1e-6);
        assert!((via_ratio.value() - 0.01).abs() < 1e-6);
        assert!((via_props.value() - via_ratio.value()).abs() < 1e-6);
    }

    #[test]
    fn high_viscosity_is_not_negligible() {
        // μ=1, ρ=1, σ=1, L=1 → Oh=1
        let oh = OhnesorgeNumber::from_properties(1.0, 1.0, 1.0, 1.0).unwrap();
        assert!((oh.value() - 1.0).abs() < 1e-6);
        assert!(!oh.is_viscosity_negligible());
    }

    #[test]
    fn zero_viscosity_is_zero_ohnesorge() {
        let oh = OhnesorgeNumber::from_properties(0.0, 1000.0, 0.072, 1.0e-3).unwrap();
        assert_eq!(oh.value(), 0.0);
        assert!(oh.is_viscosity_negligible());
    }

    #[test]
    fn rejects_bad_properties() {
        assert!(OhnesorgeNumber::from_properties(1.0, 0.0, 1.0, 1.0).is_none());
        assert!(OhnesorgeNumber::from_properties(1.0, 1.0, 0.0, 1.0).is_none());
        assert!(OhnesorgeNumber::from_properties(1.0, 1.0, 1.0, 0.0).is_none());
        assert!(OhnesorgeNumber::from_properties(-1.0, 1.0, 1.0, 1.0).is_none());
        assert!(OhnesorgeNumber::from_properties(f32::NAN, 1.0, 1.0, 1.0).is_none());
    }

    #[test]
    fn rejects_bad_ratio_inputs() {
        assert!(OhnesorgeNumber::from_capillary_and_weber(0.1, 0.0).is_none());
        assert!(OhnesorgeNumber::from_capillary_and_weber(0.1, -1.0).is_none());
        assert!(OhnesorgeNumber::from_capillary_and_weber(-0.1, 1.0).is_none());
        assert!(OhnesorgeNumber::from_capillary_and_weber(f32::INFINITY, 1.0).is_none());
    }
}
