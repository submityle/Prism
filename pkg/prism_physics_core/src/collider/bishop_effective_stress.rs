//! Bishop 非饱和有效应力参数 χ（有效应力诊断，零耦合原语）。
//!
//! Bishop (1959) 将 Terzaghi 有效应力推广到**非饱和土**：
//! ```text
//! σ' = (σ − u_a) + χ·(u_a − u_w)
//! ```
//! 其中 `σ` 为总应力，`u_a` 为孔隙气压，`u_w` 为孔隙水压：
//! - `σ − u_a` 为**净应力**（net stress）；
//! - `u_a − u_w` 为**基质吸力**（matric suction）；
//! - `χ ∈ [0, 1]` 为 Bishop 参数：`χ = 1` 对应完全饱和（退化为
//!   Terzaghi `σ' = σ − u_w`），`χ = 0` 对应完全干燥（`σ' = σ − u_a`）。
//!
//! `χ` 常由有效饱和度或吸力经验式估计：
//! ```text
//! χ = S_e = (S_r − S_r,res) / (1 − S_r,res)        (有效饱和度法)
//! χ = (s / s_ae)^(−0.55),  s > s_ae; 否则 χ = 1    (Khalili & Khabbaz, 1998)
//! ```
//!
//! 本模块是**纯只读诊断**：不构造本构、不参与碰撞/求解管线，完全 0 耦合。

/// 低于该量级的吸力阈值/饱和度残差视为退化。
const EPS: f32 = 1e-9;

/// Khalili & Khabbaz (1998) 吸力幂律指数。
const KHALILI_EXPONENT: f64 = -0.55;

/// Bishop 非饱和有效应力诊断结果。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BishopEffectiveStress {
    total_stress: f32,
    pore_air_pressure: f32,
    pore_water_pressure: f32,
    chi: f32,
}

impl BishopEffectiveStress {
    /// 由总应力、孔隙气压、孔隙水压与 Bishop 参数 χ 构造。
    ///
    /// 约定应力/压力均为同一正负号习惯（本模块对符号不敏感，直接代入
    /// Bishop 公式）。`chi` 必须落在 `[0, 1]`，所有量必须有限。
    pub fn from_components(
        total_stress: f32,
        pore_air_pressure: f32,
        pore_water_pressure: f32,
        chi: f32,
    ) -> Option<Self> {
        if !total_stress.is_finite()
            || !pore_air_pressure.is_finite()
            || !pore_water_pressure.is_finite()
            || !chi.is_finite()
        {
            return None;
        }
        if !(0.0..=1.0).contains(&chi) {
            return None;
        }
        Some(Self {
            total_stress,
            pore_air_pressure,
            pore_water_pressure,
            chi,
        })
    }

    /// 由有效饱和度估计 `χ = (S_r − S_r,res)/(1 − S_r,res)`。
    ///
    /// `saturation` 与 `residual_saturation` 均在 `[0, 1]`，且残差 < 1。
    /// 结果裁剪到 `[0, 1]`。
    pub fn chi_from_effective_saturation(saturation: f32, residual_saturation: f32) -> Option<f32> {
        if !saturation.is_finite() || !residual_saturation.is_finite() {
            return None;
        }
        if !(0.0..=1.0).contains(&saturation) || !(0.0..1.0).contains(&residual_saturation) {
            return None;
        }
        let denom = 1.0 - residual_saturation;
        if denom <= EPS {
            return None;
        }
        Some(((saturation - residual_saturation) / denom).clamp(0.0, 1.0))
    }

    /// 由 Khalili & Khabbaz (1998) 吸力幂律估计 χ：
    /// `s ≤ s_ae` 时 χ = 1，否则 `χ = (s/s_ae)^(−0.55)`。
    ///
    /// 吸力 `suction ≥ 0`，进气值 `air_entry_suction > 0`。
    pub fn chi_from_khalili_khabbaz(suction: f32, air_entry_suction: f32) -> Option<f32> {
        if !suction.is_finite() || !air_entry_suction.is_finite() {
            return None;
        }
        if suction < 0.0 || air_entry_suction <= EPS {
            return None;
        }
        if suction <= air_entry_suction {
            return Some(1.0);
        }
        // powf 为被禁用的 f32 超越函数，改在 f64 下计算后降精度。
        let ratio = f64::from(suction) / f64::from(air_entry_suction);
        let chi = ratio.powf(KHALILI_EXPONENT) as f32;
        Some(chi.clamp(0.0, 1.0))
    }

    /// 有效应力 `σ' = (σ − u_a) + χ·(u_a − u_w)`。
    pub fn effective_stress(&self) -> f32 {
        self.net_stress() + self.chi * self.matric_suction()
    }

    /// 净应力 `σ − u_a`。
    pub fn net_stress(&self) -> f32 {
        self.total_stress - self.pore_air_pressure
    }

    /// 基质吸力 `u_a − u_w`。
    pub fn matric_suction(&self) -> f32 {
        self.pore_air_pressure - self.pore_water_pressure
    }

    /// Bishop 参数 χ。
    pub fn chi(&self) -> f32 {
        self.chi
    }

    /// 饱和极限下的 Terzaghi 参照有效应力 `σ − u_w`（χ = 1 时与本模型一致）。
    pub fn terzaghi_effective_stress(&self) -> f32 {
        self.total_stress - self.pore_water_pressure
    }

    /// 是否接近饱和极限（`|χ − 1| ≤ tol`）。
    pub fn is_saturated_limit(&self, tolerance: f32) -> bool {
        (self.chi - 1.0).abs() <= tolerance.max(0.0)
    }

    /// 是否接近干燥极限（`χ ≤ tol`）。
    pub fn is_dry_limit(&self, tolerance: f32) -> bool {
        self.chi <= tolerance.max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-4;

    #[test]
    fn effective_stress_matches_bishop_equation() {
        // σ=100, u_a=20, u_w=−30 => net=80, suction=50, χ=0.5 => σ'=80+25=105。
        let s = BishopEffectiveStress::from_components(100.0, 20.0, -30.0, 0.5).unwrap();
        assert!((s.net_stress() - 80.0).abs() < TOL);
        assert!((s.matric_suction() - 50.0).abs() < TOL);
        assert!((s.effective_stress() - 105.0).abs() < TOL);
    }

    #[test]
    fn saturated_limit_reduces_to_terzaghi() {
        // χ=1 => σ' = σ − u_w。
        let s = BishopEffectiveStress::from_components(100.0, 20.0, 5.0, 1.0).unwrap();
        assert!((s.effective_stress() - s.terzaghi_effective_stress()).abs() < TOL);
        assert!((s.effective_stress() - 95.0).abs() < TOL);
        assert!(s.is_saturated_limit(1e-6));
    }

    #[test]
    fn dry_limit_reduces_to_net_stress() {
        // χ=0 => σ' = σ − u_a。
        let s = BishopEffectiveStress::from_components(100.0, 20.0, -50.0, 0.0).unwrap();
        assert!((s.effective_stress() - 80.0).abs() < TOL);
        assert!(s.is_dry_limit(1e-6));
    }

    #[test]
    fn rejects_chi_out_of_range() {
        assert!(BishopEffectiveStress::from_components(100.0, 0.0, 0.0, 1.5).is_none());
        assert!(BishopEffectiveStress::from_components(100.0, 0.0, 0.0, -0.1).is_none());
    }

    #[test]
    fn rejects_nonfinite() {
        assert!(BishopEffectiveStress::from_components(f32::NAN, 0.0, 0.0, 0.5).is_none());
        assert!(BishopEffectiveStress::from_components(100.0, 0.0, 0.0, f32::INFINITY).is_none());
    }

    #[test]
    fn chi_effective_saturation_endpoints() {
        // S_r = S_r,res => χ = 0。
        let dry = BishopEffectiveStress::chi_from_effective_saturation(0.1, 0.1).unwrap();
        assert!(dry.abs() < TOL);
        // S_r = 1 => χ = 1。
        let wet = BishopEffectiveStress::chi_from_effective_saturation(1.0, 0.1).unwrap();
        assert!((wet - 1.0).abs() < TOL);
        // 中间值: (0.55−0.1)/(1−0.1)=0.5。
        let mid = BishopEffectiveStress::chi_from_effective_saturation(0.55, 0.1).unwrap();
        assert!((mid - 0.5).abs() < TOL);
    }

    #[test]
    fn chi_effective_saturation_rejects_bad_inputs() {
        assert!(BishopEffectiveStress::chi_from_effective_saturation(1.2, 0.1).is_none());
        assert!(BishopEffectiveStress::chi_from_effective_saturation(0.5, 1.0).is_none());
    }

    #[test]
    fn chi_khalili_below_air_entry_is_one() {
        let chi = BishopEffectiveStress::chi_from_khalili_khabbaz(50.0, 100.0).unwrap();
        assert!((chi - 1.0).abs() < TOL);
    }

    #[test]
    fn chi_khalili_decreases_with_suction() {
        let near = BishopEffectiveStress::chi_from_khalili_khabbaz(200.0, 100.0).unwrap();
        let far = BishopEffectiveStress::chi_from_khalili_khabbaz(800.0, 100.0).unwrap();
        assert!(far < near);
        assert!(near <= 1.0 && far >= 0.0);
        // 解析校验: (200/100)^(−0.55) = 2^(−0.55)。
        let expected = 2.0_f64.powf(-0.55) as f32;
        assert!((near - expected).abs() < 1e-3);
    }

    #[test]
    fn chi_khalili_rejects_bad_inputs() {
        assert!(BishopEffectiveStress::chi_from_khalili_khabbaz(-10.0, 100.0).is_none());
        assert!(BishopEffectiveStress::chi_from_khalili_khabbaz(100.0, 0.0).is_none());
    }
}
