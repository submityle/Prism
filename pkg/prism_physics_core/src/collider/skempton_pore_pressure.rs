//! Skempton 孔隙水压力系数 A、B（不排水孔压响应诊断，零耦合原语）。
//!
//! Skempton (1954) 用两个无量纲系数描述**不排水**加载时的超静孔隙水压力
//! 增量：
//! ```text
//! Δu = B · [ Δσ₃ + A · (Δσ₁ − Δσ₃) ]
//! ```
//! 其中 `Δσ₁`、`Δσ₃` 为总主应力增量（三轴意义下的大、小主应力）。
//!
//! - `B` 为**饱和度系数**：各向同性加载（`Δσ₁ = Δσ₃ = Δσ_c`）时
//!   `Δu = B·Δσ_c`，故 `B = Δu / Δσ_c`。完全饱和土 `B ≈ 1`，含气土 `B < 1`。
//!   由压缩性可写
//!   ```text
//!   B = 1 / (1 + n · C_w / C_s)
//!   ```
//!   `n` 为孔隙率，`C_w` 为孔隙流体压缩系数，`C_s` 为土骨架压缩系数。
//! - `A` 为**剪切系数**：偏应力阶段（`Δσ₃ = 0`，`Δσ_d = Δσ₁`）
//!   `Δu = B·A·Δσ_d`，故 `A = Δu / (B·Δσ_d)`。`A` 随应力历史变化：
//!   正常固结/松散土趋于较大正值（剪缩生正孔压），重超固结/密实土趋于负值
//!   （剪胀生负孔压）。
//!
//! 组合系数 `Ā = B·A`（直剪/偏载阶段 `Δu/Δσ_d`）亦常用。
//!
//! 本模块是**纯只读诊断**：不构造本构、不参与碰撞/求解管线，完全 0 耦合。

/// 低于该量级的应力增量视为零（除法退化保护）。
const STRESS_EPS: f32 = 1e-9;

/// Skempton 孔压系数对 `(A, B)` 及其派生诊断。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SkemptonPorePressure {
    a: f32,
    b: f32,
}

impl SkemptonPorePressure {
    /// 直接由系数对 `(A, B)` 构造。
    ///
    /// `B` 应落在 `[0, 1]` 的物理区间，但此处不强制裁剪，只做有限性校验，
    /// 以便诊断异常标定数据。
    pub fn from_coefficients(a: f32, b: f32) -> Option<Self> {
        if !a.is_finite() || !b.is_finite() {
            return None;
        }
        Some(Self { a, b })
    }

    /// 由各向同性加载阶段反算饱和度系数 `B = Δu / Δσ_c`。
    ///
    /// `delta_cell_stress` 为围压（各向同性总应力）增量，必须非零。
    pub fn b_from_isotropic(delta_cell_stress: f32, delta_pore_pressure: f32) -> Option<f32> {
        if !delta_cell_stress.is_finite() || !delta_pore_pressure.is_finite() {
            return None;
        }
        if delta_cell_stress.abs() <= STRESS_EPS {
            return None;
        }
        Some(delta_pore_pressure / delta_cell_stress)
    }

    /// 由偏应力阶段反算剪切系数 `A = Δu / (B·Δσ_d)`。
    ///
    /// `delta_deviator_stress` 为偏应力增量 `Δσ₁ − Δσ₃`（即恒围压下的轴向
    /// 偏载），`b` 为已知饱和度系数；二者乘积必须非零。
    pub fn a_from_deviatoric(
        delta_deviator_stress: f32,
        delta_pore_pressure: f32,
        b: f32,
    ) -> Option<f32> {
        if !delta_deviator_stress.is_finite() || !delta_pore_pressure.is_finite() || !b.is_finite()
        {
            return None;
        }
        let denom = b * delta_deviator_stress;
        if denom.abs() <= STRESS_EPS {
            return None;
        }
        Some(delta_pore_pressure / denom)
    }

    /// 由压缩性比估计饱和度系数 `B = 1 / (1 + n·C_w/C_s)`。
    ///
    /// 孔隙率 `n ∈ [0, 1]`，流体压缩系数 `C_w ≥ 0`，骨架压缩系数 `C_s > 0`。
    pub fn b_from_compressibilities(
        porosity: f32,
        fluid_compressibility: f32,
        skeleton_compressibility: f32,
    ) -> Option<f32> {
        if !porosity.is_finite()
            || !fluid_compressibility.is_finite()
            || !skeleton_compressibility.is_finite()
        {
            return None;
        }
        if !(0.0..=1.0).contains(&porosity)
            || fluid_compressibility < 0.0
            || skeleton_compressibility <= STRESS_EPS
        {
            return None;
        }
        let ratio = porosity * fluid_compressibility / skeleton_compressibility;
        Some(1.0 / (1.0 + ratio))
    }

    /// 由两阶段标定一次性构造 `(A, B)`：
    /// 先用各向同性阶段求 `B`，再用偏载阶段求 `A`。
    pub fn from_triaxial_stages(
        delta_cell_stress: f32,
        delta_pore_pressure_isotropic: f32,
        delta_deviator_stress: f32,
        delta_pore_pressure_deviatoric: f32,
    ) -> Option<Self> {
        let b = Self::b_from_isotropic(delta_cell_stress, delta_pore_pressure_isotropic)?;
        let a = Self::a_from_deviatoric(delta_deviator_stress, delta_pore_pressure_deviatoric, b)?;
        Self::from_coefficients(a, b)
    }

    /// 剪切系数 `A`。
    pub fn a_coefficient(&self) -> f32 {
        self.a
    }

    /// 饱和度系数 `B`。
    pub fn b_coefficient(&self) -> f32 {
        self.b
    }

    /// 组合系数 `Ā = B·A`（偏载阶段 `Δu/Δσ_d`）。
    pub fn combined_coefficient(&self) -> f32 {
        self.b * self.a
    }

    /// 对给定总主应力增量预测不排水超静孔压
    /// `Δu = B·[Δσ₃ + A·(Δσ₁ − Δσ₃)]`。
    pub fn predict_excess_pore_pressure(
        &self,
        delta_sigma_major: f32,
        delta_sigma_minor: f32,
    ) -> f32 {
        self.b * (delta_sigma_minor + self.a * (delta_sigma_major - delta_sigma_minor))
    }

    /// 是否近似完全饱和（`|B − 1| ≤ tol`）。
    pub fn is_saturated(&self, tolerance: f32) -> bool {
        (self.b - 1.0).abs() <= tolerance.max(0.0)
    }

    /// 剪切阶段是否趋于**剪缩生正孔压**（`A > 0`，常见于正常固结/松散土）。
    pub fn is_contractive(&self) -> bool {
        self.a > 0.0
    }

    /// 剪切阶段是否趋于**剪胀生负孔压**（`A < 0`，常见于重超固结/密实土）。
    pub fn is_dilative(&self) -> bool {
        self.a < 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-4;

    #[test]
    fn b_from_isotropic_fully_saturated() {
        // 围压 +50，孔压 +50 => B = 1。
        let b = SkemptonPorePressure::b_from_isotropic(50.0, 50.0).unwrap();
        assert!((b - 1.0).abs() < TOL);
    }

    #[test]
    fn b_from_isotropic_partial_saturation() {
        let b = SkemptonPorePressure::b_from_isotropic(100.0, 60.0).unwrap();
        assert!((b - 0.6).abs() < TOL);
    }

    #[test]
    fn b_from_isotropic_rejects_zero_cell() {
        assert!(SkemptonPorePressure::b_from_isotropic(0.0, 10.0).is_none());
    }

    #[test]
    fn a_from_deviatoric_recovers_coefficient() {
        // B=1, Δσ_d=100, Δu=30 => A = 0.3。
        let a = SkemptonPorePressure::a_from_deviatoric(100.0, 30.0, 1.0).unwrap();
        assert!((a - 0.3).abs() < TOL);
    }

    #[test]
    fn a_from_deviatoric_rejects_zero_denominator() {
        assert!(SkemptonPorePressure::a_from_deviatoric(100.0, 30.0, 0.0).is_none());
        assert!(SkemptonPorePressure::a_from_deviatoric(0.0, 30.0, 1.0).is_none());
    }

    #[test]
    fn b_from_compressibilities_saturated_limit() {
        // 流体不可压 C_w=0 => B = 1。
        let b = SkemptonPorePressure::b_from_compressibilities(0.4, 0.0, 1e-6).unwrap();
        assert!((b - 1.0).abs() < TOL);
    }

    #[test]
    fn b_from_compressibilities_monotonic_in_fluid_compressibility() {
        let stiff = SkemptonPorePressure::b_from_compressibilities(0.4, 1e-7, 1e-6).unwrap();
        let soft = SkemptonPorePressure::b_from_compressibilities(0.4, 1e-5, 1e-6).unwrap();
        // 流体越可压（含气越多），B 越小。
        assert!(soft < stiff);
        assert!(stiff <= 1.0 && soft >= 0.0);
    }

    #[test]
    fn b_from_compressibilities_rejects_bad_inputs() {
        assert!(SkemptonPorePressure::b_from_compressibilities(1.5, 1e-6, 1e-6).is_none());
        assert!(SkemptonPorePressure::b_from_compressibilities(0.4, -1.0, 1e-6).is_none());
        assert!(SkemptonPorePressure::b_from_compressibilities(0.4, 1e-6, 0.0).is_none());
    }

    #[test]
    fn predict_matches_skempton_equation() {
        let s = SkemptonPorePressure::from_coefficients(0.5, 1.0).unwrap();
        // Δσ₁=100, Δσ₃=40 => Δu = 1·[40 + 0.5·60] = 70。
        assert!((s.predict_excess_pore_pressure(100.0, 40.0) - 70.0).abs() < TOL);
    }

    #[test]
    fn predict_isotropic_reduces_to_b_times_stress() {
        let s = SkemptonPorePressure::from_coefficients(0.3, 0.8).unwrap();
        // Δσ₁=Δσ₃=50 => A 项消失 => Δu = B·50 = 40。
        assert!((s.predict_excess_pore_pressure(50.0, 50.0) - 40.0).abs() < TOL);
    }

    #[test]
    fn combined_coefficient_is_product() {
        let s = SkemptonPorePressure::from_coefficients(0.4, 0.9).unwrap();
        assert!((s.combined_coefficient() - 0.36).abs() < TOL);
    }

    #[test]
    fn saturation_and_tendency_flags() {
        let sat = SkemptonPorePressure::from_coefficients(0.6, 0.999).unwrap();
        assert!(sat.is_saturated(1e-2));
        assert!(sat.is_contractive());
        assert!(!sat.is_dilative());

        let heavily_oc = SkemptonPorePressure::from_coefficients(-0.2, 1.0).unwrap();
        assert!(heavily_oc.is_dilative());
        assert!(!heavily_oc.is_contractive());
    }

    #[test]
    fn two_stage_calibration_roundtrip() {
        // 各向同性: B=1；偏载: Δσ_d=200, Δu=50 => A=0.25。
        let s = SkemptonPorePressure::from_triaxial_stages(80.0, 80.0, 200.0, 50.0).unwrap();
        assert!((s.b_coefficient() - 1.0).abs() < TOL);
        assert!((s.a_coefficient() - 0.25).abs() < TOL);
    }

    #[test]
    fn from_coefficients_rejects_nonfinite() {
        assert!(SkemptonPorePressure::from_coefficients(f32::NAN, 1.0).is_none());
        assert!(SkemptonPorePressure::from_coefficients(0.3, f32::INFINITY).is_none());
    }
}
