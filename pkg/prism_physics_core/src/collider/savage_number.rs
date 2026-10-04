//! Savage 数（Savage–Hutter 数）——区分摩擦准静态流与碰撞惯性流的无量纲判据。
//!
//! Savage（1984）提出用碰撞（惯性）应力与准静态/摩擦应力之比来刻画颗粒流
//! 所处的区制：
//!
//! ```text
//! N_sav = ρ_s · d² · γ̇² / P
//! ```
//!
//! 其中 `ρ_s` 为颗粒固体密度，`d` 为颗粒直径，`γ̇` 为剪切率，`P` 为围压
//! （法向/有效应力）。分子 `ρ_s d² γ̇²` 量级为颗粒惯性/碰撞应力，分母 `P`
//! 为约束颗粒的准静态压力。
//!
//! 经典 Savage–Hutter 判据以 `N_sav ≈ 0.1` 为摩擦—碰撞转捩点：
//! - `N_sav < 0.1`：准静态/摩擦应力主导（缓慢、持续接触的致密流）。
//! - `N_sav > 0.1`：碰撞/惯性应力主导（快速、稀疏碰撞的 grain-inertia 流）。
//!
//! 与惯性数的关系：当 `P` 取围压时，惯性数 `I = γ̇ d / √(P/ρ_s)`，于是
//! `N_sav = I²`，二者给出一致的区制判断（本模块的测试显式验证这一恒等式）。
//!
//! 这是一个纯函数式、零耦合的分析原语：仅做一次代数求值与阈值分类，不依赖
//! 任何求解器状态或时间推进，可被 DEM / 连续介质流变等上游复用。
//!
//! 它与 [`super::granular_flow_regime`] 互补：后者用间隙流体黏度（Bagnold 数）
//! 区分 macro-viscous / grain-inertia，本模块用围压区分摩擦 / 碰撞主导。

/// Savage–Hutter 摩擦—碰撞转捩的临界值 `N_sav ≈ 0.1`。
pub const SAVAGE_CRITICAL: f32 = 0.1;

/// 由 Savage 数判定的颗粒流区制。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SavageRegime {
    /// 准静态/摩擦应力主导（`N_sav < 0.1`）：缓慢、持续接触的致密流。
    FrictionalQuasiStatic,
    /// 碰撞/惯性应力主导（`N_sav ≥ 0.1`）：快速、稀疏碰撞的 grain-inertia 流。
    Collisional,
}

/// 由颗粒流状态推导出的 Savage 数及其区制分类。
///
/// 使用 [`SavageNumber::from_state`] 构造。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SavageNumber {
    solid_density: f32,
    grain_diameter: f32,
    shear_rate: f32,
    normal_stress: f32,
    savage_number: f32,
}

impl SavageNumber {
    /// 由固体密度、颗粒直径、剪切率与围压（法向应力）构造。
    ///
    /// 当任意输入非有限，或 `ρ_s ≤ 0`、`d ≤ 0`、`γ̇ < 0`、`P ≤ 0` 时返回 `None`。
    #[must_use]
    pub fn from_state(
        solid_density: f32,
        grain_diameter: f32,
        shear_rate: f32,
        normal_stress: f32,
    ) -> Option<Self> {
        if !solid_density.is_finite()
            || !grain_diameter.is_finite()
            || !shear_rate.is_finite()
            || !normal_stress.is_finite()
        {
            return None;
        }
        if solid_density <= 0.0 || grain_diameter <= 0.0 || shear_rate < 0.0 || normal_stress <= 0.0
        {
            return None;
        }

        // N_sav = ρ_s · d² · γ̇² / P。
        let savage_number =
            solid_density * grain_diameter * grain_diameter * shear_rate * shear_rate
                / normal_stress;
        if !savage_number.is_finite() {
            return None;
        }

        Some(Self {
            solid_density,
            grain_diameter,
            shear_rate,
            normal_stress,
            savage_number,
        })
    }

    /// 固体密度 `ρ_s`。
    #[must_use]
    pub fn solid_density(&self) -> f32 {
        self.solid_density
    }

    /// 颗粒直径 `d`。
    #[must_use]
    pub fn grain_diameter(&self) -> f32 {
        self.grain_diameter
    }

    /// 剪切率 `γ̇`。
    #[must_use]
    pub fn shear_rate(&self) -> f32 {
        self.shear_rate
    }

    /// 围压（法向/有效应力）`P`。
    #[must_use]
    pub fn normal_stress(&self) -> f32 {
        self.normal_stress
    }

    /// Savage 数 `N_sav = ρ_s d² γ̇² / P`。
    #[must_use]
    pub fn savage_number(&self) -> f32 {
        self.savage_number
    }

    /// 当前区制分类。
    #[must_use]
    pub fn regime(&self) -> SavageRegime {
        if self.savage_number < SAVAGE_CRITICAL {
            SavageRegime::FrictionalQuasiStatic
        } else {
            SavageRegime::Collisional
        }
    }

    /// 是否摩擦/准静态主导（`N_sav < 0.1`）。
    #[must_use]
    pub fn is_frictional_dominated(&self) -> bool {
        matches!(self.regime(), SavageRegime::FrictionalQuasiStatic)
    }

    /// 是否碰撞/惯性主导（`N_sav ≥ 0.1`）。
    #[must_use]
    pub fn is_collisional_dominated(&self) -> bool {
        matches!(self.regime(), SavageRegime::Collisional)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hand_computed_value() {
        // ρ_s=2500, d=0.002, γ̇=10, P=1000。
        // N_sav = 2500 · (0.002)² · 100 / 1000 = 2500·4e-6·100/1000 = 1e-3。
        let s = SavageNumber::from_state(2500.0, 0.002, 10.0, 1000.0).unwrap();
        assert!((s.savage_number() - 1.0e-3).abs() <= 1e-9);
        assert_eq!(s.regime(), SavageRegime::FrictionalQuasiStatic);
        assert!(s.is_frictional_dominated());
        assert!(!s.is_collisional_dominated());
    }

    #[test]
    fn collisional_regime_above_threshold() {
        // 高剪切率 → 大 N_sav。ρ_s=2500,d=0.01,γ̇=50,P=500。
        // N_sav = 2500·1e-4·2500/500 = 2500·1e-4·5 = 1.25。
        let s = SavageNumber::from_state(2500.0, 0.01, 50.0, 500.0).unwrap();
        assert!((s.savage_number() - 1.25).abs() <= 1e-5);
        assert_eq!(s.regime(), SavageRegime::Collisional);
        assert!(s.is_collisional_dominated());
    }

    #[test]
    fn threshold_boundary_is_collisional() {
        // 构造 N_sav 恰为 0.1：ρ_s=1,d=1,γ̇=√0.1,P=1 → N_sav=0.1。
        let gamma = (0.1_f64).sqrt() as f32;
        let s = SavageNumber::from_state(1.0, 1.0, gamma, 1.0).unwrap();
        assert!((s.savage_number() - 0.1).abs() <= 1e-6);
        // 0.1 不小于 0.1 → Collisional（半开区间约定）。
        assert_eq!(s.regime(), SavageRegime::Collisional);
    }

    #[test]
    fn equals_inertial_number_squared() {
        // 恒等式 N_sav = I²，其中 I = γ̇ d / √(P/ρ_s)。
        let (rho, d, gamma, p) = (2650.0_f32, 0.003_f32, 25.0_f32, 2000.0_f32);
        let s = SavageNumber::from_state(rho, d, gamma, p).unwrap();
        let inertial = gamma * d / (p / rho).sqrt();
        assert!((s.savage_number() - inertial * inertial).abs() <= 1e-4);
    }

    #[test]
    fn scales_with_shear_rate_squared() {
        let base = SavageNumber::from_state(2000.0, 0.005, 10.0, 1500.0).unwrap();
        let doubled = SavageNumber::from_state(2000.0, 0.005, 20.0, 1500.0).unwrap();
        // γ̇ 翻倍 → N_sav 变为 4 倍。
        assert!((doubled.savage_number() - 4.0 * base.savage_number()).abs() <= 1e-4);
    }

    #[test]
    fn zero_shear_is_quasi_static() {
        let s = SavageNumber::from_state(2000.0, 0.005, 0.0, 1500.0).unwrap();
        assert!(s.savage_number().abs() <= 1e-12);
        assert_eq!(s.regime(), SavageRegime::FrictionalQuasiStatic);
    }

    #[test]
    fn rejects_invalid_inputs() {
        assert!(SavageNumber::from_state(0.0, 0.002, 10.0, 1000.0).is_none());
        assert!(SavageNumber::from_state(2500.0, 0.0, 10.0, 1000.0).is_none());
        assert!(SavageNumber::from_state(2500.0, 0.002, -1.0, 1000.0).is_none());
        assert!(SavageNumber::from_state(2500.0, 0.002, 10.0, 0.0).is_none());
        assert!(SavageNumber::from_state(f32::NAN, 0.002, 10.0, 1000.0).is_none());
        assert!(SavageNumber::from_state(2500.0, 0.002, 10.0, f32::INFINITY).is_none());
    }
}
