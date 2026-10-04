//! 应力三轴度与应力状态分类（延性断裂力学的 `η`–`μ_L` 刻画）。
//!
//! 应力三轴度 `η` 定义为静水（平均）应力与 von Mises 等效应力之比：
//!
//! ```text
//!        p      σ_m
//!   η = --- = -------
//!        q      σ_vM
//! ```
//!
//! 它是延性断裂与损伤力学中刻画应力状态「拉/压/剪倾向」的核心无量纲量
//! （Bai–Wierzbicki）。几个标准参考值（拉为正约定）：
//!
//! - 单轴拉伸 `η = +1/3`。
//! - 纯剪 `η = 0`。
//! - 单轴压缩 `η = −1/3`。
//! - 等双轴拉伸 `η = +2/3`。
//!
//! 据此把应力状态按 `η` 相对单轴参考值 `±1/3` 分为拉伸主导 / 剪切主导 /
//! 压缩主导三类，并辅以 Lode-Nadai 参数 `μ_L`（三轴压缩 `−1`、纯剪 `0`、
//! 三轴拉伸 `+1`）给出应力「形状」。
//!
//! 纯函数式、零耦合：在 [`super::stress_invariants::StressInvariants`] 之上组合
//! `p`、`q`、`μ_L`，不触碰主帧循环、不依赖渲染引擎。静水应力状态（`q ≈ 0`）下
//! 三轴度无定义，构造返回 `None`。

use super::stress_invariants::StressInvariants;

/// 低于该值的 von Mises 等效应力视为静水状态，三轴度无定义。
const DEVIATORIC_EPS: f32 = 1e-9;

/// 用于分类的单轴参考三轴度阈值 `1/3`。
const UNIAXIAL_TRIAXIALITY: f32 = 1.0 / 3.0;

/// 由主应力导出的应力三轴度与状态分类。
///
/// 由 [`StressTriaxiality::from_principal_stresses`] 构造。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StressTriaxiality {
    triaxiality: f32,
    mean_stress: f32,
    von_mises: f32,
    lode_parameter: Option<f32>,
}

/// 以单轴参考三轴度 `±1/3` 划分的应力状态分类。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriaxialityState {
    /// `η ≥ +1/3`：拉伸主导（单轴拉伸及更强的拉伸三轴度）。
    TensionDominated,
    /// `|η| < 1/3`：剪切主导（纯剪附近）。
    ShearDominated,
    /// `η ≤ −1/3`：压缩主导（单轴压缩及更强的压缩三轴度）。
    CompressionDominated,
}

impl StressTriaxiality {
    /// 由一组主应力（顺序任意，拉为正）构造三轴度分解。
    ///
    /// 当任一分量非有限、或处于静水状态（`q ≤ ε`，三轴度无定义）时返回 `None`。
    pub fn from_principal_stresses(principal_stresses: [f32; 3]) -> Option<Self> {
        let invariants = StressInvariants::from_principal_stresses(principal_stresses)?;
        let von_mises = invariants.von_mises_equivalent();
        if von_mises <= DEVIATORIC_EPS {
            return None;
        }
        let mean_stress = invariants.mean_stress();
        Some(Self {
            triaxiality: mean_stress / von_mises,
            mean_stress,
            von_mises,
            lode_parameter: invariants.lode_parameter(),
        })
    }

    /// 应力三轴度 `η = p / q`。
    pub fn triaxiality(&self) -> f32 {
        self.triaxiality
    }

    /// 平均（静水）应力 `p`。
    pub fn mean_stress(&self) -> f32 {
        self.mean_stress
    }

    /// von Mises 等效应力 `q`。
    pub fn von_mises_stress(&self) -> f32 {
        self.von_mises
    }

    /// Lode-Nadai 参数 `μ_L`（纯剪时可能因 `σ1 ≈ σ3` 而无定义）。
    pub fn lode_parameter(&self) -> Option<f32> {
        self.lode_parameter
    }

    /// 以单轴参考三轴度 `±1/3` 划分的应力状态分类。
    pub fn state(&self) -> TriaxialityState {
        if self.triaxiality >= UNIAXIAL_TRIAXIALITY {
            TriaxialityState::TensionDominated
        } else if self.triaxiality <= -UNIAXIAL_TRIAXIALITY {
            TriaxialityState::CompressionDominated
        } else {
            TriaxialityState::ShearDominated
        }
    }

    /// 是否拉伸主导（`η ≥ +1/3`）。
    pub fn is_tension_dominated(&self) -> bool {
        self.triaxiality >= UNIAXIAL_TRIAXIALITY
    }

    /// 是否压缩主导（`η ≤ −1/3`）。
    pub fn is_compression_dominated(&self) -> bool {
        self.triaxiality <= -UNIAXIAL_TRIAXIALITY
    }

    /// 是否剪切主导（`|η| < 1/3`）。
    pub fn is_shear_dominated(&self) -> bool {
        self.triaxiality.abs() < UNIAXIAL_TRIAXIALITY
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    #[test]
    fn uniaxial_tension_is_one_third() {
        // σ = [s, 0, 0] -> p = s/3, q = s -> η = 1/3
        let t = StressTriaxiality::from_principal_stresses([3.0, 0.0, 0.0]).unwrap();
        assert!((t.triaxiality() - 1.0 / 3.0).abs() <= EPS);
        assert_eq!(t.state(), TriaxialityState::TensionDominated);
        assert!(t.is_tension_dominated());
    }

    #[test]
    fn uniaxial_compression_is_negative_one_third() {
        // σ = [0, 0, -s] -> p = -s/3, q = s -> η = -1/3
        let t = StressTriaxiality::from_principal_stresses([0.0, 0.0, -3.0]).unwrap();
        assert!((t.triaxiality() + 1.0 / 3.0).abs() <= EPS);
        assert_eq!(t.state(), TriaxialityState::CompressionDominated);
        assert!(t.is_compression_dominated());
    }

    #[test]
    fn pure_shear_is_zero() {
        // σ = [s, 0, -s] -> p = 0, η = 0, μ_L = 0
        let t = StressTriaxiality::from_principal_stresses([2.0, 0.0, -2.0]).unwrap();
        assert!(t.triaxiality().abs() <= EPS);
        assert_eq!(t.state(), TriaxialityState::ShearDominated);
        assert!(t.is_shear_dominated());
        assert!((t.lode_parameter().unwrap()).abs() <= EPS);
    }

    #[test]
    fn equibiaxial_tension_is_two_thirds() {
        // σ = [s, s, 0] -> p = 2s/3, q = s -> η = 2/3
        let t = StressTriaxiality::from_principal_stresses([2.0, 2.0, 0.0]).unwrap();
        assert!((t.triaxiality() - 2.0 / 3.0).abs() <= EPS);
        assert_eq!(t.state(), TriaxialityState::TensionDominated);
    }

    #[test]
    fn mean_and_von_mises_pass_through() {
        // σ = [3, 0, 0]: p = 1, q = 3
        let t = StressTriaxiality::from_principal_stresses([3.0, 0.0, 0.0]).unwrap();
        assert!((t.mean_stress() - 1.0).abs() <= EPS);
        assert!((t.von_mises_stress() - 3.0).abs() <= EPS);
    }

    #[test]
    fn hydrostatic_state_is_rejected() {
        // σ = [s, s, s] -> q = 0 -> triaxiality undefined
        assert!(StressTriaxiality::from_principal_stresses([5.0, 5.0, 5.0]).is_none());
    }

    #[test]
    fn rejects_non_finite() {
        assert!(StressTriaxiality::from_principal_stresses([f32::NAN, 0.0, 0.0]).is_none());
        assert!(StressTriaxiality::from_principal_stresses([f32::INFINITY, 0.0, 0.0]).is_none());
    }
}
