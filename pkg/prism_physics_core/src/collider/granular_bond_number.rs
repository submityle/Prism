//! 颗粒 Bond 数（粘聚数）：颗粒间粘聚力与颗粒自重的无量纲比值。
//!
//! 对湿颗粒、细粉等粘聚性体系，颗粒间毛细/范德华/固桥等粘聚力 `F_c` 相对于
//! 颗粒自重 `W = m·g` 的大小，决定了体系是“自由流动”还是“成团粘聚”。该无量纲
//! 比值称为（颗粒）Bond 数：
//!
//! ```text
//!        粘聚力     F_c
//!   Bo = ------ = -----
//!        自  重     m·g
//! ```
//!
//! - `Bo ≪ 1`：自重主导，颗粒近似无粘流动（自由流动）。
//! - `Bo ≫ 1`：粘聚力主导，颗粒易成团、形成稳定团簇与架桥。
//! - `Bo ≈ 1`：粘聚与重力相当的过渡区。
//!
//! 对球形颗粒，质量由直径与密度给出 `m = ρ_s · (π/6) · d³`，于是
//! `Bo = F_c / (ρ_s · (π/6) · d³ · g)`。
//!
//! 纯函数式、零耦合的分析原语：仅对已给出的力与颗粒属性做无量纲化，不依赖任何
//! 求解器状态或时间推进。可消费 [`super::capillary_bridge`] 等模块给出的粘聚力。

/// 球体体积系数 `π/6`，用于由直径与密度计算球形颗粒质量。
const SPHERE_VOLUME_COEFF: f32 = std::f32::consts::PI / 6.0;

/// 颗粒 Bond 数及其构造标度。
///
/// 使用 [`GranularBondNumber::from_force_weight`]（直接给定质量）或
/// [`GranularBondNumber::from_sphere`]（由直径与密度推算质量）构造。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GranularBondNumber {
    cohesive_force: f32,
    particle_weight: f32,
    bond: f32,
}

/// Bond 数判定的颗粒体系机制。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BondRegime {
    /// `Bo < 1`：自重主导，自由流动。
    FreeFlowing,
    /// `Bo ≥ 1`：粘聚力主导，成团粘聚。
    Cohesive,
}

/// 粘聚与重力相当的 Bond 阈值。
pub const BOND_CRITICAL: f32 = 1.0;

impl GranularBondNumber {
    /// 由粘聚力 `cohesive_force`、颗粒质量 `particle_mass` 与重力加速度
    /// `gravity` 构造 `Bo = F_c / (m·g)`。
    ///
    /// 要求全部参数有限、`particle_mass > 0`、`gravity > 0`、`cohesive_force ≥ 0`；
    /// 否则返回 `None`。
    pub fn from_force_weight(
        cohesive_force: f32,
        particle_mass: f32,
        gravity: f32,
    ) -> Option<Self> {
        if !cohesive_force.is_finite() || !particle_mass.is_finite() || !gravity.is_finite() {
            return None;
        }
        if cohesive_force < 0.0 || particle_mass <= 0.0 || gravity <= 0.0 {
            return None;
        }
        let weight = particle_mass * gravity;
        let bond = cohesive_force / weight;
        Some(Self {
            cohesive_force,
            particle_weight: weight,
            bond,
        })
    }

    /// 由粘聚力 `cohesive_force`、球形颗粒直径 `grain_diameter`、颗粒密度
    /// `grain_density` 与重力加速度 `gravity` 构造 Bond 数。
    ///
    /// 质量由 `m = ρ_s · (π/6) · d³` 给出。要求全部参数有限、`grain_diameter > 0`、
    /// `grain_density > 0`、`gravity > 0`、`cohesive_force ≥ 0`；否则返回 `None`。
    pub fn from_sphere(
        cohesive_force: f32,
        grain_diameter: f32,
        grain_density: f32,
        gravity: f32,
    ) -> Option<Self> {
        if !grain_diameter.is_finite() || !grain_density.is_finite() {
            return None;
        }
        if grain_diameter <= 0.0 || grain_density <= 0.0 {
            return None;
        }
        let volume = SPHERE_VOLUME_COEFF * grain_diameter * grain_diameter * grain_diameter;
        let mass = grain_density * volume;
        Self::from_force_weight(cohesive_force, mass, gravity)
    }

    /// Bond 数 `Bo`（无量纲，恒非负）。
    pub fn bond_number(&self) -> f32 {
        self.bond
    }

    /// 构造所用的粘聚力 `F_c`。
    pub fn cohesive_force(&self) -> f32 {
        self.cohesive_force
    }

    /// 颗粒自重 `W = m·g`。
    pub fn particle_weight(&self) -> f32 {
        self.particle_weight
    }

    /// 当前颗粒体系机制。
    pub fn regime(&self) -> BondRegime {
        if self.bond < BOND_CRITICAL {
            BondRegime::FreeFlowing
        } else {
            BondRegime::Cohesive
        }
    }

    /// 是否粘聚主导（`Bo ≥ 1`）。
    pub fn is_cohesive(&self) -> bool {
        self.bond >= BOND_CRITICAL
    }

    /// 是否自重主导、自由流动（`Bo < 1`）。
    pub fn is_free_flowing(&self) -> bool {
        self.bond < BOND_CRITICAL
    }

    /// 是否处于粘聚-重力平衡（`|Bo − 1| ≤ tol`）。
    pub fn is_balanced(&self, tol: f32) -> bool {
        (self.bond - BOND_CRITICAL).abs() <= tol
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    #[test]
    fn force_weight_matches_definition() {
        // F=6, m=2, g=10 -> W=20, Bo=0.3
        let bo = GranularBondNumber::from_force_weight(6.0, 2.0, 10.0).unwrap();
        assert!((bo.particle_weight() - 20.0).abs() <= EPS);
        assert!((bo.bond_number() - 0.3).abs() <= EPS);
        assert_eq!(bo.regime(), BondRegime::FreeFlowing);
        assert!(bo.is_free_flowing());
        assert!(!bo.is_cohesive());
    }

    #[test]
    fn cohesive_above_unity() {
        // F=50, m=2, g=10 -> W=20, Bo=2.5
        let bo = GranularBondNumber::from_force_weight(50.0, 2.0, 10.0).unwrap();
        assert!((bo.bond_number() - 2.5).abs() <= EPS);
        assert_eq!(bo.regime(), BondRegime::Cohesive);
        assert!(bo.is_cohesive());
    }

    #[test]
    fn sphere_mass_is_consistent() {
        // d=2 -> volume = (pi/6)*8, m = rho*volume with rho=3
        let d = 2.0_f32;
        let rho = 3.0_f32;
        let expected_mass = rho * (std::f32::consts::PI / 6.0) * d * d * d;
        let g = 10.0_f32;
        let f = 100.0_f32;
        let bo = GranularBondNumber::from_sphere(f, d, rho, g).unwrap();
        assert!((bo.particle_weight() - expected_mass * g).abs() <= 1e-3);
        assert!((bo.bond_number() - f / (expected_mass * g)).abs() <= 1e-4);
    }

    #[test]
    fn zero_cohesion_is_free_flowing() {
        let bo = GranularBondNumber::from_force_weight(0.0, 2.0, 10.0).unwrap();
        assert!(bo.bond_number().abs() <= EPS);
        assert!(bo.is_free_flowing());
    }

    #[test]
    fn balanced_within_tolerance() {
        // F=20, m=2, g=10 -> Bo=1.0 exactly
        let bo = GranularBondNumber::from_force_weight(20.0, 2.0, 10.0).unwrap();
        assert!((bo.bond_number() - 1.0).abs() <= EPS);
        assert!(bo.is_balanced(1e-3));
        assert_eq!(bo.regime(), BondRegime::Cohesive);
    }

    #[test]
    fn rejects_invalid_parameters() {
        assert!(GranularBondNumber::from_force_weight(-1.0, 1.0, 1.0).is_none());
        assert!(GranularBondNumber::from_force_weight(1.0, 0.0, 1.0).is_none());
        assert!(GranularBondNumber::from_force_weight(1.0, 1.0, 0.0).is_none());
        assert!(GranularBondNumber::from_force_weight(f32::NAN, 1.0, 1.0).is_none());
        assert!(GranularBondNumber::from_force_weight(1.0, f32::INFINITY, 1.0).is_none());
        assert!(GranularBondNumber::from_sphere(1.0, 0.0, 1.0, 1.0).is_none());
        assert!(GranularBondNumber::from_sphere(1.0, 1.0, -1.0, 1.0).is_none());
    }
}
