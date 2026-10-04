//! 运动学膨胀角与 Taylor 应力-膨胀关系（零耦合分析原语）。
//!
//! 对剪切中的密实颗粒/土体，体积应变增量 `dε_v` 与剪切应变增量 `dγ` 的比值
//! 定义了运动学**膨胀角** `ψ`（Taylor 锯齿模型约定）：
//!
//! ```text
//!   tan ψ = − dε_v / dγ
//! ```
//!
//! **符号约定（全模块统一，压缩为正）：**
//! - 体积应变增量 `dε_v > 0` 表示压缩（体积减小 / 收缩）；`dε_v < 0` 表示膨胀。
//! - 剪切应变增量 `dγ > 0` 为其大小（恒取正）。
//! - 因此 `tan ψ > 0` 对应剪胀，`tan ψ < 0` 对应剪缩，`ψ = 0` 为等容。
//!
//! Taylor（1948）能量修正给出峰值（动员）摩擦角对临界状态摩擦角与膨胀角的分解：
//!
//! ```text
//!   tan φ_mob = tan φ_cv + tan ψ
//! ```
//!
//! 即动员摩擦 = 临界状态（恒体积）摩擦 + 剪胀贡献。
//!
//! 纯函数式、零耦合：仅对给定的应变增量做运动学换算，不依赖任何求解器状态或
//! 时间推进。与 [`super::granular_rheology`] 的 `DilatancyLaw`（给出堆积分数
//! `φ(I)` 的本构律）互补——本模块给出的是运动学膨胀角，而非堆积分数本构。

/// 由应变增量得到的运动学膨胀角及其构造标度（压缩为正约定）。
///
/// 使用 [`StressDilatancy::from_strain_increments`] 构造。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StressDilatancy {
    volumetric_increment: f32,
    shear_increment: f32,
    tan_psi: f32,
    dilatancy_angle: f32,
}

impl StressDilatancy {
    /// 由体积应变增量 `volumetric_increment`（压缩为正）与剪切应变增量
    /// `shear_increment` 构造膨胀角，`tan ψ = −dε_v / dγ`。
    ///
    /// 要求两者有限且 `shear_increment > 0`；否则返回 `None`。
    pub fn from_strain_increments(volumetric_increment: f32, shear_increment: f32) -> Option<Self> {
        if !volumetric_increment.is_finite() || !shear_increment.is_finite() {
            return None;
        }
        if shear_increment <= 0.0 {
            return None;
        }
        let tan_psi = -volumetric_increment / shear_increment;
        let dilatancy_angle = f64::from(tan_psi).atan() as f32;
        Some(Self {
            volumetric_increment,
            shear_increment,
            tan_psi,
            dilatancy_angle,
        })
    }

    /// 膨胀角 `ψ`（弧度，剪胀为正，剪缩为负）。
    pub fn dilatancy_angle(&self) -> f32 {
        self.dilatancy_angle
    }

    /// 膨胀角 `ψ`（度）。
    pub fn dilatancy_angle_degrees(&self) -> f32 {
        self.dilatancy_angle.to_degrees()
    }

    /// 膨胀比 `tan ψ = −dε_v / dγ`。
    pub fn tan_dilatancy(&self) -> f32 {
        self.tan_psi
    }

    /// 构造所用的体积应变增量 `dε_v`（压缩为正）。
    pub fn volumetric_increment(&self) -> f32 {
        self.volumetric_increment
    }

    /// 构造所用的剪切应变增量 `dγ`（恒正）。
    pub fn shear_increment(&self) -> f32 {
        self.shear_increment
    }

    /// 是否剪胀（体积膨胀，`dε_v < 0`，即 `ψ > 0`）。
    pub fn is_dilating(&self) -> bool {
        self.tan_psi > 0.0
    }

    /// 是否剪缩（体积压缩，`dε_v > 0`，即 `ψ < 0`）。
    pub fn is_contracting(&self) -> bool {
        self.tan_psi < 0.0
    }

    /// 是否近似等容（`|ψ| ≤ tol`，`tol` 为弧度）。
    pub fn is_isochoric(&self, tol: f32) -> bool {
        self.dilatancy_angle.abs() <= tol
    }

    /// Taylor 应力-膨胀关系：由临界状态摩擦角 `critical_state_friction`
    /// （弧度）给出动员摩擦角 `φ_mob`，满足 `tan φ_mob = tan φ_cv + tan ψ`。
    ///
    /// 要求 `critical_state_friction` 有限且严格落在 `(−π/2, π/2)` 内（使 `tan`
    /// 有定义）；否则返回 `None`。
    pub fn taylor_mobilized_friction(&self, critical_state_friction: f32) -> Option<f32> {
        if !critical_state_friction.is_finite() {
            return None;
        }
        let half_pi = std::f32::consts::FRAC_PI_2;
        if !(-half_pi..half_pi).contains(&critical_state_friction) {
            return None;
        }
        let tan_cv = f64::from(critical_state_friction).tan();
        let tan_mob = tan_cv + f64::from(self.tan_psi);
        Some(tan_mob.atan() as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    #[test]
    fn pure_shear_is_isochoric() {
        let sd = StressDilatancy::from_strain_increments(0.0, 0.1).unwrap();
        assert!(sd.tan_dilatancy().abs() <= EPS);
        assert!(sd.dilatancy_angle().abs() <= EPS);
        assert!(sd.is_isochoric(1e-4));
        assert!(!sd.is_dilating());
        assert!(!sd.is_contracting());
    }

    #[test]
    fn dilation_gives_positive_angle() {
        // dε_v = -0.05 (expansion), dγ = 0.1 -> tan ψ = 0.5
        let sd = StressDilatancy::from_strain_increments(-0.05, 0.1).unwrap();
        assert!((sd.tan_dilatancy() - 0.5).abs() <= EPS);
        // atan(0.5) ≈ 0.4636476 rad ≈ 26.565°
        assert!((sd.dilatancy_angle() - 0.463_647_6).abs() <= 1e-4);
        assert!((sd.dilatancy_angle_degrees() - 26.565_05).abs() <= 1e-2);
        assert!(sd.is_dilating());
    }

    #[test]
    fn contraction_gives_negative_angle() {
        let sd = StressDilatancy::from_strain_increments(0.05, 0.1).unwrap();
        assert!((sd.tan_dilatancy() + 0.5).abs() <= EPS);
        assert!(sd.dilatancy_angle() < 0.0);
        assert!(sd.is_contracting());
    }

    #[test]
    fn taylor_adds_dilatancy_to_critical_state() {
        // φ_cv = 30° = 0.5235988 rad, tanφ_cv = 0.5773503
        // dilating tan ψ = 0.5 -> tan φ_mob = 1.0773503 -> φ_mob = atan ≈ 0.8226160 rad
        let sd = StressDilatancy::from_strain_increments(-0.05, 0.1).unwrap();
        let phi_cv = 30.0_f32.to_radians();
        let phi_mob = sd.taylor_mobilized_friction(phi_cv).unwrap();
        assert!((phi_mob - 0.822_616_0).abs() <= 1e-4);
        // Mobilized friction exceeds critical-state friction when dilating.
        assert!(phi_mob > phi_cv);
    }

    #[test]
    fn taylor_recovers_critical_state_without_dilatancy() {
        let sd = StressDilatancy::from_strain_increments(0.0, 0.1).unwrap();
        let phi_cv = 25.0_f32.to_radians();
        let phi_mob = sd.taylor_mobilized_friction(phi_cv).unwrap();
        assert!((phi_mob - phi_cv).abs() <= 1e-5);
    }

    #[test]
    fn taylor_rejects_out_of_range_friction() {
        let sd = StressDilatancy::from_strain_increments(-0.05, 0.1).unwrap();
        assert!(sd
            .taylor_mobilized_friction(std::f32::consts::FRAC_PI_2)
            .is_none());
        assert!(sd.taylor_mobilized_friction(f32::NAN).is_none());
    }

    #[test]
    fn rejects_invalid_increments() {
        assert!(StressDilatancy::from_strain_increments(0.1, 0.0).is_none());
        assert!(StressDilatancy::from_strain_increments(0.1, -0.1).is_none());
        assert!(StressDilatancy::from_strain_increments(f32::NAN, 0.1).is_none());
        assert!(StressDilatancy::from_strain_increments(0.1, f32::INFINITY).is_none());
    }
}
