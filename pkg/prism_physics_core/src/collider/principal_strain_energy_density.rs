//! 主应力–主应变弹性应变能密度及其体积/偏量分解。
//!
//! 给定一对**功共轭**的主应力与主应变（同一组主方向上的三元组），
//! 单位体积弹性应变能密度为
//! ```text
//! U = ½ Σ σ_i · ε_i.
//! ```
//! 将应力/应变分解为静水部分与偏量部分后，能量亦自然分解：
//! ```text
//! σ_i = s_i + p,       p = (σ1+σ2+σ3)/3,     s_i = σ_i − p,
//! ε_i = e_i + ε_v/3,   ε_v = ε1+ε2+ε3,       e_i = ε_i − ε_v/3,
//! U = ½ p·ε_v + ½ Σ s_i·e_i  =  U_vol + U_dev.
//! ```
//! 由于 `Σ s_i = 0`、`Σ e_i = 0`，交叉项消去，分解**严格精确**。
//!
//! 体积/偏量能量分解是相场断裂、损伤演化等诊断的常用驱动量，
//! 但本模块只做**只读标量诊断**：不构造本构、不参与碰撞/求解管线，
//! 完全 0 耦合。沿用本仓库**拉伸为正**约定。
//!
//! 注意：这与 `*_strain_energy_density`（由变形梯度 F 评估的超弹性
//! 本构势）不同——此处由已测得的主应力/主应变直接做功共轭积分。

/// 低于该量级的总能量视为近零（用于比例计算的退化保护）。
const ENERGY_EPS: f32 = 1e-12;

/// 主应力–主应变弹性应变能密度（含体积/偏量分解）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrincipalStrainEnergyDensity {
    total: f32,
    volumetric: f32,
    deviatoric: f32,
    mean_stress: f32,
    volumetric_strain: f32,
}

impl PrincipalStrainEnergyDensity {
    /// 由功共轭的主应力（**拉伸为正**）与主应变构造。
    ///
    /// 两个三元组必须定义在同一组主方向上（即一一对应）。
    pub fn from_principal(principal_stress: [f32; 3], principal_strain: [f32; 3]) -> Self {
        let total = 0.5
            * (principal_stress[0] * principal_strain[0]
                + principal_stress[1] * principal_strain[1]
                + principal_stress[2] * principal_strain[2]);

        let mean_stress = (principal_stress[0] + principal_stress[1] + principal_stress[2]) / 3.0;
        let volumetric_strain = principal_strain[0] + principal_strain[1] + principal_strain[2];

        // U_vol = ½ p·ε_v。
        let volumetric = 0.5 * mean_stress * volumetric_strain;
        // 偏量能量由精确分解补足，避免重复累加偏量乘积的数值误差。
        let deviatoric = total - volumetric;

        Self {
            total,
            volumetric,
            deviatoric,
            mean_stress,
            volumetric_strain,
        }
    }

    /// 总弹性应变能密度 `U = ½ Σ σ_i ε_i`。
    pub fn total_energy_density(&self) -> f32 {
        self.total
    }

    /// 体积（静水）部分 `U_vol = ½ p·ε_v`。
    pub fn volumetric_energy_density(&self) -> f32 {
        self.volumetric
    }

    /// 偏量（畸变）部分 `U_dev = U − U_vol = ½ Σ s_i e_i`。
    pub fn deviatoric_energy_density(&self) -> f32 {
        self.deviatoric
    }

    /// 平均应力 `p = (σ1+σ2+σ3)/3`（拉伸为正）。
    pub fn mean_stress(&self) -> f32 {
        self.mean_stress
    }

    /// 体应变 `ε_v = ε1+ε2+ε3`。
    pub fn volumetric_strain(&self) -> f32 {
        self.volumetric_strain
    }

    /// 偏量能量占总能量的比例 `U_dev / U`。
    ///
    /// 总能量近零时返回 `None`。
    pub fn deviatoric_fraction(&self) -> Option<f32> {
        if self.total.abs() <= ENERGY_EPS {
            return None;
        }
        Some(self.deviatoric / self.total)
    }

    /// 功是否非负（稳定材料做正功的必要诊断）。
    pub fn is_nonnegative_work(&self) -> bool {
        self.total >= 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-5;

    #[test]
    fn uniaxial_decomposition_matches_hand_values() {
        // σ=(10,0,0), ε=(0.1,0,0)。
        // U=½·10·0.1=0.5; p=10/3, ε_v=0.1; U_vol=½·(10/3)·0.1=0.166667;
        // U_dev=0.5−0.166667=0.333333。
        let u = PrincipalStrainEnergyDensity::from_principal([10.0, 0.0, 0.0], [0.1, 0.0, 0.0]);
        assert!((u.total_energy_density() - 0.5).abs() < TOL);
        assert!((u.volumetric_energy_density() - 0.5 / 3.0).abs() < TOL);
        assert!((u.deviatoric_energy_density() - (0.5 - 0.5 / 3.0)).abs() < TOL);
        assert!((u.mean_stress() - 10.0 / 3.0).abs() < TOL);
        assert!((u.volumetric_strain() - 0.1).abs() < TOL);
    }

    #[test]
    fn pure_hydrostatic_has_zero_deviatoric_energy() {
        // σ=(5,5,5), ε=(0.01,0.01,0.01) => 偏量为零。
        let u = PrincipalStrainEnergyDensity::from_principal([5.0, 5.0, 5.0], [0.01, 0.01, 0.01]);
        assert!(u.deviatoric_energy_density().abs() < TOL);
        assert!((u.total_energy_density() - u.volumetric_energy_density()).abs() < TOL);
        // U=½·(5·0.01)·3=0.075。
        assert!((u.total_energy_density() - 0.075).abs() < TOL);
    }

    #[test]
    fn pure_deviatoric_has_zero_volumetric_energy() {
        // σ=(3,−3,0) 迹为零 => p=0; ε=(0.02,−0.02,0) 体应变为零。
        let u = PrincipalStrainEnergyDensity::from_principal([3.0, -3.0, 0.0], [0.02, -0.02, 0.0]);
        assert!(u.volumetric_energy_density().abs() < TOL);
        // U=½(3·0.02+(−3)(−0.02))=0.06。
        assert!((u.total_energy_density() - 0.06).abs() < TOL);
        assert!((u.deviatoric_energy_density() - 0.06).abs() < TOL);
    }

    #[test]
    fn total_equals_volumetric_plus_deviatoric() {
        let u =
            PrincipalStrainEnergyDensity::from_principal([7.0, -2.0, 1.5], [0.03, 0.01, -0.004]);
        let sum = u.volumetric_energy_density() + u.deviatoric_energy_density();
        assert!((u.total_energy_density() - sum).abs() < TOL);
    }

    #[test]
    fn deviatoric_part_matches_explicit_sum_of_deviators() {
        // 直接按 ½ Σ s_i e_i 显式核验偏量能量。
        let s = [7.0_f32, -2.0, 1.5];
        let e = [0.03_f32, 0.01, -0.004];
        let p = (s[0] + s[1] + s[2]) / 3.0;
        let ev = e[0] + e[1] + e[2];
        let dev_s = [s[0] - p, s[1] - p, s[2] - p];
        let dev_e = [e[0] - ev / 3.0, e[1] - ev / 3.0, e[2] - ev / 3.0];
        let explicit = 0.5 * (dev_s[0] * dev_e[0] + dev_s[1] * dev_e[1] + dev_s[2] * dev_e[2]);
        let u = PrincipalStrainEnergyDensity::from_principal(s, e);
        assert!((u.deviatoric_energy_density() - explicit).abs() < TOL);
    }

    #[test]
    fn deviatoric_fraction_is_none_for_zero_energy() {
        let u = PrincipalStrainEnergyDensity::from_principal([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        assert!(u.deviatoric_fraction().is_none());
    }

    #[test]
    fn deviatoric_fraction_matches_ratio() {
        let u = PrincipalStrainEnergyDensity::from_principal([10.0, 0.0, 0.0], [0.1, 0.0, 0.0]);
        let frac = u.deviatoric_fraction().unwrap();
        assert!((frac - (u.deviatoric_energy_density() / u.total_energy_density())).abs() < TOL);
    }

    #[test]
    fn nonnegative_work_detects_sign() {
        let positive =
            PrincipalStrainEnergyDensity::from_principal([10.0, 0.0, 0.0], [0.1, 0.0, 0.0]);
        assert!(positive.is_nonnegative_work());
        let negative =
            PrincipalStrainEnergyDensity::from_principal([-10.0, 0.0, 0.0], [0.1, 0.0, 0.0]);
        assert!(!negative.is_nonnegative_work());
    }
}
