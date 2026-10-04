//! 应力张量不变量与 Lode 角分解（p–q–θ 应力状态刻画）。
//!
//! 本模块把一组主应力分解为土力学/塑性理论中标准的应力状态描述量：
//! 平均（静水）应力 `p`、偏应力第二不变量 `J2`、von Mises 等效应力 `q`、
//! 八面体剪应力 `τ_oct`，以及刻画应力状态「形状」的 Lode 参数 `μ_L` 与
//! Lode 角 `θ`。这些量共同构成颗粒材料在 `p–q–θ` 空间中的完整应力状态，
//! 是 Mohr–Coulomb、Drucker–Prager、临界状态等本构模型的通用输入。
//!
//! 约定（主应力按降序排列 `σ1 ≥ σ2 ≥ σ3`）：
//!
//! ```text
//! p     = (σ1 + σ2 + σ3) / 3
//! s_i   = σ_i − p                              （偏主应力）
//! J2    = (1/2)(s1² + s2² + s3²)
//! J3    = s1 · s2 · s3
//! q     = sqrt(3 · J2)                          （von Mises 等效应力）
//! τ_oct = sqrt(2 · J2 / 3)                       （八面体剪应力）
//! μ_L   = (2σ2 − σ1 − σ3) / (σ1 − σ3)            （Lode-Nadai 参数，∈ [−1, 1]）
//! θ     = atan(μ_L / sqrt(3))                    （Lode 角，∈ [−30°, +30°]）
//! ```
//!
//! Lode 参数的物理含义无歧义：`μ_L = −1` 为三轴压缩（`σ2 = σ3`），
//! `μ_L = +1` 为三轴拉伸（`σ2 = σ1`），`μ_L = 0` 为纯剪。静水应力状态
//! （`σ1 = σ3`）下 Lode 参数/角无定义，返回 `None`。
//!
//! 本模块仅做纯不变量计算，不触碰主帧循环、不依赖渲染引擎，可独立使用。
//! 主应力可由 [`crate::collider::contact_stress`] 的 `principal_stresses()`
//! 提供，二者互补：`contact_stress` 组装 Love–Weber 应力张量并求主应力，
//! 本模块在主应力基础上给出完整的 `p–q–θ` 状态描述。

/// 低于该值的 `(σ1 − σ3)` 视为静水状态，Lode 参数/角无定义。
const HYDROSTATIC_EPS: f32 = 1e-9;

/// 由一组主应力导出的应力不变量与 Lode 分解。
///
/// 由 [`StressInvariants::from_principal_stresses`] 构造，内部将主应力按降序
/// 排列并缓存偏主应力与不变量，随后以各 getter 提供派生量。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StressInvariants {
    /// 降序排列的主应力 `[σ1, σ2, σ3]`，满足 `σ1 ≥ σ2 ≥ σ3`。
    principal: [f32; 3],
    /// 偏主应力 `[s1, s2, s3]`，`s_i = σ_i − p`。
    deviatoric: [f32; 3],
    mean_stress: f32,
    j2: f32,
    j3: f32,
}

impl StressInvariants {
    /// 由主应力构造不变量分解。
    ///
    /// 输入的三个主应力顺序任意，内部会按降序重排。当任一分量非有限时返回 `None`。
    #[must_use]
    pub fn from_principal_stresses(principal_stresses: [f32; 3]) -> Option<Self> {
        if !principal_stresses.iter().all(|v| v.is_finite()) {
            return None;
        }
        let mut principal = principal_stresses;
        // 降序排列：σ1 ≥ σ2 ≥ σ3。输入已校验有限，total_cmp 安全且无歧义。
        principal.sort_by(|a, b| b.total_cmp(a));

        let mean_stress = (principal[0] + principal[1] + principal[2]) / 3.0;
        let deviatoric = [
            principal[0] - mean_stress,
            principal[1] - mean_stress,
            principal[2] - mean_stress,
        ];
        let j2 = 0.5
            * (deviatoric[0] * deviatoric[0]
                + deviatoric[1] * deviatoric[1]
                + deviatoric[2] * deviatoric[2]);
        let j3 = deviatoric[0] * deviatoric[1] * deviatoric[2];

        Some(Self {
            principal,
            deviatoric,
            mean_stress,
            j2,
            j3,
        })
    }

    /// 降序排列的主应力 `[σ1, σ2, σ3]`。
    #[must_use]
    pub fn principal_stresses(&self) -> [f32; 3] {
        self.principal
    }

    /// 偏主应力 `[s1, s2, s3]`，`s_i = σ_i − p`。
    #[must_use]
    pub fn deviatoric_stresses(&self) -> [f32; 3] {
        self.deviatoric
    }

    /// 平均（静水）应力 `p = (σ1 + σ2 + σ3) / 3`。
    #[must_use]
    pub fn mean_stress(&self) -> f32 {
        self.mean_stress
    }

    /// 偏应力第二不变量 `J2 = (1/2)(s1² + s2² + s3²)`。
    #[must_use]
    pub fn second_invariant_j2(&self) -> f32 {
        self.j2
    }

    /// 偏应力第三不变量 `J3 = s1 · s2 · s3`。
    #[must_use]
    pub fn third_invariant_j3(&self) -> f32 {
        self.j3
    }

    /// von Mises 等效应力 `q = sqrt(3 · J2)`。
    #[must_use]
    pub fn von_mises_equivalent(&self) -> f32 {
        (3.0 * self.j2).sqrt()
    }

    /// 八面体剪应力 `τ_oct = sqrt(2 · J2 / 3)`。
    #[must_use]
    pub fn octahedral_shear_stress(&self) -> f32 {
        (2.0 * self.j2 / 3.0).sqrt()
    }

    /// 偏应力张量的 Frobenius 范数 `sqrt(s1² + s2² + s3²) = sqrt(2 · J2)`。
    #[must_use]
    pub fn deviatoric_norm(&self) -> f32 {
        (2.0 * self.j2).sqrt()
    }

    /// Lode-Nadai 参数 `μ_L = (2σ2 − σ1 − σ3) / (σ1 − σ3)`，范围 `[−1, 1]`。
    ///
    /// `μ_L = −1` 三轴压缩，`μ_L = +1` 三轴拉伸，`μ_L = 0` 纯剪。
    /// 静水应力状态（`σ1 − σ3 ≈ 0`）下无定义，返回 `None`。
    #[must_use]
    pub fn lode_parameter(&self) -> Option<f32> {
        let span = self.principal[0] - self.principal[2];
        if span <= HYDROSTATIC_EPS {
            return None;
        }
        let mu = (2.0 * self.principal[1] - self.principal[0] - self.principal[2]) / span;
        Some(mu)
    }

    /// Lode 角 `θ = atan(μ_L / sqrt(3))`，弧度，范围 `[−π/6, +π/6]`。
    ///
    /// 静水应力状态下无定义，返回 `None`。
    #[must_use]
    pub fn lode_angle(&self) -> Option<f32> {
        let mu = self.lode_parameter()?;
        let theta = (f64::from(mu) / 3.0_f64.sqrt()).atan();
        Some(theta as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIXTH_PI: f32 = std::f32::consts::FRAC_PI_6;

    #[test]
    fn rejects_non_finite() {
        assert!(StressInvariants::from_principal_stresses([f32::NAN, 1.0, 0.0]).is_none());
        assert!(StressInvariants::from_principal_stresses([1.0, f32::INFINITY, 0.0]).is_none());
    }

    #[test]
    fn sorts_principal_descending() {
        let s = StressInvariants::from_principal_stresses([1.0, 3.0, 1.0]).unwrap();
        assert_eq!(s.principal_stresses(), [3.0, 1.0, 1.0]);
    }

    #[test]
    fn triaxial_compression_hand_values() {
        // σ = [3, 1, 1] → μ_L = −1（三轴压缩）。
        let s = StressInvariants::from_principal_stresses([3.0, 1.0, 1.0]).unwrap();
        assert!((s.mean_stress() - 5.0 / 3.0).abs() < 1e-6);
        // J2 = 4/3
        assert!((s.second_invariant_j2() - 4.0 / 3.0).abs() < 1e-6);
        // q = sqrt(3 · 4/3) = 2
        assert!((s.von_mises_equivalent() - 2.0).abs() < 1e-6);
        // J3 = (4/3)(−2/3)(−2/3) = 16/27
        assert!((s.third_invariant_j3() - 16.0 / 27.0).abs() < 1e-6);
        // τ_oct = sqrt(8/9)
        assert!((s.octahedral_shear_stress() - (8.0_f32 / 9.0).sqrt()).abs() < 1e-6);
        // deviatoric_norm = sqrt(8/3)
        assert!((s.deviatoric_norm() - (8.0_f32 / 3.0).sqrt()).abs() < 1e-6);
        assert!((s.lode_parameter().unwrap() - (-1.0)).abs() < 1e-6);
        // Lode 角 = −30°
        assert!((s.lode_angle().unwrap() - (-SIXTH_PI)).abs() < 1e-6);
    }

    #[test]
    fn triaxial_extension_hand_values() {
        // σ = [3, 3, 1] → μ_L = +1（三轴拉伸）。
        let s = StressInvariants::from_principal_stresses([3.0, 3.0, 1.0]).unwrap();
        assert!((s.mean_stress() - 7.0 / 3.0).abs() < 1e-6);
        assert!((s.second_invariant_j2() - 4.0 / 3.0).abs() < 1e-6);
        assert!((s.von_mises_equivalent() - 2.0).abs() < 1e-6);
        assert!((s.lode_parameter().unwrap() - 1.0).abs() < 1e-6);
        // Lode 角 = +30°
        assert!((s.lode_angle().unwrap() - SIXTH_PI).abs() < 1e-6);
    }

    #[test]
    fn pure_shear_hand_values() {
        // σ = [1, 0, −1] → μ_L = 0（纯剪）。
        let s = StressInvariants::from_principal_stresses([1.0, 0.0, -1.0]).unwrap();
        assert!(s.mean_stress().abs() < 1e-6);
        assert!((s.second_invariant_j2() - 1.0).abs() < 1e-6);
        assert!((s.von_mises_equivalent() - 3.0_f32.sqrt()).abs() < 1e-6);
        assert!(s.third_invariant_j3().abs() < 1e-6);
        assert!(s.lode_parameter().unwrap().abs() < 1e-6);
        assert!(s.lode_angle().unwrap().abs() < 1e-6);
    }

    #[test]
    fn hydrostatic_has_no_lode() {
        let s = StressInvariants::from_principal_stresses([2.0, 2.0, 2.0]).unwrap();
        assert_eq!(s.mean_stress(), 2.0);
        assert_eq!(s.second_invariant_j2(), 0.0);
        assert_eq!(s.von_mises_equivalent(), 0.0);
        assert_eq!(s.deviatoric_norm(), 0.0);
        assert!(s.lode_parameter().is_none());
        assert!(s.lode_angle().is_none());
    }

    #[test]
    fn mean_stress_is_translation_covariant() {
        // 叠加静水应力只改变 p，不改变偏量/Lode。
        let base = StressInvariants::from_principal_stresses([3.0, 1.0, -2.0]).unwrap();
        let shifted = StressInvariants::from_principal_stresses([13.0, 11.0, 8.0]).unwrap();
        assert!((shifted.mean_stress() - base.mean_stress() - 10.0).abs() < 1e-5);
        assert!((shifted.second_invariant_j2() - base.second_invariant_j2()).abs() < 1e-4);
        assert!((shifted.lode_parameter().unwrap() - base.lode_parameter().unwrap()).abs() < 1e-6);
    }

    #[test]
    fn von_mises_matches_octahedral_relation() {
        // q = (3 / sqrt(2)) · τ_oct 对任意状态成立。
        let s = StressInvariants::from_principal_stresses([5.0, 2.0, -1.0]).unwrap();
        let expected = 3.0 / 2.0_f32.sqrt() * s.octahedral_shear_stress();
        assert!((s.von_mises_equivalent() - expected).abs() < 1e-5);
    }
}
