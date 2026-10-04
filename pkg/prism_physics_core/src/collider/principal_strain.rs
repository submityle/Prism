//! 变形梯度 `F` 的谱分解应变度量：主伸长与各类主应变。
//!
//! 本模块对给定的变形梯度做奇异值分解 `F = U Σ Vᵀ`，奇异值 `σ_i` 即主伸长
//! `λ_i`（`F` 的右拉伸张量 `U = √(FᵀF)` 的特征值）。由主伸长可一次性导出连续
//! 介质力学中几种常用的主应变度量：
//!
//! - 主伸长 `λ₁ ≥ λ₂ ≥ λ₃ > 0`。
//! - 对数（Hencky / 真实）主应变 `εᵢ = ln λᵢ`。
//! - Green-Lagrange 主应变 `Eᵢ = (λᵢ² − 1) / 2`。
//! - 名义（工程）主应变 `eᵢ = λᵢ − 1`。
//! - 体积比 `J = λ₁ λ₂ λ₃ = det F`，体积对数应变 `ln J = Σ εᵢ`。
//! - von Mises 等效偏量对数应变
//!   `ε_eq = √(2/3 · Σ (εᵢ − ε̄)²)`，以及最大剪切对数应变 `(ε₁ − ε₃)/2`。
//!
//! 它与 [`super::finite_strain`] 互补：后者给出 `F` 的全张量度量（`C`/`b`/`E`/
//! `ε`/不变量），本模块给出其谱（主轴）表示。奇异值对转置不变，因此行主序 /
//! 列主序约定不影响结果。
//!
//! 纯函数式、零耦合的分析原语：复用 crate 自有的 [`crate::mpm::svd3`]（clean-room
//! 实现），不依赖任何求解器状态或时间推进。消费上游 `nonaffine_displacement` /
//! `tet_fem_basis` 产生的 `F`。

use crate::mpm::svd3;
use glam::Mat3;

/// 由变形梯度 `F` 谱分解得到的一组主应变度量。
///
/// 使用 [`PrincipalStrain::from_deformation_gradient`] 构造。所有数组按主伸长
/// 降序 `λ₁ ≥ λ₂ ≥ λ₃` 排列。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrincipalStrain {
    stretches: [f32; 3],
    log_strains: [f32; 3],
    green_lagrange: [f32; 3],
    nominal: [f32; 3],
    jacobian: f32,
}

impl PrincipalStrain {
    /// 从变形梯度 `F`（行主序 `[[f32; 3]; 3]`）构造全部主应变度量。
    ///
    /// 当任意分量非有限，或体积比 `J = det F ≤ 0`（材料翻转/退化）时返回 `None`。
    #[must_use]
    pub fn from_deformation_gradient(f: [[f32; 3]; 3]) -> Option<Self> {
        if !f.iter().all(|row| row.iter().all(|v| v.is_finite())) {
            return None;
        }

        // 奇异值对转置不变，列主序构造不影响主伸长结果。
        let mat = Mat3::from_cols_array_2d(&f);
        let svd = svd3(mat);
        let sigma = svd.sigma;

        // 带符号奇异值之积即 det F；svd3 仅在反射（det<0）时令某奇异值为负。
        let jacobian = sigma.x * sigma.y * sigma.z;
        if !jacobian.is_finite() || jacobian <= 0.0 {
            return None;
        }

        // 主伸长取幅值并降序排列。
        let mut stretches = [sigma.x.abs(), sigma.y.abs(), sigma.z.abs()];
        stretches.sort_by(|a, b| b.total_cmp(a));
        if stretches.iter().any(|&s| !s.is_finite() || s <= 0.0) {
            return None;
        }

        let mut log_strains = [0.0_f32; 3];
        let mut green_lagrange = [0.0_f32; 3];
        let mut nominal = [0.0_f32; 3];
        for i in 0..3 {
            let lambda = stretches[i];
            // 对数在 f64 下求值（规避 f32 ln），再降回 f32。
            log_strains[i] = f64::from(lambda).ln() as f32;
            green_lagrange[i] = (lambda * lambda - 1.0) / 2.0;
            nominal[i] = lambda - 1.0;
        }

        Some(Self {
            stretches,
            log_strains,
            green_lagrange,
            nominal,
            jacobian,
        })
    }

    /// 主伸长 `[λ₁, λ₂, λ₃]`（降序，均为正）。
    #[must_use]
    pub fn principal_stretches(&self) -> [f32; 3] {
        self.stretches
    }

    /// 对数（Hencky）主应变 `[ln λ₁, ln λ₂, ln λ₃]`。
    #[must_use]
    pub fn principal_log_strains(&self) -> [f32; 3] {
        self.log_strains
    }

    /// Green-Lagrange 主应变 `[(λᵢ² − 1)/2]`。
    #[must_use]
    pub fn principal_green_lagrange_strains(&self) -> [f32; 3] {
        self.green_lagrange
    }

    /// 名义（工程）主应变 `[λᵢ − 1]`。
    #[must_use]
    pub fn principal_nominal_strains(&self) -> [f32; 3] {
        self.nominal
    }

    /// 体积比（雅可比）`J = λ₁ λ₂ λ₃ = det F`。
    #[must_use]
    pub fn jacobian(&self) -> f32 {
        self.jacobian
    }

    /// 体积对数应变 `ln J = Σ ln λᵢ`。
    #[must_use]
    pub fn volumetric_log_strain(&self) -> f32 {
        self.log_strains[0] + self.log_strains[1] + self.log_strains[2]
    }

    /// 平均对数应变 `ε̄ = (Σ ln λᵢ) / 3`。
    #[must_use]
    pub fn mean_log_strain(&self) -> f32 {
        self.volumetric_log_strain() / 3.0
    }

    /// von Mises 等效偏量对数应变 `ε_eq = √(2/3 · Σ (εᵢ − ε̄)²)`。
    #[must_use]
    pub fn equivalent_deviatoric_log_strain(&self) -> f32 {
        let mean = self.mean_log_strain();
        let mut sum_sq = 0.0_f32;
        for &e in &self.log_strains {
            let d = e - mean;
            sum_sq += d * d;
        }
        (2.0 / 3.0 * sum_sq).sqrt()
    }

    /// 最大剪切对数应变 `(ε₁ − ε₃) / 2`（最大与最小主对数应变之差的一半）。
    #[must_use]
    pub fn max_shear_log_strain(&self) -> f32 {
        (self.log_strains[0] - self.log_strains[2]) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;
    const LN2: f32 = std::f32::consts::LN_2;

    fn arr_close(a: [f32; 3], b: [f32; 3]) -> bool {
        a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() <= EPS)
    }

    #[test]
    fn identity_is_undeformed() {
        let f = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let ps = PrincipalStrain::from_deformation_gradient(f).unwrap();
        assert!(arr_close(ps.principal_stretches(), [1.0, 1.0, 1.0]));
        assert!(arr_close(ps.principal_log_strains(), [0.0, 0.0, 0.0]));
        assert!(arr_close(
            ps.principal_green_lagrange_strains(),
            [0.0, 0.0, 0.0]
        ));
        assert!(arr_close(ps.principal_nominal_strains(), [0.0, 0.0, 0.0]));
        assert!((ps.jacobian() - 1.0).abs() <= EPS);
        assert!(ps.volumetric_log_strain().abs() <= EPS);
        assert!(ps.equivalent_deviatoric_log_strain().abs() <= EPS);
        assert!(ps.max_shear_log_strain().abs() <= EPS);
    }

    #[test]
    fn uniaxial_stretch() {
        let f = [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let ps = PrincipalStrain::from_deformation_gradient(f).unwrap();
        assert!(arr_close(ps.principal_stretches(), [2.0, 1.0, 1.0]));
        assert!(arr_close(ps.principal_log_strains(), [LN2, 0.0, 0.0]));
        assert!(arr_close(
            ps.principal_green_lagrange_strains(),
            [1.5, 0.0, 0.0]
        ));
        assert!(arr_close(ps.principal_nominal_strains(), [1.0, 0.0, 0.0]));
        assert!((ps.jacobian() - 2.0).abs() <= EPS);
        assert!((ps.volumetric_log_strain() - LN2).abs() <= EPS);
        assert!((ps.max_shear_log_strain() - LN2 / 2.0).abs() <= EPS);
    }

    #[test]
    fn isochoric_has_zero_volumetric_log_strain() {
        // F = diag(2, 1, 0.5)，J = 1。
        let f = [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.5]];
        let ps = PrincipalStrain::from_deformation_gradient(f).unwrap();
        assert!((ps.jacobian() - 1.0).abs() <= EPS);
        assert!(ps.volumetric_log_strain().abs() <= EPS);
        assert!(arr_close(ps.principal_stretches(), [2.0, 1.0, 0.5]));
        // ε = [ln2, 0, -ln2]，等效偏量 = √(2/3·2·ln2²) = ln2·2/√3。
        let expected = LN2 * 2.0 / 3.0_f32.sqrt();
        assert!((ps.equivalent_deviatoric_log_strain() - expected).abs() <= EPS);
    }

    #[test]
    fn rigid_rotation_has_zero_strain() {
        // 绕 z 轴 90° 旋转，det=1，所有主伸长应为 1。
        let f = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let ps = PrincipalStrain::from_deformation_gradient(f).unwrap();
        assert!(arr_close(ps.principal_stretches(), [1.0, 1.0, 1.0]));
        assert!(ps.equivalent_deviatoric_log_strain().abs() <= EPS);
        assert!((ps.jacobian() - 1.0).abs() <= EPS);
    }

    #[test]
    fn simple_shear_is_isochoric_but_deviatoric() {
        // 简单剪切 γ=0.5，J=1，主伸长 λ_max·λ_min=1，λ_mid=1。
        let g = 0.5_f32;
        let f = [[1.0, g, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let ps = PrincipalStrain::from_deformation_gradient(f).unwrap();
        assert!((ps.jacobian() - 1.0).abs() <= EPS);
        assert!(ps.volumetric_log_strain().abs() <= EPS);
        let s = ps.principal_stretches();
        // 解析解：λ = √(1+γ²/4) ± γ/2，中间值 1。
        let root = (1.0_f32 + g * g / 4.0).sqrt();
        assert!((s[0] - (root + g / 2.0)).abs() <= 1e-4);
        assert!((s[1] - 1.0).abs() <= 1e-4);
        assert!((s[2] - (root - g / 2.0)).abs() <= 1e-4);
        // 偏量应变非零。
        assert!(ps.equivalent_deviatoric_log_strain() > 0.1);
    }

    #[test]
    fn stretches_are_descending() {
        let f = [[0.7, 0.0, 0.0], [0.0, 1.3, 0.0], [0.0, 0.0, 1.0]];
        let ps = PrincipalStrain::from_deformation_gradient(f).unwrap();
        let s = ps.principal_stretches();
        assert!(s[0] >= s[1] && s[1] >= s[2]);
        assert!(arr_close(s, [1.3, 1.0, 0.7]));
    }

    #[test]
    fn rejects_invalid_gradients() {
        let nan = [[f32::NAN, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert!(PrincipalStrain::from_deformation_gradient(nan).is_none());
        // 翻转 det=-1。
        let inverted = [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert!(PrincipalStrain::from_deformation_gradient(inverted).is_none());
        // 退化 det=0。
        let degenerate = [[0.0; 3], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert!(PrincipalStrain::from_deformation_gradient(degenerate).is_none());
    }
}
