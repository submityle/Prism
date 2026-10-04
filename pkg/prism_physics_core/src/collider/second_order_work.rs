//! 二阶功（Hill 稳定性）诊断。
//!
//! Hill 稳定性判据以应力增量与应变增量的双点积定义**二阶功**：
//! ```text
//! d²W = dσ : dε = Σ_ij dσ_ij · dε_ij.
//! ```
//! 物质点在给定加载方向上稳定当且仅当 `d²W > 0`；当 `d²W ≤ 0` 时存在
//! 弥散型失稳（diffuse instability）的可能，典型如砂土液化、应变局部化前兆。
//! 归一化后 `d²W / (‖dσ‖·‖dε‖)` 等于两个增量方向在张量空间中的夹角余弦。
//!
//! 本模块是**纯只读诊断**：不构造本构、不参与碰撞/求解管线，完全 0 耦合。
//! 沿用本仓库**拉伸为正**约定；增量张量按对称张量处理（双点积对
//! 对称部敏感，这里不额外对称化，由调用方保证对称增量）。

/// 低于该范数视为零增量（归一化退化保护）。
const NORM_EPS: f32 = 1e-12;

/// 二阶功诊断结果。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SecondOrderWork {
    value: f32,
    stress_norm: f32,
    strain_norm: f32,
}

impl SecondOrderWork {
    /// 由对称应力/应变增量张量（3×3）构造，`d²W = Σ_ij dσ_ij·dε_ij`。
    pub fn from_tensors(stress_increment: [[f32; 3]; 3], strain_increment: [[f32; 3]; 3]) -> Self {
        let mut value = 0.0_f32;
        let mut s2 = 0.0_f32;
        let mut e2 = 0.0_f32;
        for i in 0..3 {
            for j in 0..3 {
                let ds = stress_increment[i][j];
                let de = strain_increment[i][j];
                value += ds * de;
                s2 += ds * ds;
                e2 += de * de;
            }
        }
        Self {
            value,
            stress_norm: s2.sqrt(),
            strain_norm: e2.sqrt(),
        }
    }

    /// 由功共轭的主应力/主应变增量构造，`d²W = Σ dσ_i·dε_i`。
    ///
    /// 适用于两增量共主轴的情形（主轴一一对应）。
    pub fn from_principal(stress_increment: [f32; 3], strain_increment: [f32; 3]) -> Self {
        let value = stress_increment[0] * strain_increment[0]
            + stress_increment[1] * strain_increment[1]
            + stress_increment[2] * strain_increment[2];
        let s2 = stress_increment[0] * stress_increment[0]
            + stress_increment[1] * stress_increment[1]
            + stress_increment[2] * stress_increment[2];
        let e2 = strain_increment[0] * strain_increment[0]
            + strain_increment[1] * strain_increment[1]
            + strain_increment[2] * strain_increment[2];
        Self {
            value,
            stress_norm: s2.sqrt(),
            strain_norm: e2.sqrt(),
        }
    }

    /// 二阶功标量 `d²W`。
    pub fn value(&self) -> f32 {
        self.value
    }

    /// 应力增量 Frobenius 范数 `‖dσ‖`。
    pub fn stress_increment_norm(&self) -> f32 {
        self.stress_norm
    }

    /// 应变增量 Frobenius 范数 `‖dε‖`。
    pub fn strain_increment_norm(&self) -> f32 {
        self.strain_norm
    }

    /// 是否稳定（`d²W > 0`，Hill 判据）。
    pub fn is_stable(&self) -> bool {
        self.value > 0.0
    }

    /// 是否失稳或临界（`d²W ≤ 0`，弥散失稳可能）。
    pub fn is_unstable(&self) -> bool {
        self.value <= 0.0
    }

    /// 归一化二阶功 `d²W / (‖dσ‖·‖dε‖)`，即增量方向夹角余弦，范围 `[−1, 1]`。
    ///
    /// 任一增量范数近零时返回 `None`。
    pub fn normalized(&self) -> Option<f32> {
        if self.stress_norm <= NORM_EPS || self.strain_norm <= NORM_EPS {
            return None;
        }
        Some((self.value / (self.stress_norm * self.strain_norm)).clamp(-1.0, 1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-5;

    #[test]
    fn aligned_increments_are_stable() {
        // dσ 与 dε 同向 => d²W>0。
        let w = SecondOrderWork::from_principal([2.0, 1.0, 0.5], [0.02, 0.01, 0.005]);
        // 2*0.02+1*0.01+0.5*0.005=0.04+0.01+0.0025=0.0525。
        assert!((w.value() - 0.0525).abs() < TOL);
        assert!(w.is_stable());
        assert!(!w.is_unstable());
    }

    #[test]
    fn opposed_increments_are_unstable() {
        let w = SecondOrderWork::from_principal([2.0, 1.0, 0.5], [-0.02, -0.01, -0.005]);
        assert!((w.value() + 0.0525).abs() < TOL);
        assert!(w.is_unstable());
        assert!(!w.is_stable());
    }

    #[test]
    fn zero_second_order_work_is_critical() {
        // 正交增量 => d²W=0 => 判为失稳/临界。
        let w = SecondOrderWork::from_principal([1.0, -1.0, 0.0], [0.01, 0.01, 0.0]);
        // 1*0.01+(-1)*0.01+0=0。
        assert!(w.value().abs() < TOL);
        assert!(w.is_unstable());
    }

    #[test]
    fn normalized_cosine_of_aligned_is_one() {
        let w = SecondOrderWork::from_principal([1.0, 0.0, 0.0], [0.5, 0.0, 0.0]);
        assert!((w.normalized().unwrap() - 1.0).abs() < TOL);
    }

    #[test]
    fn normalized_is_none_for_zero_increment() {
        let w = SecondOrderWork::from_principal([0.0, 0.0, 0.0], [0.01, 0.0, 0.0]);
        assert!(w.normalized().is_none());
    }

    #[test]
    fn tensor_and_principal_agree_on_diagonal() {
        // 对角增量时张量双点积应等于主值点积。
        let ds = [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.5]];
        let de = [[0.02, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.005]];
        let wt = SecondOrderWork::from_tensors(ds, de);
        let wp = SecondOrderWork::from_principal([2.0, 1.0, 0.5], [0.02, 0.01, 0.005]);
        assert!((wt.value() - wp.value()).abs() < TOL);
        assert!((wt.stress_increment_norm() - wp.stress_increment_norm()).abs() < TOL);
    }

    #[test]
    fn tensor_includes_shear_cross_terms() {
        // 纯剪切增量：对称非对角项贡献 2*dσ_xy*dε_xy。
        let ds = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        let de = [[0.0, 0.01, 0.0], [0.01, 0.0, 0.0], [0.0, 0.0, 0.0]];
        let w = SecondOrderWork::from_tensors(ds, de);
        // 两个 off-diagonal 项各 1*0.01 => 0.02。
        assert!((w.value() - 0.02).abs() < TOL);
        assert!(w.is_stable());
    }

    #[test]
    fn frobenius_norm_matches_hand_value() {
        let ds = [[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 0.0]];
        let de = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        let w = SecondOrderWork::from_tensors(ds, de);
        // ‖dσ‖=√(9+16)=5。
        assert!((w.stress_increment_norm() - 5.0).abs() < TOL);
        assert!(w.normalized().is_none());
    }
}
