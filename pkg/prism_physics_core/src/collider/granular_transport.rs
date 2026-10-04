//! 稠密颗粒气体的碰撞输运与能量耗散闭包（Lun 等人的 KTGF 理论）。
//!
//! 该模块提供两条无耦合、可独立单元验证的闭合关系式，用于稠密颗粒流的
//! 连续介质描述：
//!
//! * **体积黏性（bulk viscosity）** `xi`，刻画颗粒介质在压缩/膨胀时由于
//!   非弹性碰撞导致的额外黏性应力；
//! * **碰撞能量耗散率（collisional dissipation）** `gamma`，刻画颗粒温度
//!   （granular temperature）`Theta` 因非弹性碰撞而被耗散的速率。
//!
//! 这两条关系式来自 Lun、Savage、Jeffrey、Chepurniy（1984）的动理学理论
//! 闭合，并被广泛采用于 Gidaspow 等稠密气固两相流框架：
//!
//! ```text
//! xi    = (4/3) * rho_s * phi^2 * d * g0 * (1 + e) * sqrt(Theta / pi)
//! gamma = (12 * (1 - e^2) / (d * sqrt(pi))) * rho_s * phi^2 * g0 * Theta^(3/2)
//! ```
//!
//! 其中 `rho_s` 为颗粒材料密度，`phi` 为体积分数，`d` 为颗粒直径，
//! `g0` 为径向分布函数（接触处）值，`e` 为法向恢复系数，`Theta` 为颗粒温度。
//!
//! 本模块仅做纯闭合计算，不触碰主帧循环，不依赖渲染引擎，可完全独立使用。
//! 径向分布函数 `g0` 可由 [`crate::collider::kinetic_theory`] 的
//! Carnahan–Starling 关系式提供，颗粒温度 `Theta` 可由
//! [`crate::collider::granular_temperature`] 诊断得到。

use std::f32::consts::PI;

/// 基于 Lun 等 KTGF 闭合的稠密颗粒输运系数计算器。
///
/// 构造时固定法向恢复系数 `e`，随后可对任意 `(rho_s, phi, d, g0, Theta)`
/// 状态计算体积黏性与碰撞耗散率。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GranularTransport {
    /// 法向恢复系数 `e`，取值范围 `[0, 1]`。
    restitution: f32,
}

impl GranularTransport {
    /// 以给定的法向恢复系数 `e` 构造输运计算器。
    ///
    /// 当 `e` 非有限或不落在 `[0, 1]` 时返回 `None`。
    #[must_use]
    pub fn new(restitution: f32) -> Option<Self> {
        if !restitution.is_finite() || !(0.0..=1.0).contains(&restitution) {
            return None;
        }
        Some(Self { restitution })
    }

    /// 返回法向恢复系数 `e`。
    #[must_use]
    pub fn restitution(&self) -> f32 {
        self.restitution
    }

    /// 校验一组颗粒状态参数是否物理有效且全部有限。
    ///
    /// 要求 `rho_s > 0`、`phi ∈ [0, 1)`、`d > 0`、`g0 >= 1`、`Theta >= 0`。
    fn valid_state(rho_s: f32, phi: f32, d: f32, g0: f32, theta: f32) -> bool {
        rho_s.is_finite()
            && phi.is_finite()
            && d.is_finite()
            && g0.is_finite()
            && theta.is_finite()
            && rho_s > 0.0
            && (0.0..1.0).contains(&phi)
            && d > 0.0
            && g0 >= 1.0
            && theta >= 0.0
    }

    /// 体积黏性 `xi = (4/3) * rho_s * phi^2 * d * g0 * (1 + e) * sqrt(Theta / pi)`。
    ///
    /// 参数：
    /// * `rho_s`：颗粒材料密度（`> 0`）；
    /// * `phi`：体积分数（`[0, 1)`）；
    /// * `d`：颗粒直径（`> 0`）；
    /// * `g0`：接触处径向分布函数值（`>= 1`）；
    /// * `theta`：颗粒温度（`>= 0`）。
    ///
    /// 任一参数非法时返回 `None`。
    #[must_use]
    pub fn bulk_viscosity(&self, rho_s: f32, phi: f32, d: f32, g0: f32, theta: f32) -> Option<f32> {
        if !Self::valid_state(rho_s, phi, d, g0, theta) {
            return None;
        }
        let e = self.restitution;
        let sqrt_theta_over_pi = (theta / PI).sqrt();
        let xi = (4.0 / 3.0) * rho_s * (phi * phi) * d * g0 * (1.0 + e) * sqrt_theta_over_pi;
        Some(xi)
    }

    /// 碰撞能量耗散率
    /// `gamma = (12 * (1 - e^2) / (d * sqrt(pi))) * rho_s * phi^2 * g0 * Theta^(3/2)`。
    ///
    /// 弹性极限 `e = 1` 时耗散恒为 `0`；`Theta = 0` 时亦为 `0`。
    ///
    /// 任一参数非法时返回 `None`。
    #[must_use]
    pub fn collisional_dissipation(
        &self,
        rho_s: f32,
        phi: f32,
        d: f32,
        g0: f32,
        theta: f32,
    ) -> Option<f32> {
        if !Self::valid_state(rho_s, phi, d, g0, theta) {
            return None;
        }
        let e = self.restitution;
        let one_minus_e2 = 1.0 - e * e;
        let sqrt_pi = PI.sqrt();
        // Theta^(3/2) = Theta * sqrt(Theta)，避免使用被禁用的 powf。
        let theta_three_halves = theta * theta.sqrt();
        let prefactor = 12.0 * one_minus_e2 / (d * sqrt_pi);
        let gamma = prefactor * rho_s * (phi * phi) * g0 * theta_three_halves;
        Some(gamma)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RHO_S: f32 = 2000.0;
    const PHI: f32 = 0.5;
    const D: f32 = 0.01;
    const G0: f32 = 2.0;
    const E: f32 = 0.9;
    const THETA: f32 = 0.5;

    #[test]
    fn rejects_out_of_range_restitution() {
        assert!(GranularTransport::new(-0.01).is_none());
        assert!(GranularTransport::new(1.01).is_none());
        assert!(GranularTransport::new(f32::NAN).is_none());
        assert!(GranularTransport::new(0.0).is_some());
        assert!(GranularTransport::new(1.0).is_some());
    }

    #[test]
    fn restitution_round_trips() {
        let t = GranularTransport::new(E).unwrap();
        assert!((t.restitution() - E).abs() < 1e-6);
    }

    #[test]
    fn bulk_viscosity_matches_hand_value() {
        let t = GranularTransport::new(E).unwrap();
        let xi = t.bulk_viscosity(RHO_S, PHI, D, G0, THETA).unwrap();
        // 手算：(4/3)*2000*0.25*0.01*2*1.9*sqrt(0.5/pi) ≈ 10.1065
        assert!((xi - 10.1065).abs() < 1e-2, "xi = {xi}");
    }

    #[test]
    fn collisional_dissipation_matches_hand_value() {
        let t = GranularTransport::new(E).unwrap();
        let gamma = t.collisional_dissipation(RHO_S, PHI, D, G0, THETA).unwrap();
        // 手算：(12*0.19/(0.01*sqrt(pi)))*2000*0.25*2*(0.5^1.5) ≈ 45479.6
        assert!((gamma - 45479.6).abs() < 1.0, "gamma = {gamma}");
    }

    #[test]
    fn bulk_viscosity_scales_with_sqrt_theta() {
        let t = GranularTransport::new(E).unwrap();
        let base = t.bulk_viscosity(RHO_S, PHI, D, G0, THETA).unwrap();
        // Theta 增大 4 倍 → sqrt(Theta) 增大 2 倍 → xi 增大 2 倍。
        let scaled = t.bulk_viscosity(RHO_S, PHI, D, G0, THETA * 4.0).unwrap();
        assert!(
            (scaled / base - 2.0).abs() < 1e-3,
            "ratio = {}",
            scaled / base
        );
    }

    #[test]
    fn bulk_viscosity_scales_with_phi_squared() {
        let t = GranularTransport::new(E).unwrap();
        let base = t.bulk_viscosity(RHO_S, 0.2, D, G0, THETA).unwrap();
        // phi 0.2 → 0.4 翻倍 → phi^2 增大 4 倍。
        let scaled = t.bulk_viscosity(RHO_S, 0.4, D, G0, THETA).unwrap();
        assert!(
            (scaled / base - 4.0).abs() < 1e-3,
            "ratio = {}",
            scaled / base
        );
    }

    #[test]
    fn dissipation_scales_with_theta_three_halves() {
        let t = GranularTransport::new(E).unwrap();
        let base = t.collisional_dissipation(RHO_S, PHI, D, G0, THETA).unwrap();
        // Theta 增大 4 倍 → Theta^(3/2) 增大 8 倍。
        let scaled = t
            .collisional_dissipation(RHO_S, PHI, D, G0, THETA * 4.0)
            .unwrap();
        assert!(
            (scaled / base - 8.0).abs() < 1e-3,
            "ratio = {}",
            scaled / base
        );
    }

    #[test]
    fn elastic_limit_has_zero_dissipation() {
        let t = GranularTransport::new(1.0).unwrap();
        let gamma = t.collisional_dissipation(RHO_S, PHI, D, G0, THETA).unwrap();
        assert!(gamma.abs() < 1e-6, "gamma = {gamma}");
        // 弹性极限下体积黏性仍非零。
        let xi = t.bulk_viscosity(RHO_S, PHI, D, G0, THETA).unwrap();
        assert!(xi > 0.0);
    }

    #[test]
    fn zero_temperature_yields_zero_transport() {
        let t = GranularTransport::new(E).unwrap();
        assert_eq!(t.bulk_viscosity(RHO_S, PHI, D, G0, 0.0), Some(0.0));
        assert_eq!(t.collisional_dissipation(RHO_S, PHI, D, G0, 0.0), Some(0.0));
    }

    #[test]
    fn rejects_invalid_state() {
        let t = GranularTransport::new(E).unwrap();
        assert!(t.bulk_viscosity(0.0, PHI, D, G0, THETA).is_none());
        assert!(t.bulk_viscosity(RHO_S, 1.0, D, G0, THETA).is_none());
        assert!(t.bulk_viscosity(RHO_S, PHI, 0.0, G0, THETA).is_none());
        assert!(t.bulk_viscosity(RHO_S, PHI, D, 0.5, THETA).is_none());
        assert!(t.bulk_viscosity(RHO_S, PHI, D, G0, -0.1).is_none());
        assert!(t
            .collisional_dissipation(RHO_S, -0.1, D, G0, THETA)
            .is_none());
    }
}
