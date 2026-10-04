//! 颗粒输运的 Péclet 数：对流输运与扩散输运的无量纲比值。
//!
//! Péclet 数 `Pe` 衡量一个颗粒体系中“定向对流/剪切输运”相对于“随机自扩散”
//! 的相对强弱：
//!
//! ```text
//!        对流速率     L · v
//!   Pe = -------- = -------
//!        扩散速率       D
//! ```
//!
//! 其中 `L` 为特征长度（通常取颗粒直径或剪切带厚度），`v` 为特征对流速度，
//! `D` 为颗粒自扩散系数（可由 [`super::granular_self_diffusion`] 的均方位移给出）。
//! 等价地，`Pe` 是扩散特征时间 `τ_D = L²/D` 与对流特征时间 `τ_A = L/v` 之比
//! `Pe = τ_D / τ_A`：
//!
//! - `Pe < 1`：扩散主导，颗粒随机混合快于定向迁移。
//! - `Pe ≥ 1`：对流主导，定向迁移（剪切、重力驱动流）主导输运。
//!
//! 对剪切驱动的颗粒流，常以剪切率 `γ̇` 与长度 `L` 构造特征速度 `v = γ̇ · L`，
//! 此时 `Pe = γ̇ · L² / D`。
//!
//! 纯函数式、零耦合的分析原语：不依赖任何求解器状态或时间推进，仅做无量纲化。
//! 可消费 [`super::granular_self_diffusion`] 产出的扩散系数 `D`。

/// 颗粒输运的 Péclet 数及其构造标度。
///
/// 使用 [`GranularPeclet::from_advection`]（给定对流速度）或
/// [`GranularPeclet::from_shear`]（给定剪切率）构造。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GranularPeclet {
    length: f32,
    velocity: f32,
    diffusivity: f32,
    peclet: f32,
}

/// Péclet 数判定的输运机制。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PecletRegime {
    /// `Pe < 1`：扩散（随机混合）主导。
    DiffusionDominated,
    /// `Pe ≥ 1`：对流（定向迁移）主导。
    AdvectionDominated,
}

/// 对流与扩散平衡的 Péclet 阈值。
pub const PECLET_CRITICAL: f32 = 1.0;

impl GranularPeclet {
    /// 由特征长度 `length`、对流速度 `velocity` 与自扩散系数 `diffusivity`
    /// 构造 `Pe = length · |velocity| / diffusivity`。
    ///
    /// 要求全部参数有限、`length > 0`、`diffusivity > 0`；否则返回 `None`。
    /// `velocity` 允许为负（取其绝对值作为对流速度标度）。
    pub fn from_advection(length: f32, velocity: f32, diffusivity: f32) -> Option<Self> {
        if !length.is_finite() || !velocity.is_finite() || !diffusivity.is_finite() {
            return None;
        }
        if length <= 0.0 || diffusivity <= 0.0 {
            return None;
        }
        let speed = velocity.abs();
        let peclet = length * speed / diffusivity;
        Some(Self {
            length,
            velocity: speed,
            diffusivity,
            peclet,
        })
    }

    /// 由特征长度 `length`、剪切率 `shear_rate` 与自扩散系数 `diffusivity`
    /// 构造剪切驱动的 Péclet 数。
    ///
    /// 特征对流速度取 `v = |shear_rate| · length`，故
    /// `Pe = |shear_rate| · length² / diffusivity`。
    ///
    /// 要求全部参数有限、`length > 0`、`diffusivity > 0`；否则返回 `None`。
    pub fn from_shear(length: f32, shear_rate: f32, diffusivity: f32) -> Option<Self> {
        if !shear_rate.is_finite() {
            return None;
        }
        let velocity = shear_rate.abs() * length;
        Self::from_advection(length, velocity, diffusivity)
    }

    /// Péclet 数 `Pe`（无量纲，恒非负）。
    pub fn peclet_number(&self) -> f32 {
        self.peclet
    }

    /// 构造所用的特征长度 `L`。
    pub fn length_scale(&self) -> f32 {
        self.length
    }

    /// 构造所用的特征对流速度 `v`（恒非负）。
    pub fn velocity_scale(&self) -> f32 {
        self.velocity
    }

    /// 构造所用的自扩散系数 `D`。
    pub fn diffusivity(&self) -> f32 {
        self.diffusivity
    }

    /// 扩散特征时间 `τ_D = L² / D`。
    pub fn diffusive_time(&self) -> f32 {
        self.length * self.length / self.diffusivity
    }

    /// 对流特征时间 `τ_A = L / v`；当对流速度为零时返回 `None`（无穷大）。
    pub fn advective_time(&self) -> Option<f32> {
        if self.velocity <= 0.0 {
            return None;
        }
        Some(self.length / self.velocity)
    }

    /// 当前输运机制。
    pub fn regime(&self) -> PecletRegime {
        if self.peclet < PECLET_CRITICAL {
            PecletRegime::DiffusionDominated
        } else {
            PecletRegime::AdvectionDominated
        }
    }

    /// 是否对流主导（`Pe ≥ 1`）。
    pub fn is_advection_dominated(&self) -> bool {
        self.peclet >= PECLET_CRITICAL
    }

    /// 是否扩散主导（`Pe < 1`）。
    pub fn is_diffusion_dominated(&self) -> bool {
        self.peclet < PECLET_CRITICAL
    }

    /// 是否处于对流-扩散平衡（`|Pe − 1| ≤ tol`）。
    pub fn is_balanced(&self, tol: f32) -> bool {
        (self.peclet - PECLET_CRITICAL).abs() <= tol
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    #[test]
    fn advection_peclet_matches_definition() {
        // L=2, v=3, D=4 -> Pe = 2*3/4 = 1.5
        let pe = GranularPeclet::from_advection(2.0, 3.0, 4.0).unwrap();
        assert!((pe.peclet_number() - 1.5).abs() <= EPS);
        assert_eq!(pe.regime(), PecletRegime::AdvectionDominated);
        assert!(pe.is_advection_dominated());
        assert!(!pe.is_diffusion_dominated());
    }

    #[test]
    fn negative_velocity_uses_magnitude() {
        let pos = GranularPeclet::from_advection(2.0, 3.0, 4.0).unwrap();
        let neg = GranularPeclet::from_advection(2.0, -3.0, 4.0).unwrap();
        assert!((pos.peclet_number() - neg.peclet_number()).abs() <= EPS);
        assert!(neg.velocity_scale() >= 0.0);
    }

    #[test]
    fn diffusion_dominated_below_unity() {
        // L=1, v=1, D=4 -> Pe = 0.25
        let pe = GranularPeclet::from_advection(1.0, 1.0, 4.0).unwrap();
        assert!((pe.peclet_number() - 0.25).abs() <= EPS);
        assert_eq!(pe.regime(), PecletRegime::DiffusionDominated);
        assert!(pe.is_diffusion_dominated());
    }

    #[test]
    fn shear_peclet_matches_definition() {
        // L=2, gamma=5, D=4 -> v=10, Pe = 5*4/4 = 5
        let pe = GranularPeclet::from_shear(2.0, 5.0, 4.0).unwrap();
        assert!((pe.velocity_scale() - 10.0).abs() <= EPS);
        assert!((pe.peclet_number() - 5.0).abs() <= EPS);
    }

    #[test]
    fn characteristic_times_consistent() {
        // Pe = tau_D / tau_A
        let pe = GranularPeclet::from_advection(2.0, 3.0, 4.0).unwrap();
        let td = pe.diffusive_time();
        let ta = pe.advective_time().unwrap();
        assert!((td - 1.0).abs() <= EPS); // 4/4
        assert!((ta - 2.0_f32 / 3.0).abs() <= EPS);
        assert!((pe.peclet_number() - td / ta).abs() <= 1e-4);
    }

    #[test]
    fn zero_velocity_has_no_advective_time() {
        let pe = GranularPeclet::from_advection(2.0, 0.0, 4.0).unwrap();
        assert!((pe.peclet_number()).abs() <= EPS);
        assert_eq!(pe.regime(), PecletRegime::DiffusionDominated);
        assert!(pe.advective_time().is_none());
    }

    #[test]
    fn balanced_within_tolerance() {
        // L=4, v=1, D=4 -> Pe = 1.0 exactly
        let pe = GranularPeclet::from_advection(4.0, 1.0, 4.0).unwrap();
        assert!((pe.peclet_number() - 1.0).abs() <= EPS);
        assert!(pe.is_balanced(1e-3));
        assert_eq!(pe.regime(), PecletRegime::AdvectionDominated);
    }

    #[test]
    fn rejects_invalid_parameters() {
        assert!(GranularPeclet::from_advection(0.0, 1.0, 1.0).is_none());
        assert!(GranularPeclet::from_advection(-1.0, 1.0, 1.0).is_none());
        assert!(GranularPeclet::from_advection(1.0, 1.0, 0.0).is_none());
        assert!(GranularPeclet::from_advection(1.0, 1.0, -1.0).is_none());
        assert!(GranularPeclet::from_advection(f32::NAN, 1.0, 1.0).is_none());
        assert!(GranularPeclet::from_advection(1.0, f32::INFINITY, 1.0).is_none());
        assert!(GranularPeclet::from_shear(1.0, f32::NAN, 1.0).is_none());
    }
}
