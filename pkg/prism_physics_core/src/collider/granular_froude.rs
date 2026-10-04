//! 颗粒自由面流（泻槽/雪崩/溃坝）的 Froude 数与临界状态分析。
//!
//! Froude 数衡量惯性（对流）速度与重力表面波速之比，是自由面浅层颗粒流的
//! 核心无量纲数：
//!
//! ```text
//! Fr = u / √(g · h)
//! ```
//!
//! 其中 `u` 为（深度平均）流速，`g` 为重力加速度，`h` 为流动层厚度。
//! √(g·h) 是浅水重力波的传播速度，故：
//! - `Fr < 1`：亚临界（subcritical），扰动可向上游传播，流动“缓”。
//! - `Fr = 1`：临界，正是水跃（hydraulic jump）发生的位置。
//! - `Fr > 1`：超临界（supercritical），扰动无法上传，流动“急”。
//!
//! 对倾角为 `θ` 的斜槽，重力沿法向分量减弱，常用修正式
//! `Fr = u / √(g · h · cos θ)`，由 [`GranularFroude::from_inclined_flow`] 提供。
//!
//! 这是一个纯函数式、零耦合的分析原语：仅做一次代数求值与阈值分类，不依赖
//! 任何求解器状态或时间推进，可被 DEM / 浅水/深度平均连续介质流等上游复用。

/// 由 Froude 数判定的自由面流区制。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FroudeRegime {
    /// 亚临界（`Fr < 1`）：扰动可向上游传播。
    Subcritical,
    /// 超临界（`Fr ≥ 1`）：扰动无法向上游传播。
    Supercritical,
}

/// 由颗粒自由面流状态推导出的 Froude 数及其临界分析。
///
/// 使用 [`GranularFroude::from_flow`] 或 [`GranularFroude::from_inclined_flow`]
/// 构造。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GranularFroude {
    speed: f32,
    flow_depth: f32,
    gravity: f32,
    /// 有效法向重力 `g·cosθ`（平槽时等于 `g`）。
    effective_gravity: f32,
    froude: f32,
}

impl GranularFroude {
    /// 平槽（水平基准）自由面流的 Froude 数 `Fr = u / √(g·h)`。
    ///
    /// 当任意输入非有限，或 `u < 0`、`h ≤ 0`、`g ≤ 0` 时返回 `None`。
    #[must_use]
    pub fn from_flow(speed: f32, flow_depth: f32, gravity: f32) -> Option<Self> {
        Self::from_effective(speed, flow_depth, gravity, gravity)
    }

    /// 倾角为 `slope_angle`（弧度）的斜槽：`Fr = u / √(g·h·cosθ)`。
    ///
    /// 除 [`from_flow`](Self::from_flow) 的约束外，还要求 `θ ∈ (-π/2, π/2)`
    /// 以保证 `cosθ > 0`；否则返回 `None`。
    #[must_use]
    pub fn from_inclined_flow(
        speed: f32,
        flow_depth: f32,
        gravity: f32,
        slope_angle: f32,
    ) -> Option<Self> {
        if !slope_angle.is_finite() {
            return None;
        }
        // cos 在 f64 下求值（规避 f32 三角函数），要求 cosθ > 0。
        let cos_theta = f64::from(slope_angle).cos() as f32;
        if cos_theta <= 0.0 {
            return None;
        }
        Self::from_effective(speed, flow_depth, gravity, gravity * cos_theta)
    }

    /// 以给定的有效法向重力构造。`effective_gravity` 必须为正且有限。
    fn from_effective(
        speed: f32,
        flow_depth: f32,
        gravity: f32,
        effective_gravity: f32,
    ) -> Option<Self> {
        if !speed.is_finite()
            || !flow_depth.is_finite()
            || !gravity.is_finite()
            || !effective_gravity.is_finite()
        {
            return None;
        }
        if speed < 0.0 || flow_depth <= 0.0 || gravity <= 0.0 || effective_gravity <= 0.0 {
            return None;
        }

        let wave_speed = (effective_gravity * flow_depth).sqrt();
        if !wave_speed.is_finite() || wave_speed <= 0.0 {
            return None;
        }
        let froude = speed / wave_speed;
        if !froude.is_finite() {
            return None;
        }

        Some(Self {
            speed,
            flow_depth,
            gravity,
            effective_gravity,
            froude,
        })
    }

    /// 流速 `u`。
    #[must_use]
    pub fn speed(&self) -> f32 {
        self.speed
    }

    /// 流动层厚度 `h`。
    #[must_use]
    pub fn flow_depth(&self) -> f32 {
        self.flow_depth
    }

    /// 重力加速度 `g`。
    #[must_use]
    pub fn gravity(&self) -> f32 {
        self.gravity
    }

    /// 有效法向重力 `g·cosθ`（平槽时等于 `g`）。
    #[must_use]
    pub fn effective_gravity(&self) -> f32 {
        self.effective_gravity
    }

    /// 浅水重力波速 `c = √(g_eff·h)`。
    #[must_use]
    pub fn wave_speed(&self) -> f32 {
        (self.effective_gravity * self.flow_depth).sqrt()
    }

    /// Froude 数 `Fr = u / √(g_eff·h)`。
    #[must_use]
    pub fn froude_number(&self) -> f32 {
        self.froude
    }

    /// 当前区制分类。
    #[must_use]
    pub fn regime(&self) -> FroudeRegime {
        if self.froude < 1.0 {
            FroudeRegime::Subcritical
        } else {
            FroudeRegime::Supercritical
        }
    }

    /// 是否亚临界（`Fr < 1`）。
    #[must_use]
    pub fn is_subcritical(&self) -> bool {
        self.froude < 1.0
    }

    /// 是否超临界（`Fr > 1`）。
    #[must_use]
    pub fn is_supercritical(&self) -> bool {
        self.froude > 1.0
    }

    /// 是否在给定容差内处于临界（`|Fr − 1| ≤ tol`，水跃发生处）。
    #[must_use]
    pub fn is_critical(&self, tolerance: f32) -> bool {
        (self.froude - 1.0).abs() <= tolerance.abs()
    }

    /// 临界流速 `u_c = √(g_eff·h)`：使 `Fr = 1` 的流速（等于波速）。
    #[must_use]
    pub fn critical_speed(&self) -> f32 {
        self.wave_speed()
    }

    /// 临界深度 `h_c = u² / g_eff`：使 `Fr = 1` 的流动层厚度。
    #[must_use]
    pub fn critical_depth(&self) -> f32 {
        self.speed * self.speed / self.effective_gravity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hand_computed_subcritical() {
        // u=1, g=9.81, h=1 → √(9.81)=3.1321, Fr≈0.3193。
        let f = GranularFroude::from_flow(1.0, 1.0, 9.81).unwrap();
        assert!((f.wave_speed() - 3.132_092).abs() <= 1e-4);
        assert!((f.froude_number() - 0.319_28).abs() <= 1e-4);
        assert_eq!(f.regime(), FroudeRegime::Subcritical);
        assert!(f.is_subcritical());
        assert!(!f.is_supercritical());
    }

    #[test]
    fn supercritical() {
        // u=10, g=9.81, h=1 → Fr≈3.193。
        let f = GranularFroude::from_flow(10.0, 1.0, 9.81).unwrap();
        assert!((f.froude_number() - 3.192_8).abs() <= 1e-3);
        assert_eq!(f.regime(), FroudeRegime::Supercritical);
        assert!(f.is_supercritical());
    }

    #[test]
    fn critical_point() {
        // 令 u = √(g·h)：u=√(9.81·2)=4.4294 → Fr=1。
        let g = 9.81_f32;
        let h = 2.0_f32;
        let u = (g * h).sqrt();
        let f = GranularFroude::from_flow(u, h, g).unwrap();
        assert!((f.froude_number() - 1.0).abs() <= 1e-5);
        assert!(f.is_critical(1e-4));
        // Fr=1 归为超临界（半开区间约定），且既不是严格 subcritical 也不是严格 supercritical。
        assert_eq!(f.regime(), FroudeRegime::Supercritical);
        assert!(!f.is_subcritical());
        assert!(!f.is_supercritical());
    }

    #[test]
    fn critical_speed_and_depth_round_trip() {
        let f = GranularFroude::from_flow(3.0, 0.5, 9.81).unwrap();
        // 以临界流速重建 → Fr=1。
        let at_crit_speed =
            GranularFroude::from_flow(f.critical_speed(), f.flow_depth(), f.gravity()).unwrap();
        assert!((at_crit_speed.froude_number() - 1.0).abs() <= 1e-5);
        // 以临界深度重建 → Fr=1。
        let at_crit_depth =
            GranularFroude::from_flow(f.speed(), f.critical_depth(), f.gravity()).unwrap();
        assert!((at_crit_depth.froude_number() - 1.0).abs() <= 1e-5);
    }

    #[test]
    fn inclined_increases_froude_relative_to_flat() {
        // 斜槽 cosθ<1 → 有效重力减小 → Fr 增大。
        let flat = GranularFroude::from_flow(2.0, 1.0, 9.81).unwrap();
        let incl = GranularFroude::from_inclined_flow(2.0, 1.0, 9.81, 1.0).unwrap(); // ~57.3°
        assert!(incl.froude_number() > flat.froude_number());
        // cos(1 rad)=0.5403 → g_eff=9.81·0.5403≈5.300。
        assert!((incl.effective_gravity() - 5.300_6).abs() <= 1e-2);
    }

    #[test]
    fn inclined_zero_angle_matches_flat() {
        let flat = GranularFroude::from_flow(2.0, 1.0, 9.81).unwrap();
        let incl = GranularFroude::from_inclined_flow(2.0, 1.0, 9.81, 0.0).unwrap();
        assert!((incl.froude_number() - flat.froude_number()).abs() <= 1e-6);
    }

    #[test]
    fn froude_scales_linearly_with_speed() {
        let base = GranularFroude::from_flow(2.0, 1.0, 9.81).unwrap();
        let doubled = GranularFroude::from_flow(4.0, 1.0, 9.81).unwrap();
        assert!((doubled.froude_number() - 2.0 * base.froude_number()).abs() <= 1e-5);
    }

    #[test]
    fn zero_speed_is_subcritical() {
        let f = GranularFroude::from_flow(0.0, 1.0, 9.81).unwrap();
        assert!(f.froude_number().abs() <= 1e-12);
        assert_eq!(f.regime(), FroudeRegime::Subcritical);
    }

    #[test]
    fn rejects_invalid_inputs() {
        assert!(GranularFroude::from_flow(-1.0, 1.0, 9.81).is_none());
        assert!(GranularFroude::from_flow(1.0, 0.0, 9.81).is_none());
        assert!(GranularFroude::from_flow(1.0, 1.0, 0.0).is_none());
        assert!(GranularFroude::from_flow(f32::NAN, 1.0, 9.81).is_none());
        // 倾角 ≥ 90° → cosθ ≤ 0。
        assert!(GranularFroude::from_inclined_flow(1.0, 1.0, 9.81, 2.0).is_none());
        assert!(GranularFroude::from_inclined_flow(1.0, 1.0, 9.81, f32::NAN).is_none());
    }
}
