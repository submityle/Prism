//! `p–q` 应力路径增量刻画（三轴加载路径分析，零耦合原语）。
//!
//! 在临界状态土力学中，一次加载常用 `p–q` 平面中的应力路径描述：静水应力
//! `p`（平均应力）为横轴、von Mises 等效应力 `q` 为纵轴。给定加载前后两组
//! 主应力，可得应力增量与路径斜率：
//!
//! ```text
//!   Δp = p_f − p_i,   Δq = q_f − q_i,   路径斜率 = Δq / Δp
//! ```
//!
//! 几个标准参考路径（拉为正约定，`η = q/p` 为应力比）：
//!
//! - 常规三轴压缩排水路径 `Δq/Δp = 3`。
//! - 等 `p` 路径（纯剪加载）`Δp = 0`，路径竖直，斜率无定义。
//! - 等 `q` 固结路径 `Δq = 0`，路径水平，斜率为 `0`。
//!
//! 纯函数式、零耦合：在 [`super::stress_invariants::StressInvariants`] 之上对两组
//! 主应力做差分，不触碰主帧循环、不依赖渲染引擎。

use super::stress_invariants::StressInvariants;

/// 低于该值的 `|Δp|` 视为竖直路径，斜率无定义。
const DELTA_P_EPS: f32 = 1e-9;

/// 低于该值的 `|p|` 视为无定义应力比。
const MEAN_STRESS_EPS: f32 = 1e-9;

/// 由加载前后两组主应力得到的 `p–q` 应力路径增量。
///
/// 由 [`StressPath::from_states`] 构造。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StressPath {
    initial_p: f32,
    initial_q: f32,
    final_p: f32,
    final_q: f32,
}

impl StressPath {
    /// 由加载前主应力 `initial` 与加载后主应力 `final_stresses`（顺序任意，
    /// 拉为正）构造应力路径。任一组非有限时返回 `None`。
    pub fn from_states(initial: [f32; 3], final_stresses: [f32; 3]) -> Option<Self> {
        let inv_i = StressInvariants::from_principal_stresses(initial)?;
        let inv_f = StressInvariants::from_principal_stresses(final_stresses)?;
        Some(Self {
            initial_p: inv_i.mean_stress(),
            initial_q: inv_i.von_mises_equivalent(),
            final_p: inv_f.mean_stress(),
            final_q: inv_f.von_mises_equivalent(),
        })
    }

    /// 加载前静水应力 `p_i`。
    pub fn initial_mean_stress(&self) -> f32 {
        self.initial_p
    }

    /// 加载前等效应力 `q_i`。
    pub fn initial_von_mises(&self) -> f32 {
        self.initial_q
    }

    /// 加载后静水应力 `p_f`。
    pub fn final_mean_stress(&self) -> f32 {
        self.final_p
    }

    /// 加载后等效应力 `q_f`。
    pub fn final_von_mises(&self) -> f32 {
        self.final_q
    }

    /// 静水应力增量 `Δp = p_f − p_i`。
    pub fn delta_mean_stress(&self) -> f32 {
        self.final_p - self.initial_p
    }

    /// 等效应力增量 `Δq = q_f − q_i`。
    pub fn delta_von_mises(&self) -> f32 {
        self.final_q - self.initial_q
    }

    /// 应力路径斜率 `Δq / Δp`；竖直路径（`|Δp| ≤ ε`）无定义，返回 `None`。
    pub fn slope(&self) -> Option<f32> {
        let dp = self.delta_mean_stress();
        if dp.abs() <= DELTA_P_EPS {
            return None;
        }
        Some(self.delta_von_mises() / dp)
    }

    /// `p–q` 平面中的路径长度 `sqrt(Δp² + Δq²)`。
    pub fn path_length(&self) -> f32 {
        let dp = self.delta_mean_stress();
        let dq = self.delta_von_mises();
        (dp * dp + dq * dq).sqrt()
    }

    /// 加载前应力比 `η_i = q_i / p_i`；`|p_i| ≤ ε` 时无定义，返回 `None`。
    pub fn initial_stress_ratio(&self) -> Option<f32> {
        if self.initial_p.abs() <= MEAN_STRESS_EPS {
            return None;
        }
        Some(self.initial_q / self.initial_p)
    }

    /// 加载后应力比 `η_f = q_f / p_f`；`|p_f| ≤ ε` 时无定义，返回 `None`。
    pub fn final_stress_ratio(&self) -> Option<f32> {
        if self.final_p.abs() <= MEAN_STRESS_EPS {
            return None;
        }
        Some(self.final_q / self.final_p)
    }

    /// 是否为加载（偏应力增大，`Δq > 0`）。
    pub fn is_loading(&self) -> bool {
        self.delta_von_mises() > 0.0
    }

    /// 是否为卸载（偏应力减小，`Δq < 0`）。
    pub fn is_unloading(&self) -> bool {
        self.delta_von_mises() < 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    #[test]
    fn triaxial_compression_drained_slope_is_three() {
        // 初始各向同性 σ = [-1,-1,-1]: p=-1, q=0.
        // 终态 σ = [-1,-1,-4]: p=-2, q=3. Δp=-1, Δq=3 -> 斜率 = -3.
        // 取拉为正，常规三轴压缩路径 |Δq/Δp| = 3。
        let path = StressPath::from_states([-1.0, -1.0, -1.0], [-1.0, -1.0, -4.0]).unwrap();
        assert!((path.delta_mean_stress() + 1.0).abs() <= EPS);
        assert!((path.delta_von_mises() - 3.0).abs() <= EPS);
        assert!((path.slope().unwrap() + 3.0).abs() <= EPS);
        assert!(path.is_loading());
    }

    #[test]
    fn constant_p_path_is_vertical() {
        // 等 p 纯剪加载：σ 从 [1,0,-1] 到 [2,0,-2]，p 恒为 0。
        let path = StressPath::from_states([1.0, 0.0, -1.0], [2.0, 0.0, -2.0]).unwrap();
        assert!(path.delta_mean_stress().abs() <= EPS);
        assert!(path.slope().is_none());
        assert!(path.is_loading());
    }

    #[test]
    fn constant_q_path_is_horizontal() {
        // 等 q 固结：σ 从 [3,0,0] 到 [4,1,1]，q 恒为 3，p 从 1 升到 2。
        let path = StressPath::from_states([3.0, 0.0, 0.0], [4.0, 1.0, 1.0]).unwrap();
        assert!((path.initial_von_mises() - 3.0).abs() <= EPS);
        assert!((path.final_von_mises() - 3.0).abs() <= EPS);
        assert!(path.delta_von_mises().abs() <= EPS);
        assert!((path.slope().unwrap()).abs() <= EPS);
        assert!(!path.is_loading());
        assert!(!path.is_unloading());
    }

    #[test]
    fn path_length_is_euclidean() {
        // Δp=-1, Δq=3 -> 长度 = sqrt(1+9) = sqrt(10).
        let path = StressPath::from_states([-1.0, -1.0, -1.0], [-1.0, -1.0, -4.0]).unwrap();
        assert!((path.path_length() - 10.0_f32.sqrt()).abs() <= EPS);
    }

    #[test]
    fn stress_ratios_pass_through() {
        // 终态 σ=[4,1,1]: p=2, q=3 -> η=1.5.
        let path = StressPath::from_states([3.0, 0.0, 0.0], [4.0, 1.0, 1.0]).unwrap();
        assert!((path.final_stress_ratio().unwrap() - 1.5).abs() <= EPS);
        // 初态 σ=[3,0,0]: p=1, q=3 -> η=3.
        assert!((path.initial_stress_ratio().unwrap() - 3.0).abs() <= EPS);
    }

    #[test]
    fn unloading_is_detected() {
        // 从 [2,0,-2]（q=sqrt(12)≈3.46）到 [1,0,-1]（q≈1.73），Δq<0。
        let path = StressPath::from_states([2.0, 0.0, -2.0], [1.0, 0.0, -1.0]).unwrap();
        assert!(path.is_unloading());
        assert!(!path.is_loading());
    }

    #[test]
    fn zero_mean_stress_has_no_ratio() {
        // 纯剪 σ=[1,0,-1]: p=0 -> 应力比无定义。
        let path = StressPath::from_states([1.0, 0.0, -1.0], [2.0, 0.0, -2.0]).unwrap();
        assert!(path.initial_stress_ratio().is_none());
        assert!(path.final_stress_ratio().is_none());
    }

    #[test]
    fn rejects_non_finite() {
        assert!(StressPath::from_states([f32::NAN, 0.0, 0.0], [1.0, 0.0, 0.0]).is_none());
        assert!(StressPath::from_states([1.0, 0.0, 0.0], [f32::INFINITY, 0.0, 0.0]).is_none());
    }
}
