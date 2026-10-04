//! 颗粒自扩散系数的两快照（Einstein 关系）估计。
//!
//! 给定同一组颗粒在两个时刻的位置快照以及时间间隔 `dt`，本模块通过
//! 均方位移（mean squared displacement, MSD）与 Einstein 扩散关系估计
//! 颗粒的自扩散系数：
//!
//! ```text
//! MSD   = (1/N) * Σ |Δr_i|^2
//! D     = MSD / (6 * dt)            （三维 Einstein 关系）
//! D_x   = MSD_x / (2 * dt)          （单分量）
//! ```
//!
//! 其中 `Δr_i = r_i(t + dt) − r_i(t)`。为了把真实的扩散运动与整体平流
//! （bulk drift）区分开，本模块同时提供去漂移（drift-corrected）版本：
//! 先求整体平均位移 `d_mean`，再对 `Δr_i − d_mean` 求均方位移，从而消除
//! 均匀流动对扩散估计的污染。
//!
//! Einstein 关系是纯运动学定义，不含材料相关的约定常数，因此本模块给出的
//! 数值无歧义、可逐项手算验证。它不触碰主帧循环、不依赖渲染引擎，可独立使用，
//! 与 [`crate::collider::nonaffine_displacement`]（D²min 非仿射度量）互补：
//! 后者刻画局部非仿射重排，本模块刻画长程扩散输运。

use glam::Vec3;

/// 两快照颗粒自扩散分析结果。
///
/// 由 [`SelfDiffusionAnalysis::from_snapshots`] 构造，缓存均方位移、整体漂移
/// 与去漂移均方位移，随后以各 getter 提供扩散系数等派生量。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SelfDiffusionAnalysis {
    particle_count: usize,
    dt: f32,
    mean_displacement: Vec3,
    msd_total: f32,
    msd_components: Vec3,
    msd_fluctuating: f32,
}

impl SelfDiffusionAnalysis {
    /// 由初末两组位置快照与时间间隔 `dt` 构造分析结果。
    ///
    /// 当两组快照长度不同、为空、`dt` 非正，或任一坐标非有限时返回 `None`。
    #[must_use]
    pub fn from_snapshots(initial: &[Vec3], final_positions: &[Vec3], dt: f32) -> Option<Self> {
        if initial.is_empty()
            || initial.len() != final_positions.len()
            || !dt.is_finite()
            || dt <= 0.0
        {
            return None;
        }
        let n = initial.len();

        let mut sum_disp = Vec3::ZERO;
        for (r0, r1) in initial.iter().zip(final_positions.iter()) {
            if !r0.is_finite() || !r1.is_finite() {
                return None;
            }
            sum_disp += *r1 - *r0;
        }
        let inv_n = 1.0 / n as f32;
        let mean_displacement = sum_disp * inv_n;

        let mut msd_total = 0.0_f32;
        let mut msd_components = Vec3::ZERO;
        let mut msd_fluctuating = 0.0_f32;
        for (r0, r1) in initial.iter().zip(final_positions.iter()) {
            let d = *r1 - *r0;
            msd_total += d.length_squared();
            msd_components += d * d;
            let f = d - mean_displacement;
            msd_fluctuating += f.length_squared();
        }
        msd_total *= inv_n;
        msd_components *= inv_n;
        msd_fluctuating *= inv_n;

        Some(Self {
            particle_count: n,
            dt,
            mean_displacement,
            msd_total,
            msd_components,
            msd_fluctuating,
        })
    }

    /// 参与分析的颗粒数量。
    #[must_use]
    pub fn particle_count(&self) -> usize {
        self.particle_count
    }

    /// 两快照之间的时间间隔。
    #[must_use]
    pub fn dt(&self) -> f32 {
        self.dt
    }

    /// 整体平均位移（bulk drift）向量 `d_mean = (1/N) Σ Δr_i`。
    #[must_use]
    pub fn mean_displacement(&self) -> Vec3 {
        self.mean_displacement
    }

    /// 总均方位移 `MSD = (1/N) Σ |Δr_i|^2`（含整体漂移）。
    #[must_use]
    pub fn mean_squared_displacement(&self) -> f32 {
        self.msd_total
    }

    /// 分量均方位移 `(MSD_x, MSD_y, MSD_z)`（含整体漂移）。
    #[must_use]
    pub fn mean_squared_displacement_components(&self) -> Vec3 {
        self.msd_components
    }

    /// 去漂移均方位移 `(1/N) Σ |Δr_i − d_mean|^2`。
    #[must_use]
    pub fn fluctuating_mean_squared_displacement(&self) -> f32 {
        self.msd_fluctuating
    }

    /// 均方根位移 `sqrt(MSD)`（含整体漂移）。
    #[must_use]
    pub fn rms_displacement(&self) -> f32 {
        self.msd_total.sqrt()
    }

    /// 三维自扩散系数 `D = MSD / (6 * dt)`（含整体漂移）。
    #[must_use]
    pub fn self_diffusion_coefficient(&self) -> f32 {
        self.msd_total / (6.0 * self.dt)
    }

    /// 去漂移三维自扩散系数 `D = MSD_fluct / (6 * dt)`。
    ///
    /// 推荐在存在整体平流（如重力沉降、料斗流出）时使用该值作为真实扩散估计。
    #[must_use]
    pub fn self_diffusion_coefficient_drift_corrected(&self) -> f32 {
        self.msd_fluctuating / (6.0 * self.dt)
    }

    /// 各分量自扩散系数 `D_i = MSD_i / (2 * dt)`（含整体漂移）。
    #[must_use]
    pub fn component_diffusion(&self) -> Vec3 {
        self.msd_components / (2.0 * self.dt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_input() {
        assert!(SelfDiffusionAnalysis::from_snapshots(&[], &[], 1.0).is_none());
        assert!(SelfDiffusionAnalysis::from_snapshots(&[Vec3::ZERO], &[], 1.0).is_none());
        assert!(SelfDiffusionAnalysis::from_snapshots(&[Vec3::ZERO], &[Vec3::X], 0.0).is_none());
        assert!(SelfDiffusionAnalysis::from_snapshots(&[Vec3::ZERO], &[Vec3::X], -1.0).is_none());
        let nan = Vec3::new(f32::NAN, 0.0, 0.0);
        assert!(SelfDiffusionAnalysis::from_snapshots(&[nan], &[Vec3::X], 1.0).is_none());
    }

    #[test]
    fn symmetric_displacement_has_no_drift() {
        let initial = [Vec3::ZERO, Vec3::ZERO];
        let final_positions = [Vec3::X, Vec3::NEG_X];
        let a = SelfDiffusionAnalysis::from_snapshots(&initial, &final_positions, 0.5).unwrap();
        assert_eq!(a.particle_count(), 2);
        assert!(a.mean_displacement().length() < 1e-6);
        // MSD = (1 + 1) / 2 = 1.0
        assert!((a.mean_squared_displacement() - 1.0).abs() < 1e-6);
        // 无漂移 → 去漂移 MSD 与原始 MSD 相同。
        assert!((a.fluctuating_mean_squared_displacement() - 1.0).abs() < 1e-6);
        // D = 1 / (6 * 0.5) = 1/3
        assert!((a.self_diffusion_coefficient() - 1.0 / 3.0).abs() < 1e-6);
        assert!((a.self_diffusion_coefficient_drift_corrected() - 1.0 / 3.0).abs() < 1e-6);
        // 仅 x 方向位移：MSD_x = 1，MSD_y = MSD_z = 0。
        let c = a.mean_squared_displacement_components();
        assert!((c.x - 1.0).abs() < 1e-6);
        assert!(c.y.abs() < 1e-6 && c.z.abs() < 1e-6);
        // D_x = 1 / (2 * 0.5) = 1.0
        assert!((a.component_diffusion().x - 1.0).abs() < 1e-6);
    }

    #[test]
    fn drift_correction_removes_bulk_flow() {
        let initial = [Vec3::ZERO, Vec3::ZERO];
        let final_positions = [Vec3::new(2.0, 0.0, 0.0), Vec3::new(4.0, 0.0, 0.0)];
        let a = SelfDiffusionAnalysis::from_snapshots(&initial, &final_positions, 1.0).unwrap();
        // 平均位移 = (3, 0, 0)
        assert!((a.mean_displacement().x - 3.0).abs() < 1e-6);
        // 原始 MSD = (4 + 16) / 2 = 10
        assert!((a.mean_squared_displacement() - 10.0).abs() < 1e-6);
        // 去漂移：Δr − mean = (-1,0,0),(1,0,0) → MSD_fluct = 1
        assert!((a.fluctuating_mean_squared_displacement() - 1.0).abs() < 1e-6);
        // D_raw = 10 / 6 ≈ 1.6667
        assert!((a.self_diffusion_coefficient() - 10.0 / 6.0).abs() < 1e-5);
        // D_drift = 1 / 6 ≈ 0.16667
        assert!((a.self_diffusion_coefficient_drift_corrected() - 1.0 / 6.0).abs() < 1e-6);
        // rms = sqrt(10)
        assert!((a.rms_displacement() - 10.0_f32.sqrt()).abs() < 1e-5);
    }

    #[test]
    fn zero_displacement_yields_zero_diffusion() {
        let initial = [Vec3::new(1.0, 2.0, 3.0), Vec3::new(-1.0, 0.0, 5.0)];
        let a = SelfDiffusionAnalysis::from_snapshots(&initial, &initial, 2.0).unwrap();
        assert_eq!(a.mean_squared_displacement(), 0.0);
        assert_eq!(a.self_diffusion_coefficient(), 0.0);
        assert_eq!(a.self_diffusion_coefficient_drift_corrected(), 0.0);
        assert_eq!(a.component_diffusion(), Vec3::ZERO);
        assert_eq!(a.rms_displacement(), 0.0);
    }

    #[test]
    fn isotropic_displacement_splits_evenly() {
        // 四颗粒分别沿 ±x、±y 单位位移：各向同性，无漂移。
        let initial = [Vec3::ZERO; 4];
        let final_positions = [Vec3::X, Vec3::NEG_X, Vec3::Y, Vec3::NEG_Y];
        let a = SelfDiffusionAnalysis::from_snapshots(&initial, &final_positions, 1.0).unwrap();
        assert!(a.mean_displacement().length() < 1e-6);
        // MSD = (1+1+1+1)/4 = 1
        assert!((a.mean_squared_displacement() - 1.0).abs() < 1e-6);
        let c = a.mean_squared_displacement_components();
        // x、y 各贡献 2/4 = 0.5，z 为 0。
        assert!((c.x - 0.5).abs() < 1e-6);
        assert!((c.y - 0.5).abs() < 1e-6);
        assert!(c.z.abs() < 1e-6);
    }

    #[test]
    fn diffusion_scales_inversely_with_dt() {
        let initial = [Vec3::ZERO, Vec3::ZERO];
        let final_positions = [Vec3::X, Vec3::NEG_X];
        let a = SelfDiffusionAnalysis::from_snapshots(&initial, &final_positions, 0.5).unwrap();
        let b = SelfDiffusionAnalysis::from_snapshots(&initial, &final_positions, 1.0).unwrap();
        // 相同位移、dt 翻倍 → D 减半。
        assert!(
            (a.self_diffusion_coefficient() / b.self_diffusion_coefficient() - 2.0).abs() < 1e-5
        );
    }
}
