//! Granular temperature and velocity-fluctuation diagnostics.
//!
//! In kinetic theories of granular flow the *granular temperature* `T`
//! measures the specific kinetic energy stored in the random (fluctuating)
//! part of the grain velocity field, analogous to thermodynamic temperature
//! for molecular gases. With grain velocities `v_i` and ensemble mean
//! `⟨v⟩ = (1/N) Σ v_i`, the fluctuation is `δv_i = v_i − ⟨v⟩` and
//!
//! ```text
//! T = (1 / (3 N)) · Σ_i δv_i · δv_i
//! ```
//!
//! i.e. one third of the mean-square velocity fluctuation (the per-degree-of-
//! freedom convention). The per-axis temperatures `T_x, T_y, T_z` expose the
//! anisotropy of the fluctuations, and the mass-weighted variant recovers the
//! physical fluctuating kinetic energy when grains differ in mass.
//!
//! This module is pure analysis of a velocity field and does not couple to the
//! simulation step.

use glam::Vec3;

/// Granular-temperature diagnostics for a velocity field.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GranularTemperature {
    grain_count: usize,
    mean_velocity: Vec3,
    component_temperatures: Vec3,
    granular_temperature: f32,
    total_mass: f32,
    fluctuation_kinetic_energy: f32,
}

impl GranularTemperature {
    /// Measures the granular temperature of an equal-mass velocity field.
    ///
    /// Returns `None` if `velocities` is empty or contains a non-finite
    /// component.
    pub fn measure(velocities: &[Vec3]) -> Option<Self> {
        if velocities.is_empty() {
            return None;
        }
        for v in velocities.iter() {
            if !v.is_finite() {
                return None;
            }
        }
        let n = velocities.len();
        let inv_n = 1.0_f64 / n as f64;

        // Mean velocity in f64.
        let (mut mx, mut my, mut mz) = (0.0_f64, 0.0_f64, 0.0_f64);
        for v in velocities.iter() {
            mx += v.x as f64;
            my += v.y as f64;
            mz += v.z as f64;
        }
        mx *= inv_n;
        my *= inv_n;
        mz *= inv_n;

        // Mean-square fluctuation per axis.
        let (mut sx, mut sy, mut sz) = (0.0_f64, 0.0_f64, 0.0_f64);
        for v in velocities.iter() {
            let dx = v.x as f64 - mx;
            let dy = v.y as f64 - my;
            let dz = v.z as f64 - mz;
            sx += dx * dx;
            sy += dy * dy;
            sz += dz * dz;
        }
        let tx = sx * inv_n;
        let ty = sy * inv_n;
        let tz = sz * inv_n;
        let t = (tx + ty + tz) / 3.0;

        // Equal unit mass → fluctuating KE per grain = (1/2) Σaxis = (3/2) T·N·(1/N)…
        // total fluctuating KE (unit mass) = (1/2) Σ_i δv² = (1/2)(sx+sy+sz).
        let fluct_ke = 0.5 * (sx + sy + sz);

        Some(Self {
            grain_count: n,
            mean_velocity: Vec3::new(mx as f32, my as f32, mz as f32),
            component_temperatures: Vec3::new(tx as f32, ty as f32, tz as f32),
            granular_temperature: t as f32,
            total_mass: n as f32,
            fluctuation_kinetic_energy: fluct_ke as f32,
        })
    }

    /// Mass-weighted granular temperature.
    ///
    /// Uses the mass-weighted mean velocity `⟨v⟩ = Σ m_i v_i / Σ m_i` and the
    /// mass-weighted mean-square fluctuation `T = Σ m_i δv_i² / (3 Σ m_i)`.
    /// Returns `None` if lengths disagree, inputs are empty/non-finite, or any
    /// mass is non-positive.
    pub fn measure_weighted(velocities: &[Vec3], masses: &[f32]) -> Option<Self> {
        if velocities.is_empty() || velocities.len() != masses.len() {
            return None;
        }
        let mut total_mass = 0.0_f64;
        for (v, &m) in velocities.iter().zip(masses.iter()) {
            if !v.is_finite() || !m.is_finite() || m <= 0.0 {
                return None;
            }
            total_mass += m as f64;
        }
        if total_mass <= 0.0 {
            return None;
        }
        let inv_m = 1.0 / total_mass;

        let (mut mx, mut my, mut mz) = (0.0_f64, 0.0_f64, 0.0_f64);
        for (v, &m) in velocities.iter().zip(masses.iter()) {
            let w = m as f64;
            mx += w * v.x as f64;
            my += w * v.y as f64;
            mz += w * v.z as f64;
        }
        mx *= inv_m;
        my *= inv_m;
        mz *= inv_m;

        let (mut sx, mut sy, mut sz) = (0.0_f64, 0.0_f64, 0.0_f64);
        for (v, &m) in velocities.iter().zip(masses.iter()) {
            let w = m as f64;
            let dx = v.x as f64 - mx;
            let dy = v.y as f64 - my;
            let dz = v.z as f64 - mz;
            sx += w * dx * dx;
            sy += w * dy * dy;
            sz += w * dz * dz;
        }
        let tx = sx * inv_m;
        let ty = sy * inv_m;
        let tz = sz * inv_m;
        let t = (tx + ty + tz) / 3.0;
        // Fluctuating kinetic energy = (1/2) Σ m_i δv_i².
        let fluct_ke = 0.5 * (sx + sy + sz);

        Some(Self {
            grain_count: velocities.len(),
            mean_velocity: Vec3::new(mx as f32, my as f32, mz as f32),
            component_temperatures: Vec3::new(tx as f32, ty as f32, tz as f32),
            granular_temperature: t as f32,
            total_mass: total_mass as f32,
            fluctuation_kinetic_energy: fluct_ke as f32,
        })
    }

    /// Number of grains sampled.
    pub fn grain_count(&self) -> usize {
        self.grain_count
    }

    /// (Mass-weighted) mean velocity `⟨v⟩`.
    pub fn mean_velocity(&self) -> Vec3 {
        self.mean_velocity
    }

    /// Mean speed `|⟨v⟩|` of the bulk drift.
    pub fn mean_speed(&self) -> f32 {
        self.mean_velocity.length()
    }

    /// Per-axis granular temperatures `(T_x, T_y, T_z)`.
    pub fn component_temperatures(&self) -> Vec3 {
        self.component_temperatures
    }

    /// Granular temperature `T = (T_x + T_y + T_z)/3`.
    pub fn granular_temperature(&self) -> f32 {
        self.granular_temperature
    }

    /// RMS velocity fluctuation `√(3 T) = √⟨δv²⟩`.
    pub fn fluctuation_rms(&self) -> f32 {
        (3.0 * self.granular_temperature).max(0.0).sqrt()
    }

    /// Total (mass-weighted) fluctuating kinetic energy `(1/2) Σ m_i δv_i²`.
    pub fn fluctuation_kinetic_energy(&self) -> f32 {
        self.fluctuation_kinetic_energy
    }

    /// Total mass used in the weighting (grain count for the equal-mass case).
    pub fn total_mass(&self) -> f32 {
        self.total_mass
    }

    /// Fluctuation anisotropy: `(T_max − T_min)/T`, zero for isotropic
    /// fluctuations. Returns `0` when `T = 0`.
    pub fn anisotropy(&self) -> f32 {
        let t = self.granular_temperature;
        if t <= 0.0 {
            return 0.0;
        }
        let c = self.component_temperatures;
        let tmax = c.x.max(c.y).max(c.z);
        let tmin = c.x.min(c.y).min(c.z);
        (tmax - tmin) / t
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_bad_input() {
        assert!(GranularTemperature::measure(&[]).is_none());
        assert!(GranularTemperature::measure(&[Vec3::new(f32::NAN, 0.0, 0.0)]).is_none());
        let v = [Vec3::X, Vec3::Y];
        assert!(GranularTemperature::measure_weighted(&v, &[1.0]).is_none());
        assert!(GranularTemperature::measure_weighted(&v, &[1.0, 0.0]).is_none());
        assert!(GranularTemperature::measure_weighted(&v, &[1.0, -1.0]).is_none());
    }

    #[test]
    fn uniform_drift_has_zero_temperature() {
        // All grains share the same velocity → no fluctuation.
        let v = vec![Vec3::new(2.0, -1.0, 0.5); 16];
        let t = GranularTemperature::measure(&v).unwrap();
        assert_eq!(t.grain_count(), 16);
        assert!((t.mean_velocity() - Vec3::new(2.0, -1.0, 0.5)).length() < 1e-6);
        assert!(t.granular_temperature() < 1e-6);
        assert!(t.fluctuation_rms() < 1e-5);
        assert!(t.fluctuation_kinetic_energy() < 1e-5);
        assert!(t.anisotropy() < 1e-6);
    }

    #[test]
    fn antiparallel_pair_known_temperature() {
        // v = ±u along X with |u| = 3 → mean 0, ⟨δv²⟩ = 9, T = 3, Tx = 9.
        let u = 3.0;
        let v = [Vec3::new(u, 0.0, 0.0), Vec3::new(-u, 0.0, 0.0)];
        let t = GranularTemperature::measure(&v).unwrap();
        assert!(t.mean_speed() < 1e-6);
        assert!((t.component_temperatures().x - 9.0).abs() < 1e-5);
        assert!(t.component_temperatures().y.abs() < 1e-6);
        assert!((t.granular_temperature() - 3.0).abs() < 1e-5);
        assert!((t.fluctuation_rms() - 3.0).abs() < 1e-5);
        // Fluctuating KE (unit mass) = 1/2 (u² + u²) = u² = 9.
        assert!((t.fluctuation_kinetic_energy() - 9.0).abs() < 1e-4);
    }

    #[test]
    fn isotropic_axes_have_matching_component_temperatures() {
        // Equal-magnitude ± fluctuations on each axis → Tx = Ty = Tz.
        let a = 1.0;
        let v = vec![
            Vec3::new(a, 0.0, 0.0),
            Vec3::new(-a, 0.0, 0.0),
            Vec3::new(0.0, a, 0.0),
            Vec3::new(0.0, -a, 0.0),
            Vec3::new(0.0, 0.0, a),
            Vec3::new(0.0, 0.0, -a),
        ];
        let t = GranularTemperature::measure(&v).unwrap();
        let c = t.component_temperatures();
        assert!((c.x - c.y).abs() < 1e-6 && (c.y - c.z).abs() < 1e-6);
        assert!(t.anisotropy() < 1e-5);
    }

    #[test]
    fn mass_weighting_shifts_mean_toward_heavy_grain() {
        // Heavy grain dominates the mean velocity.
        let v = [Vec3::new(0.0, 0.0, 0.0), Vec3::new(10.0, 0.0, 0.0)];
        let m = [1.0, 9.0];
        let t = GranularTemperature::measure_weighted(&v, &m).unwrap();
        // Mean = (1·0 + 9·10)/10 = 9.
        assert!((t.mean_velocity().x - 9.0).abs() < 1e-5);
        assert!((t.total_mass() - 10.0).abs() < 1e-5);
        // δ = (-9, +1); T_x = (1·81 + 9·1)/10 = 9.
        assert!((t.component_temperatures().x - 9.0).abs() < 1e-4);
        // Fluctuating KE = 1/2 (1·81 + 9·1) = 45.
        assert!((t.fluctuation_kinetic_energy() - 45.0).abs() < 1e-3);
    }

    #[test]
    fn equal_mass_weighted_matches_unweighted() {
        let v = vec![
            Vec3::new(1.0, 2.0, -3.0),
            Vec3::new(-2.0, 0.5, 1.0),
            Vec3::new(0.0, -1.0, 2.0),
        ];
        let m = vec![1.0_f32; 3];
        let a = GranularTemperature::measure(&v).unwrap();
        let b = GranularTemperature::measure_weighted(&v, &m).unwrap();
        assert!((a.granular_temperature() - b.granular_temperature()).abs() < 1e-5);
        assert!((a.mean_velocity() - b.mean_velocity()).length() < 1e-5);
        assert!((a.fluctuation_kinetic_energy() - b.fluctuation_kinetic_energy()).abs() < 1e-4);
    }
}
