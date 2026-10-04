//! Vertical porosity / void-ratio depth profile of a granular packing.
//!
//! The single-number solid fraction from
//! [`PackingDiagnostics`](super::packing_diagnostics::PackingDiagnostics) hides
//! how density varies with depth, yet a settled pile is almost never uniform:
//! grains compact under their own weight, so the bed is denser at the bottom
//! and looser near the free surface, and a poured column often shows a dilated
//! cap. Soil mechanics captures this with a *depth profile* — the box is sliced
//! into horizontal layers and each layer reports its own porosity.
//!
//! For each z-slab this module accumulates the exact grain-material volume
//! inside the slab. A sphere that straddles a slab boundary is split
//! analytically: the volume of the spherical segment between planes `z = a` and
//! `z = b` is
//! `∫ π·(r² − (z − z_c)²) dz = π·[r²·u − u³/3]` evaluated over the clamped
//! `u = z − z_c` range, so a grain's material is distributed across exactly the
//! slabs it occupies and the per-slab volumes sum to the full `4/3·π·r³`.
//!
//! From the slab solid volume `V_s` and slab box volume `V_slab` the module
//! derives:
//!
//! * **solid fraction** `φ = V_s / V_slab`;
//! * **porosity** `n = 1 − φ` (the void share);
//! * **void ratio** `e = n / φ = V_void / V_solid`, the geotechnical standard.
//!
//! The slicing is one-dimensional along `z`, so the profile assumes grains lie
//! within the box's horizontal footprint (it uses the full slab cross-section
//! as the reference volume, like a laboratory core sample). Everything here is
//! a pure, deterministic analysis of the supplied arrays; nothing is derived
//! from Unreal Engine source.

use glam::Vec3;

/// Grain-material volume of the spherical segment of a sphere centred at height
/// `zc` with radius `r` that lies between the planes `z = a` and `z = b`
/// (`a <= b`). Computed in `f64`.
fn segment_volume(zc: f64, r: f64, a: f64, b: f64) -> f64 {
    let lo = a.max(zc - r) - zc;
    let hi = b.min(zc + r) - zc;
    if hi <= lo {
        return 0.0;
    }
    // π·[r²·u − u³/3] from lo to hi.
    let r2 = r * r;
    std::f64::consts::PI * (r2 * (hi - lo) - (hi * hi * hi - lo * lo * lo) / 3.0)
}

/// A vertical porosity profile over an axis-aligned reference box.
///
/// Build one with [`PorosityProfile::analyze`]. Layer `0` is the bottom slab
/// (lowest `z`); layer `layer_count − 1` is the top slab.
#[derive(Clone, Debug, PartialEq)]
pub struct PorosityProfile {
    box_min: Vec3,
    box_max: Vec3,
    layer_solid_volume: Vec<f64>,
}

impl PorosityProfile {
    /// Builds the profile from parallel `positions`/`radii` over the box
    /// `[box_min, box_max]`, sliced into `layers` equal-height z-slabs.
    ///
    /// Returns `None` unless the two arrays share the same length, every value
    /// is finite, every radius is strictly positive, the box is finite with a
    /// strictly positive extent on each axis, and `layers` is non-zero. An
    /// empty packing is accepted and yields zero solid volume (porosity `1`) in
    /// every layer. Grain material outside the box's z-range is clipped away.
    #[must_use]
    pub fn analyze(
        positions: &[Vec3],
        radii: &[f32],
        box_min: Vec3,
        box_max: Vec3,
        layers: usize,
    ) -> Option<Self> {
        if positions.len() != radii.len() {
            return None;
        }
        if layers == 0 {
            return None;
        }
        if !(box_min.is_finite() && box_max.is_finite()) {
            return None;
        }
        let extent = box_max - box_min;
        if extent.x <= 0.0 || extent.y <= 0.0 || extent.z <= 0.0 {
            return None;
        }
        if positions.iter().any(|p| !p.is_finite()) {
            return None;
        }
        if radii.iter().any(|r| !r.is_finite() || *r <= 0.0) {
            return None;
        }

        let z0 = box_min.z as f64;
        let layer_height = extent.z as f64 / layers as f64;
        let mut layer_solid_volume = vec![0.0_f64; layers];

        for (&p, &r) in positions.iter().zip(radii.iter()) {
            let zc = p.z as f64;
            let rd = r as f64;
            // Only the layers the sphere overlaps need touching.
            let top = (zc + rd - z0) / layer_height;
            let bot = (zc - rd - z0) / layer_height;
            let first = (bot.floor() as isize).max(0);
            let last = ((top.ceil() as isize) - 1).min(layers as isize - 1);
            let mut k = first;
            while k <= last {
                let a = z0 + k as f64 * layer_height;
                let b = a + layer_height;
                layer_solid_volume[k as usize] += segment_volume(zc, rd, a, b);
                k += 1;
            }
        }

        Some(Self {
            box_min,
            box_max,
            layer_solid_volume,
        })
    }

    /// Number of z-slabs.
    #[must_use]
    pub fn layer_count(&self) -> usize {
        self.layer_solid_volume.len()
    }

    /// Inclusive-lower, exclusive-upper z bounds `[z_k, z_{k+1})` of layer `k`,
    /// or `None` when `k` is out of range.
    #[must_use]
    pub fn layer_bounds(&self, k: usize) -> Option<(f32, f32)> {
        if k >= self.layer_solid_volume.len() {
            return None;
        }
        let h = (self.box_max.z - self.box_min.z) / self.layer_solid_volume.len() as f32;
        let lo = self.box_min.z + k as f32 * h;
        let hi = self.box_min.z + (k + 1) as f32 * h;
        Some((lo, hi))
    }

    /// Centre height of layer `k`, or `None` when `k` is out of range.
    #[must_use]
    pub fn layer_center(&self, k: usize) -> Option<f32> {
        self.layer_bounds(k).map(|(lo, hi)| 0.5 * (lo + hi))
    }

    /// Box volume of a single slab (identical for every layer).
    #[must_use]
    pub fn layer_volume(&self) -> f32 {
        let extent = self.box_max - self.box_min;
        let area = extent.x as f64 * extent.y as f64;
        (area * extent.z as f64 / self.layer_solid_volume.len() as f64) as f32
    }

    /// Grain-material volume contained in layer `k`, or `None` when out of
    /// range.
    #[must_use]
    pub fn layer_solid_volume(&self, k: usize) -> Option<f32> {
        self.layer_solid_volume.get(k).map(|&v| v as f32)
    }

    /// Solid fraction `φ = V_solid / V_slab` of layer `k`, or `None` when out
    /// of range.
    #[must_use]
    pub fn solid_fraction(&self, k: usize) -> Option<f32> {
        let &solid = self.layer_solid_volume.get(k)?;
        let extent = self.box_max - self.box_min;
        let area = extent.x as f64 * extent.y as f64;
        let slab = area * extent.z as f64 / self.layer_solid_volume.len() as f64;
        if slab <= 0.0 {
            return None;
        }
        Some((solid / slab) as f32)
    }

    /// Porosity `n = 1 − φ` of layer `k`, or `None` when out of range.
    #[must_use]
    pub fn porosity(&self, k: usize) -> Option<f32> {
        self.solid_fraction(k).map(|phi| 1.0 - phi)
    }

    /// Void ratio `e = n / φ = V_void / V_solid` of layer `k`.
    ///
    /// Returns `None` when `k` is out of range or the layer holds no solid
    /// material (the void ratio would be infinite).
    #[must_use]
    pub fn void_ratio(&self, k: usize) -> Option<f32> {
        let phi = self.solid_fraction(k)?;
        if phi <= 0.0 {
            return None;
        }
        Some((1.0 - phi) / phi)
    }

    /// Bulk (dry) density of layer `k` for a given grain `material_density`,
    /// i.e. `φ · material_density`. Returns `None` when `k` is out of range or
    /// `material_density` is non-finite or negative.
    #[must_use]
    pub fn bulk_density(&self, k: usize, material_density: f32) -> Option<f32> {
        if !material_density.is_finite() || material_density < 0.0 {
            return None;
        }
        self.solid_fraction(k).map(|phi| phi * material_density)
    }

    /// Solid fraction of every layer, bottom to top.
    #[must_use]
    pub fn solid_fraction_curve(&self) -> Vec<f32> {
        (0..self.layer_solid_volume.len())
            .map(|k| self.solid_fraction(k).unwrap_or(0.0))
            .collect()
    }

    /// Porosity of every layer, bottom to top.
    #[must_use]
    pub fn porosity_curve(&self) -> Vec<f32> {
        (0..self.layer_solid_volume.len())
            .map(|k| self.porosity(k).unwrap_or(1.0))
            .collect()
    }

    /// Total grain-material volume across all layers (clipped to the box
    /// z-range).
    #[must_use]
    pub fn total_solid_volume(&self) -> f32 {
        self.layer_solid_volume.iter().sum::<f64>() as f32
    }

    /// Overall solid fraction across the whole box.
    #[must_use]
    pub fn overall_solid_fraction(&self) -> f32 {
        let extent = self.box_max - self.box_min;
        let box_volume = extent.x as f64 * extent.y as f64 * extent.z as f64;
        (self.layer_solid_volume.iter().sum::<f64>() / box_volume) as f32
    }

    /// Overall porosity across the whole box.
    #[must_use]
    pub fn overall_porosity(&self) -> f32 {
        1.0 - self.overall_solid_fraction()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FOUR_THIRDS_PI: f64 = 4.0 / 3.0 * std::f64::consts::PI;

    fn sphere_volume(r: f32) -> f32 {
        let rd = r as f64;
        (FOUR_THIRDS_PI * rd * rd * rd) as f32
    }

    #[test]
    fn rejects_invalid_inputs() {
        let p = vec![Vec3::splat(0.5)];
        let r = vec![0.2_f32];
        let lo = Vec3::ZERO;
        let hi = Vec3::splat(1.0);
        // Length mismatch.
        assert!(PorosityProfile::analyze(&p, &[], lo, hi, 4).is_none());
        // Zero layers.
        assert!(PorosityProfile::analyze(&p, &r, lo, hi, 0).is_none());
        // Degenerate box.
        assert!(PorosityProfile::analyze(&p, &r, lo, lo, 4).is_none());
        // Non-finite box.
        assert!(PorosityProfile::analyze(&p, &r, Vec3::splat(f32::NAN), hi, 4).is_none());
        // Non-positive radius.
        assert!(PorosityProfile::analyze(&p, &[0.0], lo, hi, 4).is_none());
    }

    #[test]
    fn empty_packing_is_all_void() {
        let prof = PorosityProfile::analyze(&[], &[], Vec3::ZERO, Vec3::splat(2.0), 4).unwrap();
        assert_eq!(prof.layer_count(), 4);
        assert_eq!(prof.total_solid_volume(), 0.0);
        for k in 0..4 {
            assert_eq!(prof.solid_fraction(k).unwrap(), 0.0);
            assert_eq!(prof.porosity(k).unwrap(), 1.0);
            assert!(prof.void_ratio(k).is_none());
        }
        assert_eq!(prof.overall_porosity(), 1.0);
    }

    #[test]
    fn single_sphere_lands_in_one_layer_and_conserves_volume() {
        // Sphere fully inside the bottom layer [0,1).
        let p = vec![Vec3::new(0.5, 0.5, 0.5)];
        let r = vec![0.3_f32];
        let prof =
            PorosityProfile::analyze(&p, &r, Vec3::ZERO, Vec3::new(1.0, 1.0, 3.0), 3).unwrap();
        let expect = sphere_volume(0.3);
        assert!((prof.layer_solid_volume(0).unwrap() - expect).abs() < 1e-5);
        assert!(prof.layer_solid_volume(1).unwrap() < 1e-6);
        assert!(prof.layer_solid_volume(2).unwrap() < 1e-6);
        assert!((prof.total_solid_volume() - expect).abs() < 1e-5);
    }

    #[test]
    fn sphere_on_a_boundary_splits_evenly() {
        // Sphere r=1 centred on the plane z=1 between layers [0,1) and [1,2).
        let p = vec![Vec3::new(0.5, 0.5, 1.0)];
        let r = vec![1.0_f32];
        let prof =
            PorosityProfile::analyze(&p, &r, Vec3::ZERO, Vec3::new(1.0, 1.0, 2.0), 2).unwrap();
        let half = sphere_volume(1.0) / 2.0;
        assert!((prof.layer_solid_volume(0).unwrap() - half).abs() < 1e-4);
        assert!((prof.layer_solid_volume(1).unwrap() - half).abs() < 1e-4);
        assert!((prof.total_solid_volume() - sphere_volume(1.0)).abs() < 1e-4);
    }

    #[test]
    fn void_ratio_matches_porosity_relation() {
        let p = vec![Vec3::new(0.5, 0.5, 0.5)];
        let r = vec![0.4_f32];
        let prof = PorosityProfile::analyze(&p, &r, Vec3::ZERO, Vec3::splat(1.0), 1).unwrap();
        let phi = prof.solid_fraction(0).unwrap();
        let e = prof.void_ratio(0).unwrap();
        assert!((e - (1.0 - phi) / phi).abs() < 1e-5);
    }

    #[test]
    fn layer_geometry_and_bulk_density() {
        let prof =
            PorosityProfile::analyze(&[], &[], Vec3::ZERO, Vec3::new(1.0, 1.0, 4.0), 4).unwrap();
        let (lo, hi) = prof.layer_bounds(2).unwrap();
        assert!((lo - 2.0).abs() < 1e-6 && (hi - 3.0).abs() < 1e-6);
        assert!((prof.layer_center(2).unwrap() - 2.5).abs() < 1e-6);
        assert!((prof.layer_volume() - 1.0).abs() < 1e-6);
        assert!(prof.layer_bounds(4).is_none());
        // Empty layer → bulk density 0 regardless of material density.
        assert_eq!(prof.bulk_density(0, 2650.0).unwrap(), 0.0);
        assert!(prof.bulk_density(0, -1.0).is_none());
    }

    #[test]
    fn overall_fraction_sums_layers() {
        // Two spheres at different depths.
        let p = vec![Vec3::new(0.5, 0.5, 0.5), Vec3::new(0.5, 0.5, 2.5)];
        let r = vec![0.3_f32, 0.3_f32];
        let box_max = Vec3::new(1.0, 1.0, 3.0);
        let prof = PorosityProfile::analyze(&p, &r, Vec3::ZERO, box_max, 3).unwrap();
        let total = sphere_volume(0.3) * 2.0;
        let box_vol = 1.0 * 1.0 * 3.0;
        assert!((prof.overall_solid_fraction() - total / box_vol).abs() < 1e-5);
        assert!((prof.total_solid_volume() - total).abs() < 1e-5);
    }
}
