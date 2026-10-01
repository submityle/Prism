//! DDGI probe irradiance — octahedral radiance field with cosine accumulation,
//! temporal hysteresis blending, and octahedral gutter-border replication.
//!
//! This is the backend-neutral CPU golden reference for the *irradiance* half
//! of Majercik et al. 2019 ("Dynamic Diffuse Global Illumination with
//! Ray-Traced Irradiance Fields").  Each probe stores a small octahedral map
//! of cosine-weighted incoming radiance (its diffuse irradiance field); a
//! shading point reconstructs irradiance by bilinearly sampling the map along
//! the surface normal.
//!
//! * [`IrradianceOct`] owns the padded octahedral atlas: an `interior x
//!   interior` grid of radiance texels surrounded by a one-texel *gutter*
//!   border so hardware-style bilinear sampling stays seamless across the
//!   octahedron's folds.
//! * [`cosine_weighted_irradiance`] folds a batch of traced rays into a single
//!   texel's irradiance using the clamped-cosine (`max(0, dot)`) kernel.
//! * [`IrradianceOct::update`] re-integrates every interior texel from a fresh
//!   ray batch and blends it into the stored field with the RTXGI *hysteresis*
//!   exponential moving average, then refreshes the gutter via
//!   [`IrradianceOct::copy_border`].
//! * [`IrradianceOct::sample`] reconstructs irradiance for an arbitrary normal
//!   with bilinear filtering across the padded atlas.
//!
//! # Conventions
//! * Directions are right-handed unit `(x, y, z)` vectors and share the sibling
//!   [`crate::gi::world_space::octahedral`] mapping, so a direction, its stored
//!   texel, and the GPU texture twin all agree.
//! * Radiance / irradiance are linear RGB `Vec3` values and are kept
//!   non-negative; the only transcendental in use is the clamped cosine, so
//!   results are exact `f32` and match the GPU twin bit-for-bit.
//! * The atlas is laid out row-major over the *padded* side length
//!   `interior + 2`.  Interior texels occupy indices `1..=interior`; index `0`
//!   and `interior + 1` are gutter texels filled only by [`copy_border`].
//! * Every function is deterministic: no RNG, no I/O, no GPU, no `unsafe`.  The
//!   only allocation is the texel buffer owned by [`IrradianceOct`].
//!
//! [`copy_border`]: IrradianceOct::copy_border

use alloc::vec::Vec;
use bevy_math::{Vec2, Vec3};

use crate::gi::world_space::octahedral::{dir_to_oct, oct_to_dir};

/// Smallest denominator accepted before a cosine-weighted average falls back to
/// zero, guarding against a texel whose rays all lie in its back hemisphere.
const WEIGHT_EPSILON: f32 = 1.0e-9;

/// Integrates a batch of traced rays into a single texel's irradiance.
///
/// `texel_dir` is the hemisphere axis the texel represents (its decoded
/// octahedral direction).  Each ray contributes its `radiance` weighted by the
/// clamped cosine `max(0, dot(texel_dir, ray_dir))`, and the accumulated
/// radiance is divided by the summed weight — the cosine-weighted mean radiance
/// arriving about `texel_dir`.
///
/// Rays are normalised defensively; zero-length ray directions and non-finite
/// radiance are skipped.  When no ray lands in the texel's hemisphere (weight
/// sum below [`WEIGHT_EPSILON`]) the result is `Vec3::ZERO`.  A constant
/// radiance field therefore reconstructs to that same constant irradiance,
/// which the tests assert.
#[inline]
pub fn cosine_weighted_irradiance(texel_dir: Vec3, rays: &[(Vec3, Vec3)]) -> Vec3 {
    let axis_len_sq = texel_dir.length_squared();
    if axis_len_sq <= f32::MIN_POSITIVE {
        return Vec3::ZERO;
    }
    let axis = texel_dir * axis_len_sq.sqrt().recip();
    let mut sum = Vec3::ZERO;
    let mut weight_sum = 0.0f32;
    for &(dir, radiance) in rays {
        let len_sq = dir.length_squared();
        if len_sq <= f32::MIN_POSITIVE || !radiance.is_finite() {
            continue;
        }
        let d = dir * len_sq.sqrt().recip();
        let w = axis.dot(d).max(0.0);
        if w <= 0.0 {
            continue;
        }
        sum += radiance * w;
        weight_sum += w;
    }
    if weight_sum <= WEIGHT_EPSILON {
        return Vec3::ZERO;
    }
    let out = sum * weight_sum.recip();
    Vec3::new(out.x.max(0.0), out.y.max(0.0), out.z.max(0.0))
}

/// A single probe's octahedral irradiance field with a one-texel gutter border.
///
/// The atlas is a square `padded x padded` grid (`padded = interior + 2`) of
/// linear-RGB irradiance texels in row-major order.  The interior texels carry
/// the integrated field; the border is a replication gutter kept in sync by
/// [`copy_border`](Self::copy_border) so bilinear sampling never bleeds across
/// an octahedral seam into an unrelated hemisphere.
#[derive(Clone, Debug, PartialEq)]
pub struct IrradianceOct {
    /// Interior side length (texels per octahedral axis), always `>= 1`.
    interior: usize,
    /// Padded side length `interior + 2`.
    padded: usize,
    /// Row-major `padded * padded` RGB irradiance texels.
    texels: Vec<Vec3>,
}

impl IrradianceOct {
    /// Creates a zero-initialised field with the given interior resolution.
    ///
    /// `interior` is clamped to at least `1`.  All texels (interior and gutter)
    /// start at `Vec3::ZERO`.
    #[inline]
    pub fn new(interior: usize) -> Self {
        Self::filled(interior, Vec3::ZERO)
    }

    /// Creates a field whose every texel is initialised to `value`.
    ///
    /// `interior` is clamped to at least `1`.  Negative components of `value`
    /// are clamped to zero so the field is always a valid non-negative
    /// irradiance.
    #[inline]
    pub fn filled(interior: usize, value: Vec3) -> Self {
        let interior = interior.max(1);
        let padded = interior + 2;
        let v = Vec3::new(value.x.max(0.0), value.y.max(0.0), value.z.max(0.0));
        Self {
            interior,
            padded,
            texels: alloc::vec![v; padded * padded],
        }
    }

    /// Interior side length (texels per octahedral axis).
    #[inline]
    pub fn interior(&self) -> usize {
        self.interior
    }

    /// Padded side length `interior + 2`.
    #[inline]
    pub fn padded(&self) -> usize {
        self.padded
    }

    /// Total number of texels (`padded * padded`).
    #[inline]
    pub fn len(&self) -> usize {
        self.texels.len()
    }

    /// Always `false`: the atlas holds at least `3 * 3` texels by construction.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.texels.is_empty()
    }

    /// Row-major index for padded coordinates (caller guarantees `< padded`).
    #[inline]
    fn index(&self, x: usize, y: usize) -> usize {
        y * self.padded + x
    }

    /// Reads a texel at padded coordinates, returning `Vec3::ZERO` out of range.
    #[inline]
    pub fn texel_at(&self, x: usize, y: usize) -> Vec3 {
        if x >= self.padded || y >= self.padded {
            return Vec3::ZERO;
        }
        self.texels[self.index(x, y)]
    }

    /// Decodes the hemisphere axis an interior texel represents.
    ///
    /// `ix`/`iy` are *interior* coordinates in `0..interior`.  The texel centre
    /// sits at `(ix + 0.5) / interior` in octahedral UV space, which is decoded
    /// back to a unit direction through [`oct_to_dir`].
    #[inline]
    pub fn interior_texel_dir(&self, ix: usize, iy: usize) -> Vec3 {
        let n = self.interior as f32;
        let u = (ix.min(self.interior - 1) as f32 + 0.5) / n;
        let v = (iy.min(self.interior - 1) as f32 + 0.5) / n;
        oct_to_dir(Vec2::new(u, v))
    }

    /// Re-integrates every interior texel from `rays` and blends the result into
    /// the stored field with temporal hysteresis, then refreshes the gutter.
    ///
    /// For each interior texel the fresh irradiance is
    /// [`cosine_weighted_irradiance`] of `rays` about that texel's direction.
    /// It is blended as `stored <- lerp(fresh, stored, hysteresis)`:
    /// `hysteresis = 0` overwrites with the new estimate, `hysteresis -> 1`
    /// freezes the field (slow, stable convergence).  `hysteresis` is clamped
    /// to `[0, 1]`.  [`copy_border`](Self::copy_border) runs last so the gutter
    /// always reflects the updated interior.
    #[inline]
    pub fn update(&mut self, rays: &[(Vec3, Vec3)], hysteresis: f32) {
        let hysteresis = hysteresis.clamp(0.0, 1.0);
        for iy in 0..self.interior {
            for ix in 0..self.interior {
                let dir = self.interior_texel_dir(ix, iy);
                let fresh = cosine_weighted_irradiance(dir, rays);
                let idx = self.index(ix + 1, iy + 1);
                let prev = self.texels[idx];
                let blended = fresh + (prev - fresh) * hysteresis;
                self.texels[idx] = Vec3::new(
                    blended.x.max(0.0),
                    blended.y.max(0.0),
                    blended.z.max(0.0),
                );
            }
        }
        self.copy_border();
    }

    /// Replicates interior texels into the one-texel gutter border.
    ///
    /// The gutter makes bilinear taps near the octahedral seam read the
    /// *topologically adjacent* interior texel rather than wrapping onto an
    /// unrelated hemisphere.  With interior indices `1..=interior` inside the
    /// padded grid and `mirror(i) = interior + 1 - i` (so `1 <-> interior`):
    ///
    /// * Top row `(x, 0)` copies `(mirror(x), 1)`; bottom `(x, N+1)` copies
    ///   `(mirror(x), N)`.
    /// * Left column `(0, y)` copies `(1, mirror(y))`; right `(N+1, y)` copies
    ///   `(N, mirror(y))`.
    /// * The four corners copy the diagonally opposite interior corner.
    ///
    /// This is the standard octahedral border-copy rule; it is idempotent, so
    /// calling it twice is a no-op.
    pub fn copy_border(&mut self) {
        let n = self.interior;
        let last = self.padded - 1; // == n + 1
        // Edges (interior run i in 1..=n, mirrored to n + 1 - i).
        for i in 1..=n {
            let m = n + 1 - i;
            // Top / bottom rows mirror along x.
            let (dst, src) = (self.index(i, 0), self.index(m, 1));
            self.texels[dst] = self.texels[src];
            let (dst, src) = (self.index(i, last), self.index(m, n));
            self.texels[dst] = self.texels[src];
            // Left / right columns mirror along y.
            let (dst, src) = (self.index(0, i), self.index(1, m));
            self.texels[dst] = self.texels[src];
            let (dst, src) = (self.index(last, i), self.index(n, m));
            self.texels[dst] = self.texels[src];
        }
        // Corners wrap to the diagonally opposite interior corner.
        let (dst, src) = (self.index(0, 0), self.index(n, n));
        self.texels[dst] = self.texels[src];
        let (dst, src) = (self.index(last, 0), self.index(1, n));
        self.texels[dst] = self.texels[src];
        let (dst, src) = (self.index(0, last), self.index(n, 1));
        self.texels[dst] = self.texels[src];
        let (dst, src) = (self.index(last, last), self.index(1, 1));
        self.texels[dst] = self.texels[src];
    }

    /// Reconstructs irradiance for `normal` with bilinear filtering.
    ///
    /// The normal is encoded to octahedral UV, scaled into the padded atlas
    /// (interior texel centre `i` maps to padded coordinate `i`), and the four
    /// surrounding texels — which may include gutter texels near the seam — are
    /// bilinearly blended.  The result is always finite and non-negative.  A
    /// uniformly filled field samples back to its fill value for every normal.
    #[inline]
    pub fn sample(&self, normal: Vec3) -> Vec3 {
        let oct = dir_to_oct(normal);
        let n = self.interior as f32;
        // Interior texel centre i (padded index) is at oct = (i - 0.5)/n, so the
        // continuous padded coordinate is oct * n + 0.5.
        let fx = oct.x * n + 0.5;
        let fy = oct.y * n + 0.5;
        let x0f = fx.floor();
        let y0f = fy.floor();
        let tx = (fx - x0f).clamp(0.0, 1.0);
        let ty = (fy - y0f).clamp(0.0, 1.0);
        let last = self.padded - 1;
        let x0 = clamp_padded(x0f, last);
        let y0 = clamp_padded(y0f, last);
        let x1 = clamp_padded(x0f + 1.0, last);
        let y1 = clamp_padded(y0f + 1.0, last);

        let c00 = self.texels[self.index(x0, y0)];
        let c10 = self.texels[self.index(x1, y0)];
        let c01 = self.texels[self.index(x0, y1)];
        let c11 = self.texels[self.index(x1, y1)];
        let top = c00 + (c10 - c00) * tx;
        let bottom = c01 + (c11 - c01) * tx;
        let out = top + (bottom - top) * ty;
        Vec3::new(out.x.max(0.0), out.y.max(0.0), out.z.max(0.0))
    }
}

/// Clamps a floored, UV-scaled coordinate into `0..=last` as a padded index.
#[inline]
fn clamp_padded(value: f32, last: usize) -> usize {
    if value <= 0.0 {
        0
    } else {
        let v = value as usize;
        if v > last {
            last
        } else {
            v
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::Vec3;

    #[test]
    fn constant_field_reconstructs_constant_irradiance() {
        // Many rays all carrying the same radiance from every direction: the
        // cosine-weighted mean of a constant is that constant for any texel.
        let mut rays = Vec::new();
        let n = 10;
        for i in 0..n {
            for j in 0..n {
                let u = (i as f32 + 0.5) / n as f32;
                let v = (j as f32 + 0.5) / n as f32;
                let dir = oct_to_dir(Vec2::new(u, v));
                rays.push((dir, Vec3::new(0.4, 0.6, 0.8)));
            }
        }
        let mut field = IrradianceOct::new(6);
        field.update(&rays, 0.0);
        for normal in [Vec3::X, Vec3::Y, Vec3::Z, Vec3::NEG_X, Vec3::new(0.3, -0.5, 0.8)] {
            let e = field.sample(normal);
            assert!((e - Vec3::new(0.4, 0.6, 0.8)).length() < 2e-2, "e {e:?} for {normal:?}");
        }
    }

    #[test]
    fn cosine_weight_prefers_aligned_rays() {
        // A single bright ray along +Z: a +Z texel sees it fully, a -Z texel
        // sees nothing (back hemisphere clamped away).
        let rays = [(Vec3::Z, Vec3::splat(2.0))];
        let front = cosine_weighted_irradiance(Vec3::Z, &rays);
        let back = cosine_weighted_irradiance(Vec3::NEG_Z, &rays);
        assert!((front - Vec3::splat(2.0)).length() < 1e-6, "front {front:?}");
        assert_eq!(back, Vec3::ZERO);
    }

    #[test]
    fn empty_or_backfacing_rays_are_zero() {
        assert_eq!(cosine_weighted_irradiance(Vec3::Z, &[]), Vec3::ZERO);
        // All rays in the back hemisphere -> zero.
        let rays = [(Vec3::NEG_Z, Vec3::splat(5.0))];
        assert_eq!(cosine_weighted_irradiance(Vec3::Z, &rays), Vec3::ZERO);
        // Degenerate zero axis -> zero.
        assert_eq!(cosine_weighted_irradiance(Vec3::ZERO, &rays), Vec3::ZERO);
    }

    #[test]
    fn hysteresis_blends_toward_history() {
        let rays_a = [(Vec3::Z, Vec3::splat(1.0)), (Vec3::NEG_Z, Vec3::splat(1.0)),
                      (Vec3::X, Vec3::splat(1.0)), (Vec3::NEG_X, Vec3::splat(1.0)),
                      (Vec3::Y, Vec3::splat(1.0)), (Vec3::NEG_Y, Vec3::splat(1.0))];
        let mut field = IrradianceOct::new(4);
        field.update(&rays_a, 0.0); // establish history ~ 1.0
        let before = field.sample(Vec3::Z);
        // New brighter batch with high hysteresis moves only a little.
        let rays_b = [(Vec3::Z, Vec3::splat(5.0)), (Vec3::NEG_Z, Vec3::splat(5.0)),
                      (Vec3::X, Vec3::splat(5.0)), (Vec3::NEG_X, Vec3::splat(5.0)),
                      (Vec3::Y, Vec3::splat(5.0)), (Vec3::NEG_Y, Vec3::splat(5.0))];
        field.update(&rays_b, 0.9);
        let after = field.sample(Vec3::Z);
        assert!(after.x > before.x, "{} !> {}", after.x, before.x);
        assert!(after.x < 3.0, "high hysteresis moved too far: {}", after.x);
    }

    #[test]
    fn border_copy_is_idempotent_and_mirrors() {
        let mut field = IrradianceOct::new(4);
        // Paint interior with a gradient so mirroring is observable.
        for iy in 0..4 {
            for ix in 0..4 {
                let idx = field.index(ix + 1, iy + 1);
                field.texels[idx] = Vec3::splat((ix + iy) as f32);
            }
        }
        field.copy_border();
        let snapshot = field.clone();
        field.copy_border();
        assert_eq!(field, snapshot, "copy_border must be idempotent");

        let n = field.interior;
        let last = field.padded - 1;
        // Top-row gutter mirrors first interior row along x.
        for i in 1..=n {
            let m = n + 1 - i;
            assert_eq!(field.texel_at(i, 0), field.texel_at(m, 1));
            assert_eq!(field.texel_at(0, i), field.texel_at(1, m));
        }
        // A corner wraps to the opposite interior corner.
        assert_eq!(field.texel_at(0, 0), field.texel_at(n, n));
        assert_eq!(field.texel_at(last, last), field.texel_at(1, 1));
    }

    #[test]
    fn uniform_fill_samples_back_to_fill_value() {
        let field = IrradianceOct::filled(8, Vec3::new(0.2, 0.5, 0.9));
        for normal in [Vec3::X, Vec3::NEG_Y, Vec3::new(0.5, 0.5, -0.7), Vec3::new(-0.9, 0.1, 0.3)] {
            let e = field.sample(normal);
            assert!((e - Vec3::new(0.2, 0.5, 0.9)).length() < 1e-5, "e {e:?} for {normal:?}");
        }
    }

    #[test]
    fn construction_clamps_and_stays_finite() {
        let tiny = IrradianceOct::new(0);
        assert_eq!(tiny.interior(), 1);
        assert_eq!(tiny.padded(), 3);
        assert_eq!(tiny.len(), 9);
        assert!(!tiny.is_empty());

        // Negative fill is clamped to zero.
        let f = IrradianceOct::filled(2, Vec3::new(-1.0, -2.0, 3.0));
        assert_eq!(f.sample(Vec3::Z), Vec3::new(0.0, 0.0, 3.0));

        // Out-of-range reads are safe.
        assert_eq!(f.texel_at(999, 0), Vec3::ZERO);
    }

    #[test]
    fn results_are_deterministic() {
        let rays = [(Vec3::new(0.1, 0.2, 0.9), Vec3::splat(1.0)),
                    (Vec3::new(-0.7, 0.3, 0.6), Vec3::new(0.5, 0.2, 0.1))];
        let build = || {
            let mut f = IrradianceOct::new(6);
            f.update(&rays, 0.3);
            f
        };
        let a = build();
        let b = build();
        assert_eq!(a, b);
        assert_eq!(a.sample(Vec3::Z), b.sample(Vec3::Z));
    }
}
