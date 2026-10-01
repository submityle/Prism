//! Caustic photons: flux-carrying particles and dielectric interface transport
//! — CPU golden.
//!
//! A caustic is formed when light focused by a specular interface (water, glass)
//! lands on a diffuse receiver.  The first half of the pipeline *traces photons*:
//! light-emitted particles that carry radiant flux and refract/reflect through
//! smooth dielectric boundaries, splitting their energy according to the Fresnel
//! equations.  This module is the backend-neutral reference for that transport.
//!
//! It provides:
//!
//! * [`Photon`] — a particle with a world-space position, a unit travel
//!   direction and an RGB [`Photon::flux`].
//! * Mirror [`reflect`] and Snell [`refract`] directions, with total internal
//!   reflection (TIR) reported as `None`, matching the sign conventions of the
//!   water-surface GI reference.
//! * The exact unpolarised [`fresnel_dielectric`] reflectance that drives the
//!   energy split.
//! * [`split_at_interface`] — a *deterministic* energy split that returns the
//!   reflected photon and (unless TIR) the transmitted photon, with fluxes that
//!   sum exactly to the incident flux.
//! * [`scatter_at_interface`] — a *stochastic* single-photon continuation that
//!   Russian-roulette-selects the reflected or transmitted branch using a
//!   caller-supplied uniform `u`, keeping the carried flux unbiased.
//!
//! # Conventions
//! * A photon's `direction` is its *travel* direction (it points toward the
//!   surface it is about to hit).  Interface `normal` points *out of* the
//!   surface toward the medium the photon is currently in; a back-facing hit is
//!   detected and the normal/indices are flipped automatically.
//! * `n_i` / `n_t` are the indices of refraction on the incident / transmitted
//!   side; `eta = n_i / n_t` is the relative index fed to [`refract`].
//! * `no_std`: math via `bevy_math`; transcendentals via `bevy_math::ops`;
//!   square roots via the `f32::sqrt` method (never `f32::exp`).
//! * Every input is defensively clamped (indices to `>= 1`, cosines to
//!   `[0, 1]`), degenerate directions fall back to a fixed axis, and every
//!   result is finite — no `NaN`, no division by zero.
//! * Every function is a deterministic, allocation-free pure function: no RNG,
//!   no I/O, no GPU, no global state.

use bevy_math::Vec3;

/// Smallest index of refraction accepted; physical dielectrics have `n >= 1`.
const MIN_IOR: f32 = 1.0;

/// Clamps an index of refraction to the physical dielectric range `[1, inf)`.
#[inline]
fn clamp_ior(n: f32) -> f32 {
    if n.is_finite() {
        n.max(MIN_IOR)
    } else {
        MIN_IOR
    }
}

/// Clamps a cosine of an incidence angle to `[0, 1]`.
#[inline]
fn clamp_cos(cos: f32) -> f32 {
    if cos.is_finite() {
        cos.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Normalises `v`, returning `fallback` for a degenerate (near-zero) input.
#[inline]
fn normalize_or(v: Vec3, fallback: Vec3) -> Vec3 {
    let len_sq = v.length_squared();
    if len_sq > f32::MIN_POSITIVE {
        v * len_sq.sqrt().recip()
    } else {
        fallback
    }
}

/// A flux-carrying caustic photon.
///
/// Photons are emitted from the light, propagate in straight lines between
/// interactions, and carry radiant power (`flux`) that is partitioned by the
/// Fresnel equations at every dielectric boundary.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Photon {
    /// World-space position of the photon (updated to the hit point on a
    /// scatter).
    pub position: Vec3,
    /// Unit travel direction: the photon moves *along* this vector.
    pub direction: Vec3,
    /// RGB radiant flux carried by the photon (non-negative per channel).
    pub flux: Vec3,
}

impl Photon {
    /// Builds a photon, normalising `direction` (degenerate input falls back to
    /// `-Y`) and clamping `flux` to be non-negative and finite per channel.
    #[inline]
    pub fn new(position: Vec3, direction: Vec3, flux: Vec3) -> Self {
        Self {
            position: sanitize(position, Vec3::ZERO),
            direction: normalize_or(direction, Vec3::NEG_Y),
            flux: sanitize(flux, Vec3::ZERO).max(Vec3::ZERO),
        }
    }
}

/// Replaces any non-finite component of `v` with the matching component of
/// `fallback`.
#[inline]
fn sanitize(v: Vec3, fallback: Vec3) -> Vec3 {
    Vec3::new(
        if v.x.is_finite() { v.x } else { fallback.x },
        if v.y.is_finite() { v.y } else { fallback.y },
        if v.z.is_finite() { v.z } else { fallback.z },
    )
}

/// Exact unpolarised Fresnel reflectance of a dielectric interface.
///
/// Averages the s- and p-polarised power reflectances for light crossing from
/// medium `n_i` into medium `n_t` at incident cosine `cos_i`.  Beyond the
/// critical angle (total internal reflection, only possible when `n_i > n_t`)
/// the surface reflects everything and the result is `1`.
#[inline]
pub fn fresnel_dielectric(cos_i: f32, n_i: f32, n_t: f32) -> f32 {
    let cos_i = clamp_cos(cos_i);
    let n_i = clamp_ior(n_i);
    let n_t = clamp_ior(n_t);

    let sin_i2 = (1.0 - cos_i * cos_i).max(0.0);
    let eta = n_i / n_t;
    let sin_t2 = eta * eta * sin_i2;
    if sin_t2 >= 1.0 {
        return 1.0; // total internal reflection
    }
    let cos_t = (1.0 - sin_t2).max(0.0).sqrt();

    let r_s = (n_i * cos_i - n_t * cos_t) / (n_i * cos_i + n_t * cos_t);
    let r_p = (n_t * cos_i - n_i * cos_t) / (n_t * cos_i + n_i * cos_t);
    (0.5 * (r_s * r_s + r_p * r_p)).clamp(0.0, 1.0)
}

/// Mirror reflection of a travel direction about a surface normal.
///
/// With `incident` the photon's travel direction (pointing into the surface)
/// and `normal` pointing out toward the incident medium, returns the unit
/// reflected direction `I - 2*(N·I)*N`.  Both inputs are normalised
/// defensively; a degenerate normal falls back to `+Y`.
#[inline]
pub fn reflect(incident: Vec3, normal: Vec3) -> Vec3 {
    let i = normalize_or(incident, Vec3::NEG_Y);
    let n = normalize_or(normal, Vec3::Y);
    let r = i - 2.0 * n.dot(i) * n;
    normalize_or(r, i)
}

/// Snell refraction of a travel direction through a dielectric interface.
///
/// `incident` is the photon's travel direction (into the surface), `normal`
/// points out toward the incident medium, and `eta = n_i / n_t` is the relative
/// index.  Returns the unit refracted direction, or `None` on total internal
/// reflection (when the radicand `k = 1 - eta^2 * (1 - (N·I)^2)` is negative).
#[inline]
pub fn refract(incident: Vec3, normal: Vec3, eta: f32) -> Option<Vec3> {
    let i = normalize_or(incident, Vec3::NEG_Y);
    let n = normalize_or(normal, Vec3::Y);
    let eta = if eta.is_finite() { eta.max(0.0) } else { 1.0 };

    let cos_i = -n.dot(i);
    let k = 1.0 - eta * eta * (1.0 - cos_i * cos_i);
    if k < 0.0 {
        None
    } else {
        let t = eta * i + (eta * cos_i - k.sqrt()) * n;
        Some(normalize_or(t, i))
    }
}

/// Orients an interface so the photon always strikes its front face.
///
/// Returns `(oriented_normal, n_i, n_t, cos_i)` where `oriented_normal` opposes
/// the travel direction (`cos_i >= 0`).  If the raw `(normal, direction)` pair
/// indicates a back-face hit the normal is flipped and the two indices are
/// swapped, so the caller can pass a fixed geometric normal and let the photon's
/// side of the interface decide the medium ordering.
#[inline]
fn orient(direction: Vec3, normal: Vec3, n_i: f32, n_t: f32) -> (Vec3, f32, f32, f32) {
    let d = normalize_or(direction, Vec3::NEG_Y);
    let n = normalize_or(normal, Vec3::Y);
    let n_i = clamp_ior(n_i);
    let n_t = clamp_ior(n_t);
    let cos = -n.dot(d);
    if cos < 0.0 {
        // Photon is leaving the `n_t` medium: flip the normal and swap indices.
        (-n, n_t, n_i, (-cos).clamp(0.0, 1.0))
    } else {
        (n, n_i, n_t, cos.clamp(0.0, 1.0))
    }
}

/// The deterministic energy split of a photon at a dielectric interface.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhotonSplit {
    /// The reflected photon; always present (carries all the flux under TIR).
    pub reflected: Photon,
    /// The transmitted photon, or `None` under total internal reflection.
    pub transmitted: Option<Photon>,
}

impl PhotonSplit {
    /// Total flux carried by both branches; equals the incident flux (energy
    /// conservation), used by the tests and by downstream accounting.
    #[inline]
    pub fn total_flux(&self) -> Vec3 {
        self.reflected.flux + self.transmitted.map_or(Vec3::ZERO, |p| p.flux)
    }
}

/// Splits an incident photon's flux into reflected and transmitted photons.
///
/// The reflected photon carries `flux * R` and the transmitted photon
/// `flux * (1 - R)`, where `R` is the exact Fresnel reflectance
/// ([`fresnel_dielectric`]) at the oriented incidence.  The two fluxes sum
/// exactly to the incident flux, so no energy is created or lost.  Under total
/// internal reflection `R = 1`, the reflected photon carries all the flux and
/// `transmitted` is `None`.
///
/// Both outgoing photons start at `point` (the hit position).
#[inline]
pub fn split_at_interface(
    photon: Photon,
    point: Vec3,
    normal: Vec3,
    n_i: f32,
    n_t: f32,
) -> PhotonSplit {
    let point = sanitize(point, photon.position);
    let (n, n_i, n_t, cos_i) = orient(photon.direction, normal, n_i, n_t);
    let r = fresnel_dielectric(cos_i, n_i, n_t);

    let reflected = Photon::new(point, reflect(photon.direction, n), photon.flux * r);

    let eta = n_i / n_t;
    let transmitted = refract(photon.direction, n, eta)
        .map(|dir| Photon::new(point, dir, photon.flux * (1.0 - r)));

    PhotonSplit {
        reflected,
        transmitted,
    }
}

/// Stochastically continues a photon through a dielectric interface.
///
/// Uses the caller-supplied uniform `u ∈ [0, 1)` to Russian-roulette-select the
/// reflected branch with probability `R` (the Fresnel reflectance) and the
/// transmitted branch otherwise.  The surviving photon's flux is left
/// **unchanged** — the branch's energy `flux * R` (or `flux * (1 - R)`) divided
/// by its selection probability `R` (or `1 - R`) — so a population of photons is
/// an unbiased estimator of the deterministic split.  Under total internal
/// reflection the reflected branch is always taken.
#[inline]
pub fn scatter_at_interface(
    photon: Photon,
    point: Vec3,
    normal: Vec3,
    n_i: f32,
    n_t: f32,
    u: f32,
) -> Photon {
    let point = sanitize(point, photon.position);
    let (n, n_i, n_t, cos_i) = orient(photon.direction, normal, n_i, n_t);
    let r = fresnel_dielectric(cos_i, n_i, n_t);
    let u = if u.is_finite() { u.clamp(0.0, 1.0) } else { 0.0 };

    let eta = n_i / n_t;
    let transmit_dir = refract(photon.direction, n, eta);

    // Reflect when selected, under TIR, or if refraction degenerated.
    let take_reflection = u < r || transmit_dir.is_none();
    if take_reflection {
        Photon::new(point, reflect(photon.direction, n), photon.flux)
    } else {
        // Safe: `transmit_dir` is `Some` on this branch.
        Photon::new(point, transmit_dir.unwrap_or(photon.direction), photon.flux)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const AIR: f32 = 1.0;
    const WATER: f32 = 1.33;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn split_conserves_energy_air_to_water() {
        let incident = normalize_or(Vec3::new(0.3, -1.0, 0.1), Vec3::NEG_Y);
        let p = Photon::new(Vec3::ZERO, incident, Vec3::new(1.0, 2.0, 3.0));
        let s = split_at_interface(p, Vec3::ZERO, Vec3::Y, AIR, WATER);
        // Reflected + transmitted flux == incident flux, per channel.
        let total = s.total_flux();
        assert!((total - p.flux).length() < 1.0e-5, "total={total:?}");
        assert!(s.transmitted.is_some(), "air->water should transmit");
    }

    #[test]
    fn split_matches_fresnel_weight() {
        let p = Photon::new(Vec3::ZERO, Vec3::NEG_Y, Vec3::splat(1.0));
        let s = split_at_interface(p, Vec3::ZERO, Vec3::Y, AIR, WATER);
        // Normal incidence: reflectance ~ 2%.
        let r = fresnel_dielectric(1.0, AIR, WATER);
        assert!(approx(s.reflected.flux.x, r, 1.0e-6), "r={r}");
        let t = s.transmitted.unwrap();
        assert!(approx(t.flux.x, 1.0 - r, 1.0e-6));
    }

    #[test]
    fn refract_obeys_snell() {
        // Air -> water at 45 deg: n_i sin_i = n_t sin_t.
        let incident = normalize_or(Vec3::new(1.0, -1.0, 0.0), Vec3::NEG_Y);
        let t = refract(incident, Vec3::Y, AIR / WATER).unwrap();
        let sin_i = (1.0 - incident.dot(Vec3::NEG_Y).powi(2)).max(0.0).sqrt();
        let cos_t = (-Vec3::Y.dot(t)).abs();
        let sin_t = (1.0 - cos_t * cos_t).max(0.0).sqrt();
        assert!(
            approx(AIR * sin_i, WATER * sin_t, 1.0e-5),
            "snell: {} vs {}",
            AIR * sin_i,
            WATER * sin_t
        );
    }

    #[test]
    fn total_internal_reflection_keeps_all_flux() {
        // Water -> air past the critical angle (~48.75 deg): grazing 70 deg.
        let ang = 70.0_f32.to_radians();
        let incident = normalize_or(
            Vec3::new(ang.sin(), -ang.cos(), 0.0),
            Vec3::NEG_Y,
        );
        let p = Photon::new(Vec3::ZERO, incident, Vec3::new(2.0, 2.0, 2.0));
        let s = split_at_interface(p, Vec3::ZERO, Vec3::Y, WATER, AIR);
        assert!(s.transmitted.is_none(), "expected TIR");
        assert!((s.reflected.flux - p.flux).length() < 1.0e-6);
        // Direct direction query also reports TIR.
        assert!(refract(incident, Vec3::Y, WATER / AIR).is_none());
    }

    #[test]
    fn backface_hit_is_oriented() {
        // Photon travelling +Y with normal +Y: it hits the back face. The split
        // must still conserve energy and transmit (water -> air, normal angle).
        let p = Photon::new(Vec3::ZERO, Vec3::Y, Vec3::splat(1.0));
        let s = split_at_interface(p, Vec3::ZERO, Vec3::Y, AIR, WATER);
        assert!((s.total_flux() - p.flux).length() < 1.0e-6);
    }

    #[test]
    fn scatter_is_unbiased_split_on_average() {
        // Averaging the stochastic branch flux over the [0,1) interval must
        // reproduce the deterministic reflected/transmitted energy.
        let incident = normalize_or(Vec3::new(0.2, -1.0, 0.0), Vec3::NEG_Y);
        let p = Photon::new(Vec3::ZERO, incident, Vec3::splat(1.0));
        let r = {
            let (_, ni, nt, c) = orient(incident, Vec3::Y, AIR, WATER);
            fresnel_dielectric(c, ni, nt)
        };
        let n = 2000u32;
        let mut reflected_energy = 0.0f32;
        let mut transmitted_energy = 0.0f32;
        for i in 0..n {
            let u = (i as f32 + 0.5) / n as f32;
            let out = scatter_at_interface(p, Vec3::ZERO, Vec3::Y, AIR, WATER, u);
            // Reflected direction has +Y component; transmitted has -Y.
            if out.direction.y >= 0.0 {
                reflected_energy += out.flux.x;
            } else {
                transmitted_energy += out.flux.x;
            }
        }
        reflected_energy /= n as f32;
        transmitted_energy /= n as f32;
        assert!(approx(reflected_energy, r, 2.0e-3), "refl={reflected_energy} r={r}");
        assert!(
            approx(transmitted_energy, 1.0 - r, 2.0e-3),
            "trans={transmitted_energy} 1-r={}",
            1.0 - r
        );
    }

    #[test]
    fn scatter_under_tir_always_reflects() {
        let ang = 75.0_f32.to_radians();
        let incident = normalize_or(Vec3::new(ang.sin(), -ang.cos(), 0.0), Vec3::NEG_Y);
        let p = Photon::new(Vec3::ZERO, incident, Vec3::splat(1.0));
        for i in 0..10 {
            let u = i as f32 / 10.0;
            let out = scatter_at_interface(p, Vec3::ZERO, Vec3::Y, WATER, AIR, u);
            assert!(out.direction.y >= 0.0, "u={u} should reflect: {:?}", out.direction);
            assert!((out.flux - p.flux).length() < 1.0e-6);
        }
    }

    #[test]
    fn degenerate_inputs_do_not_nan() {
        let p = Photon::new(Vec3::ZERO, Vec3::ZERO, Vec3::new(f32::NAN, 1.0, -2.0));
        assert!(p.flux.is_finite());
        assert!(p.flux.x >= 0.0 && p.flux.z >= 0.0);
        let s = split_at_interface(p, Vec3::ZERO, Vec3::ZERO, 0.0, f32::NAN);
        assert!(s.reflected.flux.is_finite());
        assert!(s.reflected.direction.is_finite());
        let out = scatter_at_interface(p, Vec3::ZERO, Vec3::ZERO, 1.0, 1.5, f32::NAN);
        assert!(out.flux.is_finite() && out.direction.is_finite());
    }

    #[test]
    fn is_deterministic() {
        let p = Photon::new(Vec3::new(0.1, 0.2, 0.3), Vec3::new(0.0, -1.0, 0.2), Vec3::ONE);
        assert_eq!(
            split_at_interface(p, Vec3::ZERO, Vec3::Y, AIR, WATER),
            split_at_interface(p, Vec3::ZERO, Vec3::Y, AIR, WATER)
        );
        assert_eq!(
            scatter_at_interface(p, Vec3::ZERO, Vec3::Y, AIR, WATER, 0.4),
            scatter_at_interface(p, Vec3::ZERO, Vec3::Y, AIR, WATER, 0.4)
        );
    }
}
