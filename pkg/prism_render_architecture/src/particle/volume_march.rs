//! §20 volumetric renderer (`RendererKind::Volume`) ray-march integration loop
//! (`CPU` reference).
//!
//! [`super::volumetrics`] owns the *evaluation / baking* arithmetic of design
//! §20: the [`super::volumetrics::henyey_greenstein`] and
//! [`super::volumetrics::double_lobe_phase`] phase functions, the six-way
//! luminance accumulator, the `deep opacity map` transmittance sampler, and the
//! `froxel` density field. What it deliberately does *not* contain is the
//! actual **ray-march integration loop** that walks a view ray through the
//! volume, front-to-back composites step opacity, accumulates transmittance,
//! and early-terminates. This module fills exactly that gap and nothing else:
//! it is the *stepping / compositing* half, orthogonal to the *phase / lighting
//! / density* half in [`super::volumetrics`]. The two compose — a caller marches
//! with [`march_scattered`] and passes a `scatter` closure built from the
//! volumetrics phase and six-way rig — but neither reimplements the other. This
//! module never redefines a phase function.
//!
//! **No transcendental math — algebraic step opacity replaces `exp`.** The
//! textbook `Beer-Lambert` transmittance `exp(-τ)` is forbidden by the
//! workspace determinism contract (zero transcendental functions, so the `CPU`
//! reference stays bit-reproducible against a future `GPU` kernel). Instead this
//! module composites transmittance *algebraically and incrementally*, matching
//! how [`super::volumetrics`] already stores pre-integrated transmittance rather
//! than recomputing `exp`. Per step of length `ds`:
//!
//! - optical thickness `tau = sigma * ds` (extinction × step length),
//! - step opacity `alpha = tau.clamp(0.0, 1.0)` — a first-order absorber (the
//!   energy-conserving variant `tau / (1.0 + tau)` is also `exp`-free and could
//!   be substituted; it only needs the shared [`EPS`] floor on its denominator),
//! - transmittance product `transmittance *= 1.0 - alpha`,
//! - front-to-back radiance `radiance += transmittance * alpha * scatter`.
//!
//! Over a uniform medium the product telescopes to `1 - T` accumulated radiance,
//! the discrete analogue of the analytic `1 - exp(-τ)` single-scatter integral,
//! so the loop converges to the physical answer as `step_size → 0` while using
//! only `+ − × ÷` and `sqrt`.
//!
//! The only non-arithmetic operations used are `sqrt` (through [`Vec3`]),
//! `f32::floor`/`ceil`/`abs`, and integer hashing (through
//! [`super::noise::hash_lattice`] for jittered start offsets); every `f32`
//! division guards its denominator with [`EPS`], and no `f32` equality is tested
//! with bare `==`/`!=`.

use super::noise::hash_lattice;
use super::sort_cull::Aabb;
use super::Vec3;

/// General-purpose magnitude floor for guarding `f32` divisions, near-zero ray
/// directions (the parallel-slab case), and degenerate step / span spans.
///
/// Distinct from [`super::EPS_LEN_SQ`] (a squared-length threshold): this is a
/// plain magnitude tolerance, matching [`super::volumetrics::EPS`].
pub const EPS: f32 = 1e-6;

/// A world-space view ray marched through a volume.
///
/// `direction` is treated as a unit vector by the integrator; [`Ray::new`]
/// normalizes it (falling back to [`Vec3::ZERO`] for a degenerate input) so that
/// a ray parameter `t` measures world distance directly and step lengths equal
/// arc lengths.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ray {
    /// Ray origin in world space.
    pub origin: Vec3,
    /// Unit ray direction (normalized by [`Ray::new`]).
    pub direction: Vec3,
}

impl Ray {
    /// Builds a ray, normalizing `direction` to unit length.
    ///
    /// A (numerically) zero direction is stored as [`Vec3::ZERO`]; the marchers
    /// detect that and report a miss rather than dividing by zero.
    #[must_use]
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Self {
            origin,
            direction: direction.normalize_or_zero(),
        }
    }

    /// The world-space point at ray parameter `t`: `origin + direction · t`.
    #[must_use]
    pub fn at(self, t: f32) -> Vec3 {
        self.origin.add(self.direction.scale(t))
    }
}

/// Ray / axis-aligned-box intersection by the slab method (design §20).
///
/// Returns `(t_enter, t_exit)` — the ray parameters where the ray crosses the
/// near and far faces of `aabb` — or `None` when the ray misses the box or the
/// whole box lies behind the origin (`t_exit < 0`). `t_enter` may be negative
/// when the origin is *inside* the box; callers clamp it to `0` before marching.
///
/// For a ray component that is (near) parallel to a slab (`|direction| < `
/// [`EPS`]) the box is missed on that axis unless the origin already lies within
/// the slab, tested with a range `contains` check rather than a bare comparison.
/// Scalar `f32` division by a non-parallel component is safe because that
/// component's magnitude is at least [`EPS`].
#[must_use]
pub fn ray_aabb_slab(ray: Ray, aabb: Aabb) -> Option<(f32, f32)> {
    let (te, tx) = slab_clip(
        ray.origin.x,
        ray.direction.x,
        aabb.min.x,
        aabb.max.x,
        f32::NEG_INFINITY,
        f32::INFINITY,
    )?;
    let (te, tx) = slab_clip(
        ray.origin.y,
        ray.direction.y,
        aabb.min.y,
        aabb.max.y,
        te,
        tx,
    )?;
    let (te, tx) = slab_clip(
        ray.origin.z,
        ray.direction.z,
        aabb.min.z,
        aabb.max.z,
        te,
        tx,
    )?;
    if tx < te || tx < 0.0 {
        return None;
    }
    Some((te, tx))
}

/// Clips the running `[t_enter, t_exit]` interval against one axis slab.
///
/// Returns the tightened interval, or `None` when the ray leaves the slab (a
/// parallel ray whose origin is outside the slab, or a crossing that inverts the
/// interval).
#[must_use]
fn slab_clip(
    origin: f32,
    dir: f32,
    lo: f32,
    hi: f32,
    t_enter: f32,
    t_exit: f32,
) -> Option<(f32, f32)> {
    if dir.abs() < EPS {
        // Parallel to this slab: a hit requires the origin to lie within it.
        if !(lo..=hi).contains(&origin) {
            return None;
        }
        return Some((t_enter, t_exit));
    }
    let inv = 1.0 / dir;
    let a = (lo - origin) * inv;
    let b = (hi - origin) * inv;
    let (t0, t1) = if a <= b { (a, b) } else { (b, a) };
    let ne = t_enter.max(t0);
    let ex = t_exit.min(t1);
    if ex < ne {
        None
    } else {
        Some((ne, ex))
    }
}

/// Tunables for the ray-march integration loop (design §20).
///
/// `step_size` is the nominal world-space march step; the final partial step is
/// shortened to land exactly on `t_exit`. `max_steps` caps the iteration count
/// regardless of span. `density_scale` and `extinction` turn the raw sampled
/// density into an extinction coefficient `sigma`. `transmittance_cutoff` is the
/// early-out threshold: once transmittance drops below it the remaining medium
/// is effectively opaque and the loop stops.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MarchParams {
    /// Nominal world-space step length; floored at [`EPS`] internally.
    pub step_size: f32,
    /// Hard cap on the number of steps taken.
    pub max_steps: u32,
    /// Multiplier applied to the sampled density before extinction.
    pub density_scale: f32,
    /// Extinction coefficient scaling `density → sigma`.
    pub extinction: f32,
    /// Transmittance below which the march early-terminates.
    pub transmittance_cutoff: f32,
}

impl MarchParams {
    /// Builds march parameters.
    #[must_use]
    pub fn new(
        step_size: f32,
        max_steps: u32,
        density_scale: f32,
        extinction: f32,
        transmittance_cutoff: f32,
    ) -> Self {
        Self {
            step_size,
            max_steps,
            density_scale,
            extinction,
            transmittance_cutoff,
        }
    }
}

/// The result of a density-only march ([`march`]).
///
/// `transmittance` is the surviving fraction of light after crossing the volume
/// (`1.0` for empty space, approaching `0.0` for thick media). `optical_depth`
/// is the accumulated (unclamped) optical thickness `Σ sigma·ds`. `steps_taken`
/// is the number of composited steps, and `hit` is `true` when the ray actually
/// entered the volume bounds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MarchResult {
    /// Surviving light fraction in `0..=1`.
    pub transmittance: f32,
    /// Accumulated optical thickness `Σ sigma·ds`.
    pub optical_depth: f32,
    /// Number of composited steps.
    pub steps_taken: u32,
    /// Whether the ray entered the volume bounds.
    pub hit: bool,
}

/// The result of a scattered (radiance-accumulating) march ([`march_scattered`]).
///
/// `radiance` is the front-to-back accumulated in-scattered luminance,
/// `transmittance` the surviving light fraction, and `steps_taken` the number of
/// composited steps.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScatterResult {
    /// Front-to-back accumulated in-scattered luminance.
    pub radiance: f32,
    /// Surviving light fraction in `0..=1`.
    pub transmittance: f32,
    /// Number of composited steps.
    pub steps_taken: u32,
}

/// Marches a ray through `bounds`, integrating transmittance from a density
/// field (design §20).
///
/// `sample_density` maps a world point to a (non-negative) density; a caller
/// typically closes over the [`super::volumetrics`] `froxel` density field. The
/// loop enters at `max(t_enter, 0)` (so an origin *inside* the box marches from
/// the origin), steps by `params.step_size` — the final step shortened to
/// `t_exit`, the whole march capped at `params.max_steps` — and at each step
/// midpoint forms `sigma = sample_density(p) · density_scale · extinction`, then
/// composites the algebraic step opacity `alpha = (sigma·ds).clamp(0, 1)` into a
/// falling `transmittance *= 1 - alpha`. It early-terminates once transmittance
/// drops below `params.transmittance_cutoff`.
///
/// A missed box (or degenerate zero-direction ray) returns a full-transmittance,
/// `hit == false` result.
#[must_use]
pub fn march<F>(ray: Ray, bounds: Aabb, params: MarchParams, sample_density: F) -> MarchResult
where
    F: Fn(Vec3) -> f32,
{
    let miss = MarchResult {
        transmittance: 1.0,
        optical_depth: 0.0,
        steps_taken: 0,
        hit: false,
    };
    if ray.direction.length_squared() < EPS {
        return miss;
    }
    let Some((t0, t1)) = ray_aabb_slab(ray, bounds) else {
        return miss;
    };
    let t_enter = t0.max(0.0);
    let t_exit = t1;
    if t_exit - t_enter <= EPS {
        return miss;
    }

    let step = params.step_size.max(EPS);
    let cutoff = params.transmittance_cutoff.clamp(0.0, 1.0);
    let mut transmittance = 1.0f32;
    let mut optical_depth = 0.0f32;
    let mut steps_taken = 0u32;
    let mut t = t_enter;
    while t_exit - t > EPS && steps_taken < params.max_steps {
        let ds = step.min(t_exit - t);
        let p = ray.at(t + ds * 0.5);
        let density = sample_density(p).max(0.0);
        let sigma = density * params.density_scale * params.extinction;
        let tau = (sigma * ds).max(0.0);
        optical_depth += tau;
        let alpha = tau.clamp(0.0, 1.0);
        transmittance *= 1.0 - alpha;
        steps_taken += 1;
        if transmittance < cutoff {
            break;
        }
        t += ds;
    }

    MarchResult {
        transmittance: transmittance.clamp(0.0, 1.0),
        optical_depth,
        steps_taken,
        hit: true,
    }
}

/// Marches a ray through `bounds`, accumulating in-scattered radiance
/// front-to-back (design §20).
///
/// Extends [`march`] with a per-step `sample_scatter` closure that returns the
/// in-scattered luminance at a world point — the product of the phase function
/// and the light budget the caller evaluates with the [`super::volumetrics`]
/// phase / six-way rig (this module stays agnostic to how that number is
/// formed). The classic front-to-back compositing accumulates
/// `radiance += transmittance · alpha · scatter` *before* attenuating
/// `transmittance *= 1 - alpha`, so nearer steps contribute at full weight and
/// farther steps are dimmed by the medium already crossed.
#[must_use]
pub fn march_scattered<FD, FL>(
    ray: Ray,
    bounds: Aabb,
    params: MarchParams,
    sample_density: FD,
    sample_scatter: FL,
) -> ScatterResult
where
    FD: Fn(Vec3) -> f32,
    FL: Fn(Vec3) -> f32,
{
    let miss = ScatterResult {
        radiance: 0.0,
        transmittance: 1.0,
        steps_taken: 0,
    };
    if ray.direction.length_squared() < EPS {
        return miss;
    }
    let Some((t0, t1)) = ray_aabb_slab(ray, bounds) else {
        return miss;
    };
    let t_enter = t0.max(0.0);
    let t_exit = t1;
    if t_exit - t_enter <= EPS {
        return miss;
    }

    let step = params.step_size.max(EPS);
    let cutoff = params.transmittance_cutoff.clamp(0.0, 1.0);
    let mut radiance = 0.0f32;
    let mut transmittance = 1.0f32;
    let mut steps_taken = 0u32;
    let mut t = t_enter;
    while t_exit - t > EPS && steps_taken < params.max_steps {
        let ds = step.min(t_exit - t);
        let p = ray.at(t + ds * 0.5);
        let density = sample_density(p).max(0.0);
        let sigma = density * params.density_scale * params.extinction;
        let tau = (sigma * ds).max(0.0);
        let alpha = tau.clamp(0.0, 1.0);
        let scatter = sample_scatter(p);
        radiance += transmittance * alpha * scatter;
        transmittance *= 1.0 - alpha;
        steps_taken += 1;
        if transmittance < cutoff {
            break;
        }
        t += ds;
    }

    ScatterResult {
        radiance,
        transmittance: transmittance.clamp(0.0, 1.0),
        steps_taken,
    }
}

/// Scale that maps a 24-bit hash mantissa into `0..1` without a runtime divide.
///
/// Equal to `1.0 / 2^24`; multiplying keeps the mapping `exp`-free and avoids a
/// denominator entirely (the only reason [`EPS`] is not needed here).
const HASH_UNIT_SCALE: f32 = 1.0 / 16_777_216.0;

/// A per-ray start-offset jitter in `[0, step_size)` that breaks up the banding
/// artefacts of a fixed step phase (design §20).
///
/// Uses the shared integer lattice hash [`super::noise::hash_lattice`] keyed on
/// `ray_index` and `seed` so the `CPU` reference and a future `GPU` kernel agree
/// bit for bit, then maps the top 24 hash bits into `[0, 1)` and scales by
/// `step_size`. A non-positive `step_size` yields `0.0`; no transcendental math
/// or `f32` equality test is involved.
#[must_use]
pub fn jittered_start_offset(ray_index: u32, seed: u32, step_size: f32) -> f32 {
    let span = step_size.max(0.0);
    let h = hash_lattice(ray_index as i32, seed as i32, 0, seed);
    let unit = (h >> 8) as f32 * HASH_UNIT_SCALE;
    unit * span
}

#[cfg(test)]
mod tests {
    use super::super::volumetrics::henyey_greenstein;
    use super::*;

    const TOL: f32 = 1e-5;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= TOL
    }

    fn unit_box() -> Aabb {
        Aabb {
            min: Vec3::splat(-1.0),
            max: Vec3::splat(1.0),
        }
    }

    #[test]
    fn ray_box_hit_from_outside() {
        let ray = Ray::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let (te, tx) = ray_aabb_slab(ray, unit_box()).expect("ray should hit box");
        assert!(approx(te, 4.0));
        assert!(approx(tx, 6.0));
    }

    #[test]
    fn ray_box_miss_to_the_side() {
        let ray = Ray::new(Vec3::new(-5.0, 5.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        assert!(ray_aabb_slab(ray, unit_box()).is_none());
    }

    #[test]
    fn ray_box_parallel_inside_hits_and_outside_misses() {
        // Parallel to X, threaded through the box on Y/Z: hit.
        let inside = Ray::new(Vec3::new(-5.0, 0.5, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let (te, tx) = ray_aabb_slab(inside, unit_box()).expect("parallel-inside should hit");
        assert!(approx(te, 4.0));
        assert!(approx(tx, 6.0));

        // Parallel to X but offset outside the Y slab: miss.
        let outside = Ray::new(Vec3::new(-5.0, 9.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        assert!(ray_aabb_slab(outside, unit_box()).is_none());
    }

    #[test]
    fn ray_box_interior_origin_has_negative_enter() {
        let ray = Ray::new(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0));
        let (te, tx) = ray_aabb_slab(ray, unit_box()).expect("interior origin should hit");
        assert!(te < 0.0);
        assert!(approx(te, -1.0));
        assert!(approx(tx, 1.0));
    }

    #[test]
    fn ray_box_behind_returns_none() {
        // Box entirely behind the origin along the ray direction.
        let ray = Ray::new(Vec3::new(5.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        assert!(ray_aabb_slab(ray, unit_box()).is_none());
    }

    #[test]
    fn march_empty_box_is_a_miss() {
        let ray = Ray::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let params = MarchParams::new(0.1, 256, 1.0, 1.0, 0.001);
        let out = march(ray, Aabb::empty(), params, |_| 1.0);
        assert!(!out.hit);
        assert_eq!(out.steps_taken, 0);
        assert!(approx(out.transmittance, 1.0));
        assert!(approx(out.optical_depth, 0.0));
    }

    #[test]
    fn march_uniform_density_transmittance_falls_monotonically() {
        let ray = Ray::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        // No cutoff so every step runs; step count == ceil(span/step).
        let params = MarchParams::new(0.25, 64, 0.5, 1.0, 0.0);
        let out = march(ray, unit_box(), params, |_| 1.0);
        assert!(out.hit);
        // Span is 2.0 over step 0.25 => 8 steps.
        assert_eq!(out.steps_taken, 8);
        // Transmittance strictly below 1 and optical depth = sigma * span.
        assert!(out.transmittance < 1.0);
        assert!(out.transmittance > 0.0);
        assert!(approx(out.optical_depth, 0.5 * 2.0));

        // Doubling density lowers surviving transmittance (monotone in density).
        let denser = MarchParams::new(0.25, 64, 1.0, 1.0, 0.0);
        let out2 = march(ray, unit_box(), denser, |_| 1.0);
        assert!(out2.transmittance < out.transmittance);
    }

    #[test]
    fn march_early_terminates_on_cutoff() {
        let ray = Ray::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        // Very thick medium with a high cutoff: opacity saturates in a few steps.
        let params = MarchParams::new(0.1, 1024, 50.0, 1.0, 0.5);
        let out = march(ray, unit_box(), params, |_| 1.0);
        assert!(out.hit);
        // Would be 20 full steps without a cutoff; the early-out stops far sooner.
        assert!(out.steps_taken < 20);
        assert!(out.transmittance < params.transmittance_cutoff);
    }

    #[test]
    fn march_zero_direction_ray_is_a_miss() {
        let ray = Ray::new(Vec3::ZERO, Vec3::ZERO);
        let params = MarchParams::new(0.1, 32, 1.0, 1.0, 0.001);
        let out = march(ray, unit_box(), params, |_| 1.0);
        assert!(!out.hit);
        assert_eq!(out.steps_taken, 0);
    }

    #[test]
    fn scatter_radiance_grows_with_density() {
        let ray = Ray::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        // Build a constant scatter closure from the volumetrics phase function to
        // show orthogonality: this module marches, volumetrics supplies the phase.
        let phase = henyey_greenstein(0.3, 0.8);
        let scatter = |_p: Vec3| phase;

        let thin = MarchParams::new(0.2, 64, 0.2, 1.0, 0.0);
        let thick = MarchParams::new(0.2, 64, 1.0, 1.0, 0.0);
        let a = march_scattered(ray, unit_box(), thin, |_| 1.0, scatter);
        let b = march_scattered(ray, unit_box(), thick, |_| 1.0, scatter);

        assert!(a.radiance > 0.0);
        assert!(b.radiance > a.radiance);
        // Radiance ~ scatter * (1 - transmittance) for a constant medium.
        assert!(approx(b.radiance, phase * (1.0 - b.transmittance)));
        assert!(b.transmittance < a.transmittance);
    }

    #[test]
    fn scatter_empty_medium_leaves_full_transmittance() {
        let ray = Ray::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let params = MarchParams::new(0.2, 64, 1.0, 1.0, 0.0);
        let out = march_scattered(ray, unit_box(), params, |_| 0.0, |_| 1.0);
        assert!(approx(out.radiance, 0.0));
        assert!(approx(out.transmittance, 1.0));
        assert!(out.steps_taken > 0);
    }

    #[test]
    fn jitter_stays_within_step_range() {
        let step = 0.5;
        for i in 0..512u32 {
            let j = jittered_start_offset(i, 0x1234_5678, step);
            assert!(j >= 0.0);
            assert!(j < step);
        }
        // Zero step collapses the jitter to zero.
        assert!(approx(jittered_start_offset(7, 1, 0.0), 0.0));
    }

    #[test]
    fn jitter_is_deterministic_and_seed_sensitive() {
        let a = jittered_start_offset(3, 42, 1.0);
        let b = jittered_start_offset(3, 42, 1.0);
        let c = jittered_start_offset(3, 43, 1.0);
        assert!(approx(a, b));
        assert!(!approx(a, c));
    }
}
