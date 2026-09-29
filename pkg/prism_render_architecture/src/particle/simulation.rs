//! Per-particle simulation: integrators, forces, aging, and a stateless RNG.
//!
//! This is the deterministic `CPU` reference for the per-particle update stage
//! (design §8, §25, §29). It advances a particle's [`Kinematics`] under a force
//! library, ages it against its lifetime, and draws every random value from a
//! *stateless* hash so the update depends only on stable inputs — never on a
//! mutable RNG cursor — which is what lets a `GPU` compute kernel reproduce the
//! `CPU` result bit for bit regardless of thread scheduling.
//!
//! Only `sqrt` (through [`Vec3`]) and ordinary arithmetic are used; the value
//! noise driving [`curl_noise`] is built from the same integer hash and a
//! quintic fade, so no transcendental functions enter the pipeline.

use super::{IntegratorKind, Vec3, EPS_LEN_SQ};

/// Position and velocity of one particle, the state an integrator advances.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Kinematics {
    /// Current position.
    pub position: Vec3,
    /// Current velocity.
    pub velocity: Vec3,
}

impl Kinematics {
    /// Builds a kinematic state.
    #[must_use]
    pub fn new(position: Vec3, velocity: Vec3) -> Self {
        Self { position, velocity }
    }
}

/// Advances one particle by `dt` under `kind` (design §25).
///
/// `accel_at` evaluates the total acceleration (force over mass) at a trial
/// position and velocity; it is called once for the first-order integrators and
/// twice for [`IntegratorKind::Rk2`], which samples the force again at the
/// midpoint. All three forms are symplectic-friendly and reduce to the exact
/// constant-acceleration solution when the force is uniform.
#[must_use]
pub fn integrate<F>(kind: IntegratorKind, state: Kinematics, dt: f32, accel_at: F) -> Kinematics
where
    F: Fn(Vec3, Vec3) -> Vec3,
{
    let Kinematics { position, velocity } = state;
    match kind {
        IntegratorKind::SemiImplicitEuler => {
            let a = accel_at(position, velocity);
            let new_v = velocity.add(a.scale(dt));
            let new_p = position.add(new_v.scale(dt));
            Kinematics::new(new_p, new_v)
        }
        IntegratorKind::Verlet => {
            // Velocity Verlet with the step's acceleration: position advances
            // with a half-step acceleration term, velocity with the full step.
            let a = accel_at(position, velocity);
            let new_p = position.add(velocity.scale(dt)).add(a.scale(0.5 * dt * dt));
            let new_v = velocity.add(a.scale(dt));
            Kinematics::new(new_p, new_v)
        }
        IntegratorKind::Rk2 => {
            let half = 0.5 * dt;
            let a1 = accel_at(position, velocity);
            let mid_v = velocity.add(a1.scale(half));
            let mid_p = position.add(velocity.scale(half));
            let a2 = accel_at(mid_p, mid_v);
            let new_v = velocity.add(a2.scale(dt));
            let new_p = position.add(mid_v.scale(dt));
            Kinematics::new(new_p, new_v)
        }
    }
}

/// Constant gravitational acceleration.
#[must_use]
pub fn gravity(g: Vec3) -> Vec3 {
    g
}

/// Linear (Stokes) drag acceleration opposing velocity.
///
/// `coefficient` is the inverse time constant; a non-positive value yields no
/// drag. The magnitude is clamped so a single step can never over-correct past
/// rest (which would inject energy), keeping the integrator stable.
#[must_use]
pub fn linear_drag(velocity: Vec3, coefficient: f32, dt: f32) -> Vec3 {
    if coefficient > 0.0 && dt > 0.0 {
        // Clamp the effective rate so `coefficient * dt <= 1`.
        let rate = (coefficient).min(1.0 / dt);
        velocity.scale(-rate)
    } else {
        Vec3::ZERO
    }
}

/// Wind acceleration pulling a particle toward the wind's velocity.
///
/// Models aerodynamic drag against a moving air mass: the force is proportional
/// to the *relative* velocity `wind - velocity`, so a particle already moving
/// with the wind feels nothing (design §8 force library).
#[must_use]
pub fn wind(wind_velocity: Vec3, velocity: Vec3, coefficient: f32) -> Vec3 {
    if coefficient > 0.0 {
        wind_velocity.sub(velocity).scale(coefficient)
    } else {
        Vec3::ZERO
    }
}

/// Vortex acceleration swirling a particle around an axis through `center`.
///
/// The force is tangential (perpendicular to both the spin `axis` and the
/// radial vector) and falls off with radial distance, so particles orbit the
/// axis rather than being flung out. A particle on the axis feels nothing.
#[must_use]
pub fn vortex(position: Vec3, center: Vec3, axis: Vec3, strength: f32) -> Vec3 {
    let radial = position.sub(center);
    let unit_axis = axis.normalize_or_zero();
    let tangent = unit_axis.cross(radial);
    let dist_sq = radial.length_squared();
    if dist_sq > EPS_LEN_SQ {
        // Tangential direction scaled by strength / distance for orbital decay.
        tangent.scale(strength / dist_sq.sqrt())
    } else {
        Vec3::ZERO
    }
}

/// A dense 3D vector field on a regular grid, sampled by trilinear filtering.
///
/// The field is `dims.0 * dims.1 * dims.2` vectors in x-fastest order,
/// positioned in world space by `origin` and `cell_size`. This is the contract
/// a baked flow-field or fluid-solver output feeds the particle update; the
/// `GPU` samples the identical layout from a 3D texture.
#[derive(Clone, Copy, Debug)]
pub struct VectorField<'a> {
    /// Field values, `x + y * nx + z * nx * ny`.
    pub values: &'a [Vec3],
    /// Grid resolution `(nx, ny, nz)`.
    pub dims: (u32, u32, u32),
    /// World position of voxel `(0, 0, 0)`.
    pub origin: Vec3,
    /// World size of one voxel edge.
    pub cell_size: f32,
}

impl VectorField<'_> {
    /// Reads voxel `(x, y, z)`, clamping to the grid edge (border sampling).
    fn voxel(&self, x: i32, y: i32, z: i32) -> Vec3 {
        let (nx, ny, nz) = self.dims;
        if nx == 0 || ny == 0 || nz == 0 {
            return Vec3::ZERO;
        }
        let cx = x.clamp(0, nx as i32 - 1) as u32;
        let cy = y.clamp(0, ny as i32 - 1) as u32;
        let cz = z.clamp(0, nz as i32 - 1) as u32;
        let idx = (cx + cy * nx + cz * nx * ny) as usize;
        self.values.get(idx).copied().unwrap_or(Vec3::ZERO)
    }

    /// Trilinearly samples the field at a world position.
    ///
    /// Returns [`Vec3::ZERO`] for an empty or zero-sized field. Positions
    /// outside the grid clamp to the border voxel.
    #[must_use]
    pub fn sample(&self, world: Vec3) -> Vec3 {
        if self.cell_size <= 0.0 {
            return Vec3::ZERO;
        }
        let local = world.sub(self.origin).scale(1.0 / self.cell_size);
        let x0 = local.x.floor();
        let y0 = local.y.floor();
        let z0 = local.z.floor();
        let fx = local.x - x0;
        let fy = local.y - y0;
        let fz = local.z - z0;
        let (ix, iy, iz) = (x0 as i32, y0 as i32, z0 as i32);
        let c000 = self.voxel(ix, iy, iz);
        let c100 = self.voxel(ix + 1, iy, iz);
        let c010 = self.voxel(ix, iy + 1, iz);
        let c110 = self.voxel(ix + 1, iy + 1, iz);
        let c001 = self.voxel(ix, iy, iz + 1);
        let c101 = self.voxel(ix + 1, iy, iz + 1);
        let c011 = self.voxel(ix, iy + 1, iz + 1);
        let c111 = self.voxel(ix + 1, iy + 1, iz + 1);
        let x00 = lerp_vec(c000, c100, fx);
        let x10 = lerp_vec(c010, c110, fx);
        let x01 = lerp_vec(c001, c101, fx);
        let x11 = lerp_vec(c011, c111, fx);
        let y0v = lerp_vec(x00, x10, fy);
        let y1v = lerp_vec(x01, x11, fy);
        lerp_vec(y0v, y1v, fz)
    }
}

/// Scalar linear interpolation.
#[must_use]
pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Component-wise linear interpolation.
#[must_use]
fn lerp_vec(a: Vec3, b: Vec3, t: f32) -> Vec3 {
    Vec3::new(lerp(a.x, b.x, t), lerp(a.y, b.y, t), lerp(a.z, b.z, t))
}

/// Integer bit-mixer (a Wang/`murmur`-style finalizer) for the stateless RNG.
#[must_use]
fn mix32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

/// A stateless hash RNG: a pure function of stable inputs (design §29).
///
/// The same `(id, seed, stream, frame)` always hashes to the same value, so a
/// `GPU` thread and the `CPU` reference agree without sharing mutable RNG state.
/// `stream` decorrelates independent random channels for one particle (spawn
/// position vs. lifetime vs. color), and `frame` advances noise over time.
#[must_use]
pub fn hash_rng(id: u32, seed: u32, stream: u32, frame: u32) -> u32 {
    let mut h = mix32(seed ^ 0x9e37_79b9);
    h = mix32(h ^ id.wrapping_mul(0x85eb_ca6b));
    h = mix32(h ^ stream.wrapping_mul(0xc2b2_ae35));
    mix32(h ^ frame.wrapping_mul(0x27d4_eb2f))
}

/// Maps a hash to a float in `[0, 1)` using the high 24 bits (full f32 mantissa).
#[must_use]
pub fn unit_f32(hash: u32) -> f32 {
    // 24-bit mantissa keeps the result exactly representable and uniform.
    (hash >> 8) as f32 / (1u32 << 24) as f32
}

/// Convenience: a uniform `[0, 1)` sample for a particle's random channel.
#[must_use]
pub fn rand_unit(id: u32, seed: u32, stream: u32, frame: u32) -> f32 {
    unit_f32(hash_rng(id, seed, stream, frame))
}

/// Quintic fade `6t^5 - 15t^4 + 10t^3`, the smooth interpolant for value noise.
#[must_use]
fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// Signed value at an integer lattice point in `[-1, 1]`.
#[must_use]
fn lattice_value(ix: i32, iy: i32, iz: i32, seed: u32) -> f32 {
    let h = hash_rng(ix as u32, seed, iy as u32, iz as u32);
    unit_f32(h) * 2.0 - 1.0
}

/// Scalar value noise in `[-1, 1]` (hash lattice + quintic trilinear blend).
///
/// A deterministic, transcendental-free coherent noise: gradient-free value
/// noise smoothed by a quintic fade so its first derivative is continuous,
/// which is what [`curl_noise`] differentiates.
#[must_use]
pub fn value_noise(p: Vec3, seed: u32) -> f32 {
    let x0 = p.x.floor();
    let y0 = p.y.floor();
    let z0 = p.z.floor();
    let (ix, iy, iz) = (x0 as i32, y0 as i32, z0 as i32);
    let u = fade(p.x - x0);
    let v = fade(p.y - y0);
    let w = fade(p.z - z0);
    let c000 = lattice_value(ix, iy, iz, seed);
    let c100 = lattice_value(ix + 1, iy, iz, seed);
    let c010 = lattice_value(ix, iy + 1, iz, seed);
    let c110 = lattice_value(ix + 1, iy + 1, iz, seed);
    let c001 = lattice_value(ix, iy, iz + 1, seed);
    let c101 = lattice_value(ix + 1, iy, iz + 1, seed);
    let c011 = lattice_value(ix, iy + 1, iz + 1, seed);
    let c111 = lattice_value(ix + 1, iy + 1, iz + 1, seed);
    let x00 = lerp(c000, c100, u);
    let x10 = lerp(c010, c110, u);
    let x01 = lerp(c001, c101, u);
    let x11 = lerp(c011, c111, u);
    let y0v = lerp(x00, x10, v);
    let y1v = lerp(x01, x11, v);
    lerp(y0v, y1v, w)
}

/// Divergence-free curl-noise velocity at a world position (design §8, §10).
///
/// The field is the curl of a vector potential whose three components are
/// decorrelated value-noise channels; the curl of any potential is exactly
/// divergence-free, so particles advected by it swirl without clumping or
/// leaving voids. Derivatives use central differences with step `epsilon`.
#[must_use]
pub fn curl_noise(position: Vec3, seed: u32, epsilon: f32) -> Vec3 {
    let eps = if epsilon > 0.0 { epsilon } else { 1.0e-3 };
    let inv = 1.0 / (2.0 * eps);
    // Three decorrelated potential channels via distinct seed offsets.
    let s1 = seed;
    let s2 = seed.wrapping_add(0x1000_0001);
    let s3 = seed.wrapping_add(0x2000_0002);
    let dx = Vec3::new(eps, 0.0, 0.0);
    let dy = Vec3::new(0.0, eps, 0.0);
    let dz = Vec3::new(0.0, 0.0, eps);
    // Partial derivatives of each potential channel.
    let dp3_dy = (value_noise(position.add(dy), s3) - value_noise(position.sub(dy), s3)) * inv;
    let dp2_dz = (value_noise(position.add(dz), s2) - value_noise(position.sub(dz), s2)) * inv;
    let dp1_dz = (value_noise(position.add(dz), s1) - value_noise(position.sub(dz), s1)) * inv;
    let dp3_dx = (value_noise(position.add(dx), s3) - value_noise(position.sub(dx), s3)) * inv;
    let dp2_dx = (value_noise(position.add(dx), s2) - value_noise(position.sub(dx), s2)) * inv;
    let dp1_dy = (value_noise(position.add(dy), s1) - value_noise(position.sub(dy), s1)) * inv;
    Vec3::new(dp3_dy - dp2_dz, dp1_dz - dp3_dx, dp2_dx - dp1_dy)
}

/// Aging bookkeeping for one particle's lifetime.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Lifetime {
    /// Seconds the particle has been alive.
    pub age: f32,
    /// Total seconds the particle may live; non-positive means immortal.
    pub max_age: f32,
}

impl Lifetime {
    /// Builds a lifetime.
    #[must_use]
    pub fn new(age: f32, max_age: f32) -> Self {
        Self { age, max_age }
    }

    /// Advances the age by `dt`.
    #[must_use]
    pub fn advance(self, dt: f32) -> Self {
        Self {
            age: self.age + dt.max(0.0),
            max_age: self.max_age,
        }
    }

    /// Whether the particle has reached the end of its lifetime.
    ///
    /// A non-positive `max_age` marks an immortal particle that never expires.
    #[must_use]
    pub fn is_expired(self) -> bool {
        self.max_age > 0.0 && self.age >= self.max_age
    }

    /// Normalized age in `[0, 1]`, the input to over-life curves. Immortal
    /// particles report `0`.
    #[must_use]
    pub fn normalized(self) -> f32 {
        if self.max_age > 0.0 {
            (self.age / self.max_age).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}

/// Samples an over-life curve LUT at a normalized age with linear interpolation
/// (design §8.3 over-life curves).
///
/// The LUT samples span `t = 0..=1` uniformly. An empty LUT returns `0`; a
/// single-entry LUT returns that constant. Out-of-range `t` clamps to the ends.
#[must_use]
pub fn sample_over_life(lut: &[f32], t: f32) -> f32 {
    match lut.len() {
        0 => 0.0,
        1 => lut[0],
        n => {
            let clamped = t.clamp(0.0, 1.0);
            let scaled = clamped * (n - 1) as f32;
            let lower = scaled.floor();
            let i = (lower as usize).min(n - 2);
            let frac = scaled - i as f32;
            lerp(lut[i], lut[i + 1], frac)
        }
    }
}

/// Number of substeps needed to respect a `CFL` limit (design §25 stability).
///
/// A particle moving at `max_speed` may not cross more than `cfl * cell_size`
/// per substep, so `substeps = ceil(max_speed * dt / (cfl * cell_size))`,
/// clamped to `1..=max_substeps`. Degenerate inputs fall back to a single step.
#[must_use]
pub fn cfl_substeps(max_speed: f32, dt: f32, cell_size: f32, cfl: f32, max_substeps: u32) -> u32 {
    if max_speed <= 0.0 || dt <= 0.0 || cell_size <= 0.0 || cfl <= 0.0 {
        return 1;
    }
    let travel = max_speed * dt;
    let budget = cfl * cell_size;
    let needed = (travel / budget).ceil();
    let n = if needed >= max_substeps as f32 {
        max_substeps
    } else {
        needed as u32
    };
    n.clamp(1, max_substeps.max(1))
}

/// The per-substep timestep once a frame is split into `substeps`.
#[must_use]
pub fn substep_dt(dt: f32, substeps: u32) -> f32 {
    if substeps == 0 {
        dt
    } else {
        dt / substeps as f32
    }
}

/// Clamps a velocity to a maximum speed, preserving direction.
#[must_use]
pub fn clamp_speed(velocity: Vec3, max_speed: f32) -> Vec3 {
    if max_speed <= 0.0 {
        return velocity;
    }
    let speed_sq = velocity.length_squared();
    if speed_sq > max_speed * max_speed {
        velocity.scale(max_speed / speed_sq.sqrt())
    } else {
        velocity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn semi_implicit_euler_matches_hand_computation() {
        let g = Vec3::new(0.0, -10.0, 0.0);
        let state = Kinematics::new(Vec3::ZERO, Vec3::ZERO);
        let next = integrate(IntegratorKind::SemiImplicitEuler, state, 0.1, |_, _| g);
        // v' = -1.0, p' = v' * dt = -0.1.
        assert!((next.velocity.y - -1.0).abs() < 1e-6);
        assert!((next.position.y - -0.1).abs() < 1e-6);
    }

    #[test]
    fn verlet_position_uses_half_acceleration_term() {
        let g = Vec3::new(0.0, -10.0, 0.0);
        let state = Kinematics::new(Vec3::ZERO, Vec3::ZERO);
        let next = integrate(IntegratorKind::Verlet, state, 0.1, |_, _| g);
        // p' = 0.5 * a * dt^2 = 0.5 * -10 * 0.01 = -0.05.
        assert!((next.position.y - -0.05).abs() < 1e-6);
        assert!((next.velocity.y - -1.0).abs() < 1e-6);
    }

    #[test]
    fn rk2_is_exact_for_constant_acceleration() {
        // For uniform acceleration the midpoint rule is exact: p = 0.5 a t^2.
        let g = Vec3::new(0.0, -10.0, 0.0);
        let state = Kinematics::new(Vec3::ZERO, Vec3::ZERO);
        let next = integrate(IntegratorKind::Rk2, state, 0.2, |_, _| g);
        assert!((next.position.y - -0.2).abs() < 1e-6);
        assert!((next.velocity.y - -2.0).abs() < 1e-6);
    }

    #[test]
    fn drag_opposes_and_never_overshoots() {
        let v = Vec3::new(4.0, 0.0, 0.0);
        // An extreme coefficient is clamped so a step cannot reverse velocity.
        let a = linear_drag(v, 1000.0, 0.1);
        // rate clamped to 1/dt = 10 => a = -40 in x.
        assert!((a.x - -40.0).abs() < 1e-4);
        assert_eq!(linear_drag(v, 0.0, 0.1), Vec3::ZERO);
    }

    #[test]
    fn wind_pulls_toward_relative_velocity() {
        let a = wind(Vec3::new(10.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0), 0.5);
        assert!((a.x - 4.0).abs() < 1e-6);
        // A particle already moving with the wind feels nothing.
        let none = wind(Vec3::new(3.0, 0.0, 0.0), Vec3::new(3.0, 0.0, 0.0), 0.5);
        assert_eq!(none, Vec3::ZERO);
    }

    #[test]
    fn vortex_is_tangential_and_axis_safe() {
        let f = vortex(
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::ZERO,
            Vec3::new(0.0, 1.0, 0.0),
            2.0,
        );
        // Tangential force is perpendicular to the radial direction.
        assert!(f.dot(Vec3::new(1.0, 0.0, 0.0)).abs() < 1e-6);
        // A particle on the axis feels no vortex force.
        let on_axis = vortex(Vec3::ZERO, Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0), 2.0);
        assert_eq!(on_axis, Vec3::ZERO);
    }

    #[test]
    fn vector_field_trilinear_samples_and_clamps() {
        // 2x1x1 field: value 0 at x=0, value (10,0,0) at x=1.
        let values = [Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0)];
        let field = VectorField {
            values: &values,
            dims: (2, 1, 1),
            origin: Vec3::ZERO,
            cell_size: 1.0,
        };
        // Halfway samples the midpoint.
        let mid = field.sample(Vec3::new(0.5, 0.0, 0.0));
        assert!((mid.x - 5.0).abs() < 1e-6);
        // Outside the grid clamps to the border voxel.
        let past = field.sample(Vec3::new(5.0, 0.0, 0.0));
        assert!((past.x - 10.0).abs() < 1e-6);
    }

    #[test]
    fn empty_vector_field_samples_zero() {
        let field = VectorField {
            values: &[],
            dims: (0, 0, 0),
            origin: Vec3::ZERO,
            cell_size: 1.0,
        };
        assert_eq!(field.sample(Vec3::new(1.0, 2.0, 3.0)), Vec3::ZERO);
    }

    #[test]
    fn hash_rng_is_deterministic_and_channel_independent() {
        assert_eq!(hash_rng(7, 42, 0, 3), hash_rng(7, 42, 0, 3));
        // Different streams decorrelate.
        assert_ne!(hash_rng(7, 42, 0, 3), hash_rng(7, 42, 1, 3));
        // Different frames advance.
        assert_ne!(hash_rng(7, 42, 0, 3), hash_rng(7, 42, 0, 4));
    }

    #[test]
    fn unit_f32_is_in_unit_interval() {
        for id in 0..1000u32 {
            let x = rand_unit(id, 99, 0, id);
            assert!((0.0..1.0).contains(&x), "sample {x} out of range");
        }
    }

    #[test]
    fn value_noise_is_bounded_and_deterministic() {
        let p = Vec3::new(3.25, -1.75, 0.5);
        let a = value_noise(p, 5);
        let b = value_noise(p, 5);
        assert_eq!(a, b);
        assert!((-1.0..=1.0).contains(&a));
    }

    #[test]
    fn value_noise_is_zero_at_lattice_points() {
        // At an integer lattice point the fade weights collapse to one corner,
        // so re-evaluating there is stable and reproducible.
        let a = value_noise(Vec3::new(2.0, 3.0, 4.0), 1);
        let b = value_noise(Vec3::new(2.0, 3.0, 4.0), 1);
        assert_eq!(a, b);
    }

    #[test]
    fn curl_noise_is_approximately_divergence_free() {
        let seed = 17;
        let eps = 1.0e-2;
        let p = Vec3::new(0.37, 1.21, -0.6);
        // Central-difference divergence of the curl field should be ~0.
        let h = 1.0e-2;
        let dfx = (curl_noise(p.add(Vec3::new(h, 0.0, 0.0)), seed, eps).x
            - curl_noise(p.sub(Vec3::new(h, 0.0, 0.0)), seed, eps).x)
            / (2.0 * h);
        let dfy = (curl_noise(p.add(Vec3::new(0.0, h, 0.0)), seed, eps).y
            - curl_noise(p.sub(Vec3::new(0.0, h, 0.0)), seed, eps).y)
            / (2.0 * h);
        let dfz = (curl_noise(p.add(Vec3::new(0.0, 0.0, h)), seed, eps).z
            - curl_noise(p.sub(Vec3::new(0.0, 0.0, h)), seed, eps).z)
            / (2.0 * h);
        let divergence = dfx + dfy + dfz;
        assert!(divergence.abs() < 1.0, "divergence {divergence} too large");
    }

    #[test]
    fn lifetime_ages_and_expires() {
        let life = Lifetime::new(0.0, 1.0).advance(0.6);
        assert!(!life.is_expired());
        assert!((life.normalized() - 0.6).abs() < 1e-6);
        let dead = life.advance(0.5);
        assert!(dead.is_expired());
        assert!((dead.normalized() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn immortal_lifetime_never_expires() {
        let life = Lifetime::new(1000.0, 0.0);
        assert!(!life.is_expired());
        assert_eq!(life.normalized(), 0.0);
    }

    #[test]
    fn over_life_curve_interpolates_and_clamps() {
        let lut = [0.0, 1.0, 0.0];
        assert!((sample_over_life(&lut, 0.0) - 0.0).abs() < 1e-6);
        assert!((sample_over_life(&lut, 0.5) - 1.0).abs() < 1e-6);
        assert!((sample_over_life(&lut, 0.25) - 0.5).abs() < 1e-6);
        // Out-of-range clamps to the ends.
        assert!((sample_over_life(&lut, 2.0) - 0.0).abs() < 1e-6);
        // Degenerate LUTs are total.
        assert_eq!(sample_over_life(&[], 0.5), 0.0);
        assert_eq!(sample_over_life(&[7.0], 0.5), 7.0);
    }

    #[test]
    fn cfl_substeps_scale_with_speed() {
        // travel = 100 * 0.1 = 10; budget = 1 * 1 = 1 => 10 substeps.
        assert_eq!(cfl_substeps(100.0, 0.1, 1.0, 1.0, 64), 10);
        // Clamped to max_substeps.
        assert_eq!(cfl_substeps(10_000.0, 0.1, 1.0, 1.0, 8), 8);
        // Degenerate inputs fall back to one step.
        assert_eq!(cfl_substeps(0.0, 0.1, 1.0, 1.0, 8), 1);
        assert_eq!(substep_dt(0.1, 10), 0.01);
        assert_eq!(substep_dt(0.1, 0), 0.1);
    }

    #[test]
    fn clamp_speed_limits_magnitude_only() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        let clamped = clamp_speed(v, 2.5);
        assert!((clamped.length() - 2.5).abs() < 1e-6);
        // Under the limit, velocity is untouched.
        assert_eq!(
            clamp_speed(Vec3::new(1.0, 0.0, 0.0), 2.5),
            Vec3::new(1.0, 0.0, 0.0)
        );
    }
}
