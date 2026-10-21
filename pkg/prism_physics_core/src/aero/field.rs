use alloc::vec::Vec;

use glam::Vec3;

use super::sanitize::{add, finite, non_negative, normalize, smoothstep01, vec3};

pub type Real = f32;

/// The instantaneous fluid velocity and its material acceleration at a point.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WindSample {
    pub velocity: Vec3,
    pub acceleration: Vec3,
}

/// A compact, deterministic gust source with a smooth spatial and temporal envelope.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gust {
    pub center: Vec3,
    pub direction: Vec3,
    pub amplitude: Real,
    pub radius: Real,
    pub start_time: Real,
    pub end_time: Real,
    pub ramp_time: Real,
    pub frequency: Real,
    pub phase: Real,
}

impl Gust {
    #[must_use]
    pub fn new(center: Vec3, direction: Vec3, amplitude: Real, radius: Real) -> Self {
        Self {
            center,
            direction,
            amplitude,
            radius,
            start_time: 0.0,
            end_time: Real::INFINITY,
            ramp_time: 0.0,
            frequency: 0.0,
            phase: 0.0,
        }
    }

    #[must_use]
    pub fn with_time_window(mut self, start_time: Real, end_time: Real, ramp_time: Real) -> Self {
        self.start_time = start_time;
        self.end_time = end_time;
        self.ramp_time = ramp_time;
        self
    }

    #[must_use]
    pub fn with_oscillation(mut self, frequency: Real, phase: Real) -> Self {
        self.frequency = frequency;
        self.phase = phase;
        self
    }

    fn sample(&self, position: Vec3, time: Real) -> WindSample {
        let direction = normalize(self.direction, Vec3::ZERO);
        let amplitude = finite(self.amplitude);
        let radius = non_negative(self.radius);
        let radial_distance = (vec3(position) - vec3(self.center)).length();
        let spatial = if radius > 1.0e-6 {
            smoothstep01((radius - radial_distance) / radius)
        } else if radial_distance <= 1.0e-6 {
            1.0
        } else {
            0.0
        };
        if spatial == 0.0 || direction == Vec3::ZERO {
            return WindSample::default();
        }

        let time = finite(time);
        let start = finite(self.start_time);
        let end = self.end_time;
        if time < start || time > end {
            return WindSample::default();
        }
        let ramp = non_negative(self.ramp_time);
        let rise_t = (time - start) / ramp;
        let fall_t = (end - time) / ramp;
        let rise = if ramp > 1.0e-6 {
            smoothstep01(rise_t)
        } else {
            1.0
        };
        let fall = if ramp > 1.0e-6 && end.is_finite() {
            smoothstep01(fall_t)
        } else {
            1.0
        };
        let envelope = rise * fall;
        let ramp_derivative = |t: Real| {
            if ramp > 1.0e-6 && (0.0..1.0).contains(&t) {
                6.0 * t * (1.0 - t) / ramp
            } else {
                0.0
            }
        };
        let envelope_rate = ramp_derivative(rise_t) * fall - rise * ramp_derivative(fall_t);
        let omega = finite(self.frequency) * core::f32::consts::TAU;
        let phase = omega * time + finite(self.phase);
        let oscillation = if omega.abs() > 0.0 { phase.sin() } else { 1.0 };
        let oscillation_rate = if omega.abs() > 0.0 {
            omega * phase.cos()
        } else {
            0.0
        };
        let value = amplitude * spatial * envelope * oscillation;
        let acceleration = direction
            * (amplitude * spatial * (envelope_rate * oscillation + envelope * oscillation_rate));
        WindSample {
            velocity: vec3(direction * value),
            acceleration: vec3(acceleration),
        }
    }
}

/// Deterministic value-noise turbulence. The integer hash and interpolation do
/// not depend on a random generator or global state, making CPU/GPU evaluation
/// reproducible for the same seed and sample coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Turbulence {
    pub amplitude: Vec3,
    pub cell_size: Real,
    pub temporal_frequency: Real,
    pub seed: u32,
}

impl Default for Turbulence {
    fn default() -> Self {
        Self {
            amplitude: Vec3::ZERO,
            cell_size: 8.0,
            temporal_frequency: 0.25,
            seed: 0xA341_316C,
        }
    }
}

impl Turbulence {
    #[must_use]
    pub fn sample(&self, position: Vec3, time: Real) -> WindSample {
        let amplitude = vec3(self.amplitude);
        let cell = non_negative(self.cell_size).max(1.0e-3);
        if amplitude == Vec3::ZERO {
            return WindSample::default();
        }
        let frequency = finite(self.temporal_frequency);
        let t = finite(finite(time) * frequency);
        let position = vec3(vec3(position) / cell);
        let velocity = Vec3::new(
            value_noise(position, t, self.seed ^ 0x11A2_3B45),
            value_noise(position, t, self.seed ^ 0x93D7_2F19),
            value_noise(position, t, self.seed ^ 0xD1B5_4A67),
        ) * amplitude;
        let dt = 1.0e-3;
        let before = Vec3::new(
            value_noise(position, t - frequency * dt, self.seed ^ 0x11A2_3B45),
            value_noise(position, t - frequency * dt, self.seed ^ 0x93D7_2F19),
            value_noise(position, t - frequency * dt, self.seed ^ 0xD1B5_4A67),
        ) * amplitude;
        let after = Vec3::new(
            value_noise(position, t + frequency * dt, self.seed ^ 0x11A2_3B45),
            value_noise(position, t + frequency * dt, self.seed ^ 0x93D7_2F19),
            value_noise(position, t + frequency * dt, self.seed ^ 0xD1B5_4A67),
        ) * amplitude;
        WindSample {
            velocity: vec3(velocity),
            acceleration: vec3((after - before) / (2.0 * dt)),
        }
    }
}

/// A finite-radius, finite-length flow jet used for propeller slipstream or rotor downwash.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlowJet {
    pub origin: Vec3,
    pub direction: Vec3,
    pub radius: Real,
    pub length: Real,
    pub speed: Real,
    pub edge_exponent: Real,
}

impl FlowJet {
    #[must_use]
    pub fn new(origin: Vec3, direction: Vec3, radius: Real, length: Real, speed: Real) -> Self {
        Self {
            origin,
            direction,
            radius,
            length,
            speed,
            edge_exponent: 1.0,
        }
    }

    #[must_use]
    pub fn with_edge_exponent(mut self, edge_exponent: Real) -> Self {
        self.edge_exponent = edge_exponent;
        self
    }

    #[must_use]
    pub fn velocity_at(&self, position: Vec3) -> Vec3 {
        let direction = normalize(self.direction, Vec3::ZERO);
        let radius = non_negative(self.radius);
        let length = non_negative(self.length);
        if direction == Vec3::ZERO || radius <= 1.0e-6 || length <= 1.0e-6 {
            return Vec3::ZERO;
        }
        let offset = vec3(position) - vec3(self.origin);
        let axial = offset.dot(direction);
        if !(0.0..=length).contains(&axial) {
            return Vec3::ZERO;
        }
        let radial = (offset - direction * axial).length();
        if radial >= radius {
            return Vec3::ZERO;
        }
        let axial_falloff = 1.0 - axial / length;
        let radial_falloff = 1.0 - radial / radius;
        let exponent = non_negative(self.edge_exponent).max(1.0e-3);
        direction * finite(self.speed) * axial_falloff * radial_falloff.powf(exponent)
    }
}

/// Named wrapper for a propeller/jet wake.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Slipstream(pub FlowJet);

/// Named wrapper for a rotor or wing downwash field.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Downwash(pub FlowJet);

/// Multipliers applied to a panel's lift and drag in ground effect.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroundEffectScale {
    pub lift: Real,
    pub drag: Real,
}

impl Default for GroundEffectScale {
    fn default() -> Self {
        Self {
            lift: 1.0,
            drag: 1.0,
        }
    }
}

/// Plane-based ground-effect model. The plane normal points away from the ground.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroundEffect {
    pub plane_point: Vec3,
    pub plane_normal: Vec3,
    pub influence_height: Real,
    pub lift_gain: Real,
    pub drag_gain: Real,
}

impl GroundEffect {
    #[must_use]
    pub fn scale_at(&self, position: Vec3) -> GroundEffectScale {
        let normal = normalize(self.plane_normal, Vec3::Y);
        let influence = non_negative(self.influence_height);
        if influence <= 1.0e-6 {
            return GroundEffectScale::default();
        }
        let height = (vec3(position) - vec3(self.plane_point)).dot(normal);
        let proximity = if height > 0.0 {
            1.0 - height / influence
        } else {
            1.0
        };
        let weight = smoothstep01(proximity);
        GroundEffectScale {
            lift: 1.0 + non_negative(self.lift_gain) * weight,
            drag: (1.0 + finite(self.drag_gain).max(-1.0) * weight).max(0.0),
        }
    }
}

impl Default for GroundEffect {
    fn default() -> Self {
        Self {
            plane_point: Vec3::ZERO,
            plane_normal: Vec3::Y,
            influence_height: 0.0,
            lift_gain: 0.0,
            drag_gain: 0.0,
        }
    }
}

/// Extensible environment contract used by the pure rigid-body evaluator.
pub trait FlowField {
    fn sample(&self, position: Vec3, time: Real) -> WindSample;

    fn ground_effect(&self, _position: Vec3) -> GroundEffectScale {
        GroundEffectScale::default()
    }
}

/// Composite wind/flow field used by the stock M12 evaluator.
#[derive(Clone, Debug, PartialEq)]
pub struct WindField {
    pub base_velocity: Vec3,
    pub gusts: Vec<Gust>,
    pub turbulence: Turbulence,
    pub slipstreams: Vec<Slipstream>,
    pub downwash: Vec<Downwash>,
    pub ground_effect: GroundEffect,
}

impl WindField {
    #[must_use]
    pub fn new(base_velocity: Vec3) -> Self {
        Self {
            base_velocity,
            gusts: Vec::new(),
            turbulence: Turbulence::default(),
            slipstreams: Vec::new(),
            downwash: Vec::new(),
            ground_effect: GroundEffect::default(),
        }
    }

    #[must_use]
    pub fn with_gust(mut self, gust: Gust) -> Self {
        self.gusts.push(gust);
        self
    }

    #[must_use]
    pub fn with_turbulence(mut self, turbulence: Turbulence) -> Self {
        self.turbulence = turbulence;
        self
    }

    #[must_use]
    pub fn with_slipstream(mut self, slipstream: Slipstream) -> Self {
        self.slipstreams.push(slipstream);
        self
    }

    #[must_use]
    pub fn with_downwash(mut self, downwash: Downwash) -> Self {
        self.downwash.push(downwash);
        self
    }

    #[must_use]
    pub fn with_ground_effect(mut self, ground_effect: GroundEffect) -> Self {
        self.ground_effect = ground_effect;
        self
    }
}

impl Default for WindField {
    fn default() -> Self {
        Self::new(Vec3::ZERO)
    }
}

impl FlowField for WindField {
    fn sample(&self, position: Vec3, time: Real) -> WindSample {
        let mut result = WindSample {
            velocity: vec3(self.base_velocity),
            acceleration: Vec3::ZERO,
        };
        for gust in &self.gusts {
            let sample = gust.sample(position, time);
            result.velocity = add(result.velocity, sample.velocity);
            result.acceleration = add(result.acceleration, sample.acceleration);
        }
        let turbulence = self.turbulence.sample(position, time);
        result.velocity = add(result.velocity, turbulence.velocity);
        result.acceleration = add(result.acceleration, turbulence.acceleration);
        for jet in &self.slipstreams {
            result.velocity = add(result.velocity, jet.0.velocity_at(position));
        }
        for jet in &self.downwash {
            result.velocity = add(result.velocity, jet.0.velocity_at(position));
        }
        WindSample {
            velocity: vec3(result.velocity),
            acceleration: vec3(result.acceleration),
        }
    }

    fn ground_effect(&self, position: Vec3) -> GroundEffectScale {
        self.ground_effect.scale_at(position)
    }
}

fn hash4(x: i32, y: i32, z: i32, t: i32, seed: u32) -> f32 {
    let mut value = seed
        ^ (x as u32).wrapping_mul(0x9E37_79B9)
        ^ (y as u32).wrapping_mul(0x85EB_CA77)
        ^ (z as u32).wrapping_mul(0xC2B2_AE3D)
        ^ (t as u32).wrapping_mul(0x27D4_EB2F);
    value ^= value >> 16;
    value = value.wrapping_mul(0x7FEB_352D);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846C_A68B);
    value ^= value >> 16;
    (value as f32 / u32::MAX as f32) * 2.0 - 1.0
}

fn fade(value: f32) -> f32 {
    value * value * (3.0 - 2.0 * value)
}

fn value_noise(position: Vec3, time: f32, seed: u32) -> f32 {
    // Reduce lattice indices before conversion. Saturating float-to-int casts
    // lose the fractional cell and `i32::MAX + 1` panics in debug builds.
    let lattice = |value: f32| {
        let value = finite(value);
        let base = value.floor();
        (
            (f64::from(base).rem_euclid(4_294_967_296.0) as u32) as i32,
            fade(value - base),
        )
    };
    let (base_x, tx) = lattice(position.x);
    let (base_y, ty) = lattice(position.y);
    let (base_z, tz) = lattice(position.z);
    let (ti, tt) = lattice(time);
    let mut values = [0.0; 16];
    let mut index = 0;
    for dt in 0..=1 {
        for dz in 0..=1 {
            for dy in 0..=1 {
                for dx in 0..=1 {
                    values[index] = hash4(
                        base_x.wrapping_add(dx),
                        base_y.wrapping_add(dy),
                        base_z.wrapping_add(dz),
                        ti.wrapping_add(dt),
                        seed,
                    );
                    index += 1;
                }
            }
        }
    }
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    let mut temporal = [0.0; 8];
    for i in 0..8 {
        temporal[i] = lerp(values[i], values[i + 8], tt);
    }
    let mut z_values = [0.0; 4];
    for i in 0..4 {
        z_values[i] = lerp(temporal[i * 2], temporal[i * 2 + 1], tx);
    }
    let y0 = lerp(z_values[0], z_values[1], ty);
    let y1 = lerp(z_values[2], z_values[3], ty);
    lerp(y0, y1, tz)
}
