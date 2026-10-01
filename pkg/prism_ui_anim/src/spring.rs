//! A physical damped-spring integrator.
//!
//! The spring models the ODE `m * x'' + c * x' + k * x = k * target`, i.e. a
//! mass on a spring with stiffness `k`, damping `c` and mass `m`, whose rest
//! position is `target`. [`SpringState::step`] advances the state using the
//! **exact** analytic solution of the linear ODE over the step `dt`, so it is
//! unconditionally stable and converges for the under-, critically- and
//! over-damped regimes regardless of step size.

use crate::math::{absf, cosf, expf, sinf, sqrtf};

/// Spring parameters: stiffness, damping and mass.
///
/// Construct directly with [`Spring::new`] or via the presets
/// [`Spring::default`], [`Spring::gentle`], [`Spring::wobbly`] and
/// [`Spring::stiff`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spring {
    /// Spring stiffness `k` (must be positive).
    pub stiffness: f32,
    /// Damping coefficient `c` (non-negative).
    pub damping: f32,
    /// Mass `m` (must be positive).
    pub mass: f32,
}

impl Default for Spring {
    /// A balanced, snappy spring (`stiffness = 170`, `damping = 26`).
    #[inline]
    fn default() -> Self {
        Spring::new(170.0, 26.0, 1.0)
    }
}

impl Spring {
    /// Create a spring from `stiffness`, `damping` and `mass`.
    ///
    /// `stiffness` and `mass` are floored to a tiny positive value to keep the
    /// dynamics well-defined; `damping` is floored at `0.0`.
    #[inline]
    pub fn new(stiffness: f32, damping: f32, mass: f32) -> Self {
        Self {
            stiffness: if stiffness > 1e-6 { stiffness } else { 1e-6 },
            damping: if damping > 0.0 { damping } else { 0.0 },
            mass: if mass > 1e-6 { mass } else { 1e-6 },
        }
    }

    /// A soft, slow spring.
    #[inline]
    pub fn gentle() -> Self {
        Spring::new(120.0, 14.0, 1.0)
    }

    /// A bouncy, oscillating spring.
    #[inline]
    pub fn wobbly() -> Self {
        Spring::new(180.0, 12.0, 1.0)
    }

    /// A fast, tightly-damped spring.
    #[inline]
    pub fn stiff() -> Self {
        Spring::new(210.0, 20.0, 1.0)
    }

    /// Undamped natural angular frequency `omega_0 = sqrt(k / m)`.
    #[inline]
    pub fn natural_frequency(&self) -> f32 {
        sqrtf(self.stiffness / self.mass)
    }

    /// Damping ratio `zeta = c / (2 * sqrt(k * m))`.
    ///
    /// Values below `1.0` are under-damped, `1.0` is critically damped and
    /// above `1.0` is over-damped.
    #[inline]
    pub fn damping_ratio(&self) -> f32 {
        self.damping / (2.0 * sqrtf(self.stiffness * self.mass))
    }

    /// Create a fresh [`SpringState`] starting at `value` with zero velocity.
    #[inline]
    pub fn state_at(&self, value: f32) -> SpringState {
        SpringState {
            value,
            velocity: 0.0,
        }
    }
}

/// The instantaneous state of a spring: its `value` and `velocity`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpringState {
    /// Current position.
    pub value: f32,
    /// Current velocity.
    pub velocity: f32,
}

impl SpringState {
    /// Create a state with an explicit `value` and `velocity`.
    #[inline]
    pub fn new(value: f32, velocity: f32) -> Self {
        Self { value, velocity }
    }

    /// Advance the state towards `target` over the time step `dt` using the
    /// exact analytic solution for `spring`.
    ///
    /// Non-positive `dt` leaves the state unchanged.
    pub fn step(&mut self, spring: &Spring, target: f32, dt: f32) {
        if dt <= 0.0 {
            return;
        }

        let omega0 = spring.natural_frequency();
        let zeta = spring.damping_ratio();

        // Work in displacement space relative to the rest position.
        let x0 = self.value - target;
        let v0 = self.velocity;

        let (x, v) = if zeta < 1.0 - 1e-4 {
            // Under-damped.
            let omega_d = omega0 * sqrtf(1.0 - zeta * zeta);
            let decay = expf(-zeta * omega0 * dt);
            let (sin_t, cos_t) = (sinf(omega_d * dt), cosf(omega_d * dt));
            let a = x0;
            let b = (v0 + zeta * omega0 * x0) / omega_d;
            let x = decay * (a * cos_t + b * sin_t);
            let v = decay
                * ((-zeta * omega0 * a + b * omega_d) * cos_t
                    + (-zeta * omega0 * b - a * omega_d) * sin_t);
            (x, v)
        } else if zeta <= 1.0 + 1e-4 {
            // Critically damped.
            let decay = expf(-omega0 * dt);
            let c = x0;
            let d = v0 + omega0 * x0;
            let x = decay * (c + d * dt);
            let v = decay * (v0 - omega0 * d * dt);
            (x, v)
        } else {
            // Over-damped.
            let s = omega0 * sqrtf(zeta * zeta - 1.0);
            let r1 = -zeta * omega0 + s;
            let r2 = -zeta * omega0 - s;
            let a = (v0 - r2 * x0) / (r1 - r2);
            let b = x0 - a;
            let e1 = expf(r1 * dt);
            let e2 = expf(r2 * dt);
            let x = a * e1 + b * e2;
            let v = a * r1 * e1 + b * r2 * e2;
            (x, v)
        };

        self.value = target + x;
        self.velocity = v;
    }

    /// Whether the spring has effectively settled at `target`.
    ///
    /// Returns `true` once both the distance to `target` and the speed are
    /// within `eps`.
    #[inline]
    pub fn is_settled(&self, target: f32, eps: f32) -> bool {
        absf(self.value - target) <= eps && absf(self.velocity) <= eps
    }
}
