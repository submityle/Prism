//! Shallow-water-equations (`SWE`) surface stepping: `CFL` timestep selection,
//! conservative continuity flux, momentum update, and interaction injection.
//!
//! Surface bodies (rivers, ponds, flooded terrain, shorelines) are modelled as
//! a height field `h` over a regular grid with a depth-averaged velocity field
//! `(u, v)`. The continuity equation `dh/dt + d(hu)/dx + d(hv)/dz = 0` is
//! advanced in conservative flux form so total water volume is preserved to
//! rounding; the depth-averaged momentum equations add gravity-driven
//! acceleration, upwind self-advection, and linear damping. Explicit stepping
//! is only stable under the Courant-Friedrichs-Lewy (`CFL`) condition, so the
//! module exposes both the wave-speed estimate and the largest stable timestep.
//!
//! Interaction sources (a character wading, rain drops, a boat hull) inject a
//! localized depth or velocity perturbation between steps; injection is the
//! only operation that changes total volume, and it is clamped to in-range
//! cells so a stale source index can never panic.
//!
//! Everything is a pure, deterministic function over caller-owned state. The
//! only transcendental used is `sqrt` (gravity-wave celerity); iteration order
//! is fixed row-major so repeated runs are bit-for-bit reproducible.

use alloc::vec;
use alloc::vec::Vec;

use super::EPS;

/// Regular-grid layout and physical constants for a shallow-water body.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SweConfig {
    /// Cell count along x (`>= 1`).
    pub nx: u32,
    /// Cell count along z (`>= 1`).
    pub nz: u32,
    /// Cell size in meters (`> 0`); square cells.
    pub dx: f32,
    /// Gravitational acceleration used for wave celerity and the pressure term.
    pub gravity: f32,
    /// Linear velocity damping per second, in `0..=1` (drag toward rest).
    pub damping: f32,
}

impl SweConfig {
    /// Total number of cells in the grid.
    #[must_use]
    pub fn cell_count(self) -> usize {
        (self.nx as usize).saturating_mul(self.nz as usize)
    }

    /// Row-major flat index for a cell, or `None` when out of range.
    #[must_use]
    pub fn index(self, x: u32, z: u32) -> Option<usize> {
        if x >= self.nx || z >= self.nz {
            return None;
        }
        Some((z as usize) * (self.nx as usize) + (x as usize))
    }
}

/// Mutable shallow-water state: depth and depth-averaged velocity per cell.
///
/// All three vectors are row-major with `nx * nz` entries. `h` is water depth
/// (meters, non-negative in a physical state); `u`/`v` are the x/z velocity
/// components (meters per second).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SweState {
    /// Water depth per cell.
    pub h: Vec<f32>,
    /// x-velocity per cell.
    pub u: Vec<f32>,
    /// z-velocity per cell.
    pub v: Vec<f32>,
}

impl SweState {
    /// Allocates a still body of uniform `depth` and zero velocity.
    #[must_use]
    pub fn still(cfg: SweConfig, depth: f32) -> Self {
        let n = cfg.cell_count();
        Self {
            h: vec![depth; n],
            u: vec![0.0; n],
            v: vec![0.0; n],
        }
    }

    /// Total water volume `sum(h) * dx * dx`, the conserved quantity.
    #[must_use]
    pub fn total_volume(&self, cfg: SweConfig) -> f32 {
        let cell_area = cfg.dx * cfg.dx;
        let mut sum = 0.0;
        for &depth in &self.h {
            sum += depth;
        }
        sum * cell_area
    }
}

/// Gravity-wave celerity plus flow speed at one cell: `|(u,v)| + sqrt(g*h)`.
///
/// This is the local signal speed the `CFL` condition must resolve. Negative
/// depth (a transiently dry cell) is treated as zero so the square root stays
/// real.
#[must_use]
pub fn cell_wave_speed(depth: f32, u: f32, v: f32, gravity: f32) -> f32 {
    let flow = (u * u + v * v).sqrt();
    let clamped_depth = if depth > 0.0 { depth } else { 0.0 };
    flow + (gravity * clamped_depth).sqrt()
}

/// Largest signal speed over the whole grid, used to pick a stable timestep.
#[must_use]
pub fn max_wave_speed(state: &SweState, cfg: SweConfig) -> f32 {
    let mut max_speed = 0.0_f32;
    let n = state.h.len().min(state.u.len()).min(state.v.len());
    let mut i = 0;
    while i < n {
        let speed = cell_wave_speed(state.h[i], state.u[i], state.v[i], cfg.gravity);
        if speed > max_speed {
            max_speed = speed;
        }
        i += 1;
    }
    max_speed
}

/// Largest explicit timestep satisfying the `CFL` condition for a signal speed.
///
/// `dt = cfl * dx / max_speed`. A non-positive signal speed (a body fully at
/// rest with zero depth) imposes no constraint, so a large sentinel step is
/// returned; callers clamp it to their own frame budget.
#[must_use]
pub fn cfl_timestep(max_speed: f32, dx: f32, cfl_number: f32) -> f32 {
    if max_speed <= EPS {
        return f32::MAX;
    }
    cfl_number * dx / max_speed
}

/// `true` when an explicit step of `dt` is `CFL`-stable for a signal speed.
///
/// Stability requires `dt * max_speed <= cfl_number * dx` (the Courant number
/// stays within bounds). A small tolerance absorbs rounding at the boundary.
#[must_use]
pub fn is_cfl_stable(dt: f32, dx: f32, max_speed: f32, cfl_number: f32) -> bool {
    dt * max_speed <= cfl_number * dx + EPS
}

/// Advances the shallow-water state by `dt` and returns the new state.
///
/// Continuity is integrated in conservative flux form: each interior face
/// carries the averaged momentum of its two cells, and reflective (no-flux)
/// walls zero the boundary faces. Because a shared face flux enters its two
/// neighbours with opposite sign, the summed depth change telescopes to zero
/// and total volume is preserved to rounding. Momentum uses a central pressure
/// gradient, first-order upwind self-advection, and linear damping. A body at
/// rest stays exactly at rest.
#[must_use]
pub fn step(state: &SweState, cfg: SweConfig, dt: f32) -> SweState {
    let nx = cfg.nx as usize;
    let nz = cfg.nz as usize;
    let n = nx * nz;
    if n == 0 || state.h.len() < n || state.u.len() < n || state.v.len() < n {
        return state.clone();
    }
    let inv_dx = 1.0 / cfg.dx;
    let h = &state.h;
    let u = &state.u;
    let v = &state.v;

    // Momentum flux hu, hv per cell (row-major).
    let mut hu = vec![0.0_f32; n];
    let mut hv = vec![0.0_f32; n];
    let mut i = 0;
    while i < n {
        hu[i] = h[i] * u[i];
        hv[i] = h[i] * v[i];
        i += 1;
    }

    let mut new_h = vec![0.0_f32; n];
    let mut new_u = vec![0.0_f32; n];
    let mut new_v = vec![0.0_f32; n];
    let damp = (1.0 - cfg.damping * dt).max(0.0);

    let mut z = 0;
    while z < nz {
        let mut x = 0;
        while x < nx {
            let idx = z * nx + x;
            // Conservative continuity: shared face flux cancels across cells.
            let fx_right = if x + 1 < nx {
                0.5 * (hu[idx] + hu[idx + 1])
            } else {
                0.0
            };
            let fx_left = if x >= 1 {
                0.5 * (hu[idx - 1] + hu[idx])
            } else {
                0.0
            };
            let fz_up = if z + 1 < nz {
                0.5 * (hv[idx] + hv[idx + nx])
            } else {
                0.0
            };
            let fz_down = if z >= 1 {
                0.5 * (hv[idx - nx] + hv[idx])
            } else {
                0.0
            };
            new_h[idx] = h[idx] - dt * inv_dx * ((fx_right - fx_left) + (fz_up - fz_down));

            // Reflective neighbour sampling for the momentum gradients.
            let h_xp = if x + 1 < nx { h[idx + 1] } else { h[idx] };
            let h_xm = if x >= 1 { h[idx - 1] } else { h[idx] };
            let h_zp = if z + 1 < nz { h[idx + nx] } else { h[idx] };
            let h_zm = if z >= 1 { h[idx - nx] } else { h[idx] };
            let dhdx = (h_xp - h_xm) * 0.5 * inv_dx;
            let dhdz = (h_zp - h_zm) * 0.5 * inv_dx;

            // First-order upwind self-advection of the velocity field.
            let u_xp = if x + 1 < nx { u[idx + 1] } else { u[idx] };
            let u_xm = if x >= 1 { u[idx - 1] } else { u[idx] };
            let u_zp = if z + 1 < nz { u[idx + nx] } else { u[idx] };
            let u_zm = if z >= 1 { u[idx - nx] } else { u[idx] };
            let v_xp = if x + 1 < nx { v[idx + 1] } else { v[idx] };
            let v_xm = if x >= 1 { v[idx - 1] } else { v[idx] };
            let v_zp = if z + 1 < nz { v[idx + nx] } else { v[idx] };
            let v_zm = if z >= 1 { v[idx - nx] } else { v[idx] };

            let adv_u = upwind(u[idx], v[idx], u[idx], u_xm, u_xp, u_zm, u_zp, inv_dx);
            let adv_v = upwind(u[idx], v[idx], v[idx], v_xm, v_xp, v_zm, v_zp, inv_dx);

            let un = (u[idx] - dt * (adv_u + cfg.gravity * dhdx)) * damp;
            let vn = (v[idx] - dt * (adv_v + cfg.gravity * dhdz)) * damp;
            new_u[idx] = un;
            new_v[idx] = vn;
            x += 1;
        }
        z += 1;
    }

    SweState {
        h: new_h,
        u: new_u,
        v: new_v,
    }
}

/// First-order upwind advective derivative of `field` given the local velocity.
///
/// The gradient is taken from the upwind neighbour (the direction the flow
/// comes from), which keeps explicit advection stable. Returns
/// `u * dField/dx + v * dField/dz`.
#[expect(
    clippy::too_many_arguments,
    reason = "explicit stencil neighbours keep the hot loop allocation-free"
)]
fn upwind(
    flow_u: f32,
    flow_v: f32,
    center: f32,
    xm: f32,
    xp: f32,
    zm: f32,
    zp: f32,
    inv_dx: f32,
) -> f32 {
    let ddx = if flow_u > 0.0 {
        (center - xm) * inv_dx
    } else {
        (xp - center) * inv_dx
    };
    let ddz = if flow_v > 0.0 {
        (center - zm) * inv_dx
    } else {
        (zp - center) * inv_dx
    };
    flow_u * ddx + flow_v * ddz
}

/// Adds a depth perturbation at one cell (rain drop, wading splash, boat draft).
///
/// This is the only volume-changing operation; the added volume is exactly
/// `delta_depth * dx * dx`. An out-of-range cell is ignored so a stale source
/// index cannot panic. Returns `true` when the cell was in range and modified.
pub fn inject_depth(
    state: &mut SweState,
    cfg: SweConfig,
    x: u32,
    z: u32,
    delta_depth: f32,
) -> bool {
    match cfg.index(x, z) {
        Some(idx) if idx < state.h.len() => {
            state.h[idx] += delta_depth;
            true
        }
        _ => false,
    }
}

/// Adds a velocity impulse at one cell (a paddle stroke, a hull wake source).
///
/// Volume-preserving: it stirs the flow without adding water. An out-of-range
/// cell is ignored. Returns `true` when the cell was in range and modified.
pub fn inject_velocity(
    state: &mut SweState,
    cfg: SweConfig,
    x: u32,
    z: u32,
    delta_u: f32,
    delta_v: f32,
) -> bool {
    match cfg.index(x, z) {
        Some(idx) if idx < state.u.len() && idx < state.v.len() => {
            state.u[idx] += delta_u;
            state.v[idx] += delta_v;
            true
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::water::GRAVITY;

    const CFG: SweConfig = SweConfig {
        nx: 8,
        nz: 8,
        dx: 0.5,
        gravity: GRAVITY,
        damping: 0.02,
    };

    #[test]
    fn cell_wave_speed_uses_only_sqrt_and_clamps_dry_cells() {
        // Still cell: celerity = sqrt(g*h).
        let c = cell_wave_speed(2.0, 0.0, 0.0, 9.0);
        assert!((c - (9.0_f32 * 2.0).sqrt()).abs() < EPS);
        // Flow adds its magnitude.
        let c2 = cell_wave_speed(0.0, 3.0, 4.0, 9.0);
        assert!((c2 - 5.0).abs() < EPS);
        // Negative depth is treated as dry (no NaN).
        let c3 = cell_wave_speed(-1.0, 0.0, 0.0, 9.0);
        assert_eq!(c3, 0.0);
    }

    #[test]
    fn cfl_timestep_and_stability_agree() {
        let dx = 0.5;
        let cfl = 0.5;
        let speed = 4.0;
        let dt = cfl_timestep(speed, dx, cfl);
        assert!(is_cfl_stable(dt, dx, speed, cfl));
        // Anything larger is unstable.
        assert!(!is_cfl_stable(dt * 1.1, dx, speed, cfl));
        // Zero signal speed imposes no constraint.
        assert_eq!(cfl_timestep(0.0, dx, cfl), f32::MAX);
    }

    #[test]
    fn still_water_is_preserved_exactly() {
        let state = SweState::still(CFG, 1.5);
        let next = step(&state, CFG, 0.01);
        assert_eq!(next.h, state.h);
        assert_eq!(next.u, state.u);
        assert_eq!(next.v, state.v);
    }

    #[test]
    fn mass_is_conserved_under_reflective_walls() {
        let mut state = SweState::still(CFG, 1.0);
        // Seed a deterministic non-trivial bump and flow.
        for z in 0..CFG.nz {
            for x in 0..CFG.nx {
                let idx = CFG.index(x, z).unwrap();
                let fx = x as f32;
                let fz = z as f32;
                state.h[idx] = 1.0 + 0.05 * ((fx - 3.5) * (fx - 3.5) + (fz - 3.5) * (fz - 3.5));
                state.u[idx] = 0.01 * (fx - 3.5);
                state.v[idx] = -0.01 * (fz - 3.5);
            }
        }
        let initial = state.total_volume(CFG);
        let speed = max_wave_speed(&state, CFG);
        let dt = cfl_timestep(speed, CFG.dx, 0.4).min(0.005);
        let mut cur = state;
        for _ in 0..50 {
            cur = step(&cur, CFG, dt);
        }
        let final_volume = cur.total_volume(CFG);
        assert!(
            (final_volume - initial).abs() < 1e-3,
            "volume drift too large: {initial} -> {final_volume}"
        );
    }

    #[test]
    fn stepping_is_deterministic() {
        let mut state = SweState::still(CFG, 1.0);
        state.h[CFG.index(4, 4).unwrap()] = 1.4;
        let a = step(&state, CFG, 0.004);
        let b = step(&state, CFG, 0.004);
        assert_eq!(a, b);
    }

    #[test]
    fn depth_injection_changes_volume_by_exact_amount() {
        let mut state = SweState::still(CFG, 1.0);
        let before = state.total_volume(CFG);
        assert!(inject_depth(&mut state, CFG, 2, 3, 0.5));
        let after = state.total_volume(CFG);
        let cell_area = CFG.dx * CFG.dx;
        assert!((after - before - 0.5 * cell_area).abs() < EPS);
    }

    #[test]
    fn velocity_injection_preserves_volume() {
        let mut state = SweState::still(CFG, 1.0);
        let before = state.total_volume(CFG);
        assert!(inject_velocity(&mut state, CFG, 1, 1, 0.3, -0.2));
        assert_eq!(state.total_volume(CFG), before);
    }

    #[test]
    fn out_of_range_injection_is_ignored() {
        let mut state = SweState::still(CFG, 1.0);
        assert!(!inject_depth(&mut state, CFG, 99, 0, 1.0));
        assert!(!inject_velocity(&mut state, CFG, 0, 99, 1.0, 1.0));
        // State untouched.
        assert_eq!(state.total_volume(CFG), 1.0 * CFG.dx * CFG.dx * 64.0);
    }

    #[test]
    fn empty_grid_step_is_a_noop() {
        let cfg = SweConfig { nx: 0, ..CFG };
        let state = SweState::default();
        let next = step(&state, cfg, 0.01);
        assert_eq!(next, state);
    }
}
