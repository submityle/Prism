//! Dynamic foam field: semi-Lagrangian advection, flow-aware decay with
//! persistence, and reactive foam injection.
//!
//! Foam is a scalar coverage field in `0..=1` carried on the same grid as the
//! surface flow. Each frame it is advected along the surface velocity by a
//! semi-Lagrangian backtrace (unconditionally stable, non-negativity
//! preserving), then decayed exponentially. Decay is flow-aware: churning,
//! fast-moving water dissipates foam quickly, while calm eddies and shorelines
//! hold it far longer (a `persistence_floor` keeps a slow residual decay even
//! at zero flow). Breaking crests and reactive contacts (a hull, wading feet)
//! inject fresh foam through source terms.
//!
//! All operations are pure and deterministic over caller-owned grids: fixed
//! row-major iteration, bilinear sampling, and the shared `exp_approx` for the
//! decay envelope (no `f32::exp`). Advection and injection clamp to the grid so
//! stale indices never panic.

use alloc::vec;
use alloc::vec::Vec;

use super::{exp_approx, EPS};

/// Grid layout and decay tuning for a foam field.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FoamConfig {
    /// Cell count along x (`>= 1`).
    pub nx: u32,
    /// Cell count along z (`>= 1`).
    pub nz: u32,
    /// Cell size in meters (`> 0`).
    pub dx: f32,
    /// Baseline decay rate per second at or above `reference_speed`.
    pub base_decay: f32,
    /// Fraction of `base_decay` that still applies in perfectly still water,
    /// in `0..=1`; smaller values make calm foam persist longer.
    pub persistence_floor: f32,
    /// Flow speed at which decay reaches the full `base_decay` (`> 0`).
    pub reference_speed: f32,
}

impl FoamConfig {
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

/// Flow-aware decay rate (per second) at a given local flow speed.
///
/// Interpolates from `base_decay * persistence_floor` in still water up to the
/// full `base_decay` at or beyond `reference_speed`. It rises monotonically
/// with flow speed, so calm water always decays no faster than churning water.
#[must_use]
pub fn foam_decay_rate(flow_speed: f32, cfg: FoamConfig) -> f32 {
    let floor = cfg.persistence_floor.clamp(0.0, 1.0);
    let reference = if cfg.reference_speed > EPS {
        cfg.reference_speed
    } else {
        EPS
    };
    let t = (flow_speed / reference).clamp(0.0, 1.0);
    cfg.base_decay * (floor + (1.0 - floor) * t)
}

/// Exponentially decays a foam density over `dt` at a decay `rate`.
///
/// Returns `density * exp(-rate * dt)` via the shared non-negative
/// `exp_approx`. The result is non-negative, never exceeds the input, and
/// decreases monotonically as either `dt` or `rate` grows.
#[must_use]
pub fn decay_foam(density: f32, rate: f32, dt: f32) -> f32 {
    let d = if density > 0.0 { density } else { 0.0 };
    d * exp_approx(-rate * dt)
}

/// Bilinearly samples a foam field at fractional cell coordinates.
///
/// Coordinates are clamped into `[0, nx-1] x [0, nz-1]`, so sampling never
/// reads out of bounds. Because the four taps are convex-combined, sampling a
/// non-negative field yields a non-negative result.
#[must_use]
pub fn sample_bilinear(field: &[f32], cfg: FoamConfig, px: f32, pz: f32) -> f32 {
    let nx = cfg.nx as usize;
    let nz = cfg.nz as usize;
    if nx == 0 || nz == 0 || field.len() < nx * nz {
        return 0.0;
    }
    let max_x = (nx - 1) as f32;
    let max_z = (nz - 1) as f32;
    let cx = px.clamp(0.0, max_x);
    let cz = pz.clamp(0.0, max_z);
    let x0 = cx as usize;
    let z0 = cz as usize;
    let x1 = (x0 + 1).min(nx - 1);
    let z1 = (z0 + 1).min(nz - 1);
    let fx = cx - (x0 as f32);
    let fz = cz - (z0 as f32);
    let s00 = field[z0 * nx + x0];
    let s10 = field[z0 * nx + x1];
    let s01 = field[z1 * nx + x0];
    let s11 = field[z1 * nx + x1];
    let top = s00 + (s10 - s00) * fx;
    let bottom = s01 + (s11 - s01) * fx;
    top + (bottom - top) * fz
}

/// Semi-Lagrangian advection of the foam field along the flow `(u, v)`.
///
/// Each cell backtraces its centre by `-dt * velocity` (converted to cell
/// units) and bilinearly samples the previous field there. This is
/// unconditionally stable and preserves non-negativity. A uniform field is
/// returned unchanged.
#[must_use]
pub fn advect_foam_field(
    density: &[f32],
    u: &[f32],
    v: &[f32],
    cfg: FoamConfig,
    dt: f32,
) -> Vec<f32> {
    let nx = cfg.nx as usize;
    let nz = cfg.nz as usize;
    let n = nx * nz;
    if n == 0 || density.len() < n || u.len() < n || v.len() < n {
        return density.to_vec();
    }
    let inv_dx = 1.0 / cfg.dx;
    let mut out = vec![0.0_f32; n];
    let mut z = 0;
    while z < nz {
        let mut x = 0;
        while x < nx {
            let idx = z * nx + x;
            let px = (x as f32) - dt * u[idx] * inv_dx;
            let pz = (z as f32) - dt * v[idx] * inv_dx;
            out[idx] = sample_bilinear(density, cfg, px, pz);
            x += 1;
        }
        z += 1;
    }
    out
}

/// Advances the foam field one step: advect, flow-aware decay, then add
/// sources, clamping coverage to `0..=1`.
///
/// `sources[i]` is the foam generated at cell `i` this step (breaking crests,
/// reactive contacts), already scaled by `dt`. The result is non-negative and
/// never exceeds full coverage.
#[must_use]
pub fn step_foam(
    density: &[f32],
    u: &[f32],
    v: &[f32],
    sources: &[f32],
    cfg: FoamConfig,
    dt: f32,
) -> Vec<f32> {
    let mut advected = advect_foam_field(density, u, v, cfg, dt);
    let n = advected.len();
    let mut i = 0;
    while i < n {
        let flow_speed = if i < u.len() && i < v.len() {
            (u[i] * u[i] + v[i] * v[i]).sqrt()
        } else {
            0.0
        };
        let rate = foam_decay_rate(flow_speed, cfg);
        let decayed = decay_foam(advected[i], rate, dt);
        let source = sources.get(i).copied().unwrap_or(0.0).max(0.0);
        advected[i] = (decayed + source).clamp(0.0, 1.0);
        i += 1;
    }
    advected
}

/// Injects reactive foam at one cell (hull contact, wading, splash-down).
///
/// Coverage is clamped to `0..=1`. An out-of-range cell is ignored so a stale
/// contact index cannot panic. Returns `true` when a cell was modified.
pub fn inject_foam(field: &mut [f32], cfg: FoamConfig, x: u32, z: u32, amount: f32) -> bool {
    match cfg.index(x, z) {
        Some(idx) if idx < field.len() => {
            field[idx] = (field[idx] + amount).clamp(0.0, 1.0);
            true
        }
        _ => false,
    }
}

/// Writes a visibility mask: `true` where foam coverage meets `threshold`.
///
/// Fills `min(field.len(), out.len())` entries so a mismatched buffer never
/// panics; returns the number written. The renderer uses this reactive mask to
/// gate the foam shading pass to cells that actually carry foam.
pub fn reactive_mask_into(field: &[f32], threshold: f32, out: &mut [bool]) -> usize {
    let count = field.len().min(out.len());
    let mut i = 0;
    while i < count {
        out[i] = field[i] >= threshold;
        i += 1;
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    const CFG: FoamConfig = FoamConfig {
        nx: 8,
        nz: 8,
        dx: 0.5,
        base_decay: 1.0,
        persistence_floor: 0.1,
        reference_speed: 2.0,
    };

    fn zeros() -> Vec<f32> {
        vec![0.0; CFG.cell_count()]
    }

    #[test]
    fn decay_rate_rises_monotonically_with_flow_speed() {
        let mut prev = foam_decay_rate(0.0, CFG);
        let mut speed = 0.0;
        while speed <= 4.0 {
            let rate = foam_decay_rate(speed, CFG);
            assert!(rate + EPS >= prev, "decay rate must not decrease");
            prev = rate;
            speed += 0.1;
        }
        // Still water uses the persistence floor; fast water the full rate.
        assert!((foam_decay_rate(0.0, CFG) - CFG.base_decay * CFG.persistence_floor).abs() < EPS);
        assert!((foam_decay_rate(10.0, CFG) - CFG.base_decay).abs() < EPS);
    }

    #[test]
    fn decay_is_monotonic_nonnegative_and_bounded_by_input() {
        let start = 0.8;
        let mut prev = start;
        let mut dt = 0.0;
        while dt <= 3.0 {
            let d = decay_foam(start, 1.0, dt);
            assert!(d >= 0.0, "foam density went negative");
            assert!(d <= start + EPS, "decay increased density");
            assert!(d <= prev + EPS, "decay not monotonic in dt");
            prev = d;
            dt += 0.1;
        }
    }

    #[test]
    fn calm_foam_persists_longer_than_churned_foam() {
        let calm = decay_foam(1.0, foam_decay_rate(0.0, CFG), 1.0);
        let churned = decay_foam(1.0, foam_decay_rate(5.0, CFG), 1.0);
        assert!(calm > churned, "calm foam should persist longer");
    }

    #[test]
    fn uniform_field_advects_to_itself() {
        let density = vec![0.4_f32; CFG.cell_count()];
        let u = vec![0.7_f32; CFG.cell_count()];
        let v = vec![-0.3_f32; CFG.cell_count()];
        let out = advect_foam_field(&density, &u, &v, CFG, 0.05);
        for &d in &out {
            assert!((d - 0.4).abs() < 1e-4, "uniform field drifted: {d}");
        }
    }

    #[test]
    fn advection_transports_a_bump_downstream_and_stays_nonnegative() {
        let mut density = zeros();
        let src = CFG.index(3, 4).unwrap();
        density[src] = 1.0;
        // Flow +x by exactly one cell over the step (dt*u/dx = 1).
        let u = vec![CFG.dx / 0.05; CFG.cell_count()];
        let v = zeros();
        let out = advect_foam_field(&density, &u, &v, CFG, 0.05);
        for &d in &out {
            assert!(d >= 0.0);
        }
        // Downstream cell (4,4) picks up the bump; upstream source empties.
        assert!(out[CFG.index(4, 4).unwrap()] > 0.5);
    }

    #[test]
    fn step_keeps_coverage_in_unit_range_and_applies_sources() {
        let density = zeros();
        let u = zeros();
        let v = zeros();
        let mut sources = zeros();
        sources[CFG.index(2, 2).unwrap()] = 0.6;
        let out = step_foam(&density, &u, &v, &sources, CFG, 0.1);
        for &d in &out {
            assert!((0.0..=1.0).contains(&d), "coverage out of range: {d}");
        }
        assert!(out[CFG.index(2, 2).unwrap()] > 0.0);
    }

    #[test]
    fn step_is_deterministic() {
        let mut density = zeros();
        density[CFG.index(4, 4).unwrap()] = 0.9;
        let u = vec![0.2_f32; CFG.cell_count()];
        let v = vec![0.1_f32; CFG.cell_count()];
        let sources = zeros();
        let a = step_foam(&density, &u, &v, &sources, CFG, 0.05);
        let b = step_foam(&density, &u, &v, &sources, CFG, 0.05);
        assert_eq!(a, b);
    }

    #[test]
    fn injection_clamps_and_ignores_out_of_range() {
        let mut field = zeros();
        assert!(inject_foam(&mut field, CFG, 1, 1, 0.5));
        assert!(inject_foam(&mut field, CFG, 1, 1, 0.9));
        assert_eq!(field[CFG.index(1, 1).unwrap()], 1.0);
        assert!(!inject_foam(&mut field, CFG, 99, 0, 1.0));
    }

    #[test]
    fn reactive_mask_thresholds_coverage() {
        let mut field = zeros();
        field[0] = 0.05;
        field[1] = 0.5;
        field[2] = 0.9;
        let mut mask = vec![false; CFG.cell_count()];
        let written = reactive_mask_into(&field, 0.25, &mut mask);
        assert_eq!(written, CFG.cell_count());
        assert!(!mask[0]);
        assert!(mask[1]);
        assert!(mask[2]);
        // Short buffer does not panic.
        let mut small = [false; 2];
        assert_eq!(reactive_mask_into(&field, 0.25, &mut small), 2);
    }
}
