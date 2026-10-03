//! Geometric `multigrid` Poisson solver for the `FLIP`/`APIC` pressure projection.
//!
//! Removing divergence from a large liquid velocity field means solving a
//! Poisson system `A p = b`, where `A` is the discrete negative Laplacian on the
//! `MAC` grid and `b` is the scaled divergence. On big domains a plain `Jacobi`
//! or `Conjugate-Gradient` solve stalls on the smooth (low-frequency) error;
//! only a grid hierarchy converges in a bounded number of passes. This is the
//! `Houdini`/`Zhu-Bridson`-grade core that [`super::flip`] routes to when it
//! selects [`super::flip::PressureSolver::Multigrid`].
//!
//! The hierarchy is vertex-centred with standard coarsening: each level has
//! `2^L + 1` nodes per axis, the boundary layer holds a `Dirichlet` `p = 0`
//! condition (the free-surface/air boundary of a liquid), and a V-cycle does
//! pre-smooth, restrict the residual, recurse, prolong the correction, and
//! post-smooth. The smoother is damped `Jacobi`, restriction is full-weighting,
//! and prolongation is trilinear — the textbook operators, implemented as pure
//! deterministic functions over flat `f32` buffers.
//!
//! Only `sqrt` is used (for the residual norm). There are no `f32` equality
//! tests and no AI/ML. Degenerate grid sizes return an empty solve rather than
//! panicking.

use super::EPS;
use alloc::vec;
use alloc::vec::Vec;

/// Tuning for the V-cycle schedule.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MultigridConfig {
    /// Damped-`Jacobi` relaxation factor (`0 < omega < 1`).
    pub omega: f32,
    /// Pre-smoothing sweeps on the way down each level.
    pub pre_smooth: u32,
    /// Post-smoothing sweeps on the way up each level.
    pub post_smooth: u32,
    /// Sweeps at the coarsest level (effectively a direct solve).
    pub coarse_smooth: u32,
    /// Maximum V-cycles before giving up.
    pub max_cycles: u32,
    /// Residual L2-norm target that ends the solve early.
    pub tolerance: f32,
}

impl MultigridConfig {
    /// A balanced default: `omega = 0.8`, two-and-two smoothing.
    #[must_use]
    pub const fn balanced() -> Self {
        Self {
            omega: 0.8,
            pre_smooth: 2,
            post_smooth: 2,
            coarse_smooth: 24,
            max_cycles: 40,
            tolerance: 1e-5,
        }
    }

    /// Returns `true` when every field is in a usable range.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.omega > EPS && self.omega < 1.0 + EPS && self.max_cycles > 0 && self.tolerance > 0.0
    }
}

/// Result of a Poisson solve.
#[derive(Clone, Debug, PartialEq)]
pub struct SolveReport {
    /// The solved pressure field, flat `z*n*n + y*n + x`, boundary `0`.
    pub pressure: Vec<f32>,
    /// V-cycles actually run.
    pub cycles: u32,
    /// Final residual L2-norm.
    pub residual: f32,
}

/// Returns `true` when `n` is a valid vertex-centred level size (`2^L + 1`).
#[must_use]
pub fn is_valid_level_size(n: usize) -> bool {
    if n < 3 {
        return false;
    }
    let m = n - 1;
    m.is_power_of_two()
}

#[inline]
fn idx(n: usize, x: usize, y: usize, z: usize) -> usize {
    (z * n + y) * n + x
}

/// Apply the negative-Laplacian operator `A p` on an interior node grid.
///
/// Interior nodes use the 7-point stencil `(6 p_c - sum of 6 neighbours) / h^2`;
/// boundary nodes return `0` because they are fixed `Dirichlet` values, not
/// unknowns.
#[must_use]
pub fn apply_operator(p: &[f32], n: usize, h: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; n * n * n];
    if n < 3 || p.len() != n * n * n {
        return out;
    }
    let inv_h2 = 1.0 / (h * h);
    for z in 1..n - 1 {
        for y in 1..n - 1 {
            for x in 1..n - 1 {
                let c = idx(n, x, y, z);
                let s = 6.0 * p[c]
                    - p[idx(n, x - 1, y, z)]
                    - p[idx(n, x + 1, y, z)]
                    - p[idx(n, x, y - 1, z)]
                    - p[idx(n, x, y + 1, z)]
                    - p[idx(n, x, y, z - 1)]
                    - p[idx(n, x, y, z + 1)];
                out[c] = s * inv_h2;
            }
        }
    }
    out
}

/// Residual `r = b - A p`, zero on the boundary layer.
#[must_use]
pub fn residual(p: &[f32], b: &[f32], n: usize, h: f32) -> Vec<f32> {
    let ap = apply_operator(p, n, h);
    let mut r = vec![0.0f32; n * n * n];
    if b.len() != n * n * n {
        return r;
    }
    for z in 1..n - 1 {
        for y in 1..n - 1 {
            for x in 1..n - 1 {
                let c = idx(n, x, y, z);
                r[c] = b[c] - ap[c];
            }
        }
    }
    r
}

/// L2 norm of a field.
#[must_use]
pub fn l2_norm(field: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for &v in field {
        acc += v * v;
    }
    acc.sqrt()
}

/// Damped-`Jacobi` smoothing in place for `iters` sweeps.
pub fn smooth(p: &mut [f32], b: &[f32], n: usize, h: f32, omega: f32, iters: u32) {
    if n < 3 || p.len() != n * n * n || b.len() != n * n * n {
        return;
    }
    let factor = omega * h * h / 6.0;
    for _ in 0..iters {
        let r = residual(p, b, n, h);
        for z in 1..n - 1 {
            for y in 1..n - 1 {
                for x in 1..n - 1 {
                    let c = idx(n, x, y, z);
                    p[c] += factor * r[c];
                }
            }
        }
    }
}

/// Coarse level size for a fine size `nf` (`(nf - 1) / 2 + 1`).
#[must_use]
pub fn coarse_size(nf: usize) -> usize {
    (nf - 1) / 2 + 1
}

/// Full-weighting restriction of a fine field onto the coarse grid.
#[must_use]
pub fn restrict_full_weighting(fine: &[f32], nf: usize) -> Vec<f32> {
    let nc = coarse_size(nf);
    let mut out = vec![0.0f32; nc * nc * nc];
    if fine.len() != nf * nf * nf {
        return out;
    }
    let w = [0.25f32, 0.5, 0.25];
    for cz in 1..nc - 1 {
        for cy in 1..nc - 1 {
            for cx in 1..nc - 1 {
                let (fx, fy, fz) = (2 * cx, 2 * cy, 2 * cz);
                let mut acc = 0.0f32;
                for (dz, wz) in w.iter().enumerate() {
                    for (dy, wy) in w.iter().enumerate() {
                        for (dx, wx) in w.iter().enumerate() {
                            let sx = fx + dx - 1;
                            let sy = fy + dy - 1;
                            let sz = fz + dz - 1;
                            acc += wx * wy * wz * fine[idx(nf, sx, sy, sz)];
                        }
                    }
                }
                out[idx(nc, cx, cy, cz)] = acc;
            }
        }
    }
    out
}

/// Per-axis coarse contributors `(coarse_index, weight)` for a fine index.
#[inline]
fn contributors(i: usize) -> [(usize, f32); 2] {
    if i.is_multiple_of(2) {
        [(i / 2, 1.0), (i / 2, 0.0)]
    } else {
        [((i - 1) / 2, 0.5), (i.div_ceil(2), 0.5)]
    }
}

/// Trilinear prolongation of a coarse field onto the fine grid.
#[must_use]
pub fn prolong_trilinear(coarse: &[f32], nc: usize) -> Vec<f32> {
    let nf = (nc - 1) * 2 + 1;
    let mut out = vec![0.0f32; nf * nf * nf];
    if coarse.len() != nc * nc * nc {
        return out;
    }
    for z in 0..nf {
        let cz = contributors(z);
        for y in 0..nf {
            let cy = contributors(y);
            for x in 0..nf {
                let cx = contributors(x);
                let mut acc = 0.0f32;
                for &(iz, wz) in &cz {
                    if wz <= 0.0 {
                        continue;
                    }
                    for &(iy, wy) in &cy {
                        if wy <= 0.0 {
                            continue;
                        }
                        for &(ix, wx) in &cx {
                            if wx <= 0.0 {
                                continue;
                            }
                            acc += wx * wy * wz * coarse[idx(nc, ix, iy, iz)];
                        }
                    }
                }
                out[idx(nf, x, y, z)] = acc;
            }
        }
    }
    out
}

/// Recursive V-cycle: smooth, restrict residual, recurse, correct, smooth.
fn v_cycle(p: &mut [f32], b: &[f32], n: usize, h: f32, cfg: MultigridConfig) {
    if n <= 3 {
        smooth(p, b, n, h, cfg.omega, cfg.coarse_smooth);
        return;
    }
    smooth(p, b, n, h, cfg.omega, cfg.pre_smooth);
    let r = residual(p, b, n, h);
    let rc = restrict_full_weighting(&r, n);
    let nc = coarse_size(n);
    let mut ec = vec![0.0f32; nc * nc * nc];
    v_cycle(&mut ec, &rc, nc, h * 2.0, cfg);
    let ef = prolong_trilinear(&ec, nc);
    for z in 1..n - 1 {
        for y in 1..n - 1 {
            for x in 1..n - 1 {
                let c = idx(n, x, y, z);
                p[c] += ef[c];
            }
        }
    }
    smooth(p, b, n, h, cfg.omega, cfg.post_smooth);
}

/// Solve `A p = b` on an `n^3` vertex grid by repeated V-cycles.
///
/// Returns the pressure field (boundary `0`), the number of V-cycles run, and
/// the final residual L2-norm. An invalid grid size or config returns an empty
/// field with zero cycles.
#[must_use]
pub fn solve(b: &[f32], n: usize, h: f32, cfg: MultigridConfig) -> SolveReport {
    if !is_valid_level_size(n) || !cfg.is_valid() || b.len() != n * n * n || h <= EPS {
        return SolveReport {
            pressure: Vec::new(),
            cycles: 0,
            residual: 0.0,
        };
    }
    let mut p = vec![0.0f32; n * n * n];
    let mut cycles = 0u32;
    let mut res = l2_norm(&residual(&p, b, n, h));
    while cycles < cfg.max_cycles && res > cfg.tolerance {
        v_cycle(&mut p, b, n, h, cfg);
        res = l2_norm(&residual(&p, b, n, h));
        cycles += 1;
    }
    SolveReport {
        pressure: p,
        cycles,
        residual: res,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CFG: MultigridConfig = MultigridConfig::balanced();

    fn h_for(n: usize) -> f32 {
        1.0 / ((n - 1) as f32)
    }

    #[test]
    fn level_size_validation() {
        assert!(is_valid_level_size(3));
        assert!(is_valid_level_size(5));
        assert!(is_valid_level_size(9));
        assert!(is_valid_level_size(17));
        assert!(!is_valid_level_size(4));
        assert!(!is_valid_level_size(2));
        assert!(!is_valid_level_size(10));
    }

    #[test]
    fn coarse_size_halves_the_spacing() {
        assert_eq!(coarse_size(9), 5);
        assert_eq!(coarse_size(5), 3);
        assert_eq!(coarse_size(17), 9);
    }

    #[test]
    fn zero_rhs_stays_zero() {
        let n = 9;
        let b = vec![0.0f32; n * n * n];
        let rep = solve(&b, n, h_for(n), CFG);
        assert!(rep.residual < EPS);
        assert!(rep.pressure.iter().all(|&v| v.abs() < EPS));
    }

    #[test]
    fn residual_is_rhs_minus_operator() {
        let n = 5;
        let h = h_for(n);
        let mut p = vec![0.0f32; n * n * n];
        p[idx(n, 2, 2, 2)] = 1.0;
        let b = vec![0.3f32; n * n * n];
        let r = residual(&p, &b, n, h);
        let ap = apply_operator(&p, n, h);
        let c = idx(n, 2, 2, 2);
        assert!((r[c] - (b[c] - ap[c])).abs() < 1e-5);
        // Boundary residual stays zero.
        assert!(r[idx(n, 0, 0, 0)].abs() < EPS);
    }

    #[test]
    fn smoother_reduces_the_residual() {
        let n = 9;
        let h = h_for(n);
        let b = vec![1.0f32; n * n * n];
        let mut p = vec![0.0f32; n * n * n];
        let before = l2_norm(&residual(&p, &b, n, h));
        smooth(&mut p, &b, n, h, CFG.omega, 10);
        let after = l2_norm(&residual(&p, &b, n, h));
        assert!(after < before, "jacobi must reduce residual");
    }

    #[test]
    fn single_vcycle_reduces_residual() {
        let n = 9;
        let h = h_for(n);
        let b = vec![1.0f32; n * n * n];
        let mut p = vec![0.0f32; n * n * n];
        let before = l2_norm(&residual(&p, &b, n, h));
        v_cycle(&mut p, &b, n, h, CFG);
        let after = l2_norm(&residual(&p, &b, n, h));
        assert!(after < 0.5 * before, "one v-cycle must cut residual hard");
    }

    #[test]
    fn restriction_preserves_a_constant() {
        let nf = 9;
        let fine = vec![1.0f32; nf * nf * nf];
        let rc = restrict_full_weighting(&fine, nf);
        let nc = coarse_size(nf);
        // Interior coarse nodes see a full 3x3x3 of ones; weights sum to 1.
        assert!((rc[idx(nc, 2, 2, 2)] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn prolongation_is_partition_of_unity() {
        let nc = 5;
        let coarse = vec![1.0f32; nc * nc * nc];
        let fine = prolong_trilinear(&coarse, nc);
        assert!(fine.iter().all(|&v| (v - 1.0).abs() < 1e-5));
    }

    #[test]
    fn vcycle_converges_below_tolerance() {
        let n = 9;
        let b = vec![1.0f32; n * n * n];
        let rep = solve(&b, n, h_for(n), CFG);
        assert!(rep.residual <= CFG.tolerance, "residual {}", rep.residual);
        assert!(rep.cycles > 0 && rep.cycles <= CFG.max_cycles);
    }

    #[test]
    fn solver_recovers_a_manufactured_solution() {
        let n = 9;
        let h = h_for(n);
        // Method of manufactured solutions: a smooth bump that vanishes on the
        // boundary. A high-frequency field is adversarial for any
        // bounded-cycle multigrid, so (as is standard) the manufactured
        // reference is smooth and `Dirichlet`-compatible.
        let denom = (n - 1) as f32;
        let mut exact = vec![0.0f32; n * n * n];
        for z in 1..n - 1 {
            let zf = z as f32 / denom;
            for y in 1..n - 1 {
                let yf = y as f32 / denom;
                for x in 1..n - 1 {
                    let xf = x as f32 / denom;
                    let v = (xf * (1.0 - xf)) * (yf * (1.0 - yf)) * (zf * (1.0 - zf));
                    exact[idx(n, x, y, z)] = v;
                }
            }
        }
        let b = apply_operator(&exact, n, h);
        let rep = solve(&b, n, h, CFG);
        assert!(rep.residual <= CFG.tolerance);
        let mut max_err = 0.0f32;
        for (a, e) in rep.pressure.iter().zip(exact.iter()) {
            max_err = max_err.max((a - e).abs());
        }
        assert!(max_err < 1e-2, "max error {max_err}");
    }

    #[test]
    fn invalid_input_returns_empty() {
        let n = 10; // not 2^L + 1
        let b = vec![1.0f32; n * n * n];
        let rep = solve(&b, n, 0.1, CFG);
        assert!(rep.pressure.is_empty());
        assert_eq!(rep.cycles, 0);
    }

    #[test]
    fn solve_is_deterministic() {
        let n = 9;
        let b = vec![0.7f32; n * n * n];
        let a = solve(&b, n, h_for(n), CFG);
        let c = solve(&b, n, h_for(n), CFG);
        assert_eq!(a, c);
    }
}
