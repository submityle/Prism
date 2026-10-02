//! End-to-end `ReSTIR` DI resolve driven by the `ReGIR` world-space light grid.
//!
//! [`super::regir`] presamples the light set into per-cell reservoirs and
//! [`super::restir_temporal::resolve_di`] runs the temporal + spatial
//! resampling pipeline, but on their own they leave one seam to the caller: the
//! initial candidates `resolve_di` resamples have to come from somewhere, and
//! for the hand-off to stay unbiased the target function `ReGIR` evaluates when
//! it emits a candidate (`p̂_pix`) must be *exactly* the one `resolve_di` uses at
//! that pixel. Wiring the two stages by hand invites a quiet bug: a caller that
//! passes an even slightly different target to
//! [`super::regir::RegirGrid::candidates_for`] than to
//! [`super::restir_temporal::resolve_di`] silently biases the estimate without
//! any test tripping.
//!
//! This module closes that seam with a single per-pixel entry point,
//! [`resolve_di_from_regir`], that threads **one** target closure through both
//! stages. It replaces `ReSTIR` DI's `O(lights)` initial scan with the grid's
//! `O(1)` importance-aware cell lookup and feeds the resulting candidates
//! straight into the full temporal / spatial resolve — the same structure
//! `RTXDI` and `UE`'s `MegaLights` use to shade thousands of lights in real
//! time, rebuilt here from scratch.
//!
//! # Unbiasedness
//! Two unbiased stages compose. The grid emits each initial candidate with
//! `source_pdf = 1 / W_cell`, the density that makes the pixel's initial `RIS`
//! an unbiased estimator of the full many-light sum (see [`super::regir`]); the
//! resolve then folds temporal history and spatial neighbors with the unbiased
//! normalization (see [`super::restir_di::combine_unbiased`]). Because the
//! single threaded target guarantees the candidate's `p̂_pix` equals the
//! resolve's target at the current surface, the composed estimator satisfies
//! `E[contribution(y) · W] = Σ_i contribution_i` at the pixel — verified below
//! by a brute-force Monte Carlo test against the exact many-light sum.
//!
//! Pure classical Monte Carlo, deterministic in the stateless [`Rng`]: no
//! neural, learned, or data-driven components. A given seed reproduces the same
//! selection for golden tests and a future `GPU` twin.

use super::regir::RegirGrid;
use super::restir_di::DiReservoir;
use super::restir_temporal::{resolve_di, GeomReservoir, SurfaceGeometry, TemporalParams};
use super::ReservoirBudget;
use crate::particle::reservoir_sample::Rng;

/// Resolves `ReSTIR` DI for one pixel, drawing its initial candidates from the
/// `ReGIR` grid at the pixel's world position.
///
/// This is the end-to-end driver: it gathers up to `budget.initial_candidates`
/// importance-aware candidates from the cell containing `position` (an `O(1)`
/// lookup instead of scanning every light), then runs the full temporal +
/// spatial resolve via [`super::restir_temporal::resolve_di`].
///
/// * `grid` — the frame's presampled `ReGIR` grid (built with
///   [`super::regir::RegirGrid::rebuild`] or
///   [`super::regir::RegirGrid::rebuild_temporal`]).
/// * `position` — the pixel's world-space shading point, mapped to a grid cell.
///   When it lies outside the grid the gathered candidate set is empty, so the
///   initial reservoir is empty and the result rests entirely on admissible
///   history / neighbors (or is itself empty when there are none) — the caller's
///   cue to fall back to a global sampler.
/// * `current` — the pixel's surface (depth / normal), used for reuse
///   admissibility and as the point at which the target is evaluated.
/// * `history` / `spatial` — the reprojected previous-frame reservoir and this
///   frame's neighbor reservoirs, forwarded unchanged to the resolve.
/// * `params` — temporal reprojection tolerances (see [`TemporalParams`]).
/// * `target` — the single target function `p̂(surface, light)` threaded to
///   **both** stages: `ReGIR` evaluates it at `current` to weight each
///   candidate, and the resolve re-evaluates it at every reuse surface. Passing
///   one closure is exactly what keeps the composed estimator unbiased.
#[must_use]
pub fn resolve_di_from_regir<T>(
    grid: &RegirGrid,
    position: [f32; 3],
    current: SurfaceGeometry,
    history: Option<GeomReservoir>,
    spatial: &[GeomReservoir],
    budget: ReservoirBudget,
    params: TemporalParams,
    mut target: T,
    rng: &mut Rng,
) -> DiReservoir
where
    T: FnMut(&SurfaceGeometry, u32) -> f32,
{
    // Initial candidates come from the grid cell at `position`, weighted by the
    // pixel's own target evaluated at `current` — the identical function the
    // resolve uses, so the `source_pdf = 1 / W_cell` hand-off stays consistent
    // and the whole chain is unbiased.
    let candidates =
        grid.candidates_for(position, budget.initial_candidates as usize, rng, |light| {
            target(&current, light)
        });

    resolve_di(
        current,
        &candidates,
        history,
        spatial,
        budget,
        params,
        target,
        rng,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lighting::regir::RegirConfig;

    const FRONT: [f32; 3] = [0.0, 0.0, 1.0];

    fn budget(initial: u16, spatial: u16, temporal: bool) -> ReservoirBudget {
        ReservoirBudget {
            initial_candidates: initial,
            spatial_neighbors: spatial,
            temporal_reuse: temporal,
        }
    }

    fn config(dims: [u32; 3], slots: u16) -> RegirConfig {
        RegirConfig {
            grid_min: [-2.0, -2.0, -2.0],
            cell_size: 1.0,
            dims,
            reservoirs_per_cell: slots,
        }
    }

    fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    }

    fn dist2(a: [f32; 3], b: [f32; 3]) -> f32 {
        let d = sub3(a, b);
        d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
    }

    // A small point-light rig: unshadowed contribution = power / (eps + dist^2).
    struct Lights {
        pos: Vec<[f32; 3]>,
        power: Vec<f32>,
    }

    impl Lights {
        fn demo() -> Self {
            Self {
                pos: Vec::from([
                    [1.0, 0.5, 0.0],
                    [-1.5, 1.0, 0.5],
                    [0.0, -1.0, 1.0],
                    [1.5, -1.5, -0.5],
                    [-0.5, 0.0, -1.0],
                    [0.5, 1.5, 0.5],
                ]),
                power: Vec::from([3.0, 1.0, 2.5, 0.8, 1.7, 2.2]),
            }
        }
        fn count(&self) -> u32 {
            self.pos.len() as u32
        }
        fn contribution(&self, light: u32, point: [f32; 3]) -> f32 {
            let i = light as usize;
            self.power[i] / (0.05 + dist2(self.pos[i], point))
        }
        fn sum_at(&self, point: [f32; 3]) -> f32 {
            (0..self.count()).map(|l| self.contribution(l, point)).sum()
        }
    }

    fn build_grid(lights: &Lights, cfg: RegirConfig, seed: u32) -> RegirGrid {
        let mut grid = RegirGrid::new(cfg);
        let mut rng = Rng::new(seed);
        grid.rebuild(
            lights.count(),
            |center, l| lights.contribution(l, center),
            |_| 1.0 / lights.count() as f32,
            &mut rng,
        );
        grid
    }

    #[test]
    fn regir_initial_only_is_unbiased() {
        // No history, no neighbors: the end-to-end estimate must still converge
        // to the exact many-light sum at the pixel, confirming the grid's
        // `source_pdf = 1 / W_cell` hand-off threads correctly into resolve_di.
        let lights = Lights::demo();
        let cfg = config([4, 4, 4], 4);
        // A shading point well inside the grid (cell center stand-in differs
        // from the pixel, which is the whole point of the hand-off).
        let pixel = [0.3, -0.2, 0.1];
        let geom = SurfaceGeometry::new(8.0, FRONT);
        let exact = lights.sum_at(pixel);
        let b = budget(4, 0, false);

        let seeds = 400_000u32;
        let mut acc = 0.0f64;
        for s in 0..seeds {
            // Rebuild the grid each seed so grid construction randomness is part
            // of the estimator being averaged (an unbiased pipeline must survive
            // that, not just a frozen grid).
            let grid = build_grid(&lights, cfg, s.wrapping_mul(2_654_435_761).wrapping_add(1));
            let mut rng = Rng::new(s.wrapping_mul(40_503).wrapping_add(7));
            let out = resolve_di_from_regir(
                &grid,
                pixel,
                geom,
                None,
                &[],
                b,
                TemporalParams::default(),
                |_g, l| lights.contribution(l, pixel),
                &mut rng,
            );
            acc += f64::from(out.target_pdf * out.reservoir.w);
        }
        let mean = (acc / f64::from(seeds)) as f32;
        let rel_err = (mean - exact).abs() / exact;
        assert!(rel_err < 0.02, "mean {mean} vs exact {exact} ({rel_err})");
    }

    #[test]
    fn regir_with_temporal_is_unbiased() {
        // Fold in a reprojected history reservoir (itself produced by the same
        // driver on the "previous frame"): the chain stays unbiased.
        let lights = Lights::demo();
        let cfg = config([4, 4, 4], 4);
        let pixel = [0.3, -0.2, 0.1];
        let geom = SurfaceGeometry::new(8.0, FRONT);
        let exact = lights.sum_at(pixel);
        let b = budget(4, 0, true);

        let seeds = 400_000u32;
        let mut acc = 0.0f64;
        for s in 0..seeds {
            let grid = build_grid(&lights, cfg, s.wrapping_mul(2_654_435_761).wrapping_add(1));
            let mut rng = Rng::new(s.wrapping_mul(40_503).wrapping_add(7));

            // Previous frame: run the driver with no history to get a finalized
            // reservoir on the same surface, then present it as this frame's
            // reprojected (geometrically matching) history.
            let prev = resolve_di_from_regir(
                &grid,
                pixel,
                geom,
                None,
                &[],
                budget(4, 0, false),
                TemporalParams::default(),
                |_g, l| lights.contribution(l, pixel),
                &mut rng,
            );
            let history = GeomReservoir::new(prev, geom);

            let out = resolve_di_from_regir(
                &grid,
                pixel,
                geom,
                Some(history),
                &[],
                b,
                TemporalParams::default(),
                |_g, l| lights.contribution(l, pixel),
                &mut rng,
            );
            acc += f64::from(out.target_pdf * out.reservoir.w);
        }
        let mean = (acc / f64::from(seeds)) as f32;
        let rel_err = (mean - exact).abs() / exact;
        assert!(rel_err < 0.02, "mean {mean} vs exact {exact} ({rel_err})");
    }

    #[test]
    fn regir_outside_grid_yields_empty_without_history() {
        // A pixel outside the grid gathers no candidates; with no history or
        // neighbors the result is an empty reservoir (W = 0), signalling the
        // caller to fall back to a global sampler.
        let lights = Lights::demo();
        let cfg = config([2, 2, 2], 2);
        let grid = build_grid(&lights, cfg, 99);
        let mut rng = Rng::new(123);
        let far = [100.0, 0.0, 0.0];
        let geom = SurfaceGeometry::new(8.0, FRONT);
        let out = resolve_di_from_regir(
            &grid,
            far,
            geom,
            None,
            &[],
            budget(4, 0, true),
            TemporalParams::default(),
            |_g, l| lights.contribution(l, far),
            &mut rng,
        );
        assert!(out.is_empty());
        assert_eq!(out.reservoir.w, 0.0);
    }

    #[test]
    fn regir_resolve_is_deterministic() {
        // Same seed and inputs reproduce the same finalized reservoir — the
        // contract a golden test and a future GPU twin depend on.
        let lights = Lights::demo();
        let cfg = config([4, 4, 4], 4);
        let grid = build_grid(&lights, cfg, 7);
        let pixel = [0.3, -0.2, 0.1];
        let geom = SurfaceGeometry::new(8.0, FRONT);
        let run = || {
            let mut rng = Rng::new(2024);
            resolve_di_from_regir(
                &grid,
                pixel,
                geom,
                None,
                &[],
                budget(4, 0, true),
                TemporalParams::default(),
                |_g, l| lights.contribution(l, pixel),
                &mut rng,
            )
        };
        assert_eq!(run(), run());
    }
}
