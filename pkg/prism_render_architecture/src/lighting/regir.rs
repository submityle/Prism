//! World-space reservoir light grid (`ReGIR`): per-cell presampling that feeds
//! `ReSTIR` DI cheap, importance-aware initial candidates.
//!
//! With thousands of lights, generating `ReSTIR` DI's initial candidates by
//! scanning every light per pixel is too expensive. `ReGIR` (Boksansky et al.,
//! as shipped in `RTXDI`, and the same idea behind `UE`'s `MegaLights` light
//! grid) amortizes that cost: the world is diced into a coarse grid, and once
//! per frame each cell presamples the light set into a handful of small
//! reservoirs using the cell *center* as a stand-in shading point. At shading
//! time a pixel maps its world position to a cell and draws its initial
//! candidates straight from that cell's reservoirs — an `O(1)` lookup instead of
//! an `O(lights)` scan — already biased toward the lights that matter near that
//! point in space.
//!
//! This borrows the *form* of `RTXDI`/`MegaLights` grid presampling without
//! reusing any engine code. It is pure classical Monte Carlo: each cell slot is
//! a resampled-importance-sampling (`RIS`) reservoir over the light set, and the
//! stateless [`Rng`] makes the whole build deterministic for golden tests and a
//! future `GPU` twin.
//!
//! # Unbiased hand-off to `ReSTIR` DI
//! A finalized cell reservoir holds a light `y` with weight `W_cell` such that,
//! for *any* per-light function `g`, `E[g(y) · W_cell] = Σ_i g(i)` (standard
//! `RIS` unbiasedness over the enumerated light set). To reuse that draw as one
//! `ReSTIR` DI initial candidate at a pixel we must report the density `y` was
//! effectively drawn from. Folding `y` into a one-sample pixel reservoir with
//! target `p̂_pix` and finalizing yields `W_pix = (p̂_pix(y)/source_pdf)/p̂_pix(y)
//! = 1/source_pdf`, so the pixel estimate is `p̂_pix(y) · W_pix`. Matching that
//! to the cell identity `E[p̂_pix(y) · W_cell] = Σ_i p̂_pix(i)` forces
//!
//! ```text
//!     source_pdf = 1 / W_cell .
//! ```
//!
//! [`RegirGrid::candidate_at`] emits exactly that [`DiCandidate`], so streaming
//! `ReGIR` candidates through [`super::restir_di::stream_initial`] stays an
//! unbiased estimator of the full many-light sum at the pixel (verified by a
//! brute-force Monte Carlo test). No neural, learned, or data-driven components.

use alloc::vec::Vec;

use super::restir_di::{DiCandidate, DiReservoir};
use crate::particle::reservoir_sample::Rng;

/// Immutable description of the world-space grid: its origin, cell size, extent
/// in cells, and how many independent reservoir slots each cell presamples.
///
/// Slots are decorrelated presamples of the *same* cell: drawing from different
/// slots gives a pixel several near-independent initial candidates without
/// rescanning the light set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RegirConfig {
    /// World-space corner of cell `(0, 0, 0)` (minimum along every axis).
    pub grid_min: [f32; 3],
    /// Edge length of a (cubic) cell in world units (must be `> 0`).
    pub cell_size: f32,
    /// Number of cells along each axis (each must be `>= 1`).
    pub dims: [u32; 3],
    /// Independent reservoir slots presampled per cell (must be `>= 1`).
    pub reservoirs_per_cell: u16,
}

impl RegirConfig {
    /// Total number of cells (`dims.x · dims.y · dims.z`).
    #[must_use]
    pub fn cell_count(&self) -> usize {
        (self.dims[0] as usize) * (self.dims[1] as usize) * (self.dims[2] as usize)
    }

    /// Reservoir slots per cell as a `usize` (at least 1).
    #[must_use]
    pub fn slots(&self) -> usize {
        (self.reservoirs_per_cell as usize).max(1)
    }

    /// Total reservoir slots across the whole grid.
    #[must_use]
    pub fn slot_count(&self) -> usize {
        self.cell_count() * self.slots()
    }

    /// Integer cell coordinates containing `pos`, or `None` when `pos` lies
    /// outside the grid (so the caller can fall back to a global sample).
    #[must_use]
    pub fn cell_coords(&self, pos: [f32; 3]) -> Option<[u32; 3]> {
        if self.cell_size <= 0.0 || self.cell_size.is_nan() {
            return None;
        }
        let mut coords = [0u32; 3];
        for axis in 0..3 {
            let local = (pos[axis] - self.grid_min[axis]) / self.cell_size;
            if local < 0.0 || local.is_nan() {
                return None;
            }
            let idx = local as u32;
            if idx >= self.dims[axis] {
                return None;
            }
            coords[axis] = idx;
        }
        Some(coords)
    }

    /// World-space center of a cell by integer coordinates.
    #[must_use]
    pub fn cell_center(&self, coords: [u32; 3]) -> [f32; 3] {
        [
            self.grid_min[0] + (coords[0] as f32 + 0.5) * self.cell_size,
            self.grid_min[1] + (coords[1] as f32 + 0.5) * self.cell_size,
            self.grid_min[2] + (coords[2] as f32 + 0.5) * self.cell_size,
        ]
    }

    /// Row-major linear index of a cell (`x` fastest), assuming in-range coords.
    #[must_use]
    pub fn linear_index(&self, coords: [u32; 3]) -> usize {
        let (x, y, z) = (coords[0] as usize, coords[1] as usize, coords[2] as usize);
        let (dx, dy) = (self.dims[0] as usize, self.dims[1] as usize);
        (z * dy + y) * dx + x
    }

    /// Decomposes a linear cell index back into integer coordinates.
    #[must_use]
    pub fn coords_of(&self, linear: usize) -> [u32; 3] {
        let (dx, dy) = (self.dims[0] as usize, self.dims[1] as usize);
        let x = linear % dx;
        let y = (linear / dx) % dy;
        let z = linear / (dx * dy);
        [x as u32, y as u32, z as u32]
    }

    /// Linear cell index containing `pos`, or `None` when outside the grid.
    #[must_use]
    pub fn cell_index_of(&self, pos: [f32; 3]) -> Option<usize> {
        self.cell_coords(pos).map(|c| self.linear_index(c))
    }
}

/// The presampled grid: one finalized [`DiReservoir`] per cell slot.
///
/// Rebuild it once per frame with [`RegirGrid::rebuild`], then query initial
/// candidates with [`RegirGrid::candidate_at`] / [`RegirGrid::candidates_for`].
#[derive(Clone, Debug)]
pub struct RegirGrid {
    config: RegirConfig,
    /// `cell_count · slots` reservoirs, cell-major: slot `s` of cell `c` is at
    /// `c * slots + s`.
    cells: Vec<DiReservoir>,
}

impl RegirGrid {
    /// Allocates an empty grid for `config` (all slots empty until a rebuild).
    #[must_use]
    pub fn new(config: RegirConfig) -> Self {
        let mut cells = Vec::new();
        cells.resize(config.slot_count(), DiReservoir::empty());
        Self { config, cells }
    }

    /// The grid's configuration.
    #[must_use]
    pub fn config(&self) -> RegirConfig {
        self.config
    }

    /// The finalized reservoir for a cell slot, or `None` if out of range.
    #[must_use]
    pub fn slot(&self, cell: usize, slot: usize) -> Option<&DiReservoir> {
        if cell >= self.config.cell_count() || slot >= self.config.slots() {
            return None;
        }
        self.cells.get(cell * self.config.slots() + slot)
    }

    /// Rebuilds every cell slot by streaming the whole light set through `RIS`
    /// using the cell center as the stand-in shading point.
    ///
    /// * `light_count` — lights are enumerated `0..light_count`.
    /// * `target_at(center, light)` — the cell target `p̂_cell`: an unshadowed
    ///   contribution estimate of `light` at the cell `center` (power / falloff;
    ///   must be `>= 0`, and `> 0` wherever a light can contribute at a pixel in
    ///   that cell so the hand-off stays unbiased).
    /// * `source_pdf(light)` — the density each light is enumerated with. For a
    ///   full `0..light_count` sweep this is the uniform `1 / light_count`; a
    ///   caller presampling a culled subset passes that subset's pdf instead.
    ///
    /// Each of a cell's slots consumes fresh `rng` draws, so the slots are
    /// decorrelated presamples of the same cell. The build is deterministic in
    /// `rng`.
    pub fn rebuild<T, S>(
        &mut self,
        light_count: u32,
        mut target_at: T,
        mut source_pdf: S,
        rng: &mut Rng,
    ) where
        T: FnMut([f32; 3], u32) -> f32,
        S: FnMut(u32) -> f32,
    {
        let slots = self.config.slots();
        self.cells.clear();
        self.cells.reserve(self.config.slot_count());
        for cell in 0..self.config.cell_count() {
            let center = self.config.cell_center(self.config.coords_of(cell));
            for _slot in 0..slots {
                let mut reservoir = DiReservoir::empty();
                for light in 0..light_count {
                    reservoir.stream(
                        DiCandidate {
                            light_index: light,
                            target_pdf: target_at(center, light),
                            source_pdf: source_pdf(light),
                        },
                        rng.next_u01(),
                    );
                }
                reservoir.finalize();
                self.cells.push(reservoir);
            }
        }
    }

    /// Draws one `ReSTIR` DI initial candidate for `pos` from a chosen slot.
    ///
    /// Returns `None` when `pos` is outside the grid or the slot presampled no
    /// usable light (empty, or a non-positive `W`). On success the candidate's
    /// `source_pdf` is `1 / W_cell`, the density that keeps the downstream
    /// initial `RIS` unbiased (see the module docs). `pixel_target(light)` is
    /// the target `p̂_pix` evaluated at the real pixel, not the cell center.
    #[must_use]
    pub fn candidate_at<P>(
        &self,
        pos: [f32; 3],
        slot: usize,
        mut pixel_target: P,
    ) -> Option<DiCandidate>
    where
        P: FnMut(u32) -> f32,
    {
        let cell = self.config.cell_index_of(pos)?;
        let slot = slot % self.config.slots();
        let reservoir = self.cells.get(cell * self.config.slots() + slot)?;
        if reservoir.is_empty() {
            return None;
        }
        let w_cell = reservoir.reservoir.w;
        if w_cell <= 0.0 || w_cell.is_nan() {
            return None;
        }
        let light = reservoir.light_index();
        Some(DiCandidate {
            light_index: light,
            target_pdf: pixel_target(light),
            source_pdf: 1.0 / w_cell,
        })
    }

    /// Gathers up to `count` initial candidates for `pos` by cycling slots.
    ///
    /// Convenience wrapper over [`RegirGrid::candidate_at`] that fills a vector
    /// ready to hand to [`super::restir_di::stream_initial`]. Slots are visited
    /// round-robin starting at `rng`-chosen offset so repeated calls spread
    /// across the cell's presamples; empty slots are skipped. Returns fewer than
    /// `count` (possibly empty) when `pos` is outside the grid or slots are
    /// unpopulated.
    #[must_use]
    pub fn candidates_for<P>(
        &self,
        pos: [f32; 3],
        count: usize,
        rng: &mut Rng,
        mut pixel_target: P,
    ) -> Vec<DiCandidate>
    where
        P: FnMut(u32) -> f32,
    {
        let slots = self.config.slots();
        let mut out = Vec::with_capacity(count);
        if self.config.cell_index_of(pos).is_none() || count == 0 {
            return out;
        }
        let start = (rng.next_u32() as usize) % slots;
        for k in 0..count {
            let slot = (start + k) % slots;
            if let Some(c) = self.candidate_at(pos, slot, &mut pixel_target) {
                out.push(c);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn abs(x: f32) -> f32 {
        if x < 0.0 {
            -x
        } else {
            x
        }
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

    #[test]
    fn cell_coords_roundtrip_and_bounds() {
        let cfg = config([4, 4, 4], 1);
        // Inside: center of cell (0,0,0) is grid_min + 0.5.
        let c = cfg.cell_coords([-1.6, -1.9, -1.2]).unwrap();
        assert_eq!(c, [0, 0, 0]);
        assert_eq!(cfg.cell_center([0, 0, 0]), [-1.5, -1.5, -1.5]);
        // Linear <-> coords roundtrip over the whole grid.
        for linear in 0..cfg.cell_count() {
            let coords = cfg.coords_of(linear);
            assert_eq!(cfg.linear_index(coords), linear);
        }
        // Outside returns None on each face.
        assert!(cfg.cell_coords([-2.001, 0.0, 0.0]).is_none());
        assert!(cfg.cell_coords([2.0, 0.0, 0.0]).is_none()); // grid spans [-2, 2)
        assert!(cfg.cell_coords([0.0, 100.0, 0.0]).is_none());
    }

    #[test]
    fn rebuild_is_deterministic() {
        let lights = Lights::demo();
        let cfg = config([3, 3, 3], 2);
        let build = |seed| {
            let mut grid = RegirGrid::new(cfg);
            let mut rng = Rng::new(seed);
            grid.rebuild(
                lights.count(),
                |center, l| lights.contribution(l, center),
                |_| 1.0 / lights.count() as f32,
                &mut rng,
            );
            grid
        };
        let a = build(42);
        let b = build(42);
        for s in 0..cfg.slot_count() {
            assert_eq!(a.cells[s], b.cells[s]);
        }
    }

    #[test]
    fn candidate_reports_inverse_w_source_pdf() {
        let lights = Lights::demo();
        let cfg = config([3, 3, 3], 1);
        let mut grid = RegirGrid::new(cfg);
        let mut rng = Rng::new(7);
        grid.rebuild(
            lights.count(),
            |center, l| lights.contribution(l, center),
            |_| 1.0 / lights.count() as f32,
            &mut rng,
        );
        let pixel = [0.2, 0.1, 0.3];
        let cell = cfg.cell_index_of(pixel).unwrap();
        let reservoir = grid.slot(cell, 0).unwrap();
        let cand = grid
            .candidate_at(pixel, 0, |l| lights.contribution(l, pixel))
            .unwrap();
        assert_eq!(cand.light_index, reservoir.light_index());
        assert!(abs(cand.source_pdf - 1.0 / reservoir.reservoir.w) < 1e-6);
        assert!(abs(cand.target_pdf - lights.contribution(cand.light_index, pixel)) < 1e-6);
    }

    #[test]
    fn candidate_outside_grid_is_none() {
        let lights = Lights::demo();
        let cfg = config([2, 2, 2], 1);
        let mut grid = RegirGrid::new(cfg);
        let mut rng = Rng::new(1);
        grid.rebuild(
            lights.count(),
            |center, l| lights.contribution(l, center),
            |_| 1.0 / lights.count() as f32,
            &mut rng,
        );
        assert!(grid
            .candidate_at([50.0, 0.0, 0.0], 0, |l| lights
                .contribution(l, [50.0, 0.0, 0.0]))
            .is_none());
    }

    #[test]
    fn empty_light_set_yields_no_candidates() {
        let cfg = config([2, 2, 2], 2);
        let mut grid = RegirGrid::new(cfg);
        let mut rng = Rng::new(3);
        grid.rebuild(0, |_, _| 0.0, |_| 1.0, &mut rng);
        let pixel = [0.0, 0.0, 0.0];
        assert!(grid.candidate_at(pixel, 0, |_| 1.0).is_none());
        assert!(grid.candidates_for(pixel, 4, &mut rng, |_| 1.0).is_empty());
    }

    #[test]
    fn presampling_favors_bright_nearby_lights() {
        // Over many rebuilds the slot should hold the dominant light for a cell
        // far more often than a uniform 1/N pick would.
        let lights = Lights::demo();
        let cfg = config([3, 3, 3], 1);
        let pixel = [0.5, 0.5, 0.5]; // cell nearest the brightest light (light 0)
        let cell = cfg.cell_index_of(pixel).unwrap();
        let trials = 4000u32;
        let mut hits = 0u32;
        for s in 0..trials {
            let mut grid = RegirGrid::new(cfg);
            let mut rng = Rng::new(s.wrapping_mul(2_654_435_761).wrapping_add(1));
            grid.rebuild(
                lights.count(),
                |center, l| lights.contribution(l, center),
                |_| 1.0 / lights.count() as f32,
                &mut rng,
            );
            if grid.slot(cell, 0).unwrap().light_index() == 0 {
                hits += 1;
            }
        }
        let freq = hits as f32 / trials as f32;
        // Uniform would be ~1/6 ≈ 0.167; importance presampling must beat it.
        assert!(freq > 0.5, "dominant-light frequency {freq} too low");
    }

    #[test]
    fn regir_candidate_is_unbiased_for_restir_di() {
        // The core contract: a ReGIR candidate streamed into a one-sample pixel
        // reservoir is an unbiased estimator of the full many-light sum.
        let lights = Lights::demo();
        let cfg = config([3, 3, 3], 1);
        let pixel = [0.3, -0.2, 0.4];
        let exact = lights.sum_at(pixel);

        let trials = 300_000u32;
        let mut acc = 0.0f64;
        for s in 0..trials {
            let mut rng = Rng::new(s.wrapping_mul(40_503).wrapping_add(11));
            let mut grid = RegirGrid::new(cfg);
            grid.rebuild(
                lights.count(),
                |center, l| lights.contribution(l, center),
                |_| 1.0 / lights.count() as f32,
                &mut rng,
            );
            // One ReGIR candidate -> one-sample pixel reservoir -> finalize.
            if let Some(cand) = grid.candidate_at(pixel, 0, |l| lights.contribution(l, pixel)) {
                let mut r = DiReservoir::empty();
                r.stream(cand, rng.next_u01());
                r.finalize();
                acc += f64::from(r.target_pdf * r.reservoir.w);
            }
        }
        let mean = (acc / f64::from(trials)) as f32;
        let rel = abs(mean - exact) / exact;
        assert!(rel < 0.02, "ReGIR mean {mean} vs exact {exact} ({rel})");
    }

    #[test]
    fn candidates_for_fills_from_slots() {
        let lights = Lights::demo();
        let cfg = config([2, 2, 2], 4);
        let mut grid = RegirGrid::new(cfg);
        let mut rng = Rng::new(9);
        grid.rebuild(
            lights.count(),
            |center, l| lights.contribution(l, center),
            |_| 1.0 / lights.count() as f32,
            &mut rng,
        );
        let pixel = [-0.5, -0.5, -0.5];
        let cands = grid.candidates_for(pixel, 4, &mut rng, |l| lights.contribution(l, pixel));
        assert!(!cands.is_empty() && cands.len() <= 4);
        for c in &cands {
            assert!(c.source_pdf > 0.0);
            assert!((c.light_index as usize) < lights.pos.len());
        }
    }
}
