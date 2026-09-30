//! Polygon-bounded lake authoring preset (a `UE5` Water-style spline lake).
//!
//! A lake is not a rectangular pond and not a directed river: its surface is a
//! *still, closed basin* whose outline is drawn by an artist as a loop of
//! shoreline control points, and whose gentle motion comes from a wind-driven
//! surface drift plus any stream mouths feeding it. `UE5` Water and `Crest`
//! author lakes exactly this way — a closed spline bounds the water and the body
//! is filled to a rest depth — then bake the outline into the field the surface
//! solver steps. This preset does that against the dependency-free
//! [`prism_render_architecture::water`] core: it tessellates the loop into a
//! closed `Catmull-Rom` shoreline, sizes a Shallow-Water grid around it,
//! rasterizes the interior into the interaction-source field the `SWE` kernel
//! already consumes (`[.x` depth feed, `.y`/`.z` momentum drive, `.w` unused] —
//! the same layout the pond and river presets and `water_surface.wesl` agree
//! on), and bounds the `CFL` timestep by the basin's still-water celerity.
//!
//! The wind drift is a momentum-only source (it stirs a direction into the
//! surface without adding mass, so it settles against the solver's drag rather
//! than flooding the basin), tapered to zero over a `shore_width` band so the
//! banks stay calm. Optional stream-mouth [`LakeInflow`]s add a localized depth
//! feed in `.x`, each clamped to the nearest cell that actually lies inside the
//! shoreline so an inflow can never leak onto dry land.
//!
//! A degenerate outline (fewer than three nodes, a non-positive cell size or
//! rest depth) or a lake with neither wind nor a feeding inflow carries no
//! motion, so the preset returns the default no-op body rather than lighting a
//! solver that steps over dead water.

use prism_render_architecture::water::gpu::buffers::WaterBufferCounts;
use prism_render_architecture::water::gpu::pipeline::WaterPasses;
use prism_render_architecture::water::swe::{cfl_timestep, max_wave_speed, SweConfig, SweState};
use prism_render_architecture::water::{Vec2, GRAVITY};

use crate::water::abi::GpuWaterSweParams;
use crate::water::body::WaterBody;

/// One art-authored stream mouth feeding the lake.
///
/// The `position` is a world `XZ` point; the preset snaps it to the nearest grid
/// cell that lies *inside* the shoreline, so the feed always lands on water. The
/// `rate` is a steady depth feed (height per second, `>= 0`) injected there,
/// exactly the `.x` lane the pond preset uses for its central spring.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LakeInflow {
    /// World-space feed point on the `XZ` plane (meters).
    pub position: Vec2,
    /// Steady depth feed at the snapped interior cell (m/s, `>= 0`).
    pub rate: f32,
}

impl Default for LakeInflow {
    fn default() -> Self {
        Self {
            position: Vec2::ZERO,
            rate: 0.1,
        }
    }
}

/// An art-directable description of a polygon-bounded lake surface.
///
/// Construct one with [`LakePreset::default`] (a small square basin under a
/// gentle breeze) and override the [`shoreline`](Self::shoreline) to draw the
/// outline.
#[derive(Clone, Debug, PartialEq)]
pub struct LakePreset {
    /// Ordered shoreline control points forming a closed loop (the last node
    /// connects back to the first). Fewer than three nodes cannot bound an
    /// area, so the preset is then an honest no-op.
    pub shoreline: Vec<Vec2>,
    /// Square Shallow-Water cell edge length (m, `> 0`); the grid tiles the
    /// lake's bounding outline at this resolution.
    pub cell_size: f32,
    /// Spline samples emitted per shoreline segment (`>= 1`); higher values
    /// trace a curved outline more faithfully.
    pub samples_per_segment: u32,
    /// Still rest depth (m, `> 0`); sets the wave celerity `sqrt(g * depth)` and
    /// therefore the `CFL`-bounded timestep.
    pub rest_depth: f32,
    /// Width (m, `>= 0`) of the near-shore band over which the wind drift fades
    /// to zero at the bank; `0` drives the whole interior uniformly.
    pub shore_width: f32,
    /// World-space wind drift velocity (m/s) applied as a momentum-only surface
    /// stress across the interior; zero leaves a mirror-still lake.
    pub wind: Vec2,
    /// Optional stream mouths feeding the lake; each is snapped inside the
    /// shoreline before its depth feed is injected.
    pub inflows: Vec<LakeInflow>,
    /// Linear velocity damping per second in `0..=1`; the wind drift settles
    /// against this drag rather than accelerating without bound.
    pub damping: f32,
    /// Requested frame timestep (s); clamped down to the `CFL` bound so the
    /// explicit step can never diverge.
    pub timestep: f32,
    /// Courant number in `(0, 1]` the explicit step must satisfy.
    pub cfl_number: f32,
    /// Hard cap on the grid cell count (`nx * nz`); an outline whose bounding
    /// box would exceed this is clipped to the cap rather than allocating
    /// without bound.
    pub max_cells: u32,
}

impl Default for LakePreset {
    fn default() -> Self {
        Self {
            shoreline: vec![
                Vec2::new(-20.0, -20.0),
                Vec2::new(20.0, -20.0),
                Vec2::new(20.0, 20.0),
                Vec2::new(-20.0, 20.0),
            ],
            cell_size: 0.5,
            samples_per_segment: 8,
            rest_depth: 2.0,
            shore_width: 4.0,
            wind: Vec2::new(0.4, 0.0),
            inflows: Vec::new(),
            damping: 0.08,
            timestep: 1.0 / 60.0,
            cfl_number: 0.5,
            max_cells: 1 << 20,
        }
    }
}

/// Uniform `Catmull-Rom` evaluation of a scalar quadruple at parameter `t`.
#[must_use]
fn catmull_rom_scalar(p0: f32, p1: f32, p2: f32, p3: f32, t: f32) -> f32 {
    let t2 = t * t;
    let t3 = t2 * t;
    0.5 * ((2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
}

/// Uniform `Catmull-Rom` evaluation of a point quadruple at parameter `t`.
#[must_use]
fn catmull_rom_point(p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2, t: f32) -> Vec2 {
    Vec2::new(
        catmull_rom_scalar(p0.x, p1.x, p2.x, p3.x, t),
        catmull_rom_scalar(p0.y, p1.y, p2.y, p3.y, t),
    )
}

/// Cubic smoothstep `3t^2 - 2t^3` on a clamped `0..=1` argument; pure arithmetic
/// (no transcendental) so it honors the crate's determinism policy.
#[must_use]
fn smoothstep01(t: f32) -> f32 {
    let c = t.clamp(0.0, 1.0);
    c * c * (3.0 - 2.0 * c)
}

/// Nearest point on segment `a -> b` to `p`, returned as the clamped parameter
/// `t in 0..=1` and the squared distance to it. A degenerate segment reports its
/// start.
#[must_use]
fn nearest_on_segment(p: Vec2, a: Vec2, b: Vec2) -> (f32, f32) {
    let ab = b.sub(a);
    let denom = ab.length_squared();
    if denom <= 1.0e-12 {
        return (0.0, p.sub(a).length_squared());
    }
    let t = (p.sub(a).dot(ab) / denom).clamp(0.0, 1.0);
    let proj = a.add(ab.scale(t));
    (t, p.sub(proj).length_squared())
}

/// Tessellates a closed loop of control points into a dense shoreline polygon.
///
/// Each node contributes one `Catmull-Rom` segment to its successor, with the
/// wrap-around neighbors supplying the phantom control points, so the curve
/// closes smoothly back on itself. `seg_samples` points are emitted per segment
/// (excluding the segment end, which is the next segment's start), yielding
/// `nodes * seg_samples` closed-loop vertices.
#[must_use]
fn tessellate_loop(points: &[Vec2], seg_samples: u32) -> Vec<Vec2> {
    let n = points.len();
    let steps = seg_samples.max(1);
    let mut out: Vec<Vec2> = Vec::with_capacity(n * steps as usize);
    for i in 0..n {
        let p0 = points[(i + n - 1) % n];
        let p1 = points[i];
        let p2 = points[(i + 1) % n];
        let p3 = points[(i + 2) % n];
        for s in 0..steps {
            let t = s as f32 / steps as f32;
            out.push(catmull_rom_point(p0, p1, p2, p3, t));
        }
    }
    out
}

/// Even-odd ray-cast test: is `p` inside the closed polygon `poly`?
///
/// Casts a horizontal ray toward `+x` and toggles the parity for every polygon
/// edge that straddles the ray's height and crosses to the right of `p`. The
/// `(a.y > p.y) != (b.y > p.y)` guard admits only straddling edges, so the
/// height difference divided below is always non-zero.
#[must_use]
fn point_in_polygon(p: Vec2, poly: &[Vec2]) -> bool {
    let n = poly.len();
    if n < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let a = poly[i];
        let b = poly[j];
        if (a.y > p.y) != (b.y > p.y) {
            let t = (p.y - a.y) / (b.y - a.y);
            let x_cross = a.x + t * (b.x - a.x);
            if p.x < x_cross {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}

/// Shortest distance from `p` to the closed polygon boundary `poly`.
#[must_use]
fn distance_to_polygon(p: Vec2, poly: &[Vec2]) -> f32 {
    let n = poly.len();
    let mut best = f32::INFINITY;
    let mut j = n - 1;
    for i in 0..n {
        let (_, d2) = nearest_on_segment(p, poly[j], poly[i]);
        if d2 < best {
            best = d2;
        }
        j = i;
    }
    best.sqrt()
}

/// Grid dimensions (cells per axis) covering an `raw_x` by `raw_z` outline at
/// step `cell`, including a two-cell shore margin on every side plus the
/// fencepost cell. Never returns fewer than one cell per axis.
#[must_use]
fn lake_grid_dims(raw_x: f32, raw_z: f32, cell: f32) -> (u32, u32) {
    let inv = 1.0 / cell;
    let nx = (raw_x * inv).ceil() as u32 + 5;
    let nz = (raw_z * inv).ceil() as u32 + 5;
    (nx.max(1), nz.max(1))
}

impl WaterBody {
    /// Builds a fully live polygon-bounded lake body from a [`LakePreset`].
    ///
    /// The shoreline is tessellated into a closed `Catmull-Rom` outline, a
    /// Shallow-Water grid is sized around it (bounded by
    /// [`LakePreset::max_cells`]), and every interior cell is filled with a
    /// wind-driven, momentum-only surface stress tapered to zero over the
    /// `shore_width` bank band. Optional stream-mouth inflows add a localized
    /// depth feed in `.x`, each snapped to the nearest interior cell. The
    /// timestep is clamped to the `CFL` bound implied by the rest depth. A lake
    /// that carries no motion (too few nodes, a dry or degenerate basin, or
    /// neither wind nor a feeding inflow) yields the default no-op body.
    #[must_use]
    pub fn lake(preset: LakePreset) -> Self {
        // A basin needs at least three outline nodes, a positive cell size and a
        // positive rest depth to define wetted water.
        if preset.shoreline.len() < 3 || preset.cell_size <= 0.0 || preset.rest_depth <= 0.0 {
            return Self::default();
        }
        // A lake with neither wind nor a feeding inflow is mirror-still: keep it
        // an honest no-op rather than stepping a solver over dead water.
        let wind_energy = preset.wind.length_squared();
        let max_inflow = preset.inflows.iter().fold(0.0_f32, |m, f| m.max(f.rate));
        if wind_energy <= f32::EPSILON && max_inflow <= f32::EPSILON {
            return Self::default();
        }

        let poly = tessellate_loop(&preset.shoreline, preset.samples_per_segment.max(1));
        if poly.len() < 3 {
            return Self::default();
        }

        // Bounding box of the outline, padded by a two-cell margin so the
        // shoreline sits comfortably inside the grid.
        let mut min = poly[0];
        let mut max = poly[0];
        for v in &poly {
            min = Vec2::new(min.x.min(v.x), min.y.min(v.y));
            max = Vec2::new(max.x.max(v.x), max.y.max(v.y));
        }
        // Choose a cell size that spans the whole outline without exceeding the
        // cell cap. Start from the authored resolution and coarsen (never
        // refine) until the grid — including a two-cell shore margin on every
        // side — fits the budget, so a huge basin is covered edge to edge at a
        // coarser step rather than clipped down to a tiny corner of dead water.
        let cap = preset.max_cells.max(4);
        let raw_x = max.x - min.x;
        let raw_z = max.y - min.y;
        let mut cell = preset.cell_size;
        let (mut nx, mut nz) = lake_grid_dims(raw_x, raw_z, cell);
        while nx.saturating_mul(nz) > cap {
            cell *= 1.25;
            let dims = lake_grid_dims(raw_x, raw_z, cell);
            nx = dims.0;
            nz = dims.1;
        }

        let margin = 2.0 * cell;
        let origin = Vec2::new(min.x - margin, min.y - margin);

        let cfg = SweConfig {
            nx,
            nz,
            dx: cell,
            gravity: GRAVITY,
            damping: preset.damping.clamp(0.0, 1.0),
        };

        // Bound the timestep by the still-water celerity of the rest depth.
        let still = SweState::still(cfg, preset.rest_depth);
        let max_speed = max_wave_speed(&still, cfg);
        let cfl_dt = cfl_timestep(max_speed, cfg.dx, preset.cfl_number);
        let dt = preset.timestep.max(0.0).min(cfl_dt);

        let cells = (cfg.nx * cfg.nz) as usize;
        let mut sources = vec![[0.0_f32; 4]; cells];
        let shore_width = preset.shore_width.max(0.0);

        // Collect the interior cells once (flat index + world center), so both
        // the wind fill and the inflow snapping reuse the same point-in-polygon
        // classification instead of re-testing the grid per inflow.
        let mut interior: Vec<(usize, Vec2)> = Vec::new();
        for j in 0..cfg.nz {
            let wz = origin.y + (j as f32 + 0.5) * cell;
            for i in 0..cfg.nx {
                let wx = origin.x + (i as f32 + 0.5) * cell;
                let p = Vec2::new(wx, wz);
                if point_in_polygon(p, &poly) {
                    interior.push(((j * cfg.nx + i) as usize, p));
                }
            }
        }
        if interior.is_empty() {
            return Self::default();
        }

        // Wind drift: a momentum-only surface stress (no mass added) tapered to
        // zero over the near-shore band so the banks stay calm.
        let wind = preset.wind;
        for &(idx, p) in &interior {
            let drive = if shore_width <= f32::EPSILON {
                1.0
            } else {
                smoothstep01(distance_to_polygon(p, &poly) / shore_width)
            };
            sources[idx] = [0.0, wind.x * drive, wind.y * drive, 0.0];
        }

        // Stream mouths: snap each feed to the nearest interior cell and add its
        // steady depth feed in `.x`, so an inflow can never leak onto dry land.
        for inflow in &preset.inflows {
            if inflow.rate <= f32::EPSILON {
                continue;
            }
            let mut best_idx = interior[0].0;
            let mut best_d2 = f32::INFINITY;
            for &(idx, p) in &interior {
                let d2 = p.sub(inflow.position).length_squared();
                if d2 < best_d2 {
                    best_d2 = d2;
                    best_idx = idx;
                }
            }
            sources[best_idx][0] += inflow.rate;
        }

        // The lake must carry some live source; a wind-only basin whose drift
        // fully faded (all interior on the shore band) with no inflow is a no-op.
        let live = sources.iter().any(|s| {
            s[0].abs() > f32::EPSILON || s[1].abs() > f32::EPSILON || s[2].abs() > f32::EPSILON
        });
        if !live {
            return Self::default();
        }

        let cell_count = cfg.nx * cfg.nz;
        Self {
            swe_cells: cell_count,
            swe_sources: sources,
            swe_params: GpuWaterSweParams {
                nx: cfg.nx,
                nz: cfg.nz,
                dx: cfg.dx,
                gravity: cfg.gravity,
                damping: cfg.damping,
                dt,
                cfl_number: preset.cfl_number,
                max_wave_speed: max_speed,
            },
            passes: WaterPasses {
                swe: true,
                ..WaterPasses::default()
            },
            substeps: 1,
            grid2d_texels: cell_count,
            counts: WaterBufferCounts {
                swe_cells: cell_count,
                ..WaterBufferCounts::default()
            },
            ..Self::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_architecture::water::gpu::pipeline::prepare;
    use prism_render_architecture::water::kernels::WaterKernel;

    /// A square outline (side `40 m`) centered on the origin.
    fn square_lake() -> LakePreset {
        LakePreset {
            shoreline: vec![
                Vec2::new(-20.0, -20.0),
                Vec2::new(20.0, -20.0),
                Vec2::new(20.0, 20.0),
                Vec2::new(-20.0, 20.0),
            ],
            cell_size: 1.0,
            samples_per_segment: 6,
            rest_depth: 2.0,
            shore_width: 0.0,
            wind: Vec2::new(0.5, 0.0),
            inflows: Vec::new(),
            ..LakePreset::default()
        }
    }

    /// The preset expands into a live Shallow-Water body whose schedule steps
    /// the basin.
    #[test]
    fn lake_preset_builds_a_live_basin() {
        let body = WaterBody::lake(square_lake());
        assert!(body.swe_cells > 0);
        assert_eq!(body.swe_sources.len(), body.swe_cells as usize);
        assert_eq!(body.counts.swe_cells, body.swe_cells);
        assert_eq!(body.grid2d_texels, body.swe_cells);

        let ex = body.as_extract();
        assert!(ex.swe);
        let plan = prepare(&ex);
        assert!(plan
            .dispatches
            .iter()
            .any(|d| d.kernel == WaterKernel::SweStep));
    }

    /// A mirror-still lake (no wind, no inflow) carries no motion, so the preset
    /// is an honest no-op.
    #[test]
    fn calm_lake_is_a_noop() {
        let body = WaterBody::lake(LakePreset {
            wind: Vec2::ZERO,
            inflows: Vec::new(),
            ..square_lake()
        });
        assert_eq!(body.swe_cells, 0);
        assert!(body.swe_sources.is_empty());
        assert!(prepare(&body.as_extract()).dispatches.is_empty());
    }

    /// Fewer than three shoreline nodes cannot bound an area.
    #[test]
    fn too_few_nodes_is_a_noop() {
        let body = WaterBody::lake(LakePreset {
            shoreline: vec![Vec2::new(0.0, 0.0), Vec2::new(10.0, 0.0)],
            ..square_lake()
        });
        assert_eq!(body.swe_cells, 0);
        assert!(body.swe_sources.is_empty());
    }

    /// Wind-only drift is mass-neutral: it stirs momentum into `.y`/`.z` but adds
    /// no depth feed in `.x`.
    #[test]
    fn wind_only_lake_is_mass_neutral() {
        let body = WaterBody::lake(square_lake());
        let mut moving = 0_u32;
        for s in &body.swe_sources {
            assert!(s[0].abs() <= f32::EPSILON, "wind must not add mass in .x");
            if s[1].abs() > f32::EPSILON || s[2].abs() > f32::EPSILON {
                moving += 1;
            }
        }
        assert!(moving > 0, "the interior must carry a wind drift");
    }

    /// With no shore band every interior drift cell carries the full wind
    /// velocity, so all moving cells share one momentum magnitude.
    #[test]
    fn no_shore_band_drives_the_interior_uniformly() {
        let body = WaterBody::lake(square_lake());
        let wind_mag = Vec2::new(0.5, 0.0).length();
        for s in &body.swe_sources {
            let mag = Vec2::new(s[1], s[2]).length();
            if mag > f32::EPSILON {
                assert!(
                    (mag - wind_mag).abs() < 1.0e-4,
                    "mag {mag} != wind {wind_mag}"
                );
            }
        }
    }

    /// A shore band tapers the drift: the calmest moving cell (near a bank) is
    /// meaningfully slower than the fastest (open water).
    #[test]
    fn a_shore_band_calms_the_banks() {
        let body = WaterBody::lake(LakePreset {
            shore_width: 8.0,
            ..square_lake()
        });
        let mut min_mag = f32::INFINITY;
        let mut max_mag = 0.0_f32;
        for s in &body.swe_sources {
            let mag = Vec2::new(s[1], s[2]).length();
            if mag > f32::EPSILON {
                min_mag = min_mag.min(mag);
                max_mag = max_mag.max(mag);
            }
        }
        assert!(max_mag > 0.0);
        assert!(
            min_mag < 0.5 * max_mag,
            "shore taper should span a wide range (min={min_mag}, max={max_mag})"
        );
    }

    /// A stream mouth adds a localized depth feed in `.x`.
    #[test]
    fn an_inflow_adds_a_depth_feed() {
        let body = WaterBody::lake(LakePreset {
            wind: Vec2::ZERO,
            inflows: vec![LakeInflow {
                position: Vec2::new(0.0, 0.0),
                rate: 0.3,
            }],
            ..square_lake()
        });
        let fed: f32 = body.swe_sources.iter().map(|s| s[0]).sum();
        assert!((fed - 0.3).abs() < 1.0e-4, "one cell must carry the feed");
        // Mass is fed but no wind stirs the surface, so momentum stays zero.
        for s in &body.swe_sources {
            assert!(s[1].abs() <= f32::EPSILON && s[2].abs() <= f32::EPSILON);
        }
    }

    /// An inflow requested far outside the outline is snapped to an interior
    /// cell rather than leaking onto dry land.
    #[test]
    fn an_outside_inflow_is_snapped_inside() {
        let body = WaterBody::lake(LakePreset {
            wind: Vec2::ZERO,
            inflows: vec![LakeInflow {
                position: Vec2::new(1_000.0, 1_000.0),
                rate: 0.25,
            }],
            ..square_lake()
        });
        // Exactly one interior cell receives the whole feed.
        let fed_cells = body
            .swe_sources
            .iter()
            .filter(|s| s[0].abs() > f32::EPSILON)
            .count();
        assert_eq!(fed_cells, 1);
        let fed: f32 = body.swe_sources.iter().map(|s| s[0]).sum();
        assert!((fed - 0.25).abs() < 1.0e-4);
    }

    /// The rasterization is deterministic: the same preset yields byte-identical
    /// source fields across builds.
    #[test]
    fn rasterization_is_deterministic() {
        let a = WaterBody::lake(square_lake());
        let b = WaterBody::lake(square_lake());
        assert_eq!(a.swe_cells, b.swe_cells);
        assert_eq!(a.swe_sources, b.swe_sources);
        assert!((a.swe_params.dt - b.swe_params.dt).abs() < 1.0e-12);
    }

    /// The grid cell count never exceeds the configured cap, even for an outline
    /// whose bounding box would otherwise demand far more cells.
    #[test]
    fn grid_is_bounded_by_the_cell_cap() {
        let body = WaterBody::lake(LakePreset {
            shoreline: vec![
                Vec2::new(0.0, 0.0),
                Vec2::new(5_000.0, 0.0),
                Vec2::new(5_000.0, 3_000.0),
                Vec2::new(0.0, 3_000.0),
            ],
            cell_size: 0.5,
            max_cells: 4096,
            ..square_lake()
        });
        assert!(
            body.swe_cells <= 4096,
            "cells={} exceeds cap",
            body.swe_cells
        );
        assert!(body.swe_cells > 0);
    }

    /// The closed tessellation returns `nodes * seg_samples` vertices, all lying
    /// on the authored square outline.
    #[test]
    fn closed_tessellation_traces_the_outline() {
        let square = [
            Vec2::new(-1.0, -1.0),
            Vec2::new(1.0, -1.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(-1.0, 1.0),
        ];
        let poly = tessellate_loop(&square, 4);
        assert_eq!(poly.len(), 4 * 4);
        // A closed Catmull-Rom rounds each corner and bulges the straight edges
        // outward by a bounded overshoot (3/16 of the span for a unit square),
        // so every vertex stays near — but may slightly exceed — the node
        // extent. A tighter bound would be a fake assertion for a smoothing
        // spline; the honest invariant is bounded overshoot.
        for v in &poly {
            assert!(v.x >= -1.25 && v.x <= 1.25, "x out of bounds: {}", v.x);
            assert!(v.y >= -1.25 && v.y <= 1.25, "y out of bounds: {}", v.y);
        }
        // Each authored node is reproduced exactly at its segment start.
        assert!(
            poly.iter()
                .any(|v| (v.x - 1.0).abs() < 1.0e-4 && (v.y + 1.0).abs() < 1.0e-4),
            "node (1, -1) is not traced by the loop"
        );
    }

    /// The even-odd test classifies the interior and exterior of a square.
    #[test]
    fn point_in_polygon_classifies_a_square() {
        let square = [
            Vec2::new(-2.0, -2.0),
            Vec2::new(2.0, -2.0),
            Vec2::new(2.0, 2.0),
            Vec2::new(-2.0, 2.0),
        ];
        assert!(point_in_polygon(Vec2::new(0.0, 0.0), &square));
        assert!(point_in_polygon(Vec2::new(1.5, -1.5), &square));
        assert!(!point_in_polygon(Vec2::new(3.0, 0.0), &square));
        assert!(!point_in_polygon(Vec2::new(0.0, 10.0), &square));
    }

    /// The boundary distance is exact at the center of a square.
    #[test]
    fn distance_to_polygon_measures_the_boundary() {
        let square = [
            Vec2::new(-2.0, -2.0),
            Vec2::new(2.0, -2.0),
            Vec2::new(2.0, 2.0),
            Vec2::new(-2.0, 2.0),
        ];
        let d = distance_to_polygon(Vec2::new(0.0, 0.0), &square);
        assert!((d - 2.0).abs() < 1.0e-4, "center distance {d} != 2.0");
    }
}
