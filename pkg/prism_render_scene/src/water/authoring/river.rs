//! Spline-driven river authoring preset (a `UE5` Water-style spline river).
//!
//! A river is not a pond: its surface is a *directed* body of water whose course
//! is drawn by an artist as a handful of control points and whose current flows
//! downstream along that course. `UE5` Water and `Houdini` both author rivers
//! this way — a spline centerline carrying a per-node width, flow speed, and
//! depth — and then bake the spline into the flow field the surface solver
//! steps. This preset does exactly that against the dependency-free
//! [`prism_render_architecture::water`] core: it tessellates the control points
//! into a `Catmull-Rom` centerline, sizes a Shallow-Water grid around the
//! resulting course, and rasterizes the channel into the interaction-source
//! field the `SWE` kernel already consumes (`[.x` depth feed, `.y`/`.z` the
//! downstream momentum drive, `.w` unused] — the same layout the pond preset
//! and `water_surface.wesl` agree on). The `CFL` timestep is bounded by the
//! deepest reach's still-water celerity so the explicit step can never diverge.
//!
//! A degenerate course (fewer than two nodes, a zero-width or dry channel, or
//! no current anywhere) carries no directed water, so the preset returns the
//! default no-op body rather than lighting a solver that steps over a dead
//! channel.

use prism_render_architecture::water::gpu::buffers::WaterBufferCounts;
use prism_render_architecture::water::gpu::pipeline::WaterPasses;
use prism_render_architecture::water::swe::{cfl_timestep, max_wave_speed, SweConfig, SweState};
use prism_render_architecture::water::{Vec2, GRAVITY};

use crate::water::abi::GpuWaterSweParams;
use crate::water::body::WaterBody;

/// One art-authored node along a river's centerline.
///
/// Positions are world `XZ` meters (the plane the Shallow-Water grid tiles);
/// `width`, `flow_speed`, and `depth` are interpolated along the spline between
/// adjacent nodes so a course can taper, quicken, or deepen from source to mouth.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RiverControlPoint {
    /// World-space centerline position on the `XZ` plane (meters).
    pub position: Vec2,
    /// Full channel width at this node (m, `> 0`); half of this is the bank
    /// distance the current fills to either side of the centerline.
    pub width: f32,
    /// Downstream current speed at this node (m/s, `>= 0`); drives the momentum
    /// source folded into the surface step, oriented along the local tangent.
    pub flow_speed: f32,
    /// Still rest depth at this node (m, `> 0`); the deepest reach sets the
    /// wave celerity `sqrt(g * depth)` and therefore the `CFL`-bounded timestep.
    pub depth: f32,
}

impl Default for RiverControlPoint {
    fn default() -> Self {
        Self {
            position: Vec2::ZERO,
            width: 6.0,
            flow_speed: 1.5,
            depth: 1.0,
        }
    }
}

/// An art-directable description of a spline-authored river surface.
///
/// Construct one with [`RiverPreset::default`] (a short straight reach) and
/// override the [`control_points`](Self::control_points) to draw the course.
#[derive(Clone, Debug, PartialEq)]
pub struct RiverPreset {
    /// Ordered centerline nodes (source first, mouth last). Fewer than two
    /// nodes cannot define a course, so the preset is then an honest no-op.
    pub control_points: Vec<RiverControlPoint>,
    /// Square Shallow-Water cell edge length (m, `> 0`); the grid tiles the
    /// river's bounding course at this resolution.
    pub cell_size: f32,
    /// Spline samples emitted per control-point segment (`>= 1`); higher values
    /// trace a curved course more faithfully at the cost of authoring work.
    pub samples_per_segment: u32,
    /// Steady depth feed injected per step across the wetted channel (m/s,
    /// `>= 0`), keeping the course supplied; the current comes from the
    /// per-node `flow_speed`, not this feed.
    pub channel_feed: f32,
    /// Fraction of the half-width over which the source fades to zero at the
    /// bank, in `0..=1`; `0` gives a hard edge, `1` fades across the whole
    /// half-width for a soft shoreline.
    pub bank_softness: f32,
    /// Linear velocity damping per second in `0..=1`; the steady downstream
    /// drive settles against this drag rather than accelerating without bound.
    pub damping: f32,
    /// Requested frame timestep (s); clamped down to the `CFL` bound so the
    /// explicit step can never diverge.
    pub timestep: f32,
    /// Courant number in `(0, 1]` the explicit step must satisfy.
    pub cfl_number: f32,
    /// Hard cap on the grid cell count (`nx * nz`); a course whose bounding box
    /// would exceed this is clipped to the cap rather than allocating without
    /// bound. Normal authoring stays far below it.
    pub max_cells: u32,
}

impl Default for RiverPreset {
    fn default() -> Self {
        Self {
            control_points: vec![
                RiverControlPoint {
                    position: Vec2::new(0.0, 0.0),
                    ..RiverControlPoint::default()
                },
                RiverControlPoint {
                    position: Vec2::new(24.0, 0.0),
                    ..RiverControlPoint::default()
                },
            ],
            cell_size: 0.5,
            samples_per_segment: 12,
            channel_feed: 0.05,
            bank_softness: 0.5,
            damping: 0.08,
            timestep: 1.0 / 60.0,
            cfl_number: 0.5,
            max_cells: 1 << 20,
        }
    }
}

/// One tessellated point along the river centerline with the channel attributes
/// interpolated to it. `tangent` is the unit downstream direction used to orient
/// the momentum source.
#[derive(Clone, Copy, Debug)]
struct RiverSample {
    position: Vec2,
    tangent: Vec2,
    half_width: f32,
    flow_speed: f32,
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

/// Tessellates the control points into a dense centerline of [`RiverSample`]s.
///
/// Endpoints are duplicated to give the `Catmull-Rom` basis its phantom
/// neighbors so the curve passes through the first and last node. Width, flow
/// speed, and depth are interpolated with the same spline so the channel tapers
/// smoothly. Each sample's tangent is the unit direction to its successor (the
/// final sample reuses the previous tangent), i.e. the downstream heading.
fn tessellate(preset: &RiverPreset) -> Vec<RiverSample> {
    let pts = &preset.control_points;
    let n = pts.len();
    let seg_samples = preset.samples_per_segment.max(1);

    let mut positions: Vec<Vec2> = Vec::new();
    let mut half_widths: Vec<f32> = Vec::new();
    let mut flows: Vec<f32> = Vec::new();

    for seg in 0..n - 1 {
        let i0 = seg.saturating_sub(1);
        let i1 = seg;
        let i2 = seg + 1;
        let i3 = (seg + 2).min(n - 1);
        let (p0, p1, p2, p3) = (
            pts[i0].position,
            pts[i1].position,
            pts[i2].position,
            pts[i3].position,
        );
        // Emit the segment start plus interior samples; the very last segment
        // also emits its end so the mouth node is represented.
        let last_seg = seg == n - 2;
        let steps = if last_seg {
            seg_samples
        } else {
            seg_samples - 1
        };
        for s in 0..=steps {
            let t = s as f32 / seg_samples as f32;
            positions.push(catmull_rom_point(p0, p1, p2, p3, t));
            let w = catmull_rom_scalar(
                pts[i0].width,
                pts[i1].width,
                pts[i2].width,
                pts[i3].width,
                t,
            );
            half_widths.push(0.5 * w.max(0.0));
            let f = catmull_rom_scalar(
                pts[i0].flow_speed,
                pts[i1].flow_speed,
                pts[i2].flow_speed,
                pts[i3].flow_speed,
                t,
            );
            flows.push(f.max(0.0));
        }
    }

    let count = positions.len();
    let mut samples: Vec<RiverSample> = Vec::with_capacity(count);
    for k in 0..count {
        let tangent = if k + 1 < count {
            positions[k + 1].sub(positions[k]).normalize_or_zero()
        } else if k > 0 {
            samples[k - 1].tangent
        } else {
            Vec2::ZERO
        };
        samples.push(RiverSample {
            position: positions[k],
            tangent,
            half_width: half_widths[k],
            flow_speed: flows[k],
        });
    }
    // Backfill the final tangent when the whole course collapsed to one point.
    if let Some(last) = samples.last().copied()
        && last.tangent == Vec2::ZERO
        && count >= 2
    {
        let fixed = samples[count - 2].tangent;
        if let Some(slot) = samples.last_mut() {
            slot.tangent = fixed;
        }
    }
    samples
}

/// Nearest point on segment `a -> b` to `p`, returned as the clamped parameter
/// `t in 0..=1` and the squared distance to it. A degenerate segment reports
/// its start.
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

impl WaterBody {
    /// Builds a fully live spline-authored river body from a [`RiverPreset`].
    ///
    /// The control points are tessellated into a `Catmull-Rom` centerline, a
    /// Shallow-Water grid is sized around the resulting course (bounded by
    /// [`RiverPreset::max_cells`]), and the wetted channel is rasterized into the
    /// interaction-source field: a steady depth feed in the `.x` lane keeps the
    /// course supplied while a tangent-aligned momentum drive in `.y`/`.z`
    /// carries the current, both faded to zero across the bank. The timestep is
    /// clamped to the `CFL` bound implied by the deepest reach. A course that
    /// carries no directed water (too few nodes, dry or zero-width channel, or
    /// no current anywhere) yields the default no-op body.
    #[must_use]
    pub fn river(preset: RiverPreset) -> Self {
        // A course needs at least two nodes, a positive cell size, and some
        // wetted, flowing, non-dry channel to carry directed water.
        if preset.control_points.len() < 2 || preset.cell_size <= 0.0 {
            return Self::default();
        }
        let max_width = preset
            .control_points
            .iter()
            .fold(0.0_f32, |m, c| m.max(c.width));
        let max_depth = preset
            .control_points
            .iter()
            .fold(0.0_f32, |m, c| m.max(c.depth));
        let max_flow = preset
            .control_points
            .iter()
            .fold(0.0_f32, |m, c| m.max(c.flow_speed));
        if max_width <= 0.0 || max_depth <= 0.0 || max_flow <= f32::EPSILON {
            return Self::default();
        }

        let samples = tessellate(&preset);
        if samples.len() < 2 {
            return Self::default();
        }

        // Bounding box of the course, padded by a full half-width plus a two
        // cell margin so the soft bank fits inside the grid.
        let margin = 0.5 * max_width + 2.0 * preset.cell_size;
        let mut min = samples[0].position;
        let mut max = samples[0].position;
        for s in &samples {
            min = Vec2::new(min.x.min(s.position.x), min.y.min(s.position.y));
            max = Vec2::new(max.x.max(s.position.x), max.y.max(s.position.y));
        }
        let origin = Vec2::new(min.x - margin, min.y - margin);
        let span_x = (max.x - min.x) + 2.0 * margin;
        let span_z = (max.y - min.y) + 2.0 * margin;

        let inv_cell = 1.0 / preset.cell_size;
        let mut nx = (span_x * inv_cell).ceil() as u32 + 1;
        let mut nz = (span_z * inv_cell).ceil() as u32 + 1;
        nx = nx.max(1);
        nz = nz.max(1);
        // Clip the grid to the cell cap so a runaway course cannot allocate
        // without bound; a proportional cap keeps the aspect ratio sensible.
        if nx.saturating_mul(nz) > preset.max_cells && preset.max_cells >= 4 {
            let side = (preset.max_cells as f32).sqrt();
            let aspect = nx as f32 / nz as f32;
            nx = (side * aspect.sqrt()).floor().max(2.0) as u32;
            nz = (preset.max_cells / nx.max(1)).max(2);
        }

        let cfg = SweConfig {
            nx,
            nz,
            dx: preset.cell_size,
            gravity: GRAVITY,
            damping: preset.damping.clamp(0.0, 1.0),
        };

        // Bound the timestep by the deepest reach's still-water celerity.
        let still = SweState::still(cfg, max_depth);
        let max_speed = max_wave_speed(&still, cfg);
        let cfl_dt = cfl_timestep(max_speed, cfg.dx, preset.cfl_number);
        let dt = preset.timestep.max(0.0).min(cfl_dt);

        let cells = (cfg.nx * cfg.nz) as usize;
        let mut sources = vec![[0.0_f32; 4]; cells];
        let softness = preset.bank_softness.clamp(0.0, 1.0);
        let feed = preset.channel_feed.max(0.0);

        let mut live_cells = 0_u32;
        for j in 0..cfg.nz {
            let wz = origin.y + (j as f32 + 0.5) * preset.cell_size;
            for i in 0..cfg.nx {
                let wx = origin.x + (i as f32 + 0.5) * preset.cell_size;
                let p = Vec2::new(wx, wz);

                // Nearest centerline sample-segment to this cell center.
                let mut best_d2 = f32::INFINITY;
                let mut best_seg = 0_usize;
                let mut best_t = 0.0_f32;
                for k in 0..samples.len() - 1 {
                    let (t, d2) =
                        nearest_on_segment(p, samples[k].position, samples[k + 1].position);
                    if d2 < best_d2 {
                        best_d2 = d2;
                        best_seg = k;
                        best_t = t;
                    }
                }

                let a = &samples[best_seg];
                let b = &samples[best_seg + 1];
                let half_width = a.half_width + (b.half_width - a.half_width) * best_t;
                if half_width <= f32::EPSILON {
                    continue;
                }
                let dist = best_d2.sqrt();
                if dist >= half_width {
                    continue;
                }

                // Soft bank: full strength across the core, fading to zero over
                // the outer `softness` fraction of the half-width.
                let edge = softness * half_width;
                let falloff = if edge <= f32::EPSILON {
                    1.0
                } else {
                    let inner = half_width - edge;
                    if dist <= inner {
                        1.0
                    } else {
                        smoothstep01((half_width - dist) / edge)
                    }
                };
                if falloff <= f32::EPSILON {
                    continue;
                }

                let flow = a.flow_speed + (b.flow_speed - a.flow_speed) * best_t;
                // Downstream heading is the mean of the bracketing tangents.
                let tangent = a.tangent.add(b.tangent).normalize_or_zero();
                let drive = flow * falloff;

                let idx = (j * cfg.nx + i) as usize;
                sources[idx] = [
                    feed * falloff,    // .x steady depth feed
                    tangent.x * drive, // .y downstream momentum (u)
                    tangent.y * drive, // .z downstream momentum (v)
                    0.0,               // .w unused
                ];
                live_cells += 1;
            }
        }

        // The course must actually touch the grid; otherwise it is a no-op.
        if live_cells == 0 {
            return Self::default();
        }

        let cell_count = cells as u32;
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

    fn straight_reach() -> RiverPreset {
        RiverPreset {
            control_points: vec![
                RiverControlPoint {
                    position: Vec2::new(0.0, 0.0),
                    width: 4.0,
                    flow_speed: 2.0,
                    depth: 1.0,
                },
                RiverControlPoint {
                    position: Vec2::new(20.0, 0.0),
                    width: 4.0,
                    flow_speed: 2.0,
                    depth: 1.0,
                },
            ],
            cell_size: 1.0,
            samples_per_segment: 8,
            ..RiverPreset::default()
        }
    }

    /// A straight two-node reach expands into a live Shallow-Water body whose
    /// schedule steps the surface.
    #[test]
    fn straight_river_builds_a_live_body() {
        let body = WaterBody::river(straight_reach());
        assert!(body.swe_cells > 0);
        assert_eq!(body.swe_sources.len(), body.swe_cells as usize);
        assert_eq!(body.counts.swe_cells, body.swe_cells);
        assert!(body.passes.swe);

        let ex = body.as_extract();
        assert!(ex.swe);
        let plan = prepare(&ex);
        assert!(plan
            .dispatches
            .iter()
            .any(|d| d.kernel == WaterKernel::SweStep));
    }

    /// The current flows downstream (`+x` for a reach drawn along `+x`): every
    /// wetted cell carries positive `.y` momentum and negligible `.z`, and the
    /// depth feed lives in `.x`.
    #[test]
    fn current_points_downstream_along_the_course() {
        let body = WaterBody::river(straight_reach());
        let mut wetted = 0;
        for s in &body.swe_sources {
            let live = s[0].abs() > 1.0e-9 || s[1].abs() > 1.0e-9 || s[2].abs() > 1.0e-9;
            if !live {
                continue;
            }
            wetted += 1;
            assert!(s[0] >= 0.0, "depth feed is a non-negative source");
            assert!(s[1] > 0.0, "current drives downstream (+x)");
            assert!(s[2].abs() < 1.0e-3, "a straight +x reach has no cross flow");
            assert!(s[3].abs() < 1.0e-12, ".w lane is unused");
        }
        assert!(wetted > 0, "the channel wets some cells");
    }

    /// The wetted band is bounded by the channel width: no source lands farther
    /// than the half-width (plus one cell of rasterization slack) from the
    /// centerline that runs along `z = 0`.
    #[test]
    fn wetted_band_respects_the_channel_width() {
        let preset = straight_reach();
        let body = WaterBody::river(preset.clone());
        let nx = body.swe_params.nx;
        // Reconstruct the grid origin exactly as the builder does.
        let max_width = 4.0_f32;
        let margin = 0.5 * max_width + 2.0 * preset.cell_size;
        let origin_z = 0.0 - margin;
        let half_width = 0.5 * max_width;
        for (idx, s) in body.swe_sources.iter().enumerate() {
            let live = s[0].abs() > 1.0e-9 || s[1].abs() > 1.0e-9 || s[2].abs() > 1.0e-9;
            if !live {
                continue;
            }
            let j = (idx as u32) / nx;
            let wz = origin_z + (j as f32 + 0.5) * preset.cell_size;
            assert!(
                wz.abs() <= half_width + preset.cell_size,
                "source at z={wz} escapes the channel band"
            );
        }
    }

    /// The stored timestep never exceeds the `CFL` bound implied by the deepest
    /// reach's still-water celerity `sqrt(g * depth)`.
    #[test]
    fn timestep_respects_the_cfl_bound() {
        let mut preset = straight_reach();
        for cp in &mut preset.control_points {
            cp.depth = 4.0;
        }
        preset.cell_size = 0.5;
        preset.timestep = 10.0; // deliberately too large; must clamp down
        let body = WaterBody::river(preset.clone());
        let celerity = (GRAVITY * 4.0).sqrt();
        let bound = preset.cfl_number * preset.cell_size / celerity;
        assert!(body.swe_params.dt <= bound + 1.0e-6);
        assert!(body.swe_params.dt > 0.0);
        assert!((body.swe_params.max_wave_speed - celerity).abs() < 1.0e-3);
    }

    /// The `Catmull-Rom` centerline interpolates through its control points: the
    /// first and last sample sit exactly on the first and last node.
    #[test]
    fn centerline_passes_through_its_endpoints() {
        let preset = RiverPreset {
            control_points: vec![
                RiverControlPoint {
                    position: Vec2::new(1.0, 2.0),
                    ..RiverControlPoint::default()
                },
                RiverControlPoint {
                    position: Vec2::new(5.0, 9.0),
                    ..RiverControlPoint::default()
                },
                RiverControlPoint {
                    position: Vec2::new(11.0, 3.0),
                    ..RiverControlPoint::default()
                },
            ],
            ..RiverPreset::default()
        };
        let samples = tessellate(&preset);
        let first = samples.first().unwrap();
        let last = samples.last().unwrap();
        assert!(first.position.distance(Vec2::new(1.0, 2.0)) < 1.0e-4);
        assert!(last.position.distance(Vec2::new(11.0, 3.0)) < 1.0e-4);
    }

    /// A curved course bends the current: a right-angle bend produces cells
    /// whose momentum points along `+x` near the source and along a different
    /// heading near the mouth.
    #[test]
    fn a_bend_turns_the_current() {
        let preset = RiverPreset {
            control_points: vec![
                RiverControlPoint {
                    position: Vec2::new(0.0, 0.0),
                    width: 4.0,
                    flow_speed: 2.0,
                    depth: 1.0,
                },
                RiverControlPoint {
                    position: Vec2::new(20.0, 0.0),
                    width: 4.0,
                    flow_speed: 2.0,
                    depth: 1.0,
                },
                RiverControlPoint {
                    position: Vec2::new(20.0, 20.0),
                    width: 4.0,
                    flow_speed: 2.0,
                    depth: 1.0,
                },
            ],
            cell_size: 1.0,
            samples_per_segment: 12,
            ..RiverPreset::default()
        };
        let samples = tessellate(&preset);
        // The heading rotates from roughly +x at the source to roughly +z at
        // the mouth, so the summed cross product of consecutive tangents is
        // meaningfully non-zero (the course turns left).
        let mut turn = 0.0_f32;
        for k in 0..samples.len() - 1 {
            let t0 = samples[k].tangent;
            let t1 = samples[k + 1].tangent;
            turn += t0.x * t1.y - t0.y * t1.x;
        }
        assert!(turn > 0.1, "the course visibly bends (turn={turn})");
    }

    /// A course with no current is not a river: the preset is an honest no-op.
    #[test]
    fn a_currentless_course_is_a_noop() {
        let mut preset = straight_reach();
        for cp in &mut preset.control_points {
            cp.flow_speed = 0.0;
        }
        let body = WaterBody::river(preset);
        assert_eq!(body.swe_cells, 0);
        assert!(body.swe_sources.is_empty());
        assert!(prepare(&body.as_extract()).dispatches.is_empty());
    }

    /// Fewer than two nodes cannot define a course.
    #[test]
    fn a_single_node_is_a_noop() {
        let body = WaterBody::river(RiverPreset {
            control_points: vec![RiverControlPoint::default()],
            ..RiverPreset::default()
        });
        assert_eq!(body.swe_cells, 0);
        assert!(body.swe_sources.is_empty());
    }

    /// The rasterization is deterministic: the same preset yields byte-identical
    /// source fields across builds.
    #[test]
    fn rasterization_is_deterministic() {
        let a = WaterBody::river(straight_reach());
        let b = WaterBody::river(straight_reach());
        assert_eq!(a.swe_cells, b.swe_cells);
        assert_eq!(a.swe_sources, b.swe_sources);
        assert!((a.swe_params.dt - b.swe_params.dt).abs() < 1.0e-12);
    }

    /// The grid cell count never exceeds the configured cap, even for a course
    /// whose bounding box would otherwise demand far more cells.
    #[test]
    fn grid_is_bounded_by_the_cell_cap() {
        let preset = RiverPreset {
            control_points: vec![
                RiverControlPoint {
                    position: Vec2::new(0.0, 0.0),
                    width: 4.0,
                    flow_speed: 2.0,
                    depth: 1.0,
                },
                RiverControlPoint {
                    position: Vec2::new(10_000.0, 5_000.0),
                    width: 4.0,
                    flow_speed: 2.0,
                    depth: 1.0,
                },
            ],
            cell_size: 0.5,
            max_cells: 4096,
            ..RiverPreset::default()
        };
        let body = WaterBody::river(preset);
        assert!(
            body.swe_cells <= 4096,
            "cells={} exceeds cap",
            body.swe_cells
        );
    }
}
