//! Adaptive Rectangular Decomposition (ARD) wave solver.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Microsoft Project Acoustics, or Google Resonance Audio source or derived
//! code; no AI/ML.
//!
//! # Relationship
//! Implements the offline "Wave Bake" solver of design section 43. The air
//! region is split into rectangular partitions ([`crate::solver::partition`]);
//! each partition's interior is advanced analytically in a cosine (DCT) mode
//! basis, which is dispersion free for a band-limited field, and adjacent
//! partitions exchange energy through an explicit finite-difference interface
//! operator. The scheme follows the published ARD idea (analytic interior plus
//! numerical interface coupling) and is implemented from first principles with
//! a plain second-order interface patch and a frequency-independent damping
//! term; it is not derived from any middleware source.
//!
//! # Determinism and real-time
//!
//! The solve is deterministic (all transcendental math routes through
//! [`bevy_math::ops`]) and runs offline only: it allocates mode buffers and is
//! not intended for the audio callback thread. The runtime side of the crate
//! ([`crate::lookup`]) never touches this solver.

use alloc::vec;
use alloc::vec::Vec;
use bevy_math::ops;
use core::f32::consts::PI;

use super::impulse::ImpulseResponse;
use super::partition::partition_scene;
use super::scene::VoxelScene;

/// Default speed of sound in air at room temperature, in metres per second.
pub const DEFAULT_SOUND_SPEED: f32 = 343.0;

/// Configuration for a wave solve.
///
/// The time step is derived from the scene cell size by a Courant condition;
/// callers choose the physical duration, sound speed, source bandwidth, and a
/// frequency-independent damping rate that sets the reverberation decay.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct SolveConfig {
    /// Speed of sound in metres per second.
    pub sound_speed: f32,
    /// Simulated duration in seconds (sets the number of time steps).
    pub duration_s: f32,
    /// Courant number in `(0, 1]`; scales the stable interface time step.
    pub courant: f32,
    /// Gaussian source pulse width in seconds (temporal standard deviation).
    pub pulse_width_s: f32,
    /// Frequency-independent amplitude damping rate in nepers per second. The
    /// resulting reverberation time is `3 * ln(10) / damping` seconds.
    pub damping: f32,
}

impl Default for SolveConfig {
    /// A compact, generically useful configuration: room-temperature air, a
    /// short tail, a half-Courant step, a narrow pulse, and moderate damping.
    #[inline]
    fn default() -> Self {
        Self {
            sound_speed: DEFAULT_SOUND_SPEED,
            duration_s: 0.3,
            courant: 0.5,
            pulse_width_s: 0.0,
            damping: 12.0,
        }
    }
}

/// Orthonormal 1-D DCT-II / DCT-III matrix of a fixed length.
///
/// Being orthonormal, the forward transform is the stored matrix and the
/// inverse is its transpose, so one matrix serves both directions.
struct Dct1d {
    n: usize,
    // Row-major `n * n`: `m[k * n + j]` is the forward coefficient.
    m: Vec<f32>,
}

impl Dct1d {
    fn new(n: usize) -> Self {
        let mut m = vec![0.0; n * n];
        let scale = ops::sqrt(2.0 / n as f32);
        let inv_sqrt2 = 1.0 / ops::sqrt(2.0);
        for k in 0..n {
            let ck = if k == 0 { inv_sqrt2 } else { 1.0 };
            for j in 0..n {
                let angle = PI * (2 * j + 1) as f32 * k as f32 / (2 * n) as f32;
                m[k * n + j] = scale * ck * ops::cos(angle);
            }
        }
        Self { n, m }
    }
}

/// Per-partition solver state: the mode-domain pressure history, the spatial
/// pressure, scratch forcing, precomputed oscillator coefficients, and the DCT
/// matrices for each axis.
struct PartitionState {
    dims: [usize; 3],
    modes_cur: Vec<f32>,
    modes_prev: Vec<f32>,
    pressure: Vec<f32>,
    force: Vec<f32>,
    two_cos: Vec<f32>,
    fscale: Vec<f32>,
    dct: [Dct1d; 3],
}

/// One cross-partition contact between two adjacent air voxels.
struct Interface {
    part_a: usize,
    local_a: usize,
    part_b: usize,
    local_b: usize,
}

/// An ARD wave solver bound to a voxel scene and configuration.
///
/// Build it once with [`WaveSolver::new`]; each call to
/// [`WaveSolver::solve`] runs an independent impulse solve from one source
/// cell and records the pressure at a set of probe cells.
pub struct WaveSolver {
    dims: [u32; 3],
    dt: f32,
    steps: usize,
    sound_speed: f32,
    cell_size: f32,
    g2: f32,
    pulse_width_s: f32,
    partitions: Vec<PartitionState>,
    interfaces: Vec<Interface>,
    // scene.len() entries: owner `(partition, local)` for each air voxel.
    owner: Vec<Option<(u32, u32)>>,
    scratch_in: Vec<f32>,
    scratch_out: Vec<f32>,
}

impl WaveSolver {
    /// Builds a solver for `scene` under `config`, performing the rectangular
    /// decomposition and precomputing every per-mode coefficient.
    #[must_use]
    pub fn new(scene: &VoxelScene, config: SolveConfig) -> Self {
        let dims = scene.dims();
        let cell_size = scene.cell_size();
        let sound_speed = config.sound_speed.max(1.0);
        let courant = config.courant.clamp(1.0e-3, 1.0);
        // Stable explicit interface step from the 3-D Courant condition.
        let dt = courant * cell_size / (sound_speed * ops::sqrt(3.0));
        let steps = ((config.duration_s.max(dt) / dt) as usize).max(1);
        let damping = config.damping.max(0.0);
        let g = ops::exp(-damping * dt);
        let g2 = g * g;
        let pulse_width_s = if config.pulse_width_s > 0.0 {
            config.pulse_width_s
        } else {
            1.5 * dt
        };

        let parts = partition_scene(scene);
        let mut owner: Vec<Option<(u32, u32)>> = vec![None; scene.len()];
        let mut partitions = Vec::with_capacity(parts.len());

        for (pi, part) in parts.iter().enumerate() {
            let d = part.dims();
            let dims_u = [d[0] as usize, d[1] as usize, d[2] as usize];
            let count = part.len();

            // Register owners for interface discovery and probing.
            for z in part.lo[2]..part.hi[2] {
                for y in part.lo[1]..part.hi[1] {
                    for x in part.lo[0]..part.hi[0] {
                        if let (Some(gi), Some(li)) =
                            (scene.index(x, y, z), part.local_index(x, y, z))
                        {
                            owner[gi] = Some((pi as u32, li as u32));
                        }
                    }
                }
            }

            // Per-mode oscillator coefficients in DCT output order (x fastest).
            let lx = d[0] as f32 * cell_size;
            let ly = d[1] as f32 * cell_size;
            let lz = d[2] as f32 * cell_size;
            let mut two_cos = vec![0.0; count];
            let mut fscale = vec![0.0; count];
            let dt2 = dt * dt;
            for kz in 0..dims_u[2] {
                for ky in 0..dims_u[1] {
                    for kx in 0..dims_u[0] {
                        let m = (kz * dims_u[1] + ky) * dims_u[0] + kx;
                        let fx = kx as f32 / lx;
                        let fy = ky as f32 / ly;
                        let fz = kz as f32 / lz;
                        let w2 = sound_speed * sound_speed
                            * PI
                            * PI
                            * (fx * fx + fy * fy + fz * fz);
                        let w = ops::sqrt(w2);
                        let cos_wdt = ops::cos(w * dt);
                        two_cos[m] = 2.0 * g * cos_wdt;
                        fscale[m] = if w2 > 0.0 {
                            2.0 * (1.0 - cos_wdt) / w2
                        } else {
                            dt2
                        };
                    }
                }
            }

            partitions.push(PartitionState {
                dims: dims_u,
                modes_cur: vec![0.0; count],
                modes_prev: vec![0.0; count],
                pressure: vec![0.0; count],
                force: vec![0.0; count],
                two_cos,
                fscale,
                dct: [
                    Dct1d::new(dims_u[0]),
                    Dct1d::new(dims_u[1]),
                    Dct1d::new(dims_u[2]),
                ],
            });
        }

        // Discover cross-partition interfaces along +x, +y, +z.
        let mut interfaces = Vec::new();
        let [nx, ny, nz] = dims;
        let neighbors = [(1u32, 0u32, 0u32), (0, 1, 0), (0, 0, 1)];
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let Some(gi) = scene.index(x, y, z) else {
                        continue;
                    };
                    let Some((pa, la)) = owner[gi] else {
                        continue;
                    };
                    for (dx, dy, dz) in neighbors {
                        let (ox, oy, oz) = (x + dx, y + dy, z + dz);
                        let Some(gj) = scene.index(ox, oy, oz) else {
                            continue;
                        };
                        let Some((pb, lb)) = owner[gj] else {
                            continue;
                        };
                        if pa != pb {
                            interfaces.push(Interface {
                                part_a: pa as usize,
                                local_a: la as usize,
                                part_b: pb as usize,
                                local_b: lb as usize,
                            });
                        }
                    }
                }
            }
        }

        let max_dim = dims[0].max(dims[1]).max(dims[2]) as usize;
        Self {
            dims,
            dt,
            steps,
            sound_speed,
            cell_size,
            g2,
            pulse_width_s,
            partitions,
            interfaces,
            owner,
            scratch_in: vec![0.0; max_dim.max(1)],
            scratch_out: vec![0.0; max_dim.max(1)],
        }
    }

    /// Solver time step in seconds.
    #[must_use]
    #[inline]
    pub fn dt(&self) -> f32 {
        self.dt
    }

    /// Solver sample rate (`1 / dt`) in Hz.
    #[must_use]
    #[inline]
    pub fn sample_rate(&self) -> f32 {
        1.0 / self.dt
    }

    /// Number of time steps a solve runs.
    #[must_use]
    #[inline]
    pub fn steps(&self) -> usize {
        self.steps
    }

    /// Number of rectangular partitions in the decomposition.
    #[must_use]
    #[inline]
    pub fn partition_count(&self) -> usize {
        self.partitions.len()
    }

    /// Number of cross-partition interface contacts.
    #[must_use]
    #[inline]
    pub fn interface_count(&self) -> usize {
        self.interfaces.len()
    }

    fn owner_of(&self, cell: [u32; 3], scene_dims: [u32; 3]) -> Option<(usize, usize)> {
        let [x, y, z] = cell;
        if x >= scene_dims[0] || y >= scene_dims[1] || z >= scene_dims[2] {
            return None;
        }
        let nx = scene_dims[0] as usize;
        let ny = scene_dims[1] as usize;
        let gi = (z as usize * ny + y as usize) * nx + x as usize;
        self.owner[gi].map(|(p, l)| (p as usize, l as usize))
    }

    /// Runs one impulse solve: a source pulse fires at `source_cell` and the
    /// pressure at each cell in `probe_cells` is recorded over time.
    ///
    /// The returned vector matches `probe_cells` by index. A probe that is not
    /// air (or a source that is not air) simply yields a silent response for
    /// that probe. The solve is a pure function of the scene, configuration,
    /// and cell arguments.
    #[must_use]
    pub fn solve(&mut self, source_cell: [u32; 3], probe_cells: &[[u32; 3]]) -> Vec<ImpulseResponse> {
        let dims = self.dims;
        let sr = self.sample_rate();
        let mut out: Vec<ImpulseResponse> = probe_cells
            .iter()
            .map(|_| ImpulseResponse::silent(sr, self.steps))
            .collect();

        // Reset all partition state.
        for p in &mut self.partitions {
            for v in &mut p.modes_cur {
                *v = 0.0;
            }
            for v in &mut p.modes_prev {
                *v = 0.0;
            }
            for v in &mut p.pressure {
                *v = 0.0;
            }
        }

        let source = self.owner_of(source_cell, dims);
        let probes: Vec<Option<(usize, usize)>> = probe_cells
            .iter()
            .map(|c| self.owner_of(*c, dims))
            .collect();

        let c2_over_dx2 = self.sound_speed * self.sound_speed / (self.cell_size * self.cell_size);
        let sigma = self.pulse_width_s;
        let t0 = 3.0 * sigma;
        let dt = self.dt;

        for step in 0..self.steps {
            // 1. Zero the forcing buffers.
            for p in &mut self.partitions {
                for f in &mut p.force {
                    *f = 0.0;
                }
            }

            // 2. Source injection (Gaussian pulse in time).
            if let Some((ps, ls)) = source {
                let t = step as f32 * dt;
                let z = (t - t0) / sigma;
                let pulse = ops::exp(-0.5 * z * z);
                self.partitions[ps].force[ls] += pulse;
            }

            // 3. Interface forcing from the current spatial pressures.
            for iface in &self.interfaces {
                let pa = self.partitions[iface.part_a].pressure[iface.local_a];
                let pb = self.partitions[iface.part_b].pressure[iface.local_b];
                self.partitions[iface.part_a].force[iface.local_a] += c2_over_dx2 * (pb - pa);
                self.partitions[iface.part_b].force[iface.local_b] += c2_over_dx2 * (pa - pb);
            }

            // 4. Advance each partition: force -> modes -> pressure.
            let g2 = self.g2;
            for p in &mut self.partitions {
                // Forward DCT of the forcing (in place): force becomes fhat.
                forward_dct(
                    &mut p.force,
                    p.dims,
                    &p.dct,
                    &mut self.scratch_in,
                    &mut self.scratch_out,
                );
                // Mode update.
                for m in 0..p.modes_cur.len() {
                    let next =
                        p.two_cos[m] * p.modes_cur[m] - g2 * p.modes_prev[m] + p.fscale[m] * p.force[m];
                    p.modes_prev[m] = p.modes_cur[m];
                    p.modes_cur[m] = next;
                }
                // Inverse DCT of the modes into the spatial pressure.
                p.pressure.copy_from_slice(&p.modes_cur);
                inverse_dct(
                    &mut p.pressure,
                    p.dims,
                    &p.dct,
                    &mut self.scratch_in,
                    &mut self.scratch_out,
                );
            }

            // 5. Record probe pressures for this step.
            for (oi, probe) in probes.iter().enumerate() {
                if let Some((pp, lp)) = probe {
                    out[oi].samples_mut()[step] = self.partitions[*pp].pressure[*lp];
                }
            }
        }

        out
    }
}

/// Applies a 1-D transform of length `n` along a strided line.
fn apply_1d(
    buf: &mut [f32],
    start: usize,
    stride: usize,
    mat: &Dct1d,
    inverse: bool,
    tin: &mut [f32],
    tout: &mut [f32],
) {
    let n = mat.n;
    for j in 0..n {
        tin[j] = buf[start + j * stride];
    }
    for (k, out) in tout.iter_mut().enumerate().take(n) {
        let mut s = 0.0;
        for (j, &tv) in tin.iter().enumerate().take(n) {
            let coef = if inverse {
                mat.m[j * n + k]
            } else {
                mat.m[k * n + j]
            };
            s += coef * tv;
        }
        *out = s;
    }
    for k in 0..n {
        buf[start + k * stride] = tout[k];
    }
}

/// Transforms every line of `buf` along `axis` using `mat`.
fn transform_axis(
    buf: &mut [f32],
    dims: [usize; 3],
    axis: usize,
    mat: &Dct1d,
    inverse: bool,
    tin: &mut [f32],
    tout: &mut [f32],
) {
    let [nx, ny, nz] = dims;
    match axis {
        0 => {
            for z in 0..nz {
                for y in 0..ny {
                    let start = (z * ny + y) * nx;
                    apply_1d(buf, start, 1, mat, inverse, tin, tout);
                }
            }
        }
        1 => {
            for z in 0..nz {
                for x in 0..nx {
                    let start = z * nx * ny + x;
                    apply_1d(buf, start, nx, mat, inverse, tin, tout);
                }
            }
        }
        _ => {
            for y in 0..ny {
                for x in 0..nx {
                    let start = y * nx + x;
                    apply_1d(buf, start, nx * ny, mat, inverse, tin, tout);
                }
            }
        }
    }
}

/// Forward separable 3-D DCT-II in place.
fn forward_dct(
    buf: &mut [f32],
    dims: [usize; 3],
    dct: &[Dct1d; 3],
    tin: &mut [f32],
    tout: &mut [f32],
) {
    transform_axis(buf, dims, 0, &dct[0], false, tin, tout);
    transform_axis(buf, dims, 1, &dct[1], false, tin, tout);
    transform_axis(buf, dims, 2, &dct[2], false, tin, tout);
}

/// Inverse separable 3-D DCT (DCT-III) in place.
fn inverse_dct(
    buf: &mut [f32],
    dims: [usize; 3],
    dct: &[Dct1d; 3],
    tin: &mut [f32],
    tout: &mut [f32],
) {
    transform_axis(buf, dims, 0, &dct[0], true, tin, tout);
    transform_axis(buf, dims, 1, &dct[1], true, tin, tout);
    transform_axis(buf, dims, 2, &dct[2], true, tin, tout);
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::Vec3;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn dct_round_trips() {
        let dct = Dct1d::new(6);
        let original = [0.3_f32, -1.0, 2.5, 0.0, 0.7, -0.4];
        let mut buf = original;
        let mut tin = [0.0; 6];
        let mut tout = [0.0; 6];
        apply_1d(&mut buf, 0, 1, &dct, false, &mut tin, &mut tout);
        apply_1d(&mut buf, 0, 1, &dct, true, &mut tin, &mut tout);
        for (a, b) in original.iter().zip(buf.iter()) {
            assert!(approx(*a, *b, 1e-4), "round trip {a} vs {b}");
        }
    }

    #[test]
    fn source_impulse_reaches_a_distant_probe() {
        // A single open corridor; energy must travel from source to probe.
        let scene = VoxelScene::new(Vec3::ZERO, 0.2, 16, 2, 2);
        let mut solver = WaveSolver::new(
            &scene,
            SolveConfig {
                duration_s: 0.05,
                damping: 4.0,
                ..SolveConfig::default()
            },
        );
        assert_eq!(solver.partition_count(), 1);
        let irs = solver.solve([1, 1, 1], &[[14, 1, 1]]);
        assert!(irs[0].total_energy() > 0.0, "probe received no energy");
    }

    #[test]
    fn energy_crosses_a_partition_interface() {
        // A thin wall with a gap splits the room into >1 partition; energy
        // injected on one side must still reach the other.
        let mut scene = VoxelScene::new(Vec3::ZERO, 0.25, 9, 5, 1);
        for y in 0..4 {
            scene.set_solid(4, y, 0, true);
        }
        let mut solver = WaveSolver::new(
            &scene,
            SolveConfig {
                duration_s: 0.08,
                damping: 3.0,
                ..SolveConfig::default()
            },
        );
        assert!(solver.partition_count() >= 2);
        assert!(solver.interface_count() >= 1);
        let irs = solver.solve([1, 1, 0], &[[7, 1, 0]]);
        assert!(
            irs[0].total_energy() > 0.0,
            "no energy crossed the interface"
        );
    }

    #[test]
    fn damping_shortens_the_tail() {
        let scene = VoxelScene::new(Vec3::ZERO, 0.25, 8, 8, 1);
        let late_start = |ir: &ImpulseResponse| ir.window_energy(ir.len() / 2, ir.len());
        let light = {
            let mut s = WaveSolver::new(
                &scene,
                SolveConfig {
                    duration_s: 0.1,
                    damping: 2.0,
                    ..SolveConfig::default()
                },
            );
            let ir = s.solve([2, 2, 0], &[[5, 5, 0]]);
            late_start(&ir[0])
        };
        let heavy = {
            let mut s = WaveSolver::new(
                &scene,
                SolveConfig {
                    duration_s: 0.1,
                    damping: 40.0,
                    ..SolveConfig::default()
                },
            );
            let ir = s.solve([2, 2, 0], &[[5, 5, 0]]);
            late_start(&ir[0])
        };
        assert!(
            heavy < light,
            "heavier damping should leave less late energy: heavy={heavy}, light={light}"
        );
    }

    #[test]
    fn solid_probe_is_silent() {
        let mut scene = VoxelScene::new(Vec3::ZERO, 0.25, 6, 6, 1);
        scene.set_solid(5, 5, 0, true);
        let mut solver = WaveSolver::new(&scene, SolveConfig::default());
        let irs = solver.solve([1, 1, 0], &[[5, 5, 0]]);
        assert_eq!(irs[0].total_energy(), 0.0);
    }
}
