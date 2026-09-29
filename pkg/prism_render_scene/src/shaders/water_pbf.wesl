//! `GPU` compute twin of the `PBF` density solve + crest-spray emitter.
//!
//! Compute twin of the CPU golden references
//! `prism_render_architecture::water::pbf` (Position-Based Fluids, after
//! `Macklin` & Müller 2013) and the crest-spray planner in
//! `prism_render_architecture::water::breaking` / `..::water::surface_fx`. The
//! sandbox has no `GPU`, so this module is validated by naga compilation only;
//! it computes real classical `SPH`/`XPBD` numerics, no AI/ML anywhere.
//!
//! The entry-point names below are byte-identical to the ones the host
//! `prism_render_architecture::water::kernels::WaterKernel::wesl_entry_point`
//! contract emits, so the pipeline cache keys resolve against this module:
//! `water_pbf_density_solve` and `water_spray_emit`. Both are linear
//! per-particle passes dispatched at `@workgroup_size(64)`.
//!
//! ## Neighbour finding: uniform spatial hash
//!
//! Neighbour search mirrors the CPU `PbfGrid`/`PbfBins`: particles live in a
//! uniform grid whose cell edge equals the smoothing radius `h`, so a
//! particle only ever interacts with its own cell plus the 26 adjacent cells.
//! The host bins particles in ascending index order and packs the buckets into
//! a single `hash` storage buffer laid out as:
//!   - `hash[2*c + 0]` = start offset (into the index region) of cell `c`,
//!   - `hash[2*c + 1]` = particle count in cell `c`,
//!   - `hash[2*cell_count + s]` = the `s`-th binned particle index.
//! Because binning is index-ordered and every gather walks cells in a fixed
//! `(z, y, x)` order and indices in stored order, the whole solve is a pure
//! function of the input buffers — deterministic regardless of `GPU` thread
//! scheduling. Particles outside the grid contribute no neighbours and are
//! never indexed out of bounds (no panics / no `NaN` from divide-by-zero;
//! near-zero magnitudes are compared against the shared `EPS` floors).
//!
//! ## Density solve
//!
//! `water_pbf_density_solve` runs one invocation per particle. It reads the
//! frozen predicted positions (`positions_in`), estimates the `SPH` density
//! with the hand-written `Poly6` kernel, forms the density constraint
//! `C_i = rho_i / rho_0 - 1`, solves the `XPBD` scaling factor `lambda_i`,
//! and writes the projected position (`positions_out`) using the `Spiky`
//! gradient plus the anti-clustering artificial-pressure term. Reading a
//! frozen input buffer and writing a distinct output buffer keeps the Jacobi
//! projection race-free and deterministic; the host ping-pongs the two
//! bindings between solver iterations. Companion `XSPH` viscosity and vorticity
//! confinement operate on the velocity field in separate host stages (as in the
//! CPU pipeline) and are intentionally out of this density-constraint kernel.
//!
//! ## Spray emission
//!
//! `water_spray_emit` runs one invocation per candidate crest sample. It
//! mirrors `breaking_intensity` / `classify_breaking` / `plan_spray`: only a
//! fully breaking sample spews spray, the particle count is
//! `round(intensity * max_spray_count)`, and the launch velocity blends the
//! surface tangent (crest flow) with the normal (upward jet), scaled by
//! `jet_speed * intensity`. Each source writes its burst deterministically to
//! its own output slot, while an `atomic` counter accumulates the total spawned
//! particle count for the host's indirect draw; the summed total is
//! order-independent, so its value is deterministic.

/// Near-zero magnitude floor (mirrors the CPU `EPS`).
const PBF_EPS: f32 = 1.0e-6;
/// Squared-length floor guarding normalization (mirrors the CPU `EPS_LEN_SQ`).
const PBF_EPS_LEN_SQ: f32 = 1.0e-12;
/// Pi, matching the CPU `PI` used by the kernel normalizers.
const PBF_PI: f32 = 3.14159265358979323846;

/// Tuning for one `PBF` fluid domain's density solve plus the embedded uniform
/// spatial-hash grid. Mirrors the CPU `PbfParams` (Macklin & Müller) and
/// `PbfGrid`; scalar-only so it packs cleanly into a `uniform` buffer.
struct PbfParams {
    /// Minimum corner of the spatial-hash grid, world space.
    grid_origin: vec3<f32>,
    /// Grid cell edge length (`> 0`), equal to the smoothing radius.
    cell_size: f32,
    /// Target rest density `rho_0` (`> 0`).
    rest_density: f32,
    /// Per-particle mass (`> 0`) for the `SPH` density sum.
    particle_mass: f32,
    /// Smoothing radius `h` (`> 0`); support radius of both kernels.
    smoothing_radius: f32,
    /// `XPBD` relaxation/compliance term added to the `lambda` denominator.
    relaxation_epsilon: f32,
    /// Artificial-pressure strength `k` (`>= 0`) fighting particle clustering.
    artificial_pressure_k: f32,
    /// Fraction of `h` at which the artificial-pressure reference is sampled.
    artificial_pressure_delta_q: f32,
    /// Artificial-pressure exponent `n` (`>= 1`).
    artificial_pressure_n: u32,
    /// Number of particles bounding the per-particle dispatch.
    particle_count: u32,
    /// Grid cell count along x (`>= 1`).
    grid_nx: u32,
    /// Grid cell count along y (`>= 1`).
    grid_ny: u32,
    /// Grid cell count along z (`>= 1`).
    grid_nz: u32,
    /// Padding to a 16-byte multiple for `uniform` layout.
    pad: u32,
}

// Frozen predicted positions for this iteration: `xyz` = world position,
// `w` = carried scalar (e.g. inverse mass / phase), passed through untouched.
@group(0) @binding(0) var<storage, read> positions_in: array<vec4<f32>>;
// Projected positions written by this iteration (`xyz` corrected, `w` copied).
@group(0) @binding(1) var<storage, read_write> positions_out: array<vec4<f32>>;
// Packed uniform spatial hash: `[start, count]` pairs per cell followed by the
// index-ordered binned particle indices (see the module doc).
@group(0) @binding(2) var<storage, read> hash: array<u32>;
// Density-solve tuning + grid description.
@group(0) @binding(3) var<uniform> params: PbfParams;

/// `Poly6` smoothing kernel `W(r, h)` from the squared distance.
///
/// `315 / (64 * pi * h^9) * (h^2 - r^2)^3` for `0 <= r <= h`, else `0`.
/// Non-negative and non-increasing in `r`; a non-positive radius yields `0`.
fn pbf_poly6(r_squared: f32, h: f32) -> f32 {
    if h <= PBF_EPS {
        return 0.0;
    }
    let h2 = h * h;
    let r2 = max(r_squared, 0.0);
    if r2 >= h2 {
        return 0.0;
    }
    let h9 = h2 * h2 * h2 * h2 * h;
    let coeff = 315.0 / (64.0 * PBF_PI * h9);
    let d = h2 - r2;
    return coeff * d * d * d;
}

/// `Spiky` gradient kernel `grad W(r_vec, h)` for the constraint gradient.
///
/// `-45 / (pi * h^6) * (h - r)^2 * (r_vec / r)`; points from the neighbour back
/// toward the particle. Vanishes at/beyond the support radius and for a
/// coincident pair, so it never divides by zero.
fn pbf_spiky_gradient(r_vec: vec3<f32>, h: f32) -> vec3<f32> {
    if h <= PBF_EPS {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let r2 = dot(r_vec, r_vec);
    if r2 <= PBF_EPS_LEN_SQ || r2 >= h * h {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let r = sqrt(r2);
    let h6 = h * h * h * h * h * h;
    let coeff = -45.0 / (PBF_PI * h6);
    let scale = coeff * (h - r) * (h - r) / r;
    return r_vec * scale;
}

/// Integer power by repeated squaring (no `pow`, matching the CPU `powi`).
fn pbf_powi(base: f32, exp: u32) -> f32 {
    var result = 1.0;
    var b = base;
    var e = exp;
    loop {
        if e == 0u {
            break;
        }
        if (e & 1u) == 1u {
            result = result * b;
        }
        e = e >> 1u;
        if e > 0u {
            b = b * b;
        }
    }
    return result;
}

/// Artificial-pressure `s_corr = -k * (W(r,h) / W(delta_q*h, h))^n`.
///
/// A small repulsive (always `<= 0`) term that keeps particles from clumping;
/// `0` once the pair leaves the support radius or when strength/reference is
/// degenerate.
fn pbf_artificial_pressure(r_squared: f32) -> f32 {
    let k = max(params.artificial_pressure_k, 0.0);
    if k <= PBF_EPS {
        return 0.0;
    }
    let h = params.smoothing_radius;
    let dq = clamp(params.artificial_pressure_delta_q, 0.0, 1.0) * h;
    let reference = pbf_poly6(dq * dq, h);
    if reference <= PBF_EPS {
        return 0.0;
    }
    let ratio = pbf_poly6(r_squared, h) / reference;
    let power = pbf_powi(ratio, max(params.artificial_pressure_n, 1u));
    return -k * power;
}

/// Row-major flat cell index for an in-range cell coordinate.
fn pbf_flat_index(cx: u32, cy: u32, cz: u32) -> u32 {
    let nx = params.grid_nx;
    let ny = params.grid_ny;
    return cz * nx * ny + cy * nx + cx;
}

/// Number of grid cells.
fn pbf_cell_count() -> u32 {
    return params.grid_nx * params.grid_ny * params.grid_nz;
}

/// `true` plus cell coordinate when `p` falls inside the grid; `valid = false`
/// (and zeroed coords) when it lands outside, so callers skip it safely.
struct PbfCell {
    valid: bool,
    cx: u32,
    cy: u32,
    cz: u32,
}

/// Integer cell coordinate of a world position, or `valid = false` when the
/// position is outside the grid. Mirrors the CPU `PbfGrid::cell_coord`.
fn pbf_cell_coord(p: vec3<f32>) -> PbfCell {
    var out: PbfCell;
    out.valid = false;
    out.cx = 0u;
    out.cy = 0u;
    out.cz = 0u;
    if params.cell_size <= PBF_EPS {
        return out;
    }
    let local = p - params.grid_origin;
    if local.x < 0.0 || local.y < 0.0 || local.z < 0.0 {
        return out;
    }
    let cx = u32(local.x / params.cell_size);
    let cy = u32(local.y / params.cell_size);
    let cz = u32(local.z / params.cell_size);
    if cx >= params.grid_nx || cy >= params.grid_ny || cz >= params.grid_nz {
        return out;
    }
    out.valid = true;
    out.cx = cx;
    out.cy = cy;
    out.cz = cz;
    return out;
}

/// `XPBD` scaling factor `lambda_i` for particle `i`'s density constraint.
///
/// Gathers the 27-cell neighbourhood from the spatial hash, sums the `Poly6`
/// density (including the self term), assembles `grad_sum = sum_j grad W_ij`
/// and `grad_sq_sum = sum_j |grad W_ij|^2`, then returns
/// `lambda_i = -C_i / ((|grad_sum|^2 + grad_sq_sum) / rho_0^2 + epsilon)` with
/// `C_i = rho_i / rho_0 - 1`. A compressed particle (`C_i > 0`) yields
/// `lambda_i < 0` (pushes neighbours apart); a rarefied particle yields
/// `lambda_i > 0`. Out-of-grid particles and a degenerate rest density return
/// `0`.
fn pbf_lambda_at(i: u32) -> f32 {
    let rho0 = params.rest_density;
    if rho0 <= PBF_EPS {
        return 0.0;
    }
    if i >= params.particle_count {
        return 0.0;
    }
    let cell_count = pbf_cell_count();
    if cell_count == 0u {
        return 0.0;
    }
    let h = params.smoothing_radius;
    let radius_sq = params.cell_size * params.cell_size;
    let pi = positions_in[i].xyz;
    let cell = pbf_cell_coord(pi);
    if !cell.valid {
        return 0.0;
    }

    // Self term first (r = 0).
    var density = pbf_poly6(0.0, h);
    var grad_sum = vec3<f32>(0.0, 0.0, 0.0);
    var grad_sq_sum = 0.0;

    let index_base = 2u * cell_count;
    let z_lo = select(cell.cz - 1u, 0u, cell.cz == 0u);
    let z_hi = min(cell.cz + 1u, params.grid_nz - 1u);
    let y_lo = select(cell.cy - 1u, 0u, cell.cy == 0u);
    let y_hi = min(cell.cy + 1u, params.grid_ny - 1u);
    let x_lo = select(cell.cx - 1u, 0u, cell.cx == 0u);
    let x_hi = min(cell.cx + 1u, params.grid_nx - 1u);

    var z = z_lo;
    loop {
        if z > z_hi {
            break;
        }
        var y = y_lo;
        loop {
            if y > y_hi {
                break;
            }
            var x = x_lo;
            loop {
                if x > x_hi {
                    break;
                }
                let flat = pbf_flat_index(x, y, z);
                let start = hash[2u * flat];
                let count = hash[2u * flat + 1u];
                var s = 0u;
                loop {
                    if s >= count {
                        break;
                    }
                    let j = hash[index_base + start + s];
                    s = s + 1u;
                    if j == i || j >= params.particle_count {
                        continue;
                    }
                    let r_vec = pi - positions_in[j].xyz;
                    let r2 = dot(r_vec, r_vec);
                    if r2 > radius_sq {
                        continue;
                    }
                    density = density + pbf_poly6(r2, h);
                    let grad = pbf_spiky_gradient(r_vec, h);
                    grad_sum = grad_sum + grad;
                    grad_sq_sum = grad_sq_sum + dot(grad, grad);
                }
                x = x + 1u;
            }
            y = y + 1u;
        }
        z = z + 1u;
    }

    density = density * max(params.particle_mass, 0.0);
    let c_i = density / rho0 - 1.0;
    let inv_rho2 = 1.0 / (rho0 * rho0);
    let denom = (dot(grad_sum, grad_sum) + max(grad_sq_sum, 0.0)) * inv_rho2
        + max(params.relaxation_epsilon, 0.0);
    if denom <= PBF_EPS {
        return 0.0;
    }
    return -c_i / denom;
}

/// Density-constraint (`XPBD`) projection: one invocation per particle.
///
/// Computes `lambda_i`, then accumulates the position correction
/// `delta_p_i = (1 / rho_0) * sum_j (lambda_i + lambda_j + s_corr) * grad W_ij`
/// over the neighbourhood (re-gathering `lambda_j` from the frozen input, which
/// keeps the pass deterministic), and writes `positions_in[i] + delta_p_i` to
/// `positions_out[i]`. Out-of-range invocations and out-of-grid particles pass
/// the position through unchanged.
@compute @workgroup_size(64)
fn water_pbf_density_solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.particle_count {
        return;
    }
    let carried = positions_in[i].w;
    let pi = positions_in[i].xyz;
    let rho0 = params.rest_density;
    let cell_count = pbf_cell_count();
    // Default: pass the position through untouched.
    positions_out[i] = vec4<f32>(pi, carried);
    if rho0 <= PBF_EPS || cell_count == 0u {
        return;
    }
    let cell = pbf_cell_coord(pi);
    if !cell.valid {
        return;
    }

    let h = params.smoothing_radius;
    let radius_sq = params.cell_size * params.cell_size;
    let lambda_i = pbf_lambda_at(i);
    let index_base = 2u * cell_count;

    let z_lo = select(cell.cz - 1u, 0u, cell.cz == 0u);
    let z_hi = min(cell.cz + 1u, params.grid_nz - 1u);
    let y_lo = select(cell.cy - 1u, 0u, cell.cy == 0u);
    let y_hi = min(cell.cy + 1u, params.grid_ny - 1u);
    let x_lo = select(cell.cx - 1u, 0u, cell.cx == 0u);
    let x_hi = min(cell.cx + 1u, params.grid_nx - 1u);

    var delta = vec3<f32>(0.0, 0.0, 0.0);
    var z = z_lo;
    loop {
        if z > z_hi {
            break;
        }
        var y = y_lo;
        loop {
            if y > y_hi {
                break;
            }
            var x = x_lo;
            loop {
                if x > x_hi {
                    break;
                }
                let flat = pbf_flat_index(x, y, z);
                let start = hash[2u * flat];
                let count = hash[2u * flat + 1u];
                var s = 0u;
                loop {
                    if s >= count {
                        break;
                    }
                    let j = hash[index_base + start + s];
                    s = s + 1u;
                    if j == i || j >= params.particle_count {
                        continue;
                    }
                    let r_vec = pi - positions_in[j].xyz;
                    let r2 = dot(r_vec, r_vec);
                    if r2 > radius_sq {
                        continue;
                    }
                    let grad = pbf_spiky_gradient(r_vec, h);
                    let lambda_j = pbf_lambda_at(j);
                    let scorr = pbf_artificial_pressure(r2);
                    let weight = lambda_i + lambda_j + scorr;
                    delta = delta + grad * weight;
                }
                x = x + 1u;
            }
            y = y + 1u;
        }
        z = z + 1u;
    }

    delta = delta * (1.0 / rho0);
    positions_out[i] = vec4<f32>(pi + delta, carried);
}

// ---------------------------------------------------------------------------
// Crest-spray emission
// ---------------------------------------------------------------------------

/// One candidate crest sample. Mirrors the CPU `BreakingSample` metrics plus
/// the launch frame (`position` origin, `tangent` crest flow, `normal` jet).
struct SpraySource {
    /// World origin of the burst.
    position: vec3<f32>,
    /// Displacement-gradient steepness (slope magnitude).
    steepness: f32,
    /// Surface tangent (crest flow direction) for the launch.
    tangent: vec3<f32>,
    /// Choppy-displacement Jacobian (a fold at or below zero).
    jacobian: f32,
    /// Surface normal (upward jet direction).
    normal: vec3<f32>,
    /// Local crest curvature magnitude.
    curvature: f32,
}

/// One planned spray burst written per source slot.
struct SprayParticle {
    /// World origin of the burst.
    position: vec3<f32>,
    /// Number of spray particles this source emits (`0` when not breaking).
    count: u32,
    /// Launch velocity (tangent + upward jet, scaled by intensity).
    velocity: vec3<f32>,
    /// Padding to a 16-byte multiple.
    pad: u32,
}

/// Spawn buffer: an `atomic` total-particle counter followed by one burst
/// record per source slot.
struct SpraySpawn {
    /// Accumulated total spawned particle count (order-independent sum).
    counter: atomic<u32>,
    /// One burst record per source, indexed by source id.
    particles: array<SprayParticle>,
}

/// Breaking classifier thresholds + spray tuning. Mirrors the CPU
/// `BreakingCriteria` plus `jet_speed` / `max_spray_count`.
struct SprayParams {
    /// Steepness at which cresting begins (contribution saturates at twice it).
    steepness_threshold: f32,
    /// Jacobian at or below which a fold is counted.
    jacobian_fold_threshold: f32,
    /// Curvature at which the crest is sharp enough to spume.
    curvature_threshold: f32,
    /// Intensity at or above which a sample is fully breaking.
    breaking_intensity: f32,
    /// Launch-speed scale for the crest-spray jet.
    jet_speed: f32,
    /// Number of source samples bounding the dispatch.
    source_count: u32,
    /// Capacity of the per-source spawn record array.
    spawn_capacity: u32,
    /// Upper bound on spray particle count for one source.
    max_spray_count: u32,
}

// Candidate crest samples (read-only).
@group(0) @binding(0) var<storage, read> spray_sources: array<SpraySource>;
// Per-source spawn records + atomic total counter (read-write).
@group(0) @binding(1) var<storage, read_write> spray_spawn: SpraySpawn;
// Classifier thresholds + spray tuning.
@group(0) @binding(2) var<uniform> spray_params: SprayParams;

/// Saturating ramp: `0` at/below `edge`, `1` at/above `2 * edge`. Mirrors the
/// CPU `ramp_above`.
fn spray_ramp_above(value: f32, edge: f32) -> f32 {
    var scale = edge;
    if scale <= PBF_EPS {
        scale = PBF_EPS;
    }
    return clamp((value - edge) / scale, 0.0, 1.0);
}

/// Saturating ramp that grows as `value` falls below `edge`. Mirrors the CPU
/// `ramp_below`.
fn spray_ramp_below(value: f32, edge: f32) -> f32 {
    var scale = edge;
    if scale <= PBF_EPS {
        scale = PBF_EPS;
    }
    return clamp((edge - value) / scale, 0.0, 1.0);
}

/// Normalized breaking intensity in `0..=1`: the average of the steepness,
/// fold, and curvature ramps. Mirrors the CPU `breaking_intensity`.
fn spray_breaking_intensity(src: SpraySource) -> f32 {
    let steep = spray_ramp_above(src.steepness, spray_params.steepness_threshold);
    let fold = spray_ramp_below(src.jacobian, spray_params.jacobian_fold_threshold);
    let curve = spray_ramp_above(src.curvature, spray_params.curvature_threshold);
    return clamp((steep + fold + curve) / 3.0, 0.0, 1.0);
}

/// `true` when the sample is fully breaking (folded Jacobian or intensity at or
/// above the breaking threshold). Mirrors the `Breaking` arm of the CPU
/// `classify_breaking`.
fn spray_is_breaking(src: SpraySource, intensity: f32) -> bool {
    let folded = src.jacobian <= spray_params.jacobian_fold_threshold;
    return folded || intensity >= spray_params.breaking_intensity;
}

/// Normalize a vector, or return zero when it is shorter than the length floor
/// (mirrors the CPU `normalize_or_zero`).
fn spray_normalize_or_zero(v: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if len_sq > PBF_EPS_LEN_SQ {
        return v / sqrt(len_sq);
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

/// Crest-spray emitter: one invocation per candidate source.
///
/// Only fully breaking samples emit. The count is
/// `round(intensity * max_spray_count)` and the velocity is
/// `normalize(0.5 * tangent + 0.5 * normal) * jet_speed * intensity`. Each
/// source writes its burst to its own slot (`spray_spawn.particles[i]`), so the
/// per-slot output is a pure function of the input; the `atomic` counter sums
/// the total spawned particle count for the host's indirect draw. Out-of-range
/// invocations and over-capacity slots write nothing and never index out of
/// bounds.
@compute @workgroup_size(64)
fn water_spray_emit(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= spray_params.source_count || i >= spray_params.spawn_capacity {
        return;
    }
    let src = spray_sources[i];
    let intensity = spray_breaking_intensity(src);

    // Default: an empty burst for this slot.
    var record: SprayParticle;
    record.position = src.position;
    record.count = 0u;
    record.velocity = vec3<f32>(0.0, 0.0, 0.0);
    record.pad = 0u;

    if spray_is_breaking(src, intensity) {
        // Deterministic round-to-nearest (matches the CPU `+ 0.5` truncation).
        let count = u32(intensity * f32(spray_params.max_spray_count) + 0.5);
        let dir = spray_normalize_or_zero(src.tangent * 0.5 + src.normal * 0.5);
        record.count = count;
        record.velocity = dir * (spray_params.jet_speed * intensity);
        if count > 0u {
            atomicAdd(&spray_spawn.counter, count);
        }
    }

    spray_spawn.particles[i] = record;
}
