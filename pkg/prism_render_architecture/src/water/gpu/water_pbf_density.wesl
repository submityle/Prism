//! `GPU` compute twin of the `PBF` density solve (`Macklin` & Müller 2013).
//!
//! Compute twin of the `CPU` golden reference
//! `prism_render_architecture::water::pbf` (Position-Based Fluids). The sandbox
//! has no `GPU`, so this module is validated by `naga`/`wesl` compilation only;
//! it computes real classical `SPH`/`XPBD` numerics — no AI/ML anywhere, no
//! `f32` equality test (near-zero magnitudes compare against the shared `EPS`
//! floors), `sqrt` the sole float intrinsic.
//!
//! The entry-point names below are byte-identical to the ones the host
//! `prism_render_architecture::water::kernels::WaterKernel::wesl_entry_point`
//! contract emits, so the pipeline cache keys resolve against this module:
//! `pbf_compute_lambda` and `water_pbf_density_solve`. Both are linear
//! per-particle passes dispatched at `@workgroup_size(64)` over the `Particle`
//! domain and share one bind group (four storage buffers + one uniform).
//!
//! ## Neighbour finding: uniform spatial hash
//!
//! Neighbour search mirrors the `CPU` `PbfGrid`/`PbfBins`: particles live in a
//! uniform grid whose cell edge equals the smoothing radius `h`, so a particle
//! only ever interacts with its own cell plus the 26 adjacent cells. The host
//! bins particles in ascending index order and packs the buckets into a single
//! `hash` storage buffer laid out as:
//!   - `hash[2*c + 0]` = start offset (into the index region) of cell `c`,
//!   - `hash[2*c + 1]` = particle count in cell `c`,
//!   - `hash[2*cell_count + s]` = the `s`-th binned particle index.
//! Because binning is index-ordered and every gather walks cells in a fixed
//! `(z, y, x)` order and indices in stored order, the whole solve is a pure
//! function of the input buffers — deterministic regardless of `GPU` thread
//! scheduling. Out-of-grid particles contribute no neighbours and are never
//! indexed out of bounds.
//!
//! ## Two-pass split
//!
//! `pbf_compute_lambda` runs first: one invocation per particle writes
//! `lambdas[i] = lambda_i`. `water_pbf_density_solve` then reads each `lambda_j`
//! from that buffer instead of re-gathering a neighbour's 27-cell neighbourhood,
//! which turns the position solve from `O(n * k^2)` into `O(n * k)` while
//! staying numerically bit-identical. Reading a frozen `positions_in` and
//! writing a distinct `positions_out` keeps the `Jacobi` projection race-free;
//! the host ping-pongs the two bindings between solver iterations.

/// Near-zero magnitude floor (mirrors the `CPU` `EPS`).
const PBF_EPS: f32 = 1.0e-6;
/// Squared-length floor guarding normalization (mirrors the `CPU` `EPS_LEN_SQ`).
const PBF_EPS_LEN_SQ: f32 = 1.0e-12;
/// Pi, matching the `CPU` `PI` used by the kernel normalizers.
const PBF_PI: f32 = 3.14159265358979323846;

/// Tuning for one `PBF` fluid domain's density solve plus the embedded uniform
/// spatial-hash grid. Mirrors the `CPU` `PbfParams` (Macklin & Müller) and
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
// Projected positions written by the solve pass (`xyz` corrected, `w` copied).
@group(0) @binding(1) var<storage, read_write> positions_out: array<vec4<f32>>;
// Packed uniform spatial hash: `[start, count]` pairs per cell followed by the
// index-ordered binned particle indices (see the module doc).
@group(0) @binding(2) var<storage, read> hash: array<u32>;
// Density-solve tuning + grid description.
@group(0) @binding(3) var<uniform> params: PbfParams;
// Per-particle `XPBD` scaling factor `lambda_i`, written by `pbf_compute_lambda`
// and read by `water_pbf_density_solve`. `read_write` so the single module-scope
// binding serves both passes under one bind-group layout.
@group(0) @binding(4) var<storage, read_write> lambdas: array<f32>;

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

/// Integer power by repeated squaring (no `pow`, matching the `CPU` `powi`).
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

/// `valid` plus cell coordinate when `p` falls inside the grid; `valid = false`
/// (and zeroed coords) when it lands outside, so callers skip it safely.
struct PbfCell {
    valid: bool,
    cx: u32,
    cy: u32,
    cz: u32,
}

/// Integer cell coordinate of a world position, or `valid = false` when the
/// position is outside the grid. Mirrors the `CPU` `PbfGrid::cell_coord`.
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
/// `C_i = rho_i / rho_0 - 1`. Out-of-grid particles and a degenerate rest
/// density return `0`.
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

/// Lambda pass: one invocation per particle writes `lambdas[i] = lambda_i`.
///
/// This is the first of the two `PBF` density-solve passes. Evaluating every
/// particle's `XPBD` scaling factor once here (rather than recomputing a
/// neighbour's factor on demand inside the correction loop) is what keeps the
/// solve linear in the neighbour count. Out-of-range invocations are skipped;
/// `pbf_lambda_at` already returns `0` for out-of-grid or degenerate particles.
@compute @workgroup_size(64)
fn pbf_compute_lambda(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.particle_count {
        return;
    }
    lambdas[i] = pbf_lambda_at(i);
}

/// Density-constraint (`XPBD`) projection: one invocation per particle.
///
/// Accumulates the position correction
/// `delta_p_i = (1 / rho_0) * sum_j (lambda_i + lambda_j + s_corr) * grad W_ij`
/// over the neighbourhood (reading each `lambda_j` from the `lambdas` buffer the
/// `pbf_compute_lambda` pass filled), and writes `positions_in[i] + delta_p_i`
/// to `positions_out[i]`. Out-of-range invocations and out-of-grid particles
/// pass the position through unchanged.
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
    let lambda_i = lambdas[i];
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
                    let lambda_j = lambdas[j];
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
