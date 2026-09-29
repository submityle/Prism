//! `GPU` compute twin of the water-surface solvers: shallow-water stepping,
//! foam advection, and the waterline mask.
//!
//! Compute twin of three CPU golden references in the water subsystem:
//! `prism_render_architecture::water::swe::step` (shallow-water height/flux
//! stepping under a `CFL`-stable timestep), `..::water::foam::step_foam`
//! (`semi-Lagrangian` foam advection + flow-aware decay + reactive sources),
//! and `..::water::waterline` (the soft above/below-water transition mask and
//! its shoreline band). The sandbox has no `GPU`, so this shader is validated
//! by naga compilation only (see `shader_tests`); each kernel below is one
//! dispatch over a regular `nx * nz` grid, the same row-major layout the CPU
//! golden iterates.
//!
//! The entry-point names below are byte-identical to the ones the host
//! `prism_render_architecture::water::kernels::WaterKernel::wesl_entry_point`
//! contract emits, so the pipeline cache keys resolve against this module:
//! `water_swe_step`, `water_foam_advect`, `water_waterline_mask`. The sibling
//! spectral / `PBF` / `FLIP` / reconstruct kernels live in the neighbouring
//! `water.wesl` module; their bind-group layouts differ, so they cannot share
//! this module's bindings.
//!
//! Every kernel is a pure, deterministic function of its input buffers: fixed
//! row-major traversal, no atomics, no randomness, and no reads of the buffer
//! it writes (`*_in` versus `*_out`), so a dispatch reproduces its CPU golden
//! bit-for-bit given the same inputs. The `SWE` continuity flux is integrated
//! in conservative form (a shared face flux enters its two neighbours with
//! opposite sign, so the summed depth change telescopes to zero and volume is
//! preserved to rounding); only the interaction depth source deliberately
//! changes volume. Foam advection is unconditionally stable and clamps
//! coverage to `0..=1`; the waterline weights are clamped to `0..=1` and
//! monotonic in submersion depth.

/// Squared-length / speed floor below which a division is skipped, so a
/// timestep or normalization never divides by ~0. Mirrors the CPU
/// `water::EPS_LEN_SQ`.
const WATER_EPS_LEN_SQ: f32 = 1.0e-12;

/// Scalar epsilon guarding the reference-speed and wave-speed divisions.
/// Mirrors the CPU `water::EPS`.
const WATER_EPS: f32 = 1.0e-6;

/// Finite sentinel returned when a `CFL` signal speed imposes no constraint,
/// standing in for the CPU `f32::MAX`; callers clamp it to their frame budget.
const WATER_DT_UNBOUNDED: f32 = 3.0e38;

// ---------------------------------------------------------------------------
// Shared numerical helpers (mirrors of the CPU golden's private functions).
// ---------------------------------------------------------------------------

/// Non-negative `exp` approximation via 12 squarings of `1 + x/4096`, mirroring
/// the CPU `water::exp_approx`. Used for the foam decay envelope so the `GPU`
/// path matches the CPU golden instead of the hardware `exp`.
fn water_exp_approx(x: f32) -> f32 {
    var base = 1.0 + x / 4096.0;
    if base < 0.0 {
        base = 0.0;
    }
    for (var i = 0u; i < 12u; i = i + 1u) {
        base = base * base;
    }
    return base;
}

/// Gravity-wave celerity plus flow speed at one cell: `|(u,v)| + sqrt(g*h)`.
/// A transiently dry (negative-depth) cell is treated as zero depth so the
/// square root stays real. Mirrors the CPU `swe::cell_wave_speed`.
fn water_cell_wave_speed(depth: f32, u: f32, v: f32, gravity: f32) -> f32 {
    let flow = sqrt(u * u + v * v);
    var clamped_depth = depth;
    if clamped_depth < 0.0 {
        clamped_depth = 0.0;
    }
    return flow + sqrt(gravity * clamped_depth);
}

/// Largest explicit timestep satisfying the `CFL` condition for a signal speed:
/// `cfl * dx / max_speed`. A non-positive signal speed imposes no constraint
/// and returns the unbounded sentinel. Mirrors the CPU `swe::cfl_timestep`.
fn water_cfl_timestep(max_speed: f32, dx: f32, cfl_number: f32) -> f32 {
    if max_speed <= WATER_EPS {
        return WATER_DT_UNBOUNDED;
    }
    return cfl_number * dx / max_speed;
}

/// First-order upwind advective derivative `u*dF/dx + v*dF/dz` of `center`,
/// taking the gradient from the upwind neighbour (the direction the flow comes
/// from). Mirrors the CPU `swe::upwind`.
fn water_upwind(
    flow_u: f32,
    flow_v: f32,
    center: f32,
    xm: f32,
    xp: f32,
    zm: f32,
    zp: f32,
    inv_dx: f32,
) -> f32 {
    var ddx = (xp - center) * inv_dx;
    if flow_u > 0.0 {
        ddx = (center - xm) * inv_dx;
    }
    var ddz = (zp - center) * inv_dx;
    if flow_v > 0.0 {
        ddz = (center - zm) * inv_dx;
    }
    return flow_u * ddx + flow_v * ddz;
}

/// Flow-aware foam decay rate (per second): interpolates from
/// `base_decay * persistence_floor` in still water up to `base_decay` at or
/// beyond `reference_speed`, so calm foam persists longer than churned foam.
/// Mirrors the CPU `foam::foam_decay_rate`.
fn water_foam_decay_rate(
    flow_speed: f32,
    base_decay: f32,
    persistence_floor: f32,
    reference_speed: f32,
) -> f32 {
    let floor = clamp(persistence_floor, 0.0, 1.0);
    var reference = reference_speed;
    if reference <= WATER_EPS {
        reference = WATER_EPS;
    }
    let t = clamp(flow_speed / reference, 0.0, 1.0);
    return base_decay * (floor + (1.0 - floor) * t);
}

// ===========================================================================
// Kernel 1: shallow-water stepping (`water_swe_step`).
// ===========================================================================

/// Grid layout and physical constants for the shallow-water body. Mirrors the
/// CPU `swe::SweConfig` plus the host-side `CFL`/timestep scalars the CPU
/// derives once per step in `swe::cfl_timestep`.
struct WaterSweParams {
    /// Cell count along x (`>= 1`).
    nx: u32,
    /// Cell count along z (`>= 1`).
    nz: u32,
    /// Square cell size in meters (`> 0`).
    dx: f32,
    /// Gravitational acceleration (wave celerity + pressure term).
    gravity: f32,
    /// Linear velocity damping per second in `0..=1` (drag toward rest).
    damping: f32,
    /// Requested frame timestep, seconds (clamped to the `CFL` bound below).
    dt: f32,
    /// Courant number the explicit step must satisfy.
    cfl_number: f32,
    /// Largest grid signal speed this step, from the host `swe::max_wave_speed`;
    /// used once to clamp `dt` uniformly so conservation is preserved.
    max_wave_speed: f32,
}

// Read-only depth `h` and depth-averaged velocity `(u, v)` at the start of the
// step (row-major, `nx * nz`).
@group(0) @binding(0) var<storage, read> swe_h_in: array<f32>;
@group(0) @binding(1) var<storage, read> swe_u_in: array<f32>;
@group(0) @binding(2) var<storage, read> swe_v_in: array<f32>;
// Written next-step state (never read within this dispatch, so no races).
@group(0) @binding(3) var<storage, read_write> swe_h_out: array<f32>;
@group(0) @binding(4) var<storage, read_write> swe_u_out: array<f32>;
@group(0) @binding(5) var<storage, read_write> swe_v_out: array<f32>;
// Per-cell interaction source: `.x` depth delta (rain / wading / hull draft,
// the only volume-changing term), `.yz` velocity impulse, `.w` unused. Mirrors
// the CPU `swe::inject_depth` / `swe::inject_velocity` folded into the step.
@group(0) @binding(6) var<storage, read> swe_sources: array<vec4<f32>>;
// Grid + `CFL`/timestep scalars.
@group(0) @binding(7) var<uniform> swe_params: WaterSweParams;

/// Advances the shallow-water state by one `CFL`-stable timestep.
///
/// One invocation per cell. Continuity is integrated in conservative flux form
/// with reflective (no-flux) walls; momentum uses a central pressure gradient,
/// first-order upwind self-advection, and linear damping. The effective `dt` is
/// clamped to the `CFL` bound derived from `max_wave_speed`, uniformly across
/// the grid, so the flux telescoping (and thus mass conservation) is preserved.
/// A body at rest stays exactly at rest; the interaction depth source is the
/// only term that changes total volume.
@compute @workgroup_size(8, 8, 1)
fn water_swe_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let nx = swe_params.nx;
    let nz = swe_params.nz;
    let x = gid.x;
    let z = gid.y;
    if x >= nx || z >= nz {
        return;
    }
    let idx = z * nx + x;

    // Uniform, `CFL`-stable timestep (matches the CPU host clamp).
    var dt = swe_params.dt;
    let cfl_dt = water_cfl_timestep(swe_params.max_wave_speed, swe_params.dx, swe_params.cfl_number);
    if cfl_dt < dt {
        dt = cfl_dt;
    }
    let inv_dx = 1.0 / swe_params.dx;
    let gravity = swe_params.gravity;

    let h_c = swe_h_in[idx];
    let u_c = swe_u_in[idx];
    let v_c = swe_v_in[idx];
    let hu_c = h_c * u_c;
    let hv_c = h_c * v_c;

    // Conservative continuity: each shared face flux is the two-cell average and
    // cancels across neighbours; reflective walls zero the boundary faces.
    var fx_right = 0.0;
    if x + 1u < nx {
        let r = idx + 1u;
        fx_right = 0.5 * (hu_c + swe_h_in[r] * swe_u_in[r]);
    }
    var fx_left = 0.0;
    if x >= 1u {
        let l = idx - 1u;
        fx_left = 0.5 * (swe_h_in[l] * swe_u_in[l] + hu_c);
    }
    var fz_up = 0.0;
    if z + 1u < nz {
        let up = idx + nx;
        fz_up = 0.5 * (hv_c + swe_h_in[up] * swe_v_in[up]);
    }
    var fz_down = 0.0;
    if z >= 1u {
        let dn = idx - nx;
        fz_down = 0.5 * (swe_h_in[dn] * swe_v_in[dn] + hv_c);
    }

    // Reflective neighbour sampling for the momentum gradients / advection.
    var h_xp = h_c;
    var u_xp = u_c;
    var v_xp = v_c;
    if x + 1u < nx {
        let r = idx + 1u;
        h_xp = swe_h_in[r];
        u_xp = swe_u_in[r];
        v_xp = swe_v_in[r];
    }
    var h_xm = h_c;
    var u_xm = u_c;
    var v_xm = v_c;
    if x >= 1u {
        let l = idx - 1u;
        h_xm = swe_h_in[l];
        u_xm = swe_u_in[l];
        v_xm = swe_v_in[l];
    }
    var h_zp = h_c;
    var u_zp = u_c;
    var v_zp = v_c;
    if z + 1u < nz {
        let up = idx + nx;
        h_zp = swe_h_in[up];
        u_zp = swe_u_in[up];
        v_zp = swe_v_in[up];
    }
    var h_zm = h_c;
    var u_zm = u_c;
    var v_zm = v_c;
    if z >= 1u {
        let dn = idx - nx;
        h_zm = swe_h_in[dn];
        u_zm = swe_u_in[dn];
        v_zm = swe_v_in[dn];
    }

    let dhdx = (h_xp - h_xm) * 0.5 * inv_dx;
    let dhdz = (h_zp - h_zm) * 0.5 * inv_dx;
    let adv_u = water_upwind(u_c, v_c, u_c, u_xm, u_xp, u_zm, u_zp, inv_dx);
    let adv_v = water_upwind(u_c, v_c, v_c, v_xm, v_xp, v_zm, v_zp, inv_dx);
    let damp = max(1.0 - swe_params.damping * dt, 0.0);

    var new_h = h_c - dt * inv_dx * ((fx_right - fx_left) + (fz_up - fz_down));
    var new_u = (u_c - dt * (adv_u + gravity * dhdx)) * damp;
    var new_v = (v_c - dt * (adv_v + gravity * dhdz)) * damp;

    // Interaction source injection (`.x` is the only volume-changing term).
    let src = swe_sources[idx];
    new_h = new_h + src.x;
    new_u = new_u + src.y;
    new_v = new_v + src.z;

    swe_h_out[idx] = new_h;
    swe_u_out[idx] = new_u;
    swe_v_out[idx] = new_v;
}

// ===========================================================================
// Kernel 2: foam advection + decay (`water_foam_advect`).
// ===========================================================================

/// Grid layout and decay tuning for the foam field. Mirrors the CPU
/// `foam::FoamConfig`.
struct WaterFoamParams {
    /// Cell count along x (`>= 1`).
    nx: u32,
    /// Cell count along z (`>= 1`).
    nz: u32,
    /// Square cell size in meters (`> 0`).
    dx: f32,
    /// Advection / decay timestep, seconds.
    dt: f32,
    /// Baseline decay rate per second at or above `reference_speed`.
    base_decay: f32,
    /// Fraction of `base_decay` still applied in still water, `0..=1`; smaller
    /// values make calm foam persist longer.
    persistence_floor: f32,
    /// Flow speed at which decay reaches the full `base_decay` (`> 0`).
    reference_speed: f32,
}

// Read-only foam coverage `0..=1`, the surface flow field, and this step's
// reactive sources (already scaled by `dt`). Row-major, `nx * nz`.
@group(1) @binding(0) var<storage, read> foam_in: array<f32>;
@group(1) @binding(1) var<storage, read> foam_u: array<f32>;
@group(1) @binding(2) var<storage, read> foam_v: array<f32>;
@group(1) @binding(3) var<storage, read> foam_sources: array<f32>;
// Written next-step coverage (never read within this dispatch).
@group(1) @binding(4) var<storage, read_write> foam_out: array<f32>;
@group(1) @binding(5) var<uniform> foam_params: WaterFoamParams;

/// Bilinearly samples the previous foam field at fractional cell coordinates,
/// clamping into `[0, nx-1] x [0, nz-1]` so a backtrace off the grid reads the
/// clamped edge instead of out of bounds. Convex taps keep the result
/// non-negative. Mirrors the CPU `foam::sample_bilinear`.
fn water_sample_foam_bilinear(px: f32, pz: f32) -> f32 {
    let nx = foam_params.nx;
    let nz = foam_params.nz;
    if nx == 0u || nz == 0u {
        return 0.0;
    }
    let max_x = f32(nx - 1u);
    let max_z = f32(nz - 1u);
    let cx = clamp(px, 0.0, max_x);
    let cz = clamp(pz, 0.0, max_z);
    let x0 = u32(cx);
    let z0 = u32(cz);
    let x1 = min(x0 + 1u, nx - 1u);
    let z1 = min(z0 + 1u, nz - 1u);
    let fx = cx - f32(x0);
    let fz = cz - f32(z0);
    let s00 = foam_in[z0 * nx + x0];
    let s10 = foam_in[z0 * nx + x1];
    let s01 = foam_in[z1 * nx + x0];
    let s11 = foam_in[z1 * nx + x1];
    let top = s00 + (s10 - s00) * fx;
    let bottom = s01 + (s11 - s01) * fx;
    return top + (bottom - top) * fz;
}

/// Advances the foam field one step: `semi-Lagrangian` advection along the flow,
/// flow-aware exponential decay (calm cells decay slower via the persistence
/// floor), then additive reactive sources, clamped to `0..=1`.
///
/// One invocation per cell. Each cell backtraces its centre by `-dt * velocity`
/// (converted to cell units) and bilinearly samples the previous field there;
/// this is unconditionally stable and preserves non-negativity. Mirrors the CPU
/// `foam::step_foam` (`advect_foam_field` + `decay_foam` + source clamp).
@compute @workgroup_size(8, 8, 1)
fn water_foam_advect(@builtin(global_invocation_id) gid: vec3<u32>) {
    let nx = foam_params.nx;
    let nz = foam_params.nz;
    let x = gid.x;
    let z = gid.y;
    if x >= nx || z >= nz {
        return;
    }
    let idx = z * nx + x;
    let dt = foam_params.dt;
    let inv_dx = 1.0 / foam_params.dx;

    // Semi-Lagrangian backtrace + bilinear resample.
    let px = f32(x) - dt * foam_u[idx] * inv_dx;
    let pz = f32(z) - dt * foam_v[idx] * inv_dx;
    let advected = water_sample_foam_bilinear(px, pz);

    // Flow-aware exponential decay of the advected coverage.
    let flow_speed = sqrt(foam_u[idx] * foam_u[idx] + foam_v[idx] * foam_v[idx]);
    let rate = water_foam_decay_rate(
        flow_speed,
        foam_params.base_decay,
        foam_params.persistence_floor,
        foam_params.reference_speed,
    );
    var density = advected;
    if density < 0.0 {
        density = 0.0;
    }
    let decayed = density * water_exp_approx(-rate * dt);

    // Additive reactive source, then clamp coverage to `0..=1`.
    let source = max(foam_sources[idx], 0.0);
    foam_out[idx] = clamp(decayed + source, 0.0, 1.0);
}

// ===========================================================================
// Kernel 3: waterline mask (`water_waterline_mask`).
// ===========================================================================

/// Soft-waterline and shoreline-band tuning. Mirrors the CPU
/// `waterline::WaterlineParams` plus the grid extent for the 2D dispatch.
struct WaterWaterlineParams {
    /// Cell count along x (`>= 1`).
    nx: u32,
    /// Cell count along z (`>= 1`).
    nz: u32,
    /// Half-width of the soft transition straddling the surface, meters.
    transition_half_width: f32,
    /// Total water depth below which a submerged sample joins the shoreline
    /// band, meters.
    shoreline_depth: f32,
}

// Per-cell sample world height, local water-surface height, and total water
// depth (row-major, `nx * nz`).
@group(2) @binding(0) var<storage, read> waterline_sample_y: array<f32>;
@group(2) @binding(1) var<storage, read> waterline_surface_y: array<f32>;
@group(2) @binding(2) var<storage, read> waterline_depth: array<f32>;
// Packed output: `.x` soft waterline weight `0..=1`, `.y` shoreline-band weight
// `0..=1`, `.z` underwater flag (`1` submerged, else `0`), `.w` signed
// submersion depth (`water_surface_y - sample_y`).
@group(2) @binding(3) var<storage, read_write> waterline_out: array<vec4<f32>>;
@group(2) @binding(4) var<uniform> waterline_params: WaterWaterlineParams;

/// Derives the waterline mask for each sample cell.
///
/// One invocation per cell. Computes the signed submersion depth
/// `water_surface_y - sample_y`, the inclusive above/below test, the soft
/// transition weight (linear across `2 * transition_half_width`, monotonic in
/// submersion, collapsing to a hard step for a degenerate band), and the
/// shoreline-band weight (rising toward `1` as the total water depth drops
/// below `shoreline_depth`, zero for dry or deep samples). Mirrors the CPU
/// `waterline::submersion_depth` / `is_underwater` / `waterline_weight` /
/// `shoreline_band`. All weights are clamped to `0..=1`.
@compute @workgroup_size(8, 8, 1)
fn water_waterline_mask(@builtin(global_invocation_id) gid: vec3<u32>) {
    let nx = waterline_params.nx;
    let nz = waterline_params.nz;
    let x = gid.x;
    let z = gid.y;
    if x >= nx || z >= nz {
        return;
    }
    let idx = z * nx + x;

    let depth = waterline_surface_y[idx] - waterline_sample_y[idx];
    var underwater = 0.0;
    if depth >= 0.0 {
        underwater = 1.0;
    }

    // Soft waterline weight: 0 fully in air, 1 fully submerged.
    let half = waterline_params.transition_half_width;
    var weight: f32;
    if half <= WATER_EPS {
        // Degenerate band collapses to a hard step at the surface.
        weight = underwater;
    } else {
        weight = clamp((depth + half) / (2.0 * half), 0.0, 1.0);
    }

    // Shoreline band: only submerged, shallow samples; monotonic in depth.
    var band = 0.0;
    let reach = waterline_params.shoreline_depth;
    if depth >= 0.0 && reach > WATER_EPS {
        let clamped_depth = max(waterline_depth[idx], 0.0);
        band = clamp(1.0 - clamped_depth / reach, 0.0, 1.0);
    }

    waterline_out[idx] = vec4<f32>(weight, band, underwater, depth);
}
