// `wgpu` compute twin of the dependency-free `CPU` golden semi-Lagrangian foam
// advection (`prism_render_architecture::water::foam::advect_foam_field`).
//
// Foam is a scalar coverage field carried on the surface-flow grid. Each frame
// every cell backtraces its centre by `-dt * velocity` (converted to cell
// units) and bilinearly samples the previous field there — the unconditionally
// stable, non-negativity-preserving advection `Niagara`/`Crest` foam passes
// run on device. This twin evaluates that gather on the `GPU` so the foam field
// advects in the same place the rest of the water subsystem already lives.
//
// The gather is embarrassingly parallel: one invocation owns one output cell
// and reads only the *previous* field, so there is no write hazard, no barrier
// and no workgroup memory. Each cell performs the identical floating-point
// operations the `CPU` golden performs, in the same order (backtrace, clamp,
// two-tap-then-one-tap bilinear blend), so the readback is bit-identical to the
// golden up to the last-place slack of a `GPU` fused multiply-add. The kernel
// uses only the portable core-`WGSL` subset (multiply/add, `clamp`, `floor`,
// index math), so it runs unmodified on Metal, Vulkan and DX12.

// Uniform parameters for one dispatch. Layout matches `Params` in
// `src/water_foam_advect.rs` (`16` bytes, 16-byte aligned).
struct Params {
    // Cell count along x (`>= 1`).
    nx: u32,
    // Cell count along z (`>= 1`).
    nz: u32,
    // Cell size in meters (`> 0`).
    dx: f32,
    // Advection time step in seconds.
    dt: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
// Row-major `nx * nz` previous foam coverage.
@group(0) @binding(1) var<storage, read> density: array<f32>;
// Row-major `nx * nz` surface flow, x component (meters/second).
@group(0) @binding(2) var<storage, read> flow_u: array<f32>;
// Row-major `nx * nz` surface flow, z component (meters/second).
@group(0) @binding(3) var<storage, read> flow_v: array<f32>;
// Row-major `nx * nz` advected foam coverage.
@group(0) @binding(4) var<storage, read_write> out_density: array<f32>;

// Bilinear sample of the previous foam field at fractional cell coordinates,
// mirroring the golden `sample_bilinear` tap order exactly.
fn sample_bilinear(px: f32, pz: f32) -> f32 {
    let nx = params.nx;
    let nz = params.nz;
    let max_x = f32(nx - 1u);
    let max_z = f32(nz - 1u);
    let cx = clamp(px, 0.0, max_x);
    let cz = clamp(pz, 0.0, max_z);
    // `cx`/`cz` are non-negative, so truncation equals floor, matching the
    // golden's `as usize` cast.
    let x0 = u32(cx);
    let z0 = u32(cz);
    var x1 = x0 + 1u;
    if x1 > nx - 1u {
        x1 = nx - 1u;
    }
    var z1 = z0 + 1u;
    if z1 > nz - 1u {
        z1 = nz - 1u;
    }
    let fx = cx - f32(x0);
    let fz = cz - f32(z0);
    let s00 = density[z0 * nx + x0];
    let s10 = density[z0 * nx + x1];
    let s01 = density[z1 * nx + x0];
    let s11 = density[z1 * nx + x1];
    let top = s00 + (s10 - s00) * fx;
    let bottom = s01 + (s11 - s01) * fx;
    return top + (bottom - top) * fz;
}

@compute @workgroup_size(256)
fn advect_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let n = params.nx * params.nz;
    if idx >= n {
        return;
    }
    let x = idx % params.nx;
    let z = idx / params.nx;
    // `inv_dx` is the golden's single `1.0 / dx` reciprocal; the backtrace keeps
    // the golden's `dt * vel * inv_dx` multiply order.
    let inv_dx = 1.0 / params.dx;
    let px = f32(x) - params.dt * flow_u[idx] * inv_dx;
    let pz = f32(z) - params.dt * flow_v[idx] * inv_dx;
    out_density[idx] = sample_bilinear(px, pz);
}
