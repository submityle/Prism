// Voxel-density approximate strand self-collision twin: evaluates on-device the
// same per-particle gather-and-resolve the CPU golden `resolve_self_collision`
// applies (in `prism_render_architecture::hair::self_collision_voxel`), one
// thread per particle.
//
// The golden pass is two phases. Phase 1 *splats* every particle's mass and
// mass-weighted velocity into a low-resolution voxel density field with
// trilinear scatter-add - a many-writers-one-cell reduction whose GPU form
// needs float atomics (or a separate binning dispatch), so it stays host-side
// and the finished read-only field is uploaded here. Phase 2 is embarrassingly
// parallel: each particle samples the field at its *original* position and
// applies (a) density-gradient repulsion where overcrowded, (b) a two-sided
// volume-preservation restore toward the rest density, and (c) a friction blend
// of its velocity toward the local mass-weighted average. Because the field is
// built once from the input snapshot, phase 2 is a pure function of the inputs
// regardless of thread order, so this kernel mirrors the golden's resolve body
// value-for-value - a passing real-device parity test is direct evidence the
// ported kernel computes the same displaced positions and blended velocities as
// the reference, not merely that it compiles.
//
// It uses only the portable core-WGSL subset (floor, sqrt, min/max, clamp, dot,
// select, multiply/add) - no exp, no pow, no optional feature - so this twin
// runs unmodified on Metal, Vulkan and DX12. It is not bit-exact with the
// scalar CPU reference: a GPU may fuse a multiply-add, and the central-
// difference gradient divides by a cell separation, so the parity test asserts
// a per-component tolerance rather than exact equality.
//
// Provenance: standard voxel-density hair self-collision (AMD TressFX /
// Petrovic et al. 2005 volumetric hair) plus wgpu compute dispatch; no Unreal
// Engine source or derived code.

struct Params {
    // Number of particles in `positions` / `velocities` / outputs.
    particle_count: u32,
    // Target / threshold density; density above this drives repulsion.
    rest_density: f32,
    // One-sided density-gradient push-apart strength.
    repulsion: f32,
    // Hair-hair friction in [0, 1] (already clamped host-side).
    friction: f32,
    // Two-sided volume-preservation restore strength.
    target_volume_gain: f32,
    // Grid cell edge length (> 0, sanitised host-side).
    cell_size: f32,
    // Grid cell counts along x, y, z (>= 1 each, sanitised host-side).
    dim_x: u32,
    dim_y: u32,
    dim_z: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    // Grid world-space minimum corner.
    origin_x: f32,
    origin_y: f32,
    origin_z: f32,
    pad3: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Particle positions, stride 3 f32 (xyz).
@group(0) @binding(1) var<storage, read> positions: array<f32>;
// Particle velocities, stride 3 f32 (xyz).
@group(0) @binding(2) var<storage, read> velocities: array<f32>;
// Splatted per-cell density `rho`, row-major (length `cell_count`).
@group(0) @binding(3) var<storage, read> field_cells: array<f32>;
// Splatted per-cell mass-weighted velocity sum, stride 3 f32 (xyz).
@group(0) @binding(4) var<storage, read> field_vel: array<f32>;
// Output positions after the resolve, stride 3 f32 (xyz).
@group(0) @binding(5) var<storage, read_write> out_positions: array<f32>;
// Output velocities after the resolve, stride 3 f32 (xyz).
@group(0) @binding(6) var<storage, read_write> out_velocities: array<f32>;

// Matches the golden `EPS` (zero-length / near-zero-density guard).
const EPS: f32 = 1.0e-6;

fn inf() -> f32 {
    return bitcast<f32>(0x7f800000u);
}

// Finite means not NaN (v == v) and not an infinity (|v| below +inf).
fn finite_scalar(v: f32) -> bool {
    return v == v && abs(v) < inf();
}

fn finite3(p: vec3<f32>) -> bool {
    return finite_scalar(p.x) && finite_scalar(p.y) && finite_scalar(p.z);
}

// Repair any non-finite component to 0 (the golden's `sanitized`).
fn sanitize3(v: vec3<f32>) -> vec3<f32> {
    var o = v;
    if (!finite_scalar(o.x)) { o.x = 0.0; }
    if (!finite_scalar(o.y)) { o.y = 0.0; }
    if (!finite_scalar(o.z)) { o.z = 0.0; }
    return o;
}

fn origin() -> vec3<f32> {
    return vec3<f32>(params.origin_x, params.origin_y, params.origin_z);
}

fn world_max() -> vec3<f32> {
    let dims = vec3<f32>(f32(params.dim_x), f32(params.dim_y), f32(params.dim_z));
    return origin() + dims * params.cell_size;
}

// Clamp a world position into the grid's axis-aligned box (origin <= world_max
// always holds for a sanitised grid, so `clamp` is well-formed).
fn clamp_to_bounds(pos: vec3<f32>) -> vec3<f32> {
    let lo = origin();
    let hi = world_max();
    return vec3<f32>(clamp(pos.x, lo.x, hi.x), clamp(pos.y, lo.y, hi.y), clamp(pos.z, lo.z, hi.z));
}

// Flat row-major index of a cell, or -1 when out of bounds (matches the
// golden's `index_of` returning `None`).
fn index_of(cell: vec3<i32>) -> i32 {
    if (cell.x < 0 || cell.y < 0 || cell.z < 0) {
        return -1;
    }
    if (cell.x >= i32(params.dim_x) || cell.y >= i32(params.dim_y) || cell.z >= i32(params.dim_z)) {
        return -1;
    }
    return cell.x + i32(params.dim_x) * (cell.y + i32(params.dim_y) * cell.z);
}

struct Gather {
    density: f32,
    vel: vec3<f32>,
}

// Trilinear gather of the density and mass-weighted velocity-sum fields at a
// world position (the golden's `sample_density` / `sample_avg_velocity` share
// this 2x2x2 stencil; corners outside the grid contribute nothing). The corner
// order (x fastest, then y, then z) matches the reference's `for_each_corner`.
fn gather(pos: vec3<f32>) -> Gather {
    let p = sanitize3(pos);
    let inv = 1.0 / params.cell_size;
    let g = (p - origin()) * inv - vec3<f32>(0.5, 0.5, 0.5);
    let b = floor(g);
    let frac = g - b;
    let base = vec3<i32>(i32(b.x), i32(b.y), i32(b.z));

    var density = 0.0;
    var vel = vec3<f32>(0.0, 0.0, 0.0);
    for (var cz: u32 = 0u; cz < 2u; cz = cz + 1u) {
        let wz = select(frac.z, 1.0 - frac.z, cz == 0u);
        for (var cy: u32 = 0u; cy < 2u; cy = cy + 1u) {
            let wy = select(frac.y, 1.0 - frac.y, cy == 0u);
            for (var cx: u32 = 0u; cx < 2u; cx = cx + 1u) {
                let wx = select(frac.x, 1.0 - frac.x, cx == 0u);
                let w = wx * wy * wz;
                let cell = base + vec3<i32>(i32(cx), i32(cy), i32(cz));
                let idx = index_of(cell);
                if (idx >= 0) {
                    let u = u32(idx);
                    density = density + w * field_cells[u];
                    let vb = 3u * u;
                    let cv = vec3<f32>(field_vel[vb], field_vel[vb + 1u], field_vel[vb + 2u]);
                    vel = vel + cv * w;
                }
            }
        }
    }
    return Gather(density, vel);
}

// One central-difference gradient component along `offset` (a single-axis step
// of length `cell_size`), with the stencil clamped to the grid box. When the
// clamp collapses the two sample points the derivative is reported as 0.
fn grad_axis(base: vec3<f32>, offset: vec3<f32>) -> f32 {
    let hi = clamp_to_bounds(base + offset);
    let lo = clamp_to_bounds(base - offset);
    let d_hi = gather(hi).density;
    let d_lo = gather(lo).density;
    // Separation along the active axis: `offset` has one nonzero component of
    // magnitude `cell_size`, so `dot(hi - lo, offset) / cell_size` is exactly
    // `hi[axis] - lo[axis]`, matching the golden's `denom`.
    let denom = dot(hi - lo, offset) / params.cell_size;
    return select(0.0, (d_hi - d_lo) / denom, denom > EPS);
}

// Central-difference density gradient at a world position (points toward
// increasing density).
fn sample_gradient(pos: vec3<f32>) -> vec3<f32> {
    let h = params.cell_size;
    let base = clamp_to_bounds(pos);
    let gx = grad_axis(base, vec3<f32>(h, 0.0, 0.0));
    let gy = grad_axis(base, vec3<f32>(0.0, h, 0.0));
    let gz = grad_axis(base, vec3<f32>(0.0, 0.0, h));
    return vec3<f32>(gx, gy, gz);
}

// Unit vector along `v`, or zero when `v` is numerically zero.
fn normalize_or_zero(v: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if (len_sq > EPS * EPS) {
        return v * (1.0 / sqrt(len_sq));
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.particle_count) {
        return;
    }
    let bi = 3u * i;
    let pos0 = vec3<f32>(positions[bi], positions[bi + 1u], positions[bi + 2u]);
    let vel_in = vec3<f32>(velocities[bi], velocities[bi + 1u], velocities[bi + 2u]);

    // A non-finite position is left exactly as-is (the golden `continue`s before
    // it even scrubs the velocity): write both inputs straight back.
    if (!finite3(pos0)) {
        out_positions[bi] = pos0.x;
        out_positions[bi + 1u] = pos0.y;
        out_positions[bi + 2u] = pos0.z;
        out_velocities[bi] = vel_in.x;
        out_velocities[bi + 1u] = vel_in.y;
        out_velocities[bi + 2u] = vel_in.z;
        return;
    }

    // Scrub any non-finite incoming velocity to zero before the friction blend,
    // mirroring the golden's unconditional `p.velocity = p.velocity.sanitized()`
    // for a finite-position particle.
    var vel = sanitize3(vel_in);

    let gat = gather(pos0);
    let density = gat.density;
    let gradient = sample_gradient(pos0);
    let dir = normalize_or_zero(gradient);
    var avg_vel = vec3<f32>(0.0, 0.0, 0.0);
    if (density > EPS) {
        avg_vel = gat.vel * (1.0 / density);
    }

    // Signed over-density relative to the rest/target density.
    let excess = density - params.rest_density;
    // (2) Repulsion: one-sided push only where overcrowded.
    var disp_mag = 0.0;
    if (excess > 0.0) {
        disp_mag = disp_mag + params.repulsion * excess;
    }
    // (3) Volume preservation: two-sided soft restore toward rest density.
    disp_mag = disp_mag + params.target_volume_gain * excess;

    // Move down the density gradient for a positive magnitude (up it for a
    // negative one); `dir` is unit length or zero so a flat field gives no
    // spurious drift.
    let disp = dir * (-disp_mag);
    var new_pos = pos0 + disp;
    if (!finite3(new_pos)) {
        new_pos = pos0;
    }

    // (4) Hair-hair friction: blend velocity toward the local average, only
    // where there is a genuine neighbourhood (nonzero local density).
    if (density > EPS) {
        let keep = 1.0 - params.friction;
        let blended = vel * keep + avg_vel * params.friction;
        if (finite3(blended)) {
            vel = blended;
        }
    }

    out_positions[bi] = new_pos.x;
    out_positions[bi + 1u] = new_pos.y;
    out_positions[bi + 2u] = new_pos.z;
    out_velocities[bi] = vel.x;
    out_velocities[bi + 1u] = vel.y;
    out_velocities[bi + 2u] = vel.z;
}
