// Per-sample soft-barrier profile twin.
//
// Device mirror of the two `CPU` goldens
// `prism_render_architecture::hair::barrier_contact::barrier_energy` and
// `barrier_force_magnitude`. For each contact distance `d` it returns the
// soft-barrier energy `b(d)` and the repulsive force magnitude `-b'(d)` under a
// shared set of `BarrierParams` (`dhat`, `stiffness`, `d_floor`).
//
// The barrier is the rational, `C^1`, ln/exp-free potential the strand barrier
// contact pass integrates: on the clamped distance `d_e = clamp(d, d_floor,
// dhat)`,
//
//     b(d)    = k * (dhat - d_e)^2 * (1/d_e - 1/dhat)
//     -b'(d)  = k * (dhat - d_e) * [ 2 (1/d_e - 1/dhat) + (dhat - d_e)/d_e^2 ]
//
// and both are `0` in the free region `d >= dhat`.
//
// # Why one thread per sample
//
// Each sample's `(energy, force)` depends only on its own distance and the
// shared params, so this is embarrassingly parallel: one thread owns one
// distance, reads it and writes its energy/force pair. The host lays the
// distances out flat so threads never alias.
//
// # Correctness model
//
// The reference sanitizes its params (forcing `dhat > 0`, `d_floor` into
// `(0, dhat)`, `stiffness >= 0`) and treats a non-finite distance as "far"
// (energy/force `0`). The host supplies already-sanitized params and finite
// distances, so the kernel keeps the `d >= dhat` free-region early-out and the
// `clamp` to `[d_floor, dhat]` verbatim and omits the non-finite guard. The
// solve is a single closed-form evaluation (a handful of add/sub/mul/divide),
// no chained recurrence, so the only `CPU` vs `GPU` divergence is legal
// fused-multiply-add contraction and correctly rounded division; parity is
// asserted to `abs < 1e-4` or `rel < 1e-3`. Sample distances stay clear of the
// `dhat` and `d_floor` branch boundaries so the branch taken is identical on
// both sides.
//
// # Portability
//
// The kernel uses only add/sub/mul/divide/clamp/max in the portable core-`WGSL`
// subset — no `exp`, `pow`, `log` or optional device feature — so the twin runs
// unmodified on Metal, Vulkan and DX12. (The barrier is deliberately ln/exp-free
// in the `CPU` golden for exactly this portability.)
//
// Provenance: Prism's own rational soft-barrier (IPC-style, no Unreal Engine
// source or derived code) plus a `wgpu` compute dispatch.

// Shared barrier tuning plus the sample count, padded to one `16`-byte uniform
// slot. Mirrors (the sanitized subset of) `BarrierParams`.
struct Params {
    dhat: f32,
    stiffness: f32,
    d_floor: f32,
    sample_count: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Flat contact distances, one per sample.
@group(0) @binding(1) var<storage, read> dists: array<f32>;
// Per sample: barrier energy in `x`, repulsive force magnitude in `y`.
@group(0) @binding(2) var<storage, read_write> out_vals: array<vec2<f32>>;

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.sample_count) {
        return;
    }
    let d = dists[i];
    let dhat = params.dhat;
    let k = params.stiffness;
    let d_floor = params.d_floor;

    var energy: f32 = 0.0;
    var force: f32 = 0.0;
    if (d < dhat) {
        let de = clamp(d, d_floor, dhat);
        let gap = dhat - de; // >= 0
        let recip = 1.0 / de - 1.0 / dhat; // >= 0 since de <= dhat
        energy = max(k * (gap * gap) * recip, 0.0);
        let inv_de = 1.0 / de;
        let bracket = 2.0 * recip + gap * (inv_de * inv_de);
        force = max(k * gap * bracket, 0.0);
    }

    out_vals[i] = vec2<f32>(energy, force);
}
