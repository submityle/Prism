// CFL adaptive time-step reduction.
//
// One kernel reduces a velocity field to its largest squared speed so the host
// can turn it into a stable time step. It runs one invocation per particle and
// folds the field in two stages:
//
//   * Each workgroup performs a shared-memory tree reduction over its 256 lanes,
//     leaving the workgroup's local maximum squared speed in lane 0.
//   * Lane 0 folds that local maximum into a single global slot with one
//     integer atomicMax, so the whole dispatch contends the atomic once per
//     workgroup rather than once per particle.
//
// Bit trick: a squared speed `dot(v, v)` is always non-negative, and the IEEE
// 754 bit pattern of a non-negative f32 increases monotonically with its value.
// Reinterpreting each squared speed as a u32 therefore lets a plain integer
// `max` (and the final `atomicMax`) compute the floating-point maximum exactly.
// The global slot is zero-initialised, and bit pattern 0 is +0.0, the correct
// identity for a maximum of non-negative values.
//
// The kernel mirrors the `cpu_max_speed` golden twin: the fold is a maximum
// (order independent and exact on the bit patterns), and only the host's final
// square root of the reduced squared speed carries floating-point rounding, so
// parity is checked within a tight tolerance rather than bit-for-bit.
//
// Provenance: the CFL condition is a classical, openly published stability
// criterion and shared-memory tree reduction is a standard GPU technique. No
// Unreal Engine source or derived code.

struct Params {
    // Number of velocity entries.
    n: u32,
    // Padding to a 16-byte boundary.
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
// Velocity field; xyz = velocity, w unused.
@group(0) @binding(1) var<storage, read> vel: array<vec4<f32>>;
// Global maximum squared speed, held as the u32 bit pattern of the f32 value.
@group(0) @binding(2) var<storage, read_write> out_max: atomic<u32>;

// Per-workgroup scratch for the tree reduction, one slot per lane.
var<workgroup> scratch: array<u32, 256>;

@compute @workgroup_size(256)
fn reduce(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    // Out-of-range lanes seed +0.0, the identity for a maximum of non-negative
    // values, so they never disturb the fold.
    var s: f32 = 0.0;
    if (gid.x < params.n) {
        let v = vel[gid.x].xyz;
        s = dot(v, v);
    }
    scratch[lid.x] = bitcast<u32>(s);
    workgroupBarrier();

    // Halving tree reduction. `stride` is uniform across the workgroup, so the
    // barrier is reached by every lane; only the write is guarded.
    var stride = 128u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (lid.x < stride) {
            scratch[lid.x] = max(scratch[lid.x], scratch[lid.x + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (lid.x == 0u) {
        atomicMax(&out_max, scratch[0]);
    }
}
