// Groom motion-energy reduction twin.
//
// Device mirror of the CPU golden
// `prism_render_architecture::hair::sleep::groom_motion_energy`. The sleep gate
// needs one groom-global scalar: the sum over every strand particle of its
// squared implicit velocity, where velocity is stored implicitly as
// `position - prev_position` (a Verlet pair). The golden walks the flat
// particle array left to right, accumulating `(position - prev_position)`
// dotted with itself. This kernel folds the identical quantity on-device: a
// single 256-wide workgroup grid-strides across the particles, computes each
// particle's squared diff inline, and tree-reduces the partials to one scalar.
//
// Unlike the shared-lane `analysis_reduce` fold (which reduces a precomputed
// per-element lane), this kernel fuses the per-particle compute
// (`pos - prev`, then `dot`) with the sum reduction in one pass straight from
// raw particle state, so it is a distinct computation: the sibling consumes a
// finished metric buffer, this one derives the metric from positions.
//
// Buffer layout: `params` carries the particle count; `particles[2*i+0]` is
// particle `i`'s current position (xyz used) and `particles[2*i+1]` its
// previous position (xyz used); the inverse mass is irrelevant to the energy
// and is not uploaded. `out_value[0]` receives the single folded scalar.
//
// Floating-point addition is commutative but not associative, so the tree's
// pairwise order differs from the golden's left-to-right walk by a few
// low-mantissa ULP; the parity test asserts a tolerance (abs_diff < 1e-4 or
// rel_diff < 1e-3) rather than bit-for-bit. The per-particle squared diff is
// itself exact up to fma fusion. An empty groom dispatches nothing and the host
// returns the identity 0.
//
// The kernel uses only subtraction, multiply/add and workgroup shared memory in
// the portable core-WGSL subset — no exp, pow, atomics or optional device
// feature — so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard single-workgroup shared-memory tree reduction over a
// Verlet velocity proxy; no Unreal Engine source or derived code.

// Particle count bounding the grid-stride load; padded to a 16-byte uniform.
struct Params {
    particle_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Two vec4 per particle: [2*i+0] = position (xyz), [2*i+1] = prev_position (xyz).
@group(0) @binding(1) var<storage, read> particles: array<vec4<f32>>;
// Single groom-global reduced scalar; length 1.
@group(0) @binding(2) var<storage, read_write> out_value: array<f32>;

const WG: u32 = 256u;

// Per-workgroup partials for the tree reduction.
var<workgroup> partials: array<f32, 256>;

// One particle's squared implicit velocity: dot(pos - prev, pos - prev).
fn particle_energy(i: u32) -> f32 {
    let pos = particles[2u * i + 0u].xyz;
    let prev = particles[2u * i + 1u].xyz;
    let d = pos - prev;
    return dot(d, d);
}

@compute @workgroup_size(256)
fn evaluate(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let count = params.particle_count;

    // Grid-stride load: each invocation sums every WG-th particle's energy into
    // a private partial, starting from the additive identity 0.
    var local = 0.0;
    var i = tid;
    loop {
        if i >= count {
            break;
        }
        local = local + particle_energy(i);
        i = i + WG;
    }
    partials[tid] = local;
    workgroupBarrier();

    // Logarithmic tree fold: halve the active range each step until lane 0
    // holds the whole-workgroup sum.
    var stride = WG / 2u;
    loop {
        if stride == 0u {
            break;
        }
        if tid < stride {
            partials[tid] = partials[tid] + partials[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if tid == 0u {
        out_value[0] = partials[0];
    }
}
