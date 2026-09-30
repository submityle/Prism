// GPU parity twin of prism_render_architecture::cloth::sleep::max_kinetic_indicator.
//
// The golden walks every particle, skips the pinned ones (inverse mass <= 0),
// and keeps the largest velocity length_squared it sees, seeded at 0.0. This
// shader reproduces that reduction on the device: each invocation loads its own
// squared speed (0.0 when out of range or pinned, matching the golden floor),
// the workgroup collapses those values with a tree max in shared memory, and one
// lane per workgroup folds the partial into a single storage atomic.
//
// The running max is stored as the u32 bit pattern of the f32 partial. Every
// contributed value is non-negative (a squared length or the 0.0 floor), and for
// non-negative IEEE-754 floats the u32 bit order matches the float order, so
// atomicMax over the bits selects the exact same maximum the golden would. The
// host seeds the atomic with bitcast(0.0) = 0 and reinterprets the readback with
// f32::from_bits, so the CPU and GPU share one arithmetic and one seed.
//
// This module is validated by naga at build time and dispatched on a real device
// by cloth::sleep_gpu_tests.

const CLOTH_SLEEP_WORKGROUP: u32 = 64u;

struct SleepParams {
    // Number of particles bounding the dispatch (the golden slice length).
    particle_count: u32,
    // Padding to a whole 16-byte uniform stride.
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

// The running maximum, held as the u32 bit pattern of the f32 partial so the
// cross-workgroup fold can use an integer atomicMax.
@group(0) @binding(0) var<storage, read_write> max_indicator: atomic<u32>;
// Per-particle velocity: xyz is the velocity vector, w carries the inverse mass
// so the shader can skip pinned particles without a second buffer.
@group(0) @binding(1) var<storage, read> velocities: array<vec4<f32>>;
// Whole-dispatch scalars.
@group(0) @binding(2) var<uniform> params: SleepParams;

// Shared scratch for the intra-workgroup tree reduction; one slot per lane.
var<workgroup> scratch: array<f32, 64>;

@compute @workgroup_size(64)
fn sleep_max_kinetic(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let index = gid.x;
    var value = 0.0;
    if index < params.particle_count {
        let sample = velocities[index];
        let inverse_mass = sample.w;
        // Golden skips pinned particles (is_pinned == inverse_mass <= 0.0); a
        // skipped lane keeps the 0.0 floor, which never wins the max.
        if inverse_mass > 0.0 {
            // Mirror Vec3::length_squared == dot(self, self) left to right.
            value = sample.x * sample.x + sample.y * sample.y + sample.z * sample.z;
        }
    }
    scratch[lid] = value;
    workgroupBarrier();

    // Tree reduction: fold the upper half into the lower half until lane 0 holds
    // this workgroup maximum.
    var stride = CLOTH_SLEEP_WORKGROUP / 2u;
    loop {
        if stride == 0u {
            break;
        }
        if lid < stride {
            scratch[lid] = max(scratch[lid], scratch[lid + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    // One lane folds this workgroup partial into the global atomic. Non-negative
    // floats keep their order under bitcast, so integer atomicMax is exact.
    if lid == 0u {
        atomicMax(&max_indicator, bitcast<u32>(scratch[0]));
    }
}
