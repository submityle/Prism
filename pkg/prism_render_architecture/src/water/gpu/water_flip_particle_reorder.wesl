//! `GPU` compute twin of the `FLIP`/`APIC` particle reorder (gather) pass that
//! consumes the counting-sort output.
//!
//! This is the integration pass that turns the counting-sort chain (histogram
//! -> exclusive scan -> scatter) into a payoff: it gathers the packed particle
//! records into cell-major order so the measured `MAC` transfer bottleneck
//! (`P2G`/`G2P`) reads a compact face neighbourhood per workgroup instead of
//! chasing spawn-order scatter. Crucially it leaves the real-machine-validated
//! transfer kernels untouched — they run unchanged over the reordered buffer —
//! so the existing `P2G`/`G2P` parity goldens stay valid.
//!
//! Compute twin of the `CPU` golden
//! `prism_render_architecture::water::gpu::flip_particle_reorder_kernel::dispatch_flip_particle_reorder`.
//! The sandbox has no `GPU`, so this module is validated by `naga`/`wesl`
//! compilation and by its bit-exact `CPU` twin; it is pure index-gather
//! bookkeeping — no floats touched beyond a verbatim copy, no `f32` equality
//! test, no AI/ML.
//!
//! The entry-point name below is byte-identical to the one the host
//! `prism_render_architecture::water::kernels::WaterKernel::wesl_entry_point`
//! contract emits (`water_flip_particle_reorder`), so the pipeline cache keys
//! resolve against this module. It is a linear per-slot pass dispatched at
//! `@workgroup_size(64)` over the `Particle` domain, binding one read storage
//! buffer (`sorted_indices`), one read storage buffer (`particles_in`), one
//! read-write storage buffer (`particles_out`), and one uniform (`params`).
//!
//! ## What it does
//!
//! One invocation per output slot `s` reads `src = sorted_indices[s]` and
//! copies that particle's packed record (`FLIP_REORDER_PARTICLE_FLOATS` `f32`
//! lanes: position+active, velocity, three `APIC` affine rows) verbatim from
//! `particles_in[src]` into `particles_out[s]`. Because `sorted_indices` is
//! cell-major, `particles_out` is a cell-major permutation of the in-grid
//! particles. Out-of-range slots and a `src` beyond the source count are
//! skipped, never indexed out of bounds.

/// `f32` lanes per packed `FLIP`/`APIC` particle record: position+active,
/// velocity, and the three `APIC` affine rows (`vec4` padded), matching the
/// `P2G`/`G2P` particle `ABI`.
const FLIP_REORDER_PARTICLE_FLOATS: u32 = 20u;

/// Dispatch bounds for the reorder gather.
struct FlipReorderParams {
    /// Number of output slots (the in-grid particle total, `offsets[last]`).
    sorted_count: u32,
    /// Number of particles in the source pool (bounds the gather read).
    src_count: u32,
    /// Padding to a 16-byte uniform stride.
    pad0: u32,
    /// Padding to a 16-byte uniform stride.
    pad1: u32,
}

// Cell-major original particle indices from the scatter pass.
@group(0) @binding(0) var<storage, read> sorted_indices: array<u32>;
// Source particle pool, packed `FLIP_REORDER_PARTICLE_FLOATS` lanes each.
@group(0) @binding(1) var<storage, read> particles_in: array<f32>;
// Destination pool, cell-major, same packing.
@group(0) @binding(2) var<storage, read_write> particles_out: array<f32>;
// Dispatch bounds.
@group(0) @binding(3) var<uniform> params: FlipReorderParams;

/// Particle reorder: one invocation per output slot gathers its source record.
///
/// Out-of-range slots and out-of-range source indices are skipped. After this
/// pass `particles_out` holds the in-grid particles in cell-major order, ready
/// for a coherent `P2G`/`G2P` sweep.
@compute @workgroup_size(64)
fn water_flip_particle_reorder(@builtin(global_invocation_id) gid: vec3<u32>) {
    let s = gid.x;
    if s >= params.sorted_count {
        return;
    }
    let src = sorted_indices[s];
    if src >= params.src_count {
        return;
    }
    let dst_base = s * FLIP_REORDER_PARTICLE_FLOATS;
    let src_base = src * FLIP_REORDER_PARTICLE_FLOATS;
    // Bounds-guard both ends against a mis-sized buffer before copying.
    if dst_base + FLIP_REORDER_PARTICLE_FLOATS > arrayLength(&particles_out)
        || src_base + FLIP_REORDER_PARTICLE_FLOATS > arrayLength(&particles_in) {
        return;
    }
    var lane: u32 = 0u;
    loop {
        if lane >= FLIP_REORDER_PARTICLE_FLOATS {
            break;
        }
        particles_out[dst_base + lane] = particles_in[src_base + lane];
        lane = lane + 1u;
    }
}
