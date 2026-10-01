// Compute twin of Prism's adaptive variable-node transmittance curve lookup
// (prism_render_architecture::hair::adaptive_transmittance::sample_transmittance,
// which delegates to the private sample_nodes_shadow receiver-query rule). One
// thread per receiver depth evaluates the piecewise-linear reconstruction of a
// shared compressed curve: fully lit (1.0) in front of the frontmost node, the
// deepest node's value at or beyond the last node, and a linear interpolation
// of the two bracketing nodes in between.
//
// The curve nodes are a shared read-only storage buffer, flattened as (depth,
// transmittance) pairs (2 floats per node, tightly packed to dodge the std430
// vec2 stride); the per-thread variation is the receiver depth read from a flat
// storage array. The golden performs no sanitisation of the node data or the
// query depth, so neither does this kernel -- callers keep every value finite.
//
// The only arithmetic is a subtract, a divide and a mul/add a GPU may fuse, so
// CPU and GPU agree within the documented fma tolerance rather than bit-for-bit.
// It uses no transcendental math, so it runs unmodified on Metal, Vulkan and
// DX12.

struct Params {
    node_count: u32,
    query_count: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Curve nodes shared by every thread, flattened (depth, transmittance) pairs.
@group(0) @binding(1) var<storage, read> nodes: array<f32>;
// Receiver query depths, one per thread.
@group(0) @binding(2) var<storage, read> depths: array<f32>;
// One reconstructed transmittance per query depth.
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

// Depth of node `i` in the flattened (depth, transmittance) pair layout.
fn node_depth(i: u32) -> f32 {
    return nodes[i * 2u];
}

// Transmittance of node `i` in the flattened pair layout.
fn node_t(i: u32) -> f32 {
    return nodes[i * 2u + 1u];
}

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.query_count {
        return;
    }
    let depth = depths[idx];

    // Empty curve reads fully transmissive everywhere (golden: nodes.is_empty).
    if params.node_count == 0u {
        out[idx] = 1.0;
        return;
    }

    // Fully lit in front of the frontmost node.
    if depth <= node_depth(0u) {
        out[idx] = 1.0;
        return;
    }

    let last = params.node_count - 1u;
    // The deepest node's value at or beyond the last node.
    if depth >= node_depth(last) {
        out[idx] = node_t(last);
        return;
    }

    // Linear interpolation of the first bracketing pair (nodes[i], nodes[i+1])
    // whose far endpoint reaches `depth`; mirrors the golden windows(2) scan and
    // its span == 0 guard. Trip count is data-dependent (divergent within a
    // workgroup) just like the golden loop.
    var i: u32 = 0u;
    loop {
        if i >= last {
            break;
        }
        let d_hi = node_depth(i + 1u);
        if depth <= d_hi {
            let d_lo = node_depth(i);
            let t_lo = node_t(i);
            let t_hi = node_t(i + 1u);
            let span = d_hi - d_lo;
            var t: f32 = 0.0;
            if span > 0.0 {
                t = (depth - d_lo) / span;
            }
            out[idx] = t_lo + (t_hi - t_lo) * t;
            return;
        }
        i = i + 1u;
    }

    // Fallthrough matches the golden's trailing `nodes[last].transmittance`.
    out[idx] = node_t(last);
}
