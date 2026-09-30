//! `wgpu` compute twin of Prism's per-texel transmittance-sample binning
//! (`prism_render_architecture::hair::deep_transmittance::bin_samples`).
//!
//! Deep-opacity and voxel self-shadow both start from the same fan-out: a flat
//! stream of strand samples, each tagged with the light texel/ray it lands on,
//! must be routed into one bucket per texel before either the layered or the
//! froxel accumulation can run per texel (design §6/§11 自阴影分桶). The
//! reference `bin_samples` is the pure, deterministic partition pass that does
//! this — it preserves input order inside every bucket and silently drops any
//! sample whose texel index is out of range. This kernel is the on-device twin
//! of that partition.
//!
//! One invocation owns one destination texel `t`: it walks the whole indexed
//! sample stream in input order and, for every sample tagged `t`, appends its
//! `(depth, opacity)` value into `t`'s own compacted output slice
//! (`texel_ranges[t].start ..`). Because each thread writes only its own
//! disjoint slice there is no cross-thread contention and no atomics: the host
//! pre-counts each texel's population (skipping out-of-range tags exactly like
//! the golden) and hands the kernel an exclusive-prefix `texel_ranges` layout.
//! Walking the stream in ascending index and appending in encounter order makes
//! each bucket's contents bit-identical to the reference's stable push order.
//!
//! Bit-faithfulness: the kernel only copies `vec2<f32>` payloads — no arithmetic
//! — so a passing parity test proves the routing (which sample lands in which
//! bucket, and in what order) matches the golden exactly, not merely to a
//! tolerance. Out-of-range tags (`texel >= texel_count`) are never matched by
//! any thread, mirroring the golden's skip. `texel_count` is clamped to at
//! least `1` on the host, matching `TransmittanceBins::new`.

/// Binning parameters. `sample_count` bounds the per-thread stream walk;
/// `texel_count` bounds the dispatch (one invocation per destination texel).
/// `16`-byte scalar-packed so the uniform block is a multiple of 16.
struct HairBinParams {
    sample_count: u32,
    texel_count: u32,
    pad0: u32,
    pad1: u32,
}

/// One compacted per-texel output slice: entries `start .. start + count`, in
/// encounter (input) order. `count` is unused by the kernel (each thread stops
/// when the stream ends) but kept one-to-one with the reference bucket layout.
struct HairTexelRange {
    start: u32,
    count: u32,
}

@group(0) @binding(0) var<uniform> params: HairBinParams;

// Per-sample destination texel tag, in input order.
@group(0) @binding(1) var<storage, read> sample_texels: array<u32>;

// Per-sample payload, in input order: `.x` = depth, `.y` = opacity.
@group(0) @binding(2) var<storage, read> sample_values: array<vec2<f32>>;

// Per-texel compacted output ranges into `out_values` (exclusive prefix).
@group(0) @binding(3) var<storage, read> texel_ranges: array<HairTexelRange>;

// Flat per-texel bucket pool, texel-major: `out_values[start + k]` is the
// `k`-th sample (in input order) routed to that texel.
@group(0) @binding(4) var<storage, read_write> out_values: array<vec2<f32>>;

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if t >= params.texel_count {
        return;
    }

    // Scan the whole stream in input order; append every sample tagged `t` into
    // this texel's disjoint slice, preserving encounter order = golden order.
    let start = texel_ranges[t].start;
    var local = 0u;
    for (var i = 0u; i < params.sample_count; i = i + 1u) {
        if sample_texels[i] == t {
            out_values[start + local] = sample_values[i];
            local = local + 1u;
        }
    }
}
