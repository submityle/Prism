// Weather-map coverage-mass twin: sums the `coverage` channel over every cell of
// a weather field, mirroring the CPU golden `WeatherField::total_coverage` (in
// `prism_render_architecture::volumetric::weather`).
//
// `total_coverage` is the conserved quantity the advection mass tests track: a
// divergence-free wind should move cloud around without creating or destroying
// coverage, so its whole-field sum stays near-constant. The reduction itself is
// pure, so it twins cleanly. Many fields (e.g. a batch of weather snapshots, or
// the pre/post-advection pair a mass test compares) are reduced in parallel:
// each invocation owns one field, walks its own [offset, offset+count) slice of
// the shared flattened coverage array, and accumulates the sum in ascending
// cell order to match the CPU golden's left-to-right `while` loop bit for bit.
// An empty field encloses no coverage, so its result is 0, exactly as on the
// CPU. It uses only the portable core-WGSL subset, so it runs unmodified on
// Metal, Vulkan and DX12.
//
// Provenance: straightforward channel reduction over a weather grid; no Unreal
// Engine source or derived code.

struct Params {
    field_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// Where one field's coverage samples live in the shared flattened array.
// Matches `GpuFieldRange` in `src/total_coverage.rs`.
struct FieldRange {
    offset: u32,
    count: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> coverage: array<f32>;
@group(0) @binding(2) var<storage, read> ranges: array<FieldRange>;
@group(0) @binding(3) var<storage, read_write> results: array<f32>;

// Sum of one field's [offset, offset+count) coverage slice, mirroring
// `WeatherField::total_coverage`.
fn field_total(offset: u32, count: u32) -> f32 {
    var sum = 0.0;
    var i = 0u;
    loop {
        if (i >= count) {
            break;
        }
        sum = sum + coverage[offset + i];
        i = i + 1u;
    }
    return sum;
}

@compute @workgroup_size(64)
fn total_coverage_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.field_count) {
        return;
    }
    let r = ranges[idx];
    results[idx] = field_total(r.offset, r.count);
}
