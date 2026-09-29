//! GPU per-texel deep opacity map packing for hair self-shadowing.
//!
//! Compute twin of the CPU golden reference
//! `prism_render_architecture::hair::deep_opacity_layout::build_deep_opacity_map`.
//! A deep opacity map covers a whole light-space texture: thin overlapping
//! strands make a single shadow-map depth useless, so — like `UE5` Groom and
//! AMD `TressFX` self-shadow bakes — every light texel accumulates strand
//! opacity into a fixed number of depth layers whose cumulative transmittance
//! `T = product(1 - alpha)` a shading pass decodes with a constant stride
//! (design §5/§11 自阴影透射). The per-ray accumulation lives in
//! `hair_transmittance.wesl`; this kernel is the massively parallel cross-texel
//! *packing* half: one invocation owns one light texel and slices its samples
//! into `layer_count` equal-width depth layers.
//!
//! Host bookkeeping stays on the CPU, exactly as the CPU module documents the
//! *device-side sort as separate scheduling*: the caller bins strand samples
//! per texel and stably sorts each texel's slice by light-space depth
//! (`total_cmp`), then hands this kernel a dense `texel_ranges` array into a
//! shared, already-sorted `samples` buffer. Only the per-texel layer slicing
//! and `alpha`-composite fans out here — no sort on the GPU — so the packed
//! slab is bit-for-bit the CPU golden.
//!
//! Bit-faithfulness: the running product walks samples in the same pre-sorted
//! order with the same `1 - clamp(opacity, 0, 1)` factor and `<= boundary`
//! comparison, the layer boundary is spelled `start + width * (f32(i) + 1.0)`
//! with the same left-associative grouping, and `width = (end - start) /
//! f32(layer_count)`. An empty texel packs a fully transmissive row (all `1.0`)
//! with `near`/`step` of `0`, matching `pack_bucket`. A `GPU` is now available,
//! so this shader is compile- and type-checked through the same render-world
//! `ShaderCache` / `wesl` pipeline the runtime uses (see `shader_tests`), with
//! on-device dispatch/perf calibration still awaiting real hardware.

/// Deep-opacity packing parameters. `texel_count` bounds the dispatch (one
/// invocation per light texel); `layer_count` is the fixed depth-layer stride,
/// clamped to at least `1`; `start_offset` is the near bias, clamped to `>= 0`.
struct HairDeepOpacityParams {
    texel_count: u32,
    layer_count: u32,
    start_offset: f32,
}

/// One compacted, pre-sorted per-texel slice into `samples`: entries
/// `start .. start + count`, ascending in light-space depth. The host has
/// binned and stably sorted the slice, mirroring `bin_samples` + the
/// `total_cmp` sort inside `pack_bucket`.
struct HairTexelRange {
    start: u32,
    count: u32,
}

// Shared flat sample buffer: `.x` is light-space depth (`>= 0`), `.y` is the
// `alpha` opacity contribution (`0..=1`), pre-sorted ascending per texel.
@group(0) @binding(0) var<storage, read> samples: array<vec2<f32>>;

// Per-texel compacted ranges into `samples`.
@group(0) @binding(1) var<storage, read> texel_ranges: array<HairTexelRange>;

// Per-texel front depth of the first layer; length `texel_count`.
@group(0) @binding(2) var<storage, read_write> near_depth: array<f32>;

// Per-texel uniform layer width; length `texel_count`.
@group(0) @binding(3) var<storage, read_write> layer_step: array<f32>;

// Flat texel-major transmittance slab: texel `t` layer `l` at
// `transmittance[t * layer_count + l]`.
@group(0) @binding(4) var<storage, read_write> transmittance: array<f32>;

var<immediate> params: HairDeepOpacityParams;

@compute @workgroup_size(64)
fn hair_deep_opacity(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if t >= params.texel_count {
        return;
    }

    // `layer_count` is clamped to at least `1`.
    var layers = params.layer_count;
    if layers < 1u {
        layers = 1u;
    }

    let start = texel_ranges[t].start;
    let count = texel_ranges[t].count;
    let base = t * layers;

    // Empty texel: fully transmissive row, zero near/step.
    if count == 0u {
        for (var i = 0u; i < layers; i = i + 1u) {
            transmittance[base + i] = 1.0;
        }
        near_depth[t] = 0.0;
        layer_step[t] = 0.0;
        return;
    }

    let shallowest = samples[start].x;
    let deepest = samples[start + count - 1u].x;
    var so = params.start_offset;
    if so < 0.0 {
        so = 0.0;
    }
    let start_d = shallowest + so;
    let end = max(deepest, start_d);
    let width = (end - start_d) / f32(layers);

    var running = 1.0;
    var next = 0u; // index within this texel's slice, `0..count`
    for (var i = 0u; i < layers; i = i + 1u) {
        // Layer `i`'s far boundary; the last layer pins to `end` and a
        // zero-width span degenerates every boundary to `end`.
        var boundary = end;
        if i + 1u != layers {
            if width > 0.0 {
                boundary = start_d + width * (f32(i) + 1.0);
            } else {
                boundary = end;
            }
        }
        loop {
            if next >= count {
                break;
            }
            let s = samples[start + next];
            if s.x <= boundary {
                running = running * (1.0 - clamp(s.y, 0.0, 1.0));
                next = next + 1u;
            } else {
                break;
            }
        }
        transmittance[base + i] = running;
    }

    near_depth[t] = start_d;
    layer_step[t] = width;
}
