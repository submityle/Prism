//! GPU per-texel forward-scatter crossing-count packing for hair
//! dual-scattering.
//!
//! Compute twin of the CPU golden reference
//! `prism_render_architecture::hair::forward_scatter_layout::build_forward_scatter_map`.
//! It is the *additive* sibling of `hair_deep_opacity.wesl`: where deep opacity
//! bakes a cumulative transmittance `T = product(1 - alpha)`, Zinke global
//! multiple scattering needs the coverage-weighted count of strands a light ray
//! has crossed `n(d) = sum(alpha_i for depth_i <= d)` — the shading side then
//! raises the material forward attenuation `a_f^n` and angular spread
//! `n * beta_f^2` (the term light hair must not go dead without). That count is
//! a pure geometry / visibility quantity and cannot be recovered from the deep
//! opacity product (the product discards each `alpha_i`), so — like the
//! transmittance curve — it lives on the shared self-shadow layout side (design
//! §7.4 dual-scattering). The per-ray accumulation lives in
//! `prism_render_architecture::hair::dual_scattering::build_forward_scatter`;
//! this kernel is the massively parallel cross-texel *packing* half: one
//! invocation owns one light texel and slices its samples into `layer_count`
//! equal-width depth layers.
//!
//! Host bookkeeping stays on the CPU, exactly as the CPU module documents the
//! *device-side sort as separate scheduling*: the caller bins strand samples
//! per texel and stably sorts each texel's slice by light-space depth
//! (`total_cmp`), then hands this kernel a dense `texel_ranges` array into a
//! shared, already-sorted `samples` buffer. Only the per-texel layer slicing
//! and `alpha`-sum fans out here — no sort on the GPU — so the packed slab is
//! bit-for-bit the CPU golden.
//!
//! Bit-faithfulness: the running sum walks samples in the same pre-sorted order
//! with the same `clamp(opacity, 0, 1)` addend and `<= boundary` comparison,
//! the layer boundary is spelled `start + width * (f32(i) + 1.0)` with the same
//! left-associative grouping, and `width = (end - start) / f32(layer_count)`.
//! An empty texel packs a zero-crossing row (all `0.0`) with `near`/`step` of
//! `0`, matching `pack_bucket`. The curve is monotonic non-decreasing (additive)
//! rather than the non-increasing transmittance product. A `GPU` is now
//! available, so this shader is compile- and type-checked through the same
//! render-world `ShaderCache` / `wesl` pipeline the runtime uses (see
//! `shader_tests`), with on-device dispatch/perf calibration still awaiting real
//! hardware.

/// Forward-scatter packing parameters. `texel_count` bounds the dispatch (one
/// invocation per light texel); `layer_count` is the fixed depth-layer stride,
/// clamped to at least `1`; `start_offset` is the near bias, clamped to `>= 0`.
struct HairForwardScatterParams {
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
// `alpha` coverage contribution (`0..=1`), pre-sorted ascending per texel.
@group(0) @binding(0) var<storage, read> samples: array<vec2<f32>>;

// Per-texel compacted ranges into `samples`.
@group(0) @binding(1) var<storage, read> texel_ranges: array<HairTexelRange>;

// Per-texel front depth of the first layer; length `texel_count`.
@group(0) @binding(2) var<storage, read_write> near_depth: array<f32>;

// Per-texel uniform layer width; length `texel_count`.
@group(0) @binding(3) var<storage, read_write> layer_step: array<f32>;

// Flat texel-major crossing-count slab: texel `t` layer `l` at
// `crossings[t * layer_count + l]`.
@group(0) @binding(4) var<storage, read_write> crossings: array<f32>;

var<immediate> params: HairForwardScatterParams;

@compute @workgroup_size(64)
fn hair_forward_scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
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

    // Empty texel: zero-crossing row, zero near/step.
    if count == 0u {
        for (var i = 0u; i < layers; i = i + 1u) {
            crossings[base + i] = 0.0;
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

    var running = 0.0;
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
                running = running + clamp(s.y, 0.0, 1.0);
                next = next + 1u;
            } else {
                break;
            }
        }
        crossings[base + i] = running;
    }

    near_depth[t] = start_d;
    layer_step[t] = width;
}
