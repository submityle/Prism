// Order-independent-transparency k-layer composite twin.
//
// Device mirror of the CPU goldens
// `prism_render_architecture::hair::oit_frontend::composite_transmittance` and
// `composite_coverage`. Each hair pixel owns a resolved `front-to-back` stack of
// up to MAX_OIT_LAYERS (8) nearest fragments plus a flattened tail opacity; this
// kernel evaluates that pixel's multi-layer alpha-blend (MLAB) transmittance and
// its complementary coverage, one thread per pixel - the array-in/array-out form
// the OIT resolve pass consumes per pixel.
//
// Transmittance is the fraction of background light surviving the stack:
// `product_i (1 - alpha_i)` over the resolved layers times the tail's own
// `(1 - tail_alpha)`, clamped to [0, 1]. Coverage is its complement
// `1 - transmittance`, also clamped. AMD TressFX and UE5 Groom both resolve
// their per-pixel linked list with a variant of this k-layer MLAB composite.
//
// The host flattens each pixel's resolved stack into a fixed 8-slot alpha row,
// padding the unused tail slots with `0.0`. Because `(1 - 0) == 1` is the exact
// multiplicative identity in IEEE-754, multiplying through all 8 slots in layer
// order and then the tail reproduces the golden's shorter product bit-for-bit:
// the padding never perturbs the result. The alphas are pre-sanitised to
// [0, 1] on the host exactly as the golden's `HairFragment::new` does, so the
// kernel does no clamping of its own beyond the final saturation.
//
// The composite is transcendental-free - only subtracts, multiplies and a
// clamp in the portable core-WGSL subset, no `exp`, `pow` or optional device
// feature - so the twin runs unmodified on Metal, Vulkan and DX12. The result
// is a plain product chain, so it is matched against a tolerance (abs_diff <
// 1e-4 or rel_diff < 1e-3) for robustness even though no fma is involved.
//
// Provenance: d'Eon/TressFX k-layer MLAB alpha compositing plus wgpu compute
// dispatch; no Unreal Engine source or derived code.

struct Params {
    // Number of pixels in `alphas` (/8) and `tails`, and half of `out_values`.
    pixel_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Per pixel, 8 contiguous f32 alphas (resolved layers, padded with 0 past the
// stack depth), each pre-sanitised to [0, 1] on the host.
@group(0) @binding(1) var<storage, read> alphas: array<f32>;
// Per pixel, one f32: the flattened overflow tail alpha in [0, 1].
@group(0) @binding(2) var<storage, read> tails: array<f32>;
// Per pixel, two f32: `out_values[2i]` = transmittance, `out_values[2i+1]` =
// coverage, both in [0, 1].
@group(0) @binding(3) var<storage, read_write> out_values: array<f32>;

// Fixed resolved-layer slot count; matches the golden's `MAX_OIT_LAYERS`.
const MAX_OIT_LAYERS: u32 = 8u;

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.pixel_count) {
        return;
    }

    // Multiply the per-layer (1 - alpha) terms in layer order, then the tail,
    // exactly as the golden does. Padded slots carry alpha = 0 so their factor
    // is the exact identity 1.0 and does not perturb the product.
    var transmittance = 1.0;
    let base = i * MAX_OIT_LAYERS;
    for (var j = 0u; j < MAX_OIT_LAYERS; j = j + 1u) {
        transmittance = transmittance * (1.0 - alphas[base + j]);
    }
    transmittance = transmittance * (1.0 - tails[i]);
    transmittance = clamp(transmittance, 0.0, 1.0);

    let coverage = clamp(1.0 - transmittance, 0.0, 1.0);

    out_values[2u * i] = transmittance;
    out_values[2u * i + 1u] = coverage;
}
