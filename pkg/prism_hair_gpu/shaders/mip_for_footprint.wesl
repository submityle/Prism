// Compute twin of Prism's card-chain mip classifier
// (prism_render_architecture::hair::card_bake::mip_for_footprint): the mip
// level a footprint's longest side sits at within a power-of-two card chain.
// One thread per (max_dim, base_texels, max_mip) query counts how many times
// the footprint's longest side can be doubled before it reaches base_texels,
// clamped to max_mip; a zero footprint reports max_mip (coarsest level).
//
// The count is an explicit integer doubling loop (not log2): size starts at
// max_dim and is doubled while the doubled value still fits within base and the
// mip counter is below max_mip. The doubling uses saturating u32 arithmetic
// exactly like the golden's `u32::saturating_mul(2)` — a value past half of
// u32::MAX saturates to u32::MAX instead of wrapping, so a huge footprint can
// never alias back under base.
//
// The trip count is data-dependent (read from the per-thread query), so the
// loop diverges between invocations in the same workgroup. Everything is pure
// integer arithmetic with no float and no transcendental, so the device
// reproduces the scalar reference exactly; the parity test compares the integer
// results directly. Only core-WGSL integer ops are used, so it runs unmodified
// on Metal, Vulkan and DX12 with no optional device feature.

struct Params {
    element_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One query: the footprint's longest side max_dim, the full-resolution card
// side base_texels, and the coarsest mip index max_mip.
struct MipQuery {
    max_dim: u32,
    base_texels: u32,
    max_mip: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Per-thread footprint queries; length element_count.
@group(0) @binding(1) var<storage, read> queries: array<MipQuery>;
// One mip level per query, in input order.
@group(0) @binding(2) var<storage, read_write> out: array<u32>;

// Mirror the golden `u32::saturating_mul(2)`: a value whose double would
// overflow u32 saturates to u32::MAX instead of wrapping.
fn saturating_double(size: u32) -> u32 {
    if size > 0x7fffffffu {
        return 0xffffffffu;
    }
    return size * 2u;
}

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.element_count {
        return;
    }
    let max_dim = queries[idx].max_dim;
    let max_mip = queries[idx].max_mip;

    // A zero footprint reports the coarsest level (mirrors the golden's early
    // return before `base` is even formed).
    if max_dim == 0u {
        out[idx] = max_mip;
        return;
    }

    let base = max(queries[idx].base_texels, 1u);
    var mip: u32 = 0u;
    var size: u32 = max_dim;
    // Integer doubling loop (mirrors the golden's `while size*2 <= base && mip
    // < max_mip`): recompute the saturating double each iteration so a saturated
    // size can never slip back under base.
    loop {
        let doubled = saturating_double(size);
        if !(doubled <= base && mip < max_mip) {
            break;
        }
        size = doubled;
        mip = mip + 1u;
    }

    out[idx] = mip;
}
