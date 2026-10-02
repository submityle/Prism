// Lighting-channel and NPR light-layer routing — GPU twin of
// `prism_render_shading::light_routing`.
//
// Mirrors the CPU golden arm-for-arm: lighting channels are a per-light /
// per-primitive bitmask visibility gate shared by every illumination model,
// and NPR light layers are a second bitmask routing a light's contribution to
// stylized key/fill/rim accumulation bins. The channel/layer mask helpers and
// the routing gates are bit-identical to the Rust reference.
//
// One deliberate representation note: WGSL core has no 64-bit integer, so the
// cluster active-light bitset is refined one u32 word at a time here, whereas
// the CPU golden folds a full u64 in one call. A cluster with more than 32
// lights walks multiple words; each word matches the CPU reduction exactly.
//
// Self-contained (no intra-crate `import`s, matching ssgi.wesl /
// outline.wesl / volumetrics.wesl) so the WESL compile test validates it
// standalone with no GPU present.

const MAX_LIGHTING_CHANNELS: u32 = 8u;
const MAX_LIGHT_LAYERS: u32 = 4u;
const LIGHTING_CHANNEL_VALID: u32 = (1u << MAX_LIGHTING_CHANNELS) - 1u;
const LIGHT_LAYER_VALID: u32 = (1u << MAX_LIGHT_LAYERS) - 1u;

// Mirrors `LightingChannelMask::from_index`: single-bit mask, empty when
// the index is out of range (fail safe).
fn channel_mask_from_index(index: u32) -> u32 {
    if (index < MAX_LIGHTING_CHANNELS) {
        return 1u << index;
    }
    return 0u;
}

// Mirrors `LightingChannelMask::from_bits`.
fn channel_mask_from_bits(bits: u32) -> u32 {
    return bits & LIGHTING_CHANNEL_VALID;
}

// Mirrors `LightingChannelMask::contains`.
fn channel_mask_contains(mask: u32, index: u32) -> bool {
    return index < MAX_LIGHTING_CHANNELS && (mask & (1u << index)) != 0u;
}

// Mirrors `LightingChannelMask::affects`: masks interact iff they share a
// channel bit.
fn channel_mask_affects(light: u32, primitive: u32) -> bool {
    return (light & primitive) != 0u;
}

// Mirrors `LightLayerMask::from_index`.
fn layer_mask_from_index(index: u32) -> u32 {
    if (index < MAX_LIGHT_LAYERS) {
        return 1u << index;
    }
    return 0u;
}

// Mirrors `LightLayerMask::all`.
fn layer_mask_all() -> u32 {
    return LIGHT_LAYER_VALID;
}

// Mirrors `LightLayerMask::contains`.
fn layer_mask_contains(mask: u32, index: u32) -> bool {
    return index < MAX_LIGHT_LAYERS && (mask & (1u << index)) != 0u;
}

// Matches `LightRouting`: channel visibility gate + stylized layer
// routing.
struct LightRouting {
    channels: u32,
    layers: u32,
}

// Mirrors `LightRouting::lights_primitive` (PBR / hybrid visibility gate).
fn light_routing_lights_primitive(routing: LightRouting, primitive_channels: u32) -> bool {
    return channel_mask_affects(routing.channels, primitive_channels);
}

// Mirrors `LightRouting::contributes_to_layer` (NPR resolution): both the
// channel gate and the layer gate must pass.
fn light_routing_contributes_to_layer(
    routing: LightRouting,
    primitive_channels: u32,
    active_layer: u32,
) -> bool {
    return channel_mask_affects(routing.channels, primitive_channels)
        && layer_mask_contains(routing.layers, active_layer);
}

// Mirrors `cull_lights_by_channel` over a single u32 cluster word: clears
// every active bit whose light does not share a channel with the primitive.
// `channel_masks` is indexed by the light's bit position within the word;
// `count` bounds the valid records (bits at/above it are culled, matching
// the CPU golden's out-of-slice fail-safe).
fn cull_lights_by_channel_word(
    active_bits: u32,
    channel_masks: ptr<function, array<u32, 32>>,
    count: u32,
    primitive: u32,
) -> u32 {
    var remaining = active_bits;
    var result: u32 = 0u;
    loop {
        if (remaining == 0u) {
            break;
        }
        let bit = firstTrailingBit(remaining);
        remaining = remaining & (remaining - 1u);
        if (bit < count) {
            let mask = (*channel_masks)[bit];
            if (channel_mask_affects(mask, primitive)) {
                result = result | (1u << bit);
            }
        }
    }
    return result;
}

// Minimal entry so the module type-checks as a standalone compute kernel under
// the WESL compile test; the resolve pass wires real cluster light bindings
// later.
@compute @workgroup_size(64, 1, 1)
fn light_routing_probe(@builtin(global_invocation_id) gid: vec3<u32>) {
    let prim = channel_mask_from_index(1u);
    var routing: LightRouting;
    routing.channels = channel_mask_from_index(1u);
    routing.layers = layer_mask_all();

    let lit = light_routing_lights_primitive(routing, prim);
    let layered = light_routing_contributes_to_layer(routing, prim, 2u);

    var masks: array<u32, 32>;
    for (var i = 0u; i < 32u; i = i + 1u) {
        masks[i] = channel_mask_from_index(i % MAX_LIGHTING_CHANNELS);
    }
    let culled = cull_lights_by_channel_word(gid.x, &masks, 32u, prim);
    _ = culled + select(0u, 1u, lit) + select(0u, 1u, layered);
}

// --- Render-world cluster-cull refinement -----------------------------------
//
// The live GPU consumer of the per-view routing buffer. One invocation refines
// a single 32-light cluster word against a receiving primitive's channel mask,
// writing two products the resolve/composite fronts read back:
//
//   * `visible_lights[word]` — the channel-gated visibility mask (PBR / hybrid
//     and the shared cluster cull), bit `i` set iff light `word*32 + i` shares
//     a lighting channel with the primitive. Bit-identical to the CPU golden's
//     `cull_lights_by_channel` folded one u32 word at a time.
//   * `layer_lights[word * MAX_LIGHT_LAYERS + layer]` — the NPR per-layer
//     contribution mask, bit `i` set iff that light passes *both* the channel
//     gate and the light-layer gate for `layer`.
//
// Keeping both products live keeps both `LightRouting` fields hot: `channels`
// feeds every gate, `layers` feeds the per-layer split, so no committed ABI
// field is dead.

// Immediate (push-constant) block mirroring the Rust `GpuLightRoutingParams`.
struct LightRoutingParams {
    // Number of valid routing records in `routing_records`.
    light_count: u32,
    // The receiving primitive's lighting-channel bitmask.
    primitive_channels: u32,
    // Number of 32-light cluster words = ceil(light_count / 32).
    word_count: u32,
    // Pad to a 16-byte immediate block.
    _pad: u32,
}

// Per-view routing records: one `LightRouting` per light, read-only.
@group(0) @binding(0) var<storage, read> routing_records: array<LightRouting>;
// Channel-gated visibility mask, one u32 per cluster word (read-write export).
@group(0) @binding(1) var<storage, read_write> visible_lights: array<u32>;
// NPR per-layer contribution masks, `word_count * MAX_LIGHT_LAYERS` u32s
// (read-write export), laid out layer-minor: `[word * MAX_LIGHT_LAYERS + L]`.
@group(0) @binding(2) var<storage, read_write> layer_lights: array<u32>;

var<immediate> light_routing_params: LightRoutingParams;

@compute @workgroup_size(64, 1, 1)
fn light_routing_cull_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let word = gid.x;
    if (word >= light_routing_params.word_count) {
        return;
    }
    let base = word * 32u;
    let primitive = light_routing_params.primitive_channels;

    // Gather this word's channel masks and how many valid records it spans.
    // Records beyond `light_count` fail safe to an empty mask (never lit),
    // matching the CPU golden's out-of-slice behaviour.
    var masks: array<u32, 32>;
    var count = 0u;
    for (var i = 0u; i < 32u; i = i + 1u) {
        let idx = base + i;
        if (idx < light_routing_params.light_count) {
            masks[i] = channel_mask_from_bits(routing_records[idx].channels);
            count = count + 1u;
        } else {
            masks[i] = 0u;
        }
    }

    // Every record this word spans starts spatially active; the channel cull
    // clears the ones that do not share a channel with the primitive.
    var active_word: u32;
    if (count >= 32u) {
        active_word = 0xFFFFFFFFu;
    } else {
        active_word = (1u << count) - 1u;
    }
    visible_lights[word] = cull_lights_by_channel_word(active_word, &masks, count, primitive);

    // NPR resolution: one refined mask per light layer, each gated on the
    // channel gate *and* the light-layer gate (`contributes_to_layer`).
    for (var layer = 0u; layer < MAX_LIGHT_LAYERS; layer = layer + 1u) {
        var layer_result = 0u;
        for (var i = 0u; i < 32u; i = i + 1u) {
            let idx = base + i;
            if (idx < light_routing_params.light_count) {
                var routing: LightRouting;
                routing.channels = channel_mask_from_bits(routing_records[idx].channels);
                routing.layers = routing_records[idx].layers & LIGHT_LAYER_VALID;
                if (light_routing_contributes_to_layer(routing, primitive, layer)) {
                    layer_result = layer_result | (1u << i);
                }
            }
        }
        layer_lights[word * MAX_LIGHT_LAYERS + layer] = layer_result;
    }
}
