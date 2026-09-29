//! Variable-length material parameter word-heap decode (design doc §3.3, ABI v4).
//!
//! Byte-exact WESL twin of
//! `prism_render_material::SurfaceParameterBlock::unpack(mask, words).to_full()`:
//! the scene packs each surface as an über-BSDF core (12 words) followed by only
//! the lobes named in `lobe_mask`, written low bit first. Readers gather the
//! material's `parameter_size / 4` words out of the shared `array<u32>` heap
//! into a fixed `array<u32, 36>` (the fully-loaded worst case) and expand them
//! back to a full `PrismSurfaceParameters`, filling absent lobes with their
//! canonical neutral defaults.
//!
//! Kept as a single source so all five consumers (opaque, transparent,
//! visibility, shading_resolve, ssr_repack) decode identically; each reader adds
//! only a tiny inline gather because a `var<storage>` binding is a module-level
//! global and cannot be passed across module boundaries.

import prism_render_scene::shaders::material::PrismSurfaceParameters;

const PRISM_LOBE_EMISSION: u32 = 1u;      // 1 << 0
const PRISM_LOBE_CLEARCOAT: u32 = 2u;     // 1 << 1
const PRISM_LOBE_ANISOTROPY: u32 = 4u;    // 1 << 2
const PRISM_LOBE_SHEEN: u32 = 8u;         // 1 << 3
const PRISM_LOBE_SUBSURFACE: u32 = 16u;   // 1 << 4
const PRISM_LOBE_TRANSMISSION: u32 = 32u; // 1 << 5
const PRISM_LOBE_FACE: u32 = 64u;         // 1 << 6
const PRISM_LOBE_ALL: u32 = 127u;         // (1 << 7) - 1

const PRISM_SURFACE_CORE_WORDS: u32 = 12u;
const PRISM_SURFACE_LOBE_WORDS: u32 = 4u;
/// Fully-loaded worst case: core + every lobe (12 + 6*4).
const PRISM_SURFACE_MAX_WORDS: u32 = 40u;

/// Packed word count for a mask: core plus four words per present lobe. Mirrors
/// `SurfaceParameterBlock::packed_len_words`.
fn prism_packed_word_count(lobe_mask: u32) -> u32 {
    return PRISM_SURFACE_CORE_WORDS
        + countOneBits(lobe_mask & PRISM_LOBE_ALL) * PRISM_SURFACE_LOBE_WORDS;
}

/// The canonical neutral surface (`GpuSurfaceParameters::default`), used both as
/// the base an absent lobe decodes back to and as the value returned when a
/// material's declared range does not fit the bound heap.
fn prism_neutral_surface() -> PrismSurfaceParameters {
    var s: PrismSurfaceParameters;
    s.base_color = vec4<f32>(1.0, 1.0, 1.0, 1.0);
    s.emissive = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    s.metallic = 0.0;
    s.perceptual_roughness = 0.5;
    s.reflectance = 0.5;
    s.ambient_occlusion = 1.0;
    s.normal_scale = 1.0;
    s.alpha_cutoff = 0.5;
    s.transmission = 0.0;
    s.thickness = 0.0;
    s.clearcoat = 0.0;
    s.clearcoat_roughness = 0.5;
    s.anisotropy = 0.0;
    s.anisotropy_rotation = 0.0;
    s.sheen = 0.0;
    s.subsurface = 0.0;
    s.index_of_refraction = 1.5;
    s.dispersion = 0.0;
    s.face_softness = 0.1;
    return s;
}

/// Expand `words` (a material's packed block, low words first) into a full
/// surface using `lobe_mask`. Words beyond the packed length are ignored; absent
/// lobes keep their neutral default. Byte-exact twin of `SurfaceParameterBlock::
/// unpack(mask, words).to_full()`.
fn prism_unpack_surface(words: array<u32, 40>, lobe_mask: u32) -> PrismSurfaceParameters {
    var w = words;
    var s = prism_neutral_surface();

    // Core (12 words); w[10] and w[11] are padding.
    s.base_color = vec4<f32>(
        bitcast<f32>(w[0]),
        bitcast<f32>(w[1]),
        bitcast<f32>(w[2]),
        bitcast<f32>(w[3]),
    );
    s.metallic = bitcast<f32>(w[4]);
    s.perceptual_roughness = bitcast<f32>(w[5]);
    s.reflectance = bitcast<f32>(w[6]);
    s.ambient_occlusion = bitcast<f32>(w[7]);
    s.normal_scale = bitcast<f32>(w[8]);
    s.alpha_cutoff = bitcast<f32>(w[9]);

    // Present lobes, canonical low-bit-first order, four words each.
    var cursor: u32 = PRISM_SURFACE_CORE_WORDS;
    if (lobe_mask & PRISM_LOBE_EMISSION) != 0u {
        s.emissive = vec4<f32>(
            bitcast<f32>(w[cursor]),
            bitcast<f32>(w[cursor + 1u]),
            bitcast<f32>(w[cursor + 2u]),
            bitcast<f32>(w[cursor + 3u]),
        );
        cursor += PRISM_SURFACE_LOBE_WORDS;
    }
    if (lobe_mask & PRISM_LOBE_CLEARCOAT) != 0u {
        s.clearcoat = bitcast<f32>(w[cursor]);
        s.clearcoat_roughness = bitcast<f32>(w[cursor + 1u]);
        cursor += PRISM_SURFACE_LOBE_WORDS;
    }
    if (lobe_mask & PRISM_LOBE_ANISOTROPY) != 0u {
        s.anisotropy = bitcast<f32>(w[cursor]);
        s.anisotropy_rotation = bitcast<f32>(w[cursor + 1u]);
        cursor += PRISM_SURFACE_LOBE_WORDS;
    }
    if (lobe_mask & PRISM_LOBE_SHEEN) != 0u {
        s.sheen = bitcast<f32>(w[cursor]);
        cursor += PRISM_SURFACE_LOBE_WORDS;
    }
    if (lobe_mask & PRISM_LOBE_SUBSURFACE) != 0u {
        s.subsurface = bitcast<f32>(w[cursor]);
        cursor += PRISM_SURFACE_LOBE_WORDS;
    }
    if (lobe_mask & PRISM_LOBE_TRANSMISSION) != 0u {
        s.transmission = bitcast<f32>(w[cursor]);
        s.thickness = bitcast<f32>(w[cursor + 1u]);
        s.index_of_refraction = bitcast<f32>(w[cursor + 2u]);
        s.dispersion = bitcast<f32>(w[cursor + 3u]);
        cursor += PRISM_SURFACE_LOBE_WORDS;
    }
    if (lobe_mask & PRISM_LOBE_FACE) != 0u {
        s.face_softness = bitcast<f32>(w[cursor]);
        cursor += PRISM_SURFACE_LOBE_WORDS;
    }
    return s;
}
