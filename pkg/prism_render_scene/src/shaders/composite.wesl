//! HDR scene-color -> view-target composite.
//!
//! Terminal step of the visibility -> classify -> shade -> *composite* chain.
//! `shading_resolve.wesl` writes linear pre-exposure HDR radiance into the
//! `scene_color` storage texture for every *covered* pixel; uncovered pixels
//! are never touched, so their contents are undefined.  This fullscreen pass
//! runs after the core-3d main pass (which has already cleared the view target
//! to the camera clear color and drawn any non-Prism geometry) and *before*
//! Bevy's tonemapping node.  For every covered pixel it copies the resolved
//! linear radiance into the view target; uncovered pixels are `discard`ed so
//! the main pass' clear/background survives untouched.
//!
//! The only shading applied here is the *pre-exposure* multiply: the resolved
//! radiance is scaled by the eye-adaptation `adapted_exposure` the auto-exposure
//! resolve pass wrote (a stationary `1.0` when auto-exposure is disabled).  No
//! tonemapping happens here: the value written is linear pre-exposed HDR
//! radiance in the view target's working space, exactly what the downstream
//! tonemapping node expects.  Keeping tonemapping in Bevy's node avoids
//! double-mapping and lets the camera's configured tonemapper apply uniformly.

// Sentinel packed into the visibility targets for uncovered pixels. Matches
// `VisibilityPixel::is_valid` on the CPU side and `VISIBILITY_BACKGROUND` in
// `shading_resolve.wesl`.
const VISIBILITY_BACKGROUND: u32 = 0xffffffffu;

struct FullscreenVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// Single oversized triangle covering the viewport. Vertex 0 -> (-1,1),
// vertex 1 -> (3,1) [uv.x=2], vertex 2 -> (-1,-3) [uv.y=2]; the clamp to the
// [-1,1] clip box discards the excess. No vertex buffer is bound.
@vertex
fn vertex(@builtin(vertex_index) vertex_index: u32) -> FullscreenVertexOutput {
    let uv = vec2<f32>(
        f32((vertex_index << 1u) & 2u),
        f32(vertex_index & 2u),
    );
    var out: FullscreenVertexOutput;
    out.uv = uv;
    out.position = vec4<f32>(uv * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0), 0.0, 1.0);
    return out;
}

// group 0: the resolve outputs, all sampled by integer `textureLoad` so the
// composite stays a 1:1 pixel copy and needs no sampler.
@group(0) @binding(0) var scene_color: texture_2d<f32>;
@group(0) @binding(1) var visibility_ids: texture_2d<u32>;
@group(0) @binding(2) var visibility_metadata: texture_2d<u32>;

// Persistent per-view exposure state written by the auto-exposure resolve pass
// (`exposure.wesl`). `adapted_exposure` is the eye-adaptation multiplier applied
// to radiance before it reaches Bevy's tonemapping node; it stays a stationary
// `1.0` when auto-exposure is disabled, so the multiply is a no-op then. The
// three trailing fields mirror `ExposureState` in `exposure.wesl` byte-for-byte.
struct ExposureState {
    adapted_exposure: f32,
    adapted_luminance: f32,
    average_luminance: f32,
    _pad0: f32,
}
@group(0) @binding(3) var<storage, read> exposure_state: ExposureState;

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let coord = vec2<i32>(in.position.xy);
    let ids = textureLoad(visibility_ids, coord, 0);
    let metadata = textureLoad(visibility_metadata, coord, 0);
    // Uncovered pixels carry the background sentinel in both targets; leave the
    // main pass' clear/background in place by discarding here. The guard mirrors
    // the covered-pixel test in `shading_resolve.wesl` byte-for-byte.
    if ids.x == VISIBILITY_BACKGROUND || metadata.x == VISIBILITY_BACKGROUND {
        discard;
    }
    // Linear HDR radiance, pre-exposed by the eye-adaptation multiplier; Bevy's
    // tonemapping node maps it to the display. When auto-exposure is disabled
    // the multiplier is a stationary `1.0`, so this is a straight copy.
    let radiance = textureLoad(scene_color, coord, 0).rgb * exposure_state.adapted_exposure;
    return vec4<f32>(radiance, 1.0);
}
