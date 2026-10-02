//! `wgpu` compute twin of the particle shading router golden
//! ([`shading`](prism_render_architecture::particle::shading), design §12,
//! §15-§21).
//!
//! The particle subsystem decides, per emitter/renderer, how a shading model
//! specializes a draw kernel: the render phase it submits into, the per-particle
//! attribute footprint it reads, the shared lighting services it subscribes to,
//! the order-independent-transparency route it composites through, the
//! motion-vector requirements it writes, its six-way volumetric lighting
//! response, its cel-band quantization, and its deep-shadow tier. The `CPU`
//! golden
//! [`render_phase_for`](prism_render_architecture::particle::shading::render_phase_for),
//! [`attribute_footprint`](prism_render_architecture::particle::shading::attribute_footprint),
//! [`lighting_services`](prism_render_architecture::particle::shading::lighting_services),
//! [`resolve_oit_route`](prism_render_architecture::particle::shading::resolve_oit_route),
//! [`motion_vector_request`](prism_render_architecture::particle::shading::motion_vector_request),
//! [`six_way_response`](prism_render_architecture::particle::shading::six_way_response),
//! [`quantize_cel_bands`](prism_render_architecture::particle::shading::quantize_cel_bands),
//! [`resolve_deep_shadow`](prism_render_architecture::particle::shading::resolve_deep_shadow)
//! and
//! [`resolve_shading_program`](prism_render_architecture::particle::shading::resolve_shading_program)
//! own that math; [`GpuShading`] is the on-device twin that runs one thread per
//! query and reproduces every lane. A passing real-device parity test is
//! therefore direct evidence the ported kernel folds the same classification
//! branches, the same `u32` bit unions and the same directional arithmetic the
//! reference does, not merely that its shader compiles.
//!
//! # What is twinned
//!
//! Each thread evaluates, for one [`GpuShadingQuery`]:
//!
//! - the render phase for a [`BlendMode`](prism_render_architecture::particle::sort_cull::BlendMode);
//! - the attribute footprint of an
//!   [`EmberShadingModel`](prism_render_architecture::particle::EmberShadingModel)
//!   (the `u32` bit union of its lobes for a hybrid) and a standalone
//!   [`ShadingAttributeFootprint::union`](prism_render_architecture::particle::shading::ShadingAttributeFootprint::union)
//!   of two operand footprints;
//! - the lighting services a model subscribes to under a
//!   [`LightingServiceCaps`](prism_render_architecture::particle::shading::LightingServiceCaps)
//!   mask, plus the model's `needs_lighting` flag;
//! - the `OIT` route for a
//!   [`SortDecision`](prism_render_architecture::particle::sort_cull::SortDecision);
//! - the motion-vector request for a
//!   [`MotionVectorInput`](prism_render_architecture::particle::shading::MotionVectorInput);
//! - the six-way directional response for a
//!   [`SixWayLuminance`](prism_render_architecture::particle::shading::SixWayLuminance)
//!   and a light direction;
//! - the neutral isotropic
//!   [`PhaseParams`](prism_render_architecture::particle::shading::PhaseParams);
//! - the cel-band quantization of a response;
//! - the deep-shadow tier for a quality/model/volumetric triple; and
//! - the composite
//!   [`ShadingProgram`](prism_render_architecture::particle::shading::ShadingProgram).
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — integer compares and
//! bit operations, `abs`/`min`/`max`, `floor`, `sqrt`, `+ - * /` and `select` —
//! with no `sin`, `cos`, `exp`, `log`, `pow`, `round`, `smoothstep` or optional
//! device feature. The one normalization reuses `sqrt`; the cel quantization
//! reuses `floor` exactly as the reference does. It therefore runs unmodified on
//! Metal, Vulkan and DX12.
//!
//! # Correctness model
//!
//! Every classification, bitfield and boolean lane is a deterministic branch on
//! integer inputs, so `CPU` and `GPU` agree *exactly* and the parity test
//! compares them with `==`. The two continuous lanes — the six-way response and
//! the cel quantization — are closed-form multiply/add/divide with no
//! transcendental call, so they are not bit-exact only insofar as a `GPU` may
//! fuse a multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. The parity test therefore
//! asserts a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on those two
//! and an exact `==` on every discrete lane.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::shading`；pure
//! classification/bit-union plus directional algebra and a `wgpu` compute
//! dispatch; no third-party engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::lod::ParticleQuality;
use prism_render_architecture::particle::shading::{
    attribute_footprint, lighting_services, motion_vector_request, quantize_cel_bands,
    render_phase_for, resolve_deep_shadow, resolve_oit_route, resolve_shading_program,
    six_way_response, DeepShadowMode, LightingServiceCaps, LightingServices, MotionVectorInput,
    MotionVectorRequest, OitRoute, ParticleRenderPhase, PhaseParams, ShadingAttributeFootprint,
    ShadingProgram, ShadingProgramInput, SixWayLuminance,
};
use prism_render_architecture::particle::sort_cull::{BlendMode, SortDecision};
use prism_render_architecture::particle::{EmberShadingModel, ShadingBasis, SortStrategy, Vec3};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Number of threads per workgroup. `64` is the portable, warp-friendly size
/// shared by every kernel in this crate.
const WORKGROUP_SIZE: u32 = 64;

/// `BlendMode::Opaque` classification code shared by host and kernel.
const BLEND_OPAQUE: u32 = 0;
/// `BlendMode::Additive` classification code.
const BLEND_ADDITIVE: u32 = 1;
/// `BlendMode::Premultiplied` classification code.
const BLEND_PREMULTIPLIED: u32 = 2;
/// `BlendMode::AlphaBlend` classification code.
const BLEND_ALPHA: u32 = 3;

/// `ShadingBasis::Unlit` basis code.
const BASIS_UNLIT: u32 = 0;
/// `ShadingBasis::Pbr` basis code.
const BASIS_PBR: u32 = 1;
/// `ShadingBasis::Npr` basis code.
const BASIS_NPR: u32 = 2;
/// `ShadingBasis::Custom` basis code.
const BASIS_CUSTOM: u32 = 3;

/// `EmberShadingModel::Unlit` model-kind code.
const MODEL_UNLIT: u32 = 0;
/// `EmberShadingModel::Pbr` model-kind code.
const MODEL_PBR: u32 = 1;
/// `EmberShadingModel::Npr` model-kind code.
const MODEL_NPR: u32 = 2;
/// `EmberShadingModel::Custom` model-kind code.
const MODEL_CUSTOM: u32 = 3;
/// `EmberShadingModel::Hybrid` model-kind code.
const MODEL_HYBRID: u32 = 4;

/// `ParticleRenderPhase::Opaque` phase code.
const PHASE_OPAQUE: u32 = 0;
/// `ParticleRenderPhase::AlphaMask` phase code (never produced by the router).
const PHASE_ALPHA_MASK: u32 = 1;
/// `ParticleRenderPhase::Transparent` phase code.
const PHASE_TRANSPARENT: u32 = 2;

/// `SortStrategy::None` code.
const SORT_NONE: u32 = 0;
/// `SortStrategy::SharedOit` code.
const SORT_SHARED_OIT: u32 = 1;
/// `SortStrategy::ViewDepthRadix` code.
const SORT_RADIX: u32 = 2;
/// `SortStrategy::ViewDepthBitonic` code.
const SORT_BITONIC: u32 = 3;

/// `OitRoute::OrderIndependent` route code.
const OIT_ORDER_INDEPENDENT: u32 = 0;
/// `OitRoute::SharedOit` route code.
const OIT_SHARED: u32 = 1;
/// `OitRoute::StandaloneSort` route code.
const OIT_STANDALONE: u32 = 2;

/// `DeepShadowMode::None` tier code.
const DEEP_NONE: u32 = 0;
/// `DeepShadowMode::SixWay` tier code.
const DEEP_SIX_WAY: u32 = 1;
/// `DeepShadowMode::DeepOpacity` tier code.
const DEEP_OPACITY: u32 = 2;

/// `ShadingAttributeFootprint::normal` bit.
const FP_NORMAL: u32 = 1;
/// `ShadingAttributeFootprint::tangent` bit.
const FP_TANGENT: u32 = 2;
/// `ShadingAttributeFootprint::material_params` bit.
const FP_MATERIAL: u32 = 4;
/// `ShadingAttributeFootprint::ramp_lut` bit.
const FP_RAMP: u32 = 8;
/// `ShadingAttributeFootprint::custom_params` bit.
const FP_CUSTOM: u32 = 16;

/// `LightingServiceCaps::shadow_maps` input bit.
const CAP_SHADOW: u32 = 1;
/// `LightingServiceCaps::global_illumination` input bit.
const CAP_GI: u32 = 2;
/// `LightingServiceCaps::ray_tracing` input bit.
const CAP_RT: u32 = 4;

/// `LightingServices::clustered_lights` output bit.
const SVC_CLUSTERED: u32 = 1;
/// `LightingServices::shadow_maps` output bit.
const SVC_SHADOW: u32 = 2;
/// `LightingServices::global_illumination` output bit.
const SVC_GI: u32 = 4;
/// `LightingServices::ray_traced` output bit.
const SVC_RT: u32 = 8;

/// The portable core-`WGSL` shading-router kernel, embedded inline so the twin
/// ships as a single source file. Mirrors the `CPU` golden
/// [`shading`](prism_render_architecture::particle::shading) branch for branch;
/// see the module documentation for the contracts.
const SHADING_WGSL: &str = r#"
// Particle shading-router twin: one thread per query reproduces the render
// phase, attribute footprint (and footprint union), lighting services, OIT
// route, motion-vector request, six-way directional response, isotropic phase
// params, cel-band quantization, deep-shadow tier and the composite shading
// program of the CPU golden `particle::shading`. It uses only integer compares
// and bit ops, abs/min/max, floor, sqrt, + - * / and select, takes no optional
// feature, and so runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: 孪生自本仓 particle::shading; no third-party engine source.

struct Params {
    // Number of valid queries in `queries`.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One query. 112-byte std430 stride matching the host `GpuQuery`; every field is
// a 4-byte scalar so the storage array needs no vec alignment arithmetic.
struct Query {
    model_kind: u32,
    model_base: u32,
    model_overlay: u32,
    blend: u32,
    caps: u32,
    volumetric: u32,
    quality: u32,
    footprint_a: u32,
    footprint_b: u32,
    oit_blend: u32,
    particle_count: u32,
    radix_min_count: u32,
    prefer_shared_oit: u32,
    mv_visible: u32,
    mv_blend: u32,
    flipbook_rate: f32,
    flipbook_unstable_rate: f32,
    lum_right: f32,
    lum_left: f32,
    lum_up: f32,
    lum_down: f32,
    lum_front: f32,
    lum_back: f32,
    light_dir_x: f32,
    light_dir_y: f32,
    light_dir_z: f32,
    cel_response: f32,
    cel_bands: u32,
}

// One result. 96-byte std430 stride matching the host `GpuResult`.
struct Result {
    phase_code: u32,
    footprint_model: u32,
    footprint_union: u32,
    lighting: u32,
    needs_lighting: u32,
    oit_route_code: u32,
    oit_sort_strategy: u32,
    mv_write: u32,
    mv_unstable: u32,
    mv_reactive_mask: f32,
    six_way_response: f32,
    phase_g: f32,
    phase_back_lobe_weight: f32,
    phase_back_g: f32,
    cel_quantized: f32,
    deep_shadow_code: u32,
    deep_shadow_layers: u32,
    program_phase: u32,
    program_footprint: u32,
    program_lighting: u32,
    program_deep_shadow_code: u32,
    program_deep_shadow_layers: u32,
    program_needs_lighting: u32,
    pad0: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Squared-length floor guarding the six-way normalization, mirroring the
// reference `EPS_LEN_SQ`. A direct f32 == is forbidden, so a near-zero direction
// is caught by comparing its squared length against this floor.
const EPS_LEN_SQ: f32 = 1.0e-12;

// Footprint of a single shading basis lobe (`basis_footprint`): Unlit reads
// nothing, PBR reads normal + tangent + material, NPR reads normal + ramp,
// Custom reads custom params.
fn basis_footprint(basis: u32) -> u32 {
    if (basis == 1u) {
        return 1u | 2u | 4u;
    }
    if (basis == 2u) {
        return 1u | 8u;
    }
    if (basis == 3u) {
        return 16u;
    }
    return 0u;
}

// Attribute footprint of a model (`attribute_footprint`): a hybrid is the bit
// union of its two lobes.
fn attribute_footprint(kind: u32, base: u32, overlay: u32) -> u32 {
    if (kind == 0u) {
        return 0u;
    }
    if (kind == 1u) {
        return basis_footprint(1u);
    }
    if (kind == 2u) {
        return basis_footprint(2u);
    }
    if (kind == 3u) {
        return basis_footprint(3u);
    }
    return basis_footprint(base) | basis_footprint(overlay);
}

// Whether a single basis lobe consumes lighting (`basis_needs_lighting`): every
// lobe except Unlit does.
fn basis_needs_lighting(basis: u32) -> bool {
    return basis != 0u;
}

// Whether a model consumes lighting (`EmberShadingModel::needs_lighting`).
fn model_needs_lighting(kind: u32, base: u32, overlay: u32) -> bool {
    if (kind == 0u) {
        return false;
    }
    if (kind == 4u) {
        return basis_needs_lighting(base) || basis_needs_lighting(overlay);
    }
    return true;
}

// Shared lighting services (`lighting_services`): an unlit model subscribes to
// nothing; any lit model always takes clustered lights plus each optional
// service the platform caps offer.
fn lighting_services(kind: u32, base: u32, overlay: u32, caps: u32) -> u32 {
    if (!model_needs_lighting(kind, base, overlay)) {
        return 0u;
    }
    var svc = 1u;
    if ((caps & 1u) != 0u) {
        svc = svc | 2u;
    }
    if ((caps & 2u) != 0u) {
        svc = svc | 4u;
    }
    if ((caps & 4u) != 0u) {
        svc = svc | 8u;
    }
    return svc;
}

// Render phase for a blend mode (`render_phase_for`): opaque lands in the opaque
// phase, every blended mode in the transparent phase.
fn render_phase_for(blend: u32) -> u32 {
    if (blend == 0u) {
        return 0u;
    }
    return 2u;
}

// Sort strategy for an OIT decision (`choose_sort_strategy`): order-independent
// or trivial counts need no sort; an order-dependent blend prefers the shared
// OIT path, else a radix sort for large counts and a bitonic sort for small.
fn choose_sort_strategy(blend: u32, count: u32, radix_min: u32, prefer: u32) -> u32 {
    let needs_sort = (blend == 3u);
    if (!needs_sort || count <= 1u) {
        return 0u;
    }
    if (prefer != 0u) {
        return 1u;
    }
    if (count >= radix_min) {
        return 2u;
    }
    return 3u;
}

// Clamp-and-quantize into cel bands (`quantize_cel_bands`): clamp to 0..=1 (so
// NaN collapses to the low band), pass through when `bands <= 1`, else snap to
// one of `bands` evenly spaced levels by flooring `clamped * bands`.
fn quantize_cel_bands(response: f32, bands: u32) -> f32 {
    var clamped = 0.0;
    if (response > 1.0) {
        clamped = 1.0;
    } else if (response > 0.0) {
        clamped = response;
    } else {
        clamped = 0.0;
    }
    if (bands <= 1u) {
        return clamped;
    }
    let steps = f32(bands);
    var idx = floor(clamped * steps);
    let top = steps - 1.0;
    if (idx > top) {
        idx = top;
    }
    return idx / top;
}

// Six-way directional response (`six_way_response`): normalize-or-zero the light
// direction, then sum each axis's baked luminance weighted by the positive
// projection of the direction onto that axis.
fn six_way_response(
    right: f32,
    left: f32,
    up: f32,
    down: f32,
    front: f32,
    back: f32,
    dir: vec3<f32>,
) -> f32 {
    let len_sq = dir.x * dir.x + dir.y * dir.y + dir.z * dir.z;
    var d = vec3<f32>(0.0, 0.0, 0.0);
    if (len_sq > EPS_LEN_SQ) {
        d = dir * (1.0 / sqrt(len_sq));
    }
    var x = -d.x * left;
    if (d.x >= 0.0) {
        x = d.x * right;
    }
    var y = -d.y * down;
    if (d.y >= 0.0) {
        y = d.y * up;
    }
    var z = -d.z * back;
    if (d.z >= 0.0) {
        z = d.z * front;
    }
    return x + y + z;
}

// Deep-shadow tier (`resolve_deep_shadow`): none for non-volumetric or unlit;
// otherwise the quality ladder (SixWay, then DeepOpacity with 4/8/16 layers).
// Returns (code, layers).
fn resolve_deep_shadow(
    quality: u32,
    kind: u32,
    base: u32,
    overlay: u32,
    volumetric: u32,
) -> vec2<u32> {
    if (volumetric == 0u || !model_needs_lighting(kind, base, overlay)) {
        return vec2<u32>(0u, 0u);
    }
    if (quality == 0u) {
        return vec2<u32>(1u, 0u);
    }
    if (quality == 1u) {
        return vec2<u32>(2u, 4u);
    }
    if (quality == 2u) {
        return vec2<u32>(2u, 8u);
    }
    return vec2<u32>(2u, 16u);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var r: Result;

    r.phase_code = render_phase_for(q.blend);
    r.footprint_model = attribute_footprint(q.model_kind, q.model_base, q.model_overlay);
    r.footprint_union = q.footprint_a | q.footprint_b;
    r.lighting = lighting_services(q.model_kind, q.model_base, q.model_overlay, q.caps);
    r.needs_lighting = select(0u, 1u, model_needs_lighting(q.model_kind, q.model_base, q.model_overlay));

    let strat = choose_sort_strategy(q.oit_blend, q.particle_count, q.radix_min_count, q.prefer_shared_oit);
    if (strat == 0u) {
        r.oit_route_code = 0u;
        r.oit_sort_strategy = 0u;
    } else if (strat == 1u) {
        r.oit_route_code = 1u;
        r.oit_sort_strategy = 1u;
    } else {
        r.oit_route_code = 2u;
        r.oit_sort_strategy = strat;
    }

    if (q.mv_visible == 0u) {
        r.mv_write = 0u;
        r.mv_unstable = 0u;
        r.mv_reactive_mask = 0.0;
    } else {
        let unstable = (q.flipbook_rate > 0.0) && (q.flipbook_rate >= q.flipbook_unstable_rate);
        var bias = 0.5;
        if (q.mv_blend == 0u) {
            bias = 0.0;
        }
        var mask = bias;
        if (unstable) {
            mask = 1.0;
        }
        r.mv_write = 1u;
        r.mv_unstable = select(0u, 1u, unstable);
        r.mv_reactive_mask = mask;
    }

    r.six_way_response = six_way_response(
        q.lum_right,
        q.lum_left,
        q.lum_up,
        q.lum_down,
        q.lum_front,
        q.lum_back,
        vec3<f32>(q.light_dir_x, q.light_dir_y, q.light_dir_z),
    );

    r.phase_g = 0.0;
    r.phase_back_lobe_weight = 0.0;
    r.phase_back_g = 0.0;

    r.cel_quantized = quantize_cel_bands(q.cel_response, q.cel_bands);

    let ds = resolve_deep_shadow(q.quality, q.model_kind, q.model_base, q.model_overlay, q.volumetric);
    r.deep_shadow_code = ds.x;
    r.deep_shadow_layers = ds.y;

    r.program_phase = r.phase_code;
    r.program_footprint = r.footprint_model;
    r.program_lighting = r.lighting;
    r.program_deep_shadow_code = ds.x;
    r.program_deep_shadow_layers = ds.y;
    r.program_needs_lighting = r.needs_lighting;

    r.pad0 = 0u;

    results[idx] = r;
}
"#;

/// One shading-router query bundling every input the twinned functions consume.
///
/// Mirrors a single evaluation of the reference
/// [`render_phase_for`](prism_render_architecture::particle::shading::render_phase_for),
/// [`attribute_footprint`](prism_render_architecture::particle::shading::attribute_footprint),
/// [`lighting_services`](prism_render_architecture::particle::shading::lighting_services),
/// [`resolve_oit_route`](prism_render_architecture::particle::shading::resolve_oit_route),
/// [`motion_vector_request`](prism_render_architecture::particle::shading::motion_vector_request),
/// [`six_way_response`](prism_render_architecture::particle::shading::six_way_response),
/// [`quantize_cel_bands`](prism_render_architecture::particle::shading::quantize_cel_bands),
/// [`resolve_deep_shadow`](prism_render_architecture::particle::shading::resolve_deep_shadow)
/// and
/// [`resolve_shading_program`](prism_render_architecture::particle::shading::resolve_shading_program).
/// Derives only [`PartialEq`] (no `Eq`/`Hash`) because it holds `f32` lighting
/// and quantization inputs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuShadingQuery {
    /// The shading model driving the footprint, lighting and shadow lanes.
    pub model: EmberShadingModel,
    /// The renderer's blend mode driving the render phase and program.
    pub blend: BlendMode,
    /// The platform's available shared lighting services.
    pub caps: LightingServiceCaps,
    /// Whether the renderer is volumetric (drives the deep-shadow lane).
    pub volumetric: bool,
    /// The resolved quality tier (drives the deep-shadow ladder).
    pub quality: ParticleQuality,
    /// First operand of the standalone footprint-union lane.
    pub footprint_a: ShadingAttributeFootprint,
    /// Second operand of the standalone footprint-union lane.
    pub footprint_b: ShadingAttributeFootprint,
    /// The sort decision driving the `OIT`-route lane.
    pub oit: SortDecision,
    /// The motion-vector decision inputs.
    pub motion: MotionVectorInput,
    /// The pre-integrated six-way luminance rig.
    pub lum: SixWayLuminance,
    /// The light direction fed to the six-way rig (need not be normalized).
    pub light_dir: Vec3,
    /// The lighting response fed to the cel-band quantization lane.
    pub cel_response: f32,
    /// The number of cel bands to quantize into.
    pub cel_bands: u32,
}

/// The resolved verdict for one query, the host-side mirror of the kernel's
/// `Result` lane.
///
/// Every discrete field is reproduced exactly; the two continuous fields
/// (`six_way_response` and `cel_quantized`, plus the `f32` members of
/// `phase_params` and `motion`) are reproduced within the parity tolerance.
/// Derives only [`PartialEq`] (no `Eq`/`Hash`) because it holds `f32` fields.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuShadingResult {
    /// The render phase for `blend`.
    pub phase: ParticleRenderPhase,
    /// The attribute footprint of `model`.
    pub footprint_model: ShadingAttributeFootprint,
    /// The union of `footprint_a` and `footprint_b`.
    pub footprint_union: ShadingAttributeFootprint,
    /// The shared lighting services `model` subscribes to under `caps`.
    pub lighting: LightingServices,
    /// Whether `model` consumes lighting at all.
    pub needs_lighting: bool,
    /// The `OIT` route for `oit`.
    pub oit_route: OitRoute,
    /// The motion-vector request for `motion`.
    pub motion: MotionVectorRequest,
    /// The six-way directional response for `lum` and `light_dir`.
    pub six_way_response: f32,
    /// The neutral isotropic phase parameters.
    pub phase_params: PhaseParams,
    /// The cel-band quantization of `cel_response`.
    pub cel_quantized: f32,
    /// The deep-shadow tier for `quality`, `model` and `volumetric`.
    pub deep_shadow: DeepShadowMode,
    /// The composite shading program.
    pub program: ShadingProgram,
}

/// Uniform parameters for one dispatch. `repr(C)` `std430` layout matching
/// `Params` in [`SHADING_WGSL`]: the query count and three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid queries.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// One query as uploaded. `112`-byte `std430` stride matching `Query` in the
/// shader; every field is a 4-byte scalar so no vec padding is needed.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// `EmberShadingModel` kind code.
    model_kind: u32,
    /// Hybrid base-lobe basis code (unused for non-hybrid models).
    model_base: u32,
    /// Hybrid overlay-lobe basis code (unused for non-hybrid models).
    model_overlay: u32,
    /// `BlendMode` code for the render phase and program.
    blend: u32,
    /// Lighting-service capability bits.
    caps: u32,
    /// Volumetric flag (`0`/`1`).
    volumetric: u32,
    /// `ParticleQuality` code.
    quality: u32,
    /// First footprint-union operand bits.
    footprint_a: u32,
    /// Second footprint-union operand bits.
    footprint_b: u32,
    /// `BlendMode` code for the `OIT` decision.
    oit_blend: u32,
    /// Live particle count for the `OIT` decision.
    particle_count: u32,
    /// Radix threshold for the `OIT` decision.
    radix_min_count: u32,
    /// Prefer-shared-`OIT` flag (`0`/`1`).
    prefer_shared_oit: u32,
    /// Motion-vector visibility flag (`0`/`1`).
    mv_visible: u32,
    /// `BlendMode` code for the motion-vector decision.
    mv_blend: u32,
    /// `Flipbook` advance rate.
    flipbook_rate: f32,
    /// `Flipbook` instability threshold rate.
    flipbook_unstable_rate: f32,
    /// Six-way luminance toward `+X`.
    lum_right: f32,
    /// Six-way luminance toward `-X`.
    lum_left: f32,
    /// Six-way luminance toward `+Y`.
    lum_up: f32,
    /// Six-way luminance toward `-Y`.
    lum_down: f32,
    /// Six-way luminance toward `+Z`.
    lum_front: f32,
    /// Six-way luminance toward `-Z`.
    lum_back: f32,
    /// Light direction `x`.
    light_dir_x: f32,
    /// Light direction `y`.
    light_dir_y: f32,
    /// Light direction `z`.
    light_dir_z: f32,
    /// Cel-band response input.
    cel_response: f32,
    /// Cel-band count.
    cel_bands: u32,
}

impl GpuQuery {
    /// Packs a [`GpuShadingQuery`] into the `std430` upload layout.
    fn from_query(query: &GpuShadingQuery) -> GpuQuery {
        let (model_kind, model_base, model_overlay) = model_codes(query.model);
        GpuQuery {
            model_kind,
            model_base,
            model_overlay,
            blend: blend_code(query.blend),
            caps: caps_bits(query.caps),
            volumetric: u32::from(query.volumetric),
            quality: quality_code(query.quality),
            footprint_a: footprint_bits(query.footprint_a),
            footprint_b: footprint_bits(query.footprint_b),
            oit_blend: blend_code(query.oit.blend),
            particle_count: query.oit.particle_count,
            radix_min_count: query.oit.radix_min_count,
            prefer_shared_oit: u32::from(query.oit.prefer_shared_oit),
            mv_visible: u32::from(query.motion.visible),
            mv_blend: blend_code(query.motion.blend),
            flipbook_rate: query.motion.flipbook_rate,
            flipbook_unstable_rate: query.motion.flipbook_unstable_rate,
            lum_right: query.lum.right,
            lum_left: query.lum.left,
            lum_up: query.lum.up,
            lum_down: query.lum.down,
            lum_front: query.lum.front,
            lum_back: query.lum.back,
            light_dir_x: query.light_dir.x,
            light_dir_y: query.light_dir.y,
            light_dir_z: query.light_dir.z,
            cel_response: query.cel_response,
            cel_bands: query.cel_bands,
        }
    }
}

/// One result as read back. `96`-byte `std430` stride matching `Result` in the
/// shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Render-phase code.
    phase_code: u32,
    /// Model attribute-footprint bits.
    footprint_model: u32,
    /// Operand footprint-union bits.
    footprint_union: u32,
    /// Lighting-services bits.
    lighting: u32,
    /// Needs-lighting flag (`0`/`1`).
    needs_lighting: u32,
    /// `OIT`-route code.
    oit_route_code: u32,
    /// Standalone sort-strategy code for the `OIT` route.
    oit_sort_strategy: u32,
    /// Write-motion-vectors flag (`0`/`1`).
    mv_write: u32,
    /// Temporally-unstable flag (`0`/`1`).
    mv_unstable: u32,
    /// Reactive-mask weight.
    mv_reactive_mask: f32,
    /// Six-way directional response.
    six_way_response: f32,
    /// Isotropic phase anisotropy `g`.
    phase_g: f32,
    /// Isotropic back-lobe weight.
    phase_back_lobe_weight: f32,
    /// Isotropic back-lobe anisotropy.
    phase_back_g: f32,
    /// Cel-band quantization result.
    cel_quantized: f32,
    /// Deep-shadow tier code.
    deep_shadow_code: u32,
    /// Deep-shadow opacity-layer count.
    deep_shadow_layers: u32,
    /// Program render-phase code.
    program_phase: u32,
    /// Program attribute-footprint bits.
    program_footprint: u32,
    /// Program lighting-services bits.
    program_lighting: u32,
    /// Program deep-shadow tier code.
    program_deep_shadow_code: u32,
    /// Program deep-shadow opacity-layer count.
    program_deep_shadow_layers: u32,
    /// Program needs-lighting flag (`0`/`1`).
    program_needs_lighting: u32,
    /// Padding word.
    pad0: u32,
}

/// Maps a [`BlendMode`](prism_render_architecture::particle::sort_cull::BlendMode)
/// to its classification code.
fn blend_code(blend: BlendMode) -> u32 {
    match blend {
        BlendMode::Opaque => BLEND_OPAQUE,
        BlendMode::Additive => BLEND_ADDITIVE,
        BlendMode::Premultiplied => BLEND_PREMULTIPLIED,
        BlendMode::AlphaBlend => BLEND_ALPHA,
    }
}

/// Maps a [`ShadingBasis`](prism_render_architecture::particle::ShadingBasis) to
/// its basis code.
fn basis_code(basis: ShadingBasis) -> u32 {
    match basis {
        ShadingBasis::Unlit => BASIS_UNLIT,
        ShadingBasis::Pbr => BASIS_PBR,
        ShadingBasis::Npr => BASIS_NPR,
        ShadingBasis::Custom(_) => BASIS_CUSTOM,
    }
}

/// Encodes an
/// [`EmberShadingModel`](prism_render_architecture::particle::EmberShadingModel)
/// as `(kind, base, overlay)` codes; `base`/`overlay` are meaningful only for a
/// hybrid.
fn model_codes(model: EmberShadingModel) -> (u32, u32, u32) {
    match model {
        EmberShadingModel::Unlit => (MODEL_UNLIT, 0, 0),
        EmberShadingModel::Pbr => (MODEL_PBR, 0, 0),
        EmberShadingModel::Npr => (MODEL_NPR, 0, 0),
        EmberShadingModel::Custom(_) => (MODEL_CUSTOM, 0, 0),
        EmberShadingModel::Hybrid { base, overlay, .. } => {
            (MODEL_HYBRID, basis_code(base), basis_code(overlay))
        }
    }
}

/// Maps a
/// [`ParticleQuality`](prism_render_architecture::particle::lod::ParticleQuality)
/// to its tier code.
fn quality_code(quality: ParticleQuality) -> u32 {
    match quality {
        ParticleQuality::Low => 0,
        ParticleQuality::Medium => 1,
        ParticleQuality::High => 2,
        ParticleQuality::Ultra => 3,
    }
}

/// Packs the platform
/// [`LightingServiceCaps`](prism_render_architecture::particle::shading::LightingServiceCaps)
/// into the capability input bits.
fn caps_bits(caps: LightingServiceCaps) -> u32 {
    let mut bits = 0;
    if caps.shadow_maps {
        bits |= CAP_SHADOW;
    }
    if caps.global_illumination {
        bits |= CAP_GI;
    }
    if caps.ray_tracing {
        bits |= CAP_RT;
    }
    bits
}

/// Packs a
/// [`ShadingAttributeFootprint`](prism_render_architecture::particle::shading::ShadingAttributeFootprint)
/// into its five-bit field.
fn footprint_bits(fp: ShadingAttributeFootprint) -> u32 {
    let mut bits = 0;
    if fp.normal {
        bits |= FP_NORMAL;
    }
    if fp.tangent {
        bits |= FP_TANGENT;
    }
    if fp.material_params {
        bits |= FP_MATERIAL;
    }
    if fp.ramp_lut {
        bits |= FP_RAMP;
    }
    if fp.custom_params {
        bits |= FP_CUSTOM;
    }
    bits
}

/// Rebuilds a
/// [`ShadingAttributeFootprint`](prism_render_architecture::particle::shading::ShadingAttributeFootprint)
/// from its five-bit field.
fn footprint_from_bits(bits: u32) -> ShadingAttributeFootprint {
    ShadingAttributeFootprint {
        normal: (bits & FP_NORMAL) != 0,
        tangent: (bits & FP_TANGENT) != 0,
        material_params: (bits & FP_MATERIAL) != 0,
        ramp_lut: (bits & FP_RAMP) != 0,
        custom_params: (bits & FP_CUSTOM) != 0,
    }
}

/// Rebuilds a
/// [`LightingServices`](prism_render_architecture::particle::shading::LightingServices)
/// from its bit field.
fn lighting_from_bits(bits: u32) -> LightingServices {
    LightingServices {
        clustered_lights: (bits & SVC_CLUSTERED) != 0,
        shadow_maps: (bits & SVC_SHADOW) != 0,
        global_illumination: (bits & SVC_GI) != 0,
        ray_traced: (bits & SVC_RT) != 0,
    }
}

/// Rebuilds a
/// [`ParticleRenderPhase`](prism_render_architecture::particle::shading::ParticleRenderPhase)
/// from its phase code.
///
/// # Panics
///
/// Panics on an unknown code, which can only arise from a kernel/host layout
/// mismatch and indicates a broken port.
fn phase_from_code(code: u32) -> ParticleRenderPhase {
    match code {
        PHASE_OPAQUE => ParticleRenderPhase::Opaque,
        PHASE_ALPHA_MASK => ParticleRenderPhase::AlphaMask,
        PHASE_TRANSPARENT => ParticleRenderPhase::Transparent,
        other => panic!("unknown render-phase code {other}"),
    }
}

/// Rebuilds a
/// [`SortStrategy`](prism_render_architecture::particle::SortStrategy) from its
/// code (only the standalone radix/bitonic codes are expected here).
///
/// # Panics
///
/// Panics on an unexpected code, which indicates a broken port.
fn sort_strategy_from_code(code: u32) -> SortStrategy {
    match code {
        SORT_NONE => SortStrategy::None,
        SORT_SHARED_OIT => SortStrategy::SharedOit,
        SORT_RADIX => SortStrategy::ViewDepthRadix,
        SORT_BITONIC => SortStrategy::ViewDepthBitonic,
        other => panic!("unknown sort-strategy code {other}"),
    }
}

/// Rebuilds an
/// [`OitRoute`](prism_render_architecture::particle::shading::OitRoute) from its
/// route code and standalone sort-strategy code.
///
/// # Panics
///
/// Panics on an unknown route code, which indicates a broken port.
fn oit_route_from_codes(route: u32, strategy: u32) -> OitRoute {
    match route {
        OIT_ORDER_INDEPENDENT => OitRoute::OrderIndependent,
        OIT_SHARED => OitRoute::SharedOit,
        OIT_STANDALONE => OitRoute::StandaloneSort(sort_strategy_from_code(strategy)),
        other => panic!("unknown OIT-route code {other}"),
    }
}

/// Rebuilds a
/// [`DeepShadowMode`](prism_render_architecture::particle::shading::DeepShadowMode)
/// from its tier code and layer count.
///
/// # Panics
///
/// Panics on an unknown tier code, which indicates a broken port.
fn deep_shadow_from(code: u32, layers: u32) -> DeepShadowMode {
    match code {
        DEEP_NONE => DeepShadowMode::None,
        DEEP_SIX_WAY => DeepShadowMode::SixWay,
        DEEP_OPACITY => DeepShadowMode::DeepOpacity { layers },
        other => panic!("unknown deep-shadow code {other}"),
    }
}

/// Maps one kernel `Result` lane back to the host [`GpuShadingResult`].
fn decode_result(raw: &GpuResult) -> GpuShadingResult {
    GpuShadingResult {
        phase: phase_from_code(raw.phase_code),
        footprint_model: footprint_from_bits(raw.footprint_model),
        footprint_union: footprint_from_bits(raw.footprint_union),
        lighting: lighting_from_bits(raw.lighting),
        needs_lighting: raw.needs_lighting == 1,
        oit_route: oit_route_from_codes(raw.oit_route_code, raw.oit_sort_strategy),
        motion: MotionVectorRequest {
            write_motion_vectors: raw.mv_write == 1,
            temporally_unstable: raw.mv_unstable == 1,
            reactive_mask: raw.mv_reactive_mask,
        },
        six_way_response: raw.six_way_response,
        phase_params: PhaseParams {
            g: raw.phase_g,
            back_lobe_weight: raw.phase_back_lobe_weight,
            back_g: raw.phase_back_g,
        },
        cel_quantized: raw.cel_quantized,
        deep_shadow: deep_shadow_from(raw.deep_shadow_code, raw.deep_shadow_layers),
        program: ShadingProgram {
            phase: phase_from_code(raw.program_phase),
            footprint: footprint_from_bits(raw.program_footprint),
            lighting: lighting_from_bits(raw.program_lighting),
            deep_shadow: deep_shadow_from(
                raw.program_deep_shadow_code,
                raw.program_deep_shadow_layers,
            ),
            needs_lighting: raw.program_needs_lighting == 1,
        },
    }
}

/// The `CPU` golden verdict for one query, dispatching to the reference entry
/// points so callers (and the parity test) can pin the twin lane for lane.
#[must_use]
pub fn cpu_reference(query: &GpuShadingQuery) -> GpuShadingResult {
    GpuShadingResult {
        phase: render_phase_for(query.blend),
        footprint_model: attribute_footprint(query.model),
        footprint_union: query.footprint_a.union(query.footprint_b),
        lighting: lighting_services(query.model, query.caps),
        needs_lighting: query.model.needs_lighting(),
        oit_route: resolve_oit_route(query.oit),
        motion: motion_vector_request(query.motion),
        six_way_response: six_way_response(query.lum, query.light_dir),
        phase_params: PhaseParams::isotropic(),
        cel_quantized: quantize_cel_bands(query.cel_response, query.cel_bands),
        deep_shadow: resolve_deep_shadow(query.quality, query.model, query.volumetric),
        program: resolve_shading_program(ShadingProgramInput {
            model: query.model,
            blend: query.blend,
            caps: query.caps,
            volumetric: query.volumetric,
            quality: query.quality,
        }),
    }
}

/// A compiled, reusable shading-router pipeline.
pub struct GpuShading {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuShading {
    /// Compiles the shading-router kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuShading {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_shading"),
            source: ShaderSource::Wgsl(SHADING_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_shading_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_shading_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_shading_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuShading {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries`, returning one [`GpuShadingResult`] per
    /// query in input order.
    ///
    /// The returned result for query `q` mirrors the reference
    /// [`cpu_reference`] evaluated on `q`. An empty `queries` slice yields an
    /// empty result — storage buffers cannot be zero-sized, so it is handled by
    /// an early return before any dispatch.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[GpuShadingQuery]) -> Vec<GpuShadingResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_params = GpuParams {
            count: queries.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let gpu_queries: Vec<GpuQuery> = queries.iter().map(GpuQuery::from_query).collect();

        let out_bytes = (queries.len() * size_of::<GpuResult>()) as u64;

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_shading_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_shading_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_shading_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_shading_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_shading_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: queries_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: results_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_shading_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_shading_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, flattened to a 1-D dispatch.
            let groups = (queries.len() as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&results_buf, 0, &results_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        results_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = results_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let raw = bytemuck::cast_slice::<u8, GpuResult>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(raw.len(), queries.len());

        raw.iter().map(decode_result).collect()
    }
}

/// Builds a compute-visible buffer binding layout entry.
fn buffer_entry(binding: u32, ty: BufferBindingType) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
