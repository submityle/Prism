//! The four-equal-citizens shading router, motion-vector requirements, `OIT`
//! routing, and the volumetric six-way lighting / deep-shadow contracts.
//!
//! Shading is an emitter- (or per-particle-) level axis with four first-class
//! citizens — `Unlit`, `PBR`, `NPR`, and a user `Custom` closure — plus a
//! per-particle `Hybrid` blend (design §16). Whichever citizen an emitter
//! picks, it consumes the *same* shared base services (clustered + ray-traced
//! lighting, virtual shadow maps, global illumination); the shading axis only
//! decides *how* the surface responds to that lighting, so `NPR` is not a
//! second-class effect (design §16-§19). This module turns a shading model plus
//! its renderer's blend mode and the platform's lighting capabilities into a
//! compile-time specialization descriptor:
//!
//! - **Attribute footprint** — which per-particle attributes the chosen closure
//!   actually reads, so an `Unlit` kernel never allocates normals or material
//!   parameters (design §5.1, §16).
//! - **Lighting services** — the shared base services the closure subscribes to;
//!   identical for `PBR` and `NPR`, empty for `Unlit` (design §16).
//! - **Render phase & `OIT` routing** — the transparent/opaque phase a renderer
//!   lands in and whether an order-dependent blend routes through the scene's
//!   shared order-independent transparency path or a standalone depth sort
//!   (design §12, §15). This reuses the [`super::sort_cull`] strategy matrix.
//! - **Motion vectors** — every visible renderer writes motion vectors for
//!   `TAA` / temporal upsampling, and fast `flipbook` animation is flagged
//!   temporally unstable so the upsampler lowers its history weight (design
//!   §21).
//! - **Volumetric lighting** — the six-way pre-integrated lighting rig and the
//!   deep-opacity self-shadow tier that the `PBR`/`NPR` volumetric closures call
//!   (design §17, §18, §20).
//!
//! Every function here is a pure, deterministic classification or a small piece
//! of directional arithmetic; no transcendental math is used, so the `CPU`
//! contract matches the eventual `GPU` draw kernels bit for bit. The kernels and
//! `WESL` closure codegen themselves are pending the `GPU` backend; the phase
//! functions that need transcendental evaluation (Henyey-Greenstein) are carried
//! as parameter contracts and evaluated in-shader.

use super::lod::ParticleQuality;
use super::sort_cull::{choose_sort_strategy, BlendMode, SortDecision};
use super::{EmberShadingModel, ShadingBasis, SortStrategy, Vec3};

/// The render phase a particle renderer submits into (design §15).
///
/// Mirrors the scene's `Opaque3d` / `AlphaMask3d` / `Transparent3d` phases so
/// particles interleave with the rest of the frame instead of forming a
/// separate compositing pass.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ParticleRenderPhase {
    /// Depth-tested opaque draw, no blending.
    Opaque,
    /// Depth-tested alpha-masked (cutout) draw; authored via an opaque pipeline
    /// with an alpha test rather than a blend mode.
    AlphaMask,
    /// Blended draw in the transparent phase (additive, premultiplied, or
    /// straight alpha).
    Transparent,
}

/// Maps a renderer's blend mode to the render phase it submits into.
///
/// Opaque draws land in the opaque phase; every blended mode (additive,
/// premultiplied, straight alpha) lands in the transparent phase. Alpha-mask is
/// authored as an opaque-pipeline flag rather than a blend mode, so it is not
/// produced here.
#[must_use]
pub fn render_phase_for(blend: BlendMode) -> ParticleRenderPhase {
    match blend {
        BlendMode::Opaque => ParticleRenderPhase::Opaque,
        BlendMode::Additive | BlendMode::Premultiplied | BlendMode::AlphaBlend => {
            ParticleRenderPhase::Transparent
        }
    }
}

/// The per-particle attributes a shading closure reads.
///
/// Compile-time specialization allocates only the attributes a closure needs, so
/// an `Unlit` kernel carries no normal or material parameters (design §16). The
/// footprint of a [`EmberShadingModel::Hybrid`] is the union of its two lobes.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct ShadingAttributeFootprint {
    /// A shading normal (billboard/normal-map for sprites, real normal for
    /// mesh particles) is read.
    pub normal: bool,
    /// A tangent frame is read (anisotropy / normal mapping).
    pub tangent: bool,
    /// Physical material parameters (roughness / metallic / etc.) are read.
    pub material_params: bool,
    /// A 1D ramp / gradient `LUT` is sampled (stylized `NPR` shading).
    pub ramp_lut: bool,
    /// User-authored custom closure parameters are read.
    pub custom_params: bool,
}

impl ShadingAttributeFootprint {
    /// The union of two footprints (used for hybrid blends).
    #[must_use]
    pub fn union(self, other: Self) -> Self {
        ShadingAttributeFootprint {
            normal: self.normal || other.normal,
            tangent: self.tangent || other.tangent,
            material_params: self.material_params || other.material_params,
            ramp_lut: self.ramp_lut || other.ramp_lut,
            custom_params: self.custom_params || other.custom_params,
        }
    }
}

/// The attribute footprint of a single shading basis lobe.
#[must_use]
fn basis_footprint(basis: ShadingBasis) -> ShadingAttributeFootprint {
    match basis {
        ShadingBasis::Unlit => ShadingAttributeFootprint::default(),
        ShadingBasis::Pbr => ShadingAttributeFootprint {
            normal: true,
            tangent: true,
            material_params: true,
            ..ShadingAttributeFootprint::default()
        },
        ShadingBasis::Npr => ShadingAttributeFootprint {
            normal: true,
            ramp_lut: true,
            ..ShadingAttributeFootprint::default()
        },
        ShadingBasis::Custom(_) => ShadingAttributeFootprint {
            custom_params: true,
            ..ShadingAttributeFootprint::default()
        },
    }
}

/// The per-particle attribute footprint required by a shading model.
#[must_use]
pub fn attribute_footprint(model: EmberShadingModel) -> ShadingAttributeFootprint {
    match model {
        EmberShadingModel::Unlit => basis_footprint(ShadingBasis::Unlit),
        EmberShadingModel::Pbr => basis_footprint(ShadingBasis::Pbr),
        EmberShadingModel::Npr => basis_footprint(ShadingBasis::Npr),
        EmberShadingModel::Custom(id) => basis_footprint(ShadingBasis::Custom(id)),
        EmberShadingModel::Hybrid { base, overlay, .. } => {
            basis_footprint(base).union(basis_footprint(overlay))
        }
    }
}

/// The shared lighting base services a platform makes available (design §16).
///
/// These gate the optional services in [`LightingServices`]; clustered lighting
/// is always available for a lit model, but shadows / `GI` / ray tracing are
/// only subscribed to when both the platform offers them and the model is lit.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct LightingServiceCaps {
    /// Virtual shadow maps are available to receive.
    pub shadow_maps: bool,
    /// Screen-space / probe global illumination is available.
    pub global_illumination: bool,
    /// Ray-traced lighting (`bevy_solari`-style) is available.
    pub ray_tracing: bool,
}

/// The shared base services a shading model subscribes to this frame.
///
/// Determined by whether the model needs lighting at all (design §16): `Unlit`
/// subscribes to nothing, while `PBR` and `NPR` subscribe *identically* — the
/// shading axis never gates which shared services a lit model can consume.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct LightingServices {
    /// Clustered (forward+) light evaluation.
    pub clustered_lights: bool,
    /// Virtual shadow-map reception.
    pub shadow_maps: bool,
    /// Global illumination (probe / screen-space).
    pub global_illumination: bool,
    /// Ray-traced lighting contribution.
    pub ray_traced: bool,
}

/// Resolves which shared lighting services a shading model consumes.
///
/// An unlit model consumes none. Any lit model (`PBR`, `NPR`, a lit `Custom`, or
/// a hybrid with a lit lobe) always consumes clustered lights and additionally
/// each optional service the platform offers — the same set regardless of the
/// `PBR`-vs-`NPR` choice (design §16, §18).
#[must_use]
pub fn lighting_services(model: EmberShadingModel, caps: LightingServiceCaps) -> LightingServices {
    if !model.needs_lighting() {
        return LightingServices::default();
    }
    LightingServices {
        clustered_lights: true,
        shadow_maps: caps.shadow_maps,
        global_illumination: caps.global_illumination,
        ray_traced: caps.ray_tracing,
    }
}

/// How an emitter's transparent particles reach the frame buffer (design §12).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum OitRoute {
    /// Order-independent (opaque / additive / premultiplied): composite
    /// directly, no sort and no `OIT`.
    OrderIndependent,
    /// Order-dependent blend routed through the scene's shared order-independent
    /// transparency path (unified compositing with meshes, hair, cloth).
    SharedOit,
    /// Order-dependent blend with no shared `OIT` available: fall back to a
    /// standalone per-emitter view-depth sort.
    StandaloneSort(SortStrategy),
}

/// Resolves the transparency-compositing route for a renderer (design §12).
///
/// Reuses the [`super::sort_cull`] strategy matrix: order-independent blends
/// need neither a sort nor `OIT`; an order-dependent blend prefers the shared
/// `OIT` path and otherwise falls back to a standalone radix/bitonic sort sized
/// by particle count.
#[must_use]
pub fn resolve_oit_route(decision: SortDecision) -> OitRoute {
    match choose_sort_strategy(decision) {
        SortStrategy::None => OitRoute::OrderIndependent,
        SortStrategy::SharedOit => OitRoute::SharedOit,
        explicit => OitRoute::StandaloneSort(explicit),
    }
}

/// Inputs to the motion-vector requirement decision (design §21).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MotionVectorInput {
    /// Whether the renderer survived culling and draws this frame.
    pub visible: bool,
    /// The renderer's blend mode; blended draws feed a reactive mask so the
    /// temporal upsampler weights their history less.
    pub blend: BlendMode,
    /// `Flipbook` / `UV`-animation advance rate (frames per second). Fast
    /// animation is temporally unstable and rejects history.
    pub flipbook_rate: f32,
    /// Rate at or above which the `flipbook` is flagged temporally unstable.
    pub flipbook_unstable_rate: f32,
}

/// The motion-vector and temporal-upsampling requirements for a renderer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MotionVectorRequest {
    /// Whether the draw kernel writes per-particle motion vectors.
    pub write_motion_vectors: bool,
    /// Whether fast `flipbook` animation marks the renderer temporally
    /// unstable (upsampler should drop history).
    pub temporally_unstable: bool,
    /// Reactive-mask weight in `0..=1` fed to the temporal upsampler; higher
    /// means less history reuse (reduces ghosting on high-frequency particles).
    pub reactive_mask: f32,
}

/// Resolves the motion-vector requirements for a renderer (design §21).
///
/// Every *visible* renderer writes motion vectors so `TAA` / temporal
/// upsampling can reproject it. Culled renderers write nothing. Blended
/// (transparent) renderers get a partial reactive mask, and fast `flipbook`
/// animation saturates the mask and flags the renderer temporally unstable.
#[must_use]
pub fn motion_vector_request(input: MotionVectorInput) -> MotionVectorRequest {
    if !input.visible {
        return MotionVectorRequest {
            write_motion_vectors: false,
            temporally_unstable: false,
            reactive_mask: 0.0,
        };
    }
    let temporally_unstable =
        input.flipbook_rate > 0.0 && input.flipbook_rate >= input.flipbook_unstable_rate;
    let transparent_bias = match input.blend {
        BlendMode::Opaque => 0.0,
        BlendMode::Additive | BlendMode::Premultiplied | BlendMode::AlphaBlend => 0.5,
    };
    let reactive_mask = if temporally_unstable {
        1.0
    } else {
        transparent_bias
    };
    MotionVectorRequest {
        write_motion_vectors: true,
        temporally_unstable,
        reactive_mask,
    }
}

/// Pre-integrated six-axis in/out luminance for the six-way lighting rig.
///
/// The six-way rig bakes incoming/outgoing luminance along the six principal
/// axes and interpolates them by the light direction at runtime, a cheap volume
/// scattering approximation used by production `VFX` stacks (design §20). Each
/// field is the luminance seen when the light points along that axis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SixWayLuminance {
    /// Luminance toward `+X`.
    pub right: f32,
    /// Luminance toward `-X`.
    pub left: f32,
    /// Luminance toward `+Y`.
    pub up: f32,
    /// Luminance toward `-Y`.
    pub down: f32,
    /// Luminance toward `+Z`.
    pub front: f32,
    /// Luminance toward `-Z`.
    pub back: f32,
}

/// Interpolates the six-way rig for a light direction (design §20).
///
/// `light_dir` points from the particle toward the light and need not be
/// normalized (it is normalized internally; a zero direction yields zero). Each
/// axis contributes its baked luminance weighted by the positive projection of
/// the light direction onto that axis, so the result is a smooth directional
/// blend with no transcendental math.
#[must_use]
pub fn six_way_response(lum: SixWayLuminance, light_dir: Vec3) -> f32 {
    let d = light_dir.normalize_or_zero();
    let x = if d.x >= 0.0 {
        d.x * lum.right
    } else {
        -d.x * lum.left
    };
    let y = if d.y >= 0.0 {
        d.y * lum.up
    } else {
        -d.y * lum.down
    };
    let z = if d.z >= 0.0 {
        d.z * lum.front
    } else {
        -d.z * lum.back
    };
    x + y + z
}

/// Phase-function parameters for volumetric scattering (design §17, §20).
///
/// Carries the Henyey-Greenstein anisotropy and an optional back lobe for the
/// double-lobe (front + back scatter) response that gives real smoke its edge
/// glow. The phase function itself needs transcendental evaluation and runs
/// in-shader (pending the `GPU` backend); this struct is the authored contract.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhaseParams {
    /// Primary Henyey-Greenstein anisotropy `g` in `-1..=1` (forward `> 0`).
    pub g: f32,
    /// Weight of the back-scatter lobe in `0..=1` (`0` = single lobe).
    pub back_lobe_weight: f32,
    /// Back-scatter lobe anisotropy (typically negative).
    pub back_g: f32,
}

impl PhaseParams {
    /// An isotropic single-lobe phase (`g = 0`, no back lobe).
    #[must_use]
    pub fn isotropic() -> Self {
        PhaseParams {
            g: 0.0,
            back_lobe_weight: 0.0,
            back_g: 0.0,
        }
    }
}

/// Quantizes a lighting response into discrete cel bands (design §18, §20).
///
/// The stylized (`NPR`) volumetric path replaces smooth scattering with a small
/// number of hard shadow bands. `response` is clamped to `0..=1`; with `bands`
/// steps the result snaps to one of `bands` evenly spaced levels spanning
/// `0..=1`. `bands <= 1` returns the clamped value unchanged (no banding).
#[must_use]
pub fn quantize_cel_bands(response: f32, bands: u32) -> f32 {
    // Clamp explicitly so `NaN` collapses to `0.0` (the low band) rather than
    // propagating through the banding arithmetic.
    let clamped = if response > 1.0 {
        1.0
    } else if response > 0.0 {
        response
    } else {
        0.0
    };
    if bands <= 1 {
        return clamped;
    }
    let steps = bands as f32;
    let idx = (clamped * steps).floor();
    let top = steps - 1.0;
    let idx = if idx > top { top } else { idx };
    idx / top
}

/// The self-shadow / volumetric shadowing tier for a volumetric renderer
/// (design §20).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DeepShadowMode {
    /// No self-shadow (unlit energy effects, or non-volumetric renderers).
    None,
    /// Six-way pre-integrated rig only: cheap directional shading, no
    /// per-sample occlusion.
    SixWay,
    /// A deep-opacity map with `layers` transmittance layers for internal
    /// self-shadowing.
    DeepOpacity {
        /// Number of transmittance layers recorded along the light ray.
        layers: u32,
    },
}

/// Resolves the volumetric self-shadow tier for a renderer (design §20).
///
/// Non-volumetric renderers and unlit models cast no self-shadow. A lit
/// volumetric renderer climbs the quality ladder: the lowest quality uses only
/// the six-way rig, while higher qualities record progressively deeper opacity
/// maps for accurate internal self-shadowing.
#[must_use]
pub fn resolve_deep_shadow(
    quality: ParticleQuality,
    model: EmberShadingModel,
    volumetric: bool,
) -> DeepShadowMode {
    if !volumetric || !model.needs_lighting() {
        return DeepShadowMode::None;
    }
    match quality {
        ParticleQuality::Low => DeepShadowMode::SixWay,
        ParticleQuality::Medium => DeepShadowMode::DeepOpacity { layers: 4 },
        ParticleQuality::High => DeepShadowMode::DeepOpacity { layers: 8 },
        ParticleQuality::Ultra => DeepShadowMode::DeepOpacity { layers: 16 },
    }
}

/// The full set of inputs that specialize a particle draw kernel (design §16).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShadingProgramInput {
    /// The emitter's shading model.
    pub model: EmberShadingModel,
    /// The renderer's blend mode.
    pub blend: BlendMode,
    /// The platform's available shared lighting services.
    pub caps: LightingServiceCaps,
    /// Whether the renderer is a volumetric (smoke/fluid) renderer.
    pub volumetric: bool,
    /// The resolved quality tier (drives the deep-shadow ladder).
    pub quality: ParticleQuality,
}

/// The compile-time specialization descriptor for a `Renderer × ShadingModel`
/// draw kernel (design §16).
///
/// Bundles everything the kernel needs to be generated without reusing any
/// attribute, lighting service, or shadow structure the model does not touch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShadingProgram {
    /// The render phase the kernel submits into.
    pub phase: ParticleRenderPhase,
    /// The per-particle attributes the kernel reads.
    pub footprint: ShadingAttributeFootprint,
    /// The shared lighting services the kernel subscribes to.
    pub lighting: LightingServices,
    /// The volumetric self-shadow tier the kernel uses.
    pub deep_shadow: DeepShadowMode,
    /// Whether the model consumes lighting at all (cached from the model).
    pub needs_lighting: bool,
}

/// Resolves the full draw-kernel specialization for a renderer (design §16).
///
/// Composes the render phase, attribute footprint, lighting services, and
/// deep-shadow tier into one descriptor so the kernel generator (pending the
/// `GPU` backend) allocates exactly what the chosen shading citizen needs.
#[must_use]
pub fn resolve_shading_program(input: ShadingProgramInput) -> ShadingProgram {
    ShadingProgram {
        phase: render_phase_for(input.blend),
        footprint: attribute_footprint(input.model),
        lighting: lighting_services(input.model, input.caps),
        deep_shadow: resolve_deep_shadow(input.quality, input.model, input.volumetric),
        needs_lighting: input.model.needs_lighting(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_caps() -> LightingServiceCaps {
        LightingServiceCaps {
            shadow_maps: true,
            global_illumination: true,
            ray_tracing: true,
        }
    }

    #[test]
    fn render_phase_maps_blends() {
        assert_eq!(
            render_phase_for(BlendMode::Opaque),
            ParticleRenderPhase::Opaque
        );
        assert_eq!(
            render_phase_for(BlendMode::Additive),
            ParticleRenderPhase::Transparent
        );
        assert_eq!(
            render_phase_for(BlendMode::Premultiplied),
            ParticleRenderPhase::Transparent
        );
        assert_eq!(
            render_phase_for(BlendMode::AlphaBlend),
            ParticleRenderPhase::Transparent
        );
    }

    #[test]
    fn unlit_footprint_is_empty() {
        let fp = attribute_footprint(EmberShadingModel::Unlit);
        assert_eq!(fp, ShadingAttributeFootprint::default());
        assert!(!fp.normal);
        assert!(!fp.material_params);
    }

    #[test]
    fn pbr_reads_normal_tangent_and_material() {
        let fp = attribute_footprint(EmberShadingModel::Pbr);
        assert!(fp.normal);
        assert!(fp.tangent);
        assert!(fp.material_params);
        assert!(!fp.ramp_lut);
        assert!(!fp.custom_params);
    }

    #[test]
    fn npr_reads_normal_and_ramp() {
        let fp = attribute_footprint(EmberShadingModel::Npr);
        assert!(fp.normal);
        assert!(fp.ramp_lut);
        assert!(!fp.tangent);
        assert!(!fp.material_params);
    }

    #[test]
    fn custom_reads_custom_params() {
        let fp = attribute_footprint(EmberShadingModel::Custom(7));
        assert!(fp.custom_params);
        assert!(!fp.normal);
    }

    #[test]
    fn hybrid_footprint_is_union_of_lobes() {
        let fp = attribute_footprint(EmberShadingModel::Hybrid {
            base: ShadingBasis::Pbr,
            overlay: ShadingBasis::Npr,
            weight: 0.5,
        });
        // PBR contributes tangent + material_params; NPR contributes ramp_lut;
        // both contribute normal.
        assert!(fp.normal);
        assert!(fp.tangent);
        assert!(fp.material_params);
        assert!(fp.ramp_lut);
        assert!(!fp.custom_params);
    }

    #[test]
    fn unlit_subscribes_to_no_lighting_services() {
        let svc = lighting_services(EmberShadingModel::Unlit, all_caps());
        assert_eq!(svc, LightingServices::default());
    }

    #[test]
    fn pbr_and_npr_subscribe_to_the_same_services() {
        let pbr = lighting_services(EmberShadingModel::Pbr, all_caps());
        let npr = lighting_services(EmberShadingModel::Npr, all_caps());
        assert_eq!(pbr, npr);
        assert!(pbr.clustered_lights);
        assert!(pbr.shadow_maps);
        assert!(pbr.global_illumination);
        assert!(pbr.ray_traced);
    }

    #[test]
    fn lighting_services_gate_on_platform_caps() {
        let caps = LightingServiceCaps {
            shadow_maps: true,
            global_illumination: false,
            ray_tracing: false,
        };
        let svc = lighting_services(EmberShadingModel::Pbr, caps);
        assert!(svc.clustered_lights);
        assert!(svc.shadow_maps);
        assert!(!svc.global_illumination);
        assert!(!svc.ray_traced);
    }

    #[test]
    fn pure_unlit_hybrid_needs_no_lighting() {
        let model = EmberShadingModel::Hybrid {
            base: ShadingBasis::Unlit,
            overlay: ShadingBasis::Unlit,
            weight: 0.5,
        };
        assert_eq!(
            lighting_services(model, all_caps()),
            LightingServices::default()
        );
    }

    fn oit_decision(blend: BlendMode, count: u32, prefer_oit: bool) -> SortDecision {
        SortDecision {
            blend,
            particle_count: count,
            radix_min_count: 4096,
            prefer_shared_oit: prefer_oit,
        }
    }

    #[test]
    fn order_independent_blends_skip_oit() {
        assert_eq!(
            resolve_oit_route(oit_decision(BlendMode::Additive, 10_000, true)),
            OitRoute::OrderIndependent
        );
        assert_eq!(
            resolve_oit_route(oit_decision(BlendMode::Premultiplied, 10_000, true)),
            OitRoute::OrderIndependent
        );
        assert_eq!(
            resolve_oit_route(oit_decision(BlendMode::Opaque, 10_000, true)),
            OitRoute::OrderIndependent
        );
    }

    #[test]
    fn alpha_blend_prefers_shared_oit() {
        assert_eq!(
            resolve_oit_route(oit_decision(BlendMode::AlphaBlend, 10_000, true)),
            OitRoute::SharedOit
        );
    }

    #[test]
    fn alpha_blend_without_oit_falls_back_to_sort() {
        assert_eq!(
            resolve_oit_route(oit_decision(BlendMode::AlphaBlend, 10_000, false)),
            OitRoute::StandaloneSort(SortStrategy::ViewDepthRadix)
        );
        assert_eq!(
            resolve_oit_route(oit_decision(BlendMode::AlphaBlend, 100, false)),
            OitRoute::StandaloneSort(SortStrategy::ViewDepthBitonic)
        );
    }

    #[test]
    fn culled_renderer_writes_no_motion_vectors() {
        let req = motion_vector_request(MotionVectorInput {
            visible: false,
            blend: BlendMode::AlphaBlend,
            flipbook_rate: 60.0,
            flipbook_unstable_rate: 30.0,
        });
        assert!(!req.write_motion_vectors);
        assert!(!req.temporally_unstable);
        assert_eq!(req.reactive_mask, 0.0);
    }

    #[test]
    fn visible_opaque_writes_motion_vectors_without_reactive_mask() {
        let req = motion_vector_request(MotionVectorInput {
            visible: true,
            blend: BlendMode::Opaque,
            flipbook_rate: 0.0,
            flipbook_unstable_rate: 30.0,
        });
        assert!(req.write_motion_vectors);
        assert!(!req.temporally_unstable);
        assert_eq!(req.reactive_mask, 0.0);
    }

    #[test]
    fn transparent_gets_partial_reactive_mask() {
        let req = motion_vector_request(MotionVectorInput {
            visible: true,
            blend: BlendMode::AlphaBlend,
            flipbook_rate: 0.0,
            flipbook_unstable_rate: 30.0,
        });
        assert!(req.write_motion_vectors);
        assert!(!req.temporally_unstable);
        assert_eq!(req.reactive_mask, 0.5);
    }

    #[test]
    fn fast_flipbook_saturates_reactive_mask() {
        let req = motion_vector_request(MotionVectorInput {
            visible: true,
            blend: BlendMode::Opaque,
            flipbook_rate: 45.0,
            flipbook_unstable_rate: 30.0,
        });
        assert!(req.temporally_unstable);
        assert_eq!(req.reactive_mask, 1.0);
    }

    #[test]
    fn six_way_picks_the_axis_facing_the_light() {
        let lum = SixWayLuminance {
            right: 1.0,
            left: 0.1,
            up: 0.2,
            down: 0.3,
            front: 0.4,
            back: 0.5,
        };
        // Light straight along +X selects the `right` luminance.
        assert!((six_way_response(lum, Vec3::new(2.0, 0.0, 0.0)) - 1.0).abs() < 1e-6);
        // Light straight along -X selects the `left` luminance.
        assert!((six_way_response(lum, Vec3::new(-1.0, 0.0, 0.0)) - 0.1).abs() < 1e-6);
        // A zero direction yields no response.
        assert_eq!(six_way_response(lum, Vec3::ZERO), 0.0);
    }

    #[test]
    fn six_way_blends_between_axes() {
        let lum = SixWayLuminance {
            right: 1.0,
            left: 0.0,
            up: 1.0,
            down: 0.0,
            front: 0.0,
            back: 0.0,
        };
        // 45 degrees between +X and +Y: each axis contributes 1/sqrt(2).
        let r = six_way_response(lum, Vec3::new(1.0, 1.0, 0.0));
        let expected = 2.0 * 0.5_f32.sqrt();
        assert!((r - expected).abs() < 1e-6, "got {r}");
    }

    #[test]
    fn cel_banding_snaps_to_levels() {
        // 3 bands snap to levels 0, 0.5, 1.0 by flooring `value * bands`.
        assert_eq!(quantize_cel_bands(0.0, 3), 0.0);
        assert_eq!(quantize_cel_bands(0.3, 3), 0.0);
        assert_eq!(quantize_cel_bands(0.4, 3), 0.5);
        assert_eq!(quantize_cel_bands(0.5, 3), 0.5);
        assert_eq!(quantize_cel_bands(0.9, 3), 1.0);
        assert_eq!(quantize_cel_bands(1.0, 3), 1.0);
    }

    #[test]
    fn cel_banding_clamps_and_passes_through() {
        assert_eq!(quantize_cel_bands(2.0, 3), 1.0);
        assert_eq!(quantize_cel_bands(-1.0, 3), 0.0);
        // NaN collapses to the low band.
        assert_eq!(quantize_cel_bands(f32::NAN, 3), 0.0);
        // One band (or zero) means no banding: clamped value returned.
        assert_eq!(quantize_cel_bands(0.37, 1), 0.37);
    }

    #[test]
    fn deep_shadow_off_for_non_volumetric_or_unlit() {
        assert_eq!(
            resolve_deep_shadow(ParticleQuality::Ultra, EmberShadingModel::Pbr, false),
            DeepShadowMode::None
        );
        assert_eq!(
            resolve_deep_shadow(ParticleQuality::Ultra, EmberShadingModel::Unlit, true),
            DeepShadowMode::None
        );
    }

    #[test]
    fn deep_shadow_climbs_quality_ladder() {
        assert_eq!(
            resolve_deep_shadow(ParticleQuality::Low, EmberShadingModel::Pbr, true),
            DeepShadowMode::SixWay
        );
        assert_eq!(
            resolve_deep_shadow(ParticleQuality::Medium, EmberShadingModel::Npr, true),
            DeepShadowMode::DeepOpacity { layers: 4 }
        );
        assert_eq!(
            resolve_deep_shadow(ParticleQuality::High, EmberShadingModel::Pbr, true),
            DeepShadowMode::DeepOpacity { layers: 8 }
        );
        assert_eq!(
            resolve_deep_shadow(ParticleQuality::Ultra, EmberShadingModel::Pbr, true),
            DeepShadowMode::DeepOpacity { layers: 16 }
        );
    }

    #[test]
    fn shading_program_composes_every_facet() {
        let program = resolve_shading_program(ShadingProgramInput {
            model: EmberShadingModel::Pbr,
            blend: BlendMode::AlphaBlend,
            caps: all_caps(),
            volumetric: true,
            quality: ParticleQuality::High,
        });
        assert_eq!(program.phase, ParticleRenderPhase::Transparent);
        assert!(program.footprint.material_params);
        assert!(program.lighting.clustered_lights);
        assert_eq!(
            program.deep_shadow,
            DeepShadowMode::DeepOpacity { layers: 8 }
        );
        assert!(program.needs_lighting);
    }

    #[test]
    fn shading_program_for_unlit_additive_is_minimal() {
        let program = resolve_shading_program(ShadingProgramInput {
            model: EmberShadingModel::Unlit,
            blend: BlendMode::Additive,
            caps: all_caps(),
            volumetric: true,
            quality: ParticleQuality::Ultra,
        });
        assert_eq!(program.phase, ParticleRenderPhase::Transparent);
        assert_eq!(program.footprint, ShadingAttributeFootprint::default());
        assert_eq!(program.lighting, LightingServices::default());
        assert_eq!(program.deep_shadow, DeepShadowMode::None);
        assert!(!program.needs_lighting);
    }

    #[test]
    fn phase_params_isotropic_is_neutral() {
        let p = PhaseParams::isotropic();
        assert_eq!(p.g, 0.0);
        assert_eq!(p.back_lobe_weight, 0.0);
        assert_eq!(p.back_g, 0.0);
    }
}
