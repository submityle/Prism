//! The `WESL` shading-closure code emitter: turns a resolved [`ShadingProgram`]
//! into the inline `WESL` closure fragments and bind-group declarations a draw
//! kernel needs (design §16).
//!
//! [`super::shading`] resolves a [`ShadingProgram`] *descriptor* — render phase,
//! attribute footprint, shared lighting services, and deep-shadow tier — but it
//! stops short of emitting any shader source; its own documentation flags the
//! `WESL` closure codegen as pending. This module closes that gap. It mirrors the
//! simulation-side emission paradigm in [`super::modules`], where every built-in
//! module implements `emit_wesl` against a shared `CodegenCtx` that accumulates
//! inline fragments and de-duplicates binding declarations. The shading path is
//! deliberately isomorphic: given the four equal-citizen shading models — `Unlit`,
//! `PBR`, `NPR`, and a user `Custom` closure — plus the per-particle `Hybrid`
//! blend, it emits the matching closure call(s) and declares exactly the bindings
//! the chosen citizen touches (design §16).
//!
//! The emission rules follow the specialization contract of the descriptor:
//!
//! - **Attribute bindings** are declared only when the program's
//!   [`ShadingAttributeFootprint`] actually reads them, so an `Unlit` kernel
//!   declares no normal, tangent, material, ramp, or custom-parameter binding
//!   (design §5.1, §16).
//! - **Lighting bindings** are declared only for the shared services the program
//!   subscribes to; `PBR` and `NPR` subscribe identically, `Unlit` subscribes to
//!   nothing (design §16, §18).
//! - **Deep-shadow binding** is declared only when the program carries a
//!   volumetric self-shadow tier (design §20).
//! - **`Hybrid`** emits both lobe closures and a `blend_shading` combine weighted
//!   by the per-particle blend weight; **`Custom`** emits a stable
//!   `shade_custom_<id>` call as the user-authored closure extension point
//!   (design §19).
//!
//! This layer is pure string assembly: it performs no arithmetic and uses no
//! transcendental math, so it never participates in the deterministic-`CPU`
//! contract. The Henyey-Greenstein (`HG`) phase and other transcendental shading
//! math remain parameter contracts evaluated in-shader, exactly as
//! [`super::shading`] documents; nothing here evaluates them on the `CPU`. The
//! only numeric work is formatting a blend weight into a `WESL` float literal.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use super::modules::{CodegenCtx, ResourceDecl};
use super::shading::{DeepShadowMode, LightingServices, ShadingAttributeFootprint, ShadingProgram};
use super::{EmberShadingModel, ShadingBasis};

/// The `WESL` bind-group index for the per-particle attribute inputs a shading
/// closure reads (normals, tangents, material parameters, ramps, custom params).
///
/// Kept distinct from the simulation bind groups in [`super::modules`] so a draw
/// kernel can bind shading inputs without disturbing the simulation layout
/// (design §16). The value `2` reserves groups `0`/`1` for the shared
/// per-view/per-system data the renderer already owns.
const SHADING_ATTR_GROUP: u32 = 2;

/// The `WESL` bind-group index for the shared lighting services every lit
/// closure subscribes to (clustered lights, shadow maps, global illumination,
/// ray-traced scene, deep-opacity self-shadow). The value `3` follows the
/// attribute group so the two never collide (design §16).
const LIGHTING_GROUP: u32 = 3;

// --- Attribute bindings (group `SHADING_ATTR_GROUP`) -----------------------
//
// Each slot is declared only when the resolved footprint reads it; the binding
// index is a stable per-slot reservation, not a packed/allocated value.

/// Per-particle shading normals (billboard/mapped for sprites, geometric for
/// mesh particles). Binding slot `0`.
const BIND_NORMAL: ResourceDecl = ResourceDecl::buffer("shading_normal_buf", SHADING_ATTR_GROUP, 0);
/// Per-particle tangent frames (anisotropy / normal mapping). Binding slot `1`.
const BIND_TANGENT: ResourceDecl =
    ResourceDecl::buffer("shading_tangent_buf", SHADING_ATTR_GROUP, 1);
/// Physical material parameters (roughness / metallic / etc.). Binding slot `2`.
const BIND_MATERIAL: ResourceDecl =
    ResourceDecl::uniform("shading_material_params", SHADING_ATTR_GROUP, 2);
/// A 1D ramp / gradient look-up table for stylized `NPR` shading. Slot `3`.
const BIND_RAMP: ResourceDecl = ResourceDecl::curve("shading_ramp_lut", SHADING_ATTR_GROUP, 3);
/// User-authored `Custom` closure parameters. Binding slot `4`.
const BIND_CUSTOM: ResourceDecl =
    ResourceDecl::uniform("shading_custom_params", SHADING_ATTR_GROUP, 4);

// --- Lighting bindings (group `LIGHTING_GROUP`) ----------------------------

/// Clustered (forward+) light list every lit closure iterates. Binding slot `0`.
const BIND_CLUSTERS: ResourceDecl = ResourceDecl::buffer("lighting_clusters", LIGHTING_GROUP, 0);
/// Virtual shadow-map atlas the closure samples for occlusion. Slot `1`.
const BIND_SHADOW_MAPS: ResourceDecl =
    ResourceDecl::texture2d("lighting_shadow_maps", LIGHTING_GROUP, 1);
/// Global-illumination probe volume (screen-space / probe `GI`). Slot `2`.
const BIND_GI_PROBES: ResourceDecl =
    ResourceDecl::texture3d("lighting_gi_probes", LIGHTING_GROUP, 2);
/// Ray-traced scene acceleration structure / lighting buffer. Slot `3`.
const BIND_RAY_SCENE: ResourceDecl = ResourceDecl::buffer("lighting_ray_scene", LIGHTING_GROUP, 3);
/// Deep-opacity / six-way self-shadow volume for volumetric closures. Slot `4`.
const BIND_DEEP_OPACITY: ResourceDecl =
    ResourceDecl::texture3d("shading_deep_opacity", LIGHTING_GROUP, 4);

/// The `WESL` identifier the final shaded result is bound to in every emitted
/// fragment. Downstream fragments (output write, motion vectors) read from this
/// name, so it is stable across all shading models.
const SHADED_RESULT: &str = "shaded_color";

/// The emitter that turns one resolved [`ShadingProgram`] plus its originating
/// [`EmberShadingModel`] into `WESL` closure fragments and binding declarations
/// (design §16).
///
/// The [`ShadingProgram`] descriptor caches the attribute footprint, lighting
/// services, and deep-shadow tier, but it does not retain the model's variant
/// data (the `Hybrid` lobes and weight, or the `Custom` handle), so this emitter
/// carries the [`EmberShadingModel`] alongside it. It is the shading-side analog
/// of a simulation `EmberModule`: `emit_wesl` declares the bindings the program
/// touches and pushes the closure call fragments, de-duplicating bindings through
/// the shared [`CodegenCtx`] exactly like the simulation path.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShadingClosure {
    /// The emitter's (or particle's) shading model, carrying `Hybrid` lobes /
    /// weight and the `Custom` handle that the descriptor does not retain.
    pub model: EmberShadingModel,
    /// The resolved specialization descriptor (footprint, lighting, shadow tier).
    pub program: ShadingProgram,
}

impl ShadingClosure {
    /// Pairs a shading model with its resolved program descriptor.
    #[must_use]
    pub fn new(model: EmberShadingModel, program: ShadingProgram) -> Self {
        ShadingClosure { model, program }
    }

    /// The ordered set of bind-group declarations this closure needs.
    ///
    /// Only the slots the program's footprint and lighting services actually
    /// read are included, so an unused attribute never declares a binding
    /// (design §16). The order is a stable attribute-then-lighting-then-shadow
    /// walk so the assembled kernel is deterministic.
    #[must_use]
    pub fn required_bindings(&self) -> Vec<ResourceDecl> {
        let mut out = Vec::new();
        let footprint: ShadingAttributeFootprint = self.program.footprint;
        if footprint.normal {
            out.push(BIND_NORMAL);
        }
        if footprint.tangent {
            out.push(BIND_TANGENT);
        }
        if footprint.material_params {
            out.push(BIND_MATERIAL);
        }
        if footprint.ramp_lut {
            out.push(BIND_RAMP);
        }
        if footprint.custom_params {
            out.push(BIND_CUSTOM);
        }
        let lighting: LightingServices = self.program.lighting;
        if lighting.clustered_lights {
            out.push(BIND_CLUSTERS);
        }
        if lighting.shadow_maps {
            out.push(BIND_SHADOW_MAPS);
        }
        if lighting.global_illumination {
            out.push(BIND_GI_PROBES);
        }
        if lighting.ray_traced {
            out.push(BIND_RAY_SCENE);
        }
        if !matches!(self.program.deep_shadow, DeepShadowMode::None) {
            out.push(BIND_DEEP_OPACITY);
        }
        out
    }

    /// Emits this closure's bindings and `WESL` fragments into `ctx`.
    ///
    /// Mirrors `EmberModule::emit_wesl` in [`super::modules`]: it first declares
    /// every required binding (de-duplicated by the context) and then pushes the
    /// closure call fragment(s) in emission order. The emitted source binds the
    /// final result to [`SHADED_RESULT`] regardless of the shading model.
    pub fn emit_wesl(&self, ctx: &mut CodegenCtx) {
        for decl in self.required_bindings() {
            let _ = ctx.declare_binding(decl);
        }
        match self.model {
            EmberShadingModel::Unlit => {
                ctx.push_fragment(single_closure_fragment(ShadingBasis::Unlit));
            }
            EmberShadingModel::Pbr => {
                ctx.push_fragment(single_closure_fragment(ShadingBasis::Pbr));
            }
            EmberShadingModel::Npr => {
                ctx.push_fragment(single_closure_fragment(ShadingBasis::Npr));
            }
            EmberShadingModel::Custom(id) => {
                ctx.push_fragment(single_closure_fragment(ShadingBasis::Custom(id)));
            }
            EmberShadingModel::Hybrid {
                base,
                overlay,
                weight,
            } => {
                emit_hybrid(ctx, base, overlay, weight);
            }
        }
    }
}

/// Emits the closure call(s) and bindings for a `model`/`program` pair into `ctx`.
///
/// A free-function convenience over [`ShadingClosure::emit_wesl`] for callers
/// that already hold the model and program separately.
pub fn emit_shading_closure(
    model: EmberShadingModel,
    program: ShadingProgram,
    ctx: &mut CodegenCtx,
) {
    ShadingClosure::new(model, program).emit_wesl(ctx);
}

/// The `WESL` closure function name for a single [`ShadingBasis`] lobe.
///
/// `Unlit`/`PBR`/`NPR` map to the shared built-in closures; a `Custom` basis maps
/// to the stable `shade_custom_<id>` extension-point symbol the user supplies.
#[must_use]
fn basis_closure_name(basis: ShadingBasis) -> String {
    match basis {
        ShadingBasis::Unlit => String::from("shade_unlit"),
        ShadingBasis::Pbr => String::from("shade_pbr"),
        ShadingBasis::Npr => String::from("shade_npr"),
        ShadingBasis::Custom(id) => format!("shade_custom_{id}"),
    }
}

/// A single-closure assignment fragment: `let shaded_color = <closure>(particle_index);`.
///
/// A `Custom` basis additionally carries a trailing comment marking the
/// user-authored `WESL` extension point (design §19).
#[must_use]
fn single_closure_fragment(basis: ShadingBasis) -> String {
    let name = basis_closure_name(basis);
    let mut fragment = format!("let {SHADED_RESULT} = {name}(particle_index);");
    if matches!(basis, ShadingBasis::Custom(_)) {
        fragment.push_str(" // user WESL closure extension point");
    }
    fragment
}

/// Emits the two-lobe `Hybrid` blend: each lobe closure into a named local, then
/// a `blend_shading` combine weighted by the per-particle blend weight.
///
/// The lobe locals are `shading_base` and `shading_overlay`; the combine binds
/// the final [`SHADED_RESULT`]. The weight is clamped to `0..=1` and formatted as
/// a `WESL` float literal so a pure-`base` (`0`) or pure-`overlay` (`1`) blend
/// still emits a well-formed literal (design §19).
fn emit_hybrid(ctx: &mut CodegenCtx, base: ShadingBasis, overlay: ShadingBasis, weight: f32) {
    let base_name = basis_closure_name(base);
    let overlay_name = basis_closure_name(overlay);
    ctx.push_fragment(format!("let shading_base = {base_name}(particle_index);"));
    ctx.push_fragment(format!(
        "let shading_overlay = {overlay_name}(particle_index);"
    ));
    let literal = float_literal(sanitized_weight(weight));
    ctx.push_fragment(format!(
        "let {SHADED_RESULT} = blend_shading(shading_base, shading_overlay, {literal});"
    ));
}

/// Clamps a blend weight into the `0..=1` contract, mapping any non-finite input
/// to `0.0` (pure base) so the emitted literal is always well formed.
#[must_use]
fn sanitized_weight(weight: f32) -> f32 {
    if weight.is_finite() {
        // `clamp` orders against finite bounds only; the non-finite case is
        // handled above so this never sees a `NaN`.
        weight.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Formats a finite `f32` as a `WESL`/`WGSL` float literal.
///
/// Rust's default `f32` formatting is deterministic but omits the fractional
/// part for whole numbers (for example `1`), which `WGSL` would parse as an
/// integer literal. A decimal suffix is appended when the rendered text carries
/// neither a decimal point nor an exponent so the result is always a float.
#[must_use]
fn float_literal(value: f32) -> String {
    let mut text = format!("{value}");
    let has_fraction =
        text.contains('.') || text.contains('e') || text.contains('E');
    if !has_fraction {
        text.push_str(".0");
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::particle::lod::ParticleQuality;
    use crate::particle::shading::{
        resolve_shading_program, LightingServiceCaps, ShadingProgramInput,
    };
    use crate::particle::sort_cull::BlendMode;

    /// Full lighting capabilities so lit closures subscribe to every service.
    fn all_caps() -> LightingServiceCaps {
        LightingServiceCaps {
            shadow_maps: true,
            global_illumination: true,
            ray_tracing: true,
        }
    }

    /// Resolves a program for a model with full caps, non-volumetric, high quality.
    fn program_for(model: EmberShadingModel, volumetric: bool) -> ShadingProgram {
        resolve_shading_program(ShadingProgramInput {
            model,
            blend: BlendMode::AlphaBlend,
            caps: all_caps(),
            volumetric,
            quality: ParticleQuality::High,
        })
    }

    fn emit(model: EmberShadingModel, volumetric: bool) -> CodegenCtx {
        let program = program_for(model, volumetric);
        let mut ctx = CodegenCtx::new();
        ShadingClosure::new(model, program).emit_wesl(&mut ctx);
        ctx
    }

    fn binding_names(ctx: &CodegenCtx) -> Vec<&'static str> {
        ctx.bindings().iter().map(|decl| decl.name).collect()
    }

    #[test]
    fn unlit_emits_closure_and_declares_no_bindings() {
        let ctx = emit(EmberShadingModel::Unlit, false);
        let source = ctx.assembled_source();
        assert!(source.contains("shade_unlit(particle_index)"), "{source}");
        assert!(source.contains("let shaded_color ="), "{source}");
        // An unlit kernel reads no attributes and subscribes to no lighting.
        assert_eq!(ctx.binding_count(), 0);
    }

    #[test]
    fn pbr_emits_closure_and_declares_material_and_lighting_bindings() {
        let ctx = emit(EmberShadingModel::Pbr, false);
        let source = ctx.assembled_source();
        assert!(source.contains("shade_pbr(particle_index)"), "{source}");
        let names = binding_names(&ctx);
        // Footprint: normal + tangent + material params.
        assert!(names.contains(&"shading_normal_buf"));
        assert!(names.contains(&"shading_tangent_buf"));
        assert!(names.contains(&"shading_material_params"));
        // Lit model with full caps subscribes to every shared service.
        assert!(names.contains(&"lighting_clusters"));
        assert!(names.contains(&"lighting_shadow_maps"));
        assert!(names.contains(&"lighting_gi_probes"));
        assert!(names.contains(&"lighting_ray_scene"));
        // PBR reads no ramp or custom parameters.
        assert!(!names.contains(&"shading_ramp_lut"));
        assert!(!names.contains(&"shading_custom_params"));
    }

    #[test]
    fn npr_emits_closure_and_declares_normal_and_ramp() {
        let ctx = emit(EmberShadingModel::Npr, false);
        let source = ctx.assembled_source();
        assert!(source.contains("shade_npr(particle_index)"), "{source}");
        let names = binding_names(&ctx);
        assert!(names.contains(&"shading_normal_buf"));
        assert!(names.contains(&"shading_ramp_lut"));
        // NPR reads no tangent frame or material parameters.
        assert!(!names.contains(&"shading_tangent_buf"));
        assert!(!names.contains(&"shading_material_params"));
    }

    #[test]
    fn custom_emits_handle_qualified_extension_point() {
        let ctx = emit(EmberShadingModel::Custom(7), false);
        let source = ctx.assembled_source();
        // The handle is encoded in the closure symbol, not in a binding name.
        assert!(source.contains("shade_custom_7(particle_index)"), "{source}");
        assert!(
            source.contains("user WESL closure extension point"),
            "{source}"
        );
        // Custom closures read their custom-parameter block.
        assert!(binding_names(&ctx).contains(&"shading_custom_params"));
    }

    #[test]
    fn hybrid_emits_both_lobes_and_a_weighted_blend() {
        let model = EmberShadingModel::Hybrid {
            base: ShadingBasis::Pbr,
            overlay: ShadingBasis::Npr,
            weight: 0.25,
        };
        let ctx = emit(model, false);
        let source = ctx.assembled_source();
        assert!(
            source.contains("let shading_base = shade_pbr(particle_index);"),
            "{source}"
        );
        assert!(
            source.contains("let shading_overlay = shade_npr(particle_index);"),
            "{source}"
        );
        assert!(
            source.contains("blend_shading(shading_base, shading_overlay, 0.25)"),
            "{source}"
        );
        // Footprint is the union of both lobes: PBR's material/tangent plus
        // NPR's ramp, all over a shared normal.
        let names = binding_names(&ctx);
        assert!(names.contains(&"shading_normal_buf"));
        assert!(names.contains(&"shading_tangent_buf"));
        assert!(names.contains(&"shading_material_params"));
        assert!(names.contains(&"shading_ramp_lut"));
    }

    #[test]
    fn hybrid_pure_bound_weights_emit_float_literals() {
        let base_zero = EmberShadingModel::Hybrid {
            base: ShadingBasis::Pbr,
            overlay: ShadingBasis::Npr,
            weight: 0.0,
        };
        let overlay_one = EmberShadingModel::Hybrid {
            base: ShadingBasis::Pbr,
            overlay: ShadingBasis::Npr,
            weight: 1.0,
        };
        assert!(emit(base_zero, false)
            .assembled_source()
            .contains("shading_overlay, 0.0)"));
        assert!(emit(overlay_one, false)
            .assembled_source()
            .contains("shading_overlay, 1.0)"));
    }

    #[test]
    fn hybrid_out_of_range_weight_is_clamped() {
        let over = EmberShadingModel::Hybrid {
            base: ShadingBasis::Unlit,
            overlay: ShadingBasis::Npr,
            weight: 4.0,
        };
        assert!(emit(over, false)
            .assembled_source()
            .contains("shading_overlay, 1.0)"));
        let non_finite = EmberShadingModel::Hybrid {
            base: ShadingBasis::Unlit,
            overlay: ShadingBasis::Npr,
            weight: f32::NAN,
        };
        assert!(emit(non_finite, false)
            .assembled_source()
            .contains("shading_overlay, 0.0)"));
    }

    #[test]
    fn hybrid_custom_lobe_emits_handle_qualified_call() {
        let model = EmberShadingModel::Hybrid {
            base: ShadingBasis::Custom(3),
            overlay: ShadingBasis::Unlit,
            weight: 0.5,
        };
        let source = emit(model, false).assembled_source();
        assert!(
            source.contains("let shading_base = shade_custom_3(particle_index);"),
            "{source}"
        );
        assert!(
            source.contains("let shading_overlay = shade_unlit(particle_index);"),
            "{source}"
        );
    }

    #[test]
    fn deep_shadow_binding_follows_the_program_tier() {
        // A volumetric lit renderer carries a deep-shadow tier, so the
        // self-shadow volume binding is declared.
        let volumetric = emit(EmberShadingModel::Pbr, true);
        assert!(binding_names(&volumetric).contains(&"shading_deep_opacity"));
        // A non-volumetric renderer has no deep-shadow tier and declares none.
        let flat = emit(EmberShadingModel::Pbr, false);
        assert!(!binding_names(&flat).contains(&"shading_deep_opacity"));
    }

    #[test]
    fn unlit_subscribes_to_no_lighting_even_when_volumetric() {
        // Unlit never needs lighting, so a volumetric unlit renderer still
        // declares neither lighting nor deep-shadow bindings.
        let ctx = emit(EmberShadingModel::Unlit, true);
        assert_eq!(ctx.binding_count(), 0);
    }

    #[test]
    fn assembled_source_composes_fragments() {
        let model = EmberShadingModel::Hybrid {
            base: ShadingBasis::Pbr,
            overlay: ShadingBasis::Npr,
            weight: 0.5,
        };
        let ctx = emit(model, false);
        // Hybrid pushes three fragments (base, overlay, blend); the assembled
        // source joins them with newlines.
        assert_eq!(ctx.fragment_count(), 3);
        let source = ctx.assembled_source();
        assert_eq!(source.lines().count(), 3);
        assert!(source.contains('\n'));
    }

    #[test]
    fn emission_is_deterministic() {
        let model = EmberShadingModel::Hybrid {
            base: ShadingBasis::Pbr,
            overlay: ShadingBasis::Custom(9),
            weight: 0.75,
        };
        let first = emit(model, true);
        let second = emit(model, true);
        assert_eq!(first.assembled_source(), second.assembled_source());
        assert_eq!(binding_names(&first), binding_names(&second));
    }

    #[test]
    fn free_function_matches_method_emission() {
        let model = EmberShadingModel::Pbr;
        let program = program_for(model, false);
        let mut via_fn = CodegenCtx::new();
        emit_shading_closure(model, program, &mut via_fn);
        let mut via_method = CodegenCtx::new();
        ShadingClosure::new(model, program).emit_wesl(&mut via_method);
        assert_eq!(via_fn.assembled_source(), via_method.assembled_source());
        assert_eq!(binding_names(&via_fn), binding_names(&via_method));
    }

    #[test]
    fn required_bindings_match_declared_bindings() {
        let model = EmberShadingModel::Npr;
        let program = program_for(model, true);
        let closure = ShadingClosure::new(model, program);
        let mut ctx = CodegenCtx::new();
        closure.emit_wesl(&mut ctx);
        let expected: Vec<&'static str> =
            closure.required_bindings().iter().map(|d| d.name).collect();
        assert_eq!(binding_names(&ctx), expected);
    }
}
