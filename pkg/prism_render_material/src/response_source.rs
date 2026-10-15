//! Finite, executable response-function packages, not metadata-only registrations.
//!
//! Authors supply WGSL **function bodies** (one per declared static permutation).
//! They are type checked with the fixed ABI and compiled into an actual standalone
//! compute artifact. The scene linker reuses these very bodies and supplies the
//! same read-only light/custom-data operations from its resident resources.
//! No imports, new bindings, global declarations or escaping the function body
//! are supported. Device compilation of the final linked resolve is still required.

use crate::{CompiledEntry, CompiledEntryError};
use alloc::{format, string::String, sync::Arc, vec::Vec};
use prism_render_architecture::shader_package::ShaderPackageManifest;

pub const RESPONSE_FUNCTION_ABI: &str = include_str!("response_abi.wgsl");
pub const MAX_RESPONSE_VARIANTS: usize = 256;
pub const MAX_RESPONSE_BODY_BYTES: usize = 64 * 1024;

/// Immutable source whose exact functions have passed Naga and SPIR-V validation.
#[derive(Clone, Debug)]
pub struct ResponseSource {
    bodies: Arc<[String]>,
    standalone: Arc<str>,
}

impl ResponseSource {
    pub fn bodies(&self) -> &[String] {
        &self.bodies
    }
    /// Real standalone compute implementation of this package's functions.
    /// Bindings: input records, unweighted light samples, area primitives,
    /// packed custom words, and 16-word output records (all in group zero).
    pub fn standalone_wgsl(&self) -> &str {
        &self.standalone
    }
}

/// The same header is used both by package validation and the scene linker.
pub fn response_function(name: &str, body: &str) -> String {
    format!("fn {name}(surface: RegisteredSurfaceData, lighting: RegisteredLightingContext, view: RegisteredView, custom: RegisteredCustomData) -> RegisteredResponse {{\n{body}\n}}\n")
}

pub const STANDALONE_RESOURCES: &str = r#"
struct PrismResponseInput {
    surface: RegisteredSurfaceData,
    lighting: RegisteredLightingContext,
    view: RegisteredView,
    custom: RegisteredCustomData,
}
struct PrismResponseWords { words: array<f32, 16>, }
@group(0) @binding(0) var<storage, read> prism_response_inputs: array<PrismResponseInput>;
@group(0) @binding(1) var<storage, read> prism_response_lights: array<RegisteredLight>;
@group(0) @binding(2) var<storage, read> prism_response_areas: array<RegisteredArea>;
@group(0) @binding(3) var<storage, read> prism_response_custom: array<u32>;
@group(0) @binding(4) var<storage, read_write> prism_response_outputs: array<PrismResponseWords>;
fn registered_light_count(lighting: RegisteredLightingContext) -> u32 {
    return lighting.directional_count + lighting.punctual_count;
}
fn registered_light(lighting: RegisteredLightingContext, index: u32) -> RegisteredLight {
    var invalid: RegisteredLight;
    if index >= registered_light_count(lighting) || index >= arrayLength(&prism_response_lights) { return invalid; }
    return prism_response_lights[index];
}
fn registered_area(lighting: RegisteredLightingContext, index: u32) -> RegisteredArea {
    var invalid: RegisteredArea;
    if index >= lighting.area_count || index >= arrayLength(&prism_response_areas) { return invalid; }
    return prism_response_areas[index];
}
fn registered_custom_word(custom: RegisteredCustomData, index: u32) -> u32 {
    let header = custom.header;
    let base = header.parameter_offset;
    if index >= header.parameter_size / 4u
        || base >= arrayLength(&prism_response_custom)
        || index >= arrayLength(&prism_response_custom) - base { return 0u; }
    return prism_response_custom[base + index];
}
"#;

pub const PACK_RESPONSE: &str = r#"
fn registered_pack(response: RegisteredResponse) -> array<f32, 16> {
    return array<f32, 16>(
        response.direct.x, response.direct.y, response.direct.z,
        response.indirect_diffuse.x, response.indirect_diffuse.y, response.indirect_diffuse.z,
        response.reflection.x, response.reflection.y, response.reflection.z,
        response.transmission.x, response.transmission.y, response.transmission.z,
        response.emission.x, response.emission.y, response.emission.z, response.alpha);
}
"#;

impl CompiledEntry {
    /// Compiles every finite variant to a real executable artifact. This does
    /// not attach source to an unrelated SPIR-V binary or invent a hash response.
    pub fn from_response_wgsl(
        manifest: ShaderPackageManifest,
        bodies: Vec<String>,
    ) -> Result<Self, CompiledEntryError> {
        let fail = |message: String| CompiledEntryError::ResponseSource(message);
        if bodies.is_empty()
            || bodies.len() > MAX_RESPONSE_VARIANTS
            || bodies.len() as u64 != manifest.permutation_count()
        {
            return Err(fail(
                "finite variant count must match the manifest (1..=256)".into(),
            ));
        }
        // The reserved prefix is the standalone adapter's private namespace,
        // never part of the author ABI. It also prevents cross-variant calls.
        for body in &bodies {
            if body.trim().is_empty()
                || body.len() > MAX_RESPONSE_BODY_BYTES
                || body.contains("prism_response_")
            {
                return Err(fail("empty/oversized body or reserved identifier".into()));
            }
        }
        let mut source =
            format!("{RESPONSE_FUNCTION_ABI}\n{STANDALONE_RESOURCES}\n{PACK_RESPONSE}\n");
        for (index, body) in bodies.iter().enumerate() {
            source.push_str(&response_function(
                &format!("prism_response_variant_{index}"),
                body,
            ));
        }
        source.push_str(&format!("@compute @workgroup_size(1) fn {}(@builtin(global_invocation_id) id: vec3<u32>) {{\nif id.x >= arrayLength(&prism_response_inputs) || id.x >= arrayLength(&prism_response_outputs) {{ return; }}\nlet input = prism_response_inputs[id.x];\nvar response: RegisteredResponse;\nif input.custom.permutation_high != 0u {{ return; }}\nswitch input.custom.permutation_low {{\n", manifest.entry_point));
        for index in 0..bodies.len() {
            source.push_str(&format!("case {index}u: {{ response = prism_response_variant_{index}(input.surface, input.lighting, input.view, input.custom); }}\n"));
        }
        source.push_str("default: { return; }\n}\nprism_response_outputs[id.x].words = registered_pack(response);\n}\n");
        let module =
            naga::front::wgsl::parse_str(&source).map_err(|e| fail(e.emit_to_string(&source)))?;
        // Compare the complete declaration set with the ABI adapter, so a body
        // cannot close its wrapper and add globals/functions or a second entry.
        if module.entry_points.len() != 1
            || module.global_variables.len() != 5
            || module.functions.len() != bodies.len() + 5
            || module.entry_points[0].name != manifest.entry_point
        {
            return Err(fail("response body escaped its function boundary".into()));
        }
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .map_err(|e| fail(e.to_string()))?;
        let words =
            naga::back::spv::write_vec(&module, &info, &naga::back::spv::Options::default(), None)
                .map_err(|e| fail(e.to_string()))?;
        let mut compiled = Self::from_spirv(manifest, words)?;
        compiled.response_source = Some(ResponseSource {
            bodies: bodies.into(),
            standalone: source.into(),
        });
        Ok(compiled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_architecture::{
        abi::{AbiHash, AbiVersion},
        shader_package::{
            ArtifactDigest, PermutationSpace, ShaderPackageId, ShaderPackageVariantManifest,
        },
    };
    pub fn manifest(count: u32) -> ShaderPackageManifest {
        ShaderPackageManifest {
            id: ShaderPackageId::new("actual-response"),
            source_revision: "1".into(),
            entry_point: "main".into(),
            abi_version: AbiVersion::from_parts(1, 0),
            abi_hash: AbiHash::from_bytes(RESPONSE_FUNCTION_ABI.as_bytes()),
            variants: ShaderPackageVariantManifest::from_digests(
                &PermutationSpace::new([count]),
                (0..count).map(|index| ArtifactDigest([index as u8 + 1; 32])),
            )
            .unwrap(),
        }
    }
    #[test]
    fn real_source_artifact_changes_with_author_code() {
        let a = CompiledEntry::from_response_wgsl(manifest(1), vec!["return RegisteredResponse(vec3(1.0), vec3(2.0), vec3(3.0), vec3(4.0), vec3(5.0), 0.75);".into()]).unwrap();
        let b = CompiledEntry::from_response_wgsl(manifest(1), vec!["return RegisteredResponse(vec3(7.0), vec3(2.0), vec3(3.0), vec3(4.0), vec3(5.0), 0.75);".into()]).unwrap();
        assert_ne!(a.digest(), b.digest());
        assert!(a.spirv().len() > 100);
        assert_eq!(a.response_source().unwrap().bodies().len(), 1);
    }
    #[test]
    fn validates_body_types_boundary_and_all_variants() {
        for body in [
            "return 7u;",
            "}",
            "return prism_response_inputs[0].surface;",
            "var a: RegisteredResponse; return a; } fn escaped() {",
        ] {
            assert!(
                CompiledEntry::from_response_wgsl(manifest(1), vec![body.into()]).is_err(),
                "accepted {body}"
            );
        }
        assert!(CompiledEntry::from_response_wgsl(
            manifest(2),
            vec!["var r: RegisteredResponse; return r;".into()]
        )
        .is_err());
    }
}
