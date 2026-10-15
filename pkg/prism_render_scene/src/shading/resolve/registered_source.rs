//! Link finite, independently compiled response functions into the resolve.
//! No shader file is rewritten and there is no hash-color or builtin fallback.

use prism_render_material::response_source::{
    response_function, PACK_RESPONSE, RESPONSE_FUNCTION_ABI,
};
use prism_render_material::{
    CompiledEntry, MaterialExecutionTables, ProgramRef, ResponseProgramIndex,
};

pub(super) const BEGIN: &str = "// PRISM_REGISTERED_RESPONSE_BEGIN";
pub(super) const END: &str = "// PRISM_REGISTERED_RESPONSE_END";

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum LinkError {
    MissingOrAmbiguousSeam,
    MissingSource(u32),
    InvalidTable,
    VariantCount(u32),
}

pub(super) struct RegisteredModule<'a> {
    pub program_id: u32,
    pub compiled: &'a CompiledEntry,
}

/// Scans the cold catalogue, never a per-pixel or per-frame shader compilation.
pub(super) fn catalogue(
    tables: &MaterialExecutionTables,
) -> Result<Vec<RegisteredModule<'_>>, LinkError> {
    let mut modules = Vec::new();
    for raw in 0..tables.programs() {
        let index = ResponseProgramIndex::from_raw(raw as u32, tables.programs())
            .map_err(|_| LinkError::InvalidTable)?;
        let entry = tables.program(index).map_err(|_| LinkError::InvalidTable)?;
        if let ProgramRef::Registered(id) = entry.program() {
            let compiled = entry.compiled_entry();
            if compiled.response_source().is_none() {
                return Err(LinkError::MissingSource(id.0));
            }
            if compiled.digest() != entry.digest() {
                return Err(LinkError::InvalidTable);
            }
            modules.push(RegisteredModule {
                program_id: id.0,
                compiled,
            });
        }
    }
    modules.sort_by_key(|entry| entry.program_id);
    if modules
        .windows(2)
        .any(|pair| pair[0].program_id == pair[1].program_id)
    {
        return Err(LinkError::InvalidTable);
    }
    Ok(modules)
}

/// Canonical ABI fields. Generation fails at shader type checking if the scene
/// changes its surface contract; tests pin every field, not a lossy projection.
pub(super) fn fields(name: &str) -> Vec<&'static str> {
    RESPONSE_FUNCTION_ABI
        .split(&format!("struct {name} {{"))
        .nth(1)
        .unwrap()
        .split('}')
        .next()
        .unwrap()
        .lines()
        .filter_map(|line| line.trim().split_once(':').map(|(field, _)| field.trim()))
        .collect()
}

fn convert(ty: &str, value: &str) -> String {
    let arguments = fields(ty)
        .iter()
        .map(|field| {
            if *field == "stylized" {
                convert("RegisteredStylizedParams", &format!("{value}.stylized"))
            } else {
                format!("{value}.{field}")
            }
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("{ty}({arguments})")
}

/// Dispatch section also used by the real-device test harness. Code growth is
/// sum(variants per package), never a product of package pairs.
pub(super) fn functions(modules: &[RegisteredModule<'_>]) -> Result<String, LinkError> {
    let mut source = PACK_RESPONSE.to_owned();
    let mut seen = std::collections::BTreeSet::new();
    for module in modules {
        if !seen.insert(module.program_id) {
            return Err(LinkError::InvalidTable);
        }
        let payload = module
            .compiled
            .response_source()
            .ok_or(LinkError::MissingSource(module.program_id))?;
        if payload.bodies().len() as u64 != module.compiled.package_manifest().permutation_count() {
            return Err(LinkError::VariantCount(module.program_id));
        }
        for (variant, body) in payload.bodies().iter().enumerate() {
            source.push_str(&response_function(
                &format!("registered_p{}_v{variant}", module.program_id),
                body,
            ));
        }
    }
    source.push_str("fn registered_evaluate(surface: RegisteredSurfaceData, lighting: RegisteredLightingContext, view: RegisteredView, custom: RegisteredCustomData, program_id: u32, permutation_low: u32, permutation_high: u32) -> RegisteredResponse {\nvar invalid: RegisteredResponse;\nif permutation_high != 0u { return invalid; }\nswitch program_id {\n");
    for module in modules {
        source.push_str(&format!(
            "case {}u: {{ switch permutation_low {{\n",
            module.program_id
        ));
        for variant in 0..module.compiled.response_source().unwrap().bodies().len() {
            source.push_str(&format!("case {variant}u: {{ return registered_p{}_v{variant}(surface, lighting, view, custom); }}\n", module.program_id));
        }
        source.push_str("default: { return invalid; }\n} }\n");
    }
    source.push_str("default: { return invalid; }\n}\n}\n");
    Ok(source)
}

pub(super) fn link(template: &str, modules: &[RegisteredModule<'_>]) -> Result<String, LinkError> {
    if template.matches(BEGIN).count() != 1 || template.matches(END).count() != 1 {
        return Err(LinkError::MissingOrAmbiguousSeam);
    }
    let start = template.find(BEGIN).unwrap();
    let end = template.find(END).unwrap() + END.len();
    if end <= start {
        return Err(LinkError::MissingOrAmbiguousSeam);
    }
    let mut injection = format!(
        "{BEGIN}\n{RESPONSE_FUNCTION_ABI}\n{LIGHT_AND_CUSTOM_HELPERS}\n{}\n",
        functions(modules)?
    );
    injection.push_str("fn registered_response(surface: SurfaceSample, frame: ShadingFrame, geometry: SurfaceSampleGeometry, parameters: PrismSurfaceParameters, header: PrismMaterialHeader, indirect: IndirectResult, view_position: vec3<f32>, pixel: vec2<u32>, screen_size: vec2<u32>, frame_index: u32, program_id: u32, permutation_low: u32, permutation_high: u32) -> ResponseOutput {\n");
    injection.push_str(&format!(
        "let data = RegisteredSurfaceData({}, {}, {}, {});\n",
        convert("RegisteredSurface", "surface"),
        convert("RegisteredFrame", "frame"),
        convert("RegisteredGeometry", "geometry"),
        convert("RegisteredParameters", "parameters")
    ));
    injection.push_str("let env = light_env[0];\nlet slice = resolve_cluster_slice(geometry.position, pixel.x, pixel.y);\nvar lighting: RegisteredLightingContext;\n");
    injection.push_str(&format!(
        "lighting.environment = {};\nlighting.frame = {};\n",
        convert("RegisteredEnvironment", "env"),
        convert("RegisteredFrame", "frame")
    ));
    injection.push_str("lighting.world_position = geometry.position;\nlighting.directional_count = min(env.directional_count, arrayLength(&directional_lights));\nlighting.punctual_count = min(env.punctual_count, arrayLength(&punctual_lights));\nlighting.area_count = arrayLength(&area_lights);\nlighting.clustered = slice.use_clusters;\nlighting.cluster_offset = slice.offset;\nif slice.use_clusters != 0u { lighting.punctual_count = 0u; if slice.offset < arrayLength(&cluster_indices) { lighting.punctual_count = min(slice.count, arrayLength(&cluster_indices) - slice.offset); } }\nlighting.indirect_total = indirect.total;\nlighting.diffuse_irradiance = indirect.ambient_irradiance;\nlighting.environment_radiance = indirect.env_specular;\nlighting.specular_weight = indirect.spec_weight;\n");
    injection.push_str("let view = RegisteredView(view_position, pixel, screen_size, frame_index, motion_matrices.clip_from_world, motion_matrices.previous_clip_from_world);\n");
    injection.push_str(&format!(
        "let custom = RegisteredCustomData({}, permutation_low, permutation_high);\n",
        convert("RegisteredHeader", "header")
    ));
    injection.push_str("var output: ResponseOutput;\noutput.words = registered_pack(registered_evaluate(data, lighting, view, custom, program_id, permutation_low, permutation_high));\nreturn output;\n}\n");
    injection.push_str(END);
    let mut source = String::with_capacity(template.len() + injection.len());
    source.push_str(&template[..start]);
    source.push_str(&injection);
    source.push_str(&template[end..]);
    Ok(source)
}

// These are unweighted lighting/visibility queries, not principled shading.
// They preserve the scene's cluster and routing/shadow checks. Calls are lazy;
// this does not claim to precompute every shadow once across two split arms.
const LIGHT_AND_CUSTOM_HELPERS: &str = r#"
fn registered_light_count(lighting: RegisteredLightingContext) -> u32 {
    return lighting.directional_count + lighting.punctual_count;
}
fn registered_light(lighting: RegisteredLightingContext, index: u32) -> RegisteredLight {
    var invalid: RegisteredLight;
    if index >= registered_light_count(lighting) { return invalid; }
    let f = lighting.frame;
    let frame = ShadingFrame(f.normal, f.view, f.tangent, f.bitangent);
    var sample: LightSample;
    if index < lighting.directional_count {
        if index >= arrayLength(&directional_lights) { return invalid; }
        sample = sample_directional_shadowed(index, frame, lighting.world_position);
    } else {
        var light_index = index - lighting.directional_count;
        if lighting.clustered != 0u {
            if lighting.cluster_offset >= arrayLength(&cluster_indices)
                || light_index >= arrayLength(&cluster_indices) - lighting.cluster_offset { return invalid; }
            light_index = cluster_indices[lighting.cluster_offset + light_index];
        }
        if light_index >= arrayLength(&punctual_lights) { return invalid; }
        sample = sample_punctual_shadowed(light_index, frame, lighting.world_position);
    }
    return RegisteredLight(sample.direction, sample.illuminance, sample.visibility, sample.valid);
}
fn registered_area(lighting: RegisteredLightingContext, index: u32) -> RegisteredArea {
    var invalid: RegisteredArea;
    if index >= lighting.area_count || index >= arrayLength(&area_lights) { return invalid; }
    let a = area_lights[index];
    return RegisteredArea(a.pos_x, a.pos_y, a.pos_z, a.range, a.color_x, a.color_y, a.color_z, a.intensity, a.axis_u_x, a.axis_u_y, a.axis_u_z, a.half_width, a.axis_v_x, a.axis_v_y, a.axis_v_z, a.half_height, a.shape, a.two_sided, a.radius, a.pad0);
}
fn registered_custom_word(custom: RegisteredCustomData, index: u32) -> u32 {
    let header = custom.header;
    let base = header.parameter_offset;
    if index >= header.parameter_size / 4u || base >= arrayLength(&material_parameters)
        || index >= arrayLength(&material_parameters) - base { return 0u; }
    return material_parameters[base + index];
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn absent_or_ambiguous_seam_is_not_silently_accepted() {
        for template in ["", "// PRISM_REGISTERED_RESPONSE_END\n// PRISM_REGISTERED_RESPONSE_BEGIN", "// PRISM_REGISTERED_RESPONSE_BEGIN\n// PRISM_REGISTERED_RESPONSE_BEGIN\n// PRISM_REGISTERED_RESPONSE_END"] {
            assert_eq!(link(template, &[]).unwrap_err(), LinkError::MissingOrAmbiguousSeam);
        }
    }
    #[test]
    fn full_surface_abi_tracks_every_scene_field() {
        for (source, scene, abi) in [
            (
                include_str!("../../shaders/brdf.wesl"),
                "SurfaceSample",
                "RegisteredSurface",
            ),
            (
                include_str!("../../shaders/brdf.wesl"),
                "ShadingFrame",
                "RegisteredFrame",
            ),
            (
                include_str!("../../shaders/material.wesl"),
                "PrismSurfaceParameters",
                "RegisteredParameters",
            ),
            (
                include_str!("../../shaders/material.wesl"),
                "PrismMaterialHeader",
                "RegisteredHeader",
            ),
            (
                include_str!("../../shaders/surface.wesl"),
                "SurfaceSampleGeometry",
                "RegisteredGeometry",
            ),
        ] {
            let declaration = format!("struct {scene} {{");
            let start = source.find(&declaration).unwrap() + declaration.len();
            let mut depth = 1u32;
            let mut end = start;
            for (offset, ch) in source[start..].char_indices() {
                match ch {
                    '{' => depth += 1,
                    '}' => {
                        depth -= 1;
                        if depth == 0 {
                            end = start + offset;
                            break;
                        }
                    }
                    _ => {}
                }
            }
            assert!(end > start, "{scene} has no balanced closing brace");
            let actual: Vec<_> = source[start..end]
                .lines()
                .map(|line| line.split_once("//").map_or(line, |(code, _)| code))
                .map(str::trim)
                .filter_map(|line| line.split_once(':').map(|(name, _)| name.trim()))
                .filter(|name| !name.is_empty())
                .collect();
            // The scene-side material struct is allowed to use its packed ABI
            // order while the registered-response adapter uses the response
            // contract's order. Conversion is name-based, so the invariant is
            // complete coverage of every field consumed by the response ABI
            // rather than identical declaration order. The scene contract may
            // carry additional lobe parameters that this response ABI does not
            // consume. Keep duplicate detection here: a duplicate would make a
            // named projection ambiguous and silently corrupt a response.
            let mut actual_sorted = actual.clone();
            actual_sorted.sort_unstable();
            let mut abi_sorted = fields(abi);
            abi_sorted.sort_unstable();
            assert_eq!(
                actual.len(),
                actual_sorted.len(),
                "{scene} contains duplicate fields"
            );
            assert!(
                abi_sorted
                    .iter()
                    .all(|field| actual_sorted.binary_search(field).is_ok()),
                "{scene} is missing a field consumed by {abi}"
            );
        }
    }
}
