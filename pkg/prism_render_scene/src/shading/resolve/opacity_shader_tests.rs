//! Source-contract regressions; no GPU execution or full WESL type checking.
//! Also runnable alone with `rustc --test` without building the render world.

const RESOLVE: &str = include_str!("../../shaders/shading_resolve.wesl");
const OPACITY: &str = "clamp(parameters.geometry_opacity, 0.0, 1.0)";

fn code(source: &str) -> String {
    source
        .lines()
        .map(|line| line.split_once("//").map_or(line, |(code, _)| code))
        .collect::<String>()
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect()
}

fn builtin_response_has_authored_opacity(source: &str) -> bool {
    let source = code(source);
    let Some((_, arm)) = source.split_once("fnevaluate_response_arm(") else {
        return false;
    };
    let Some((arm, _)) = arm.split_once("@compute") else {
        return false;
    };
    let Some((_, output)) = arm.split_once("arm.output=ResponseOutput(array<f32,16>(") else {
        return false;
    };
    let Some((words, _)) = output.split_once("));") else {
        return false;
    };
    // Pin the final lane of the real builtin output, not a comment, another
    // function, or the neutral/registered response's alpha.
    words.ends_with(&format!(
        "emissive.x,emissive.y,emissive.z,{},",
        code(OPACITY)
    ))
}

#[test]
fn builtin_response_exports_authored_opacity_in_the_alpha_lane() {
    assert!(builtin_response_has_authored_opacity(RESOLVE));
    assert!(code(RESOLVE).contains("CompositionControl(0.0,0u,left_arm.output.words[15],1u,)"));
}

#[test]
fn builtin_response_opacity_rejects_wrong_sources_and_decoy_markers() {
    assert_eq!(RESOLVE.matches(OPACITY).count(), 1);
    for wrong in [
        "1.0",
        "0.0",
        "parameters.geometry_opacity",
        "clamp(surface.geometry_opacity, 0.0, 1.0)",
        "clamp(parameters.base_color.w, 0.0, 1.0)",
        "clamp(sampled.base_alpha, 0.0, 1.0)",
        "clamp(surface.transmission, 0.0, 1.0)",
        "clamp(parameters.geometry_opacity * 0.5, 0.0, 1.0)",
    ] {
        let mutated = RESOLVE.replacen(OPACITY, wrong, 1);
        // A correct expression in a comment or unrelated function must not
        // hide a broken production output (the old global assertion could).
        let decoy = format!(
            "// {OPACITY}\n{mutated}\nfn opacity_decoy(parameters: PrismSurfaceParameters) -> f32 {{ return {OPACITY}; }}"
        );
        assert!(
            !builtin_response_has_authored_opacity(&decoy),
            "accepted wrong response alpha: {wrong}"
        );
    }
}

#[test]
fn dynamic_bsdf_modulation_has_no_geometry_opacity_to_export() {
    let brdf = code(include_str!("../../shaders/brdf.wesl"));
    let (_, surface) = brdf.split_once("structSurfaceSample{").unwrap();
    let (surface, _) = surface.split_once('}').unwrap();
    assert!(!surface.contains("geometry_opacity:"));

    let parameters = code(include_str!(
        "../../../../prism_render_material/src/shaders/material.wesl"
    ));
    let (_, parameters) = parameters
        .split_once("structPrismSurfaceParameters{")
        .unwrap();
    let (parameters, _) = parameters.split_once('}').unwrap();
    assert!(parameters.contains("geometry_opacity:f32,"));

    let dynamic = code(include_str!("../../shaders/dynamic.wesl"));
    assert!(!dynamic.contains("geometry_opacity"));
    assert!(dynamic.contains("letcoverage=dyn_dissolve_coverage(dissolve,surface_cell);"));
    assert!(dynamic.contains("letrim=DISSOLVE_EDGE_GAIN*coverage*(1.0-coverage);"));
    assert!(dynamic.contains("s.emissive=s.emissive*(1.0+rim);"));
}

#[test]
fn evaluated_material_keeps_geometry_opacity_separate_from_sampled_alpha() {
    let resolve = code(RESOLVE);
    // The immutable parameter block is decoded from the live material heap,
    // not a default or an opacity reconstructed from evaluated radiance.
    assert!(resolve.contains("words[i]=material_parameters[word_offset+i];"));
    assert!(resolve.contains("returnprism_unpack_surface(words,lobe_mask);"));
    assert!(resolve.contains(
        "letparameters=load_surface_parameters(material_header.parameter_offset,material_header.lobe_mask);"
    ));
    let unpack = code(include_str!(
        "../../../../prism_render_material/src/shaders/material_unpack.wesl"
    ));
    let (_, unpack) = unpack.split_once("fnprism_unpack_surface(").unwrap();
    assert!(unpack.contains("s.geometry_opacity=bitcast<f32>(w[18]);"));

    // Texture alpha is evaluated, but it is a different quantity from the
    // material's geometry opacity. Do not replace response alpha with it.
    let sampling = code(include_str!(
        "../../../../prism_render_material/src/shaders/material_sample.wesl"
    ));
    assert!(sampling.contains("out.base_alpha=parameters.base_color.w;"));
    assert!(sampling.contains("out.base_alpha=out.base_alpha*texel.w;"));
    assert!(!sampling.contains("geometry_opacity"));

    // CPU SurfaceSample *does* carry opacity, unlike the GPU BSDF struct.
    // Its evaluated value is still a passthrough: dynamic modulation never
    // writes it, so reading the GPU parameter block preserves that semantics.
    let cpu_resolve = code(include_str!(
        "../../../../prism_render_shading/src/resolve.rs"
    ));
    assert!(cpu_resolve.contains("geometry_opacity:parameters.geometry_opacity,"));
    let cpu_dynamic = code(include_str!(
        "../../../../prism_render_shading/src/dynamic.rs"
    ));
    let (_, modulation) = cpu_dynamic
        .split_once("pubfnapply_dynamic_channels(")
        .unwrap();
    let (modulation, _) = modulation.split_once("pubfndissolve_coverage(").unwrap();
    assert!(!modulation.contains("geometry_opacity"));
}

#[test]
fn response_emission_uses_the_dynamically_evaluated_surface() {
    let resolve = code(RESOLVE);
    let dynamic = resolve.find("surface=apply_dynamic_channels(").unwrap();
    let emission = resolve.find("letemissive=surface.emissive;").unwrap();
    let response = resolve.find("letleft_arm=evaluate_response_arm(").unwrap();
    assert!(dynamic < emission && emission < response);
    let (_, call) = resolve[response..]
        .split_once("evaluate_response_arm(")
        .unwrap();
    assert!(call.starts_with(
        "response_kind,response_program_id,surface,lit,frame,geometry,parameters,material_header,indirect_result,indirect,emissive,"
    ));
}

#[test]
fn raster_coverage_combines_base_alpha_opacity_and_the_shared_dissolve_mask() {
    let opaque = code(include_str!("../../shaders/opaque.wesl"));
    assert!(opaque.contains("letdissolve=bitcast<f32>(material_header.dynamic_dissolve);"));
    assert!(opaque.contains("letcoverage=dyn_dissolve_coverage(dissolve,in.uv);"));
    assert!(opaque.contains("surface.base_color.a*surface.geometry_opacity*coverage,"));

    let transparent = code(include_str!("../../shaders/transparent.wesl"));
    assert!(transparent.contains("letdissolve=bitcast<f32>(material_header.dynamic_dissolve);"));
    assert!(transparent.contains(
        "letalpha=clamp(surface.base_color.a*surface.geometry_opacity*dyn_dissolve_coverage(dissolve,in.uv),0.0,1.0,);"
    ));
    assert!(transparent.contains("oit_accumulate(color,alpha,in.view_depth)"));
}

#[test]
fn material_parameter_heap_guards_cannot_wrap_u32_ranges() {
    let range_guard = "letheap_words=arrayLength(&material_parameters);ifword_offset>heap_words||n>heap_words-word_offset{";
    let resolve_guard = "letparameter_heap_words=arrayLength(&material_parameters);ifmaterial_header.parameter_offset>parameter_heap_words||parameter_words>parameter_heap_words-material_header.parameter_offset{";
    for (name, source) in [
        ("opaque", include_str!("../../shaders/opaque.wesl")),
        (
            "transparent",
            include_str!("../../shaders/transparent.wesl"),
        ),
        ("shading_resolve", RESOLVE),
        ("ssr_repack", include_str!("../../shaders/ssr_repack.wesl")),
    ] {
        let source = code(source);
        assert!(
            source.contains(range_guard),
            "{name} parameter loader must use subtraction-based bounds checks"
        );
        if name == "shading_resolve" || name == "ssr_repack" {
            assert!(
                source.contains(resolve_guard),
                "{name} dispatch must validate the complete parameter range"
            );
        }
    }
}

#[test]
fn material_texture_row_scans_cannot_wrap_u32_ranges() {
    let resolve = code(RESOLVE);
    assert!(!resolve.contains("lettex_end=tex_offset+tex_count;"));
    assert!(resolve.contains("i<tex_physical&&i-tex_offset<tex_count;"));
    assert!(!resolve
        .contains("lettex_end=material_header.texture_offset+material_header.texture_count;"));
    assert!(resolve.contains(
        "i<tex_physical&&i-material_header.texture_offset<material_header.texture_count;"
    ));

    let ssr = code(include_str!("../../shaders/ssr_repack.wesl"));
    assert!(
        !ssr.contains("lettex_end=material_header.texture_offset+material_header.texture_count;")
    );
    assert!(ssr.contains(
        "i<tex_physical&&i-material_header.texture_offset<material_header.texture_count;"
    ));
}
