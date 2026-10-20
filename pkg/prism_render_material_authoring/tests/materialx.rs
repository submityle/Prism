#![allow(missing_docs, reason = "integration tests are self-describing")]

use prism_render_material_authoring::{
    adapt_openpbr_surface, export_materialx, import_openpbr_surface, parse_materialx, Connection,
    DiagnosticCode, Severity,
};

const SOURCE: &str = r#"<?xml version="1.0"?>
<materialx version="1.38">
  <nodedef name="ND_open_pbr_surface" node="open_pbr_surface">
    <input name="base_color" type="color3" value="0.8, 0.8, 0.8"/>
    <input name="specular_ior" type="float" value="1.5"/>
    <output name="out" type="surfaceshader"/>
  </nodedef>
  <nodegraph name="Graph">
    <constant name="base" type="color3" value="0.2 0.4 0.8"/>
    <open_pbr_surface name="surface">
      <input name="base_color" type="color3" nodename="base"/>
      <input name="specular_ior" type="float" value="1.45"/>
      <input name="coat_weight" type="float" value="0.3"/>
    </open_pbr_surface>
    <output name="out" type="surfaceshader" nodename="surface"/>
  </nodegraph>
</materialx>
"#;

#[test]
fn parses_structure_and_imports_open_pbr_literals_and_constant_connections() {
    let document = parse_materialx(SOURCE).expect("supported MaterialX subset");
    assert_eq!(document.nodedefs[0].category, "open_pbr_surface");
    assert_eq!(document.nodegraphs[0].nodes.len(), 2);

    let surface = import_openpbr_surface(&document, "Graph", "surface").expect("surface import");
    assert_eq!(surface.base_color, [0.2, 0.4, 0.8, 1.0]);
    assert_eq!(surface.specular_ior, 1.45);
    assert_eq!(surface.coat_weight, 0.3);
    assert!(surface.unsupported_fields.is_empty());
}

#[test]
fn export_is_deterministic_and_structurally_round_trips() {
    let document = parse_materialx(SOURCE).expect("parse");
    let first = export_materialx(&document);
    let second = export_materialx(&parse_materialx(&first).expect("exported XML parses"));
    assert_eq!(first, second);
    assert!(first.contains("<open_pbr_surface"));
    assert!(first.contains("value=\"0.2, 0.4, 0.8\""));
}

#[test]
fn duplicate_nodedef_ports_are_rejected_instead_of_overwriting_schema() {
    let duplicate_input = parse_materialx(
        r#"<materialx><nodedef name="ND_surface" node="open_pbr_surface">
          <input name="base_color" type="color3"/><input name="base_color" type="color3"/>
        </nodedef></materialx>"#,
    )
    .expect_err("duplicate input declarations are ambiguous");
    assert_eq!(duplicate_input.0.code, DiagnosticCode::InvalidReference);
    assert!(duplicate_input.to_string().contains("base_color"));

    let duplicate_output = parse_materialx(
        r#"<materialx><nodedef name="ND_surface" node="open_pbr_surface">
          <output name="out" type="surfaceshader"/><output name="out" type="surfaceshader"/>
        </nodedef></materialx>"#,
    )
    .expect_err("duplicate output declarations are ambiguous");
    assert_eq!(duplicate_output.0.code, DiagnosticCode::InvalidReference);
    assert!(duplicate_output.to_string().contains("out"));
}

#[test]
fn author_node_instances_fail_closed_on_invalid_nodedef_contracts() {
    let cases = [
        (
            r#"<materialx><nodegraph name="g"><add name="n" nodedef="ND_missing"/></nodegraph></materialx>"#,
            DiagnosticCode::InvalidReference,
            "unknown nodedef",
        ),
        (
            r#"<materialx><nodedef name="ND_noise" node="noise"><input name="in" type="float" value="0"/><output name="out" type="float"/></nodedef><nodegraph name="g"><noise name="n" nodedef="ND_noise"/></nodegraph></materialx>"#,
            DiagnosticCode::UnsupportedMaterial,
            "no executable PCM author_node",
        ),
        (
            r#"<materialx><nodedef name="ND_add" node="add"><input name="a" type="float" value="1"/><input name="b" type="float" value="2"/><output name="out" type="float"/></nodedef><nodegraph name="g"><multiply name="n" nodedef="ND_add"/></nodegraph></materialx>"#,
            DiagnosticCode::InvalidStructure,
            "does not match",
        ),
        (
            r#"<materialx><nodedef name="ND_add" node="add"><input name="a" type="float"/><input name="b" type="float" value="2"/><output name="out" type="float"/></nodedef><nodegraph name="g"><add name="n" nodedef="ND_add"/></nodegraph></materialx>"#,
            DiagnosticCode::InvalidStructure,
            "omits required",
        ),
        (
            r#"<materialx><nodedef name="ND_add" node="add"><input name="a" type="float" value="1"/><input name="b" type="float" value="2"/><output name="out" type="float"/></nodedef><nodegraph name="g"><add name="n" nodedef="ND_add"><input name="extra" type="float" value="3"/></add></nodegraph></materialx>"#,
            DiagnosticCode::InvalidStructure,
            "is not declared",
        ),
        (
            r#"<materialx><nodedef name="ND_add" node="add"><input name="a" type="float" value="1"/><input name="b" type="float" value="2"/><output name="out" type="float"/></nodedef><nodegraph name="g"><add name="n" nodedef="ND_add"><input name="a" type="color3" value="1, 1, 1"/></add></nodegraph></materialx>"#,
            DiagnosticCode::InvalidValue,
            "expected 'float'",
        ),
        (
            r#"<materialx><nodedef name="ND_add" node="add"><input name="a" type="float" value="1"/><input name="b" type="float" value="2"/><output name="out" type="float"/></nodedef><nodegraph name="g"><add name="n" nodedef="ND_add"><output name="other" type="float"/></add></nodegraph></materialx>"#,
            DiagnosticCode::InvalidReference,
            "is not declared",
        ),
        (
            r#"<materialx><nodedef name="ND_base" node="add"><input name="a" type="float" value="1"/><input name="b" type="float" value="2"/><output name="out" type="float"/></nodedef><nodedef name="ND_child" node="add" inherit="ND_base"><input name="a" type="float" value="1"/><input name="b" type="float" value="2"/><output name="out" type="float"/></nodedef></materialx>"#,
            DiagnosticCode::UnsupportedMaterial,
            "inherited author-node semantics are unsupported",
        ),
        (
            r#"<materialx><nodedef name="ND_expand" node="convert"><input name="in" type="float" value="1"/><output name="out" type="color3"/></nodedef><nodegraph name="g"><convert name="n" nodedef="ND_expand"/></nodegraph></materialx>"#,
            DiagnosticCode::InvalidValue,
            "same dimension",
        ),
        (
            r#"<materialx><nodedef name="ND_boolean" node="convert"><input name="in" type="boolean" value="true"/><output name="out" type="boolean"/></nodedef><nodegraph name="g"><convert name="n" nodedef="ND_boolean"/></nodegraph></materialx>"#,
            DiagnosticCode::InvalidValue,
            "supported numeric",
        ),
    ];

    for (source, code, message) in cases {
        let error = parse_materialx(source).expect_err(message);
        assert_eq!(error.0.code, code, "{source}");
        assert!(error.to_string().contains(message), "{error}");
    }
}

#[test]
fn duplicate_author_node_instance_ports_are_rejected() {
    let source = r#"<materialx>
      <nodedef name="ND_add" node="add">
        <input name="a" type="float" value="1"/><input name="b" type="float" value="2"/>
        <output name="out" type="float"/>
      </nodedef>
      <nodegraph name="g"><add name="n" nodedef="ND_add">
        <input name="a" type="float" value="3"/><input name="a" type="float" value="4"/>
      </add></nodegraph>
    </materialx>"#;
    let error = parse_materialx(source).expect_err("duplicate instance input is ambiguous");
    assert_eq!(error.0.code, DiagnosticCode::InvalidReference);
    assert!(error
        .to_string()
        .contains("duplicate document identifier 'a'"));
}

#[test]
fn unknown_node_is_rejected_with_location_and_machine_code() {
    let error = parse_materialx(
        r#"<materialx><nodegraph name="g"><mystery name="x"/></nodegraph></materialx>"#,
    )
    .expect_err("unknown semantics must not be dropped");
    assert_eq!(error.0.code, DiagnosticCode::UnsupportedMaterial);
    assert!(error.0.location.is_some());
    assert!(error.to_string().contains("mystery"));
}

#[test]
fn malformed_connection_is_diagnostic() {
    let error = parse_materialx(
        r#"<materialx><nodegraph name="g"><constant name="c" type="float" value="1"/><standard_surface name="s"><input name="base" nodename="missing"/></standard_surface></nodegraph></materialx>"#,
    )
    .expect_err("dangling node reference must be rejected");
    assert_eq!(error.0.code, DiagnosticCode::InvalidReference);
    assert!(error.to_string().contains("missing"));
}

#[test]
fn unsupported_surface_input_is_reported_without_being_silently_applied() {
    let document = parse_materialx(
        r#"<materialx><nodegraph name="g"><standard_surface name="s"><input name="thinfilm_thickness" type="float" value="200"/></standard_surface></nodegraph></materialx>"#,
    )
    .expect("structural parser accepts a supported surface node");
    let surface = import_openpbr_surface(&document, "g", "s").expect("surface import");
    assert_eq!(surface.unsupported_fields, ["thinfilm_thickness"]);
}

#[test]
fn cross_graph_node_connection_parses_imports_and_round_trips() {
    let source = r#"<materialx>
  <nodegraph name="SurfaceGraph">
    <open_pbr_surface name="surface">
      <input name="base_color" type="color3" nodegraph="Constants" nodename="base"/>
    </open_pbr_surface>
    <output name="out" type="surfaceshader" nodename="surface"/>
  </nodegraph>
  <nodegraph name="Constants">
    <constant name="base" type="color3" value="0.1, 0.3, 0.7"/>
    <output name="color" type="color3" nodename="base"/>
  </nodegraph>
</materialx>"#;

    let document = parse_materialx(source).expect("cross-graph connection is valid");
    let surface = import_openpbr_surface(&document, "SurfaceGraph", "surface")
        .expect("constant connected from another graph is importable");
    assert_eq!(surface.base_color, [0.1, 0.3, 0.7, 1.0]);

    let first_export = export_materialx(&document);
    assert!(first_export.contains("nodegraph=\"Constants\" nodename=\"base\""));
    let reparsed = parse_materialx(&first_export).expect("export retains cross-graph semantics");
    let second_export = export_materialx(&reparsed);
    assert_eq!(first_export, second_export);
    assert_eq!(
        import_openpbr_surface(&reparsed, "SurfaceGraph", "surface")
            .expect("reimport")
            .base_color,
        [0.1, 0.3, 0.7, 1.0]
    );
}

#[test]
fn node_names_are_scoped_to_their_nodegraph() {
    let source = r#"<materialx>
  <nodegraph name="First">
    <constant name="shared" type="color3" value="0.1 0.2 0.3"/>
  </nodegraph>
  <nodegraph name="Second">
    <constant name="shared" type="color3" value="0.7 0.8 0.9"/>
  </nodegraph>
</materialx>"#;

    let document = parse_materialx(source).expect("same node name is legal in separate graphs");
    assert_eq!(document.nodegraphs[0].nodes[0].name, "shared");
    assert_eq!(document.nodegraphs[1].nodes[0].name, "shared");

    let first_export = export_materialx(&document);
    let second_export = export_materialx(
        &parse_materialx(&first_export).expect("scoped identifiers survive export"),
    );
    assert_eq!(first_export, second_export);
}

#[test]
fn adapter_maps_openpbr_fields_into_the_prism_runtime_block() {
    let document = parse_materialx(
        r#"<materialx><nodegraph name="g"><open_pbr_surface name="s">
          <input name="base_color" type="color3" value="0.1, 0.2, 0.3"/>
          <input name="base_weight" type="float" value="0.75"/>
          <input name="specular_weight" type="float" value="0.8"/>
          <input name="specular_ior" type="float" value="1.7"/>
          <input name="specular_roughness_anisotropy" type="float" value="0.35"/>
          <input name="emission_color" type="color3" value="0.4, 0.5, 0.6"/>
          <input name="emission_luminance" type="float" value="12"/>
          <input name="coat_color" type="color3" value="0.7, 0.8, 0.9"/>
          <input name="coat_roughness_anisotropy" type="float" value="0.2"/>
          <input name="subsurface_radius_scale" type="color3" value="1, 0.5, 0.25"/>
          <input name="transmission_color" type="color3" value="0.9, 0.8, 0.7"/>
          <input name="thin_film_thickness" type="float" value="0.35"/>
          <input name="transmission_depth" type="float" value="2.5"/>
          <input name="geometry_opacity" type="float" value="0.6"/>
          <input name="geometry_thin_walled" type="boolean" value="true"/>
        </open_pbr_surface></nodegraph></materialx>"#,
    )
    .expect("valid supported subset");

    let adapted = adapt_openpbr_surface(&document, "g", "s").expect("adapter succeeds");
    let p = adapted.parameters;
    assert_eq!(p.base_color, [0.1, 0.2, 0.3, 1.0]);
    assert_eq!(p.base_weight, 0.75);
    assert_eq!(p.specular_weight, 0.8);
    assert_eq!(p.specular_ior, 1.7);
    assert_eq!(p.anisotropy, 0.35);
    assert_eq!(p.emission_color, [0.4, 0.5, 0.6, 1.0]);
    assert_eq!(p.emission_luminance, 12.0);
    assert_eq!(p.coat_color, [0.7, 0.8, 0.9, 1.0]);
    assert_eq!(p.coat_roughness_anisotropy, 0.2);
    assert_eq!(p.subsurface_radius_scale, [1.0, 0.5, 0.25, 1.0]);
    assert_eq!(p.transmission_color, [0.9, 0.8, 0.7, 1.0]);
    assert_eq!(p.thin_film_thickness, 0.35);
    assert_eq!(p.thickness, 2.5);
    assert_eq!(p.geometry_opacity, 0.6);
    assert_eq!(p.thin_walled, 1.0);
    assert!(adapted.report.unsupported_fields.is_empty());
    assert!(adapted.report.diagnostics.is_empty());
    assert!(adapted
        .report
        .supported_fields
        .contains(&"geometry_thin_walled".into()));
}

#[test]
fn adapter_covers_every_direct_openpbr_runtime_field_mapping() {
    use prism_render_material::GpuSurfaceParameters;

    let scalar_fields = [
        "base_metalness",
        "base_weight",
        "specular_roughness",
        "specular_weight",
        "ambient_occlusion",
        "normal_scale",
        "alpha_cutoff",
        "base_diffuse_roughness",
        "specular_ior",
        "emission_luminance",
        "coat_weight",
        "coat_roughness",
        "coat_ior",
        "coat_roughness_anisotropy",
        "coat_darkening",
        "anisotropy_rotation",
        "fuzz_weight",
        "fuzz_roughness",
        "subsurface_weight",
        "subsurface_radius",
        "subsurface_scatter_anisotropy",
        "index_of_refraction",
        "transmission_dispersion_scale",
        "transmission_scatter_anisotropy",
        "transmission_dispersion_abbe_number",
        "thin_film_weight",
        "thin_film_thickness",
        "thin_film_ior",
        "transmission_weight",
        "thickness",
        "geometry_opacity",
        "specular_roughness_anisotropy",
    ];
    let color_fields = [
        "base_color",
        "emission_color",
        "specular_color",
        "coat_color",
        "fuzz_color",
        "subsurface_color",
        "subsurface_radius_scale",
        "transmission_color",
        "transmission_scatter",
    ];
    let mut inputs = scalar_fields
        .iter()
        .map(|name| format!(r#"<input name="{name}" type="float" value="0.25"/>"#))
        .collect::<Vec<_>>();
    inputs.extend(
        color_fields
            .iter()
            .map(|name| format!(r#"<input name="{name}" type="color3" value="0.1, 0.2, 0.3"/>"#)),
    );
    inputs.push(r#"<input name="geometry_thin_walled" type="boolean" value="true"/>"#.into());
    let xml = format!(
        "<materialx><nodegraph name=\"g\"><open_pbr_surface name=\"s\">{}</open_pbr_surface></nodegraph></materialx>",
        inputs.join("\n")
    );
    let document = parse_materialx(&xml).expect("all declared OpenPBR inputs parse");
    let adapted = adapt_openpbr_surface(&document, "g", "s").expect("all mappings adapt");

    assert!(adapted.report.unsupported_fields.is_empty());
    assert_eq!(adapted.report.diagnostics, []);
    assert_eq!(
        adapted.report.applied_fields.len(),
        scalar_fields.len() + color_fields.len() + 1
    );
    assert_eq!(
        adapted.report.supported_fields,
        adapted.report.applied_fields
    );
    assert_eq!(
        adapted.parameters,
        GpuSurfaceParameters {
            base_color: [0.1, 0.2, 0.3, 1.0],
            emission_color: [0.1, 0.2, 0.3, 1.0],
            emission_luminance: 0.25,
            base_metalness: 0.25,
            specular_roughness: 0.25,
            specular_weight: 0.25,
            ambient_occlusion: 0.25,
            normal_scale: 0.25,
            alpha_cutoff: 0.25,
            base_weight: 0.25,
            base_diffuse_roughness: 0.25,
            specular_color: [0.1, 0.2, 0.3, 1.0],
            specular_ior: 0.25,
            thin_walled: 1.0,
            geometry_opacity: 0.25,
            transmission_weight: 0.25,
            thickness: 0.25,
            coat_weight: 0.25,
            coat_roughness: 0.25,
            coat_ior: 0.25,
            coat_color: [0.1, 0.2, 0.3, 1.0],
            coat_roughness_anisotropy: 0.25,
            coat_darkening: 0.25,
            anisotropy: 0.25,
            anisotropy_rotation: 0.25,
            fuzz_weight: 0.25,
            fuzz_roughness: 0.25,
            fuzz_color: [0.1, 0.2, 0.3, 1.0],
            subsurface_weight: 0.25,
            subsurface_color: [0.1, 0.2, 0.3, 1.0],
            subsurface_radius: 0.25,
            subsurface_radius_scale: [0.1, 0.2, 0.3, 1.0],
            subsurface_scatter_anisotropy: 0.25,
            index_of_refraction: 0.25,
            transmission_dispersion_scale: 0.25,
            transmission_color: [0.1, 0.2, 0.3, 1.0],
            transmission_scatter: [0.1, 0.2, 0.3, 1.0],
            thin_film_weight: 0.25,
            thin_film_thickness: 0.25,
            thin_film_ior: 0.25,
            transmission_scatter_anisotropy: 0.25,
            transmission_dispersion_abbe_number: 0.25,
            ..GpuSurfaceParameters::default()
        }
    );
}

#[test]
fn unsupported_fields_are_reported_as_loss_without_mutating_runtime_defaults() {
    let document = parse_materialx(
        r#"<materialx><nodegraph name="g"><open_pbr_surface name="s">
          <input name="base_color" type="color3" value="0.2, 0.3, 0.4"/>
          <input name="coat_normal" type="vector3" value="0, 0, 1"/>
          <input name="artist_custom_parameter" type="float" value="42"/>
        </open_pbr_surface></nodegraph></materialx>"#,
    )
    .expect("the subset parser preserves supported surface-node inputs");

    let adapted = adapt_openpbr_surface(&document, "g", "s")
        .expect("unsupported fields are recoverable losses");
    assert_eq!(adapted.parameters.base_color, [0.2, 0.3, 0.4, 1.0]);
    assert_eq!(adapted.parameters.coat_weight, 0.0);
    assert_eq!(
        adapted.report.unsupported_fields,
        ["artist_custom_parameter", "coat_normal"]
    );
    assert_eq!(adapted.report.diagnostics.len(), 2);
    assert!(adapted.report.diagnostics.iter().all(|d| {
        d.code == DiagnosticCode::UnsupportedField && d.severity == Severity::Warning
    }));
}

#[test]
fn pure_expression_connections_are_evaluated_into_runtime_parameters() {
    let document = parse_materialx(
        r#"<materialx><nodegraph name="g">
          <constant name="a" type="float" value="0.2"/>
          <constant name="b" type="float" value="0.8"/>
          <multiply name="mix" type="float"><input name="in1" nodename="a"/><input name="in2" nodename="b"/></multiply>
          <open_pbr_surface name="s"><input name="specular_roughness" type="float" nodename="mix"/></open_pbr_surface>
        </nodegraph></materialx>"#,
    )
    .expect("structurally valid graph");
    let adapted = adapt_openpbr_surface(&document, "g", "s")
        .expect("bounded pure expression is part of the authoring subset");
    assert!((adapted.parameters.specular_roughness - 0.16).abs() < 1e-6);
    assert_eq!(adapted.report.applied_fields, ["specular_roughness"]);
}

#[test]
fn unsupported_expression_nodes_still_require_a_bake() {
    let document = parse_materialx(
        r#"<materialx><nodegraph name="g">
          <image name="texture" type="color3"/>
          <open_pbr_surface name="s"><input name="base_color" type="color3" nodename="texture"/></open_pbr_surface>
        </nodegraph></materialx>"#,
    )
    .expect("image nodes are structurally preserved");
    let error = adapt_openpbr_surface(&document, "g", "s")
        .expect_err("texture evaluation must not become a default or fake constant");
    assert_eq!(error.0.code, DiagnosticCode::UnsafeConnection);
    assert!(error.to_string().contains("requires a bake/runtime"));
}

#[test]
fn interface_connections_resolve_and_non_default_outputs_remain_explicitly_unsafe() {
    let interface_document = parse_materialx(
        r#"<materialx><nodegraph name="g"><input name="base" type="color3" value="0.1, 0.2, 0.3"/><open_pbr_surface name="s"><input name="base_color" type="color3" interfacename="base"/></open_pbr_surface></nodegraph></materialx>"#,
    )
    .expect("interface declaration is structurally valid");
    let interface_surface = adapt_openpbr_surface(&interface_document, "g", "s")
        .expect("graph interface input resolves to its literal");
    assert_eq!(
        interface_surface.parameters.base_color,
        [0.1, 0.2, 0.3, 1.0]
    );

    let round_trip = export_materialx(&interface_document);
    let reparsed = parse_materialx(&round_trip).expect("interface survives export");
    assert_eq!(
        adapt_openpbr_surface(&reparsed, "g", "s")
            .expect("reimported interface")
            .parameters
            .base_color,
        [0.1, 0.2, 0.3, 1.0]
    );

    let output_document = parse_materialx(
        r#"<materialx><nodegraph name="g"><constant name="c" type="float" value="0.2"/><open_pbr_surface name="s"><input name="specular_roughness" type="float" nodename="c" output="alternate"/></open_pbr_surface></nodegraph></materialx>"#,
    )
    .expect("constant connection is structurally valid");
    let output_error = adapt_openpbr_surface(&output_document, "g", "s")
        .expect_err("non-default output selection is not guessed");
    assert_eq!(output_error.0.code, DiagnosticCode::UnsafeConnection);

    let operator_output_document = parse_materialx(
        r#"<materialx><nodegraph name="g"><constant name="a" type="float" value="0.1"/><constant name="b" type="float" value="0.2"/><add name="sum"><input name="in1" nodename="a"/><input name="in2" nodename="b"/><output name="alternate" type="float"/></add><open_pbr_surface name="s"><input name="specular_roughness" type="float" nodename="sum" output="alternate"/></open_pbr_surface></nodegraph></materialx>"#,
    )
    .expect("operator output declaration is structurally valid");
    let operator_output_error = adapt_openpbr_surface(&operator_output_document, "g", "s")
        .expect_err("operator outputs without an evaluator must not alias the default output");
    assert_eq!(
        operator_output_error.0.code,
        DiagnosticCode::UnsafeConnection
    );
}

#[test]
fn graph_outputs_resolve_and_type_mismatched_connections_are_not_guessed() {
    let graph_output_document = parse_materialx(
        r#"<materialx>
          <nodegraph name="values"><constant name="c" type="float" value="0.2"/><output name="out" type="float" nodename="c"/></nodegraph>
          <nodegraph name="surface"><open_pbr_surface name="s"><input name="specular_roughness" type="float" nodegraph="values"/></open_pbr_surface></nodegraph>
        </materialx>"#,
    )
    .expect("graph output reference is structurally valid");
    let graph_output_surface = adapt_openpbr_surface(&graph_output_document, "surface", "s")
        .expect("graph output resolves through its connected constant");
    assert_eq!(graph_output_surface.parameters.specular_roughness, 0.2);

    let mismatched_document = parse_materialx(
        r#"<materialx><nodegraph name="g"><constant name="c" type="color3" value="0.2, 0.3, 0.4"/><open_pbr_surface name="s"><input name="specular_roughness" type="float" nodename="c"/></open_pbr_surface></nodegraph></materialx>"#,
    )
    .expect("connection structure parses");
    let mismatch = adapt_openpbr_surface(&mismatched_document, "g", "s")
        .expect_err("source and input types must agree");
    assert_eq!(mismatch.0.code, DiagnosticCode::InvalidValue);
}

#[test]
fn interface_and_graph_output_error_paths_remain_diagnostic() {
    let missing_interface = parse_materialx(
        r#"<materialx><nodegraph name="g"><open_pbr_surface name="s"><input name="base_color" type="color3" interfacename="missing"/></open_pbr_surface></nodegraph></materialx>"#,
    )
    .expect_err("missing graph interface must not become a default");
    assert_eq!(missing_interface.0.code, DiagnosticCode::InvalidReference);

    let unconnected_output = parse_materialx(
        r#"<materialx><nodegraph name="values"><output name="out" type="float"/></nodegraph><nodegraph name="surface"><open_pbr_surface name="s"><input name="specular_roughness" type="float" nodegraph="values" output="out"/></open_pbr_surface></nodegraph></materialx>"#,
    )
    .expect("unconnected graph output is structurally representable");
    let unconnected_error = adapt_openpbr_surface(&unconnected_output, "surface", "s")
        .expect_err("unconnected graph output must not become a default");
    assert_eq!(unconnected_error.0.code, DiagnosticCode::UnsafeConnection);

    let ambiguous_output = parse_materialx(
        r#"<materialx><nodegraph name="values"><constant name="a" type="float" value="0.2"/><constant name="b" type="float" value="0.4"/><output name="a_out" type="float" nodename="a"/><output name="b_out" type="float" nodename="b"/></nodegraph><nodegraph name="surface"><open_pbr_surface name="s"><input name="specular_roughness" type="float" nodegraph="values"/></open_pbr_surface></nodegraph></materialx>"#,
    )
    .expect("implicit graph output connection is structurally representable");
    let ambiguous_error = adapt_openpbr_surface(&ambiguous_output, "surface", "s")
        .expect_err("multiple graph outputs require an explicit output name");
    assert_eq!(ambiguous_error.0.code, DiagnosticCode::InvalidReference);
}

#[test]
fn omitted_supported_fields_keep_runtime_defaults_and_alias_collisions_fail() {
    let omitted = parse_materialx(
        r#"<materialx><nodegraph name="g"><open_pbr_surface name="s"><input name="coat_ior" type="float"/></open_pbr_surface></nodegraph></materialx>"#,
    )
    .expect("omitted value is valid");
    let adapted = adapt_openpbr_surface(&omitted, "g", "s").expect("default is retained");
    assert_eq!(adapted.parameters.coat_ior, 1.6);
    assert_eq!(adapted.report.supported_fields, ["coat_ior"]);
    assert!(adapted.report.applied_fields.is_empty());

    let duplicate = parse_materialx(
        r#"<materialx><nodegraph name="g"><standard_surface name="s"><input name="base" type="float" value="0.5"/><input name="base_weight" type="float" value="0.8"/></standard_surface></nodegraph></materialx>"#,
    )
    .expect("both aliases are structurally valid");
    let error = adapt_openpbr_surface(&duplicate, "g", "s")
        .expect_err("conflicting source inputs must not silently overwrite each other");
    assert_eq!(error.0.code, DiagnosticCode::InvalidStructure);
}

#[test]
fn malformed_supported_literals_fail_instead_of_falling_back_to_defaults() {
    for raw in ["0.1, 0.2", "NaN", "inf"] {
        let xml = format!(
            "<materialx><nodegraph name=\"g\"><open_pbr_surface name=\"s\"><input name=\"base_color\" type=\"color3\" value=\"{raw}\"/></open_pbr_surface></nodegraph></materialx>"
        );
        let document = parse_materialx(&xml).expect("literal is structurally valid");
        let error = adapt_openpbr_surface(&document, "g", "s").expect_err("invalid color rejected");
        assert_eq!(error.0.code, DiagnosticCode::InvalidValue);
    }

    let document = parse_materialx(
        r#"<materialx><nodegraph name="g"><open_pbr_surface name="s"><input name="specular_ior" type="float" value="1, 2"/></open_pbr_surface></nodegraph></materialx>"#,
    )
    .expect("literal is structurally valid");
    assert_eq!(
        adapt_openpbr_surface(&document, "g", "s")
            .unwrap_err()
            .0
            .code,
        DiagnosticCode::InvalidValue
    );
}

#[test]
fn standard_surface_aliases_map_to_runtime_fields_without_claiming_color_opacity() {
    let document = parse_materialx(
        r#"<materialx><nodegraph name="g"><standard_surface name="s">
          <input name="base" type="float" value="0.65"/>
          <input name="metalness" type="float" value="0.4"/>
          <input name="specular_IOR" type="float" value="1.4"/>
          <input name="transmission" type="float" value="0.25"/>
          <input name="opacity" type="color3" value="0.5, 0.6, 0.7"/>
          <input name="emission_color" type="color3" value="0.9, 0.8, 0.7"/>
          <input name="emission" type="float" value="0.3"/>
        </standard_surface></nodegraph></materialx>"#,
    )
    .expect("valid Standard Surface structure");
    let adapted = adapt_openpbr_surface(&document, "g", "s").expect("supported aliases map");
    assert_eq!(adapted.parameters.base_weight, 0.65);
    assert_eq!(adapted.parameters.base_metalness, 0.4);
    assert_eq!(adapted.parameters.specular_ior, 1.4);
    assert_eq!(adapted.parameters.transmission_weight, 0.25);
    assert_eq!(adapted.parameters.emission_color, [0.9, 0.8, 0.7, 1.0]);
    assert_eq!(adapted.parameters.emission_luminance, 0.0);
    assert_eq!(adapted.report.unsupported_fields, ["emission", "opacity"]);
}

#[test]
fn reusable_nodegraph_instances_resolve_defaults_overrides_and_selected_outputs() {
    let document = parse_materialx(r#"<materialx>
      <nodedef name="ND_scale" node="multiply"><input name="factor" type="float" value="2"/><input name="value" type="float" value="0.25"/><output name="out" type="float"/><output name="twice" type="float"/></nodedef>
      <nodegraph name="NG_scale" nodedef="ND_scale">
        <input name="factor" type="float" value="3"/><input name="value" type="float" value="0.2"/>
        <multiply name="product"><input name="in1" interfacename="factor"/><input name="in2" interfacename="value"/></multiply>
        <constant name="alternate" type="float" value="0.9"/>
        <output name="out" type="float" nodename="product"/><output name="twice" type="float" nodename="alternate"/>
      </nodegraph>
      <nodegraph name="surface">
        <constant name="override" type="float" value="0.4"/>
        <multiply name="instance" nodedef="ND_scale"><input name="factor" nodename="override"/></multiply>
        <open_pbr_surface name="s"><input name="specular_roughness" type="float" nodename="instance" output="out"/></open_pbr_surface>
      </nodegraph>
    </materialx>"#).expect("reusable graph definition and instance parse");
    let adapted =
        adapt_openpbr_surface(&document, "surface", "s").expect("graph instance evaluates");
    assert!((adapted.parameters.specular_roughness - 0.08).abs() < 1.0e-6);
    let surface_graph = document
        .nodegraphs
        .iter()
        .find(|graph| graph.name == "surface")
        .unwrap();
    let selected = prism_render_material_authoring::evaluate_connection(
        &document,
        surface_graph,
        &Connection {
            node: Some("instance".into()),
            output: Some("twice".into()),
            nodegraph: None,
        },
    )
    .expect("selected alternate graph output evaluates");
    assert_eq!(selected.value, "0.9");

    let round_trip =
        parse_materialx(&export_materialx(&document)).expect("graph binding round trips");
    assert_eq!(round_trip, document);
}

#[test]
fn reusable_nodegraph_instance_falls_back_to_nodedef_interface_defaults() {
    let document = parse_materialx(r#"<materialx>
      <nodedef name="ND_value" node="convert">
        <input name="in" type="float" value="0.375"/>
        <output name="out" type="float"/>
      </nodedef>
      <nodegraph name="NG_value" nodedef="ND_value">
        <input name="in" type="float"/>
        <output name="out" type="float" nodename="source"/>
        <convert name="source"><input name="in" interfacename="in"/></convert>
      </nodegraph>
      <nodegraph name="surface">
        <convert name="instance" nodedef="ND_value"/>
        <open_pbr_surface name="s"><input name="specular_roughness" type="float" nodename="instance"/></open_pbr_surface>
      </nodegraph>
    </materialx>"#).expect("nodedef and implementation interfaces agree");

    let adapted = adapt_openpbr_surface(&document, "surface", "s")
        .expect("omitted instance input resolves through nodedef default");
    assert_eq!(adapted.parameters.specular_roughness, 0.375);
}

#[test]
fn reusable_nodegraph_instance_does_not_mask_implementation_interface_connections() {
    let document = parse_materialx(r#"<materialx>
      <nodedef name="ND_value" node="convert">
        <input name="in" type="float" value="0.375"/>
        <output name="out" type="float"/>
      </nodedef>
      <nodegraph name="NG_value" nodedef="ND_value">
        <input name="in" type="float" nodename="implementation_default"/>
        <constant name="implementation_default" type="float" value="0.625"/>
        <output name="out" type="float" nodename="source"/>
        <convert name="source"><input name="in" interfacename="in"/></convert>
      </nodegraph>
      <nodegraph name="surface">
        <convert name="instance" nodedef="ND_value"/>
        <open_pbr_surface name="s"><input name="specular_roughness" type="float" nodename="instance"/></open_pbr_surface>
      </nodegraph>
    </materialx>"#).expect("connected implementation interface is valid");

    let adapted = adapt_openpbr_surface(&document, "surface", "s")
        .expect("implementation connection takes precedence over nodedef fallback");
    assert_eq!(adapted.parameters.specular_roughness, 0.625);
}

#[test]
fn reusable_nodegraph_instances_nest_and_detect_cycles() {
    let nested = parse_materialx(r#"<materialx>
      <nodedef name="ND_inner" node="add"><input name="x" type="float" value="0.1"/><output name="out" type="float"/></nodedef>
      <nodegraph name="NG_inner" nodedef="ND_inner"><input name="x" type="float" value="0.2"/><add name="sum"><input name="in1" interfacename="x"/><input name="in2" type="float" value="0.1"/></add><output name="out" type="float" nodename="sum"/></nodegraph>
      <nodedef name="ND_outer" node="add"><input name="x" type="float" value="0"/><output name="out" type="float"/></nodedef>
      <nodegraph name="NG_outer" nodedef="ND_outer"><input name="x" type="float" value="0"/><add name="inner_instance" nodedef="ND_inner"><input name="x" interfacename="x"/></add><output name="out" type="float" nodename="inner_instance"/></nodegraph>
      <nodegraph name="surface"><add name="outer" nodedef="ND_outer"/><open_pbr_surface name="s"><input name="specular_roughness" type="float" nodename="outer"/></open_pbr_surface></nodegraph>
    </materialx>"#).expect("nested graphs parse");
    let value = prism_render_material_authoring::evaluate_connection(
        &nested,
        &nested.nodegraphs[2],
        &Connection {
            node: Some("outer".into()),
            output: None,
            nodegraph: None,
        },
    )
    .expect("nested graph evaluates");
    assert_eq!(value.value, "0.1");

    let cyclic = parse_materialx(r#"<materialx>
      <nodedef name="ND_loop" node="add"><input name="x" type="float" value="1"/><output name="out" type="float"/></nodedef>
      <nodegraph name="NG_loop" nodedef="ND_loop"><input name="x" type="float" value="1"/><add name="again" nodedef="ND_loop"><input name="x" interfacename="x"/></add><output name="out" type="float" nodename="again"/></nodegraph>
      <nodegraph name="surface"><add name="loop" nodedef="ND_loop"/><open_pbr_surface name="s"><input name="specular_roughness" type="float" nodename="loop"/></open_pbr_surface></nodegraph>
    </materialx>"#).expect("cycle is a runtime graph property");
    let error =
        adapt_openpbr_surface(&cyclic, "surface", "s").expect_err("recursive graph rejected");
    assert_eq!(error.0.code, DiagnosticCode::UnsafeConnection);
    assert!(error.to_string().contains("cyclic"));
}

#[test]
fn reusable_nodegraph_instances_report_input_and_output_type_mismatches() {
    let wrong_input = parse_materialx(r#"<materialx><nodedef name="ND_value" node="constant"><input name="value" type="float" value="0"/><output name="out" type="float"/></nodedef><nodegraph name="NG_value" nodedef="ND_value"><input name="value" type="float" value="0.2"/><convert name="source"><input name="in" interfacename="value"/></convert><output name="out" type="float" nodename="source"/></nodegraph><nodegraph name="surface"><constant name="color" type="color3" value="0.1, 0.2, 0.3"/><constant name="instance" nodedef="ND_value"><input name="value" nodename="color"/></constant><open_pbr_surface name="s"><input name="specular_roughness" type="float" nodename="instance"/></open_pbr_surface></nodegraph></materialx>"#).expect("instance connection is structurally valid");
    let error = adapt_openpbr_surface(&wrong_input, "surface", "s")
        .expect_err("input type mismatch rejected");
    assert_eq!(error.0.code, DiagnosticCode::InvalidValue);

    let malformed_untyped_override = parse_materialx(r#"<materialx>
      <nodedef name="ND_value" node="convert"><input name="in" type="float" value="0"/><output name="out" type="float"/></nodedef>
      <nodegraph name="NG_value" nodedef="ND_value"><input name="in" type="float"/><convert name="source"><input name="in" interfacename="in"/></convert><output name="out" type="float" nodename="source"/></nodegraph>
      <nodegraph name="surface"><convert name="instance" nodedef="ND_value"><input name="in" value="not-a-number"/></convert><open_pbr_surface name="s"><input name="specular_roughness" type="float" nodename="instance"/></open_pbr_surface></nodegraph>
    </materialx>"#).expect("untyped override is structurally valid");
    let error = adapt_openpbr_surface(&malformed_untyped_override, "surface", "s")
        .expect_err("untyped numeric override must be parsed before type coercion");
    assert_eq!(error.0.code, DiagnosticCode::InvalidValue);
    assert!(error.to_string().contains("invalid numeric literal"));

    let missing_output = parse_materialx(r#"<materialx><nodedef name="ND_multi" node="add"><input name="x" type="float" value="0"/><output name="a" type="float"/><output name="b" type="float"/></nodedef><nodegraph name="NG_multi" nodedef="ND_multi"><input name="x" type="float" value="0"/><add name="source"><input name="in1" interfacename="x"/><input name="in2" interfacename="x"/></add><output name="a" type="float" nodename="source"/><output name="b" type="float" nodename="source"/></nodegraph><nodegraph name="surface"><add name="instance" nodedef="ND_multi"/><open_pbr_surface name="s"><input name="specular_roughness" type="float" nodename="instance"/></open_pbr_surface></nodegraph></materialx>"#).expect("ambiguous graph output is structurally valid");
    let error = adapt_openpbr_surface(&missing_output, "surface", "s")
        .expect_err("ambiguous output rejected");
    assert_eq!(error.0.code, DiagnosticCode::InvalidReference);

    let wrong_output = parse_materialx(r#"<materialx><nodedef name="ND_wrong_output" node="constant"><input name="x" type="float" value="0"/><output name="out" type="float"/></nodedef><nodegraph name="NG_wrong_output" nodedef="ND_wrong_output"><input name="x" type="float" value="0"/><constant name="color" type="color3" value="0.1, 0.2, 0.3"/><output name="out" type="float" nodename="color"/></nodegraph><nodegraph name="surface"><constant name="instance" nodedef="ND_wrong_output"/><open_pbr_surface name="s"><input name="specular_roughness" type="float" nodename="instance"/></open_pbr_surface></nodegraph></materialx>"#).expect("output type is checked during evaluation");
    let error = adapt_openpbr_surface(&wrong_output, "surface", "s")
        .expect_err("implementation output type mismatch rejected");
    assert_eq!(error.0.code, DiagnosticCode::InvalidValue);
}

#[test]
fn reusable_nodegraph_instances_reject_invalid_instance_contracts_at_parse_time() {
    let cases = [
        (
            r#"<materialx><nodedef name="ND_value" node="constant"><input name="value" type="float" value="0"/><output name="out" type="float"/></nodedef><nodegraph name="NG_value" nodedef="ND_value"><input name="value" type="float"/><constant name="source" type="float" value="0"/><output name="other" type="float" nodename="source"/></nodegraph><nodegraph name="surface"><add name="instance" nodedef="ND_value"/></nodegraph></materialx>"#,
            DiagnosticCode::InvalidStructure,
            "does not match nodedef",
        ),
        (
            r#"<materialx><nodedef name="ND_value" node="constant"><input name="value" type="float" value="0"/><output name="out" type="float"/></nodedef><nodegraph name="NG_value" nodedef="ND_value"><input name="value" type="float"/><constant name="source" type="float" value="0"/><output name="out" type="float" nodename="source"/></nodegraph><nodegraph name="surface"><constant name="instance" type="float" value="1" nodedef="ND_value"/></nodegraph></materialx>"#,
            DiagnosticCode::InvalidStructure,
            "cannot carry a node-level value",
        ),
        (
            r#"<materialx><nodedef name="ND_value" node="constant"><input name="value" type="float" value="0"/><output name="out" type="float"/></nodedef><nodegraph name="NG_value" nodedef="ND_value"><input name="value" type="float"/><constant name="source" type="float" value="0"/><output name="out" type="float" nodename="source"/></nodegraph><nodegraph name="surface"><constant name="instance" nodedef="ND_value"><input name="unknown" type="float" value="1"/></constant></nodegraph></materialx>"#,
            DiagnosticCode::InvalidStructure,
            "is not declared by nodegraph",
        ),
        (
            r#"<materialx><nodedef name="ND_value" node="constant"><input name="value" type="float" value="0"/><output name="out" type="float"/></nodedef><nodegraph name="NG_value" nodedef="ND_value"><input name="value" type="float"/><constant name="source" type="float" value="0"/><output name="out" type="float" nodename="source"/></nodegraph><nodegraph name="surface"><constant name="instance" nodedef="ND_value"><output name="other" type="float"/></constant></nodegraph></materialx>"#,
            DiagnosticCode::InvalidReference,
            "is not declared by nodedef",
        ),
    ];

    for (source, code, message) in cases {
        let error = parse_materialx(source).expect_err(message);
        assert_eq!(error.0.code, code, "{source}");
        assert!(error.to_string().contains(message), "{error}");
    }
}
