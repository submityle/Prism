use prism_render_architecture::abi::GenerationalHandle;
use prism_render_architecture::reference_pt::{sampler::Rng, Vec3};
use prism_render_material::{
    GpuSurfaceParameters, MaterialVtFormat, MaterialVtSourceDescriptor, MaterialVtSourceSnapshot,
    ProceduralVtBakeConfig,
};
use prism_render_material_authoring::{
    export_materialx, parse_materialx, AuthorNodeIdentity, DiagnosticCode, MaterialArtifact,
    MaterialArtifactError, MaterialArtifactRequest, MaterialVtAuthoringBinding,
    MaterialVtPublication, ReferenceMaterialError, ReferenceShadeContext,
};

fn source(texture_id: u32) -> MaterialVtSourceSnapshot {
    let descriptor =
        MaterialVtSourceDescriptor::new(texture_id, 2, 2, 1, 2, 2, MaterialVtFormat::Rgba8Unorm)
            .expect("valid two-level source descriptor");
    MaterialVtSourceSnapshot::new(descriptor, vec![128; 20]).expect("valid source payload")
}

fn image_surface_document() -> prism_render_material_authoring::Document {
    parse_materialx(
        r#"<materialx>
          <nodegraph name="surface">
            <image name="albedo" type="color3" file="albedo.exr"/>
            <open_pbr_surface name="surface">
              <input name="base_color" type="color3" nodename="albedo"/>
            </open_pbr_surface>
          </nodegraph>
        </materialx>"#,
    )
    .expect("valid image surface")
}

fn image_binding() -> MaterialVtAuthoringBinding {
    MaterialVtAuthoringBinding {
        graph: "surface".into(),
        node: "albedo".into(),
        semantic: 0,
        bindless_index: 17,
        generation: 4,
        sampler_index: 9,
        source: source(31),
    }
}

#[test]
fn generic_node_element_preserves_structure_and_runtime_category() {
    let document = parse_materialx(
        r#"<materialx><nodegraph name="g">
          <node name="sum" type="add">
            <input name="in1" type="float" value="0.2"/>
            <input name="in2" type="float" value="0.3"/>
          </node>
        </nodegraph></materialx>"#,
    )
    .expect("generic node is supported");
    let node = &document.nodegraphs[0].nodes[0];
    assert!(node.generic_element);
    assert_eq!(node.category, "add");
    let round_trip = parse_materialx(&export_materialx(&document)).expect("round trip parses");
    assert_eq!(round_trip, document);
}

#[test]
fn direct_nodedef_identity_is_canonical_and_graph_instances_are_not_mislabeled() {
    let document = parse_materialx(
        r#"<materialx>
          <nodedef name="ND_plus" node="plus">
            <input name="in1" type="float" value="0.1"/>
            <input name="in2" type="float" value="0.2"/>
            <output name="out" type="float"/>
          </nodedef>
          <nodedef name="ND_compound" node="plus">
            <input name="in1" type="float" value="0.1"/>
            <input name="in2" type="float" value="0.2"/>
            <output name="out" type="float"/>
          </nodedef>
          <nodegraph name="NG_compound" nodedef="ND_compound">
            <input name="in1" type="float" value="0.1"/>
            <input name="in2" type="float" value="0.2"/>
            <plus name="sum">
              <input name="in1" interfacename="in1"/>
              <input name="in2" interfacename="in2"/>
            </plus>
            <output name="out" type="float" nodename="sum"/>
          </nodegraph>
          <nodegraph name="g">
            <plus name="direct" nodedef="ND_plus"/>
            <plus name="compound" nodedef="ND_compound"/>
          </nodegraph>
        </materialx>"#,
    )
    .expect("valid direct and graph-backed definitions");
    let graph = document
        .nodegraphs
        .iter()
        .find(|graph| graph.name == "g")
        .expect("instance graph");
    let direct = prism_render_material_authoring::author_node_identity(&document, &graph.nodes[0])
        .expect("direct identity resolves")
        .expect("direct node has identity");
    assert_eq!(
        direct,
        AuthorNodeIdentity {
            nodedef: "ND_plus".into(),
            source_category: "plus".into(),
            canonical_category: "add".into(),
            pcm_uid: "org.prism.author_node.materialx.add".into(),
        }
    );
    assert!(
        prism_render_material_authoring::author_node_identity(&document, &graph.nodes[1])
            .expect("compound identity resolves")
            .is_none()
    );
}

#[test]
fn artifact_build_publishes_existing_texture_row_and_runtime_record() {
    let document = image_surface_document();
    let artifact = MaterialArtifact::build(
        &document,
        MaterialArtifactRequest::surface("surface", "surface"),
        [image_binding()],
    )
    .expect("image artifact builds");
    assert_eq!(artifact.surface.texture_inputs[0].graph, "surface");
    assert_eq!(artifact.surface.texture_inputs[0].node, "albedo");
    assert_eq!(artifact.textures.len(), 1);
    let rows = artifact.texture_rows();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].index, 17);
    assert_eq!(rows[0].generation, 4);
    assert_eq!(rows[0].semantic, 0);
    assert_eq!(rows[0].virtual_texture_id, 31);
    assert_eq!(rows[0].virtual_texture_width, 2);
    assert_eq!(rows[0].virtual_texture_page_size, 2);

    let record = artifact
        .material_record(GenerationalHandle::new(8, 2), 11)
        .expect("u32 revision is representable");
    assert_eq!(record.handle, GenerationalHandle::new(8, 2));
    assert_eq!(record.revision, 11);
    assert_eq!(record.textures, rows);

    let response = artifact
        .compile_runtime_response(GenerationalHandle::new(8, 2), 11)
        .expect("surface artifact compiles to the runtime response seam");
    assert_eq!(response.record, record);
    assert!(response.procedural.is_none());
}

#[test]
fn artifact_reports_missing_and_mismatched_texture_bindings() {
    let document = image_surface_document();
    let missing = MaterialArtifact::build(
        &document,
        MaterialArtifactRequest::surface("surface", "surface"),
        [],
    )
    .expect_err("unbound authored image must not become a scalar default");
    assert_eq!(
        missing,
        MaterialArtifactError::MissingTextureBinding("surface::albedo".into())
    );

    let mut binding = image_binding();
    binding.semantic = 1;
    let mismatch = MaterialArtifact::build(
        &document,
        MaterialArtifactRequest::surface("surface", "surface"),
        [binding],
    )
    .expect_err("wrong texture semantic must be rejected");
    assert_eq!(
        mismatch,
        MaterialArtifactError::TextureSemanticMismatch {
            field: "base_color".into(),
            node: "surface::albedo".into(),
            actual: 1,
            expected: 0,
        }
    );
}

#[test]
fn texture_bindings_respect_materialx_graph_scope() {
    let document = parse_materialx(
        r#"<materialx>
          <nodegraph name="surface_a">
            <image name="albedo" type="color3" file="a.exr"/>
            <open_pbr_surface name="surface">
              <input name="base_color" type="color3" nodename="albedo"/>
            </open_pbr_surface>
          </nodegraph>
          <nodegraph name="surface_b">
            <image name="albedo" type="color3" file="b.exr"/>
          </nodegraph>
        </materialx>"#,
    )
    .expect("same node names are legal in different MaterialX graphs");
    let mut binding = image_binding();
    binding.graph = "surface_b".into();
    let error = MaterialArtifact::build(
        &document,
        MaterialArtifactRequest::surface("surface_a", "surface"),
        [binding],
    )
    .expect_err("a same-named image from another graph must not satisfy the surface");
    assert_eq!(
        error,
        MaterialArtifactError::MissingTextureBinding("surface_a::albedo".into())
    );
}

#[test]
fn nested_graph_texture_parameter_reaches_the_surface_vt_binding() {
    let document = parse_materialx(
        r#"<materialx>
          <nodedef name="ND_texture_passthrough" node="add">
            <input name="source" type="color3"/>
            <output name="out" type="color3"/>
          </nodedef>
          <nodegraph name="NG_texture_passthrough" nodedef="ND_texture_passthrough">
            <input name="source" type="color3"/>
            <normalmap name="normal">
              <input name="in" interfacename="source"/>
            </normalmap>
            <output name="out" type="color3" nodename="normal"/>
          </nodegraph>
          <nodegraph name="surface">
            <image name="albedo" type="color3" file="albedo.exr"/>
            <add name="texture" nodedef="ND_texture_passthrough">
              <input name="source" nodename="albedo"/>
            </add>
            <open_pbr_surface name="surface">
              <input name="base_color" type="color3" nodename="texture"/>
            </open_pbr_surface>
          </nodegraph>
        </materialx>"#,
    )
    .expect("nested texture graph should parse");
    let artifact = MaterialArtifact::build(
        &document,
        MaterialArtifactRequest::surface("surface", "surface"),
        [image_binding()],
    )
    .expect("nested image parameter should resolve to the existing VT row");
    assert_eq!(artifact.surface.texture_inputs[0].node, "albedo");
    assert_eq!(artifact.texture_rows()[0].virtual_texture_id, 31);
}

#[test]
fn artifact_bakes_procedural_graph_through_existing_vt_snapshot_api() {
    let document = parse_materialx(
        r#"<materialx>
          <nodegraph name="surface">
            <constant name="roughness" type="float" value="0.4"/>
            <open_pbr_surface name="surface">
              <input name="specular_roughness" type="float" nodename="roughness"/>
            </open_pbr_surface>
          </nodegraph>
          <nodegraph name="procedural">
            <constant name="value" type="float" value="0.25"/>
            <output name="out" type="float" nodename="value"/>
          </nodegraph>
        </materialx>"#,
    )
    .expect("valid surface and procedural graph");
    let artifact = MaterialArtifact::build(
        &document,
        MaterialArtifactRequest::surface("surface", "surface").with_procedural_vt(
            "procedural",
            ProceduralVtBakeConfig::new(47, 2, 2, 2),
            MaterialVtPublication {
                semantic: 7,
                bindless_index: 21,
                generation: 5,
                sampler_index: 4,
            },
        ),
        [],
    )
    .expect("procedural artifact builds");
    assert_eq!(artifact.procedural_graph.as_ref().unwrap().nodes.len(), 1);
    let source = artifact.procedural_source.as_ref().expect("baked source");
    assert_eq!(source.descriptor().texture_id(), 47);
    assert_eq!(source.descriptor().width(), 2);
    assert_eq!(source.bytes().len(), 20);
    assert_eq!(&source.bytes()[0..4], &[64, 64, 64, 64]);
    assert_eq!(artifact.texture_rows().len(), 1);
    assert_eq!(artifact.texture_rows()[0].semantic, 7);
    assert_eq!(artifact.texture_rows()[0].index, 21);
    assert_eq!(artifact.texture_rows()[0].generation, 5);
    assert_eq!(artifact.texture_rows()[0].sampler_index, 4);

    let response = artifact
        .compile_runtime_response(GenerationalHandle::new(3, 1), 12)
        .expect("procedural artifact compiles its normalized response");
    let normalized = response.procedural.expect("procedural response is present");
    assert_eq!(normalized.nodes.len(), 1);
    assert_eq!(response.record.textures, artifact.texture_rows());
}

#[test]
fn invalid_procedural_response_is_rejected_before_publication() {
    let document = parse_materialx(
        r#"<materialx><nodegraph name="surface"><open_pbr_surface name="surface"/></nodegraph></materialx>"#,
    )
    .expect("surface parses");
    let mut artifact = MaterialArtifact::build(
        &document,
        MaterialArtifactRequest::surface("surface", "surface"),
        [],
    )
    .expect("surface artifact builds");
    artifact.procedural_graph = Some(prism_render_material::MaterialGraph::default());
    let error = artifact
        .compile_runtime_response(GenerationalHandle::new(4, 1), 1)
        .expect_err("invalid procedural graph must not be published");
    assert_eq!(
        error,
        MaterialArtifactError::RuntimeValidation(
            prism_render_material::MaterialValidationError::MissingOutput
        )
    );
}

#[test]
fn procedural_publication_metadata_is_required_and_revision_overflow_fails_closed() {
    let document = parse_materialx(
        r#"<materialx>
          <nodegraph name="surface">
            <open_pbr_surface name="surface"/>
          </nodegraph>
          <nodegraph name="procedural">
            <constant name="value" type="float" value="0.25"/>
            <output name="out" type="float" nodename="value"/>
          </nodegraph>
        </materialx>"#,
    )
    .expect("valid procedural surface document");
    let missing = MaterialArtifact::build(
        &document,
        MaterialArtifactRequest::surface("surface", "surface"),
        [],
    )
    .expect("surface-only artifact remains valid");
    assert_eq!(
        missing
            .material_record(GenerationalHandle::new(1, 1), u64::from(u32::MAX) + 1)
            .expect_err("revision truncation must be rejected"),
        MaterialArtifactError::RevisionOverflow(u64::from(u32::MAX) + 1)
    );

    let error = MaterialArtifact::build(
        &document,
        MaterialArtifactRequest {
            surface_graph: "surface".into(),
            surface_node: "surface".into(),
            procedural_graph: Some("procedural".into()),
            procedural_vt: Some(ProceduralVtBakeConfig::new(48, 1, 1, 1)),
            procedural_publication: None,
            plan: prism_render_material::ShadingPlan::default(),
            render_class: prism_render_material::MaterialRenderClass::default(),
            render_domain: prism_render_material::RenderDomain::Surface,
        },
        [],
    )
    .expect_err("procedural source without publication metadata is not publishable");
    assert_eq!(error, MaterialArtifactError::MissingProceduralPublication);
}

#[test]
fn authoring_surface_exposes_independent_reference_material_and_rejects_dropped_features() {
    let document = parse_materialx(
        r#"<materialx>
          <nodegraph name="surface">
            <open_pbr_surface name="surface">
              <input name="base_color" type="color3" value="0.2, 0.4, 0.6"/>
              <input name="specular_roughness" type="float" value="0.25"/>
            </open_pbr_surface>
          </nodegraph>
        </materialx>"#,
    )
    .expect("reference fixture parses");
    let artifact = MaterialArtifact::build(
        &document,
        MaterialArtifactRequest::surface("surface", "surface"),
        [],
    )
    .expect("reference fixture builds as an artifact");
    let reference = artifact
        .reference_material()
        .expect("supported scalar surface maps to OpenPBR reference inputs");
    assert_eq!(reference.openpbr.base_color.to_array(), [0.2, 0.4, 0.6]);
    assert_eq!(reference.openpbr.perceptual_roughness, 0.25);

    let mut anisotropic = artifact.surface.clone();
    anisotropic.parameters.anisotropy = 0.2;
    assert_eq!(
        prism_render_material_authoring::reference_material(&anisotropic)
            .expect_err("anisotropy needs a tangent frame not carried by this authoring boundary"),
        ReferenceMaterialError::UnsupportedField {
            field: "anisotropy"
        }
    );
}

#[test]
fn canonical_reference_adapter_carries_supported_fields_and_shade_context() {
    let parameters = GpuSurfaceParameters {
        base_color: [0.21, 0.32, 0.43, 0.67],
        emission_color: [0.7, 0.5, 0.3, 1.0],
        emission_luminance: 12.0,
        base_metalness: 0.37,
        specular_roughness: 0.28,
        specular_weight: 0.74,
        base_weight: 0.83,
        base_diffuse_roughness: 0.19,
        specular_color: [0.91, 0.82, 0.73, 1.0],
        specular_ior: 1.61,
        thin_walled: 1.0,
        geometry_opacity: 0.63,
        transmission_weight: 0.42,
        thickness: 0.71,
        coat_weight: 0.46,
        coat_roughness: 0.23,
        coat_ior: 1.57,
        coat_color: [0.88, 0.77, 0.66, 1.0],
        coat_darkening: 0.39,
        fuzz_weight: 0.27,
        fuzz_roughness: 0.51,
        fuzz_color: [0.41, 0.52, 0.63, 1.0],
        subsurface_weight: 0.18,
        subsurface_color: [0.72, 0.61, 0.5, 1.0],
        subsurface_radius: 0.93,
        subsurface_radius_scale: [0.9, 0.6, 0.3, 1.0],
        subsurface_scatter_anisotropy: -0.22,
        index_of_refraction: 1.48,
        transmission_dispersion_scale: 0.64,
        transmission_color: [0.79, 0.68, 0.57, 1.0],
        transmission_scatter: [0.11, 0.22, 0.33, 1.0],
        thin_film_weight: 0.31,
        thin_film_thickness: 0.41,
        thin_film_ior: 1.38,
        transmission_scatter_anisotropy: 0.36,
        transmission_dispersion_abbe_number: 31.0,
        glint_weight: 0.44,
        glint_roughness: 0.24,
        glint_density: 2.7,
        glint_bin_half_width: 0.08,
        ..GpuSurfaceParameters::default()
    };
    let context = ReferenceShadeContext {
        glint_state: -1.0,
        glint_cell: [3.0, -2.0],
        glint_footprint_area: 0.27,
    };
    let reference =
        prism_render_material_authoring::reference_material_from_parameters_with_context(
            parameters, context,
        )
        .expect("supported canonical OpenPBR fields must map without loss");
    let openpbr = reference.openpbr;

    assert_eq!(openpbr.base_color, Vec3::new(0.21, 0.32, 0.43));
    assert_eq!(openpbr.base_weight, parameters.base_weight);
    assert_eq!(openpbr.diffuse_roughness, parameters.base_diffuse_roughness);
    assert_eq!(openpbr.metallic, parameters.base_metalness);
    assert_eq!(openpbr.perceptual_roughness, parameters.specular_roughness);
    assert_eq!(openpbr.specular_color, Vec3::new(0.91, 0.82, 0.73));
    assert_eq!(openpbr.transmission_scatter, Vec3::new(0.11, 0.22, 0.33));
    assert_eq!(openpbr.coat_color, Vec3::new(0.88, 0.77, 0.66));
    assert_eq!(openpbr.fuzz_color, Vec3::new(0.41, 0.52, 0.63));
    assert_eq!(openpbr.subsurface_radius_scale, Vec3::new(0.9, 0.6, 0.3));
    assert_eq!(openpbr.iridescence_thickness_nm, 410.0);
    assert_eq!(openpbr.glint_weight, parameters.glint_weight);
    assert_eq!(openpbr.glint_state, context.glint_state);
    assert_eq!(openpbr.glint_cell, context.glint_cell);
    assert_eq!(openpbr.glint_footprint_area, context.glint_footprint_area);
    assert_eq!(reference.base_alpha, parameters.base_color[3]);
    assert_eq!(reference.geometry_opacity, parameters.geometry_opacity);
    assert_eq!(reference.ambient_occlusion, parameters.ambient_occlusion);
    assert_eq!(reference.alpha_cutoff, parameters.alpha_cutoff);
    for (actual, expected) in reference.emission.into_iter().zip([8.4, 6.0, 3.6]) {
        assert!((actual - expected).abs() < 1.0e-6);
    }

    let wo = Vec3::new(0.2, 0.96, 0.05).normalize_or_zero();
    let wi = Vec3::new(-0.3, 0.87, 0.39).normalize_or_zero();
    let value = reference.evaluate(wo, wi, Vec3::new(0.0, 1.0, 0.0));
    assert!(value.is_finite());
    let mut rng = Rng::seed(0xA11CE);
    let sample = reference
        .sample(wo, Vec3::new(0.0, 1.0, 0.0), &mut rng)
        .expect("mapped OpenPBR closure must provide a CPU sample");
    assert!(sample.value.is_finite());
    assert!(sample.pdf.is_finite() && sample.pdf > 0.0);
    assert!(
        (sample.pdf - reference.pdf(wo, sample.direction, Vec3::new(0.0, 1.0, 0.0))).abs() < 1.0e-6
    );
}

#[test]
fn active_glint_requires_context_instead_of_being_zeroed() {
    let mut parameters = GpuSurfaceParameters::default();
    parameters.glint_weight = 0.3;
    assert_eq!(
        prism_render_material_authoring::reference_material_from_parameters(parameters)
            .expect_err("active glint without a shade footprint must fail closed"),
        ReferenceMaterialError::MissingShadeContext { field: "glint" }
    );
    let material =
        prism_render_material_authoring::reference_material_from_parameters_with_context(
            parameters,
            ReferenceShadeContext {
                glint_state: 0.6,
                glint_cell: [1.0, 2.0],
                glint_footprint_area: 0.2,
            },
        )
        .expect("explicit glint context completes the mapping");
    assert_eq!(material.openpbr.glint_weight, 0.3);
    assert_eq!(material.openpbr.glint_state, 0.6);
}

#[test]
fn nested_graph_external_parameter_without_default_is_rejected_structurally() {
    let error = parse_materialx(
        r#"<materialx>
          <nodedef name="ND_compound" node="add">
            <input name="value" type="float"/>
            <output name="out" type="float"/>
          </nodedef>
          <nodegraph name="NG_compound" nodedef="ND_compound">
            <input name="value" type="float"/>
            <add name="sum">
              <input name="in1" interfacename="value"/>
              <input name="in2" type="float" value="0.1"/>
            </add>
            <output name="out" type="float" nodename="sum"/>
          </nodegraph>
          <nodegraph name="surface">
            <add name="instance" nodedef="ND_compound"/>
            <open_pbr_surface name="surface">
              <input name="specular_roughness" type="float" nodename="instance"/>
            </open_pbr_surface>
          </nodegraph>
        </materialx>"#,
    )
    .expect_err("an unbound forwarded parameter must not become an implicit default");
    assert_eq!(error.0.code, DiagnosticCode::InvalidStructure);
    assert!(error.to_string().contains("unbound external parameter"));
}
