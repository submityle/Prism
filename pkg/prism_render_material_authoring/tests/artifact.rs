use prism_render_architecture::abi::GenerationalHandle;
use prism_render_material::{
    MaterialVtFormat, MaterialVtSourceDescriptor, MaterialVtSourceSnapshot, ProceduralVtBakeConfig,
};
use prism_render_material_authoring::{
    export_materialx, parse_materialx, AuthorNodeIdentity, DiagnosticCode, MaterialArtifact,
    MaterialArtifactError, MaterialArtifactRequest, MaterialVtAuthoringBinding,
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

    let record = artifact.material_record(GenerationalHandle::new(8, 2), 11);
    assert_eq!(record.handle, GenerationalHandle::new(8, 2));
    assert_eq!(record.revision, 11);
    assert_eq!(record.textures, rows);
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
        MaterialArtifactError::MissingTextureBinding("albedo".into())
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
            node: "albedo".into(),
            actual: 1,
            expected: 0,
        }
    );
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
        MaterialArtifactRequest::surface("surface", "surface")
            .with_procedural_vt("procedural", ProceduralVtBakeConfig::new(47, 2, 2, 2)),
        [],
    )
    .expect("procedural artifact builds");
    assert_eq!(artifact.procedural_graph.as_ref().unwrap().nodes.len(), 1);
    let source = artifact.procedural_source.as_ref().expect("baked source");
    assert_eq!(source.descriptor().texture_id(), 47);
    assert_eq!(source.descriptor().width(), 2);
    assert_eq!(source.bytes().len(), 20);
    assert_eq!(&source.bytes()[0..4], &[64, 64, 64, 64]);
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
