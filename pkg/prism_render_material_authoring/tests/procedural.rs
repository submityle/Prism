#![allow(missing_docs, reason = "integration tests are self-describing")]

use prism_render_material::{MaterialNode, MaterialVtFormat, ProceduralVtBakeConfig};
use prism_render_material_authoring::{
    bake_procedural_graph, lower_procedural_graph, parse_materialx, DiagnosticCode,
    ProceduralGraphError,
};

const SOURCE: &str = r#"<materialx version="1.38">
  <nodegraph name="procedural">
    <texcoord name="uv" />
    <noise name="noise">
      <input name="scale" type="float" value="4" />
      <input name="seed" type="integer" value="7" />
    </noise>
    <curve name="ramp" points="0,0.1;1,0.9">
      <input name="input" nodename="noise" />
    </curve>
    <constant name="white" type="color4" value="1,1,1,1" />
    <blend name="mix">
      <input name="a" nodename="ramp" />
      <input name="b" nodename="white" />
      <input name="weight" nodename="uv" />
    </blend>
    <output name="out" nodename="mix" />
  </nodegraph>
</materialx>"#;

#[test]
fn materialx_procedural_graph_lowers_and_bakes_to_vt() {
    let document = parse_materialx(SOURCE).expect("procedural graph should parse");
    let graph = &document.nodegraphs[0];
    let lowered = lower_procedural_graph(&document, graph).expect("graph should lower");
    assert!(lowered
        .nodes
        .values()
        .any(|node| matches!(node, MaterialNode::Noise { .. })));
    let snapshot =
        bake_procedural_graph(&document, graph, ProceduralVtBakeConfig::new(42, 4, 2, 2))
            .expect("procedural graph should bake");
    assert_eq!(snapshot.descriptor().format(), MaterialVtFormat::Rgba8Unorm);
    assert_eq!(snapshot.descriptor().mip_count(), 3);
    assert!(snapshot.bytes().iter().any(|value| *value != 0));
}

#[test]
fn unsupported_procedural_nodes_fail_closed() {
    let document = parse_materialx(
        r#"<materialx><nodegraph name="g">
          <image name="texture" />
          <output name="out" nodename="texture" />
        </nodegraph></materialx>"#,
    )
    .expect("structural parser should preserve the node");
    let error = lower_procedural_graph(&document, &document.nodegraphs[0])
        .expect_err("texture nodes require a separate source binding");
    assert!(matches!(
        error,
        ProceduralGraphError::UnsupportedCategory { .. }
    ));
}

#[test]
fn nested_procedural_graph_expands_external_parameter_bindings() {
    let document = parse_materialx(
        r#"<materialx>
          <nodedef name="ND_scale" node="multiply">
            <input name="factor" type="float" value="2"/>
            <input name="value" type="float"/>
            <output name="out" type="float"/>
          </nodedef>
          <nodegraph name="NG_scale" nodedef="ND_scale">
            <input name="factor" type="float" value="2"/>
            <input name="value" type="float"/>
            <multiply name="product">
              <input name="in1" interfacename="factor"/>
              <input name="in2" interfacename="value"/>
            </multiply>
            <output name="out" type="float" nodename="product"/>
          </nodegraph>
          <nodegraph name="procedural">
            <constant name="source" type="float" value="0.25"/>
            <multiply name="instance" nodedef="ND_scale">
              <input name="value" nodename="source"/>
            </multiply>
            <output name="out" nodename="instance"/>
          </nodegraph>
        </materialx>"#,
    )
    .expect("nested procedural graph should parse");
    let graph = document
        .nodegraphs
        .iter()
        .find(|graph| graph.name == "procedural")
        .expect("procedural graph");
    let lowered = lower_procedural_graph(&document, graph).expect("compound graph lowers");
    assert!(lowered
        .nodes
        .values()
        .any(|node| matches!(node, MaterialNode::Multiply(_, _))));
    let snapshot =
        bake_procedural_graph(&document, graph, ProceduralVtBakeConfig::new(43, 1, 1, 1))
            .expect("compound graph bakes");
    assert_eq!(&snapshot.bytes()[0..4], &[128, 128, 128, 128]);
}

#[test]
fn procedural_noise_requires_authored_scale_and_seed() {
    let document = parse_materialx(
        r#"<materialx><nodegraph name="g">
          <noise name="noise"><input name="scale" type="float" value="4"/></noise>
          <output name="out" nodename="noise"/>
        </nodegraph></materialx>"#,
    )
    .expect("noise graph is structurally valid");
    let error = lower_procedural_graph(&document, &document.nodegraphs[0])
        .expect_err("missing seed must not become a hidden deterministic default");
    assert_eq!(
        error,
        ProceduralGraphError::InvalidInput {
            node: "noise".into(),
            input: "seed".into(),
        }
    );
}

#[test]
fn procedural_subtract_is_lowered_without_changing_runtime_abi() {
    let document = parse_materialx(
        r#"<materialx><nodegraph name="g">
          <subtract name="difference">
            <input name="a" type="float" value="0.75"/>
            <input name="b" type="float" value="0.25"/>
          </subtract>
          <output name="out" nodename="difference"/>
        </nodegraph></materialx>"#,
    )
    .expect("subtract graph is structurally valid");
    let snapshot = bake_procedural_graph(
        &document,
        &document.nodegraphs[0],
        ProceduralVtBakeConfig::new(44, 1, 1, 1),
    )
    .expect("subtract lowers to add and multiply");
    assert_eq!(&snapshot.bytes()[0..4], &[128, 128, 128, 128]);
}

#[test]
fn procedural_cross_graph_connection_is_lowered_without_a_hidden_default() {
    let document = parse_materialx(
        r#"<materialx>
          <nodegraph name="values">
            <constant name="value" type="float" value="0.75"/>
            <output name="out" type="float" nodename="value"/>
          </nodegraph>
          <nodegraph name="procedural">
            <multiply name="scale">
              <input name="in1" type="float" nodegraph="values"/>
              <input name="in2" type="float" value="0.5"/>
            </multiply>
            <output name="out" type="float" nodename="scale"/>
          </nodegraph>
        </materialx>"#,
    )
    .expect("cross-graph procedural connection should parse");
    let graph = document
        .nodegraphs
        .iter()
        .find(|graph| graph.name == "procedural")
        .expect("procedural graph");
    let snapshot =
        bake_procedural_graph(&document, graph, ProceduralVtBakeConfig::new(45, 1, 1, 1))
            .expect("cross-graph constant should lower and bake");
    assert_eq!(&snapshot.bytes()[0..4], &[96, 96, 96, 96]);
}

#[test]
fn nested_procedural_graph_missing_external_parameter_is_structured_error() {
    let error = parse_materialx(
        r#"<materialx>
          <nodedef name="ND_passthrough" node="convert">
            <input name="value" type="float"/>
            <output name="out" type="float"/>
          </nodedef>
          <nodegraph name="NG_passthrough" nodedef="ND_passthrough">
            <input name="value" type="float"/>
            <convert name="copy"><input name="in" interfacename="value"/></convert>
            <output name="out" type="float" nodename="copy"/>
          </nodegraph>
          <nodegraph name="procedural">
            <convert name="instance" nodedef="ND_passthrough"/>
            <output name="out" type="float" nodename="instance"/>
          </nodegraph>
        </materialx>"#,
    )
    .expect_err("missing nested external parameter must fail at the authoring boundary");
    assert_eq!(error.0.code, DiagnosticCode::InvalidStructure);
    assert!(error
        .to_string()
        .contains("unbound external parameter 'value'"));
}
