#![allow(missing_docs, reason = "integration tests are self-describing")]

use prism_render_material::{MaterialNode, MaterialVtFormat, ProceduralVtBakeConfig};
use prism_render_material_authoring::{
    bake_procedural_graph, lower_procedural_graph, parse_materialx, ProceduralGraphError,
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
