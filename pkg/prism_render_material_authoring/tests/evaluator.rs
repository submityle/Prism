use prism_render_material_authoring::{
    author_node_manifest, evaluate_connection, export_materialx, parse_materialx, Connection,
    DiagnosticCode,
};

#[test]
fn evaluates_nested_numeric_expression_deterministically() {
    let doc = parse_materialx(
        r#"<materialx><nodegraph name="g">
      <constant name="a" type="float" value="0.2"/><constant name="b" type="float" value="0.3"/>
      <add name="sum"><input name="in1" nodename="a"/><input name="in2" nodename="b"/></add>
    </nodegraph></materialx>"#,
    )
    .unwrap();
    let graph = &doc.nodegraphs[0];
    let value = evaluate_connection(
        &doc,
        graph,
        &Connection {
            node: Some("sum".into()),
            output: None,
            nodegraph: None,
        },
    )
    .unwrap();
    assert_eq!(value.value, "0.5");
}

#[test]
fn evaluator_rejects_cycles_and_non_finite_results() {
    let cyclic = parse_materialx(
        r#"<materialx><nodegraph name="g">
      <add name="a"><input name="x" nodename="b"/><input name="y" value="1" type="float"/></add>
      <add name="b"><input name="x" nodename="a"/><input name="y" value="1" type="float"/></add>
    </nodegraph></materialx>"#,
    )
    .unwrap();
    let graph = &cyclic.nodegraphs[0];
    let error = evaluate_connection(
        &cyclic,
        graph,
        &Connection {
            node: Some("a".into()),
            output: None,
            nodegraph: None,
        },
    )
    .unwrap_err();
    assert_eq!(error.0.code, DiagnosticCode::UnsafeConnection);

    let divide = parse_materialx(
        r#"<materialx><nodegraph name="g">
      <constant name="a" type="float" value="1"/><constant name="z" type="float" value="0"/>
      <divide name="d"><input name="x" nodename="a"/><input name="y" nodename="z"/></divide>
    </nodegraph></materialx>"#,
    )
    .unwrap();
    let graph = &divide.nodegraphs[0];
    let error = evaluate_connection(
        &divide,
        graph,
        &Connection {
            node: Some("d".into()),
            output: None,
            nodegraph: None,
        },
    )
    .unwrap_err();
    assert_eq!(error.0.code, DiagnosticCode::InvalidValue);
}

#[test]
fn evaluator_rejects_reversed_clamp_bounds_without_panicking() {
    let document = parse_materialx(
        r#"<materialx><nodegraph name="g">
      <constant name="value" type="float" value="0.5"/>
      <constant name="low" type="float" value="1"/>
      <constant name="high" type="float" value="0"/>
      <clamp name="bounded"><input name="in" nodename="value"/><input name="low" nodename="low"/><input name="high" nodename="high"/></clamp>
    </nodegraph></materialx>"#,
    )
    .unwrap();
    let graph = &document.nodegraphs[0];
    let error = evaluate_connection(
        &document,
        graph,
        &Connection {
            node: Some("bounded".into()),
            output: None,
            nodegraph: None,
        },
    )
    .unwrap_err();
    assert_eq!(error.0.code, DiagnosticCode::InvalidValue);
    assert!(error.to_string().contains("lower bound"));
}

#[test]
fn evaluator_resolves_graph_outputs_and_interfaces_through_nested_expressions() {
    let document = parse_materialx(
        r#"<materialx><nodegraph name="g"><input name="scale" type="float" value="0.5"/><constant name="base" type="float" value="0.4"/><multiply name="scaled"><input name="in1" nodename="base"/><input name="in2" interfacename="scale"/></multiply><output name="out" type="float" nodename="scaled"/></nodegraph></materialx>"#,
    )
    .unwrap();
    let graph = &document.nodegraphs[0];
    let value = evaluate_connection(
        &document,
        graph,
        &Connection {
            node: None,
            output: Some("out".into()),
            nodegraph: Some("g".into()),
        },
    )
    .unwrap();
    assert_eq!(value.type_name, "float");
    let numeric = value.value.parse::<f32>().unwrap();
    assert!((numeric - 0.2).abs() < 1e-6);
}

#[test]
fn evaluator_rejects_graph_output_cycles() {
    let document = parse_materialx(
        r#"<materialx><nodegraph name="g"><output name="a" type="float" nodegraph="g" output="b"/><output name="b" type="float" nodegraph="g" output="a"/></nodegraph></materialx>"#,
    )
    .unwrap();
    let graph = &document.nodegraphs[0];
    let error = evaluate_connection(
        &document,
        graph,
        &Connection {
            node: None,
            output: Some("a".into()),
            nodegraph: None,
        },
    )
    .unwrap_err();
    assert_eq!(error.0.code, DiagnosticCode::UnsafeConnection);
    assert!(error.to_string().contains("cyclic graph output"));
}

#[test]
fn nodedef_author_node_uses_defaults_overrides_and_stable_pcm_identity() {
    let source = r#"<materialx>
      <nodedef name="ND_tint_add" node="add">
        <input name="base" type="color3" value="0.1, 0.2, 0.3"/>
        <input name="tint" type="color3" value="0.2, 0.3, 0.4"/>
        <output name="out" type="color3"/>
      </nodedef>
      <nodegraph name="g">
        <add name="defaulted" nodedef="ND_tint_add"/>
        <add name="overridden" nodedef="ND_tint_add">
          <input name="tint" type="color3" value="0.9, 0.8, 0.7"/>
        </add>
      </nodegraph>
    </materialx>"#;
    let document = parse_materialx(source).expect("valid author-node instances");
    let graph = &document.nodegraphs[0];
    let evaluate = |name: &str| {
        evaluate_connection(
            &document,
            graph,
            &Connection {
                node: Some(name.into()),
                output: Some("out".into()),
                nodegraph: None,
            },
        )
        .expect("author node evaluates")
    };
    assert_eq!(evaluate("defaulted").value, "0.3, 0.5, 0.7");
    assert_eq!(evaluate("overridden").value, "1, 1, 1");

    let manifest = author_node_manifest(&document, &graph.nodes[0])
        .expect("binding resolves")
        .expect("node has a nodedef");
    assert_eq!(manifest.uid(), "org.prism.author_node.materialx.add");
    assert_eq!(manifest.registry_slot(), None);
}

#[test]
fn nodedef_author_node_round_trip_preserves_identity_and_evaluation() {
    let source = r#"<materialx>
      <nodedef name="ND_scale" node="multiply">
        <input name="value" type="float"/>
        <input name="scale" type="float" value="2"/>
        <output name="result" type="float"/>
      </nodedef>
      <nodegraph name="g">
        <multiply name="scaled" nodedef="ND_scale">
          <input name="value" type="float" value="1.5"/>
        </multiply>
      </nodegraph>
    </materialx>"#;
    let first = parse_materialx(source).expect("parse");
    let exported = export_materialx(&first);
    let second = parse_materialx(&exported).expect("reparse");
    let node = &second.nodegraphs[0].nodes[0];
    let manifest = author_node_manifest(&second, node)
        .expect("binding resolves")
        .expect("node has a nodedef");
    let value = evaluate_connection(
        &second,
        &second.nodegraphs[0],
        &Connection {
            node: Some("scaled".into()),
            output: Some("result".into()),
            nodegraph: None,
        },
    )
    .expect("round-tripped author node evaluates");
    assert_eq!(manifest.uid(), "org.prism.author_node.materialx.multiply");
    assert_eq!(value.type_name, "float");
    assert_eq!(value.value, "3");
}

#[test]
fn nodedef_convert_preserves_components_across_equal_width_numeric_types() {
    let document = parse_materialx(
        r#"<materialx>
      <nodedef name="ND_color_to_vector" node="convert">
        <input name="in" type="color3"/>
        <output name="out" type="vector3"/>
      </nodedef>
      <nodegraph name="g">
        <convert name="converted" nodedef="ND_color_to_vector">
          <input name="in" type="color3" value="0.1, 0.2, 0.3"/>
        </convert>
      </nodegraph>
    </materialx>"#,
    )
    .expect("equal-width numeric convert is supported");
    let value = evaluate_connection(
        &document,
        &document.nodegraphs[0],
        &Connection {
            node: Some("converted".into()),
            output: Some("out".into()),
            nodegraph: None,
        },
    )
    .expect("convert evaluates");
    assert_eq!(value.type_name, "vector3");
    assert_eq!(value.value, "0.1, 0.2, 0.3");
}

#[test]
fn graph_backed_nodedef_keeps_graph_semantics_without_builtin_author_node_alias() {
    let document = parse_materialx(
        r#"<materialx>
      <nodedef name="ND_graph_add" node="add">
        <input name="x" type="float" value="2"/>
        <input name="y" type="float" value="3"/>
        <output name="out" type="float"/>
      </nodedef>
      <nodegraph name="NG_graph_add" nodedef="ND_graph_add">
        <input name="x" type="float" value="2"/>
        <input name="y" type="float" value="3"/>
        <multiply name="product">
          <input name="in1" interfacename="x"/>
          <input name="in2" interfacename="y"/>
        </multiply>
        <output name="out" type="float" nodename="product"/>
      </nodegraph>
      <nodegraph name="use">
        <add name="instance" nodedef="ND_graph_add"/>
      </nodegraph>
    </materialx>"#,
    )
    .expect("graph-backed nodedef is structurally valid");
    let instance = &document.nodegraphs[1].nodes[0];
    assert!(author_node_manifest(&document, instance)
        .expect("graph-backed nodes are not invalid")
        .is_none());

    let value = evaluate_connection(
        &document,
        &document.nodegraphs[1],
        &Connection {
            node: Some("instance".into()),
            output: Some("out".into()),
            nodegraph: None,
        },
    )
    .expect("nodegraph implementation evaluates");
    assert_eq!(value.value, "6");
}

#[test]
fn every_schema_author_node_has_a_real_instance_semantics_and_pcm_identity() {
    let document = parse_materialx(
        r#"<materialx>
      <nodedef name="ND_add" node="add"><input name="a" type="float"/><input name="b" type="float"/><output name="out" type="float"/></nodedef>
      <nodedef name="ND_clamp" node="clamp"><input name="value" type="float"/><input name="low" type="float"/><input name="high" type="float"/><output name="out" type="float"/></nodedef>
      <nodedef name="ND_convert" node="convert"><input name="in" type="color3"/><output name="out" type="vector3"/></nodedef>
      <nodedef name="ND_divide" node="divide"><input name="a" type="float"/><input name="b" type="float"/><output name="out" type="float"/></nodedef>
      <nodedef name="ND_mix" node="mix"><input name="a" type="float"/><input name="b" type="float"/><input name="t" type="float"/><output name="out" type="float"/></nodedef>
      <nodedef name="ND_multiply" node="multiply"><input name="a" type="float"/><input name="b" type="float"/><output name="out" type="float"/></nodedef>
      <nodedef name="ND_subtract" node="subtract"><input name="a" type="float"/><input name="b" type="float"/><output name="out" type="float"/></nodedef>
      <nodegraph name="g">
        <add name="add" nodedef="ND_add"><input name="a" type="float" value="1.25"/><input name="b" type="float" value="2.75"/></add>
        <clamp name="clamp" nodedef="ND_clamp"><input name="value" type="float" value="2.0"/><input name="low" type="float" value="0.0"/><input name="high" type="float" value="1.0"/></clamp>
        <convert name="convert" nodedef="ND_convert"><input name="in" type="color3" value="0.1, 0.2, 0.3"/></convert>
        <divide name="divide" nodedef="ND_divide"><input name="a" type="float" value="6.0"/><input name="b" type="float" value="2.0"/></divide>
        <mix name="mix" nodedef="ND_mix"><input name="a" type="float" value="0.0"/><input name="b" type="float" value="1.0"/><input name="t" type="float" value="0.25"/></mix>
        <multiply name="multiply" nodedef="ND_multiply"><input name="a" type="float" value="2.0"/><input name="b" type="float" value="3.0"/></multiply>
        <subtract name="subtract" nodedef="ND_subtract"><input name="a" type="float" value="4.0"/><input name="b" type="float" value="1.5"/></subtract>
      </nodegraph>
    </materialx>"#,
    )
    .expect("all schema-declared author-node definitions are valid");
    let graph = &document.nodegraphs[0];
    let expected = [
        ("add", "org.prism.author_node.materialx.add", "4"),
        ("clamp", "org.prism.author_node.materialx.clamp", "1"),
        (
            "convert",
            "org.prism.author_node.materialx.convert",
            "0.1, 0.2, 0.3",
        ),
        ("divide", "org.prism.author_node.materialx.divide", "3"),
        ("mix", "org.prism.author_node.materialx.mix", "0.25"),
        ("multiply", "org.prism.author_node.materialx.multiply", "6"),
        (
            "subtract",
            "org.prism.author_node.materialx.subtract",
            "2.5",
        ),
    ];
    for (node_name, uid, expected_value) in expected {
        let node = graph
            .nodes
            .iter()
            .find(|node| node.name == node_name)
            .expect("author-node instance exists");
        assert_eq!(
            author_node_manifest(&document, node)
                .expect("schema and PCM author-node identities agree")
                .expect("direct nodedef resolves to an author node")
                .uid(),
            uid
        );
        let value = evaluate_connection(
            &document,
            graph,
            &Connection {
                node: Some(node_name.into()),
                output: Some("out".into()),
                nodegraph: None,
            },
        )
        .expect("author-node instance evaluates");
        assert_eq!(value.value, expected_value, "{node_name}");
    }
}
