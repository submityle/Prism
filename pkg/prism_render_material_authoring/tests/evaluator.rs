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
