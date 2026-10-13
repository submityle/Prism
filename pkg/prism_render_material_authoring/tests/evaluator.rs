use prism_render_material_authoring::{
    evaluate_connection, parse_materialx, Connection, DiagnosticCode,
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
