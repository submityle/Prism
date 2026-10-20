use prism_editor_graph::{Graph, NodeSpec, PortSpec, PortType};

#[test]
fn public_api_supports_a_diamond_with_deterministic_order() {
    let mut graph = Graph::new();
    let source = graph
        .add_node(
            NodeSpec::new("source", ())
                .with_port(PortSpec::output("out", PortType::named("value"))),
        )
        .expect("source");
    let left = graph
        .add_node(
            NodeSpec::new("left", ())
                .with_port(PortSpec::input("in", PortType::named("value")))
                .with_port(PortSpec::output("out", PortType::named("value"))),
        )
        .expect("left");
    let right = graph
        .add_node(
            NodeSpec::new("right", ())
                .with_port(PortSpec::input("in", PortType::named("value")))
                .with_port(PortSpec::output("out", PortType::named("value"))),
        )
        .expect("right");
    let sink = graph
        .add_node(
            NodeSpec::new("sink", ()).with_port(PortSpec::input("in", PortType::named("value"))),
        )
        .expect("sink");

    graph
        .connect(
            graph.port_named(source, "out").expect("source output"),
            graph.port_named(left, "in").expect("left input"),
        )
        .expect("source to left");
    graph
        .connect(
            graph.port_named(source, "out").expect("source output"),
            graph.port_named(right, "in").expect("right input"),
        )
        .expect("source to right");
    graph
        .connect(
            graph.port_named(left, "out").expect("left output"),
            graph.port_named(sink, "in").expect("sink input"),
        )
        .expect("left to sink");
    assert_eq!(
        graph.topological_order().expect("acyclic"),
        vec![source, left, right, sink]
    );
}
