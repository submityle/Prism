use std::fmt::Write;

use crate::{Connection, Document, Input, Node, NodeDef, NodeGraph, Output, TypedValue};

/// Exports the supported IR as deterministic, UTF-8 MaterialX XML.
///
/// Definitions and graphs are sorted by identifier by the parser and all
/// attribute maps use `BTreeMap`; source order of inputs/nodes/outputs remains
/// intentional and is preserved. No opaque or unknown content is invented.
pub fn export_materialx(document: &Document) -> String {
    let mut out = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<materialx");
    write_attributes(&mut out, &document.attributes, &[]);
    out.push_str(">\n");
    for node_def in &document.nodedefs {
        export_nodedef(&mut out, node_def, 1);
    }
    for graph in &document.nodegraphs {
        export_nodegraph(&mut out, graph, 1);
    }
    out.push_str("</materialx>\n");
    out
}

fn export_nodedef(out: &mut String, node_def: &NodeDef, indent: usize) {
    line(
        out,
        indent,
        &format!(
            "<nodedef name=\"{}\" node=\"{}\"{}>",
            escape(&node_def.name),
            escape(&node_def.category),
            attributes(&node_def.attributes)
        ),
    );
    for input in &node_def.inputs {
        export_input(out, input, indent + 1, true);
    }
    for output in &node_def.outputs {
        export_output(out, output, indent + 1, true);
    }
    line(out, indent, "</nodedef>");
}

fn export_nodegraph(out: &mut String, graph: &NodeGraph, indent: usize) {
    line(
        out,
        indent,
        &format!(
            "<nodegraph name=\"{}\"{}>",
            escape(&graph.name),
            attributes(&graph.attributes)
        ),
    );
    for node in &graph.nodes {
        export_node(out, node, indent + 1);
    }
    for output in &graph.outputs {
        export_output(out, output, indent + 1, false);
    }
    line(out, indent, "</nodegraph>");
}

fn export_node(out: &mut String, node: &Node, indent: usize) {
    let mut attrs = format!(" name=\"{}\"", escape(&node.name));
    if node.category == "node" {
        attrs.push_str(" type=\"node\"");
    }
    if let Some(value) = &node.value {
        attrs.push_str(&format!(
            " type=\"{}\"{}",
            escape(&value.type_name),
            value_attributes(value)
        ));
    }
    attrs.push_str(&attributes(&node.attributes));
    if node.inputs.is_empty() && node.outputs.is_empty() {
        line(out, indent, &format!("<{}{} />", node.category, attrs));
        return;
    }
    line(out, indent, &format!("<{}{}>", node.category, attrs));
    for input in &node.inputs {
        export_input(out, input, indent + 1, false);
    }
    for output in &node.outputs {
        export_output(out, output, indent + 1, false);
    }
    line(out, indent, &format!("</{}>", node.category));
}

fn export_input(out: &mut String, input: &Input, indent: usize, declaration: bool) {
    let mut attrs = format!(" name=\"{}\"", escape(&input.name));
    if let Some(type_name) = &input.type_name {
        attrs.push_str(&format!(" type=\"{}\"", escape(type_name)));
    }
    if let Some(value) = &input.value {
        attrs.push_str(&value_attributes(value));
    }
    if let Some(connection) = &input.connection {
        attrs.push_str(&connection_attributes(connection));
    }
    attrs.push_str(&attributes(&input.attributes));
    let _ = declaration;
    line(out, indent, &format!("<input{} />", attrs));
}

fn export_output(out: &mut String, output: &Output, indent: usize, _declaration: bool) {
    let mut attrs = format!(" name=\"{}\"", escape(&output.name));
    if let Some(type_name) = &output.type_name {
        attrs.push_str(&format!(" type=\"{}\"", escape(type_name)));
    }
    if let Some(value) = &output.value {
        attrs.push_str(&value_attributes(value));
    }
    if let Some(connection) = &output.connection {
        attrs.push_str(&connection_attributes(connection));
    }
    line(out, indent, &format!("<output{} />", attrs));
}

fn value_attributes(value: &TypedValue) -> String {
    format!(" value=\"{}\"", escape(&value.value))
}

fn connection_attributes(connection: &Connection) -> String {
    let mut attrs = String::new();
    if let Some(graph) = &connection.nodegraph {
        attrs.push_str(&format!(" nodegraph=\"{}\"", escape(graph)));
        if let Some(node) = &connection.node {
            attrs.push_str(&format!(" nodename=\"{}\"", escape(node)));
        }
    } else if let Some(node) = &connection.node {
        attrs.push_str(&format!(" nodename=\"{}\"", escape(node)));
    }
    if let Some(output) = &connection.output {
        attrs.push_str(&format!(" output=\"{}\"", escape(output)));
    }
    attrs
}

fn write_attributes(
    out: &mut String,
    map: &std::collections::BTreeMap<String, String>,
    omit: &[&str],
) {
    for (key, value) in map {
        if !omit.contains(&key.as_str()) {
            let _ = write!(out, " {}=\"{}\"", key, escape(value));
        }
    }
}

fn attributes(map: &std::collections::BTreeMap<String, String>) -> String {
    let mut out = String::new();
    write_attributes(&mut out, map, &[]);
    out
}

fn line(out: &mut String, indent: usize, content: &str) {
    out.push_str(&"  ".repeat(indent));
    out.push_str(content);
    out.push('\n');
}

fn escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}
