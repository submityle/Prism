use std::collections::{BTreeMap, BTreeSet};

use roxmltree::{Document as XmlDocument, Node as XmlNode};

use crate::{
    diagnostic::{location, AuthoringError, DiagnosticCode},
    Connection, Document, Input, Node, NodeDef, NodeGraph, Output, TypedValue,
};

const ROOT_ATTRIBUTES: &[&str] = &[
    "version",
    "xsi",
    "xmlns",
    "xmlns:xsi",
    "xsi:noNamespaceSchemaLocation",
];
const INPUT_ATTRIBUTES: &[&str] = &[
    "name",
    "type",
    "value",
    "nodename",
    "nodegraph",
    "output",
    "interfacename",
    "enum",
    "unit",
    "colorspace",
];
const NODE_ATTRIBUTES: &[&str] = &["name", "type", "value", "nodedef", "doc", "namespace"];

/// Parses the explicitly supported MaterialX subset.
///
/// Supported structure: `materialx`, `nodedef` with input/output declarations,
/// `nodegraph` with interface inputs, node/input/output children, literal input
/// values, and `nodename`/`nodegraph` output connections. Unknown elements and
/// node categories are errors; callers must not assume omitted data was
/// preserved.
pub fn parse_materialx(source: &str) -> Result<Document, AuthoringError> {
    let xml = XmlDocument::parse(source).map_err(|error| {
        AuthoringError::new(
            DiagnosticCode::InvalidXml,
            error.to_string(),
            Some(crate::SourceLocation {
                line: error.pos().row,
                column: error.pos().col,
            }),
        )
    })?;
    let root = xml.root_element();
    if root.tag_name().name() != "materialx" {
        return Err(fail(
            &xml,
            root,
            DiagnosticCode::InvalidStructure,
            "root element must be <materialx>",
        ));
    }
    let mut result = Document {
        attributes: attrs(&xml, root, ROOT_ATTRIBUTES)?,
        ..Document::default()
    };
    // MaterialX identifiers have two distinct scopes here: nodedef/nodegraph
    // names are document-scoped, while node names are scoped to their graph.
    // Keeping these sets separate is required for legal same-named nodes in
    // different graphs.
    let mut nodedef_ids = BTreeSet::new();
    let mut nodegraph_ids = BTreeSet::new();
    for child in root.children().filter(XmlNode::is_element) {
        match child.tag_name().name() {
            "nodedef" => {
                let def = parse_nodedef(&xml, child)?;
                insert_id(&xml, &mut nodedef_ids, &def.name, child)?;
                result.nodedefs.push(def);
            }
            "nodegraph" => {
                let graph = parse_nodegraph(&xml, child, &mut nodegraph_ids)?;
                result.nodegraphs.push(graph);
            }
            _ => {
                return Err(fail(
                    &xml,
                    child,
                    DiagnosticCode::UnsupportedElement,
                    format!(
                        "unsupported <{}> under <materialx>",
                        child.tag_name().name()
                    ),
                ))
            }
        }
    }
    result.nodedefs.sort_by(|a, b| a.name.cmp(&b.name));
    result.nodegraphs.sort_by(|a, b| a.name.cmp(&b.name));
    validate_references(&xml, &result)?;
    Ok(result)
}

fn parse_nodedef(
    xml: &XmlDocument<'_>,
    element: XmlNode<'_, '_>,
) -> Result<NodeDef, AuthoringError> {
    let name = required(xml, element, "name")?;
    let category = element.attribute("node").unwrap_or("").to_owned();
    if category.is_empty() {
        return Err(fail(
            xml,
            element,
            DiagnosticCode::InvalidStructure,
            "<nodedef> requires non-empty node attribute",
        ));
    }
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    let mut input_ids = BTreeSet::new();
    let mut output_ids = BTreeSet::new();
    for child in element.children().filter(XmlNode::is_element) {
        match child.tag_name().name() {
            "input" => {
                let input = parse_input(xml, child, true)?;
                insert_id(xml, &mut input_ids, &input.name, child)?;
                inputs.push(input);
            }
            "output" => {
                let output = parse_output(xml, child, true)?;
                insert_id(xml, &mut output_ids, &output.name, child)?;
                outputs.push(output);
            }
            _ => {
                return Err(fail(
                    xml,
                    child,
                    DiagnosticCode::UnsupportedElement,
                    format!("unsupported <{}> in nodedef", child.tag_name().name()),
                ))
            }
        }
    }
    Ok(NodeDef {
        name,
        category,
        inputs,
        outputs,
        attributes: attrs(
            xml,
            element,
            &[
                "name",
                "node",
                "version",
                "isdefaultversion",
                "inherit",
                "doc",
                "namespace",
            ],
        )?,
    })
}

fn parse_nodegraph(
    xml: &XmlDocument<'_>,
    element: XmlNode<'_, '_>,
    graph_ids: &mut BTreeSet<String>,
) -> Result<NodeGraph, AuthoringError> {
    let name = required(xml, element, "name")?;
    insert_id(xml, graph_ids, &name, element)?;
    let mut interface_ids = BTreeSet::new();
    let mut node_ids = BTreeSet::new();
    let mut inputs = Vec::new();
    let mut nodes = Vec::new();
    let mut outputs = Vec::new();
    for child in element.children().filter(XmlNode::is_element) {
        match child.tag_name().name() {
            "input" => {
                let input = parse_input(xml, child, false)?;
                insert_id(xml, &mut interface_ids, &input.name, child)?;
                inputs.push(input);
            }
            "node" => {
                let node = parse_node(xml, child)?;
                insert_id(xml, &mut node_ids, &node.name, child)?;
                nodes.push(node);
            }
            "output" => outputs.push(parse_output(xml, child, false)?),
            _category if child.attribute("name").is_some() => {
                let node = parse_node(xml, child)?;
                insert_id(xml, &mut node_ids, &node.name, child)?;
                nodes.push(node);
            }
            _ => {
                return Err(fail(
                    xml,
                    child,
                    DiagnosticCode::UnsupportedElement,
                    format!("unsupported <{}> in nodegraph", child.tag_name().name()),
                ))
            }
        }
    }
    Ok(NodeGraph {
        name,
        inputs,
        nodes,
        outputs,
        attributes: attrs(xml, element, &["name", "doc", "namespace"])?,
    })
}

fn parse_node(xml: &XmlDocument<'_>, element: XmlNode<'_, '_>) -> Result<Node, AuthoringError> {
    let name = required(xml, element, "name")?;
    let category = if element.tag_name().name() == "node" {
        element.attribute("type").unwrap_or("").to_owned()
    } else {
        element.tag_name().name().to_owned()
    };
    if category.is_empty() {
        return Err(fail(
            xml,
            element,
            DiagnosticCode::InvalidStructure,
            "<node> requires non-empty type attribute",
        ));
    }
    // A strict allow-list guarantees unsupported node semantics cannot disappear during round-trip.
    const SUPPORTED: &[&str] = &[
        "constant",
        "add",
        "subtract",
        "multiply",
        "divide",
        "mix",
        "convert",
        "clamp",
        "open_pbr_surface",
        "standard_surface",
        "image",
        "texcoord",
        "normalmap",
        "dotproduct",
        "ifequal",
        "switch",
    ];
    if !SUPPORTED.contains(&category.as_str()) {
        return Err(fail(
            xml,
            element,
            DiagnosticCode::UnsupportedMaterial,
            format!("unsupported node category '{category}' on node '{name}'"),
        ));
    }
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    for child in element.children().filter(XmlNode::is_element) {
        match child.tag_name().name() {
            "input" => inputs.push(parse_input(xml, child, false)?),
            "output" => outputs.push(parse_output(xml, child, false)?),
            _ => {
                return Err(fail(
                    xml,
                    child,
                    DiagnosticCode::UnsupportedElement,
                    format!("unsupported <{}> in node '{name}'", child.tag_name().name()),
                ))
            }
        }
    }
    let value = element.attribute("value").map(|value| TypedValue {
        type_name: element.attribute("type").unwrap_or("string").to_owned(),
        value: canonical_value(value),
    });
    Ok(Node {
        name,
        category,
        value,
        inputs,
        outputs,
        attributes: attrs(xml, element, NODE_ATTRIBUTES)?,
    })
}

fn parse_input(
    xml: &XmlDocument<'_>,
    element: XmlNode<'_, '_>,
    declaration: bool,
) -> Result<Input, AuthoringError> {
    let name = required(xml, element, "name")?;
    let type_name = element.attribute("type").map(str::to_owned);
    if declaration && type_name.is_none() {
        return Err(fail(
            xml,
            element,
            DiagnosticCode::InvalidStructure,
            format!("nodedef input '{name}' requires a type"),
        ));
    }
    let literal = element.attribute("value");
    let node = element.attribute("nodename");
    let graph = element.attribute("nodegraph");
    let interface = element.attribute("interfacename");
    if literal.is_some() && (node.is_some() || graph.is_some()) {
        return Err(fail(
            xml,
            element,
            DiagnosticCode::InvalidStructure,
            format!("input '{name}' cannot contain both a literal and a connection"),
        ));
    }
    if interface.is_some() && (literal.is_some() || node.is_some() || graph.is_some()) {
        return Err(fail(
            xml,
            element,
            DiagnosticCode::InvalidStructure,
            format!("input '{name}' cannot contain interfacename with a literal or connection"),
        ));
    }
    if interface.is_some_and(str::is_empty) {
        return Err(fail(
            xml,
            element,
            DiagnosticCode::InvalidStructure,
            format!("input '{name}' requires a non-empty interfacename"),
        ));
    }
    let value = literal.map(|value| TypedValue {
        type_name: type_name.clone().unwrap_or_else(|| "string".into()),
        value: canonical_value(value),
    });
    let connection = node.or(graph).map(|_| Connection {
        node: node.map(str::to_owned),
        output: element.attribute("output").map(str::to_owned),
        nodegraph: graph.map(str::to_owned),
    });
    Ok(Input {
        name,
        type_name,
        value,
        connection,
        attributes: attrs(xml, element, INPUT_ATTRIBUTES)?,
    })
}

fn parse_output(
    xml: &XmlDocument<'_>,
    element: XmlNode<'_, '_>,
    declaration: bool,
) -> Result<Output, AuthoringError> {
    let name = required(xml, element, "name")?;
    let type_name = element.attribute("type").map(str::to_owned);
    if declaration && type_name.is_none() {
        return Err(fail(
            xml,
            element,
            DiagnosticCode::InvalidStructure,
            format!("nodedef output '{name}' requires a type"),
        ));
    }
    let value = element.attribute("value").map(|value| TypedValue {
        type_name: type_name.clone().unwrap_or_else(|| "string".into()),
        value: canonical_value(value),
    });
    let node = element.attribute("nodename");
    let graph = element.attribute("nodegraph");
    if !declaration && value.is_some() {
        return Err(fail(
            xml,
            element,
            DiagnosticCode::InvalidStructure,
            "nodegraph output cannot carry a literal value",
        ));
    }
    if declaration && (node.is_some() || graph.is_some()) {
        return Err(fail(
            xml,
            element,
            DiagnosticCode::InvalidStructure,
            "nodedef output cannot carry a connection",
        ));
    }
    let connection = node.or(graph).map(|_| Connection {
        node: node.map(str::to_owned),
        output: element.attribute("output").map(str::to_owned),
        nodegraph: graph.map(str::to_owned),
    });
    Ok(Output {
        name,
        type_name,
        connection,
        value,
    })
}

fn validate_references(xml: &XmlDocument<'_>, doc: &Document) -> Result<(), AuthoringError> {
    let graphs: BTreeMap<_, _> = doc
        .nodegraphs
        .iter()
        .map(|g| (g.name.as_str(), g))
        .collect();
    for graph in &doc.nodegraphs {
        let nodes: BTreeSet<_> = graph.nodes.iter().map(|n| n.name.as_str()).collect();
        for input in graph
            .inputs
            .iter()
            .chain(graph.nodes.iter().flat_map(|n| &n.inputs))
        {
            if let Some(interface) = input.attributes.get("interfacename")
                && !graph
                    .inputs
                    .iter()
                    .any(|candidate| candidate.name == *interface)
            {
                return Err(AuthoringError::new(
                    DiagnosticCode::InvalidReference,
                    format!(
                        "interface input '{interface}' does not exist in nodegraph '{}'",
                        graph.name
                    ),
                    None,
                ));
            }
            if let Some(connection) = input.connection.as_ref() {
                validate_connection(&graphs, graph, &nodes, connection)?;
            }
        }
        for output in &graph.outputs {
            if let Some(connection) = output.connection.as_ref() {
                validate_connection(&graphs, graph, &nodes, connection)?;
            }
        }
    }
    let _ = xml;
    Ok(())
}

fn validate_connection(
    graphs: &BTreeMap<&str, &NodeGraph>,
    graph: &NodeGraph,
    nodes: &BTreeSet<&str>,
    connection: &Connection,
) -> Result<(), AuthoringError> {
    if let Some(graph_name) = connection.nodegraph.as_deref() {
        let Some(target_graph) = graphs.get(graph_name) else {
            return Err(AuthoringError::new(
                DiagnosticCode::InvalidReference,
                format!("nodegraph reference '{graph_name}' does not exist"),
                None,
            ));
        };
        if let Some(node_name) = connection.node.as_deref() {
            if !target_graph.nodes.iter().any(|node| node.name == node_name) {
                return Err(AuthoringError::new(
                    DiagnosticCode::InvalidReference,
                    format!(
                        "node reference '{node_name}' does not exist in nodegraph '{graph_name}'"
                    ),
                    None,
                ));
            }
        } else if let Some(output) = connection.output.as_deref()
            && !target_graph.outputs.iter().any(|item| item.name == output)
        {
            return Err(AuthoringError::new(
                DiagnosticCode::InvalidReference,
                format!("output '{output}' does not exist on nodegraph '{graph_name}'"),
                None,
            ));
        }
    } else if connection
        .node
        .as_deref()
        .is_none_or(|name| !nodes.contains(name))
    {
        return Err(AuthoringError::new(
            DiagnosticCode::InvalidReference,
            format!(
                "node reference '{}' does not exist in nodegraph '{}'",
                connection.node.as_deref().unwrap_or("<missing>"),
                graph.name
            ),
            None,
        ));
    }
    Ok(())
}

fn attrs(
    xml: &XmlDocument<'_>,
    node: XmlNode<'_, '_>,
    allowed: &[&str],
) -> Result<BTreeMap<String, String>, AuthoringError> {
    let mut result = BTreeMap::new();
    for attribute in node.attributes() {
        let name = attribute.name();
        if !allowed.contains(&name) && !name.starts_with("xmlns") && !name.starts_with("xsi:") {
            return Err(fail(
                xml,
                node,
                DiagnosticCode::UnsupportedElement,
                format!(
                    "unsupported attribute '{name}' on <{}>",
                    node.tag_name().name()
                ),
            ));
        }
        if !matches!(
            name,
            "name" | "type" | "value" | "node" | "nodename" | "nodegraph" | "output"
        ) {
            result.insert(name.to_owned(), attribute.value().to_owned());
        }
    }
    Ok(result)
}

fn required(
    xml: &XmlDocument<'_>,
    node: XmlNode<'_, '_>,
    key: &str,
) -> Result<String, AuthoringError> {
    node.attribute(key)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| {
            fail(
                xml,
                node,
                DiagnosticCode::InvalidStructure,
                format!("<{}> requires non-empty '{key}'", node.tag_name().name()),
            )
        })
}
fn insert_id(
    xml: &XmlDocument<'_>,
    ids: &mut BTreeSet<String>,
    name: &str,
    node: XmlNode<'_, '_>,
) -> Result<(), AuthoringError> {
    if !ids.insert(name.to_owned()) {
        return Err(fail(
            xml,
            node,
            DiagnosticCode::InvalidReference,
            format!("duplicate document identifier '{name}'"),
        ));
    }
    Ok(())
}
fn fail(
    xml: &XmlDocument<'_>,
    node: XmlNode<'_, '_>,
    code: DiagnosticCode,
    message: impl Into<String>,
) -> AuthoringError {
    AuthoringError::new(code, message, location(xml, node))
}
fn canonical_value(value: &str) -> String {
    value
        .split(|character: char| character == ',' || character.is_whitespace())
        .filter(|token| !token.is_empty())
        .collect::<Vec<_>>()
        .join(", ")
}
