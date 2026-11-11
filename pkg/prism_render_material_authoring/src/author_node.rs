//! MaterialX `nodedef` to PCM `author_node` binding and validation.

use std::sync::OnceLock;

use prism_material_pcm::{
    builtin::{builtin_author_node, AuthorNodeBuiltin},
    ExtensionManifest,
};
use prism_material_schema::{load_surface_schema, IdentityExtension, IdentityKind, SurfaceSchema};

use crate::{AuthoringError, DiagnosticCode, Document, Node, NodeDef};

const NODEDEF_ATTRIBUTE: &str = "nodedef";
static MATERIAL_SCHEMA: OnceLock<Result<SurfaceSchema, String>> = OnceLock::new();

/// The stable semantic binding of a direct `nodedef` author node.
///
/// MaterialX category spellings are authoring syntax. The canonical category
/// and PCM UID are the identity consumed by the bake/evaluator boundary; an
/// alias such as `plus` therefore cannot create a second runtime identity.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AuthorNodeIdentity {
    /// The referenced MaterialX `nodedef` name.
    pub nodedef: String,
    /// Category written by the definition (`plus`, `add`, and so on).
    pub source_category: String,
    /// Canonical category used by the PCM catalogue.
    pub canonical_category: String,
    /// Stable identity-only PCM UID.
    pub pcm_uid: String,
}

/// Return the stable PCM author-node manifest selected by a node instance.
///
/// `None` means the node is either a built-in MaterialX category without a
/// `nodedef` binding, or its `nodedef` is implemented by a nodegraph. Graph
/// implementations retain their own semantics and must not be mislabeled as
/// the category's built-in operator. A directly evaluated definition is
/// resolved by binary search and must map to an executable pure-numeric
/// author-node category.
pub fn author_node_manifest(
    document: &Document,
    node: &Node,
) -> Result<Option<&'static ExtensionManifest>, AuthoringError> {
    resolve(document, node).map(|binding| binding.map(|(_, builtin)| builtin.manifest()))
}

/// Resolve a node's direct author-node identity for artifact manifests.
///
/// Graph-backed definitions intentionally return `None`: their semantics are
/// the nested graph body and are lowered/evaluated as graph instances rather
/// than pretending to be one of the fixed PCM operator identities.
pub fn author_node_identity(
    document: &Document,
    node: &Node,
) -> Result<Option<AuthorNodeIdentity>, AuthoringError> {
    let Some((definition, builtin)) = resolve(document, node)? else {
        return Ok(None);
    };
    Ok(Some(AuthorNodeIdentity {
        nodedef: definition.name.clone(),
        source_category: definition.category.clone(),
        canonical_category: builtin.category().to_owned(),
        pcm_uid: builtin.manifest().uid().to_owned(),
    }))
}

/// Normalize the bounded MaterialX operator vocabulary to its stable PCM
/// identity. Aliases are authoring syntax only and must never create a second
/// runtime extension identity.
pub(crate) fn canonical_category(category: &str) -> Option<&'static str> {
    match category {
        "add" | "plus" => Some("add"),
        "clamp" => Some("clamp"),
        "convert" => Some("convert"),
        "divide" => Some("divide"),
        "mix" | "blend" => Some("mix"),
        "multiply" | "mult" => Some("multiply"),
        "subtract" => Some("subtract"),
        _ => None,
    }
}

pub(crate) fn resolve<'a>(
    document: &'a Document,
    node: &Node,
) -> Result<Option<(&'a NodeDef, &'static AuthorNodeBuiltin)>, AuthoringError> {
    let Some(name) = node.attributes.get(NODEDEF_ATTRIBUTE) else {
        return Ok(None);
    };
    if document
        .nodegraphs
        .iter()
        .any(|graph| graph.nodedef.as_deref() == Some(name.as_str()))
    {
        return Ok(None);
    }
    let definition = document
        .nodedefs
        .binary_search_by(|definition| definition.name.as_str().cmp(name.as_str()))
        .ok()
        .map(|index| &document.nodedefs[index])
        .ok_or_else(|| {
            error(
                DiagnosticCode::InvalidReference,
                format!("node '{}' references unknown nodedef '{name}'", node.name),
            )
        })?;
    let canonical = canonical_category(&definition.category).ok_or_else(|| {
        error(
            DiagnosticCode::UnsupportedMaterial,
            format!(
                "nodedef '{}' category '{}' has no executable PCM author_node",
                definition.name, definition.category
            ),
        )
    })?;
    let builtin = builtin_author_node(canonical).ok_or_else(|| {
        error(
            DiagnosticCode::InvalidReference,
            format!("author_node category '{canonical}' is absent from the PCM catalogue"),
        )
    })?;
    validate_schema_binding(canonical, builtin)?;
    Ok(Some((definition, builtin)))
}

pub(crate) fn validate_document(document: &Document) -> Result<(), AuthoringError> {
    for definition in &document.nodedefs {
        if let Some(parent) = definition.attributes.get("inherit") {
            return Err(error(
                DiagnosticCode::UnsupportedMaterial,
                format!(
                    "nodedef '{}' inherits '{parent}', but inherited author-node semantics are unsupported",
                    definition.name
                ),
            ));
        }
    }

    // Validate direct author-node definitions even when the document contains
    // no instance yet. Otherwise an invalid library nodedef would survive
    // import and fail only after a later graph references it.
    for definition in &document.nodedefs {
        if document
            .nodegraphs
            .iter()
            .any(|graph| graph.nodedef.as_deref() == Some(definition.name.as_str()))
        {
            continue;
        }
        let Some(canonical) = canonical_category(&definition.category) else {
            if matches!(
                definition.category.as_str(),
                "open_pbr_surface" | "standard_surface"
            ) {
                continue;
            }
            return Err(error(
                DiagnosticCode::UnsupportedMaterial,
                format!(
                    "nodedef '{}' category '{}' has no graph implementation or executable PCM author_node",
                    definition.name, definition.category
                ),
            ));
        };
        let builtin = builtin_author_node(canonical).ok_or_else(|| {
            error(
                DiagnosticCode::InvalidReference,
                format!("author_node category '{canonical}' is absent from the PCM catalogue"),
            )
        })?;
        validate_schema_binding(canonical, builtin)?;
        validate_operator_signature(definition)?;
    }

    for node in document
        .nodegraphs
        .iter()
        .flat_map(|graph| graph.nodes.iter())
    {
        if let Some(name) = node.attributes.get(NODEDEF_ATTRIBUTE)
            && let Some(implementation) = document
                .nodegraphs
                .iter()
                .find(|graph| graph.nodedef.as_deref() == Some(name.as_str()))
        {
            let definition = document
                .nodedefs
                .binary_search_by(|definition| definition.name.cmp(name))
                .ok()
                .map(|index| &document.nodedefs[index])
                .ok_or_else(|| {
                    error(
                        DiagnosticCode::InvalidReference,
                        format!("node '{}' references unknown nodedef '{name}'", node.name),
                    )
                })?;
            validate_graph_instance(node, definition, implementation)?;
            continue;
        }
        let Some((definition, _)) = resolve(document, node)? else {
            continue;
        };
        validate_instance(node, definition)?;
    }
    Ok(())
}

fn validate_graph_instance(
    node: &Node,
    definition: &NodeDef,
    implementation: &crate::NodeGraph,
) -> Result<(), AuthoringError> {
    let category_matches = node.category == definition.category
        || matches!(
            (
                canonical_category(&node.category),
                canonical_category(&definition.category)
            ),
            (Some(actual), Some(expected)) if actual == expected
        );
    if !node.generic_element && !category_matches {
        return Err(error(
            DiagnosticCode::InvalidStructure,
            format!(
                "node '{}' category '{}' does not match nodedef '{}' category '{}'",
                node.name, node.category, definition.name, definition.category
            ),
        ));
    }
    if node.value.is_some() {
        return Err(error(
            DiagnosticCode::InvalidStructure,
            format!(
                "nodegraph instance '{}' cannot carry a node-level value",
                node.name
            ),
        ));
    }

    for input in &node.inputs {
        if input.explicit_default().is_none()
            && input.connection.is_none()
            && !input.attributes.contains_key("interfacename")
        {
            return Err(error(
                DiagnosticCode::InvalidStructure,
                format!(
                    "node '{}' input '{}' has no value or connection",
                    node.name, input.name
                ),
            ));
        }
        let declared = implementation
            .inputs
            .iter()
            .find(|candidate| candidate.name == input.name)
            .ok_or_else(|| {
                error(
                    DiagnosticCode::InvalidStructure,
                    format!(
                        "node '{}' input '{}' is not declared by nodegraph '{}'",
                        node.name, input.name, implementation.name
                    ),
                )
            })?;
        let definition_declared = definition
            .inputs
            .iter()
            .find(|candidate| candidate.name == input.name)
            .ok_or_else(|| {
                error(
                    DiagnosticCode::InvalidStructure,
                    format!(
                        "node '{}' input '{}' is not declared by nodedef '{}'",
                        node.name, input.name, definition.name
                    ),
                )
            })?;
        if let (Some(actual), Some(expected)) = (
            input.type_name.as_deref(),
            definition_declared.type_name.as_deref(),
        ) {
            require_type_match(
                &format!("node '{}' input '{}'", node.name, input.name),
                actual,
                expected,
            )?;
        }
        if let (Some(actual), Some(expected)) = (
            declared.type_name.as_deref(),
            definition_declared.type_name.as_deref(),
        ) {
            require_type_match(
                &format!(
                    "nodegraph '{}' interface '{}'",
                    implementation.name, declared.name
                ),
                actual,
                expected,
            )?;
        }
    }

    // An instance may omit an input only when the implementation supplies a
    // concrete interface value/connection or the definition supplies a
    // default. Anything else would become an evaluator-time implicit default
    // and would make the round trip lossy.
    for declared in &definition.inputs {
        if node.inputs.iter().any(|input| input.name == declared.name) {
            continue;
        }
        let implementation_input = implementation
            .inputs
            .iter()
            .find(|input| input.name == declared.name);
        // An `interfacename` on the implementation only forwards the value;
        // it is not a value for an instance that omitted the parameter.
        let has_implementation_binding = implementation_input
            .is_some_and(|input| input.explicit_default().is_some() || input.connection.is_some());
        if declared.explicit_default().is_none() && !has_implementation_binding {
            return Err(error(
                DiagnosticCode::InvalidStructure,
                format!(
                    "node '{}' omits unbound external parameter '{}' of nodedef '{}'",
                    node.name, declared.name, definition.name
                ),
            ));
        }
    }

    for output in &node.outputs {
        let declared = definition
            .outputs
            .iter()
            .find(|candidate| candidate.name == output.name)
            .ok_or_else(|| {
                error(
                    DiagnosticCode::InvalidReference,
                    format!(
                        "node '{}' output '{}' is not declared by nodedef '{}'",
                        node.name, output.name, definition.name
                    ),
                )
            })?;
        if let (Some(actual), Some(expected)) =
            (output.type_name.as_deref(), declared.type_name.as_deref())
        {
            require_type_match(
                &format!("node '{}' output '{}'", node.name, output.name),
                actual,
                expected,
            )?;
        }
    }
    Ok(())
}

fn validate_instance(node: &Node, definition: &NodeDef) -> Result<(), AuthoringError> {
    if node.category != definition.category {
        return Err(error(
            DiagnosticCode::InvalidStructure,
            format!(
                "node '{}' category '{}' does not match nodedef '{}' category '{}'",
                node.name, node.category, definition.name, definition.category
            ),
        ));
    }
    if node.value.is_some() {
        return Err(error(
            DiagnosticCode::InvalidStructure,
            format!(
                "author node '{}' cannot carry a node-level value",
                node.name
            ),
        ));
    }
    if definition.outputs.is_empty() {
        return Err(error(
            DiagnosticCode::InvalidStructure,
            format!("nodedef '{}' requires at least one output", definition.name),
        ));
    }
    validate_operator_signature(definition)?;

    for input in &node.inputs {
        if input.explicit_default().is_none()
            && input.connection.is_none()
            && !input.attributes.contains_key("interfacename")
        {
            return Err(error(
                DiagnosticCode::InvalidStructure,
                format!(
                    "node '{}' input '{}' has no value or connection",
                    node.name, input.name
                ),
            ));
        }
        let declared = definition
            .inputs
            .iter()
            .find(|candidate| candidate.name == input.name)
            .ok_or_else(|| {
                error(
                    DiagnosticCode::InvalidStructure,
                    format!(
                        "node '{}' input '{}' is not declared by nodedef '{}'",
                        node.name, input.name, definition.name
                    ),
                )
            })?;
        if let Some(actual) = input.type_name.as_deref() {
            require_type_match(
                &format!("node '{}' input '{}'", node.name, input.name),
                actual,
                declared.type_name.as_deref().expect("nodedef input type"),
            )?;
        }
    }
    for declared in &definition.inputs {
        if !node.inputs.iter().any(|input| input.name == declared.name)
            && declared.explicit_default().is_none()
        {
            return Err(error(
                DiagnosticCode::InvalidStructure,
                format!(
                    "node '{}' omits required nodedef input '{}'",
                    node.name, declared.name
                ),
            ));
        }
    }

    for output in &node.outputs {
        let declared = definition
            .outputs
            .iter()
            .find(|candidate| candidate.name == output.name)
            .ok_or_else(|| {
                error(
                    DiagnosticCode::InvalidReference,
                    format!(
                        "node '{}' output '{}' is not declared by nodedef '{}'",
                        node.name, output.name, definition.name
                    ),
                )
            })?;
        if let Some(actual) = output.type_name.as_deref() {
            require_type_match(
                &format!("node '{}' output '{}'", node.name, output.name),
                actual,
                declared.type_name.as_deref().expect("nodedef output type"),
            )?;
        }
    }
    Ok(())
}

fn validate_operator_signature(definition: &NodeDef) -> Result<(), AuthoringError> {
    let category = canonical_category(&definition.category).ok_or_else(|| {
        error(
            DiagnosticCode::UnsupportedMaterial,
            format!(
                "nodedef '{}' category '{}' has no bounded author-node contract",
                definition.name, definition.category
            ),
        )
    })?;
    let expected_inputs = match category {
        "convert" => 1,
        "add" | "subtract" | "multiply" | "divide" => 2,
        "mix" | "clamp" => 3,
        _ => {
            return Err(error(
                DiagnosticCode::UnsupportedMaterial,
                format!(
                    "nodedef '{}' category '{}' has no bounded author-node contract",
                    definition.name, definition.category
                ),
            ))
        }
    };
    if definition.inputs.len() != expected_inputs {
        return Err(error(
            DiagnosticCode::InvalidStructure,
            format!(
                "nodedef '{}' category '{}' requires {expected_inputs} inputs, found {}",
                definition.name,
                definition.category,
                definition.inputs.len()
            ),
        ));
    }
    if definition.outputs.len() != 1 {
        return Err(error(
            DiagnosticCode::InvalidStructure,
            format!(
                "nodedef '{}' category '{}' requires exactly one output, found {}",
                definition.name,
                definition.category,
                definition.outputs.len()
            ),
        ));
    }

    let input_type = |index: usize| {
        definition.inputs[index]
            .type_name
            .as_deref()
            .expect("nodedef input type")
    };
    let output_type = definition.outputs[0]
        .type_name
        .as_deref()
        .expect("nodedef output type");
    match category {
        "convert" => {
            let input = input_type(0);
            let input_width = numeric_width(input);
            let output_width = numeric_width(output_type);
            if input_width.is_none() || output_width.is_none() || input_width != output_width {
                return Err(error(
                    DiagnosticCode::InvalidValue,
                    format!(
                        "convert nodedef '{}' requires supported numeric input/output types with the same dimension, found '{input}' -> '{output_type}'",
                        definition.name
                    ),
                ));
            }
        }
        "add" | "subtract" | "multiply" | "divide" => {
            require_type_match("binary input types", input_type(1), input_type(0))?;
            require_type_match("binary output type", output_type, input_type(0))?;
        }
        "mix" => {
            require_type_match("mix value types", input_type(1), input_type(0))?;
            require_type_match("mix amount type", input_type(2), "float")?;
            require_type_match("mix output type", output_type, input_type(0))?;
        }
        "clamp" => {
            require_type_match("clamp low type", input_type(1), "float")?;
            require_type_match("clamp high type", input_type(2), "float")?;
            require_type_match("clamp output type", output_type, input_type(0))?;
        }
        _ => {
            return Err(error(
                DiagnosticCode::UnsupportedMaterial,
                format!(
                    "nodedef '{}' category '{}' has no bounded author-node contract",
                    definition.name, definition.category
                ),
            ))
        }
    }
    Ok(())
}

fn validate_schema_binding(
    category: &str,
    builtin: &'static AuthorNodeBuiltin,
) -> Result<(), AuthoringError> {
    let schema_entry = author_node_schema_entry(category)?;
    let manifest = builtin.manifest();
    if schema_entry.uid != manifest.uid()
        || schema_entry.kind.as_str() != manifest.kind().as_str()
        || schema_entry.namespace.as_str() != manifest.namespace().as_str()
        || schema_entry.vendor != manifest.vendor()
        || schema_entry.stability.as_str() != manifest.stability().as_str()
    {
        return Err(error(
            DiagnosticCode::InvalidReference,
            format!("author_node '{category}' disagrees with the schema/PCM identity contract"),
        ));
    }
    Ok(())
}

fn author_node_schema_entry(category: &str) -> Result<IdentityExtension, AuthoringError> {
    let schema = MATERIAL_SCHEMA
        .get_or_init(|| load_surface_schema().map_err(|schema_error| schema_error.to_string()));
    let schema = schema.as_ref().map_err(|schema_error| {
        error(
            DiagnosticCode::InvalidReference,
            format!("embedded material schema is invalid: {schema_error}"),
        )
    })?;
    schema
        .identity_extensions
        .iter()
        .find(|extension| extension.kind == IdentityKind::AuthorNode && extension.id == category)
        .cloned()
        .ok_or_else(|| {
            error(
                DiagnosticCode::InvalidReference,
                format!("author_node category '{category}' has no schema identity extension"),
            )
        })
}

pub(crate) fn numeric_width(type_name: &str) -> Option<usize> {
    match type_name {
        "float" => Some(1),
        "vector2" => Some(2),
        "color3" | "vector3" => Some(3),
        "color4" | "vector4" => Some(4),
        _ => None,
    }
}

fn require_type_match(context: &str, actual: &str, expected: &str) -> Result<(), AuthoringError> {
    if actual == expected {
        return Ok(());
    }
    Err(error(
        DiagnosticCode::InvalidValue,
        format!("{context} has type '{actual}', expected '{expected}'"),
    ))
}

fn error(code: DiagnosticCode, message: impl Into<String>) -> AuthoringError {
    AuthoringError::new(code, message, None)
}
