//! Path-bearing projection over the reflection kernel's inspector tree.

use crate::FieldPath;
use prism_reflect::{inspect, Access, InspectorNode, Reflect, ReflectRef, TypeRegistry};

/// An inspector tree with a stable field path attached to every addressable node.
#[derive(Clone, Debug, PartialEq)]
pub struct InspectorProjection {
    /// The projected root node.
    pub root: InspectorProjectionNode,
}

/// One node from `prism_reflect::inspect` plus its semantic path.
#[derive(Clone, Debug, PartialEq)]
pub struct InspectorProjectionNode {
    /// The path to this node. Set is `None` when the dependency's path grammar
    /// cannot address the underlying collection item.
    pub path: Option<FieldPath>,
    /// The original reflection inspector node and its metadata hints.
    pub node: InspectorNode,
    /// Child nodes with path information projected recursively.
    pub children: Vec<InspectorProjectionNode>,
}

/// Build a path-bearing tree by projecting the existing reflection inspector.
#[must_use]
pub fn project_inspector(root: &dyn Reflect, registry: &TypeRegistry) -> InspectorProjection {
    let inspected = inspect(root, registry);
    InspectorProjection {
        root: project_node(root, &inspected, Some(FieldPath::root())),
    }
}

fn project_node(
    value: &dyn Reflect,
    node: &InspectorNode,
    path: Option<FieldPath>,
) -> InspectorProjectionNode {
    let children = match value.reflect_ref() {
        ReflectRef::Struct(structure) => node
            .children
            .iter()
            .filter_map(|child| {
                structure.field(&child.label).map(|field| {
                    let child_path = path
                        .as_ref()
                        .and_then(|path| path.with_access(Access::Field(child.label.clone())).ok());
                    project_node(field, child, child_path)
                })
            })
            .collect(),
        ReflectRef::TupleStruct(tuple) => node
            .children
            .iter()
            .filter_map(|child| child.label.parse::<usize>().ok())
            .filter_map(|index| tuple.field(index).map(|field| (index, field)))
            .filter_map(|(index, field)| {
                node.children
                    .iter()
                    .find(|child| child.label == index.to_string())
                    .map(|child| {
                        let child_path = path
                            .as_ref()
                            .and_then(|path| path.with_access(Access::TupleIndex(index)).ok());
                        project_node(field, child, child_path)
                    })
            })
            .collect(),
        ReflectRef::Enum(enum_value) => node
            .children
            .iter()
            .filter_map(|child| child.label.parse::<usize>().ok())
            .filter_map(|index| enum_value.field_at(index).map(|field| (index, field)))
            .filter_map(|(index, field)| {
                node.children
                    .iter()
                    .find(|child| child.label == index.to_string())
                    .map(|child| {
                        let child_path = path
                            .as_ref()
                            .and_then(|path| path.with_access(Access::TupleIndex(index)).ok());
                        project_node(field, child, child_path)
                    })
            })
            .collect(),
        ReflectRef::List(list) => node
            .children
            .iter()
            .filter_map(|child| child.label.parse::<usize>().ok())
            .filter_map(|index| list.get(index).map(|field| (index, field)))
            .filter_map(|(index, field)| {
                node.children
                    .iter()
                    .find(|child| child.label == index.to_string())
                    .map(|child| {
                        let child_path = path
                            .as_ref()
                            .and_then(|path| path.with_access(Access::ListIndex(index)).ok());
                        project_node(field, child, child_path)
                    })
            })
            .collect(),
        ReflectRef::Array(array) => node
            .children
            .iter()
            .filter_map(|child| child.label.parse::<usize>().ok())
            .filter_map(|index| array.get(index).map(|field| (index, field)))
            .filter_map(|(index, field)| {
                node.children
                    .iter()
                    .find(|child| child.label == index.to_string())
                    .map(|child| {
                        let child_path = path
                            .as_ref()
                            .and_then(|path| path.with_access(Access::ListIndex(index)).ok());
                        project_node(field, child, child_path)
                    })
            })
            .collect(),
        ReflectRef::Map(map) => node
            .children
            .iter()
            .zip(map.iter_reflect())
            .map(|(child, (key, field))| {
                let child_path = key
                    .downcast_ref::<String>()
                    .and_then(|key| path.as_ref()?.with_access(Access::Key(key.clone())).ok());
                project_node(field, child, child_path)
            })
            .collect(),
        ReflectRef::Set(set) => node
            .children
            .iter()
            .zip(set.iter_reflect())
            .map(|(child, field)| project_node(field, child, None))
            .collect(),
        ReflectRef::Value(_) => Vec::new(),
        _ => Vec::new(),
    };

    InspectorProjectionNode {
        path,
        node: node.clone(),
        children,
    }
}

use std::string::String;
