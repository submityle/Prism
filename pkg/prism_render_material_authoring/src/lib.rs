//! MaterialX authoring frontend for Prism.
//!
//! This crate parses a documented MaterialX subset into a loss-aware authoring
//! IR and depends on Prism's runtime material crate only at the explicit
//! OpenPBR/Standard Surface adapter boundary. That adapter evaluates literals,
//! graph interfaces, graph outputs, and a bounded pure numeric expression
//! subset. The bounded procedural subset can also be lowered into the runtime
//! material IR and baked into the production material-VT source contract;
//! unsupported nodes remain explicit errors instead of being treated as
//! defaults.

#![allow(
    clippy::doc_markdown,
    clippy::std_instead_of_alloc,
    reason = "MaterialX/OpenPBR are domain names; the crate intentionally uses std collections"
)]

mod archetype;
mod artifact;
mod author_node;
mod diagnostic;
mod evaluator;
mod export;
mod import;
mod ir;
mod parse;
mod procedural;
mod reference;

pub use archetype::MaterialArchetype;
pub use artifact::{
    MaterialArtifact, MaterialArtifactError, MaterialArtifactRequest, MaterialRuntimeResponse,
    MaterialVtAuthoringBinding, MaterialVtPublication,
};
pub use author_node::{author_node_identity, author_node_manifest, AuthorNodeIdentity};
pub use diagnostic::{AuthoringError, Diagnostic, DiagnosticCode, Severity, SourceLocation};
pub use evaluator::evaluate_connection;
pub use export::export_materialx;
pub use import::{adapt_openpbr_surface, AdapterReport, RuntimeSurface, RuntimeTextureInput};
pub use ir::{
    value_type_kind, Connection, Document, DocumentElement, Input, Node, NodeDef, NodeGraph,
    Output, TypedValue, ValueTypeKind,
};
pub use parse::parse_materialx;
pub use procedural::{bake_procedural_graph, lower_procedural_graph, ProceduralGraphError};
pub use reference::{
    reference_material, reference_material_from_parameters,
    reference_material_from_parameters_with_context, reference_material_with_context,
    ReferenceMaterial, ReferenceMaterialError, ReferenceShadeContext,
};
