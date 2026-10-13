//! MaterialX authoring frontend for Prism.
//!
//! This crate parses a documented MaterialX subset into a loss-aware authoring
//! IR and depends on Prism's runtime material crate only at the explicit
//! OpenPBR/Standard Surface adapter boundary. That adapter evaluates literals
//! and direct `constant` connections; arbitrary graph expressions and graph
//! interfaces are rejected instead of being treated as defaults.

#![allow(
    clippy::doc_markdown,
    clippy::std_instead_of_alloc,
    reason = "MaterialX/OpenPBR are domain names; the crate intentionally uses std collections"
)]

mod diagnostic;
mod evaluator;
mod export;
mod import;
mod ir;
mod parse;

pub use diagnostic::{AuthoringError, Diagnostic, DiagnosticCode, Severity, SourceLocation};
pub use evaluator::evaluate_connection;
pub use export::export_materialx;
pub use import::{
    adapt_openpbr_surface, import_openpbr_surface, AdapterReport, OpenPbrSurface, RuntimeSurface,
};
pub use ir::{Connection, Document, Input, Node, NodeDef, NodeGraph, Output, TypedValue};
pub use parse::parse_materialx;
