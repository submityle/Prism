use core::fmt;

/// Severity of an authoring diagnostic.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Severity {
    /// The document cannot be safely interpreted or exported.
    Error,
    /// The document is usable, but a conversion has a known limitation.
    Warning,
}

/// Stable machine-readable diagnostic category.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DiagnosticCode {
    /// XML syntax or root element is invalid.
    InvalidXml,
    /// An element is outside the supported MaterialX subset.
    UnsupportedElement,
    /// A required attribute is absent or invalid.
    InvalidStructure,
    /// An identifier is duplicated or a reference cannot be resolved.
    InvalidReference,
    /// A typed literal could not be parsed.
    InvalidValue,
    /// The selected node is not a supported OpenPBR surface.
    UnsupportedMaterial,
    /// A valid OpenPBR field is not represented by the first-stage adapter.
    UnsupportedField,
    /// A connection points at a graph expression that this restricted
    /// adapter cannot evaluate without baking or executing MaterialX.
    UnsafeConnection,
}

/// One diagnostic with an optional XML source location.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Diagnostic {
    /// Stable category.
    pub code: DiagnosticCode,
    /// Severity.
    pub severity: Severity,
    /// Human-readable explanation.
    pub message: String,
    /// XML location when available.
    pub location: Option<SourceLocation>,
}

/// One-based XML source location.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SourceLocation {
    /// One-based line.
    pub line: u32,
    /// One-based column.
    pub column: u32,
}

/// A fatal frontend error carrying a precise diagnostic.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AuthoringError(pub Diagnostic);

impl fmt::Display for AuthoringError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(location) = self.0.location {
            write!(
                f,
                "{} at {}:{}: {}",
                code_name(self.0.code),
                location.line,
                location.column,
                self.0.message
            )
        } else {
            write!(f, "{}: {}", code_name(self.0.code), self.0.message)
        }
    }
}

impl std::error::Error for AuthoringError {}

impl AuthoringError {
    pub(crate) fn new(
        code: DiagnosticCode,
        message: impl Into<String>,
        location: Option<SourceLocation>,
    ) -> Self {
        Self(Diagnostic {
            code,
            severity: Severity::Error,
            message: message.into(),
            location,
        })
    }
}

pub(crate) fn location(
    document: &roxmltree::Document<'_>,
    node: roxmltree::Node<'_, '_>,
) -> Option<SourceLocation> {
    let pos = document.text_pos_at(node.range().start);
    Some(SourceLocation {
        line: pos.row,
        column: pos.col,
    })
}

fn code_name(code: DiagnosticCode) -> &'static str {
    match code {
        DiagnosticCode::InvalidXml => "invalid_xml",
        DiagnosticCode::UnsupportedElement => "unsupported_element",
        DiagnosticCode::InvalidStructure => "invalid_structure",
        DiagnosticCode::InvalidReference => "invalid_reference",
        DiagnosticCode::InvalidValue => "invalid_value",
        DiagnosticCode::UnsupportedMaterial => "unsupported_material",
        DiagnosticCode::UnsupportedField => "unsupported_field",
        DiagnosticCode::UnsafeConnection => "unsafe_connection",
    }
}
