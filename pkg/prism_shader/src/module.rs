//! An authored shader module and its declared imports.
//!
//! A [`ShaderModule`] pairs a name with its source text and records the modules
//! it depends on. Dependencies are declared with import directives at the top
//! of the source:
//!
//! ```text
//! #import common/brdf
//! #include "lighting/pbr"
//! ```
//!
//! Both `#import` and `#include` are accepted and treated identically. The path
//! may optionally be wrapped in double quotes or angle brackets, which are
//! stripped. Import lines are recorded in [`ShaderModule::imports`] and removed
//! from the body returned by [`ShaderModule::body`], so the composer can splice
//! dependency bodies in resolved order without the directive lines leaking into
//! the final source.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// An authored shader module: a name, its source, and its declared imports.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ShaderModule {
    /// The module's unique registry name.
    name: String,
    /// The full authored source, including any import directive lines.
    source: String,
    /// The resolved import paths, in declaration order.
    imports: Vec<String>,
}

/// Extracts the import path from an import directive argument, if the line is
/// one. Returns `None` for non-import lines.
fn parse_import(line: &str) -> Option<&str> {
    let trimmed = line.trim_start();
    let rest = trimmed.strip_prefix('#')?;
    let rest = rest.trim_start();
    let path = rest
        .strip_prefix("import")
        .or_else(|| rest.strip_prefix("include"))?;
    // The keyword must be followed by whitespace, not be a prefix of a longer
    // identifier such as `importance`.
    if !path.starts_with(char::is_whitespace) {
        return None;
    }
    Some(strip_delimiters(path.trim()))
}

/// Strips a single matching pair of `"..."` or `<...>` delimiters from `path`.
fn strip_delimiters(path: &str) -> &str {
    if let Some(inner) = path
        .strip_prefix('"')
        .and_then(|rest| rest.strip_suffix('"'))
    {
        return inner;
    }
    if let Some(inner) = path
        .strip_prefix('<')
        .and_then(|rest| rest.strip_suffix('>'))
    {
        return inner;
    }
    path
}

impl ShaderModule {
    /// Creates a module from a name and source, parsing its import directives.
    ///
    /// Import lines whose extracted path is empty are ignored.
    #[must_use]
    pub fn new(name: impl Into<String>, source: impl Into<String>) -> Self {
        let name = name.into();
        let source = source.into();
        let mut imports = Vec::new();
        for line in source.lines() {
            if let Some(path) = parse_import(line)
                && !path.is_empty()
            {
                imports.push(path.to_string());
            }
        }
        Self {
            name,
            source,
            imports,
        }
    }

    /// The module's registry name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The full authored source, including import directive lines.
    #[must_use]
    pub fn source(&self) -> &str {
        &self.source
    }

    /// The declared import paths, in declaration order.
    #[must_use]
    pub fn imports(&self) -> &[String] {
        &self.imports
    }

    /// The source with its import directive lines removed.
    ///
    /// This is what the composer concatenates: the pure body, so import
    /// directives never reach the preprocessor or the backend.
    #[must_use]
    pub fn body(&self) -> String {
        let mut lines: Vec<&str> = Vec::new();
        for line in self.source.lines() {
            if parse_import(line).is_none() {
                lines.push(line);
            }
        }
        lines.join("\n")
    }
}
