//! The end-to-end render pipeline.
//!
//! [`render`] ties the three stages together: it [`negotiate`]s a schema
//! version, [`sanitize`](crate::Sandbox::sanitize)s the untrusted tree against
//! the sandbox, and [`decode`]s the result into a [`prism_ui::Element`]. An
//! unsupported version short-circuits to a single fallback element without
//! touching the untrusted tree at all.

use alloc::vec::Vec;

use prism_ui::Element;

use crate::decode::{decode, fallback_element};
use crate::sandbox::{Diagnostic, Sandbox};
use crate::schema::RemoteDocument;
use crate::version::{negotiate, Resolution, VersionSet};

/// The product of [`render`]: the negotiated resolution, the decoded element
/// and the diagnostics gathered while sanitizing.
#[derive(Clone, Debug)]
pub struct RenderOutcome {
    /// How client and server schema versions were reconciled.
    pub resolution: Resolution,
    /// The decoded, render-ready element tree.
    pub element: Element,
    /// Every rewrite the sandbox performed while sanitizing.
    pub diagnostics: Vec<Diagnostic>,
}

/// Renders an untrusted [`RemoteDocument`] into a safe [`prism_ui::Element`].
///
/// The steps are, in order:
///
/// 1. negotiate the document's declared version against `client`,
/// 2. if unrenderable, return a fallback element immediately,
/// 3. otherwise sanitize the tree against `sandbox` and decode the result.
///
/// # Examples
///
/// ```
/// use prism_ui_sdui::{render, CapabilitySet, RemoteDocument, RemoteNode, Sandbox, Version, VersionSet};
///
/// let mut client = VersionSet::new();
/// client.insert(Version::new(1, 0));
/// let sandbox = Sandbox::new(CapabilitySet::new().allow_kinds(["box", "text"]));
///
/// let doc = RemoteDocument::new(
///     Version::new(1, 0),
///     RemoteNode::new("box").child(RemoteNode::text("hi")),
/// );
/// let outcome = render(&client, &sandbox, &doc);
/// assert!(outcome.diagnostics.is_empty());
/// assert_eq!(outcome.element.child_elements()[0].text_content(), Some("hi"));
/// ```
#[must_use]
pub fn render(client: &VersionSet, sandbox: &Sandbox, document: &RemoteDocument) -> RenderOutcome {
    let resolution = negotiate(client, document.version());
    if !resolution.is_renderable() {
        return RenderOutcome {
            resolution,
            element: fallback_element(Some("unsupported schema version")),
            diagnostics: Vec::new(),
        };
    }

    let sanitized = sandbox.sanitize(document.root());
    let element = decode(&sanitized.root);
    RenderOutcome {
        resolution,
        element,
        diagnostics: sanitized.diagnostics,
    }
}

#[cfg(test)]
mod tests {
    use super::render;
    use crate::capability::CapabilitySet;
    use crate::decode::FALLBACK_CLASS;
    use crate::sandbox::{DiagnosticKind, Sandbox};
    use crate::schema::{RemoteDocument, RemoteNode};
    use crate::version::{Resolution, Version, VersionSet};
    use prism_ui::ElementKind;

    fn client() -> VersionSet {
        [Version::new(2, 0), Version::new(2, 1)]
            .into_iter()
            .collect()
    }

    fn sandbox() -> Sandbox {
        Sandbox::new(CapabilitySet::new().allow_kinds(["box", "text"]))
    }

    #[test]
    fn renders_whitelisted_document_cleanly() {
        let doc = RemoteDocument::new(
            Version::new(2, 1),
            RemoteNode::new("box").child(RemoteNode::text("hi")),
        );
        let outcome = render(&client(), &sandbox(), &doc);
        assert_eq!(outcome.resolution, Resolution::Exact(Version::new(2, 1)));
        assert!(outcome.diagnostics.is_empty());
        assert_eq!(
            outcome.element.child_elements()[0].text_content(),
            Some("hi")
        );
    }

    #[test]
    fn unsupported_version_short_circuits_to_fallback() {
        let doc = RemoteDocument::new(Version::new(9, 0), RemoteNode::new("box"));
        let outcome = render(&client(), &sandbox(), &doc);
        assert!(matches!(outcome.resolution, Resolution::Unsupported { .. }));
        assert_eq!(outcome.element.kind(), &ElementKind::Box);
        assert_eq!(outcome.element.class_names(), &[FALLBACK_CLASS.to_string()]);
        assert!(outcome.diagnostics.is_empty());
    }

    #[test]
    fn downgrade_still_sanitizes_and_decodes() {
        // Server is ahead; the client renders at 2.1 and the unknown kind
        // degrades to a placeholder.
        let doc = RemoteDocument::new(
            Version::new(2, 5),
            RemoteNode::new("box").child(RemoteNode::new("future_widget")),
        );
        let outcome = render(&client(), &sandbox(), &doc);
        assert_eq!(
            outcome.resolution,
            Resolution::Downgraded {
                requested: Version::new(2, 5),
                resolved: Version::new(2, 1),
            }
        );
        assert_eq!(outcome.diagnostics.len(), 1);
        assert_eq!(outcome.diagnostics[0].kind, DiagnosticKind::RejectedKind);
        assert_eq!(
            outcome.element.child_elements()[0].class_names(),
            &[FALLBACK_CLASS.to_string()]
        );
    }
}
