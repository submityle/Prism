//! Safe decoding of a sanitized tree into [`prism_ui::Element`]s.
//!
//! [`decode`] is total: it only ever produces real elements for the built-in
//! kinds and for custom kinds that survived the [`Sandbox`](crate::Sandbox),
//! and it maps the reserved [`KIND_FALLBACK`] placeholder to an
//! error-boundary-style element. It must be fed a sanitized tree; running it on
//! raw untrusted input would bypass the capability check.

use prism_ui::Element;

use crate::schema::{RemoteNode, KIND_BOX, KIND_FALLBACK, KIND_TEXT};

/// The style class applied to fallback placeholder elements.
pub const FALLBACK_CLASS: &str = "sdui-fallback";

/// The message shown by a fallback element when no reason was recorded.
pub const DEFAULT_FALLBACK_MESSAGE: &str = "unavailable";

/// Builds an error-boundary-style placeholder element.
///
/// This is the single safe landing spot for anything that could not be
/// rendered: rejected nodes, unsupported schema versions and depth-capped
/// subtrees all funnel here, mirroring the error-boundary pattern used by
/// `prism_ui_async`.
///
/// # Examples
///
/// ```
/// use prism_ui_sdui::fallback_element;
///
/// let el = fallback_element(Some("blocked"));
/// assert_eq!(el.child_elements()[0].text_content(), Some("blocked"));
/// ```
#[must_use]
pub fn fallback_element(reason: Option<&str>) -> Element {
    Element::box_()
        .class(FALLBACK_CLASS)
        .child(Element::text(reason.unwrap_or(DEFAULT_FALLBACK_MESSAGE)))
}

/// Decodes a sanitized [`RemoteNode`] into a [`prism_ui::Element`].
///
/// Built-in kinds map to their native elements, the reserved placeholder kind
/// maps to [`fallback_element`], and any other (already whitelisted) kind maps
/// to a [`custom`](Element::custom) element carrying the kind name. Style
/// tokens become style classes and children are decoded recursively.
///
/// # Examples
///
/// ```
/// use prism_ui::ElementKind;
/// use prism_ui_sdui::{decode, RemoteNode};
///
/// let node = RemoteNode::new("box").child(RemoteNode::text("hi"));
/// let element = decode(&node);
/// assert_eq!(element.kind(), &ElementKind::Box);
/// assert_eq!(element.child_elements()[0].text_content(), Some("hi"));
/// ```
#[must_use]
pub fn decode(node: &RemoteNode) -> Element {
    // The placeholder kind is a leaf: it carries its reason as text and never
    // exposes untrusted children.
    if node.kind() == KIND_FALLBACK {
        return fallback_element(node.text_content());
    }

    let mut element = match node.kind() {
        KIND_TEXT => Element::text(node.text_content().unwrap_or_default()),
        KIND_BOX => Element::box_(),
        other => Element::custom(other),
    };

    for token in node.style_tokens() {
        element = element.class(token.clone());
    }

    for child in node.child_nodes() {
        element = element.child(decode(child));
    }

    element
}

#[cfg(test)]
mod tests {
    use super::{decode, fallback_element, DEFAULT_FALLBACK_MESSAGE, FALLBACK_CLASS};
    use crate::schema::RemoteNode;
    use prism_ui::ElementKind;

    #[test]
    fn decodes_box_text_and_custom_kinds() {
        let node = RemoteNode::new("box")
            .style("card")
            .child(RemoteNode::text("hello"))
            .child(RemoteNode::new("gauge"));
        let el = decode(&node);
        assert_eq!(el.kind(), &ElementKind::Box);
        assert_eq!(el.class_names(), &["card".to_string()]);
        assert_eq!(el.child_elements()[0].kind(), &ElementKind::Text);
        assert_eq!(el.child_elements()[0].text_content(), Some("hello"));
        assert_eq!(
            el.child_elements()[1].kind(),
            &ElementKind::Custom("gauge".to_string())
        );
    }

    #[test]
    fn placeholder_decodes_to_fallback_with_reason() {
        let node = RemoteNode::fallback("blocked kind `script`");
        let el = decode(&node);
        assert_eq!(el.kind(), &ElementKind::Box);
        assert_eq!(el.class_names(), &[FALLBACK_CLASS.to_string()]);
        assert_eq!(
            el.child_elements()[0].text_content(),
            Some("blocked kind `script`")
        );
    }

    #[test]
    fn fallback_element_uses_default_message_when_absent() {
        let el = fallback_element(None);
        assert_eq!(
            el.child_elements()[0].text_content(),
            Some(DEFAULT_FALLBACK_MESSAGE)
        );
    }

    #[test]
    fn placeholder_ignores_untrusted_children() {
        // Even if a placeholder somehow carried children, decode drops them.
        let node = RemoteNode::fallback("x").child(RemoteNode::new("script"));
        let el = decode(&node);
        assert_eq!(el.child_elements().len(), 1);
        assert_eq!(el.child_elements()[0].text_content(), Some("x"));
    }
}
