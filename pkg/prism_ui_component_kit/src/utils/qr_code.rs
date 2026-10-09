//! [`QrCode`] — a data-only QR placeholder resolved by the backend.
//!
//! The kit is `no_std` and does no rasterization. [`QrCode`] records *what* to
//! encode (`data`) and *how big* to draw it (`size_px`) and emits a single
//! [`Element::custom`] node named `pk-qrcode`, carrying the payload as a text
//! child and the `pk-qrcode` class. Turning that payload into actual modules is
//! a backend / `qr`-feature concern: a capable backend recognizes the custom
//! name, reads the text child, and rasterizes the matrix at the requested size.
//!
//! Until then the node is inert data (payload + sizing hint), so the control
//! composes and unit-tests without a running runtime.

use alloc::string::String;

use prism_ui::Element;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// The default rendered edge length, in logical pixels, used by
/// [`QrCodeProps::new`].
pub const DEFAULT_QR_SIZE_PX: f32 = 160.0;

/// Props for [`QrCode`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct QrCodeProps {
    /// The payload to encode (URL, text, etc.).
    pub data: String,
    /// The rendered edge length, in logical pixels. A square code.
    pub size_px: f32,
}

impl QrCodeProps {
    /// Creates props for `data` at the [default size](DEFAULT_QR_SIZE_PX).
    #[must_use]
    pub fn new(data: impl Into<String>) -> Self {
        Self {
            data: data.into(),
            size_px: DEFAULT_QR_SIZE_PX,
        }
    }

    /// Sets the payload to encode.
    #[must_use]
    pub fn data(mut self, data: impl Into<String>) -> Self {
        self.data = data.into();
        self
    }

    /// Sets the rendered edge length, in logical pixels.
    #[must_use]
    pub fn size_px(mut self, size_px: f32) -> Self {
        self.size_px = size_px;
        self
    }
}

/// The QR-code control. Zero-sized; config lives in [`QrCodeProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct QrCode;

impl Component for QrCode {
    type Props = QrCodeProps;

    fn render(&self, props: &Self::Props) -> Element {
        use prism_ui_style::{StyleProp, StyleValue};

        Element::custom("pk-qrcode")
            .class("pk-qrcode")
            .style(StyleProp::Width, StyleValue::px(props.size_px))
            .style(StyleProp::Height, StyleValue::px(props.size_px))
            .child(Element::text(props.data.clone()))
    }
}

/// Registers `pk-qrcode`: a square surface with a light fill and small radius
/// so the (backend-rendered) code sits on a clean token-backed card. The
/// concrete pixel size is an inline hint from `size_px`; the class owns only
/// the themeable surface.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, StyleProp, StyleValue};

    use crate::preset::tok;

    sheet.insert(
        Class::new("pk-qrcode")
            .with(StyleProp::BackgroundColor, tok("color.background"))
            .with(StyleProp::BorderRadius, tok("radius.sm"))
            .with(StyleProp::BorderWidth, StyleValue::px(1.0))
            .with(StyleProp::BorderColor, tok("color.separator")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ui::ElementKind;
    use prism_ui_style::{StyleProp, StyleValue};

    fn render(props: QrCodeProps) -> Element {
        QrCode.render(&props)
    }

    #[test]
    fn renders_custom_qrcode_element_with_class() {
        let el = render(QrCodeProps::new("https://example.com"));
        assert_eq!(el.kind(), &ElementKind::Custom("pk-qrcode".into()));
        assert_eq!(el.class_names(), ["pk-qrcode"]);
    }

    #[test]
    fn carries_payload_as_text_child() {
        let el = render(QrCodeProps::new("WIFI:T:WPA;S:net;P:pw;;"));
        let child = &el.child_elements()[0];
        assert_eq!(child.kind(), &ElementKind::Text);
        assert_eq!(child.text_content(), Some("WIFI:T:WPA;S:net;P:pw;;"));
    }

    #[test]
    fn size_px_becomes_inline_square() {
        let el = render(QrCodeProps::new("x").size_px(96.0));
        let inline = el.inline_pairs();
        assert!(inline.contains(&(StyleProp::Width, StyleValue::px(96.0))));
        assert!(inline.contains(&(StyleProp::Height, StyleValue::px(96.0))));
    }

    #[test]
    fn new_uses_default_size() {
        let props = QrCodeProps::new("x");
        assert_eq!(props.size_px, DEFAULT_QR_SIZE_PX);
    }

    #[test]
    fn register_adds_class() {
        let mut sheet = StyleSheet::new();
        register_styles(&mut sheet);
        assert!(sheet.get("pk-qrcode").is_some());
    }
}
