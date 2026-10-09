//! `utils/` controls. See the kit design doc, section 5.11.
//!
//! A grab-bag of structural and utility controls that don't belong to a visual
//! family: [`Portal`] (teleport marker), [`VisuallyHidden`] (a11y-only text),
//! [`Watermark`] (repeated overlay mark), [`CopyButton`] (clipboard-state
//! label) and [`QrCode`] (data-only QR placeholder).
//!
//! Each control emits a data-only [`Element`](prism_ui::Element) carrying only
//! kit class names (or a backend-recognized custom element name); this module's
//! [`register_styles`] owns the token-backed [`Class`](prism_ui_style::Class)
//! rules for those names. Controls stay plain
//! [`Component`](prism_ui_component::Component)s so they compose and unit-test
//! without a running runtime.

use crate::preset::StyleSheet;

pub mod copy_button;
pub mod portal;
pub mod qr_code;
pub mod visually_hidden;
pub mod watermark;

pub use copy_button::{CopyButton, CopyButtonProps};
pub use portal::{Portal, PortalProps};
pub use qr_code::{QrCode, QrCodeProps, DEFAULT_QR_SIZE_PX};
pub use visually_hidden::{VisuallyHidden, VisuallyHiddenProps};
pub use watermark::{Watermark, WatermarkProps};

/// Registers every `utils/` control's token-backed classes into `sheet`.
pub fn register_styles(sheet: &mut StyleSheet) {
    portal::register_styles(sheet);
    visually_hidden::register_styles(sheet);
    watermark::register_styles(sheet);
    copy_button::register_styles(sheet);
    qr_code::register_styles(sheet);
}
