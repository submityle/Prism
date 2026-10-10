//! Gallery instances for the `utils` family. Dev-only; see `gallery/mod.rs`.
//!
//! Skips controls that paint nothing visible (`VisuallyHidden`, `Portal`).

use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_component::mount_component;

use super::Showcase;
use crate::utils::{
    CopyButton, CopyButtonProps, QrCode, QrCodeProps, Watermark, WatermarkProps,
};

/// Real, named instances of every *visible* `utils` control.
#[must_use]
pub fn instances() -> Vec<Showcase> {
    alloc::vec![
        Showcase::new(
            "Watermark / Mark",
            mount_component(
                &Watermark,
                WatermarkProps::new("CONFIDENTIAL")
                    .child(Element::box_().child(Element::text("Document body"))),
            ),
        ),
        Showcase::new(
            "CopyButton / Copied",
            mount_component(
                &CopyButton,
                CopyButtonProps::new("loom-key-123", "Copy key").copied(true),
            ),
        ),
        Showcase::new(
            "QrCode / Data",
            mount_component(&QrCode, QrCodeProps::new("https://example.com").size_px(96.0)),
        ),
    ]
}
