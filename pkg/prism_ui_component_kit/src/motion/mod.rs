//! `motion/` controls. See the kit design doc, section 5.
//!
//! Each control emits a data-only [`Element`](prism_ui::Element) carrying only
//! kit class names; this module's [`register_styles`] owns the token-backed
//! [`Class`](prism_ui_style::Class) rules for those names. Controls stay plain
//! [`Component`](prism_ui_component::Component)s so they compose and unit-test
//! without a running runtime.

use crate::preset::StyleSheet;

/// Registers every `motion/` control's token-backed classes into `sheet`.
///
/// Currently a no-op placeholder; control submodules append their registrars
/// here as they land.
pub fn register_styles(_sheet: &mut StyleSheet) {}
