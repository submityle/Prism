//! `motion/` controls. See the kit design doc, section 5.10.
//!
//! Motion in the kit is **data-only**: controls carry no timers and no clock.
//! Each emits an [`Element`](prism_ui::Element) tagged with kit class names and
//! an externally-driven lifecycle (a [`TransitionPhase`] or an `open` bool);
//! this module's [`register_styles`] owns the token-backed
//! [`Class`](prism_ui_style::Class) rules for those names. The visible effects
//! the stack can express today are [`Opacity`](prism_ui_style::StyleProp::Opacity)
//! and [`Height`](prism_ui_style::StyleProp::Height); directional/scale intent
//! lives in the class name for a capable backend to honor (as glass blur does).
//!
//! * [`Transition`] — the reusable enter/leave scheduler primitive + wrapper.
//! * [`Fade`] / [`Slide`] / [`Scale`] — thin phase-driven transitions.
//! * [`Collapse`] — an `open`/closed height+opacity reveal (used by Accordion).

use crate::preset::StyleSheet;

pub mod collapse;
pub mod fade;
pub mod scale;
pub mod slide;
pub mod transition;

pub use collapse::{Collapse, CollapseProps};
pub use fade::{Fade, FadeProps};
pub use scale::{Scale, ScaleProps};
pub use slide::{Slide, SlideFrom, SlideProps};
pub use transition::{Easing, Transition, TransitionPhase, TransitionProps, TransitionSpec};

/// Registers every `motion/` control's token-backed classes into `sheet`.
pub fn register_styles(sheet: &mut StyleSheet) {
    transition::register_styles(sheet);
    fade::register_styles(sheet);
    slide::register_styles(sheet);
    scale::register_styles(sheet);
    collapse::register_styles(sheet);
}
