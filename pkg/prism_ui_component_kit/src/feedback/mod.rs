//! `feedback/` controls. See the kit design doc, section 5.
//!
//! Each control emits a data-only [`Element`](prism_ui::Element) carrying only
//! kit class names; this module's [`register_styles`] owns the token-backed
//! [`Class`](prism_ui_style::Class) rules for those names. Controls stay plain
//! [`Component`](prism_ui_component::Component)s so they compose and unit-test
//! without a running runtime.
//!
//! The family's overlay-backed controls share one base per architecture
//! invariant K7: [`Popover`] is the single frosted-glass surface, and both
//! [`Tooltip`] and [`Dialog`] compose it in their own `render`. The remaining
//! controls — [`Toast`], [`Alert`], [`Spinner`], [`ProgressBar`], [`Skeleton`],
//! [`Banner`] and [`Callout`] — are standalone status surfaces.

use crate::preset::StyleSheet;

pub mod alert;
pub mod banner;
pub mod callout;
pub mod dialog;
pub mod popover;
pub mod progress_bar;
pub mod skeleton;
pub mod spinner;
pub mod toast;
pub mod tooltip;

pub use alert::{Alert, AlertProps};
pub use banner::{Banner, BannerProps};
pub use callout::{Callout, CalloutProps};
pub use dialog::{Dialog, DialogProps};
pub use popover::{Placement, Popover, PopoverProps};
pub use progress_bar::{ProgressBar, ProgressBarProps};
pub use skeleton::{Skeleton, SkeletonProps};
pub use spinner::{Spinner, SpinnerProps};
pub use toast::{Toast, ToastProps};
pub use tooltip::{Tooltip, TooltipProps};

/// Registers every `feedback/` control's token-backed classes into `sheet`.
///
/// The overlay base ([`popover`]) registers first so its surface rules are in
/// place before the controls that compose it.
pub fn register_styles(sheet: &mut StyleSheet) {
    popover::register_styles(sheet);
    tooltip::register_styles(sheet);
    dialog::register_styles(sheet);
    toast::register_styles(sheet);
    alert::register_styles(sheet);
    spinner::register_styles(sheet);
    progress_bar::register_styles(sheet);
    skeleton::register_styles(sheet);
    banner::register_styles(sheet);
    callout::register_styles(sheet);
}
