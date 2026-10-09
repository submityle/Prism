//! `editor/` controls — the authoring / inspector family. See the kit design
//! doc, sections 5.8 and 12.
//!
//! Each control emits a data-only [`Element`](prism_ui::Element) carrying only
//! kit class names; this module's [`register_styles`] owns the token-backed
//! [`Class`](prism_ui_style::Class) rules for those names. Controls stay plain
//! [`Component`](prism_ui_component::Component)s so they compose and unit-test
//! without a running runtime.
//!
//! This family ships the editor/tooling controls: [`PropertyGrid`]
//! ([`PropertyInspector`]), [`VectorField`], [`TransformField`],
//! [`Knob`]/[`Dial`], [`Gauge`], [`DockPanel`], [`Chart`],
//! [`Sequencer`]/[`Timeline`], [`AssetBrowser`]/[`AssetGrid`],
//! [`Console`]/[`LogView`] and [`Ruler`]/[`Guides`].

use crate::preset::StyleSheet;

pub mod asset_browser;
pub mod chart;
pub mod console_log;
pub mod dock_panel;
pub mod gauge;
pub mod knob;
pub mod property_grid;
pub mod ruler;
pub mod sequencer;
pub mod transform_field;
pub mod vector_field;

pub use asset_browser::{AssetBrowser, AssetBrowserProps, AssetGrid, AssetItem};
pub use chart::{Chart, ChartKind, ChartProps};
pub use console_log::{Console, ConsoleProps, LogEntry, LogLevel, LogView};
pub use dock_panel::{DockPanel, DockPanelProps, DockSide};
pub use gauge::{Gauge, GaugeProps};
pub use knob::{Dial, Knob, KnobProps};
pub use property_grid::{PropertyGrid, PropertyGridProps, PropertyInspector, PropertyRow};
pub use ruler::{Guides, Ruler, RulerOrientation, RulerProps};
pub use sequencer::{Sequencer, SequencerProps, SequencerTrack, Timeline};
pub use transform_field::{TransformField, TransformFieldProps};
pub use vector_field::{VectorField, VectorFieldProps};

/// Registers every `editor/` control's token-backed classes into `sheet`.
pub fn register_styles(sheet: &mut StyleSheet) {
    property_grid::register_styles(sheet);
    vector_field::register_styles(sheet);
    transform_field::register_styles(sheet);
    knob::register_styles(sheet);
    gauge::register_styles(sheet);
    dock_panel::register_styles(sheet);
    chart::register_styles(sheet);
    sequencer::register_styles(sheet);
    asset_browser::register_styles(sheet);
    console_log::register_styles(sheet);
    ruler::register_styles(sheet);
}
