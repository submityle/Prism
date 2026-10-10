//! Gallery instances for the `editor` family. Dev-only; see `gallery/mod.rs`.

use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_component::mount_component;

use super::Showcase;
use crate::editor::{
    AssetBrowser, AssetBrowserProps, AssetItem, Chart, ChartKind, ChartProps, Console, ConsoleProps,
    DockPanel, DockPanelProps, DockSide, Gauge, GaugeProps, Knob, KnobProps, LogEntry, LogLevel,
    PropertyGrid, PropertyGridProps, PropertyRow, Ruler, RulerOrientation, RulerProps, Sequencer,
    SequencerProps, SequencerTrack, TransformField, TransformFieldProps, VectorField,
    VectorFieldProps,
};

/// Real, named instances of every `editor` control.
#[must_use]
pub fn instances() -> Vec<Showcase> {
    alloc::vec![
        Showcase::new(
            "PropertyGrid / Rows",
            mount_component(
                &PropertyGrid,
                PropertyGridProps::new().rows([
                    PropertyRow::new("Name", Element::text("Cube")),
                    PropertyRow::new("Visible", Element::text("true")),
                ]),
            ),
        ),
        Showcase::new(
            "VectorField / XYZ",
            mount_component(
                &VectorField,
                VectorFieldProps::xyz().values(["0.0", "1.5", "-2.0"]),
            ),
        ),
        Showcase::new(
            "TransformField / TRS",
            mount_component(
                &TransformField,
                TransformFieldProps::new()
                    .translation(VectorFieldProps::xyz().values(["0.0", "0.0", "0.0"]))
                    .rotation(VectorFieldProps::xyz().values(["0.0", "90.0", "0.0"]))
                    .scale(VectorFieldProps::xyz().values(["1.0", "1.0", "1.0"])),
            ),
        ),
        Showcase::new(
            "Knob / Value",
            mount_component(&Knob, KnobProps::new(0.6)),
        ),
        Showcase::new(
            "Gauge / Value",
            mount_component(&Gauge, GaugeProps::new(0.72).text("72%")),
        ),
        Showcase::new(
            "DockPanel / Left",
            mount_component(
                &DockPanel,
                DockPanelProps::new("Inspector")
                    .side(DockSide::Left)
                    .child(Element::text("Panel body")),
            ),
        ),
        Showcase::new(
            "Chart / Bar",
            mount_component(
                &Chart,
                ChartProps::new()
                    .title("Frame time")
                    .kind(ChartKind::Bar)
                    .plot(Element::box_().child(Element::text("plot")))
                    .legend(Element::text("ms")),
            ),
        ),
        Showcase::new(
            "Sequencer / Tracks",
            mount_component(
                &Sequencer,
                SequencerProps::new()
                    .tracks([
                        SequencerTrack::new("Position").keyframes([0.0, 0.5, 1.0]),
                        SequencerTrack::new("Rotation").keyframes([0.25, 0.75]),
                    ])
                    .playhead(0.5),
            ),
        ),
        Showcase::new(
            "AssetBrowser / Grid",
            mount_component(
                &AssetBrowser,
                AssetBrowserProps::new()
                    .toolbar(Element::text("Assets"))
                    .items([
                        AssetItem::new("hero.png"),
                        AssetItem::new("tree.glb"),
                        AssetItem::new("sky.hdr"),
                    ]),
            ),
        ),
        Showcase::new(
            "Console / Log",
            mount_component(
                &Console,
                ConsoleProps::new()
                    .search(Element::text("filter"))
                    .entries([
                        LogEntry::new(LogLevel::Info, "Build started"),
                        LogEntry::new(LogLevel::Warn, "Deprecated API"),
                        LogEntry::new(LogLevel::Error, "Link failed"),
                    ]),
            ),
        ),
        Showcase::new(
            "Ruler / Horizontal",
            mount_component(
                &Ruler,
                RulerProps::new(10)
                    .orientation(RulerOrientation::Horizontal)
                    .guide(3.0),
            ),
        ),
    ]
}
