//! [`Sequencer`] / [`Timeline`] — an animation / audio timeline.
//!
//! This is the **engine** timeline (tracks + keyframes + playhead), distinct
//! from the `display` event [`Timeline`](crate::display::Timeline). It renders
//! a `pk-sequencer` surface whose children are `pk-sequencer__track` rows; each
//! track lays out `pk-sequencer__keyframe` markers, and a `pk-sequencer__playhead`
//! marks the current time. Keyframe and playhead positions are data-derived
//! inline offsets (margin-left as a percentage), never color/shadow literals;
//! all surface values resolve from theme tokens via [`crate::preset`].

use alloc::string::String;
use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// Clamps `value` into the inclusive range `0.0..=1.0`.
///
/// Hand-rolled replacement for the std-only `f32::clamp`; `NaN` collapses to
/// `0.0` because neither comparison holds for it.
fn clamp01(value: f32) -> f32 {
    if value > 1.0 {
        1.0
    } else if value > 0.0 {
        value
    } else {
        0.0
    }
}

/// A single sequencer track: a label plus keyframe positions in `0.0..=1.0`.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct SequencerTrack {
    /// The track's label text.
    pub label: String,
    /// Keyframe positions along the track, each in `0.0..=1.0`.
    pub keyframes: Vec<f32>,
}

impl SequencerTrack {
    /// Creates a labelled track with no keyframes.
    #[must_use]
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            ..Self::default()
        }
    }

    /// Replaces the keyframe positions.
    #[must_use]
    pub fn keyframes<I: IntoIterator<Item = f32>>(mut self, keyframes: I) -> Self {
        self.keyframes = keyframes.into_iter().collect();
        self
    }
}

/// Props for [`Sequencer`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct SequencerProps {
    /// The tracks, rendered top-to-bottom.
    pub tracks: Vec<SequencerTrack>,
    /// The playhead position, in `0.0..=1.0`.
    pub playhead: f32,
}

impl SequencerProps {
    /// Creates empty sequencer props with the playhead at the start.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a track.
    #[must_use]
    pub fn track(mut self, track: SequencerTrack) -> Self {
        self.tracks.push(track);
        self
    }

    /// Replaces the tracks.
    #[must_use]
    pub fn tracks<I: IntoIterator<Item = SequencerTrack>>(mut self, tracks: I) -> Self {
        self.tracks = tracks.into_iter().collect();
        self
    }

    /// Sets the playhead position (clamped to `0.0..=1.0`).
    #[must_use]
    pub fn playhead(mut self, playhead: f32) -> Self {
        self.playhead = clamp01(playhead);
        self
    }
}

/// The sequencer control. Zero-sized; config lives in [`SequencerProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Sequencer;

/// Alias matching the `Timeline` naming used by the design doc (section 12).
pub type Timeline = Sequencer;

impl Sequencer {
    /// The accessibility role a sequencer exposes.
    #[must_use]
    pub const fn role() -> Role {
        Role::Group
    }
}

impl Component for Sequencer {
    type Props = SequencerProps;

    fn render(&self, props: &Self::Props) -> Element {
        use prism_ui_style::{StyleProp, StyleValue};

        let mut el = Element::box_().class("pk-sequencer");
        for track in &props.tracks {
            let label = Element::text(track.label.clone()).class("pk-sequencer__label");
            let mut lane = Element::box_().class("pk-sequencer__lane");
            for position in &track.keyframes {
                let marker = Element::box_().class("pk-sequencer__keyframe").style(
                    StyleProp::MarginLeft,
                    StyleValue::percent(clamp01(*position) * 100.0),
                );
                lane = lane.child(marker);
            }
            let row = Element::box_()
                .class("pk-sequencer__track")
                .child(label)
                .child(lane);
            el = el.child(row);
        }

        let playhead = Element::box_().class("pk-sequencer__playhead").style(
            StyleProp::MarginLeft,
            StyleValue::percent(clamp01(props.playhead) * 100.0),
        );
        el.child(playhead)
    }
}

/// Registers the `pk-sequencer` family: surface, track, label, lane, keyframe,
/// playhead.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    sheet.insert(
        Class::new("pk-sequencer")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::Gap, tok("space.xxs"))
            .with(StyleProp::BackgroundColor, tok("color.surface.secondary"))
            .with(StyleProp::BorderRadius, tok("radius.md"))
            .with_padding_x(tok("space.sm"))
            .with_padding_y(tok("space.sm")),
    );

    sheet.insert(
        Class::new("pk-sequencer__track")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::Gap, tok("space.sm")),
    );

    sheet.insert(
        Class::new("pk-sequencer__label")
            .with(StyleProp::Color, tok("color.label.secondary"))
            .with(StyleProp::FontSize, tok("font.size.caption1"))
            .with(StyleProp::MinWidth, StyleValue::px(72.0)),
    );

    sheet.insert(
        Class::new("pk-sequencer__lane")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::FlexGrow, StyleValue::number(1.0))
            .with(StyleProp::Height, StyleValue::px(20.0))
            .with(StyleProp::BackgroundColor, tok("color.fill.tertiary"))
            .with(StyleProp::BorderRadius, tok("radius.sm")),
    );

    sheet.insert(
        Class::new("pk-sequencer__keyframe")
            .with(StyleProp::Width, StyleValue::px(8.0))
            .with(StyleProp::Height, StyleValue::px(8.0))
            .with(StyleProp::MinWidth, StyleValue::px(8.0))
            .with(StyleProp::FlexShrink, StyleValue::number(0.0))
            .with(StyleProp::BorderRadius, tok("radius.xs"))
            .with(StyleProp::BackgroundColor, tok("color.tint")),
    );

    sheet.insert(
        Class::new("pk-sequencer__playhead")
            .with(StyleProp::Width, StyleValue::px(2.0))
            .with(StyleProp::Height, StyleValue::px(16.0))
            .with(StyleProp::BackgroundColor, tok("color.red")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ui_style::{StyleProp, StyleValue};

    fn render(props: SequencerProps) -> Element {
        Sequencer.render(&props)
    }

    #[test]
    fn empty_sequencer_has_only_playhead() {
        let el = render(SequencerProps::new());
        assert_eq!(el.class_names(), ["pk-sequencer"]);
        let children = el.child_elements();
        assert_eq!(children.len(), 1);
        assert_eq!(children[0].class_names(), ["pk-sequencer__playhead"]);
    }

    #[test]
    fn track_lays_out_label_then_lane_with_keyframes() {
        let el = render(
            SequencerProps::new()
                .track(SequencerTrack::new("Opacity").keyframes([0.0, 0.5, 1.0])),
        );
        let row = &el.child_elements()[0];
        assert_eq!(row.class_names(), ["pk-sequencer__track"]);
        assert_eq!(row.child_elements()[0].class_names(), ["pk-sequencer__label"]);
        let lane = &row.child_elements()[1];
        assert_eq!(lane.class_names(), ["pk-sequencer__lane"]);
        assert_eq!(lane.child_elements().len(), 3);
    }

    #[test]
    fn keyframe_position_is_inline_margin_percent() {
        let el = render(
            SequencerProps::new().track(SequencerTrack::new("x").keyframes([0.25])),
        );
        let marker = &el.child_elements()[0].child_elements()[1].child_elements()[0];
        assert_eq!(
            marker.inline_pairs(),
            [(StyleProp::MarginLeft, StyleValue::percent(25.0))]
        );
    }

    #[test]
    fn playhead_is_clamped_and_positioned() {
        let el = render(SequencerProps::new().playhead(2.0));
        let playhead = el.child_elements().last().unwrap();
        assert_eq!(
            playhead.inline_pairs(),
            [(StyleProp::MarginLeft, StyleValue::percent(100.0))]
        );
    }

    #[test]
    fn role_is_group() {
        assert_eq!(Sequencer::role(), Role::Group);
    }

    #[test]
    fn register_adds_classes() {
        let mut sheet = StyleSheet::new();
        register_styles(&mut sheet);
        for name in [
            "pk-sequencer",
            "pk-sequencer__track",
            "pk-sequencer__label",
            "pk-sequencer__lane",
            "pk-sequencer__keyframe",
            "pk-sequencer__playhead",
        ] {
            assert!(sheet.get(name).is_some(), "missing {name}");
        }
    }
}
