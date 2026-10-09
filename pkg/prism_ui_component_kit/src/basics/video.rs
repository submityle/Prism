//! [`Video`] (aliased [`MediaPlayer`]) — a media-player shell with chrome.
//!
//! A video renders a `pk-video` container stacking a `pk-video__surface`
//! (the picture placeholder) over an optional `pk-video__controls` bar. The
//! control bar is a glass strip holding a `pk-video__play` toggle, a
//! `pk-video__time` readout, a `pk-video__progress` track whose
//! `pk-video__progress-fill` width is the clamped playback fraction, and a
//! `pk-video__fullscreen` affordance. The fill width is the one legitimately
//! dynamic value and rides as an inline length; every color comes from a token
//! so [`crate::preset`] owns the palette. There is no real media decode or
//! backdrop blur here — this is the data-only shell every backend keys off.

use alloc::string::String;

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::kit::classes;
use crate::preset::StyleSheet;

/// Props for [`Video`] / [`MediaPlayer`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct VideoProps {
    /// The media source (resolved by the backend).
    pub source: Option<String>,
    /// A poster frame shown before playback (resolved by the backend).
    pub poster: Option<String>,
    /// Whether playback is currently running.
    pub playing: bool,
    /// Whether the control bar is rendered.
    pub controls: bool,
    /// The current playback position, in seconds.
    pub current: f32,
    /// The total media duration, in seconds.
    pub duration: f32,
}

impl VideoProps {
    /// Creates player props with the control bar shown (the common default).
    #[must_use]
    pub fn new() -> Self {
        Self {
            controls: true,
            ..Self::default()
        }
    }

    /// Sets the media source.
    #[must_use]
    pub fn source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Sets the poster frame.
    #[must_use]
    pub fn poster(mut self, poster: impl Into<String>) -> Self {
        self.poster = Some(poster.into());
        self
    }

    /// Sets whether playback is running.
    #[must_use]
    pub fn playing(mut self, playing: bool) -> Self {
        self.playing = playing;
        self
    }

    /// Sets whether the control bar is rendered.
    #[must_use]
    pub fn controls(mut self, controls: bool) -> Self {
        self.controls = controls;
        self
    }

    /// Sets the current playback position (seconds).
    #[must_use]
    pub fn current(mut self, current: f32) -> Self {
        self.current = current;
        self
    }

    /// Sets the total media duration (seconds).
    #[must_use]
    pub fn duration(mut self, duration: f32) -> Self {
        self.duration = duration;
        self
    }
}

/// Clamps `value` into `0.0..=1.0` without relying on `f32::clamp` (keeps the
/// control trivially `no_std` and NaN-safe: a non-ordered value falls to 0).
#[must_use]
fn clamp01(value: f32) -> f32 {
    if value > 1.0 {
        1.0
    } else if value > 0.0 {
        value
    } else {
        0.0
    }
}

/// The playback fraction in `0.0..=1.0`. A non-positive duration reads as 0 so
/// an un-loaded player shows an empty track rather than a NaN width.
#[must_use]
fn progress_fraction(current: f32, duration: f32) -> f32 {
    if duration > 0.0 {
        clamp01(current / duration)
    } else {
        0.0
    }
}

/// The media-player control. Zero-sized; config lives in [`VideoProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Video;

/// `MediaPlayer` is the semantic alias for [`Video`].
pub type MediaPlayer = Video;

impl Video {
    /// The accessibility role a media player exposes.
    #[must_use]
    pub const fn role() -> Role {
        Role::Group
    }
}

impl Component for Video {
    type Props = VideoProps;

    fn render(&self, props: &Self::Props) -> Element {
        use prism_ui_style::{StyleProp, StyleValue};

        let mut el = Element::box_();
        let mods: &[&str] = if props.playing { &["playing"] } else { &[] };
        for name in classes("pk-video", mods) {
            el = el.class(name);
        }

        // Surface: the picture placeholder. Carries the source/poster as data.
        let mut surface = Element::box_().class("pk-video__surface");
        if let Some(source) = props.source.as_ref() {
            surface = surface.child(Element::text(source.clone()).class("pk-video__source"));
        }
        if let Some(poster) = props.poster.as_ref() {
            surface = surface.child(Element::text(poster.clone()).class("pk-video__poster"));
        }
        el = el.child(surface);

        if props.controls {
            let play_state = if props.playing {
                "pk-video__play--playing"
            } else {
                "pk-video__play--paused"
            };
            let play = Element::box_().class("pk-video__play").class(play_state);
            let time = Element::box_().class("pk-video__time");

            let fill = Element::box_().class("pk-video__progress-fill").style(
                StyleProp::Width,
                // The only inline value: a dynamic, continuous width.
                StyleValue::percent(progress_fraction(props.current, props.duration) * 100.0),
            );
            let progress = Element::box_().class("pk-video__progress").child(fill);

            let fullscreen = Element::box_().class("pk-video__fullscreen");

            let controls = Element::box_()
                .class("pk-video__controls")
                .child(play)
                .child(time)
                .child(progress)
                .child(fullscreen);
            el = el.child(controls);
        }

        el
    }
}

/// Registers the `pk-video` class family: the container (+ playing modifier),
/// the surface placeholder, and the glass control bar with its play toggle,
/// time readout, progress track + fill, and fullscreen affordance.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, InteractionState, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    // Container: a rounded column stacking the surface over the control bar.
    sheet.insert(
        Class::new("pk-video")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::Gap, tok("space.xs"))
            .with(StyleProp::BorderRadius, tok("radius.lg")),
    );
    // Playing is a backend-readable state marker; no own paint.
    sheet.insert(Class::new("pk-video--playing"));

    // Surface: the picture placeholder on a quiet secondary fill. Sizing is a
    // backend concern (ratio); the base just gives it presence and a floor.
    sheet.insert(
        Class::new("pk-video__surface")
            .with(StyleProp::FlexGrow, StyleValue::number(1.0))
            .with(StyleProp::MinHeight, StyleValue::px(160.0))
            .with(StyleProp::BackgroundColor, tok("color.fill.secondary"))
            .with(StyleProp::BorderRadius, tok("radius.md")),
    );
    // Source/poster are data-only children; no own paint.
    sheet.insert(Class::new("pk-video__source"));
    sheet.insert(Class::new("pk-video__poster"));

    // Control bar: a frosted glass strip laying its affordances in a row.
    sheet.insert(
        Class::new("pk-video__controls")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::Gap, tok("space.sm"))
            .with_padding_x(tok("space.md"))
            .with_padding_y(tok("space.xs"))
            .with(StyleProp::BorderRadius, tok("radius.md"))
            .with_glass(0.0, tok("glass.tint"), Some(tok("glass.highlight")))
            .with_shadow(0.0, 4.0, 12.0, tok("glass.shadow")),
    );

    // Play toggle: a capsule button on a neutral fill, dimming on interaction.
    sheet.insert(
        Class::new("pk-video__play")
            .with(StyleProp::Width, StyleValue::px(32.0))
            .with(StyleProp::Height, StyleValue::px(32.0))
            .with(StyleProp::BorderRadius, tok("radius.capsule"))
            .with(StyleProp::BackgroundColor, tok("color.fill"))
            .with(StyleProp::Color, tok("color.label"))
            .with_state(InteractionState::Hover, StyleProp::Opacity, StyleValue::number(0.9))
            .with_state(InteractionState::Pressed, StyleProp::Opacity, StyleValue::number(0.78)),
    );
    // Play/pause glyph swap is a backend concern keyed off these markers.
    sheet.insert(Class::new("pk-video__play--playing"));
    sheet.insert(Class::new("pk-video__play--paused"));

    // Time readout: a quiet footnote-sized label.
    sheet.insert(
        Class::new("pk-video__time")
            .with(StyleProp::FontSize, tok("font.size.footnote"))
            .with(StyleProp::Color, tok("color.label.secondary")),
    );

    // Progress track: a thin capsule rail that grows to fill the bar.
    sheet.insert(
        Class::new("pk-video__progress")
            .with(StyleProp::FlexGrow, StyleValue::number(1.0))
            .with(StyleProp::Height, StyleValue::px(4.0))
            .with(StyleProp::BorderRadius, tok("radius.capsule"))
            .with(StyleProp::BackgroundColor, tok("color.fill.tertiary")),
    );
    // Fill: full height, rounded, accent-tinted; width rides inline per-frame.
    sheet.insert(
        Class::new("pk-video__progress-fill")
            .with(StyleProp::Height, StyleValue::percent(100.0))
            .with(StyleProp::BorderRadius, tok("radius.capsule"))
            .with(StyleProp::BackgroundColor, tok("color.tint")),
    );

    // Fullscreen affordance: a quiet secondary-label control.
    sheet.insert(
        Class::new("pk-video__fullscreen")
            .with(StyleProp::Width, StyleValue::px(24.0))
            .with(StyleProp::Height, StyleValue::px(24.0))
            .with(StyleProp::Color, tok("color.label.secondary"))
            .with_state(InteractionState::Hover, StyleProp::Opacity, StyleValue::number(0.9))
            .with_state(InteractionState::Pressed, StyleProp::Opacity, StyleValue::number(0.78)),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ui_style::{Length, StyleProp, StyleValue};

    fn render(props: VideoProps) -> Element {
        Video.render(&props)
    }

    fn controls_bar(el: &Element) -> Option<&Element> {
        el.child_elements()
            .iter()
            .find(|c| c.class_names().iter().any(|n| n == "pk-video__controls"))
    }

    fn fill_width(el: &Element) -> f32 {
        let controls = controls_bar(el).expect("control bar");
        let progress = controls
            .child_elements()
            .iter()
            .find(|c| c.class_names().iter().any(|n| n == "pk-video__progress"))
            .expect("progress track");
        let fill = &progress.child_elements()[0];
        for (prop, value) in fill.inline_pairs() {
            if *prop == StyleProp::Width
                && let StyleValue::Length(Length::Percent(p)) = value
            {
                return *p;
            }
        }
        panic!("fill has no percent width");
    }

    #[test]
    fn surface_only_when_controls_hidden() {
        let el = render(VideoProps::new().controls(false));
        assert_eq!(el.class_names(), ["pk-video"]);
        let kids = el.child_elements();
        assert_eq!(kids.len(), 1);
        assert_eq!(kids[0].class_names(), ["pk-video__surface"]);
    }

    #[test]
    fn controls_render_bar_after_surface() {
        let el = render(VideoProps::new());
        let kids = el.child_elements();
        assert_eq!(kids.len(), 2);
        assert_eq!(kids[0].class_names(), ["pk-video__surface"]);
        assert!(controls_bar(&el).is_some());
    }

    #[test]
    fn playing_adds_container_and_play_modifier() {
        let el = render(VideoProps::new().playing(true));
        assert_eq!(el.class_names(), ["pk-video", "pk-video--playing"]);
        let controls = controls_bar(&el).expect("control bar");
        let play = &controls.child_elements()[0];
        assert_eq!(play.class_names(), ["pk-video__play", "pk-video__play--playing"]);
    }

    #[test]
    fn paused_marks_play_as_paused() {
        let el = render(VideoProps::new());
        let controls = controls_bar(&el).expect("control bar");
        let play = &controls.child_elements()[0];
        assert_eq!(play.class_names(), ["pk-video__play", "pk-video__play--paused"]);
    }

    #[test]
    fn progress_width_tracks_position() {
        let el = render(VideoProps::new().current(30.0).duration(120.0));
        assert!((fill_width(&el) - 25.0).abs() < f32::EPSILON);
    }

    #[test]
    fn progress_width_clamps_and_guards_zero_duration() {
        assert!((fill_width(&render(VideoProps::new().current(10.0).duration(0.0))) - 0.0).abs() < f32::EPSILON);
        assert!((fill_width(&render(VideoProps::new().current(99.0).duration(10.0))) - 100.0).abs() < f32::EPSILON);
    }

    #[test]
    fn carries_source_and_poster_into_surface() {
        let el = render(VideoProps::new().source("clip.mp4").poster("frame.png").controls(false));
        let surface = &el.child_elements()[0];
        let kids = surface.child_elements();
        assert_eq!(kids[0].class_names(), ["pk-video__source"]);
        assert_eq!(kids[0].text_content(), Some("clip.mp4"));
        assert_eq!(kids[1].class_names(), ["pk-video__poster"]);
        assert_eq!(kids[1].text_content(), Some("frame.png"));
    }

    #[test]
    fn role_is_group() {
        assert_eq!(Video::role(), Role::Group);
    }

    #[test]
    fn alias_resolves_to_video() {
        let el = MediaPlayer::default().render(&VideoProps::new().controls(false));
        assert_eq!(el.class_names(), ["pk-video"]);
    }
}
