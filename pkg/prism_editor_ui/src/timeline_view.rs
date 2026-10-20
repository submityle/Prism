//! Keyed Loom sequencer rows and a bounded sampled curve editor.
use crate::{
    TimelineKeySelection, TimelinePanelAction as Action, TimelinePanelController,
    TimelinePanelError as Error, UiEvent, UiTrigger,
};
use prism_editor_timeline::{
    Interpolation, PlaybackConfig, PlaybackMode, PlaybackRate, PlaybackStatus, TimelineEditor,
    TimelineSnapshotToken,
};
use prism_ui::{
    style::{Keyword, StyleProp, StyleValue},
    Element,
};
use prism_ui_component::Component;
use prism_ui_component_kit::inputs::{SearchField, SearchFieldProps};
use std::{collections::BTreeMap, sync::Arc};

/// One actual curve sample projected to logical canvas pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimelineCurvePoint {
    /// Integer sample tick.
    pub tick: u64,
    /// Actual sampled channel value.
    pub value: f64,
    /// Logical horizontal pixel position.
    pub x: f32,
    /// Logical vertical pixel position in the 96-pixel curve.
    pub y: f32,
}
#[derive(Clone, Debug)]
enum Recipe {
    Action(Action),
    Select(TimelineKeySelection),
    Seek,
    Tick,
    Values,
    Incoming,
    Outgoing,
    Move,
    Start,
    Span,
    Channel,
    TrackName,
    TrackDimension,
    EventName,
    PlaybackStart,
    PlaybackEnd,
    Frequency,
    RateNumerator,
    RateDenominator,
}

/// Concrete retained Loom view, exact bindings and document freshness evidence.
#[derive(Clone, Debug)]
pub struct BoundTimelinePanel {
    element: Element,
    bindings: BTreeMap<(String, UiTrigger), Recipe>,
    pub(crate) identity: Arc<()>,
    pub(crate) token: TimelineSnapshotToken,
    curve: Vec<TimelineCurvePoint>,
    visible_keys: usize,
    truncated: bool,
    window: crate::TimelineWindow,
    tangents: (Vec<f64>, Vec<f64>),
    playback_config: PlaybackConfig,
    playback_rate: PlaybackRate,
}
impl BoundTimelinePanel {
    /// Actual tree to mount or reconcile.
    #[must_use]
    pub fn element(&self) -> &Element {
        &self.element
    }
    /// Exact interactive keys and their supported semantic triggers.
    pub fn bindings(&self) -> impl Iterator<Item = (&str, UiTrigger)> {
        self.bindings
            .keys()
            .map(|(key, trigger)| (key.as_str(), *trigger))
    }
    /// Curve samples represented by the visible curve canvas.
    #[must_use]
    pub fn curve(&self) -> &[TimelineCurvePoint] {
        &self.curve
    }
    /// Number of visible key markers built, at most 1024.
    #[must_use]
    pub const fn visible_keys(&self) -> usize {
        self.visible_keys
    }
    /// True when more keys intersect the window than the marker budget allows.
    #[must_use]
    pub const fn truncated(&self) -> bool {
        self.truncated
    }
    /// Resolves a semantic event without mutating the document. The controller
    /// additionally enforces freshness through `handle_event`.
    pub fn dispatch(&self, target: &str, event: &UiEvent) -> Result<Action, Error> {
        let trigger = match event {
            UiEvent::Activate => UiTrigger::Activate,
            UiEvent::Select { .. } => UiTrigger::Select,
            UiEvent::Commit { .. } => UiTrigger::Commit,
        };
        let recipe = self
            .bindings
            .get(&(target.to_owned(), trigger))
            .ok_or(Error::UnboundEvent)?;
        match (recipe, event) {
            (Recipe::Action(action), UiEvent::Activate) => Ok(action.clone()),
            (Recipe::Select(target), UiEvent::Select { mode }) => Ok(Action::Select {
                target: *target,
                mode: *mode,
            }),
            (recipe, UiEvent::Commit { value }) => {
                let unsigned = || {
                    value
                        .as_unsigned()
                        .or_else(|| value.as_text()?.trim().parse().ok())
                        .ok_or(Error::InvalidValue)
                };
                match recipe {
                    Recipe::PlaybackStart => Ok(Action::PlaybackConfig(PlaybackConfig {
                        start_tick: unsigned()?,
                        ..self.playback_config
                    })),
                    Recipe::PlaybackEnd => Ok(Action::PlaybackConfig(PlaybackConfig {
                        end_tick: unsigned()?,
                        ..self.playback_config
                    })),
                    Recipe::Frequency => Ok(Action::PlaybackConfig(PlaybackConfig {
                        ticks_per_second: unsigned()?,
                        ..self.playback_config
                    })),
                    Recipe::RateNumerator => Ok(Action::PlaybackRate(PlaybackRate::new(
                        unsigned()?,
                        self.playback_rate.ratio().1,
                    )?)),
                    Recipe::RateDenominator => Ok(Action::PlaybackRate(PlaybackRate::new(
                        self.playback_rate.ratio().0,
                        unsigned()?,
                    )?)),
                    Recipe::Seek => Ok(Action::Seek(unsigned()?)),
                    Recipe::Tick => Ok(Action::KeyTick(unsigned()?)),
                    Recipe::Channel => Ok(Action::Channel(
                        usize::try_from(unsigned()?).map_err(|_| Error::InvalidValue)?,
                    )),
                    Recipe::TrackDimension => Ok(Action::TrackDimension(
                        usize::try_from(unsigned()?).map_err(|_| Error::InvalidValue)?,
                    )),
                    Recipe::TrackName | Recipe::EventName => {
                        let name = value.as_text().ok_or(Error::InvalidValue)?;
                        if name.trim().is_empty() || name.len() > crate::MAX_ACTION_TEXT_BYTES {
                            return Err(Error::InvalidValue);
                        }
                        Ok(if matches!(recipe, Recipe::TrackName) {
                            Action::TrackName(name.to_owned())
                        } else {
                            Action::EventName(name.to_owned())
                        })
                    }
                    Recipe::Start => Ok(Action::Window(crate::TimelineWindow {
                        start: unsigned()?,
                        ..self.window
                    })),
                    Recipe::Span => Ok(Action::Window(crate::TimelineWindow {
                        span: unsigned()?,
                        ..self.window
                    })),
                    Recipe::Move => Ok(Action::MoveSelection(
                        value
                            .as_integer()
                            .or_else(|| value.as_text()?.trim().parse().ok())
                            .ok_or(Error::InvalidValue)?,
                    )),
                    Recipe::Values => Ok(Action::KeyValues(vector(
                        value.as_text().ok_or(Error::InvalidValue)?,
                    )?)),
                    Recipe::Incoming => Ok(Action::KeyTangents {
                        incoming: vector(value.as_text().ok_or(Error::InvalidValue)?)?,
                        outgoing: self.tangents.1.clone(),
                    }),
                    Recipe::Outgoing => Ok(Action::KeyTangents {
                        incoming: self.tangents.0.clone(),
                        outgoing: vector(value.as_text().ok_or(Error::InvalidValue)?)?,
                    }),
                    _ => Err(Error::UnboundEvent),
                }
            }
            _ => Err(Error::UnboundEvent),
        }
    }
}

pub(crate) fn build(
    controller: &TimelinePanelController,
    editor: &TimelineEditor,
) -> Result<BoundTimelinePanel, Error> {
    let window = controller.window;
    let mut view = BoundTimelinePanel {
        element: Element::box_(),
        bindings: BTreeMap::new(),
        identity: controller.identity.clone(),
        token: editor.snapshot_token(),
        curve: Vec::new(),
        visible_keys: 0,
        truncated: false,
        window,
        tangents: (Vec::new(), Vec::new()),
        playback_config: controller.playback.config(),
        playback_rate: controller.playback.rate(),
    };
    let config = controller.playback.config();
    let status = controller.playback.status();
    let transport = row()
        .key_str("timeline:transport")
        .child(button(
            &mut view,
            "timeline:play",
            "Play",
            (status == PlaybackStatus::Paused).then_some(Action::Play),
        ))
        .child(button(
            &mut view,
            "timeline:pause",
            "Pause",
            (status == PlaybackStatus::Playing).then_some(Action::Pause),
        ))
        .child(button(
            &mut view,
            "timeline:stop",
            "Stop",
            Some(Action::Stop),
        ))
        .child(button(
            &mut view,
            "timeline:loop",
            "Toggle loop",
            Some(Action::PlaybackConfig(PlaybackConfig {
                mode: if config.mode == PlaybackMode::Loop {
                    PlaybackMode::Once
                } else {
                    PlaybackMode::Loop
                },
                ..config
            })),
        ))
        .child(Element::text(format!("{status:?} / {:?}", config.mode)));
    let playback_bounds = row()
        .child(field(
            &mut view,
            "timeline:playback-start",
            "Playback start",
            &config.start_tick.to_string(),
            Recipe::PlaybackStart,
        ))
        .child(field(
            &mut view,
            "timeline:playback-end",
            "Playback end",
            &config.end_tick.to_string(),
            Recipe::PlaybackEnd,
        ))
        .child(field(
            &mut view,
            "timeline:frequency",
            "Ticks per second",
            &config.ticks_per_second.to_string(),
            Recipe::Frequency,
        ));
    let rate = controller.playback.rate().ratio();
    let playback_rate = row()
        .child(field(
            &mut view,
            "timeline:rate-numerator",
            "Speed numerator",
            &rate.0.to_string(),
            Recipe::RateNumerator,
        ))
        .child(field(
            &mut view,
            "timeline:rate-denominator",
            "Speed denominator",
            &rate.1.to_string(),
            Recipe::RateDenominator,
        ));
    let toolbar = row()
        .child(button(
            &mut view,
            "timeline:undo",
            "Undo",
            (editor.undo_len() > 0).then_some(Action::Undo),
        ))
        .child(button(
            &mut view,
            "timeline:redo",
            "Redo",
            (editor.redo_len() > 0).then_some(Action::Redo),
        ))
        .child(field(
            &mut view,
            "timeline:seek",
            "Playhead tick",
            &controller.playhead().to_string(),
            Recipe::Seek,
        ))
        .child(field(
            &mut view,
            "timeline:start",
            "First tick",
            &window.start.to_string(),
            Recipe::Start,
        ))
        .child(field(
            &mut view,
            "timeline:span",
            "Visible ticks",
            &window.span.to_string(),
            Recipe::Span,
        ));
    let tracks: Vec<_> = editor
        .timeline()
        .tracks_from(window.first_track)
        .take(window.rows + 1)
        .collect();
    let creation = row()
        .child(field(
            &mut view,
            "timeline:track-name",
            "New track name",
            &controller.track_name,
            Recipe::TrackName,
        ))
        .child(field(
            &mut view,
            "timeline:track-dimension",
            "Track channels",
            &controller.track_dimension.to_string(),
            Recipe::TrackDimension,
        ))
        .child(button(
            &mut view,
            "timeline:add-track",
            "Add track",
            Some(Action::AddTrack),
        ))
        .child(field(
            &mut view,
            "timeline:event-name",
            "New event name",
            &controller.event_name,
            Recipe::EventName,
        ))
        .child(button(
            &mut view,
            "timeline:add-event",
            "Add event at playhead",
            Some(Action::AddEvent),
        ));
    let previous = editor
        .timeline()
        .tracks_before(window.first_track)
        .rev()
        .take(window.rows)
        .last();
    let next = tracks.get(window.rows).map(|track| track.id());
    let navigation = row()
        .child(button(
            &mut view,
            "timeline:previous-tracks",
            "Previous tracks",
            previous.map(|track| {
                Action::Window(crate::TimelineWindow {
                    first_track: track.id(),
                    ..window
                })
            }),
        ))
        .child(button(
            &mut view,
            "timeline:next-tracks",
            "Next tracks",
            next.map(|id| {
                Action::Window(crate::TimelineWindow {
                    first_track: id,
                    ..window
                })
            }),
        ))
        .child(button(
            &mut view,
            "timeline:earlier",
            "Earlier",
            (window.start > 0).then_some(Action::Window(crate::TimelineWindow {
                start: window.start.saturating_sub(window.span / 2 + 1),
                ..window
            })),
        ))
        .child(button(
            &mut view,
            "timeline:later",
            "Later",
            (window.end() < u64::MAX).then_some(Action::Window(crate::TimelineWindow {
                start: window.start.saturating_add(window.span / 2 + 1),
                ..window
            })),
        ))
        .child(button(
            &mut view,
            "timeline:zoom-in",
            "Zoom in",
            (window.span > 1).then_some(Action::Window(crate::TimelineWindow {
                span: (window.span / 2).max(1),
                ..window
            })),
        ))
        .child(button(
            &mut view,
            "timeline:zoom-out",
            "Zoom out",
            (window.span < u64::MAX).then_some(Action::Window(crate::TimelineWindow {
                span: window.span.saturating_mul(2),
                ..window
            })),
        ));
    let mut rows = column().key_str("timeline:tracks");
    for track in tracks.into_iter().take(window.rows) {
        let mut lane =
            canvas(window.width, 28.0).key_str(format!("timeline:lane:{}", track.id().raw()));
        if (window.start..=window.end()).contains(&controller.playhead()) {
            lane = lane.child(mark(
                window.x(controller.playhead()),
                0.0,
                1.0,
                28.0,
                "color.primary",
            ));
        }
        for key in track.keys_in(window.start..=window.end()).take(1025) {
            if view.visible_keys == 1024 {
                view.truncated = true;
                break;
            }
            view.visible_keys += 1;
            let target = TimelineKeySelection {
                track: track.id(),
                key: key.id(),
            };
            let identity = format!("timeline:key:{}:{}", track.id().raw(), key.id().raw());
            view.bindings.insert(
                (identity.clone(), UiTrigger::Select),
                Recipe::Select(target),
            );
            lane = lane.child(
                mark(
                    window.x(key.tick()).clamp(0.0, window.width - 8.0),
                    8.0,
                    8.0,
                    12.0,
                    if controller.selection.contains(&target) {
                        "color.primary"
                    } else {
                        "color.text"
                    },
                )
                .key_str(identity),
            );
        }
        let header = row()
            .child(Element::text(track.name()))
            .child(button(
                &mut view,
                &format!("timeline:insert-key:{}", track.id().raw()),
                "Insert at playhead",
                Some(Action::InsertKey(track.id())),
            ))
            .child(button(
                &mut view,
                &format!("timeline:delete-track:{}", track.id().raw()),
                "Delete track",
                Some(Action::RemoveTrack(track.id())),
            ));
        rows = rows.child(
            column()
                .key_str(format!("timeline:track:{}", track.id().raw()))
                .child(header)
                .child(lane),
        );
    }
    if view.truncated {
        rows = rows.child(
            Element::text("Key marker limit reached; zoom in or show fewer tracks")
                .key_str("timeline:truncated"),
        );
    }
    let mut events = column().key_str("timeline:events");
    for (index, event) in editor
        .timeline()
        .events_in(window.start..=window.end())
        .take(129)
        .enumerate()
    {
        if index == 128 {
            events = events.child(Element::text("Event limit reached; zoom in"));
            break;
        }
        events = events.child(
            row()
                .child(Element::text(format!("{}: {}", event.tick, event.name)))
                .child(button(
                    &mut view,
                    &format!("timeline:delete-event:{}", event.id),
                    "Delete event",
                    Some(Action::RemoveEvent(event.id)),
                )),
        );
    }
    let selection = row()
        .child(Element::text(format!(
            "{} keys selected",
            controller.selection.len()
        )))
        .child(button(
            &mut view,
            "timeline:clear-selection",
            "Clear",
            (!controller.selection.is_empty()).then_some(Action::ClearSelection),
        ))
        .child(button(
            &mut view,
            "timeline:delete",
            "Delete keys",
            (!controller.selection.is_empty()).then_some(Action::DeleteSelection),
        ))
        .child(field(
            &mut view,
            "timeline:move",
            "Move ticks (+/-)",
            "0",
            Recipe::Move,
        ));
    let mut properties = column().key_str("timeline:properties");
    if let Ok(target) = controller.single() {
        let key = crate::timeline::lookup(editor, target)?;
        let track = editor
            .timeline()
            .track(target.track)
            .expect("selected key has a track");
        view.tangents = (key.incoming().to_vec(), key.outgoing().to_vec());
        properties = properties
            .child(field(
                &mut view,
                "timeline:key-tick",
                "Key tick",
                &key.tick().to_string(),
                Recipe::Tick,
            ))
            .child(field(
                &mut view,
                "timeline:key-values",
                "Values (comma separated)",
                &numbers(key.values()),
                Recipe::Values,
            ))
            .child(field(
                &mut view,
                "timeline:incoming",
                "Incoming slopes per tick",
                &numbers(key.incoming()),
                Recipe::Incoming,
            ))
            .child(field(
                &mut view,
                "timeline:outgoing",
                "Outgoing slopes per tick",
                &numbers(key.outgoing()),
                Recipe::Outgoing,
            ))
            .child(field(
                &mut view,
                "timeline:channel",
                "Curve channel (zero based)",
                &controller.channel.to_string(),
                Recipe::Channel,
            ));
        let mut modes = row();
        for (name, mode) in [
            ("step", Interpolation::Step),
            ("linear", Interpolation::Linear),
            ("cubic", Interpolation::Cubic),
        ] {
            modes = modes.child(button(
                &mut view,
                &format!("timeline:interpolation:{name}"),
                name,
                Some(Action::KeyInterpolation(mode)),
            ));
        }
        properties = properties.child(modes);
        let channel = controller.channel.min(track.dimension() - 1);
        for index in 0..128u128 {
            let tick = window
                .start
                .saturating_add((u128::from(window.span) * index / 127) as u64);
            if let Some(values) = track.sample(tick)? {
                view.curve.push(TimelineCurvePoint {
                    tick,
                    value: values[channel],
                    x: window.x(tick),
                    y: 0.0,
                });
            }
        }
        let min = view
            .curve
            .iter()
            .map(|point| point.value)
            .fold(f64::INFINITY, f64::min);
        let max = view
            .curve
            .iter()
            .map(|point| point.value)
            .fold(f64::NEG_INFINITY, f64::max);
        let scale = min.abs().max(max.abs()).max(1.0);
        let lower = min / scale;
        let upper = max / scale;
        let mut curve = canvas(window.width, 96.0).key_str("timeline:curve");
        for (index, point) in view.curve.iter_mut().enumerate() {
            point.y = if lower == upper {
                48.0
            } else {
                (92.0 - (point.value / scale - lower) / (upper - lower) * 88.0) as f32
            };
            curve = curve.child(
                mark(
                    point.x.clamp(0.0, window.width - 3.0),
                    point.y,
                    3.0,
                    3.0,
                    "color.primary",
                )
                .key_str(format!("timeline:curve:{index}")),
            );
        }
        properties = properties
            .child(Element::text(format!(
                "Channel {channel}, range [{min}, {max}]"
            )))
            .child(curve);
    } else {
        properties = properties.child(Element::text(
            "Select one key to edit values, interpolation and tangents",
        ));
    }
    view.element = column()
        .key_str("timeline:panel")
        .class("prism-editor-timeline")
        .child(Element::text("Sequencer"))
        .child(transport)
        .child(playback_bounds)
        .child(playback_rate)
        .child(toolbar)
        .child(creation)
        .child(navigation)
        .child(rows)
        .child(events)
        .child(selection)
        .child(properties);
    Ok(view)
}
fn vector(text: &str) -> Result<Vec<f64>, Error> {
    if text.len() > crate::MAX_ACTION_TEXT_BYTES {
        return Err(Error::InvalidValue);
    }
    let mut values = Vec::new();
    for item in text.split(',') {
        if values.len() == 64 {
            return Err(Error::InvalidValue);
        }
        let value: f64 = item.trim().parse().map_err(|_| Error::InvalidValue)?;
        if !value.is_finite() {
            return Err(Error::InvalidValue);
        }
        values.push(value);
    }
    Ok(values)
}
fn numbers(values: &[f64]) -> String {
    values
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}
fn field(
    view: &mut BoundTimelinePanel,
    key: &str,
    label: &str,
    value: &str,
    recipe: Recipe,
) -> Element {
    view.bindings
        .insert((key.to_owned(), UiTrigger::Commit), recipe);
    SearchField
        .render(&SearchFieldProps::new().value(value).placeholder(label))
        .key_str(key)
}
fn button(
    view: &mut BoundTimelinePanel,
    key: &str,
    label: &str,
    action: Option<Action>,
) -> Element {
    let enabled = action.is_some();
    if let Some(action) = action {
        view.bindings.insert(
            (key.to_owned(), UiTrigger::Activate),
            Recipe::Action(action),
        );
    }
    Element::box_()
        .key_str(key)
        .class(if enabled { "is-enabled" } else { "is-disabled" })
        .style(
            StyleProp::Opacity,
            StyleValue::number(if enabled { 1.0 } else { 0.45 }),
        )
        .child(Element::text(label))
}
fn column() -> Element {
    Element::box_()
        .style(StyleProp::Display, StyleValue::keyword(Keyword::Flex))
        .style(
            StyleProp::FlexDirection,
            StyleValue::keyword(Keyword::Column),
        )
        .style(StyleProp::Gap, StyleValue::px(4.0))
}
fn row() -> Element {
    Element::box_()
        .style(StyleProp::Display, StyleValue::keyword(Keyword::Flex))
        .style(StyleProp::FlexDirection, StyleValue::keyword(Keyword::Row))
        .style(StyleProp::Gap, StyleValue::px(8.0))
}
fn canvas(width: f32, height: f32) -> Element {
    Element::box_()
        .style(StyleProp::Position, StyleValue::keyword(Keyword::Relative))
        .style(StyleProp::Width, StyleValue::px(width))
        .style(StyleProp::Height, StyleValue::px(height))
        .style(StyleProp::FlexShrink, StyleValue::number(0.0))
        .style(StyleProp::BackgroundColor, StyleValue::token("color.bg"))
}
fn mark(x: f32, y: f32, width: f32, height: f32, color: &str) -> Element {
    Element::box_()
        .style(StyleProp::Position, StyleValue::keyword(Keyword::Absolute))
        .style(StyleProp::Left, StyleValue::px(x))
        .style(StyleProp::Top, StyleValue::px(y))
        .style(StyleProp::Width, StyleValue::px(width))
        .style(StyleProp::Height, StyleValue::px(height))
        .style(StyleProp::BackgroundColor, StyleValue::token(color))
}
