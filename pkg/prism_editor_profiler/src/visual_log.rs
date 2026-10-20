//! Structured visual-log events and deterministic timeline projections.

use std::collections::BTreeMap;

use crate::{validate_limit, validate_text, ProfilerError, ProfilerLimits};

/// Stable identity for one visual-log event.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct VisualEventId(u64);

impl VisualEventId {
    /// Creates an event identity from a producer-assigned value.
    #[must_use]
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    /// Returns the producer-assigned value.
    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }
}

/// Backend-neutral debug geometry or annotation.
#[derive(Clone, Debug, PartialEq)]
pub enum VisualPrimitive {
    /// A point at a world-space position.
    Point([f32; 3]),
    /// A line segment between two positions.
    Line {
        /// Segment start.
        start: [f32; 3],
        /// Segment end.
        end: [f32; 3],
    },
    /// An axis-aligned box.
    Box {
        /// Minimum corner.
        min: [f32; 3],
        /// Maximum corner.
        max: [f32; 3],
    },
    /// A text annotation.
    Text(String),
}

/// One structured event suitable for a visual-log timeline.
#[derive(Clone, Debug, PartialEq)]
pub struct VisualEvent {
    id: VisualEventId,
    start_ns: u64,
    end_ns: u64,
    track: String,
    label: String,
    subject: Option<u64>,
    primitive: VisualPrimitive,
}

impl VisualEvent {
    /// Creates an event. Range, text, and numeric validation occurs on insert.
    #[must_use]
    pub fn new(
        id: VisualEventId,
        start_ns: u64,
        end_ns: u64,
        track: impl Into<String>,
        label: impl Into<String>,
        subject: Option<u64>,
        primitive: VisualPrimitive,
    ) -> Self {
        Self {
            id,
            start_ns,
            end_ns,
            track: track.into(),
            label: label.into(),
            subject,
            primitive,
        }
    }

    /// Returns the stable event identity.
    #[must_use]
    pub const fn id(&self) -> VisualEventId {
        self.id
    }

    /// Returns the inclusive start timestamp.
    #[must_use]
    pub const fn start_ns(&self) -> u64 {
        self.start_ns
    }

    /// Returns the exclusive end timestamp.
    #[must_use]
    pub const fn end_ns(&self) -> u64 {
        self.end_ns
    }

    /// Returns the track name.
    #[must_use]
    pub fn track(&self) -> &str {
        &self.track
    }

    /// Returns the annotation label.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Returns the optional runtime subject identity.
    #[must_use]
    pub const fn subject(&self) -> Option<u64> {
        self.subject
    }

    /// Returns the structured primitive.
    #[must_use]
    pub const fn primitive(&self) -> &VisualPrimitive {
        &self.primitive
    }
}

/// Events belonging to one stable track.
#[derive(Clone, Debug, PartialEq)]
pub struct VisualTrack {
    name: String,
    events: Vec<VisualEvent>,
}

impl VisualTrack {
    /// Returns the track name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns events in stable timeline order.
    #[must_use]
    pub fn events(&self) -> &[VisualEvent] {
        &self.events
    }
}

/// Result of projecting a visual-log time window into tracks.
#[derive(Clone, Debug, PartialEq)]
pub struct TrackProjection {
    tracks: Vec<VisualTrack>,
    event_count: usize,
}

impl TrackProjection {
    /// Returns tracks in name order.
    #[must_use]
    pub fn tracks(&self) -> &[VisualTrack] {
        &self.tracks
    }

    /// Returns the number of projected events.
    #[must_use]
    pub const fn event_count(&self) -> usize {
        self.event_count
    }
}

/// Bounded visual-log storage with stable event ordering.
#[derive(Clone, Debug, PartialEq)]
pub struct VisualLog {
    limits: ProfilerLimits,
    events: Vec<VisualEvent>,
    tracks: BTreeMap<String, usize>,
}

impl VisualLog {
    /// Creates an empty visual log.
    #[must_use]
    pub const fn new(limits: ProfilerLimits) -> Self {
        Self {
            limits,
            events: Vec::new(),
            tracks: BTreeMap::new(),
        }
    }

    /// Records one structured event after validating its bounds.
    pub fn record(&mut self, event: VisualEvent) -> Result<(), ProfilerError> {
        validate_limit(
            self.events.len() + 1,
            self.limits.max_visual_events,
            "visual events",
        )?;
        validate_text(&event.track, "visual track", &self.limits)?;
        validate_text(&event.label, "visual label", &self.limits)?;
        if event.start_ns >= event.end_ns {
            return Err(ProfilerError::InvalidRange {
                field: "visual event",
            });
        }
        if self.events.iter().any(|existing| existing.id == event.id) {
            return Err(ProfilerError::DuplicateId {
                resource: "visual event",
                id: event.id.raw(),
            });
        }
        validate_primitive(&event.primitive, &self.limits)?;
        if !self.tracks.contains_key(&event.track) {
            validate_limit(
                self.tracks.len() + 1,
                self.limits.max_visual_tracks,
                "visual tracks",
            )?;
            self.tracks.insert(event.track.clone(), self.tracks.len());
        }
        let position = self
            .events
            .binary_search_by_key(
                &(event.start_ns, event.end_ns, event.track.clone(), event.id),
                |existing| {
                    (
                        existing.start_ns,
                        existing.end_ns,
                        existing.track.clone(),
                        existing.id,
                    )
                },
            )
            .unwrap_or_else(|position| position);
        self.events.insert(position, event);
        Ok(())
    }

    /// Returns all events in stable time/track/id order.
    #[must_use]
    pub fn events(&self) -> &[VisualEvent] {
        &self.events
    }

    /// Projects events intersecting `[start_ns, end_ns)` into stable tracks.
    pub fn project(
        &self,
        start_ns: u64,
        end_ns: u64,
        max_events: usize,
    ) -> Result<TrackProjection, ProfilerError> {
        if start_ns >= end_ns {
            return Err(ProfilerError::InvalidRange {
                field: "projection",
            });
        }
        let limit = max_events.min(self.limits.max_projection_events);
        let mut grouped: BTreeMap<String, Vec<VisualEvent>> = BTreeMap::new();
        let mut event_count = 0;
        for event in &self.events {
            if event.start_ns < end_ns && event.end_ns > start_ns {
                if event_count == limit {
                    break;
                }
                grouped
                    .entry(event.track.clone())
                    .or_default()
                    .push(event.clone());
                event_count += 1;
            }
        }
        let tracks = grouped
            .into_iter()
            .map(|(name, events)| VisualTrack { name, events })
            .collect();
        Ok(TrackProjection {
            tracks,
            event_count,
        })
    }
}

fn validate_primitive(
    primitive: &VisualPrimitive,
    limits: &ProfilerLimits,
) -> Result<(), ProfilerError> {
    match primitive {
        VisualPrimitive::Point(point) => validate_points(point),
        VisualPrimitive::Line { start, end } => {
            validate_points(start)?;
            validate_points(end)
        }
        VisualPrimitive::Box { min, max } => {
            validate_points(min)?;
            validate_points(max)?;
            if min.iter().zip(max).any(|(low, high)| low > high) {
                return Err(ProfilerError::InvalidRange {
                    field: "visual box",
                });
            }
            Ok(())
        }
        VisualPrimitive::Text(text) => validate_text(text, "visual primitive text", limits),
    }
}

fn validate_points(points: &[f32; 3]) -> Result<(), ProfilerError> {
    if points.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(ProfilerError::NonFiniteValue {
            field: "visual position",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projection_groups_intersecting_events_stably() {
        let mut log = VisualLog::new(ProfilerLimits::default());
        log.record(VisualEvent::new(
            VisualEventId::from_raw(2),
            20,
            30,
            "ai",
            "late",
            Some(4),
            VisualPrimitive::Point([0.0, 0.0, 0.0]),
        ))
        .unwrap();
        log.record(VisualEvent::new(
            VisualEventId::from_raw(1),
            0,
            10,
            "navigation",
            "path",
            None,
            VisualPrimitive::Line {
                start: [0.0, 0.0, 0.0],
                end: [1.0, 0.0, 0.0],
            },
        ))
        .unwrap();
        let projection = log.project(5, 25, 8).unwrap();
        assert_eq!(projection.event_count(), 2);
        assert_eq!(
            projection
                .tracks()
                .iter()
                .map(VisualTrack::name)
                .collect::<Vec<_>>(),
            vec!["ai", "navigation"]
        );
    }

    #[test]
    fn visual_input_is_bounded_and_validated() {
        let mut log = VisualLog::new(ProfilerLimits {
            max_visual_events: 1,
            ..ProfilerLimits::default()
        });
        log.record(VisualEvent::new(
            VisualEventId::from_raw(1),
            0,
            1,
            "debug",
            "point",
            None,
            VisualPrimitive::Point([0.0, 0.0, 0.0]),
        ))
        .unwrap();
        let result = log.record(VisualEvent::new(
            VisualEventId::from_raw(2),
            1,
            2,
            "debug",
            "bad",
            None,
            VisualPrimitive::Point([f32::NAN, 0.0, 0.0]),
        ));
        assert!(matches!(
            result,
            Err(ProfilerError::LimitExceeded {
                resource: "visual events",
                ..
            })
        ));
    }
}
