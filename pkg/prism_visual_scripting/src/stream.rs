//! Durable event-stream windows with event-time ordering and bounded buffering.

use crate::{Value, VsError, VsResult};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WindowSpec {
    Tumbling { size_ms: u64 },
    Sliding { size_ms: u64, step_ms: u64 },
}

impl WindowSpec {
    pub fn validate(&self) -> VsResult<()> {
        match self {
            Self::Tumbling { size_ms } if *size_ms == 0 => {
                Err(VsError::Validate("tumbling window size must be > 0".into()))
            }
            Self::Sliding { size_ms, step_ms } if *size_ms == 0 || *step_ms == 0 => Err(
                VsError::Validate("sliding window size and step must be > 0".into()),
            ),
            Self::Sliding { size_ms, step_ms } if step_ms > size_ms => Err(VsError::Validate(
                "sliding window step must be <= size".into(),
            )),
            _ => Ok(()),
        }
    }

    fn starts_for(&self, event_time_ms: u64) -> Vec<u64> {
        match self {
            Self::Tumbling { size_ms } => {
                let start = (event_time_ms / size_ms) * size_ms;
                start
                    .checked_add(*size_ms)
                    .map_or_else(Vec::new, |_| vec![start])
            }
            Self::Sliding { size_ms, step_ms } => {
                let latest = (event_time_ms / step_ms) * step_ms;
                let earliest = event_time_ms.saturating_sub(size_ms - 1);
                let first = (earliest / step_ms) * step_ms;
                (first..=latest)
                    .step_by(*step_ms as usize)
                    .filter(|start| {
                        start
                            .checked_add(*size_ms)
                            .is_some_and(|end| event_time_ms < end)
                    })
                    .collect()
            }
        }
    }

    fn end(&self, start: u64) -> Option<u64> {
        match self {
            Self::Tumbling { size_ms } | Self::Sliding { size_ms, .. } => {
                start.checked_add(*size_ms)
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackpressurePolicy {
    Reject,
    DropOldest,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamEvent {
    pub event_id: String,
    pub event_time_ms: u64,
    pub payload: Value,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClosedWindow {
    pub start_ms: u64,
    pub end_ms: u64,
    pub events: Vec<StreamEvent>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamIngest {
    Accepted,
    Duplicate,
    DroppedOldest,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LateEvent {
    TooLate { watermark_ms: u64 },
    AlreadyClosed { window_start_ms: u64 },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StreamIngestError {
    Late { reason: LateEvent },
    Backpressure { capacity: usize, required: usize },
    Arithmetic { message: String },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WindowedStream {
    pub spec: WindowSpec,
    pub allowed_lateness_ms: u64,
    pub max_buffered_events: usize,
    pub backpressure: BackpressurePolicy,
    pub watermark_ms: Option<u64>,
    pub max_event_time_ms: Option<u64>,
    pub windows: BTreeMap<u64, Vec<StreamEvent>>,
    pub closed_windows: BTreeSet<u64>,
    pub seen_event_ids: BTreeSet<String>,
    dropped_events: u64,
}

impl WindowedStream {
    pub fn new(
        spec: WindowSpec,
        allowed_lateness_ms: u64,
        max_buffered_events: usize,
        backpressure: BackpressurePolicy,
    ) -> VsResult<Self> {
        spec.validate()?;
        if max_buffered_events == 0 {
            return Err(VsError::Validate(
                "stream max_buffered_events must be > 0".into(),
            ));
        }
        Ok(Self {
            spec,
            allowed_lateness_ms,
            max_buffered_events,
            backpressure,
            watermark_ms: None,
            max_event_time_ms: None,
            windows: BTreeMap::new(),
            closed_windows: BTreeSet::new(),
            seen_event_ids: BTreeSet::new(),
            dropped_events: 0,
        })
    }

    pub fn buffered_events(&self) -> usize {
        self.windows.values().map(Vec::len).sum()
    }

    pub fn dropped_events(&self) -> u64 {
        self.dropped_events
    }

    pub fn ingest(
        &mut self,
        event: StreamEvent,
    ) -> Result<(StreamIngest, Vec<ClosedWindow>), StreamIngestError> {
        if !self.seen_event_ids.insert(event.event_id.clone()) {
            return Ok((StreamIngest::Duplicate, Vec::new()));
        }
        let current_watermark = self.watermark_ms.unwrap_or(0);
        let starts = self.spec.starts_for(event.event_time_ms);
        if starts.is_empty() || starts.iter().any(|start| self.spec.end(*start).is_none()) {
            self.seen_event_ids.remove(&event.event_id);
            return Err(StreamIngestError::Arithmetic {
                message: "event window end time exhausted".into(),
            });
        }
        if starts
            .iter()
            .any(|start| self.closed_windows.contains(start))
        {
            self.seen_event_ids.remove(&event.event_id);
            return Err(StreamIngestError::Late {
                reason: LateEvent::AlreadyClosed {
                    window_start_ms: starts[0],
                },
            });
        }
        let lateness_deadline = match event.event_time_ms.checked_add(self.allowed_lateness_ms) {
            Some(deadline) => deadline,
            None => {
                self.seen_event_ids.remove(&event.event_id);
                return Err(StreamIngestError::Arithmetic {
                    message: "event lateness deadline exhausted".into(),
                });
            }
        };
        if lateness_deadline <= current_watermark {
            self.seen_event_ids.remove(&event.event_id);
            return Err(StreamIngestError::Late {
                reason: LateEvent::TooLate {
                    watermark_ms: current_watermark,
                },
            });
        }
        if starts.len() > self.max_buffered_events {
            self.seen_event_ids.remove(&event.event_id);
            return Err(StreamIngestError::Backpressure {
                capacity: self.max_buffered_events,
                required: starts.len(),
            });
        }
        let mut result = StreamIngest::Accepted;
        let required = self
            .buffered_events()
            .checked_add(starts.len())
            .ok_or_else(|| StreamIngestError::Arithmetic {
                message: "buffered event count exhausted".into(),
            })?;
        if required > self.max_buffered_events {
            match self.backpressure {
                BackpressurePolicy::Reject => {
                    self.seen_event_ids.remove(&event.event_id);
                    return Err(StreamIngestError::Backpressure {
                        capacity: self.max_buffered_events,
                        required,
                    });
                }
                BackpressurePolicy::DropOldest => {
                    self.drop_oldest_until(self.max_buffered_events.saturating_sub(starts.len()));
                    result = StreamIngest::DroppedOldest;
                }
            }
        }
        self.max_event_time_ms = Some(
            self.max_event_time_ms
                .map_or(event.event_time_ms, |max| max.max(event.event_time_ms)),
        );
        for start in starts {
            self.windows.entry(start).or_default().push(event.clone());
        }
        self.sort_windows();
        let watermark = self
            .max_event_time_ms
            .expect("ingest assigned max event time")
            .saturating_sub(self.allowed_lateness_ms);
        let closed = self.advance_watermark(watermark);
        Ok((result, closed))
    }

    pub fn advance_watermark(&mut self, watermark_ms: u64) -> Vec<ClosedWindow> {
        if self
            .watermark_ms
            .is_some_and(|current| watermark_ms <= current)
        {
            return Vec::new();
        }
        self.watermark_ms = Some(watermark_ms);
        let closable: Vec<u64> = self
            .windows
            .keys()
            .copied()
            .filter(|start| self.spec.end(*start).is_some_and(|end| end <= watermark_ms))
            .collect();
        closable
            .into_iter()
            .filter_map(|start| {
                self.windows.remove(&start).map(|mut events| {
                    events.sort_by_key(|event| (event.event_time_ms, event.event_id.clone()));
                    self.closed_windows.insert(start);
                    ClosedWindow {
                        start_ms: start,
                        end_ms: self.spec.end(start).expect("validated window end"),
                        events,
                    }
                })
            })
            .collect()
    }

    fn sort_windows(&mut self) {
        for events in self.windows.values_mut() {
            events.sort_by_key(|event| (event.event_time_ms, event.event_id.clone()));
        }
    }

    fn drop_oldest_until(&mut self, target: usize) {
        while self.buffered_events() > target {
            let Some(start) = self
                .windows
                .iter()
                .filter_map(|(start, events)| events.first().map(|event| (*start, event)))
                .min_by_key(|(start, event)| (event.event_time_ms, event.event_id.clone(), *start))
                .map(|(start, _)| start)
            else {
                break;
            };
            let should_remove = if let Some(events) = self.windows.get_mut(&start) {
                events.remove(0);
                events.is_empty()
            } else {
                false
            };
            if should_remove {
                self.windows.remove(&start);
            }
            self.dropped_events = self.dropped_events.saturating_add(1);
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AggregationSpec {
    Count,
    Sum,
    Average,
    Min,
    Max,
    DistinctCount,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WindowAggregate {
    pub start_ms: u64,
    pub end_ms: u64,
    pub event_count: u64,
    pub value: Value,
    pub source_digest: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ReductionOutcome {
    Applied { aggregate: WindowAggregate },
    Replay { aggregate: WindowAggregate },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WindowReducer {
    pub spec: AggregationSpec,
    pub aggregates: BTreeMap<u64, WindowAggregate>,
}

impl WindowReducer {
    pub fn new(spec: AggregationSpec) -> Self {
        Self {
            spec,
            aggregates: BTreeMap::new(),
        }
    }

    pub fn reduce(&mut self, window: &ClosedWindow) -> VsResult<ReductionOutcome> {
        validate_closed_window(window)?;
        let source_digest = window_digest(window)?;
        if let Some(existing) = self.aggregates.get(&window.start_ms) {
            if existing.source_digest != source_digest || existing.end_ms != window.end_ms {
                return Err(VsError::Runtime(format!(
                    "window {} was replayed with different content",
                    window.start_ms
                )));
            }
            return Ok(ReductionOutcome::Replay {
                aggregate: existing.clone(),
            });
        }
        let value = aggregate_events(&self.spec, &window.events)?;
        let aggregate = WindowAggregate {
            start_ms: window.start_ms,
            end_ms: window.end_ms,
            event_count: window.events.len() as u64,
            value,
            source_digest,
        };
        self.aggregates.insert(window.start_ms, aggregate.clone());
        Ok(ReductionOutcome::Applied { aggregate })
    }
}

fn validate_closed_window(window: &ClosedWindow) -> VsResult<()> {
    if window.end_ms <= window.start_ms {
        return Err(VsError::Validate(
            "closed window end must be greater than start".into(),
        ));
    }
    let mut ids = BTreeSet::new();
    for event in &window.events {
        if event.event_id.is_empty() {
            return Err(VsError::Validate(
                "closed window event id must not be empty".into(),
            ));
        }
        if !ids.insert(&event.event_id) {
            return Err(VsError::Validate(format!(
                "closed window contains duplicate event {}",
                event.event_id
            )));
        }
        if event.event_time_ms < window.start_ms || event.event_time_ms >= window.end_ms {
            return Err(VsError::Validate(format!(
                "event {} is outside closed window bounds",
                event.event_id
            )));
        }
        let non_finite = match &event.payload {
            Value::F32(value) => !value.is_finite(),
            Value::F64(value) => !value.is_finite(),
            _ => false,
        };
        if non_finite {
            return Err(VsError::Validate(format!(
                "stream event {} has a non-finite numeric payload",
                event.event_id
            )));
        }
    }
    Ok(())
}

fn window_digest(window: &ClosedWindow) -> VsResult<[u8; 32]> {
    let encoded = serde_json::to_vec(window)
        .map_err(|error| VsError::Runtime(format!("window digest failed: {error}")))?;
    Ok(Sha256::digest(encoded).into())
}

fn aggregate_events(spec: &AggregationSpec, events: &[StreamEvent]) -> VsResult<Value> {
    match spec {
        AggregationSpec::Count => Ok(Value::I64(count_as_i64(events.len())?)),
        AggregationSpec::DistinctCount => {
            let mut values = BTreeSet::new();
            for event in events {
                let encoded = serde_json::to_vec(&event.payload).map_err(|error| {
                    VsError::Runtime(format!("distinct value encoding failed: {error}"))
                })?;
                values.insert(Sha256::digest(encoded).to_vec());
            }
            Ok(Value::I64(count_as_i64(values.len())?))
        }
        AggregationSpec::Sum => finite_aggregate(numeric_values(events)?.iter().sum(), "sum"),
        AggregationSpec::Average => {
            let values = numeric_values(events)?;
            if values.is_empty() {
                return Err(VsError::Runtime("cannot average an empty window".into()));
            }
            let average = values.iter().sum::<f64>() / values.len() as f64;
            finite_aggregate(average, "average")
        }
        AggregationSpec::Min => numeric_extreme(events, f64::min, "min"),
        AggregationSpec::Max => numeric_extreme(events, f64::max, "max"),
    }
}

fn numeric_values(events: &[StreamEvent]) -> VsResult<Vec<f64>> {
    events
        .iter()
        .map(|event| {
            let value = event
                .payload
                .as_f64()
                .ok_or_else(|| VsError::TypeMismatch {
                    expected: crate::ValueType::F64,
                    got: event.payload.ty(),
                    location: format!("stream event {} payload", event.event_id),
                })?;
            if !value.is_finite() {
                return Err(VsError::Validate(format!(
                    "stream event {} has a non-finite numeric payload",
                    event.event_id
                )));
            }
            Ok(value)
        })
        .collect()
}

fn numeric_extreme(
    events: &[StreamEvent],
    combine: fn(f64, f64) -> f64,
    name: &str,
) -> VsResult<Value> {
    let mut values = numeric_values(events)?.into_iter();
    let first = values
        .next()
        .ok_or_else(|| VsError::Runtime(format!("cannot compute {name} of an empty window")))?;
    Ok(Value::F64(values.fold(first, combine)))
}

fn finite_aggregate(value: f64, name: &str) -> VsResult<Value> {
    if !value.is_finite() {
        return Err(VsError::Runtime(format!(
            "stream {name} aggregate is non-finite"
        )));
    }
    Ok(Value::F64(value))
}

fn count_as_i64(count: usize) -> VsResult<i64> {
    i64::try_from(count).map_err(|_| VsError::Runtime("stream count exceeds i64".into()))
}
