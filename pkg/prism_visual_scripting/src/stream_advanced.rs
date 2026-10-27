//! Keyed stream joins, session windows, partition watermarks, and source offsets.

use crate::{Value, VsError, VsResult};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KeyedStreamEvent {
    pub event_id: String,
    pub key: String,
    pub partition: String,
    pub event_time_ms: u64,
    pub payload: Value,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JoinSide {
    Left,
    Right,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JoinedEvent {
    pub key: String,
    pub left: KeyedStreamEvent,
    pub right: KeyedStreamEvent,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IntervalJoin {
    pub before_ms: u64,
    pub after_ms: u64,
    pub max_buffered_events: usize,
    pub watermark_ms: Option<u64>,
    left: BTreeMap<String, Vec<KeyedStreamEvent>>,
    right: BTreeMap<String, Vec<KeyedStreamEvent>>,
    seen: BTreeSet<(JoinSide, String)>,
}

impl IntervalJoin {
    pub fn new(before_ms: u64, after_ms: u64, max_buffered_events: usize) -> VsResult<Self> {
        if max_buffered_events == 0 {
            return Err(VsError::Validate(
                "join buffer capacity must be greater than zero".into(),
            ));
        }
        Ok(Self {
            before_ms,
            after_ms,
            max_buffered_events,
            watermark_ms: None,
            left: BTreeMap::new(),
            right: BTreeMap::new(),
            seen: BTreeSet::new(),
        })
    }

    pub fn buffered_events(&self) -> usize {
        self.left.values().map(Vec::len).sum::<usize>()
            + self.right.values().map(Vec::len).sum::<usize>()
    }

    pub fn ingest(
        &mut self,
        side: JoinSide,
        event: KeyedStreamEvent,
    ) -> VsResult<Vec<JoinedEvent>> {
        validate_keyed_event(&event)?;
        if self
            .watermark_ms
            .is_some_and(|watermark| event.event_time_ms < watermark)
        {
            return Err(VsError::Runtime(format!(
                "join event {} is behind watermark",
                event.event_id
            )));
        }
        if !self.seen.insert((side, event.event_id.clone())) {
            return Ok(Vec::new());
        }
        if self.buffered_events() == self.max_buffered_events {
            self.seen.remove(&(side, event.event_id.clone()));
            return Err(VsError::Runtime("join buffer capacity exceeded".into()));
        }

        let opposite = match side {
            JoinSide::Left => self.right.get(&event.key),
            JoinSide::Right => self.left.get(&event.key),
        };
        let mut joined = opposite
            .into_iter()
            .flatten()
            .filter(|candidate| match side {
                JoinSide::Left => checked_interval(
                    candidate.event_time_ms,
                    event.event_time_ms,
                    self.before_ms,
                    self.after_ms,
                ),
                JoinSide::Right => checked_interval(
                    event.event_time_ms,
                    candidate.event_time_ms,
                    self.before_ms,
                    self.after_ms,
                ),
            })
            .map(|candidate| match side {
                JoinSide::Left => JoinedEvent {
                    key: event.key.clone(),
                    left: event.clone(),
                    right: candidate.clone(),
                },
                JoinSide::Right => JoinedEvent {
                    key: event.key.clone(),
                    left: candidate.clone(),
                    right: event.clone(),
                },
            })
            .collect::<Vec<_>>();
        joined.sort_by_key(|item| {
            (
                item.left.event_time_ms,
                item.left.event_id.clone(),
                item.right.event_time_ms,
                item.right.event_id.clone(),
            )
        });
        let target = match side {
            JoinSide::Left => &mut self.left,
            JoinSide::Right => &mut self.right,
        };
        let events = target.entry(event.key.clone()).or_default();
        events.push(event);
        events.sort_by_key(|item| (item.event_time_ms, item.event_id.clone()));
        Ok(joined)
    }

    pub fn advance_watermark(&mut self, watermark_ms: u64) -> usize {
        if self
            .watermark_ms
            .is_some_and(|current| watermark_ms <= current)
        {
            return 0;
        }
        self.watermark_ms = Some(watermark_ms);
        let before = self.buffered_events();
        retain_join_events(&mut self.left, |event| {
            event
                .event_time_ms
                .checked_add(self.after_ms)
                .is_some_and(|end| end >= watermark_ms)
        });
        retain_join_events(&mut self.right, |event| {
            event
                .event_time_ms
                .checked_add(self.before_ms)
                .is_some_and(|end| end >= watermark_ms)
        });
        before - self.buffered_events()
    }
}

fn retain_join_events(
    events: &mut BTreeMap<String, Vec<KeyedStreamEvent>>,
    mut retain: impl FnMut(&KeyedStreamEvent) -> bool,
) {
    events.retain(|_, values| {
        values.retain(&mut retain);
        !values.is_empty()
    });
}

fn in_interval(value: u64, start: u64, end: u64) -> bool {
    value >= start && value <= end
}

fn checked_interval(value: u64, center: u64, before_ms: u64, after_ms: u64) -> bool {
    let Some(start) = center.checked_sub(before_ms) else {
        return false;
    };
    let Some(end) = center.checked_add(after_ms) else {
        return false;
    };
    in_interval(value, start, end)
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SessionWindow {
    pub key: String,
    pub start_ms: u64,
    pub end_ms: u64,
    pub events: Vec<KeyedStreamEvent>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SessionWindowStream {
    pub gap_ms: u64,
    pub max_buffered_events: usize,
    pub watermark_ms: Option<u64>,
    pub sessions: BTreeMap<String, Vec<SessionWindow>>,
    seen_event_ids: BTreeSet<String>,
}

impl SessionWindowStream {
    pub fn new(gap_ms: u64, max_buffered_events: usize) -> VsResult<Self> {
        if gap_ms == 0 || max_buffered_events == 0 {
            return Err(VsError::Validate(
                "session gap and capacity must be greater than zero".into(),
            ));
        }
        Ok(Self {
            gap_ms,
            max_buffered_events,
            watermark_ms: None,
            sessions: BTreeMap::new(),
            seen_event_ids: BTreeSet::new(),
        })
    }

    pub fn buffered_events(&self) -> usize {
        self.sessions
            .values()
            .flatten()
            .map(|session| session.events.len())
            .sum()
    }

    pub fn ingest(&mut self, event: KeyedStreamEvent) -> VsResult<bool> {
        validate_keyed_event(&event)?;
        if self
            .watermark_ms
            .is_some_and(|watermark| event.event_time_ms < watermark)
        {
            return Err(VsError::Runtime(format!(
                "session event {} is behind watermark",
                event.event_id
            )));
        }
        if !self.seen_event_ids.insert(event.event_id.clone()) {
            return Ok(false);
        }
        if self.buffered_events() == self.max_buffered_events {
            self.seen_event_ids.remove(&event.event_id);
            return Err(VsError::Runtime("session buffer capacity exceeded".into()));
        }
        let sessions = self.sessions.entry(event.key.clone()).or_default();
        let mut merged = SessionWindow {
            key: event.key.clone(),
            start_ms: event.event_time_ms,
            end_ms: event.event_time_ms,
            events: vec![event],
        };
        let mut index = 0;
        while index < sessions.len() {
            let touches = sessions[index]
                .end_ms
                .checked_add(self.gap_ms)
                .is_some_and(|end| merged.start_ms <= end)
                && merged
                    .end_ms
                    .checked_add(self.gap_ms)
                    .is_some_and(|end| sessions[index].start_ms <= end);
            if touches {
                let session = sessions.remove(index);
                merged.start_ms = merged.start_ms.min(session.start_ms);
                merged.end_ms = merged.end_ms.max(session.end_ms);
                merged.events.extend(session.events);
            } else {
                index += 1;
            }
        }
        merged
            .events
            .sort_by_key(|item| (item.event_time_ms, item.event_id.clone()));
        sessions.push(merged);
        sessions.sort_by_key(|session| session.start_ms);
        Ok(true)
    }

    pub fn advance_watermark(&mut self, watermark_ms: u64) -> Vec<SessionWindow> {
        if self
            .watermark_ms
            .is_some_and(|current| watermark_ms <= current)
        {
            return Vec::new();
        }
        self.watermark_ms = Some(watermark_ms);
        let mut closed = Vec::new();
        self.sessions.retain(|_, sessions| {
            let mut retained = Vec::new();
            for session in std::mem::take(sessions) {
                if session
                    .end_ms
                    .checked_add(self.gap_ms)
                    .is_some_and(|end| end <= watermark_ms)
                {
                    closed.push(session);
                } else {
                    retained.push(session);
                }
            }
            *sessions = retained;
            !sessions.is_empty()
        });
        closed.sort_by_key(|session| (session.start_ms, session.key.clone()));
        closed
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PartitionProgress {
    pub max_event_time_ms: u64,
    pub idle: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PartitionWatermarks {
    pub allowed_lateness_ms: u64,
    pub watermark_ms: Option<u64>,
    pub partitions: BTreeMap<String, PartitionProgress>,
}

impl PartitionWatermarks {
    pub fn new(allowed_lateness_ms: u64) -> Self {
        Self {
            allowed_lateness_ms,
            watermark_ms: None,
            partitions: BTreeMap::new(),
        }
    }

    pub fn observe(&mut self, partition: &str, event_time_ms: u64) -> VsResult<Option<u64>> {
        if partition.is_empty() {
            return Err(VsError::Validate(
                "stream partition must not be empty".into(),
            ));
        }
        let progress = self
            .partitions
            .entry(partition.to_string())
            .or_insert(PartitionProgress {
                max_event_time_ms: event_time_ms,
                idle: false,
            });
        progress.max_event_time_ms = progress.max_event_time_ms.max(event_time_ms);
        progress.idle = false;
        Ok(self.recompute())
    }

    pub fn set_idle(&mut self, partition: &str, idle: bool) -> VsResult<Option<u64>> {
        let progress = self
            .partitions
            .get_mut(partition)
            .ok_or_else(|| VsError::Runtime(format!("unknown stream partition {partition}")))?;
        progress.idle = idle;
        Ok(self.recompute())
    }

    fn recompute(&mut self) -> Option<u64> {
        let candidate = self
            .partitions
            .values()
            .filter(|partition| !partition.idle)
            .map(|partition| {
                partition
                    .max_event_time_ms
                    .saturating_sub(self.allowed_lateness_ms)
            })
            .min();
        if let Some(candidate) = candidate
            && self.watermark_ms.is_none_or(|current| candidate > current)
        {
            self.watermark_ms = Some(candidate);
        }
        self.watermark_ms
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceOffset {
    pub offset: u64,
    pub payload_digest: [u8; 32],
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceOffsets {
    pub partitions: BTreeMap<String, SourceOffset>,
}

impl SourceOffsets {
    pub fn commit(&mut self, partition: &str, offset: u64, payload: &[u8]) -> VsResult<bool> {
        if partition.is_empty() {
            return Err(VsError::Validate(
                "source partition must not be empty".into(),
            ));
        }
        let payload_digest = Sha256::digest(payload).into();
        if let Some(current) = self.partitions.get(partition) {
            if offset < current.offset {
                return Err(VsError::Runtime("source offset moved backwards".into()));
            }
            if offset == current.offset {
                if payload_digest != current.payload_digest {
                    return Err(VsError::Runtime(
                        "source offset was replayed with different content".into(),
                    ));
                }
                return Ok(false);
            }
        }
        self.partitions.insert(
            partition.to_string(),
            SourceOffset {
                offset,
                payload_digest,
            },
        );
        Ok(true)
    }
}

fn validate_keyed_event(event: &KeyedStreamEvent) -> VsResult<()> {
    if event.event_id.is_empty() || event.key.is_empty() || event.partition.is_empty() {
        return Err(VsError::Validate(
            "keyed stream event identity, key, and partition must not be empty".into(),
        ));
    }
    Ok(())
}
