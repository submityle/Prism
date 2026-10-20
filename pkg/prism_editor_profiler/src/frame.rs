//! Deterministic frame and nested span data.

use std::collections::{BTreeMap, BTreeSet};

use crate::{validate_limit, validate_text, ProfilerError, ProfilerLimits};

/// Stable identity for a captured frame.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct FrameId(u64);

impl FrameId {
    /// Creates an identifier from a producer-assigned value.
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

/// Stable identity for a span inside a frame.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct SpanId(u64);

impl SpanId {
    /// Creates an identifier from a producer-assigned value.
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

/// One normalized span with frame-relative nanosecond timing.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Span {
    id: SpanId,
    parent: Option<SpanId>,
    start_ns: u64,
    duration_ns: u64,
    name: String,
    lane: String,
}

impl Span {
    /// Creates a span. Frame insertion performs all limit and range checks.
    #[must_use]
    pub fn new(
        id: SpanId,
        parent: Option<SpanId>,
        start_ns: u64,
        duration_ns: u64,
        name: impl Into<String>,
        lane: impl Into<String>,
    ) -> Self {
        Self {
            id,
            parent,
            start_ns,
            duration_ns,
            name: name.into(),
            lane: lane.into(),
        }
    }

    /// Returns the stable span identity.
    #[must_use]
    pub const fn id(&self) -> SpanId {
        self.id
    }

    /// Returns the optional containing span.
    #[must_use]
    pub const fn parent(&self) -> Option<SpanId> {
        self.parent
    }

    /// Returns the frame-relative start time.
    #[must_use]
    pub const fn start_ns(&self) -> u64 {
        self.start_ns
    }

    /// Returns the span duration.
    #[must_use]
    pub const fn duration_ns(&self) -> u64 {
        self.duration_ns
    }

    /// Returns the display name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the stable lane/category name.
    #[must_use]
    pub fn lane(&self) -> &str {
        &self.lane
    }

    fn end_ns(&self) -> Result<u64, ProfilerError> {
        self.start_ns
            .checked_add(self.duration_ns)
            .ok_or(ProfilerError::ArithmeticOverflow)
    }
}

/// A captured frame and its validated frame-local spans.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Frame {
    id: FrameId,
    sequence: u64,
    timestamp_ns: u64,
    duration_ns: u64,
    spans: Vec<Span>,
}

impl Frame {
    /// Validates and creates a frame from unordered span input.
    pub fn try_new(
        id: FrameId,
        sequence: u64,
        timestamp_ns: u64,
        duration_ns: u64,
        mut spans: Vec<Span>,
        limits: &ProfilerLimits,
    ) -> Result<Self, ProfilerError> {
        validate_spans(duration_ns, &spans, limits)?;
        spans.sort_by_key(|span| (span.start_ns, span.duration_ns, span.id));
        Ok(Self {
            id,
            sequence,
            timestamp_ns,
            duration_ns,
            spans,
        })
    }

    /// Returns the stable frame identity.
    #[must_use]
    pub const fn id(&self) -> FrameId {
        self.id
    }

    /// Returns the producer sequence number.
    #[must_use]
    pub const fn sequence(&self) -> u64 {
        self.sequence
    }

    /// Returns the capture timestamp.
    #[must_use]
    pub const fn timestamp_ns(&self) -> u64 {
        self.timestamp_ns
    }

    /// Returns the frame duration.
    #[must_use]
    pub const fn duration_ns(&self) -> u64 {
        self.duration_ns
    }

    /// Returns spans in deterministic start/duration/id order.
    #[must_use]
    pub fn spans(&self) -> &[Span] {
        &self.spans
    }

    /// Builds a stable hierarchy from the validated flat span list.
    #[must_use]
    pub fn span_tree(&self) -> SpanTree {
        let mut children: BTreeMap<Option<SpanId>, Vec<SpanId>> = BTreeMap::new();
        let by_id: BTreeMap<SpanId, &Span> =
            self.spans.iter().map(|span| (span.id, span)).collect();
        for span in &self.spans {
            children.entry(span.parent).or_default().push(span.id);
        }
        for ids in children.values_mut() {
            ids.sort_by_key(|id| {
                let span = by_id[id];
                (span.start_ns, span.duration_ns, span.id)
            });
        }
        fn build(
            id: SpanId,
            by_id: &BTreeMap<SpanId, &Span>,
            children: &BTreeMap<Option<SpanId>, Vec<SpanId>>,
        ) -> SpanNode {
            let span = by_id[&id].clone();
            let child_nodes = children
                .get(&Some(id))
                .into_iter()
                .flatten()
                .map(|child| build(*child, by_id, children))
                .collect();
            SpanNode {
                span,
                children: child_nodes,
            }
        }
        let roots = children
            .get(&None)
            .into_iter()
            .flatten()
            .map(|id| build(*id, &by_id, &children))
            .collect();
        SpanTree { roots }
    }
}

/// A hierarchy node in a [`SpanTree`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpanNode {
    span: Span,
    children: Vec<SpanNode>,
}

impl SpanNode {
    /// Returns this node's span.
    #[must_use]
    pub const fn span(&self) -> &Span {
        &self.span
    }

    /// Returns children in stable timing order.
    #[must_use]
    pub fn children(&self) -> &[SpanNode] {
        &self.children
    }
}

/// Stable roots for one frame's span hierarchy.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpanTree {
    roots: Vec<SpanNode>,
}

impl SpanTree {
    /// Returns root spans in stable timing order.
    #[must_use]
    pub fn roots(&self) -> &[SpanNode] {
        &self.roots
    }
}

/// Bounded frame timeline with deterministic sequence/id ordering.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FrameStore {
    limits: ProfilerLimits,
    frames: Vec<Frame>,
}

impl FrameStore {
    /// Creates an empty frame store.
    #[must_use]
    pub const fn new(limits: ProfilerLimits) -> Self {
        Self {
            limits,
            frames: Vec::new(),
        }
    }

    /// Inserts a frame into its deterministic sequence position.
    pub fn insert(&mut self, frame: Frame) -> Result<(), ProfilerError> {
        validate_limit(self.frames.len() + 1, self.limits.max_frames, "frames")?;
        validate_spans(frame.duration_ns, &frame.spans, &self.limits)?;
        if self.frames.iter().any(|existing| existing.id == frame.id) {
            return Err(ProfilerError::DuplicateId {
                resource: "frame",
                id: frame.id.raw(),
            });
        }
        let position = self
            .frames
            .binary_search_by_key(&(frame.sequence, frame.id), |existing| {
                (existing.sequence, existing.id)
            })
            .unwrap_or_else(|position| position);
        self.frames.insert(position, frame);
        Ok(())
    }

    /// Returns frames in sequence/id order.
    #[must_use]
    pub fn frames(&self) -> &[Frame] {
        &self.frames
    }
}

fn validate_spans(
    frame_duration_ns: u64,
    spans: &[Span],
    limits: &ProfilerLimits,
) -> Result<(), ProfilerError> {
    validate_limit(spans.len(), limits.max_spans_per_frame, "spans per frame")?;
    let mut ids = BTreeSet::new();
    let mut parents = BTreeMap::new();
    for span in spans {
        validate_text(&span.name, "span name", limits)?;
        validate_text(&span.lane, "span lane", limits)?;
        if !ids.insert(span.id) {
            return Err(ProfilerError::DuplicateId {
                resource: "span",
                id: span.id.raw(),
            });
        }
        if span.end_ns()? > frame_duration_ns {
            return Err(ProfilerError::InvalidRange { field: "span" });
        }
        parents.insert(span.id, span.parent);
    }
    for span in spans {
        if let Some(parent) = span.parent {
            if !ids.contains(&parent) {
                return Err(ProfilerError::MissingId {
                    resource: "parent span",
                    id: parent.raw(),
                });
            }
            let parent_span = spans
                .iter()
                .find(|candidate| candidate.id == parent)
                .ok_or(ProfilerError::MissingId {
                    resource: "parent span",
                    id: parent.raw(),
                })?;
            if span.start_ns < parent_span.start_ns || span.end_ns()? > parent_span.end_ns()? {
                return Err(ProfilerError::InvalidRange {
                    field: "nested span",
                });
            }
        }
        let mut depth = 1usize;
        let mut current = span.parent;
        let mut visited = BTreeSet::new();
        while let Some(parent) = current {
            if !visited.insert(parent) {
                return Err(ProfilerError::Cycle);
            }
            depth = depth
                .checked_add(1)
                .ok_or(ProfilerError::ArithmeticOverflow)?;
            if depth > limits.max_span_depth {
                return Err(ProfilerError::LimitExceeded {
                    resource: "span depth",
                    limit: limits.max_span_depth,
                    actual: depth,
                });
            }
            current = parents.get(&parent).copied().flatten();
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits() -> ProfilerLimits {
        ProfilerLimits {
            max_frames: 2,
            max_spans_per_frame: 8,
            max_span_depth: 3,
            ..ProfilerLimits::default()
        }
    }

    #[test]
    fn span_tree_is_stable_for_unordered_input() {
        let limits = limits();
        let frame = Frame::try_new(
            FrameId::from_raw(7),
            1,
            100,
            100,
            vec![
                Span::new(
                    SpanId::from_raw(3),
                    Some(SpanId::from_raw(1)),
                    30,
                    10,
                    "child-b",
                    "cpu",
                ),
                Span::new(SpanId::from_raw(1), None, 0, 80, "root", "cpu"),
                Span::new(
                    SpanId::from_raw(2),
                    Some(SpanId::from_raw(1)),
                    10,
                    10,
                    "child-a",
                    "cpu",
                ),
            ],
            &limits,
        )
        .expect("valid frame");
        let tree = frame.span_tree();
        assert_eq!(tree.roots()[0].span().id(), SpanId::from_raw(1));
        assert_eq!(
            tree.roots()[0].children()[0].span().id(),
            SpanId::from_raw(2)
        );
        assert_eq!(
            tree.roots()[0].children()[1].span().id(),
            SpanId::from_raw(3)
        );
    }

    #[test]
    fn frame_rejects_depth_and_missing_parent() {
        let limits = limits();
        let missing = Frame::try_new(
            FrameId::from_raw(1),
            1,
            0,
            10,
            vec![Span::new(
                SpanId::from_raw(1),
                Some(SpanId::from_raw(9)),
                0,
                1,
                "x",
                "cpu",
            )],
            &limits,
        );
        assert!(matches!(missing, Err(ProfilerError::MissingId { .. })));
        let deep = Frame::try_new(
            FrameId::from_raw(2),
            1,
            0,
            10,
            vec![
                Span::new(SpanId::from_raw(1), None, 0, 10, "a", "cpu"),
                Span::new(
                    SpanId::from_raw(2),
                    Some(SpanId::from_raw(1)),
                    0,
                    9,
                    "b",
                    "cpu",
                ),
                Span::new(
                    SpanId::from_raw(3),
                    Some(SpanId::from_raw(2)),
                    0,
                    8,
                    "c",
                    "cpu",
                ),
                Span::new(
                    SpanId::from_raw(4),
                    Some(SpanId::from_raw(3)),
                    0,
                    7,
                    "d",
                    "cpu",
                ),
            ],
            &limits,
        );
        assert!(matches!(
            deep,
            Err(ProfilerError::LimitExceeded {
                resource: "span depth",
                ..
            })
        ));
    }

    #[test]
    fn frame_store_orders_and_bounds_frames() {
        let limits = limits();
        let make = |id, sequence| {
            Frame::try_new(FrameId::from_raw(id), sequence, 0, 1, Vec::new(), &limits).unwrap()
        };
        let mut store = FrameStore::new(limits);
        store.insert(make(2, 2)).unwrap();
        store.insert(make(1, 1)).unwrap();
        assert_eq!(
            store.frames().iter().map(Frame::id).collect::<Vec<_>>(),
            vec![FrameId::from_raw(1), FrameId::from_raw(2)]
        );
        assert!(matches!(
            store.insert(make(3, 3)),
            Err(ProfilerError::LimitExceeded {
                resource: "frames",
                ..
            })
        ));
    }
}
