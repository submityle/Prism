//! Bounded nested clocks over borrowed real documents, without runtime domains.
use crate::{
    Playback, PlaybackConfig, PlaybackError, PlaybackMode, PlaybackRate, PlaybackStatus,
    TimelineDocument, TimelineError, TimelineEvent, TimelineLimits, TrackId,
};
use alloc::{
    collections::{BTreeMap, BTreeSet},
    vec::Vec,
};
use core::{fmt, time::Duration};

/// Stable identity of a registered sequence document.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct SequenceId(pub u128);
/// Stable identity of a clip, unique across the registry.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct SequenceClipId(pub u128);

/// Nonempty half-open integer tick interval [start, end).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SequenceRange {
    start: u64,
    end: u64,
}
impl SequenceRange {
    /// Rejects empty or reversed intervals.
    pub fn new(start: u64, end: u64) -> Result<Self, SequenceError> {
        if start >= end {
            return Err(SequenceError::InvalidRange);
        }
        Ok(Self { start, end })
    }
    /// Inclusive first tick.
    pub const fn start(self) -> u64 {
        self.start
    }
    /// Exclusive end tick.
    pub const fn end(self) -> u64 {
        self.end
    }
    fn len(self) -> u64 {
        self.end - self.start
    }
    fn contains(self, tick: u64) -> bool {
        self.start <= tick && tick < self.end
    }
}

/// One child instance on a parent document clock. Values are copied into the registry.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SequenceClip {
    /// Globally unique stable clip identity.
    pub id: SequenceClipId,
    /// Registered child sequence.
    pub source: SequenceId,
    /// Active interval measured in parent document ticks.
    pub parent: SequenceRange,
    /// Repeated interval measured in child document ticks.
    pub source_range: SequenceRange,
    /// Positive speed multiplier in addition to child-Hz / parent-Hz.
    pub rate: PlaybackRate,
    /// Finite number of source traversals. Parent interval can clip them early.
    pub repetitions: u64,
}

/// Explicit storage, traversal and output policies; zero output budgets are valid.
#[derive(Clone, Copy, Debug)]
pub struct SequenceLimits {
    /// Registered document references, at most 4096.
    pub max_sequences: usize,
    /// Total owned clips, at most 65536.
    pub max_clips: usize,
    /// Longest sequence path including root, at most 64.
    pub max_depth: usize,
    /// Node/clip visits per evaluation, at most one million.
    pub max_visits: usize,
    /// Event occurrences per frame, at most one million.
    pub max_events: usize,
    /// Total tracks inspected for destination sampling, at most one million.
    pub max_samples: usize,
    /// Actual codec validation policy for every borrowed document.
    pub document: TimelineLimits,
    /// Maximum encoded bytes for one borrowed document, at most 64 MiB.
    pub max_document_bytes: usize,
}
impl Default for SequenceLimits {
    fn default() -> Self {
        Self {
            max_sequences: 256,
            max_clips: 4096,
            max_depth: 16,
            max_visits: 16384,
            max_events: 4096,
            max_samples: 16384,
            document: TimelineLimits::default(),
            max_document_bytes: 8 * 1024 * 1024,
        }
    }
}

/// Rejected registry or evaluation operation; no partial player advancement.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SequenceError {
    /// Invalid resource policy.
    InvalidLimits,
    /// Empty or reversed tick range.
    InvalidRange,
    /// Repetition count must be positive.
    InvalidRepeat,
    /// Existing sequence identity.
    DuplicateSequence(SequenceId),
    /// Existing clip identity anywhere in the registry.
    DuplicateClip(SequenceClipId),
    /// Missing registered source or root.
    MissingSequence(SequenceId),
    /// Edge would introduce a cycle.
    Cycle,
    /// Named resource policy exceeded.
    Limit(&'static str),
    /// Exact integer calculation cannot be represented.
    ArithmeticOverflow,
    /// Real document validation or track evaluation failure.
    Timeline(TimelineError),
    /// Root transport rejected the operation.
    Playback(PlaybackError),
}
impl fmt::Display for SequenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "sequence {self:?}")
    }
}
impl core::error::Error for SequenceError {}
impl From<PlaybackError> for SequenceError {
    fn from(error: PlaybackError) -> Self {
        Self::Playback(error)
    }
}

struct Node<'a> {
    document: &'a TimelineDocument,
    clips: BTreeMap<SequenceClipId, SequenceClip>,
}
/// Owns bounded topology while borrowing immutable documents. A live player
/// borrows the registry, so document/topology edits require ending that borrow.
pub struct SequenceRegistry<'a> {
    limits: SequenceLimits,
    nodes: BTreeMap<SequenceId, Node<'a>>,
    clips: BTreeSet<SequenceClipId>,
}
impl<'a> SequenceRegistry<'a> {
    /// Creates an empty registry with hard caps on configurable budgets.
    pub fn new(limits: SequenceLimits) -> Result<Self, SequenceError> {
        if limits.max_sequences == 0
            || limits.max_sequences > 4096
            || limits.max_clips > 65536
            || limits.max_depth == 0
            || limits.max_depth > 64
            || limits.max_visits == 0
            || limits.max_visits > 1_000_000
            || limits.max_events > 1_000_000
            || limits.max_samples > 1_000_000
            || limits.max_document_bytes == 0
            || limits.max_document_bytes > 64 * 1024 * 1024
        {
            return Err(SequenceError::InvalidLimits);
        }
        Ok(Self {
            limits,
            nodes: BTreeMap::new(),
            clips: BTreeSet::new(),
        })
    }
    /// Immutable resource policy.
    pub const fn limits(&self) -> SequenceLimits {
        self.limits
    }
    /// Borrows the original document, without cloning payloads.
    pub fn document(&self, id: SequenceId) -> Option<&'a TimelineDocument> {
        self.nodes.get(&id).map(|node| node.document)
    }
    /// Clips in stable identity order.
    pub fn clips(
        &self,
        id: SequenceId,
    ) -> Result<impl Iterator<Item = &SequenceClip>, SequenceError> {
        Ok(self.node(id)?.clips.values())
    }
    /// Registers a borrowed document after real codec validation. The temporary
    /// bounded encoding is discarded; no snapshot or document clone is retained.
    pub fn insert(
        &mut self,
        id: SequenceId,
        document: &'a TimelineDocument,
    ) -> Result<(), SequenceError> {
        if self.nodes.contains_key(&id) {
            return Err(SequenceError::DuplicateSequence(id));
        }
        if self.nodes.len() >= self.limits.max_sequences {
            return Err(SequenceError::Limit("sequences"));
        }
        document
            .encode(self.limits.document, self.limits.max_document_bytes)
            .map_err(SequenceError::Timeline)?;
        self.nodes.insert(
            id,
            Node {
                document,
                clips: BTreeMap::new(),
            },
        );
        Ok(())
    }
    /// Adds a validated edge atomically. Missing references, cycles, depth,
    /// repeated identity or unrepresentable remapping preserve existing topology.
    pub fn add_clip(
        &mut self,
        parent: SequenceId,
        clip: SequenceClip,
    ) -> Result<(), SequenceError> {
        let parent_hz = self.node(parent)?.document.ticks_per_second;
        let source_hz = self.node(clip.source)?.document.ticks_per_second;
        if self.clips.contains(&clip.id) {
            return Err(SequenceError::DuplicateClip(clip.id));
        }
        if self.clips.len() >= self.limits.max_clips {
            return Err(SequenceError::Limit("clips"));
        }
        if clip.repetitions == 0 {
            return Err(SequenceError::InvalidRepeat);
        }
        let clock = Clock::new(&clip, parent_hz, source_hz);
        clock.forward(clip.parent.len() - 1)?;
        let id = clip.id;
        self.nodes
            .get_mut(&parent)
            .expect("checked node")
            .clips
            .insert(id, clip);
        if let Err(error) = self.validate_graph() {
            self.nodes
                .get_mut(&parent)
                .expect("checked node")
                .clips
                .remove(&id);
            return Err(error);
        }
        self.clips.insert(id);
        Ok(())
    }
    fn node(&self, id: SequenceId) -> Result<&Node<'a>, SequenceError> {
        self.nodes
            .get(&id)
            .ok_or(SequenceError::MissingSequence(id))
    }
    fn validate_graph(&self) -> Result<(), SequenceError> {
        let mut visiting = BTreeSet::new();
        let mut depths = BTreeMap::new();
        for id in self.nodes.keys() {
            self.depth(*id, &mut visiting, &mut depths)?;
        }
        Ok(())
    }
    fn depth(
        &self,
        id: SequenceId,
        visiting: &mut BTreeSet<SequenceId>,
        depths: &mut BTreeMap<SequenceId, usize>,
    ) -> Result<usize, SequenceError> {
        if visiting.contains(&id) {
            return Err(SequenceError::Cycle);
        }
        if let Some(depth) = depths.get(&id) {
            return Ok(*depth);
        }
        if visiting.len() >= self.limits.max_depth {
            return Err(SequenceError::Limit("depth"));
        }
        visiting.insert(id);
        let mut depth = 1;
        for clip in self.node(id)?.clips.values() {
            depth = depth.max(1 + self.depth(clip.source, visiting, depths)?);
            if depth > self.limits.max_depth {
                return Err(SequenceError::Limit("depth"));
            }
        }
        visiting.remove(&id);
        depths.insert(id, depth);
        Ok(depth)
    }
}

/// One occurrence of a stable clip in a finite repeat traversal.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct SequenceInstance {
    /// Clip identity.
    pub clip: SequenceClipId,
    /// Zero-based source traversal.
    pub iteration: u64,
}
/// Real destination samples for one active document instance. Overlaps remain
/// separate instances; providers decide blending and domain interpretation.
#[derive(Clone, Debug, PartialEq)]
pub struct SequenceSample<'a> {
    /// Stable registered sequence.
    pub sequence: SequenceId,
    /// Original borrowed document.
    pub document: &'a TimelineDocument,
    /// Root has an empty path; child paths disambiguate reused documents.
    pub path: Vec<SequenceInstance>,
    /// Integer local source tick after remapping.
    pub tick: u64,
    /// Actual track samples.
    pub tracks: BTreeMap<TrackId, Vec<f64>>,
}
/// One real event occurrence, carrying no camera/audio-specific policy.
#[derive(Clone, Debug, PartialEq)]
pub struct SequenceEvent<'a> {
    /// Source sequence.
    pub sequence: SequenceId,
    /// Original borrowed document.
    pub document: &'a TimelineDocument,
    /// Original event; name and stable ID are never synthesized.
    pub event: &'a TimelineEvent,
    /// Instance/repetition path.
    pub path: Vec<SequenceInstance>,
    /// Earliest root tick crossing this event, including skipped local ticks.
    pub root_tick: u64,
    /// Root transport loop generation, reset by seek.
    pub root_iteration: u128,
}
/// Bounded complete evaluation, ready for explicit provider dispatch.
#[derive(Clone, Debug, PartialEq)]
pub struct SequenceFrame<'a> {
    /// Root destination tick.
    pub tick: u64,
    /// Root loop traversal.
    pub iteration: u128,
    /// Transport status after evaluation.
    pub status: PlaybackStatus,
    /// Active destination instances in stable path order.
    pub samples: Vec<SequenceSample<'a>>,
    /// Events ordered by root iteration/tick, path, then source tick/event ID.
    pub events: Vec<SequenceEvent<'a>>,
}
/// Domain-neutral adapter invoked only on explicit successful frame dispatch.
pub trait SequenceProvider {
    /// Provider-owned failure type and side-effect policy.
    type Error;
    /// Apply an actual document instance's numeric channels.
    fn sample(&mut self, sample: &SequenceSample<'_>) -> Result<(), Self::Error>;
    /// Consume an actual authored event occurrence.
    fn event(&mut self, event: &SequenceEvent<'_>) -> Result<(), Self::Error>;
}
impl SequenceFrame<'_> {
    /// Dispatches destination samples followed by ordered events. Provider errors
    /// stop dispatch; already applied external side effects are not rolled back.
    pub fn dispatch<P: SequenceProvider>(&self, provider: &mut P) -> Result<(), P::Error> {
        for sample in &self.samples {
            provider.sample(sample)?;
        }
        for event in &self.events {
            provider.event(event)?;
        }
        Ok(())
    }
}

/// Root transport bound to immutable registry topology. All failures preserve
/// position, fractional time and event cursor, and return no partial frame.
pub struct SequencePlayer<'r, 'a> {
    registry: &'r SequenceRegistry<'a>,
    root: SequenceId,
    playback: Playback,
    origin_pending: bool,
}
impl<'r, 'a> SequencePlayer<'r, 'a> {
    /// Creates a paused root clock. Child frequency conversion is automatic.
    pub fn new(
        registry: &'r SequenceRegistry<'a>,
        root: SequenceId,
        range: SequenceRange,
        mode: PlaybackMode,
    ) -> Result<Self, SequenceError> {
        let hz = registry.node(root)?.document.ticks_per_second;
        Ok(Self {
            registry,
            root,
            playback: Playback::new(PlaybackConfig {
                ticks_per_second: hz,
                start_tick: range.start,
                end_tick: range.end - 1,
                mode,
                max_events: registry.limits.max_events,
            })?,
            origin_pending: true,
        })
    }
    /// Current root tick.
    pub const fn tick(&self) -> u64 {
        self.playback.tick()
    }
    /// Root transport status.
    pub const fn status(&self) -> PlaybackStatus {
        self.playback.status()
    }
    /// Root loop generation, reset by seek.
    pub const fn iteration(&self) -> u128 {
        self.playback.iteration()
    }
    /// Current root speed multiplier.
    pub const fn rate(&self) -> PlaybackRate {
        self.playback.rate()
    }
    /// Exact fractional root tick.
    pub const fn fractional_tick(&self) -> (u128, u128) {
        self.playback.fractional_tick()
    }
    /// Positive root speed; changing it resets only the fractional tick.
    pub fn set_rate(&mut self, rate: PlaybackRate) {
        self.playback.set_rate(rate);
    }
    /// Starts/resumes, unless a one-shot root has finished.
    pub fn play(&mut self) -> bool {
        self.playback.play()
    }
    /// Pauses without consuming time or rearming events.
    pub fn pause(&mut self) {
        self.playback.pause();
    }
    /// Samples destination without crossed events; re-arms only the actual local
    /// destination events for the next playing advance, even at zero elapsed.
    pub fn seek(&mut self, tick: u64) -> Result<SequenceFrame<'a>, SequenceError> {
        self.check_root_sample_budget()?;
        let mut next = self.playback.clone();
        next.seek(&self.registry.node(self.root)?.document.timeline, tick)?;
        let mut evaluation = Evaluation::new(self.registry);
        evaluation.samples(self.root, tick, &mut Vec::new())?;
        self.playback = next;
        self.origin_pending = true;
        Ok(self.frame(evaluation))
    }
    /// Consumes caller elapsed time using the real root transport. Collects all
    /// nested events crossed by fast clocks, loops and source range boundaries.
    pub fn advance(&mut self, elapsed: Duration) -> Result<SequenceFrame<'a>, SequenceError> {
        self.check_root_sample_budget()?;
        let mut next = self.playback.clone();
        next.advance(&self.registry.node(self.root)?.document.timeline, elapsed)?;
        let mut evaluation = Evaluation::new(self.registry);
        if self.status() == PlaybackStatus::Playing {
            if self.origin_pending {
                evaluation.point(
                    self.root,
                    self.tick(),
                    &mut Vec::new(),
                    self.tick(),
                    self.playback.iteration(),
                )?;
            }
            let old_iteration = self.playback.iteration();
            let new_iteration = next.iteration();
            if old_iteration == new_iteration {
                if next.tick() > self.tick() {
                    evaluation.range(
                        self.root,
                        self.tick() + 1,
                        next.tick(),
                        &mut Vec::new(),
                        old_iteration,
                    )?;
                }
            } else {
                let config = self.playback.config();
                if self.tick() < config.end_tick {
                    evaluation.range(
                        self.root,
                        self.tick() + 1,
                        config.end_tick,
                        &mut Vec::new(),
                        old_iteration,
                    )?;
                }
                // Each root traversal costs a visit, including event-free loops.
                if new_iteration - old_iteration > self.registry.limits.max_visits as u128 {
                    return Err(SequenceError::Limit("visits"));
                }
                for iteration in old_iteration + 1..=new_iteration {
                    let end = if iteration == new_iteration {
                        next.tick()
                    } else {
                        config.end_tick
                    };
                    evaluation.range(
                        self.root,
                        config.start_tick,
                        end,
                        &mut Vec::new(),
                        iteration,
                    )?;
                }
            }
        }
        evaluation.samples(self.root, next.tick(), &mut Vec::new())?;
        evaluation.events.sort_by(|a, b| {
            (
                a.root_iteration,
                a.root_tick,
                &a.path,
                a.event.tick,
                a.event.id,
            )
                .cmp(&(
                    b.root_iteration,
                    b.root_tick,
                    &b.path,
                    b.event.tick,
                    b.event.id,
                ))
        });
        if self.status() == PlaybackStatus::Playing {
            self.origin_pending = false;
        }
        self.playback = next;
        Ok(self.frame(evaluation))
    }
    fn check_root_sample_budget(&self) -> Result<(), SequenceError> {
        // The existing transport also samples its real root timeline. Check its
        // allocation budget before invoking it, as well as in nested evaluation.
        if self.registry.nodes[&self.root]
            .document
            .timeline
            .track_count()
            > self.registry.limits.max_samples
        {
            return Err(SequenceError::Limit("samples"));
        }
        Ok(())
    }
    fn frame(&self, evaluation: Evaluation<'_, 'a>) -> SequenceFrame<'a> {
        SequenceFrame {
            tick: self.tick(),
            iteration: self.playback.iteration(),
            status: self.status(),
            samples: evaluation.samples,
            events: evaluation.events,
        }
    }
}

#[derive(Clone, Copy)]
struct Clock {
    numerator: u128,
    denominator: u128,
}
impl Clock {
    fn new(clip: &SequenceClip, parent_hz: u64, source_hz: u64) -> Self {
        let (n, d) = clip.rate.ratio();
        Self {
            numerator: source_hz as u128 * n as u128,
            denominator: parent_hz as u128 * d as u128,
        }
    }
    fn forward(self, offset: u64) -> Result<u128, SequenceError> {
        Ok(mul_div(offset as u128, self.numerator, self.denominator)?.0)
    }
    fn inverse(self, offset: u128) -> Result<u64, SequenceError> {
        let (whole, remainder) = mul_div(offset, self.denominator, self.numerator)?;
        let rounded = whole
            .checked_add(u128::from(remainder != 0))
            .ok_or(SequenceError::ArithmeticOverflow)?;
        u64::try_from(rounded).map_err(|_| SequenceError::ArithmeticOverflow)
    }
}
// Exact quotient/remainder multiplication, without requiring a 256-bit product.
fn mul_div(value: u128, factor: u128, denominator: u128) -> Result<(u128, u128), SequenceError> {
    let (base, fraction) = (value / denominator, value % denominator);
    let (mut whole, mut remainder) = (0u128, 0u128);
    for bit in (0..128 - factor.leading_zeros()).rev() {
        let (carry, next) = add_mod(remainder, remainder, denominator);
        remainder = next;
        whole = whole
            .checked_mul(2)
            .and_then(|n| n.checked_add(carry))
            .ok_or(SequenceError::ArithmeticOverflow)?;
        if factor & (1u128 << bit) != 0 {
            let (carry, next) = add_mod(remainder, fraction, denominator);
            remainder = next;
            whole = whole
                .checked_add(base)
                .and_then(|n| n.checked_add(carry))
                .ok_or(SequenceError::ArithmeticOverflow)?;
        }
    }
    Ok((whole, remainder))
}
fn add_mod(a: u128, b: u128, modulus: u128) -> (u128, u128) {
    if a >= modulus - b {
        (1, a - (modulus - b))
    } else {
        (0, a + b)
    }
}
struct Edge<'a> {
    clip: &'a SequenceClip,
    clock: Clock,
    iteration: u64,
}
struct Evaluation<'r, 'a> {
    registry: &'r SequenceRegistry<'a>,
    visits: usize,
    inspected_tracks: usize,
    samples: Vec<SequenceSample<'a>>,
    events: Vec<SequenceEvent<'a>>,
}
impl<'r, 'a> Evaluation<'r, 'a> {
    fn new(registry: &'r SequenceRegistry<'a>) -> Self {
        Self {
            registry,
            visits: 0,
            inspected_tracks: 0,
            samples: Vec::new(),
            events: Vec::new(),
        }
    }
    fn visit(&mut self) -> Result<(), SequenceError> {
        if self.visits >= self.registry.limits.max_visits {
            return Err(SequenceError::Limit("visits"));
        }
        self.visits += 1;
        Ok(())
    }
    fn path(edges: &[Edge<'_>]) -> Vec<SequenceInstance> {
        edges
            .iter()
            .map(|edge| SequenceInstance {
                clip: edge.clip.id,
                iteration: edge.iteration,
            })
            .collect()
    }
    fn clock(&self, parent: SequenceId, clip: &SequenceClip) -> Clock {
        Clock::new(
            clip,
            self.registry.nodes[&parent].document.ticks_per_second,
            self.registry.nodes[&clip.source].document.ticks_per_second,
        )
    }
    fn active(
        &self,
        parent: SequenceId,
        clip: &SequenceClip,
        tick: u64,
    ) -> Result<Option<(Clock, u64, u64)>, SequenceError> {
        if !clip.parent.contains(tick) {
            return Ok(None);
        }
        let clock = self.clock(parent, clip);
        let offset = clock.forward(tick - clip.parent.start)?;
        let length = clip.source_range.len() as u128;
        if offset >= length * clip.repetitions as u128 {
            return Ok(None);
        }
        Ok(Some((
            clock,
            (offset / length) as u64,
            clip.source_range.start + (offset % length) as u64,
        )))
    }
    fn samples(
        &mut self,
        id: SequenceId,
        tick: u64,
        edges: &mut Vec<Edge<'r>>,
    ) -> Result<(), SequenceError> {
        self.visit()?;
        let node = self.registry.node(id)?;
        let count = node.document.timeline.track_count();
        if count > self.registry.limits.max_samples - self.inspected_tracks {
            return Err(SequenceError::Limit("samples"));
        }
        self.inspected_tracks += count;
        let tracks = node
            .document
            .timeline
            .sample(tick)
            .map_err(SequenceError::Timeline)?;
        self.samples.push(SequenceSample {
            sequence: id,
            document: node.document,
            tick,
            path: Self::path(edges),
            tracks,
        });
        for clip in node.clips.values() {
            self.visit()?;
            if let Some((clock, iteration, local)) = self.active(id, clip, tick)? {
                edges.push(Edge {
                    clip,
                    clock,
                    iteration,
                });
                self.samples(clip.source, local, edges)?;
                edges.pop();
            }
        }
        Ok(())
    }
    fn event(
        &mut self,
        id: SequenceId,
        event: &'a TimelineEvent,
        edges: &[Edge<'_>],
        root_tick: u64,
        root_iteration: u128,
    ) -> Result<(), SequenceError> {
        if self.events.len() >= self.registry.limits.max_events {
            return Err(SequenceError::Limit("events"));
        }
        self.events.push(SequenceEvent {
            sequence: id,
            document: self.registry.nodes[&id].document,
            event,
            path: Self::path(edges),
            root_tick,
            root_iteration,
        });
        Ok(())
    }
    fn point(
        &mut self,
        id: SequenceId,
        tick: u64,
        edges: &mut Vec<Edge<'r>>,
        root_tick: u64,
        iteration: u128,
    ) -> Result<(), SequenceError> {
        self.visit()?;
        let node = self.registry.node(id)?;
        for event in node.document.timeline.events_at(tick) {
            self.event(id, event, edges, root_tick, iteration)?;
        }
        for clip in node.clips.values() {
            self.visit()?;
            if let Some((clock, repeat, local)) = self.active(id, clip, tick)? {
                edges.push(Edge {
                    clip,
                    clock,
                    iteration: repeat,
                });
                self.point(clip.source, local, edges, root_tick, iteration)?;
                edges.pop();
            }
        }
        Ok(())
    }
    fn root_tick(mut tick: u64, edges: &[Edge<'_>]) -> Result<u64, SequenceError> {
        for edge in edges.iter().rev() {
            let offset = edge.iteration as u128 * edge.clip.source_range.len() as u128
                + (tick - edge.clip.source_range.start) as u128;
            tick = edge
                .clip
                .parent
                .start
                .checked_add(edge.clock.inverse(offset)?)
                .ok_or(SequenceError::ArithmeticOverflow)?;
        }
        Ok(tick)
    }
    fn range(
        &mut self,
        id: SequenceId,
        start: u64,
        end: u64,
        edges: &mut Vec<Edge<'r>>,
        iteration: u128,
    ) -> Result<(), SequenceError> {
        self.visit()?;
        let node = self.registry.node(id)?;
        for event in node.document.timeline.events_in(start..=end) {
            self.event(
                id,
                event,
                edges,
                Self::root_tick(event.tick, edges)?,
                iteration,
            )?;
        }
        for clip in node.clips.values() {
            self.visit()?;
            let first = start.max(clip.parent.start);
            let last = end.min(clip.parent.end - 1);
            if first > last {
                continue;
            }
            let clock = self.clock(id, clip);
            let low = if first == clip.parent.start {
                0
            } else {
                clock
                    .forward(first - clip.parent.start - 1)?
                    .checked_add(1)
                    .ok_or(SequenceError::ArithmeticOverflow)?
            };
            let length = clip.source_range.len() as u128;
            let high = clock
                .forward(last - clip.parent.start)?
                .min(length * clip.repetitions as u128 - 1);
            if low > high {
                continue;
            }
            for repeat in low / length..=high / length {
                let local_start = if repeat == low / length {
                    (low % length) as u64
                } else {
                    0
                };
                let local_end = if repeat == high / length {
                    (high % length) as u64
                } else {
                    clip.source_range.len() - 1
                };
                edges.push(Edge {
                    clip,
                    clock,
                    iteration: repeat as u64,
                });
                self.range(
                    clip.source,
                    clip.source_range.start + local_start,
                    clip.source_range.start + local_end,
                    edges,
                    iteration,
                )?;
                edges.pop();
            }
        }
        Ok(())
    }
}
