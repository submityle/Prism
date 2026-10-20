//! Native dock contributions, retaining the host's actual models/controllers.
use super::*;
use alloc::collections::BTreeSet;
use prism_editor_timeline::{
    PlaybackStatus, TimelineEditor, TimelineEvent, TimelineSnapshotToken, TrackId,
};
use prism_editor_ui::{
    BoundProfilerPanel, BoundTimelinePanel, BoundWorldPanel, ProfilerController, ProfilerSources,
    TimelinePanelController, WorldPanelController,
};
use std::{
    any::Any,
    time::{Duration, Instant},
};

const TAB_HEIGHT: f32 = 28.0;
const TICK: Duration = Duration::from_millis(16);

/// A real Loom subtree and its semantic targets. Explicit string keys must be
/// unique within the tree and avoid the editor/content/native namespaces.
pub struct DesktopPanelView {
    /// Actual keyed Loom tree.
    pub element: Element,
    /// Exact target/trigger pairs supported by this snapshot.
    pub bindings: Vec<(String, UiTrigger)>,
}
impl DesktopPanelView {
    fn validate(&self) -> Result<(), DesktopUiError> {
        let invalid = || {
            DesktopUiError::Panel(
                "Invalid, duplicate or excessive native panel tree/bindings".into(),
            )
        };
        if self.bindings.len() > 4096 {
            return Err(invalid());
        }
        let mut keys = BTreeSet::new();
        let mut stack = vec![(&self.element, 0usize)];
        let mut count = 0usize;
        while let Some((element, depth)) = stack.pop() {
            count += 1;
            if count > 16_384 || depth > 128 {
                return Err(invalid());
            }
            if let Some(Key::Str(key)) = element.explicit_key()
                && (key.len() > MAX_ACTION_TEXT_BYTES
                    || key.starts_with("editor.")
                    || key.starts_with("content:")
                    || key.starts_with("native:")
                    || !keys.insert(key.as_str()))
            {
                return Err(invalid());
            }
            stack.extend(
                element
                    .child_elements()
                    .iter()
                    .map(|child| (child, depth + 1)),
            );
        }
        let unique: BTreeSet<_> = self.bindings.iter().collect();
        if unique.len() != self.bindings.len()
            || self
                .bindings
                .iter()
                .any(|(key, _)| !keys.contains(key.as_str()))
        {
            return Err(invalid());
        }
        Ok(())
    }
}

/// Result of host acquisition. `fenced` means unsaved data, pending I/O or an
/// unacknowledged failure. A future deadline wakes native `WaitUntil`, even hidden.
#[derive(Clone, Copy, Debug, Default)]
pub struct DesktopPanelUpdate {
    /// Published controller or source changes requiring reconciliation.
    pub changed: bool,
    /// Unsaved host data, outstanding I/O or unresolved publication errors.
    pub fenced: bool,
    /// Next nonblocking acquisition deadline on the host monotonic clock.
    pub wake_at: Option<Instant>,
}

/// A bounded native dock slot. Builds real elements and handles their exact
/// bindings. Poll must be nonblocking; completion is reported by the host.
pub trait DesktopPanel: Any {
    /// Build the actual view and retain its freshness evidence for dispatch.
    fn build(&mut self) -> Result<DesktopPanelView, String>;
    /// Dispatch against that exact view, preserving data on failure.
    fn event(&mut self, target: &str, event: &UiEvent) -> Result<(), String>;
    /// Acquire completed host work without blocking the native event loop.
    fn poll(&mut self, _now: Instant) -> Result<bool, String> {
        Ok(false)
    }
    /// Next required native wake; None permits idle sleep.
    fn next_wake(&self) -> Option<Instant> {
        None
    }
    /// Prevent scene save/reload/close and guarded removal while data is unsafe.
    fn fenced(&self) -> bool {
        false
    }
    /// Authoring stamp only; transport ticks must not invalidate text drafts.
    fn draft_revision(&self) -> u64 {
        0
    }
}

/// Owned delivery retained verbatim if the host rejects a frame. Events carry
/// their actual source identities and loop iteration, not synthesized commands.
#[derive(Clone, Debug, PartialEq)]
pub struct DesktopTimelineFrame {
    /// Actual destination tick.
    pub tick: u64,
    /// Destination traversal number.
    pub iteration: u128,
    /// Transport status after advancement.
    pub status: PlaybackStatus,
    /// Actual numeric channel samples, keyed by host track identity.
    pub samples: BTreeMap<TrackId, Vec<f64>>,
    /// Crossed source events paired with their traversal numbers.
    pub events: Vec<(TimelineEvent, u128)>,
}
type TimelineConsumer = dyn FnMut(&DesktopTimelineFrame) -> Result<(), String>;

/// Host-supplied authoring document, real UI transport controller and consumer.
/// A consumer returning Err must accept nothing; retry delivers the same frame.
pub struct DesktopTimelinePanel {
    /// Host authoring model; access mutable state through `native_panel_mut`.
    pub editor: TimelineEditor,
    /// Actual sequencer controller, including Playback.
    pub controller: TimelinePanelController,
    consumer: Box<TimelineConsumer>,
    pub(super) view: Option<BoundTimelinePanel>,
    saved: TimelineSnapshotToken,
    pub(super) anchor: Option<Instant>,
    pub(super) wake: Option<Instant>,
    pending: Option<DesktopTimelineFrame>,
    pub(super) failed: bool,
    pub(super) deliver: bool,
}
impl DesktopTimelinePanel {
    /// Mount existing authoring/transport state with an all-or-nothing consumer.
    pub fn new(
        editor: TimelineEditor,
        controller: TimelinePanelController,
        consumer: impl FnMut(&DesktopTimelineFrame) -> Result<(), String> + 'static,
    ) -> Self {
        let saved = editor.snapshot_token();
        Self {
            editor,
            controller,
            consumer: Box::new(consumer),
            saved,
            view: None,
            anchor: None,
            wake: None,
            pending: None,
            failed: false,
            deliver: true,
        }
    }
    /// Call only after persisting this exact authoring snapshot successfully.
    pub fn acknowledge_saved(&mut self, token: &TimelineSnapshotToken) -> bool {
        if *token != self.editor.snapshot_token() {
            return false;
        }
        self.saved = token.clone();
        true
    }
    /// Explicitly retry a rejected delivery or failed advance. No elapsed time
    /// is silently discarded; advance errors retain the old clock anchor.
    pub fn retry(&mut self) {
        self.failed = false;
        self.wake = Some(Instant::now());
    }
    /// Failed frame retained until the consumer accepts it through retry.
    pub fn pending_frame(&self) -> Option<&DesktopTimelineFrame> {
        self.pending.as_ref()
    }
}
impl DesktopPanel for DesktopTimelinePanel {
    fn build(&mut self) -> Result<DesktopPanelView, String> {
        let view = self
            .controller
            .build(&self.editor)
            .map_err(|e| e.to_string())?;
        let result = DesktopPanelView {
            element: view.element().clone(),
            bindings: view.bindings().map(|(k, t)| (k.to_owned(), t)).collect(),
        };
        self.view = Some(view);
        Ok(result)
    }
    fn event(&mut self, target: &str, event: &UiEvent) -> Result<(), String> {
        if self.failed || self.pending.is_some() {
            return Err("Timeline delivery failed; retry the retained frame before editing".into());
        }
        let playing = self.controller.playback().status() == PlaybackStatus::Playing;
        self.controller
            .handle_event(
                &mut self.editor,
                self.view.as_ref().ok_or("Timeline is not rendered")?,
                target,
                event,
            )
            .map_err(|e| e.to_string())?;
        // Preserve elapsed time on ordinary authoring/navigation events. A play
        // transition starts a fresh anchor; paused time is never replayed.
        if !playing || self.controller.playback().status() != PlaybackStatus::Playing {
            self.anchor = None;
        }
        self.deliver = true;
        self.wake = Some(Instant::now());
        Ok(())
    }
    fn poll(&mut self, now: Instant) -> Result<bool, String> {
        if self.failed || self.controller.drag_preview().is_some() {
            return Ok(false);
        }
        if self.anchor.is_some_and(|anchor| now < anchor) {
            self.failed = true;
            self.wake = None;
            return Err("Timeline monotonic clock moved backwards".into());
        }
        if self.pending.is_none() {
            let playing = self.controller.playback().status() == PlaybackStatus::Playing;
            if !self.deliver && (!playing || self.wake.is_some_and(|w| now < w)) {
                return Ok(false);
            }
            let elapsed = if playing {
                match self.anchor {
                    Some(old) => match now.checked_duration_since(old) {
                        Some(elapsed) => elapsed,
                        None => {
                            self.failed = true;
                            self.wake = None;
                            return Err("Timeline monotonic clock moved backwards".into());
                        }
                    },
                    None => Duration::ZERO,
                }
            } else {
                Duration::ZERO
            };
            let frame = match self.controller.advance(&self.editor, elapsed) {
                Ok(f) => f,
                Err(e) => {
                    self.failed = true;
                    self.wake = None;
                    return Err(e.to_string());
                }
            };
            self.pending = Some(DesktopTimelineFrame {
                tick: frame.tick,
                iteration: frame.iteration,
                status: frame.status,
                samples: frame.samples,
                events: frame
                    .events
                    .into_iter()
                    .map(|e| (e.event.clone(), e.iteration))
                    .collect(),
            });
            self.anchor = Some(now);
        }
        if let Err(error) = (self.consumer)(self.pending.as_ref().expect("pending frame")) {
            self.failed = true;
            self.wake = None;
            return Err(error);
        }
        self.pending = None;
        self.deliver = false;
        self.wake =
            (self.controller.playback().status() == PlaybackStatus::Playing).then(|| now + TICK);
        Ok(true)
    }
    fn next_wake(&self) -> Option<Instant> {
        self.wake
    }
    fn fenced(&self) -> bool {
        self.failed
            || self.pending.is_some()
            || self.controller.drag_preview().is_some()
            || self.saved != self.editor.snapshot_token()
    }
    fn draft_revision(&self) -> u64 {
        self.editor.revision()
    }
}

type WorldProvider =
    dyn FnMut(&mut WorldPanelController, Instant) -> Result<DesktopPanelUpdate, String>;
/// The provider owns I/O policy and must retain real intent/transition/save
/// tokens until completion. App never turns an intent into a success receipt.
pub struct DesktopWorldPanel {
    /// Actual world controller, including its authoring data and intent queue.
    pub controller: WorldPanelController,
    provider: Box<WorldProvider>,
    view: Option<BoundWorldPanel>,
    update: DesktopPanelUpdate,
    failed: bool,
}
impl DesktopWorldPanel {
    /// Attach the host's real world/I/O provider. It must report metadata dirty
    /// state and pending publication using `DesktopPanelUpdate::fenced`.
    pub fn new(
        controller: WorldPanelController,
        provider: impl FnMut(&mut WorldPanelController, Instant) -> Result<DesktopPanelUpdate, String>
            + 'static,
    ) -> Self {
        Self {
            controller,
            provider: Box::new(provider),
            view: None,
            update: DesktopPanelUpdate::default(),
            failed: false,
        }
    }
    /// Resume provider acquisition after the host resolves an error.
    pub fn retry(&mut self) {
        self.failed = false;
    }
}
impl DesktopPanel for DesktopWorldPanel {
    fn build(&mut self) -> Result<DesktopPanelView, String> {
        let view = self.controller.build();
        let result = DesktopPanelView {
            element: view.element().clone(),
            bindings: view.bindings().map(|(k, t)| (k.to_owned(), t)).collect(),
        };
        self.view = Some(view);
        Ok(result)
    }
    fn event(&mut self, target: &str, event: &UiEvent) -> Result<(), String> {
        if self.failed {
            return Err("World provider failed; retry before issuing another operation".into());
        }
        self.controller
            .handle_event(
                self.view.as_ref().ok_or("World is not rendered")?,
                target,
                event,
            )
            .map_err(|e| e.to_string())
    }
    fn poll(&mut self, now: Instant) -> Result<bool, String> {
        if self.failed {
            return Ok(false);
        }
        match (self.provider)(&mut self.controller, now) {
            Ok(update) => {
                self.update = update;
                Ok(update.changed)
            }
            Err(e) => {
                self.failed = true;
                Err(e)
            }
        }
    }
    fn next_wake(&self) -> Option<Instant> {
        if self.failed {
            None
        } else {
            self.update.wake_at
        }
    }
    fn fenced(&self) -> bool {
        self.failed
            || self.update.fenced
            || self.controller.pending_intents() != 0
            || self
                .controller
                .world()
                .model()
                .cells()
                .any(|cell| cell.is_dirty() || cell.residency().is_pending())
    }
}

type ProfilerProvider =
    dyn FnMut(Instant) -> Result<(Option<ProfilerSources>, DesktopPanelUpdate), String>;
/// Publishes actual completed capture snapshots, keeping CPU/GPU clocks distinct.
pub struct DesktopProfilerPanel {
    /// Actual snapshot navigation controller.
    pub controller: ProfilerController,
    provider: Box<ProfilerProvider>,
    view: Option<BoundProfilerPanel>,
    update: DesktopPanelUpdate,
    failed: bool,
}
impl DesktopProfilerPanel {
    /// Attach a producer of immutable completed CPU/GPU capture stores.
    pub fn new(
        controller: ProfilerController,
        provider: impl FnMut(Instant) -> Result<(Option<ProfilerSources>, DesktopPanelUpdate), String>
            + 'static,
    ) -> Self {
        Self {
            controller,
            provider: Box::new(provider),
            view: None,
            update: DesktopPanelUpdate::default(),
            failed: false,
        }
    }
    /// Resume capture acquisition after the host resolves an error.
    pub fn retry(&mut self) {
        self.failed = false;
    }
}
impl DesktopPanel for DesktopProfilerPanel {
    fn build(&mut self) -> Result<DesktopPanelView, String> {
        let view = self.controller.build();
        let result = DesktopPanelView {
            element: view.element().clone(),
            bindings: view.bindings().map(|(k, t)| (k.to_owned(), t)).collect(),
        };
        self.view = Some(view);
        Ok(result)
    }
    fn event(&mut self, target: &str, event: &UiEvent) -> Result<(), String> {
        self.controller
            .handle_event(
                self.view.as_ref().ok_or("Profiler is not rendered")?,
                target,
                event,
            )
            .map(|_| ())
            .map_err(|e| e.to_string())
    }
    fn poll(&mut self, now: Instant) -> Result<bool, String> {
        if self.failed {
            return Ok(false);
        }
        match (self.provider)(now) {
            Ok((sources, update)) => {
                let changed = sources.is_some() || update.changed;
                if let Some(sources) = sources {
                    self.controller.publish(sources);
                }
                self.update = update;
                Ok(changed)
            }
            Err(e) => {
                self.failed = true;
                Err(e)
            }
        }
    }
    fn next_wake(&self) -> Option<Instant> {
        if self.failed {
            None
        } else {
            self.update.wake_at
        }
    }
    fn fenced(&self) -> bool {
        self.failed || self.update.fenced
    }
}

pub(super) struct Slot {
    pub(super) id: String,
    label: String,
    pub(super) host: Box<dyn DesktopPanel>,
    view: Option<DesktopPanelView>,
    scroll: f32,
    scroll_max: f32,
}
#[derive(Default)]
pub(super) struct NativePanels {
    pub(super) slots: Vec<Slot>,
    active: Option<String>,
    pub stale: bool,
    draft_stamp: Option<u64>,
    pub(super) geometry: BTreeMap<String, (Rect, Rect)>,
    pub(super) capture: Option<panel_pointer::PanelCapture>,
}
impl NativePanels {
    pub fn mounted(&self) -> bool {
        !self.slots.is_empty()
    }
    pub fn active(&self) -> bool {
        self.active.is_some()
    }
    pub(super) fn slot(&self) -> Option<&Slot> {
        self.slots
            .iter()
            .find(|s| Some(&s.id) == self.active.as_ref())
    }
    pub(super) fn slot_mut(&mut self) -> Option<&mut Slot> {
        self.slots
            .iter_mut()
            .find(|s| Some(&s.id) == self.active.as_ref())
    }
    pub fn fenced(&self) -> bool {
        self.slots.iter().any(|s| s.host.fenced())
    }
    pub fn rebuild(&mut self) -> Result<(), DesktopUiError> {
        if let Some(slot) = self.slot_mut() {
            let view = slot.host.build().map_err(DesktopUiError::Panel)?;
            view.validate()?;
            slot.view = Some(view);
        }
        Ok(())
    }
    pub fn bindings(&self) -> impl Iterator<Item = (&str, UiTrigger)> {
        self.slot()
            .into_iter()
            .flat_map(|s| s.view.as_ref())
            .flat_map(|v| v.bindings.iter().map(|(k, t)| (k.as_str(), *t)))
    }
    pub fn is_target(&self, target: &str) -> bool {
        self.bindings().any(|(k, _)| k == target)
    }
    pub fn tabs(&self, content: bool) -> Element {
        let mut tabs = Element::box_()
            .key_str("editor.paneltabs")
            .class("desktop-toolbar")
            .style(StyleProp::Height, StyleValue::px(TAB_HEIGHT));
        for (id, label) in content
            .then_some(("content", "Content"))
            .into_iter()
            .chain(self.slots.iter().map(|s| (s.id.as_str(), s.label.as_str())))
        {
            let selected = if id == "content" {
                self.active.is_none()
            } else {
                self.active.as_deref() == Some(id)
            };
            tabs = tabs.child(
                Element::box_()
                    .key_str(format!("native:tab:{id}"))
                    .class("desktop-button")
                    .child(Element::text(format!(
                        "{}{}",
                        if selected { "● " } else { "" },
                        label
                    ))),
            );
        }
        tabs
    }
    pub fn body(&self, height: f32) -> Option<Element> {
        let slot = self.slot()?;
        Some(
            Element::box_()
                .key_str("editor.nativepanel")
                .class("desktop-browser")
                .style(StyleProp::Height, StyleValue::px(height))
                .child(
                    Element::box_()
                        .key_str("native:body")
                        .style(StyleProp::FlexShrink, StyleValue::number(0.0))
                        .style(StyleProp::MarginTop, StyleValue::px(-slot.scroll))
                        .child(slot.view.as_ref()?.element.clone()),
                ),
        )
    }
    pub fn layout(&mut self, height: f32, body: f32) -> bool {
        if let Some(slot) = self.slot_mut() {
            slot.scroll_max = (body - height).max(0.0);
            if slot.scroll > slot.scroll_max {
                slot.scroll = slot.scroll_max;
                return true;
            }
        }
        false
    }
    pub fn scroll(&mut self, delta: i32) -> bool {
        let Some(slot) = self.slot_mut() else {
            return false;
        };
        let next = (slot.scroll + delta as f32 * 32.0).clamp(0.0, slot.scroll_max);
        let changed = next != slot.scroll;
        slot.scroll = next;
        changed
    }
}

impl EditorDesktopUi {
    /// At most eight slots; IDs are ASCII alphanumeric/hyphen, labels bounded.
    /// Takes the actual host object and returns it unchanged on guarded removal.
    pub fn mount_native_panel(
        &mut self,
        id: &str,
        label: &str,
        mut host: Box<dyn DesktopPanel>,
    ) -> Result<(), DesktopUiError> {
        if self.focus.is_some()
            || self.panels.fenced()
            || self.panels.slots.len() >= 8
            || id.is_empty()
            || id.len() > 48
            || id == "content"
            || !id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-')
            || label.is_empty()
            || label.len() > 64
            || label.chars().any(char::is_control)
            || self.panels.slots.iter().any(|s| s.id == id)
        {
            return self.reject(DesktopUiError::Panel(
                "Native panel mount rejected by identity, capacity or edit fence".into(),
            ));
        }
        let view = host.build().map_err(DesktopUiError::Panel)?;
        view.validate()?;
        self.panels.slots.push(Slot {
            id: id.into(),
            label: label.into(),
            host,
            view: Some(view),
            scroll: 0.0,
            scroll_max: 0.0,
        });
        if self.content.is_none() && self.panels.active.is_none() {
            self.panels.active = Some(id.into());
        }
        self.invalidated = true;
        self.refresh()?;
        Ok(())
    }
    /// Return the original host object only after its data and I/O are safe.
    pub fn unmount_native_panel(
        &mut self,
        id: &str,
    ) -> Result<Box<dyn DesktopPanel>, DesktopUiError> {
        let index = self
            .panels
            .slots
            .iter()
            .position(|s| s.id == id)
            .ok_or_else(|| DesktopUiError::Panel("Unknown native panel".into()))?;
        if self.focus.is_some() || self.panels.slots[index].host.fenced() {
            return self.reject(DesktopUiError::Panel(
                "Native panel has uncommitted data or I/O".into(),
            ));
        }
        let previous_active = self.panels.active.clone();
        let slot = self.panels.slots.remove(index);
        if self.panels.active.as_deref() == Some(id) {
            self.panels.active = if self.content.is_some() {
                None
            } else {
                self.panels.slots.first().map(|s| s.id.clone())
            };
        }
        self.invalidated = true;
        if let Err(error) = self.refresh() {
            // A remaining provider may reject its new view. Keep ownership of
            // the removed host until removal can actually be presented.
            self.panels.slots.insert(index, slot);
            self.panels.active = previous_active;
            self.panels.stale = true;
            self.invalidated = true;
            return self.reject(error);
        }
        Ok(slot.host)
    }
    /// Read the original typed host model/controller/provider without cloning.
    pub fn native_panel<T: DesktopPanel>(&self, id: &str) -> Option<&T> {
        (self.panels.slots.iter().find(|s| s.id == id)?.host.as_ref() as &dyn Any).downcast_ref()
    }
    /// Host mutation invalidates displayed geometry; drafts must be resolved first.
    pub fn native_panel_mut<T: DesktopPanel>(
        &mut self,
        id: &str,
    ) -> Result<&mut T, DesktopUiError> {
        if self.focus.is_some() {
            return Err(DesktopUiError::Panel(
                "Resolve the focused draft before host mutation".into(),
            ));
        }
        self.cancel_panel_pointer()?;
        self.invalidated = true;
        self.panels.stale = true;
        (self
            .panels
            .slots
            .iter_mut()
            .find(|s| s.id == id)
            .ok_or(DesktopUiError::InvalidProjection)?
            .host
            .as_mut() as &mut dyn Any)
            .downcast_mut()
            .ok_or(DesktopUiError::InvalidProjection)
    }
    /// Poll from a monotonic host clock; never requires a GPU frame or visible tab.
    pub fn poll_native_panels(&mut self, now: Instant) -> Result<bool, DesktopUiError> {
        let mut changed = false;
        let mut visible = false;
        let mut error = None;
        for slot in &mut self.panels.slots {
            match slot.host.poll(now) {
                Ok(value) => {
                    changed |= value;
                    visible |= value && self.panels.active.as_ref() == Some(&slot.id);
                }
                Err(e) => {
                    changed = true;
                    visible = true;
                    error = Some(e);
                }
            }
        }
        self.invalidated |= visible;
        self.panels.stale |= visible;
        if let Some(e) = error {
            return self.reject(DesktopUiError::Panel(e));
        }
        Ok(changed)
    }
    /// Earliest transport/acquisition deadline, including hidden panels.
    pub fn next_native_panel_wake(&self) -> Option<Instant> {
        self.panels
            .slots
            .iter()
            .filter_map(|s| s.host.next_wake())
            .min()
    }
    /// Active contribution ID; None selects the content browser.
    pub fn active_native_panel(&self) -> Option<&str> {
        self.panels.active.as_deref()
    }
    /// Switch visible content without dropping drafts or hidden source state.
    pub fn switch_native_panel(&mut self, id: Option<&str>) -> Result<(), DesktopUiError> {
        if self.focus.is_some()
            || self.panel_pointer_active()
            || id.is_none() && self.content.is_none()
            || id.is_some_and(|id| !self.panels.slots.iter().any(|s| s.id == id))
        {
            return self.reject(DesktopUiError::Panel(
                "Resolve the draft or choose a mounted panel".into(),
            ));
        }
        self.panels.active = id.map(str::to_owned);
        self.invalidated = true;
        self.refresh()?;
        Ok(())
    }
    pub(super) fn panel_event(
        &mut self,
        target: &str,
        event: &UiEvent,
    ) -> Result<bool, DesktopUiError> {
        if let Some(id) = target.strip_prefix("native:tab:") {
            self.switch_native_panel((id != "content").then_some(id))?;
            return Ok(true);
        }
        if self.focus.is_some() && !matches!(event, UiEvent::Commit { .. }) {
            return self.reject(DesktopUiError::Panel(
                "Resolve the panel draft before another action".into(),
            ));
        }
        let stamp = self.panels.draft_stamp;
        let slot = self
            .panels
            .slot_mut()
            .ok_or(DesktopUiError::InvalidProjection)?;
        if matches!(event, UiEvent::Commit { .. }) && stamp != Some(slot.host.draft_revision()) {
            return self.reject(DesktopUiError::Panel(
                "Panel authoring changed; cancel and reopen the draft".into(),
            ));
        }
        if let Err(e) = slot.host.event(target, event) {
            return self.reject(DesktopUiError::Panel(e));
        }
        self.focus = None;
        self.panels.draft_stamp = None;
        self.invalidated = true;
        self.refresh()?;
        Ok(true)
    }
    pub(super) fn focus_panel(&mut self, target: &str) -> Result<(), DesktopUiError> {
        fn value(el: &Element, target: &str) -> Option<String> {
            if matches!(el.explicit_key(),Some(Key::Str(k)) if k==target) {
                let value = el
                    .child_elements()
                    .iter()
                    .find(|e| e.class_names().iter().any(|c| is_input_value(c)))?;
                return Some(value.text_content().unwrap_or("").to_owned());
            }
            el.child_elements().iter().find_map(|e| value(e, target))
        }
        let slot = self
            .panels
            .slot()
            .ok_or(DesktopUiError::InvalidProjection)?;
        let buffer = value(
            &slot
                .view
                .as_ref()
                .ok_or(DesktopUiError::InvalidProjection)?
                .element,
            target,
        )
        .ok_or(DesktopUiError::InvalidProjection)?;
        self.panels.draft_stamp = Some(slot.host.draft_revision());
        self.focus = Some(Focus {
            target: target.into(),
            buffer,
            preedit: String::new(),
            replace: true,
        });
        self.invalidated = true;
        Ok(())
    }
    pub(super) fn panel_tabs_height(&self) -> f32 {
        if self.panels.mounted() {
            TAB_HEIGHT
        } else {
            0.0
        }
    }
    pub(super) fn panel_body_height(&self) -> f32 {
        (self.content_height() - self.panel_tabs_height()).max(0.0)
    }
}
