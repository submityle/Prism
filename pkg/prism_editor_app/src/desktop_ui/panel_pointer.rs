//! Capture uses the same retained logical rectangles as painting and clipping.
use super::*;
use prism_editor_ui::{TimelinePanelAction, TimelineTangentSide};
use std::{any::Any, time::Instant};

pub(super) struct PanelCapture {
    id: String,
    origin: [f32; 2],
}

fn is_drag_target(key: &str) -> bool {
    key.starts_with("timeline:key:")
        || matches!(
            key,
            "timeline:tangent:incoming" | "timeline:tangent:outgoing"
        )
}

impl EditorDesktopUi {
    /// Whether a timeline key/tangent owns the native pointer until release/cancel.
    pub fn panel_pointer_active(&self) -> bool {
        self.panels.capture.is_some()
    }

    /// Capture an actual visible key or tangent using retained Loom geometry.
    /// Returns false for ordinary controls, which continue through click routing.
    pub fn panel_pointer_press(
        &mut self,
        x: f32,
        y: f32,
        mode: SelectionMode,
    ) -> Result<bool, DesktopUiError> {
        if !x.is_finite() || !y.is_finite() {
            return self.reject(DesktopUiError::InvalidPointer);
        }
        let stale = self.panels.stale;
        let old_marker_hit = self
            .panels
            .geometry
            .iter()
            .any(|(key, (_, clip))| is_drag_target(key) && contains(*clip, x, y));
        self.refresh()?;
        // A pointer delivered for a prior layout cannot acquire a new target.
        if stale {
            return Ok(old_marker_hit);
        }
        if self.panel_pointer_active() {
            return self.reject(DesktopUiError::Panel("Pointer is already captured".into()));
        }
        let target = self.panels.geometry.iter().find_map(|(key, (_, clip))| {
            (is_drag_target(key) && contains(*clip, x, y)).then(|| key.clone())
        });
        let Some(target) = target else {
            return Ok(false);
        };
        if self.focus.is_some() {
            return self.reject(DesktopUiError::Panel(
                "Resolve the draft before dragging".into(),
            ));
        }
        let (origin_key, side) = match target.as_str() {
            "timeline:tangent:incoming" => (
                "timeline:curve".to_owned(),
                Some(TimelineTangentSide::Incoming),
            ),
            "timeline:tangent:outgoing" => (
                "timeline:curve".to_owned(),
                Some(TimelineTangentSide::Outgoing),
            ),
            _ => {
                let track = target
                    .strip_prefix("timeline:key:")
                    .and_then(|s| s.split(':').next())
                    .ok_or(DesktopUiError::InvalidProjection)?;
                (format!("timeline:lane:{track}"), None)
            }
        };
        let origin = self
            .panels
            .geometry
            .get(&origin_key)
            .map(|(rect, _)| [rect.left(), rect.top()])
            .ok_or(DesktopUiError::InvalidProjection)?;
        let slot = self
            .panels
            .slot_mut()
            .ok_or(DesktopUiError::InvalidProjection)?;
        let Some(panel) =
            (slot.host.as_mut() as &mut dyn Any).downcast_mut::<DesktopTimelinePanel>()
        else {
            return Ok(false);
        };
        if panel.failed || panel.pending_frame().is_some() {
            return self.reject(DesktopUiError::Panel(
                "Resolve failed timeline delivery before dragging".into(),
            ));
        }
        let view = panel
            .view
            .as_ref()
            .ok_or(DesktopUiError::InvalidProjection)?;
        let result = if let Some(side) = side {
            panel
                .controller
                .begin_tangent_drag(&panel.editor, view, side)
        } else {
            let TimelinePanelAction::Select { target, .. } = view
                .dispatch(&target, &UiEvent::Select { mode })
                .map_err(|e| DesktopUiError::Panel(e.to_string()))?
            else {
                return Err(DesktopUiError::InvalidProjection);
            };
            panel.controller.begin_key_drag(
                &mut panel.editor,
                view,
                target,
                f64::from(x - origin[0]),
                mode,
            )
        };
        if let Err(e) = result {
            return self.reject(DesktopUiError::Panel(e.to_string()));
        }
        panel.anchor = None;
        panel.wake = None;
        self.panels.capture = Some(PanelCapture {
            id: slot.id.clone(),
            origin,
        });
        self.invalidated = true;
        self.refresh()?;
        Ok(true)
    }

    /// Move a captured pointer, including outside the original pane. Invalid
    /// coordinates retain the last preview and surface the controller error.
    pub fn panel_pointer_move(&mut self, x: f32, y: f32) -> Result<bool, DesktopUiError> {
        let Some(capture) = &self.panels.capture else {
            return Ok(false);
        };
        let origin = capture.origin;
        let slot = self
            .panels
            .slots
            .iter_mut()
            .find(|s| s.id == capture.id)
            .ok_or(DesktopUiError::InvalidProjection)?;
        let panel = (slot.host.as_mut() as &mut dyn Any)
            .downcast_mut::<DesktopTimelinePanel>()
            .ok_or(DesktopUiError::InvalidProjection)?;
        if let Err(e) = panel.controller.update_drag(
            &panel.editor,
            [f64::from(x - origin[0]), f64::from(y - origin[1])],
        ) {
            return self.reject(DesktopUiError::Panel(e.to_string()));
        }
        self.invalidated = true;
        self.refresh()?;
        Ok(true)
    }

    /// Update the final native location, then publish one undoable transaction.
    /// Missing/invalid release cancels; an earlier valid preview is never committed.
    pub fn panel_pointer_release(
        &mut self,
        position: Option<[f32; 2]>,
    ) -> Result<bool, DesktopUiError> {
        if !self.panel_pointer_active() {
            return Ok(false);
        }
        let Some([x, y]) = position else {
            return self.cancel_panel_pointer();
        };
        if let Err(error) = self.panel_pointer_move(x, y) {
            self.cancel_panel_pointer()?;
            return self.reject(error);
        }
        let capture = self.panels.capture.as_ref().expect("active pointer");
        let slot = self
            .panels
            .slots
            .iter_mut()
            .find(|s| s.id == capture.id)
            .ok_or(DesktopUiError::InvalidProjection)?;
        let panel = (slot.host.as_mut() as &mut dyn Any)
            .downcast_mut::<DesktopTimelinePanel>()
            .ok_or(DesktopUiError::InvalidProjection)?;
        if let Err(e) = panel.controller.commit_drag(&mut panel.editor) {
            let error = DesktopUiError::Panel(e.to_string());
            self.cancel_panel_pointer()?;
            return self.reject(error);
        }
        panel.deliver = true;
        panel.wake = Some(Instant::now());
        self.panels.capture = None;
        self.invalidated = true;
        self.refresh()?;
        Ok(true)
    }

    /// Escape, focus loss, resize or host document replacement drops preview only.
    pub fn cancel_panel_pointer(&mut self) -> Result<bool, DesktopUiError> {
        let Some(capture) = self.panels.capture.take() else {
            return Ok(false);
        };
        if let Some(slot) = self.panels.slots.iter_mut().find(|s| s.id == capture.id)
            && let Some(panel) =
                (slot.host.as_mut() as &mut dyn Any).downcast_mut::<DesktopTimelinePanel>()
        {
            panel.controller.cancel_drag();
            panel.anchor = None;
            panel.deliver = true;
            panel.wake = Some(Instant::now());
        }
        self.invalidated = true;
        self.refresh()?;
        Ok(true)
    }
}
