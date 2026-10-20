//! Production graph ownership, admission and native image publication.
use super::*;
use prism_editor_profiler::{
    gpu_preview::TexturePreviewLimits,
    runtime_preview::{RuntimePreviewFrame, RuntimePreviewSubmission},
};
use prism_editor_ui::{
    BoundFrameGraphPanel, BoundFrameGraphPreview, FrameGraphPreviewAction,
    FrameGraphPreviewRequest, FrameGraphPreviewReservation,
};
use std::time::{Duration, Instant};

const POLL: Duration = Duration::from_millis(8);
const MAX_RETAINED: usize = 8;

/// Host runtime producer. Record an actual frame and invoke the consumer exactly
/// once synchronously. Its queue may be borrowed from this source. Imported
/// resources remain host-owned until the panel's completion fence has cleared.
pub trait DesktopRuntimePreviewSource {
    /// Called only for an explicit capture request, never while idle. Errors
    /// before invoking the consumer must leave the graph unsubmitted.
    fn capture(
        &mut self,
        submit: &mut dyn FnMut(RuntimePreviewFrame<'_>) -> Result<(), String>,
    ) -> Result<(), String>;
}

struct Flight {
    request: FrameGraphPreviewRequest,
    cancelled: bool,
}
struct Retiring {
    submission: RuntimePreviewSubmission,
    reservation: Option<FrameGraphPreviewReservation>,
}
#[derive(Default)]
pub(super) struct RuntimePreviewHost {
    source: Option<Box<dyn DesktopRuntimePreviewSource>>,
    limits: TexturePreviewLimits,
    requested: bool,
    flights: Vec<Flight>,
    retiring: Vec<Retiring>,
    pub(super) wake: Option<Instant>,
    pub(super) error: Option<String>,
}
impl RuntimePreviewHost {
    pub(super) fn fenced(&self) -> bool {
        self.requested
            || !self.flights.is_empty()
            || !self.retiring.is_empty()
            || self.error.is_some()
    }
}

/// Real CPU publication paired with its actual retained canvas and dock clip.
#[derive(Clone)]
pub struct DesktopFrameGraphImage {
    /// Exact display publication and zoom/pan state.
    pub view: BoundFrameGraphPreview,
    /// Logical retained image canvas.
    pub rect: Rect,
    /// Effective logical dock/window clip.
    pub clip: Rect,
}

impl DesktopFrameGraphPanel {
    /// Attach the real runtime producer and per-capture staging budget.
    /// Replacement requires all submitted work and errors to be resolved.
    pub fn mount_runtime_preview(
        &mut self,
        source: Box<dyn DesktopRuntimePreviewSource>,
        limits: TexturePreviewLimits,
    ) -> Result<(), String> {
        if self.runtime.fenced() {
            return Err("Runtime preview is fenced".into());
        }
        self.runtime.source = Some(source);
        self.runtime.limits = limits;
        Ok(())
    }
    /// Graphs retained through readback and actual graph completion.
    pub fn runtime_previews_pending(&self) -> usize {
        self.runtime.flights.len() + self.runtime.retiring.len()
    }
    /// Isolate pixels without undoing submitted rendering. Hidden panels keep
    /// polling; save/close/removal remain fenced through actual GPU drainage.
    pub fn cancel_runtime_preview(&mut self) -> Result<(), String> {
        self.runtime.requested = false;
        for flight in &mut self.runtime.flights {
            if !flight.cancelled {
                self.controller
                    .preview_mut()
                    .cancel_submission(&flight.request)
                    .map_err(|e| e.to_string())?;
                flight.cancelled = true;
            }
        }
        self.refresh_bound_view();
        Ok(())
    }
    pub(super) fn cancel_stale_previews(&mut self) -> Result<(), String> {
        for flight in &mut self.runtime.flights {
            let current = self.controller.preview().binding().is_some_and(|binding| {
                flight.request.binding().matches(
                    binding.report(),
                    binding.pass(),
                    binding.resource(),
                )
            }) && self.controller.preview().subresource()
                == Some([flight.request.mip(), flight.request.layer()]);
            if !current && !flight.cancelled {
                self.controller
                    .preview_mut()
                    .cancel_submission(&flight.request)
                    .map_err(|e| e.to_string())?;
                flight.cancelled = true;
            }
        }
        Ok(())
    }
    fn submit_runtime_frame(&mut self, frame: RuntimePreviewFrame<'_>) -> Result<(), String> {
        if self.runtime_previews_pending() >= MAX_RETAINED {
            return Err("Runtime preview graph completion capacity exhausted".into());
        }
        let selected = self
            .controller
            .preview()
            .binding()
            .ok_or("Select a real pass access first")?;
        // Stable authority keys cross frame boundaries; table positions do not.
        let pass_key = selected.report().passes()[selected.pass() as usize].key;
        let slot = selected.report().resources()[selected.resource() as usize].slot;
        let [mip, layer] = self
            .controller
            .preview()
            .subresource()
            .ok_or("No selected subresource")?;
        let pass = frame
            .report()
            .passes()
            .iter()
            .position(|p| p.key == pass_key)
            .ok_or("Selected pass is absent in runtime frame")? as u32;
        let resource = frame
            .report()
            .resources()
            .iter()
            .position(|r| r.slot == slot)
            .ok_or("Selected resource is absent in runtime frame")? as u32;
        let binding = frame.binding(pass, resource).map_err(|e| e.to_string())?;
        self.cancel_runtime_preview()?;
        self.controller
            .publish(Some(frame.report().clone()))
            .map_err(|e| e.to_string())?;
        self.controller
            .preview_mut()
            .select(binding, mip, layer)
            .map_err(|e| e.to_string())?;
        let reservation = self
            .controller
            .preview_mut()
            .reserve(self.runtime.limits)
            .map_err(|e| e.to_string())?;
        let [mip, layer] = reservation.subresource();
        let submission = match frame.submit(
            reservation.binding().clone(),
            mip,
            layer,
            reservation.capture_limits(),
        ) {
            Ok(submission) => submission,
            Err(failure) => {
                self.controller
                    .preview_mut()
                    .abandon(&reservation)
                    .map_err(|e| e.to_string())?;
                // Returned original frame drops here without submission.
                return Err(failure.error.to_string());
            }
        };
        match self
            .controller
            .preview_mut()
            .bind_submission(&reservation, submission)
        {
            Ok(request) => self.runtime.flights.push(Flight {
                request,
                cancelled: false,
            }),
            Err(rejected) => {
                let (error, mut submission) = *rejected;
                let _ = submission.cancel();
                self.runtime.retiring.push(Retiring {
                    submission,
                    reservation: Some(reservation),
                });
                return Err(error.to_string());
            }
        }
        self.refresh_bound_view();
        Ok(())
    }
    pub(super) fn poll_runtime(&mut self, now: Instant) -> Result<bool, String> {
        let mut changed = false;
        let mut error = None;
        for index in (0..self.runtime.flights.len()).rev() {
            let request = self.runtime.flights[index].request.clone();
            match self.controller.preview_mut().poll_submission(&request) {
                Ok(Some((submission, publication))) => {
                    self.runtime.flights.remove(index);
                    self.runtime.retiring.push(Retiring {
                        submission,
                        reservation: None,
                    });
                    changed = true;
                    if let Err(e) = publication {
                        error = Some(e.to_string());
                    }
                }
                Ok(None) => {}
                Err(e) => error = Some(e.to_string()),
            }
        }
        for index in (0..self.runtime.retiring.len()).rev() {
            let retiring = &mut self.runtime.retiring[index];
            if let Err(e) = retiring.submission.poll() {
                error = Some(e.to_string());
            }
            if retiring.submission.occupied_slots() == 0
                && retiring.submission.allocated_bytes() == 0
                && retiring.submission.is_complete()
            {
                let retired = self.runtime.retiring.remove(index);
                // Rejected binding retained its charge; release after drainage.
                if let Some(reservation) = retired.reservation
                    && let Err(e) = self.controller.preview_mut().abandon(&reservation)
                {
                    error = Some(e.to_string());
                }
                changed = true;
            }
        }
        if self.runtime.requested && error.is_none() && self.runtime.error.is_none() {
            self.runtime.requested = false;
            if let Some(mut source) = self.runtime.source.take() {
                let mut called = false;
                let mut rejected = None;
                let result = source.capture(&mut |frame| {
                    if called {
                        return Err("Runtime source supplied more than one frame".into());
                    }
                    called = true;
                    let result = self.submit_runtime_frame(frame);
                    if let Err(e) = &result {
                        rejected = Some(e.clone());
                    }
                    result
                });
                self.runtime.source = Some(source);
                error = rejected.or_else(|| result.err()).or_else(|| {
                    (!called).then(|| "Runtime source supplied no actual frame".into())
                });
                changed = true;
            } else {
                error = Some("No production runtime preview source is mounted".into());
            }
        }
        self.runtime.wake =
            (self.runtime.requested || self.runtime_previews_pending() != 0).then_some(now + POLL);
        if let Some(error) = error {
            self.runtime.error = Some(error.clone());
            return Err(error);
        }
        if changed {
            self.refresh_bound_view();
        }
        Ok(changed)
    }
    pub(super) fn runtime_event(&mut self, target: &str, event: &UiEvent) -> Result<(), String> {
        if target == "frame-graph:runtime:cancel" && matches!(event, UiEvent::Activate) {
            return self.cancel_runtime_preview();
        }
        if target == "frame-graph:runtime:retry" && matches!(event, UiEvent::Activate) {
            self.retry();
            return Ok(());
        }
        let action = self
            .controller
            .handle_preview_event(
                self.view.as_ref().ok_or("Frame graph is not rendered")?,
                target,
                event,
            )
            .map_err(|e| e.to_string())?;
        if action == FrameGraphPreviewAction::Capture {
            if self.runtime.source.is_none() {
                return Err("No production runtime preview source is mounted".into());
            }
            if self.runtime.error.is_some() {
                return Err("Resolve runtime preview failure and retry first".into());
            }
            self.runtime.requested = true;
            self.runtime.wake = Some(Instant::now());
        }
        self.cancel_stale_previews()?;
        self.refresh_bound_view();
        Ok(())
    }
    pub(super) fn decorate_preview(
        &self,
        view: &BoundFrameGraphPanel,
        result: &mut DesktopPanelView,
    ) {
        if let Some(preview) = view.preview() {
            result
                .bindings
                .extend(preview.bindings().map(|(k, t)| (k.to_owned(), t)));
        }
        let mut controls = Element::box_();
        for (key, label) in [
            (
                "frame-graph:runtime:cancel",
                "Cancel / drain runtime preview",
            ),
            (
                "frame-graph:runtime:retry",
                "Acknowledge preview error / retry",
            ),
        ] {
            controls = controls.child(Element::text(label).key_str(key).class("desktop-button"));
            result.bindings.push((key.into(), UiTrigger::Activate));
        }
        if let Some(error) = &self.runtime.error {
            controls = controls.child(Element::text(error.clone()));
        }
        result.element = Element::box_()
            .child(result.element.clone())
            .child(controls);
    }
}
impl EditorDesktopUi {
    /// Current active panel's actual image and retained canvas.
    pub fn frame_graph_image(&self) -> Option<&DesktopFrameGraphImage> {
        self.frame_graph_image.as_ref()
    }
}
