//! Host bindings, isolated previews and native transform gestures.
use super::*;
use crate::{ControllerFieldEdit, PreparedControllerEdit};
use prism_editor_viewport::{
    GizmoAxis, GizmoInteraction, GizmoMode, ScreenGizmoLayout, ScreenPoint, SnapSettings,
    TransformDelta,
};
use prism_reflect::Reflect;

const AXES: [GizmoAxis; 3] = [GizmoAxis::X, GizmoAxis::Y, GizmoAxis::Z];

/// Explicit host schema contract. Methods must be deterministic and free of
/// external mutations. Edits describe cumulative local TRS changes from the
/// unchanged drag-start model. Multi-selection applies the same delta to each
/// object's own pivot, not an inferred hierarchy or shared project pivot.
pub trait SceneTransformAdapter {
    /// Maps a bound model to a mesh's world matrix, including host submesh offsets.
    fn model(
        &self,
        item: &SelectionItem,
        mesh: SceneObjectId,
        root: &dyn Reflect,
    ) -> Result<SceneIdMatrix, DesktopError>;
    /// Returns explicit scalar field edits for this identity and cumulative delta.
    /// An unsupported or locked selection must return an error, not skip it.
    fn edits(
        &self,
        item: &SelectionItem,
        root: &dyn Reflect,
        delta: TransformDelta,
    ) -> Result<Vec<ControllerFieldEdit>, DesktopError>;
    /// Supplies screen axes and input calibration. Override for a host camera,
    /// coordinate convention, or projection policy. Default is world-aligned.
    fn frame(
        &self,
        model: SceneIdMatrix,
        camera: SceneIdMatrix,
        bounds: DesktopViewportBounds,
    ) -> Result<Option<GizmoFrame>, DesktopError> {
        GizmoFrame::project(model[3], camera, bounds)
    }
}

/// Logical viewport-local axis geometry and world units per logical pixel.
/// Rotation uses one degree per pixel; scale uses one percent per pixel.
#[derive(Clone, Copy, Debug)]
pub struct GizmoFrame {
    /// The same geometry is used for rendering and hit testing.
    pub layout: ScreenGizmoLayout,
    /// Positive finite translation calibration for X/Y/Z.
    pub world_units_per_pixel: [f64; 3],
}

impl GizmoFrame {
    /// Projects world axes with a constant 72 logical pixel handle length.
    /// Hides the gizmo when behind/outside the camera or any axis is edge-on;
    /// hosts can override frame for alternate degenerate-axis interaction.
    pub fn project(
        origin: [f32; 4],
        camera: SceneIdMatrix,
        bounds: DesktopViewportBounds,
    ) -> Result<Option<Self>, DesktopError> {
        let project = |p: [f32; 4]| -> Option<ScreenPoint> {
            let c: [f32; 4] =
                core::array::from_fn(|row| (0..4).map(|col| camera[col][row] * p[col]).sum());
            if c.iter().any(|v| !v.is_finite()) || c[3] <= 1e-6 || c[2] < 0.0 || c[2] > c[3] {
                return None;
            }
            Some(ScreenPoint::new(
                (c[0] / c[3] + 1.0) * bounds.logical.size.width * 0.5,
                (1.0 - c[1] / c[3]) * bounds.logical.size.height * 0.5,
            ))
        };
        let Some(start) = project(origin) else {
            return Ok(None);
        };
        if start.x() < 0.0
            || start.y() < 0.0
            || start.x() >= bounds.logical.size.width
            || start.y() >= bounds.logical.size.height
        {
            return Ok(None);
        }
        let mut endpoints = [start; 3];
        let mut units = [0.0; 3];
        for i in 0..3 {
            // Use a short world probe to avoid crossing the depth clip planes.
            let mut p = origin;
            p[i] += 0.01;
            let Some(end) = project(p) else {
                return Ok(None);
            };
            let dx = end.x() - start.x();
            let dy = end.y() - start.y();
            let length = prism_math::ops::hypot(dx, dy);
            if length < 1e-5 {
                return Ok(None);
            }
            endpoints[i] = ScreenPoint::new(
                start.x() + dx / length * 72.0,
                start.y() + dy / length * 72.0,
            );
            units[i] = 0.01 / f64::from(length);
        }
        Ok(Some(Self {
            layout: ScreenGizmoLayout::new(start, endpoints, 6.0).map_err(renderer_error)?,
            world_units_per_pixel: units,
        }))
    }

    fn validate(self) -> Result<Self, DesktopError> {
        if self
            .world_units_per_pixel
            .iter()
            .any(|u| !u.is_finite() || *u <= 0.0)
        {
            return Err(message("invalid gizmo translation calibration"));
        }
        Ok(self)
    }
}

pub(super) struct GizmoState {
    pub mode: GizmoMode,
    pub snap: SnapSettings,
    drag: Option<Drag>,
}

impl Default for GizmoState {
    fn default() -> Self {
        Self {
            mode: GizmoMode::Translate,
            snap: SnapSettings::default(),
            drag: None,
        }
    }
}

impl GizmoState {
    pub(super) fn drag_active(&self) -> bool {
        self.drag.is_some()
    }
}

struct Drag {
    interaction: GizmoInteraction,
    frame: GizmoFrame,
    start: ScreenPoint,
    raw: i64,
    delta: TransformDelta,
    items: Vec<SelectionItem>,
    original: Vec<(SceneObjectId, SceneIdMatrix)>,
    prepared: Option<PreparedControllerEdit>,
}

impl DesktopScene {
    /// Installs a host-owned transform contract. Call before starting gestures.
    pub fn set_transform_adapter<A: SceneTransformAdapter + 'static>(
        &mut self,
        adapter: A,
    ) -> Result<(), DesktopError> {
        if self.gizmo.drag.is_some() {
            return Err(message("cannot replace transform bindings during a drag"));
        }
        self.transform_adapter = Some(Box::new(adapter));
        self.transform_revision = None;
        Ok(())
    }

    /// Sets the transform tool and snapping; changes are fenced during a drag.
    pub fn set_gizmo_tool(
        &mut self,
        mode: GizmoMode,
        snap: SnapSettings,
    ) -> Result<(), DesktopError> {
        if self.gizmo.drag.is_some() {
            return Err(message("finish or cancel the active gizmo first"));
        }
        self.gizmo.mode = mode;
        self.gizmo.snap = snap;
        Ok(())
    }

    /// Current screen geometry, using the primary object's pivot.
    pub fn gizmo_frame(
        &self,
        controller: &EditorController,
    ) -> Result<Option<GizmoFrame>, DesktopError> {
        if let Some(drag) = &self.gizmo.drag {
            return Ok(Some(drag.frame));
        }
        let (Some(adapter), Some(bounds), Some(primary)) = (
            &self.transform_adapter,
            self.bounds,
            controller.selection().primary(),
        ) else {
            return Ok(None);
        };
        let Some(object) = self.transform_indices.get(primary).and_then(|indices| {
            indices
                .iter()
                .map(|i| &self.objects[*i])
                .find(|object| object.visible)
        }) else {
            return Ok(None);
        };
        adapter
            .frame(object.model, self.matrix, bounds)?
            .map(GizmoFrame::validate)
            .transpose()
    }

    /// Hit-tests before point/rectangle selection and leases the real document.
    pub fn gizmo_press(
        &mut self,
        controller: &mut EditorController,
        position: [f32; 2],
    ) -> Result<bool, DesktopError> {
        if self.gizmo.drag.is_some() {
            return Err(message("gizmo already active"));
        }
        self.sync_transforms(controller)?;
        let Some(frame) = self.gizmo_frame(controller)? else {
            return Ok(false);
        };
        let start = ScreenPoint::new(position[0], position[1]);
        let Some(interaction) = GizmoInteraction::begin_from_pointer(
            frame.layout,
            start,
            self.gizmo.mode,
            self.gizmo.snap,
        ) else {
            return Ok(false);
        };
        let items = controller.selection().items().to_vec();
        let adapter = self
            .transform_adapter
            .as_ref()
            .ok_or_else(|| message("missing transform binding"))?;
        // Validate the complete selection and identity bindings before capture.
        let mut edits = Vec::new();
        let identity = TransformDelta {
            scale_milliunits: [1000; 3],
            ..TransformDelta::default()
        };
        for item in &items {
            if !self.transform_indices.contains_key(item) {
                return Err(message("selected object has no scene transform binding"));
            }
            let fields = adapter.edits(
                item,
                controller.object_root(item).map_err(renderer_error)?,
                identity,
            )?;
            validate_targets(item, &fields)?;
            edits.extend(fields);
        }
        let identity_candidate = controller.prepare_edits(edits).map_err(renderer_error)?;
        if items
            .iter()
            .any(|item| identity_candidate.root(item).is_some())
        {
            return Err(message("host identity delta must preserve the model"));
        }
        let original = items
            .iter()
            .flat_map(|item| {
                self.transform_indices[item]
                    .iter()
                    .map(|i| (self.objects[*i].id, self.objects[*i].model))
            })
            .collect();
        controller.begin_edit().map_err(renderer_error)?;
        self.cancel_pending();
        self.gizmo.drag = Some(Drag {
            interaction,
            frame,
            start,
            raw: 0,
            delta: identity,
            items,
            original,
            prepared: None,
        });
        Ok(true)
    }

    /// Updates isolated candidate models and previews only the affected meshes.
    /// Any adapter, conversion, metadata or GPU preflight failure cancels the
    /// entire gesture and restores all original matrices.
    pub fn gizmo_move(
        &mut self,
        controller: &mut EditorController,
        position: [f32; 2],
    ) -> Result<bool, DesktopError> {
        let Some(mut drag) = self.gizmo.drag.take() else {
            return Ok(false);
        };
        let result = (|| {
            if position.iter().any(|v| !v.is_finite()) {
                return Err(message("nonfinite gizmo pointer"));
            }
            let axis = drag.interaction.axis();
            let i = axis.index() as usize;
            let origin = drag.frame.layout.origin();
            let end = drag.frame.layout.endpoint(axis);
            let dx = f64::from(end.x() - origin.x());
            let dy = f64::from(end.y() - origin.y());
            let pixels = ((f64::from(position[0]) - f64::from(drag.start.x())) * dx
                + (f64::from(position[1]) - f64::from(drag.start.y())) * dy)
                / dx.hypot(dy);
            let factor = match self.gizmo.mode {
                GizmoMode::Translate => drag.frame.world_units_per_pixel[i] * 1000.0,
                GizmoMode::Rotate => 1000.0,
                GizmoMode::Scale => 10.0,
            };
            let raw = pixels * factor;
            if !raw.is_finite() || raw.abs() > f64::from(i32::MAX) {
                return Err(message("gizmo delta exceeds supported range"));
            }
            let raw = raw.round() as i64;
            let change = raw - drag.raw;
            drag.raw = raw;
            let mut input = TransformDelta {
                scale_milliunits: [1000; 3],
                ..TransformDelta::default()
            };
            match self.gizmo.mode {
                GizmoMode::Translate => input.translation_milliunits[i] = change,
                GizmoMode::Rotate => {
                    input.rotation_millidegrees[i] =
                        i32::try_from(change).map_err(renderer_error)?;
                }
                GizmoMode::Scale => input.scale_milliunits[i] += change,
            }
            let mut delta = drag
                .interaction
                .update(input)
                .ok_or_else(|| message("inactive gizmo"))?;
            // A non-divisor scale snap must not alter the other two axes or
            // turn a return to the drag origin into a nonidentity transform.
            for other in 0..3 {
                if other != i {
                    delta.scale_milliunits[other] = 1000;
                }
            }
            if raw == 0 {
                delta = TransformDelta {
                    scale_milliunits: [1000; 3],
                    ..TransformDelta::default()
                };
            }
            if delta == drag.delta {
                return Ok(false);
            }
            let adapter = self
                .transform_adapter
                .as_ref()
                .ok_or_else(|| message("missing transform binding"))?;
            let mut edits = Vec::new();
            for item in &drag.items {
                let fields = adapter.edits(
                    item,
                    controller.object_root(item).map_err(renderer_error)?,
                    delta,
                )?;
                validate_targets(item, &fields)?;
                edits.extend(fields);
            }
            let prepared = controller.prepare_edits(edits).map_err(renderer_error)?;
            let mut models = Vec::new();
            for item in &drag.items {
                let root = prepared
                    .root(item)
                    .unwrap_or(controller.object_root(item).map_err(renderer_error)?);
                for index in &self.transform_indices[item] {
                    let id = self.objects[*index].id;
                    models.push((id, adapter.model(item, id, root)?));
                }
            }
            // Reserve another complete batch for cancellation before preview.
            self.scene
                .generation()
                .checked_add((models.len() as u64).saturating_mul(2))
                .ok_or_else(|| message("scene generation exhausted"))?;
            self.apply_models(&models)?;
            drag.delta = delta;
            drag.prepared = Some(prepared);
            Ok(true)
        })();
        if result.is_err() {
            drag.interaction.cancel();
            controller.end_edit();
            self.apply_models(&drag.original)?;
        } else {
            self.gizmo.drag = Some(drag);
        }
        result
    }

    /// Publishes one real command on release, or restores the entire preview.
    pub fn gizmo_release(
        &mut self,
        controller: &mut EditorController,
    ) -> Result<bool, DesktopError> {
        let Some(mut drag) = self.gizmo.drag.take() else {
            return Ok(false);
        };
        drag.interaction.finish();
        controller.end_edit();
        if let Some(prepared) = drag.prepared
            && let Err(error) = controller.commit_edits(prepared)
        {
            self.apply_models(&drag.original)?;
            return Err(renderer_error(error));
        }
        // Force the next frame through the authoritative model path.
        Ok(true)
    }

    /// Cancels a gesture without any model, dirty, revision or history mutation.
    pub fn gizmo_cancel(
        &mut self,
        controller: &mut EditorController,
    ) -> Result<bool, DesktopError> {
        let Some(mut drag) = self.gizmo.drag.take() else {
            return Ok(false);
        };
        drag.interaction.cancel();
        controller.end_edit();
        self.apply_models(&drag.original)?;
        Ok(true)
    }

    pub(super) fn sync_transforms(
        &mut self,
        controller: &EditorController,
    ) -> Result<(), DesktopError> {
        if self.gizmo.drag.is_some()
            || self.transform_adapter.is_none()
            || self.transform_revision == Some(controller.model_revision())
        {
            return Ok(());
        }
        let items = if let Some(revision) = self.transform_revision {
            controller.changed_objects_since(revision)
        } else {
            None
        };
        let items = items.unwrap_or_else(|| {
            self.transform_indices.clear();
            for (index, object) in self.objects.iter().enumerate() {
                self.transform_indices
                    .entry(object.selection.clone())
                    .or_default()
                    .push(index);
            }
            self.transform_indices.keys().cloned().collect()
        });
        let adapter = self
            .transform_adapter
            .as_ref()
            .ok_or_else(|| message("missing transform binding"))?;
        let mut models = Vec::new();
        for item in items {
            if let Some(indices) = self.transform_indices.get(&item) {
                let root = controller.object_root(&item).map_err(renderer_error)?;
                for index in indices {
                    let id = self.objects[*index].id;
                    models.push((id, adapter.model(&item, id, root)?));
                }
            }
        }
        self.apply_models(&models)?;
        self.transform_revision = Some(controller.model_revision());
        Ok(())
    }

    fn apply_models(
        &mut self,
        models: &[(SceneObjectId, SceneIdMatrix)],
    ) -> Result<(), DesktopError> {
        // Preflight every matrix, identity and generation before touching GPU/CPU.
        for (id, model) in models {
            validate_matrix(model)?;
            if self.object_model(*id).is_none() {
                return Err(message("unbound scene transform"));
            }
        }
        self.scene
            .generation()
            .checked_add(models.len() as u64)
            .ok_or_else(|| message("scene generation exhausted"))?;
        for (id, model) in models {
            self.update_object_model(*id, *model)?;
        }
        Ok(())
    }

    pub(super) fn paint_gizmo(
        &self,
        gpu: &DesktopGpu,
        controller: &EditorController,
        encoder: &mut CommandEncoder,
        target: &TextureView,
        bounds: DesktopViewportBounds,
    ) -> Result<(), DesktopError> {
        if let (Some(frame), Some(native), Some(region)) = (
            self.gizmo_frame(controller)?,
            &self.gpu,
            preview_region(&bounds)?,
        ) {
            let active = self
                .gizmo
                .drag
                .as_ref()
                .map(|d| d.interaction.axis().index());
            native.gizmo.paint(
                gpu,
                encoder,
                target,
                frame.layout,
                active,
                bounds,
                region.scissor,
            );
        }
        Ok(())
    }
}

fn validate_targets(
    item: &SelectionItem,
    edits: &[ControllerFieldEdit],
) -> Result<(), DesktopError> {
    if edits.is_empty() || edits.iter().any(|edit| &edit.item != item) {
        return Err(message(
            "host must bind every selected object and only edit that identity",
        ));
    }
    Ok(())
}

pub(super) struct GizmoPainter {
    pipeline: wgpu::RenderPipeline,
    uniform: wgpu::Buffer,
    binding: wgpu::BindGroup,
}

impl GizmoPainter {
    pub fn new(gpu: &DesktopGpu) -> Self {
        let device = &gpu.device;
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gizmo axes"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gizmo screen axes"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let binding = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gizmo axes"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform.as_entire_binding(),
            }],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("native gizmo axes"),
            source: wgpu::ShaderSource::Wgsl(include_str!("desktop_gizmo.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gizmo axes"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("native gizmo axes"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: gpu.format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview_mask: None,
            cache: None,
        });
        Self {
            pipeline,
            uniform,
            binding,
        }
    }

    fn paint(
        &self,
        gpu: &DesktopGpu,
        encoder: &mut CommandEncoder,
        target: &TextureView,
        layout: ScreenGizmoLayout,
        active: Option<u8>,
        bounds: DesktopViewportBounds,
        scissor: [u32; 4],
    ) {
        let sx = bounds.physical.size.width / bounds.logical.size.width;
        let sy = bounds.physical.size.height / bounds.logical.size.height;
        let point = |p: ScreenPoint| {
            [
                bounds.physical.left() + p.x() * sx,
                bounds.physical.top() + p.y() * sy,
            ]
        };
        let mut values = Vec::new();
        values.extend(point(layout.origin()));
        values.extend([active.map_or(-1.0, f32::from), sx.max(sy)]);
        for axis in AXES {
            values.extend(point(layout.endpoint(axis)));
            values.extend([0.0, 0.0]);
        }
        let bytes: Vec<u8> = values.into_iter().flat_map(f32::to_le_bytes).collect();
        gpu.queue.write_buffer(&self.uniform, 0, &bytes);
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("native gizmo overlay"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        pass.set_scissor_rect(scissor[0], scissor[1], scissor[2], scissor[3]);
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.binding, &[]);
        pass.draw(0..3, 0..1);
    }
}
