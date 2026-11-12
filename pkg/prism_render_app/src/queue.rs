//! The mesh queue: the draw items a frame's phases will record.
//!
//! Bevy's render world turns the visible set into render-phase items (a
//! `queue_material_meshes`-style stage): each item is a fully specialized draw —
//! the pipeline to bind, the mesh, and the material. Prism's equivalent is
//! [`RenderQueues`]: a set of per-phase item lists the queue stage fills from
//! the visible/binned set, then sorts deterministically so a frame records the
//! same order every run.
//!
//! The queue is phased by [`RenderPhase`] (opaque / transparent / stylized /
//! custom) — the material-model-agnostic phases of the single execution path —
//! not by material model, so PBR / NPR / hybrid / custom draws share one queue
//! and differ only in the pipeline they specialize to.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec;

use prism_ecs::prelude::Resource;
use prism_render_architecture::gpu_scene::{GeometryHandle, SceneHandle, SceneMaterialHandle};
use prism_render_driver::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayoutId, BindingResource, BufferDescriptor,
    BufferId, BufferUsages, IndexFormat, RenderDevice, RenderPipelineDescriptor, RenderPipelineId,
    RenderQueue, VertexBufferLayout,
};
use prism_render_material::{
    material_pipeline_state, pipeline_state_for_material, MaterialRecord, MaterialRenderClass,
    MaterialRenderPhase,
};

use crate::draw::{
    BindGroupBindingId, DrawFunctionId, DrawFunctions, RenderBindGroupBinding,
    RenderBindGroupBuffer, RenderBindGroupBufferAccess,
};
use crate::forward_lit::{surface_uv1_layout, SurfaceFrame, SurfaceMaterial};
use crate::material::MaterialParameterBuffers;
use crate::material::virtual_texture::NativeVtBinding;
use crate::mesh_pipeline::{MeshPipelineKey, MeshPipelineSpecializer};
use crate::sets::RenderPhase;

/// GPU state for the default surface mesh material contract.
#[derive(Resource)]
pub struct CanonicalMaterialBindings {
    generation: Option<u64>,
    layout: Option<BindGroupLayoutId>,
    frame: Option<BufferId>,
    materials: Option<BufferId>,
    bind_group: Option<prism_render_driver::BindGroupId>,
    vt_layout: Option<BindGroupLayoutId>,
    vt_bind_group: Option<prism_render_driver::BindGroupId>,
    vt_resources: Option<(
        u64,
        BindGroupLayoutId,
        BufferId,
        BufferId,
        prism_render_driver::TextureViewId,
        prism_render_driver::SamplerId,
    )>,
    canonical_buffers: Option<[BufferId; 3]>,
    material_stride: u32,
    material_capacity: u32,
    frame_stride: u32,
    frame_capacity: u32,
    frame_slots: u32,
    explicit_frame_batch: bool,
    frame_values: BTreeMap<u32, SurfaceFrame>,
    prepared_frame_slots: BTreeSet<u32>,
    bound_frame_slots: BTreeSet<u32>,
    material_values: BTreeMap<u32, SurfaceMaterial>,
}

impl CanonicalMaterialBindings {
    /// Creates the binding owner. Device resources are allocated lazily once
    /// the active mesh provider exposes its group-0 layout.
    #[must_use]
    pub fn new(initial_capacity: u32) -> Self {
        Self {
            generation: None,
            layout: None,
            frame: None,
            materials: None,
            bind_group: None,
            vt_layout: None,
            vt_bind_group: None,
            vt_resources: None,
            canonical_buffers: None,
            material_stride: 0,
            material_capacity: initial_capacity.max(1),
            frame_stride: 0,
            frame_capacity: 1,
            frame_slots: 1,
            explicit_frame_batch: false,
            frame_values: BTreeMap::new(),
            prepared_frame_slots: BTreeSet::new(),
            bound_frame_slots: BTreeSet::new(),
            material_values: BTreeMap::new(),
        }
    }

    /// Rebuilds the device-local group when the provider/device generation
    /// changes. Old ids are forgotten rather than passed to a new device.
    pub fn ensure_layout(
        &mut self,
        device: &dyn RenderDevice,
        generation: u64,
        layout: Option<BindGroupLayoutId>,
    ) -> bool {
        let Some(layout) = layout else {
            if self.generation == Some(generation) {
                self.release_gpu(device);
            } else {
                self.frame = None;
                self.materials = None;
                self.bind_group = None;
                self.canonical_buffers = None;
            }
            self.generation = Some(generation);
            self.layout = None;
            self.vt_layout = None;
            self.vt_resources = None;
            self.frame = None;
            self.materials = None;
            self.bind_group = None;
            self.invalidate_uploads();
            return true;
        };
        if self.generation == Some(generation)
            && self.layout == Some(layout)
            && self.frame.is_some()
            && self.materials.is_some()
        {
            return true;
        }
        if self.generation == Some(generation) {
            self.release_gpu(device);
        } else {
            self.frame = None;
            self.materials = None;
            self.bind_group = None;
            self.canonical_buffers = None;
            self.vt_bind_group = None;
            self.vt_resources = None;
            self.vt_layout = None;
        }
        self.generation = Some(generation);
        self.layout = Some(layout);
        // These mirrors describe uploads into the old uniform allocation, not
        // values resident on an arbitrary replacement device or provider.
        self.invalidate_uploads();
        self.allocate_gpu(device)
    }

    /// Installs the native surface VT layout and drops any group from another
    /// provider/device generation. Queue preflight has already drained use.
    pub fn ensure_vt_layout(
        &mut self,
        device: &dyn RenderDevice,
        generation: u64,
        layout: Option<BindGroupLayoutId>,
    ) -> bool {
        if self.generation != Some(generation) {
            self.vt_bind_group = None;
            self.vt_resources = None;
            self.vt_layout = layout;
            return false;
        }
        if self.vt_layout == layout {
            return true;
        }
        if let Some(group) = self.vt_bind_group.take() {
            device.destroy_bind_group(group);
        }
        self.vt_resources = None;
        self.vt_layout = layout;
        true
    }

    /// Whether the next complete view/material batch would destroy or replace
    /// resources that may still be referenced by a submitted frame.
    ///
    /// A cold owner has nothing to drain: waiting there would turn a valid
    /// first frame into a false completion failure for hosts that intentionally
    /// delay completion proofs. Once GPU-owned buffers or bind groups exist,
    /// growth, provider/device changes, or a changed VT snapshot require the
    /// caller to prove completion before the preparation methods below can
    /// retire them.
    pub fn requires_completion_before_batch<D: RenderDevice + Send + Sync + 'static>(
        &self,
        device: &dyn RenderDevice,
        generation: u64,
        layout: Option<BindGroupLayoutId>,
        vt_layout: Option<BindGroupLayoutId>,
        materials: &MaterialParameterBuffers,
        frame_slots: usize,
        vt_snapshot: Option<&NativeVtBinding<D>>,
    ) -> bool {
        let has_gpu_resources = self.frame.is_some()
            || self.materials.is_some()
            || self.bind_group.is_some()
            || self.vt_bind_group.is_some();
        if has_gpu_resources
            && (self.generation != Some(generation) || self.layout != layout)
        {
            return true;
        }
        if self.vt_bind_group.is_some() && self.vt_layout != vt_layout {
            return true;
        }

        if self.frame.is_some() && self.materials.is_some() {
            let material_rows = materials.header_capacity().max(self.material_capacity);
            let frame_slots = u32::try_from(frame_slots).unwrap_or(u32::MAX);
            let capacity_change = match (
                material_allocation(&device.capabilities().limits, material_rows),
                frame_allocation(&device.capabilities().limits, frame_slots.max(1)),
            ) {
                (Some((material_stride, material_capacity)), Some((frame_stride, frame_capacity))) => {
                    material_stride != self.material_stride
                        || material_capacity != self.material_capacity
                        || frame_stride != self.frame_stride
                        || frame_capacity != self.frame_capacity
                }
                _ => false,
            };
            if capacity_change {
                return true;
            }
        }

        if self.bind_group.is_some() {
            let canonical = match (
                materials.headers(),
                materials.parameters(),
                materials.textures(),
            ) {
                (Some(headers), Some(parameters), Some(textures)) => {
                    Some([headers, parameters, textures])
                }
                _ => None,
            };
            if self.canonical_buffers != canonical {
                return true;
            }
        }

        if self.vt_bind_group.is_some()
            && let (Some(vt_layout), Some(snapshot)) = (vt_layout, vt_snapshot)
        {
            let key = (
                generation,
                vt_layout,
                snapshot.page_table(),
                snapshot.config(),
                snapshot.atlas_view(),
                snapshot.sampler(),
            );
            if self.vt_resources != Some(key) {
                return true;
            }
        }
        false
    }

    /// Gets/creates group 1 from one already-captured active VT snapshot.
    /// A generation mismatch or incomplete layout fails closed.
    pub fn vt_binding<D: RenderDevice + Send + Sync + 'static>(
        &mut self,
        device: &dyn RenderDevice,
        snapshot: &NativeVtBinding<D>,
    ) -> Option<RenderBindGroupBinding> {
        let generation = self.generation?;
        let layout = self.vt_layout?;
        if snapshot.device_generation() != generation {
            return None;
        }
        let page_table = snapshot.page_table();
        let config = snapshot.config();
        let atlas = snapshot.atlas_view();
        let sampler = snapshot.sampler();
        let key = (generation, layout, page_table, config, atlas, sampler);
        if self.vt_resources != Some(key) {
            let descriptor = BindGroupDescriptor {
                label: Some("prism canonical surface Material-VT group 1".into()),
                layout: Some(layout),
                entries: alloc::vec![
                    BindGroupEntry {
                        binding: crate::forward_lit::SURFACE_VT_PAGE_TABLE_BINDING,
                        resource: BindingResource::Buffer {
                            buffer: page_table,
                            offset: 0,
                            size: None
                        }
                    },
                    BindGroupEntry {
                        binding: crate::forward_lit::SURFACE_VT_CONFIG_BINDING,
                        resource: BindingResource::Buffer {
                            buffer: config,
                            offset: 0,
                            size: None
                        }
                    },
                    BindGroupEntry {
                        binding: crate::forward_lit::SURFACE_VT_ATLAS_BINDING,
                        resource: BindingResource::TextureView(atlas)
                    },
                    BindGroupEntry {
                        binding: crate::forward_lit::SURFACE_VT_SAMPLER_BINDING,
                        resource: BindingResource::Sampler(sampler)
                    },
                ],
            };
            let group = device.create_bind_group(&descriptor);
            if let Some(old) = self.vt_bind_group.replace(group) {
                device.destroy_bind_group(old);
            }
            self.vt_resources = Some(key);
        }
        Some(RenderBindGroupBinding {
            identity: crate::draw::BindGroupBindingId(1),
            index: 1,
            bind_group: self.vt_bind_group?,
            dynamic_offsets: Vec::new(),
            buffers: alloc::vec![
                RenderBindGroupBuffer {
                    buffer: page_table,
                    access: RenderBindGroupBufferAccess::Storage
                },
                RenderBindGroupBuffer {
                    buffer: config,
                    access: RenderBindGroupBufferAccess::Storage
                },
            ],
            sampled_textures: alloc::vec![atlas],
            external_owners: alloc::vec![snapshot.external_owner()],
        })
    }

    /// The texture view currently published through the frame graph, allowing
    /// queue preparation to revoke precisely the previous VT import.
    pub fn vt_import_view(&self) -> Option<prism_render_driver::TextureViewId> {
        self.vt_resources.map(|(_, _, _, _, view, _)| view)
    }

    /// Reserves the complete frame's material address space before any draw is
    /// queued. A draw never reallocates resources that another queued draw uses.
    /// Returns false if the device cannot address the requested rows.
    pub fn prepare_capacity(&mut self, device: &dyn RenderDevice, needed: u32) -> bool {
        if !self.reserve_capacities(device, needed, 1) {
            return false;
        }
        // Legacy single-view callers may refresh canonical storage without
        // re-uploading an unchanged slot 0. Keep that slot's readiness intact.
        self.frame_slots = 1;
        self.explicit_frame_batch = false;
        self.frame_values.retain(|&slot, _| slot == 0);
        self.prepared_frame_slots.retain(|&slot| slot == 0);
        self.bound_frame_slots.clear();
        true
    }

    /// Reserves both uniform arenas and begins a new view-binding batch.
    ///
    /// Call once with the complete material/view counts, before generating
    /// **any** camera bindings. Slots are dense in `0..frame_slots`; zero views
    /// disables all frame bindings. Every active slot must be prepared again
    /// in this batch, even if its resident bytes are unchanged. Removed slots
    /// lose their upload mirrors. Uploads and binding lookup never grow arenas.
    ///
    /// The caller must have submitted or discarded all bindings from the old
    /// batch before reserving (including a same-capacity reserve). Growth can
    /// destroy their buffers/group; reusing slots can change their contents.
    /// A limit/overflow failure preserves the old resources and batch intact.
    pub fn prepare_capacities(
        &mut self,
        device: &dyn RenderDevice,
        material_rows: u32,
        frame_slots: u32,
    ) -> bool {
        if !self.reserve_capacities(device, material_rows, frame_slots) {
            return false;
        }
        self.frame_slots = frame_slots;
        self.explicit_frame_batch = true;
        self.frame_values.retain(|&slot, _| slot < frame_slots);
        self.prepared_frame_slots.clear();
        self.bound_frame_slots.clear();
        true
    }

    fn reserve_capacities(
        &mut self,
        device: &dyn RenderDevice,
        material_rows: u32,
        frame_slots: u32,
    ) -> bool {
        if self.layout.is_none() {
            return false;
        }
        let limits = &device.capabilities().limits;
        let Some((stride, capacity)) =
            material_allocation(limits, material_rows.max(self.material_capacity))
        else {
            return false;
        };
        let Some((frame_stride, frame_capacity)) =
            frame_allocation(limits, frame_slots.max(self.frame_capacity))
        else {
            return false;
        };
        if self.materials.is_some()
            && self.frame.is_some()
            && stride == self.material_stride
            && frame_stride == self.frame_stride
            && material_rows <= self.material_capacity
            && frame_slots <= self.frame_capacity
        {
            return true;
        }
        self.release_gpu(device);
        self.material_stride = stride;
        self.material_capacity = capacity;
        self.frame_stride = frame_stride;
        self.frame_capacity = frame_capacity;
        // No draw has been emitted yet. Only live rows requested this frame
        // need uploading; replaying every historical row wastes bandwidth.
        self.allocate_gpu(device)
    }

    /// Binds the authoritative canonical buffers after all material uploads and
    /// capacity changes, before emitting any draws. No parameter data is copied.
    /// Buffer growth changes the group identity even when material revisions
    /// are unchanged; callers cannot accidentally reuse a freed heap binding.
    pub fn prepare_material_buffers(
        &mut self,
        device: &dyn RenderDevice,
        buffers: &crate::MaterialParameterBuffers,
    ) -> bool {
        let limits = &device.capabilities().limits;
        let header_bytes = u64::from(buffers.header_capacity())
            * core::mem::size_of::<prism_render_material::GpuMaterialHeader>() as u64;
        let parameter_bytes = u64::from(buffers.parameter_capacity_words()) * 4;
        if limits.max_storage_buffers_per_shader_stage < 3
            || header_bytes > limits.max_storage_buffer_binding_size
            || parameter_bytes > limits.max_storage_buffer_binding_size
            || u64::from(buffers.texture_capacity_rows())
                .saturating_mul(crate::material::texture_row_bytes())
                > limits.max_storage_buffer_binding_size
        {
            return false;
        }
        let (
            Some(layout),
            Some(frame),
            Some(materials),
            Some(headers),
            Some(parameters),
            Some(textures),
        ) = (
            self.layout,
            self.frame,
            self.materials,
            buffers.headers(),
            buffers.parameters(),
            buffers.textures(),
        )
        else {
            return false;
        };
        let canonical = [headers, parameters, textures];
        if self.canonical_buffers == Some(canonical) && self.bind_group.is_some() {
            return true;
        }
        let group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism canonical surface material group".into()),
            layout: Some(layout),
            entries: vec![
                BindGroupEntry {
                    binding: crate::forward_lit::SURFACE_FRAME_BINDING,
                    resource: BindingResource::Buffer {
                        buffer: frame,
                        offset: 0,
                        size: Some(SurfaceFrame::SIZE_BYTES),
                    },
                },
                BindGroupEntry {
                    binding: crate::forward_lit::SURFACE_MATERIAL_BINDING,
                    resource: BindingResource::Buffer {
                        buffer: materials,
                        offset: 0,
                        size: Some(SurfaceMaterial::SIZE_BYTES),
                    },
                },
                BindGroupEntry {
                    binding: crate::forward_lit::SURFACE_HEADERS_BINDING,
                    resource: BindingResource::Buffer {
                        buffer: headers,
                        offset: 0,
                        size: None,
                    },
                },
                BindGroupEntry {
                    binding: crate::forward_lit::SURFACE_PARAMETERS_BINDING,
                    resource: BindingResource::Buffer {
                        buffer: parameters,
                        offset: 0,
                        size: None,
                    },
                },
                BindGroupEntry {
                    binding: crate::forward_lit::SURFACE_TEXTURES_BINDING,
                    resource: BindingResource::Buffer {
                        buffer: textures,
                        offset: 0,
                        size: None,
                    },
                },
            ],
        });
        if let Some(old) = self.bind_group.replace(group) {
            device.destroy_bind_group(old);
        }
        self.canonical_buffers = Some(canonical);
        true
    }

    /// Uploads the frame/view uniform for a validated native surface view.
    /// Returns false without writing when no view is available.
    pub fn prepare_frame(
        &mut self,
        queue: &dyn RenderQueue,
        view: Option<&crate::surface_views::SurfaceView>,
    ) -> bool {
        self.prepare_frame_slot(queue, 0, view)
    }

    /// Uploads one view's current/previous matrices at its aligned frame slot.
    /// Returns false without writes for an absent view, unreserved slot, or
    /// missing allocation. Within an explicit `prepare_capacities` batch, a
    /// slot that has emitted a binding cannot be overwritten with other bytes;
    /// identical repeats are allowed. Start a new batch to reuse such a slot.
    pub fn prepare_frame_slot(
        &mut self,
        queue: &dyn RenderQueue,
        slot: u32,
        view: Option<&crate::surface_views::SurfaceView>,
    ) -> bool {
        let Some(view) = view else {
            return false;
        };
        let (Some(buffer), Some(offset)) = (self.frame, self.frame_offset(slot)) else {
            return false;
        };
        let value = SurfaceFrame::from_view(view);
        if self.frame_values.get(&slot) != Some(&value) {
            if self.bound_frame_slots.contains(&slot) {
                return false;
            }
            queue.write_buffer(buffer, u64::from(offset), bytemuck::bytes_of(&value));
            self.frame_values.insert(slot, value);
        }
        self.prepared_frame_slots.insert(slot);
        true
    }

    /// Resolves a scene material to the physical dynamic group. The scene
    /// handle and the active ABI row are checked before a binding is emitted.
    pub fn binding_for(
        &mut self,
        queue: &dyn RenderQueue,
        handle: SceneMaterialHandle,
        record: &MaterialRecord,
        buffers: &crate::MaterialParameterBuffers,
    ) -> Option<RenderBindGroupBinding> {
        self.binding_for_frame_slot(queue, 0, handle, record, buffers)
    }

    /// Resolves a material with the selected view's dynamic frame offset.
    /// Requires a reserved slot prepared in the current batch, and validates
    /// canonical storage/identity before any material write. All views share
    /// one physical group and material arena; only the first dynamic offset
    /// varies by view. This never allocates GPU resources.
    pub fn binding_for_frame_slot(
        &mut self,
        queue: &dyn RenderQueue,
        slot: u32,
        handle: SceneMaterialHandle,
        record: &MaterialRecord,
        buffers: &crate::MaterialParameterBuffers,
    ) -> Option<RenderBindGroupBinding> {
        let frame_offset = self.frame_offset(slot)?;
        if !self.prepared_frame_slots.contains(&slot) {
            return None;
        }
        if record.handle != handle || handle.index >= self.material_capacity {
            return None;
        }
        let dynamic_offset = handle.index.checked_mul(self.material_stride)?;
        let frame = self.frame?;
        let materials = self.materials?;
        let bind_group = self.bind_group?;
        let headers = buffers.headers()?;
        let parameters = buffers.parameters()?;
        let textures = buffers.textures()?;
        if self.canonical_buffers != Some([headers, parameters, textures]) {
            return None;
        }
        let header = buffers.header(handle.index)?;
        if header.is_active == 0
            || header.generation != handle.generation
            || header.revision != record.revision
        {
            return None;
        }
        let material = SurfaceMaterial {
            index: handle.index,
            generation: handle.generation,
            revision: record.revision,
            execution_plan_index: header.execution_plan_index,
        };
        if self.material_values.get(&handle.index) != Some(&material) {
            queue.write_buffer(
                materials,
                u64::from(dynamic_offset),
                bytemuck::bytes_of(&material),
            );
            self.material_values.insert(handle.index, material);
        }
        if self.explicit_frame_batch {
            self.bound_frame_slots.insert(slot);
        }
        Some(RenderBindGroupBinding {
            identity: crate::draw::canonical::SURFACE_MATERIAL,
            index: crate::draw::canonical::SURFACE_GROUP,
            bind_group,
            dynamic_offsets: vec![frame_offset, dynamic_offset],
            buffers: vec![
                RenderBindGroupBuffer {
                    buffer: frame,
                    access: RenderBindGroupBufferAccess::Uniform,
                },
                RenderBindGroupBuffer {
                    buffer: materials,
                    access: RenderBindGroupBufferAccess::Uniform,
                },
                RenderBindGroupBuffer {
                    buffer: headers,
                    access: RenderBindGroupBufferAccess::Storage,
                },
                RenderBindGroupBuffer {
                    buffer: parameters,
                    access: RenderBindGroupBufferAccess::Storage,
                },
                RenderBindGroupBuffer {
                    buffer: textures,
                    access: RenderBindGroupBufferAccess::Storage,
                },
            ],
            sampled_textures: Vec::new(),
            external_owners: Vec::new(),
        })
    }

    /// Releases the owner resources during an orderly shutdown.
    pub fn destroy(&mut self, device: &dyn RenderDevice) {
        self.release_gpu(device);
        self.generation = None;
        self.layout = None;
        self.vt_layout = None;
    }

    fn allocate_gpu(&mut self, device: &dyn RenderDevice) -> bool {
        let Some(_) = self.layout else {
            return false;
        };
        let Some((stride, capacity)) =
            material_allocation(&device.capabilities().limits, self.material_capacity)
        else {
            return false;
        };
        let Some((frame_stride, frame_capacity)) =
            frame_allocation(&device.capabilities().limits, self.frame_capacity)
        else {
            return false;
        };
        self.material_stride = stride;
        self.material_capacity = capacity;
        self.frame_stride = frame_stride;
        self.frame_capacity = frame_capacity;
        self.frame = Some(device.create_buffer(&BufferDescriptor {
            label: Some("prism canonical surface frame".into()),
            size: u64::from(frame_capacity - 1) * u64::from(frame_stride)
                + SurfaceFrame::SIZE_BYTES,
            usage: BufferUsages::UNIFORM.union(BufferUsages::COPY_DST),
            mapped_at_creation: false,
        }));
        self.materials = Some(device.create_buffer(&BufferDescriptor {
            label: Some("prism canonical surface materials".into()),
            size: u64::from(self.material_stride) * u64::from(self.material_capacity),
            usage: BufferUsages::UNIFORM.union(BufferUsages::COPY_DST),
            mapped_at_creation: false,
        }));
        true
    }

    fn release_gpu(&mut self, device: &dyn RenderDevice) {
        self.invalidate_uploads();
        self.canonical_buffers = None;
        if let Some(group) = self.bind_group.take() {
            device.destroy_bind_group(group);
        }
        if let Some(group) = self.vt_bind_group.take() {
            device.destroy_bind_group(group);
        }
        self.vt_resources = None;
        if let Some(frame) = self.frame.take() {
            device.destroy_buffer(frame);
        }
        if let Some(materials) = self.materials.take() {
            device.destroy_buffer(materials);
        }
        self.vt_layout = None;
    }

    fn invalidate_uploads(&mut self) {
        self.frame_values.clear();
        self.prepared_frame_slots.clear();
        self.bound_frame_slots.clear();
        self.material_values.clear();
    }

    fn frame_offset(&self, slot: u32) -> Option<u32> {
        if slot >= self.frame_slots || slot >= self.frame_capacity || self.frame_stride == 0 {
            return None;
        }
        slot.checked_mul(self.frame_stride)
    }
}

fn frame_allocation(limits: &prism_render_driver::Limits, needed: u32) -> Option<(u32, u32)> {
    if limits.max_uniform_buffer_binding_size < SurfaceFrame::SIZE_BYTES
        || limits.max_buffer_size < SurfaceFrame::SIZE_BYTES
    {
        return None;
    }
    let alignment = u64::from(limits.min_uniform_buffer_offset_alignment.max(1));
    let stride = u32::try_from(SurfaceFrame::SIZE_BYTES.div_ceil(alignment) * alignment).ok()?;
    // The final slot needs only its payload, not trailing alignment padding.
    // Keep a legal single-view allocation even when the binding alignment is
    // larger than the payload ABI.
    let max_capacity = ((limits.max_buffer_size - SurfaceFrame::SIZE_BYTES) / u64::from(stride)
        + 1)
    .min(u64::from(u32::MAX / stride) + 1)
    .min(u64::from(u32::MAX));
    let needed = needed.max(1);
    if u64::from(needed) > max_capacity {
        return None;
    }
    let capacity =
        u64::from(needed.checked_next_power_of_two().unwrap_or(u32::MAX)).min(max_capacity) as u32;
    Some((stride, capacity))
}

fn material_allocation(limits: &prism_render_driver::Limits, needed: u32) -> Option<(u32, u32)> {
    if limits.max_uniform_buffer_binding_size < SurfaceMaterial::SIZE_BYTES
        || limits.max_buffer_size < SurfaceMaterial::SIZE_BYTES
    {
        return None;
    }
    let alignment = u64::from(limits.min_uniform_buffer_offset_alignment.max(1));
    let stride = SurfaceMaterial::SIZE_BYTES.div_ceil(alignment) * alignment;
    let stride = u32::try_from(stride).ok()?;
    let max_capacity = (limits.max_buffer_size / u64::from(stride))
        .min(u64::from(u32::MAX / stride) + 1)
        .min(u64::from(u32::MAX));
    let needed = needed.max(1);
    if u64::from(needed) > max_capacity {
        return None;
    }
    // Geometric growth where possible, without rejecting a legal exact fit
    // just because the next power of two exceeds the adapter's buffer limit.
    let capacity =
        u64::from(needed.checked_next_power_of_two().unwrap_or(u32::MAX)).min(max_capacity) as u32;
    Some((stride, capacity))
}

#[cfg(test)]
#[path = "queue/bindings_tests.rs"]
mod bindings_tests;

/// A resolved visible draw ready to queue: the scene has already resolved the
/// work item's geometry layout and material into a variant + phase.
///
/// The queue specializes it to a pipeline (through
/// [`MeshPipelineSpecializer`]) and appends a [`PhaseItem`].
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct DrawRequest {
    /// The phase the draw belongs to (from
    /// [`phase_for_render_class`]).
    pub phase: RenderPhase,
    /// The mesh pipeline variant (material class + fixed-function state).
    pub variant: MeshPipelineKey,
    /// The mesh's vertex layouts, folded into the pipeline key.
    pub layout: Vec<VertexBufferLayout>,
    /// The resolved LOD level to draw.
    pub lod_level: u32,
    /// The geometry to draw.
    pub geometry: GeometryHandle,
    /// The vertex buffers bound at slots `0..vertices.len()`.
    pub vertices: Vec<BufferId>,
    /// Exact byte length of each prepared vertex stream (excluding spare capacity).
    pub vertex_sizes: Vec<u64>,
    /// The index buffer and format, or `None` for a non-indexed draw.
    pub indices: Option<(BufferId, IndexFormat)>,
    /// Exact byte length of the prepared index data, when indexed.
    pub index_size: Option<u64>,
    /// The first element (index or vertex) to draw.
    pub first: u32,
    /// The number of elements to draw.
    pub count: u32,
    /// The base vertex added to each index (`0` for non-indexed).
    pub base_vertex: i32,
    /// The material whose bindings the draw uses.
    pub material: SceneMaterialHandle,
    /// The scene instance the draw came from.
    pub scene: SceneHandle,
    /// The caller's ordering key within the phase.
    pub order: u64,
    /// The draw function that records the item's commands.
    pub draw: DrawFunctionId,
}

/// A resolved request together with the physical bind groups selected for this
/// draw (for example, its material/view groups).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct DrawRequestWithBindings {
    /// The resolved mesh draw request.
    pub request: DrawRequest,
    /// Per-draw physical groups and their graph-visible buffer declarations.
    pub bind_groups: Vec<RenderBindGroupBinding>,
}

impl DrawRequest {
    /// Attaches the actual per-draw groups selected for this request.
    #[must_use]
    pub fn with_bind_groups(
        self,
        bind_groups: Vec<RenderBindGroupBinding>,
    ) -> DrawRequestWithBindings {
        DrawRequestWithBindings {
            request: self,
            bind_groups,
        }
    }
}

impl DrawRequest {
    /// Returns the explicit bind-group identity carried by this request's draw
    /// function contract.
    #[must_use]
    pub fn bind_group_identity(&self, functions: &DrawFunctions) -> Option<BindGroupBindingId> {
        functions.bind_group_identity(self.draw)
    }

    /// Returns the complete bind-group contract carried by this request.
    #[must_use]
    pub fn bind_groups<'a>(&self, functions: &'a DrawFunctions) -> &'a [RenderBindGroupBinding] {
        functions.bind_groups(self.draw)
    }

    /// Returns the required per-item binding contracts for this request.
    #[must_use]
    pub fn dynamic_bind_group_contracts<'a>(
        &self,
        functions: &'a DrawFunctions,
    ) -> &'a [crate::draw::RenderBindGroupBindingContract] {
        functions.dynamic_bind_group_contracts(self.draw)
    }
}

impl DrawRequestWithBindings {
    /// Returns the actual per-draw groups carried by this request.
    #[must_use]
    pub fn dynamic_bind_groups(&self) -> &[RenderBindGroupBinding] {
        &self.bind_groups
    }
}

/// The phase a surface draw belongs to, from its material's blend family.
///
/// Selection is by [`MaterialRenderClass`] **only**. `illumination`
/// (Lit / Stylized / Unlit / Custom) is the *response* axis
/// (`prism_material_pipeline_refactor_zh.md` §3.5/§8.5), not a phase: a stylized
/// or custom surface draws in the same opaque/transparent phase as a lit one and
/// differs only in the response its shader dispatches to. The
/// [`RenderPhase::Stylized`] and [`RenderPhase::Custom`] phases host extra
/// *passes* (an NPR outline pass, a project-injected pass), not the surface
/// draws of those materials.
#[must_use]
pub fn phase_for_render_class(class: MaterialRenderClass) -> RenderPhase {
    phase_from_material(pipeline_state_for_material(
        prism_render_architecture::render_fabric::RenderDomain::Surface,
        class,
        prism_render_material::MaterialFeatureFlags::default(),
    ))
}

/// Resolve the complete pipeline-state family from the domain, blend class,
/// and feature flags. Domain-specific paths are deliberately explicit so a
/// domain migration cannot silently change phase ordering.
#[must_use]
pub fn phase_for_material(record: &MaterialRecord) -> RenderPhase {
    phase_from_material(material_pipeline_state(record))
}

const fn phase_from_material(state: prism_render_material::MaterialPipelineState) -> RenderPhase {
    match state.phase {
        MaterialRenderPhase::Opaque => RenderPhase::Opaque,
        MaterialRenderPhase::Transparent => RenderPhase::Transparent,
        MaterialRenderPhase::Custom => RenderPhase::Custom,
    }
}

/// One queued draw: the pipeline to bind plus the mesh/material it draws.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct PhaseItem {
    /// The specialized pipeline this draw binds.
    pub pipeline: RenderPipelineId,
    /// The draw function that records this item's commands.
    pub draw: DrawFunctionId,
    /// The geometry (mesh) to draw.
    pub geometry: GeometryHandle,
    /// The vertex buffers bound at slots `0..vertices.len()`.
    pub vertices: Vec<BufferId>,
    /// Exact byte length of each prepared vertex stream.
    pub vertex_sizes: Vec<u64>,
    /// The index buffer and format, or `None` for a non-indexed draw.
    pub indices: Option<(BufferId, IndexFormat)>,
    /// Exact byte length of the prepared index data, when indexed.
    pub index_size: Option<u64>,
    /// The first element to draw.
    pub first: u32,
    /// The number of elements to draw.
    pub count: u32,
    /// The base vertex added to each index.
    pub base_vertex: i32,
    /// The indirect argument range, for the GPU-driven bin path.
    ///
    /// When set, the draw function issues one multi-draw from this range instead
    /// of the item's direct element range (the vertex/index buffers are still
    /// bound; the arguments supply counts, first index, and base vertex).
    pub indirect: Option<IndirectArgs>,
    /// The resolved LOD level this draw renders.
    pub lod_level: u32,
    /// The material whose bindings this draw uses.
    pub material: SceneMaterialHandle,
    /// The scene instance this draw came from.
    pub scene: SceneHandle,
    /// The caller's ordering key within the phase (e.g. a depth or bin key).
    pub order: u64,
}

/// A queued phase item together with its actual per-draw bind groups.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct PhaseItemWithBindings {
    /// The phase item whose draw function and pipeline are recorded.
    pub item: PhaseItem,
    /// Per-draw physical groups and their graph-visible buffer declarations.
    pub bind_groups: Vec<RenderBindGroupBinding>,
}

impl PhaseItem {
    /// Attaches the actual per-draw groups selected for this item.
    #[must_use]
    pub fn with_bind_groups(
        self,
        bind_groups: Vec<RenderBindGroupBinding>,
    ) -> PhaseItemWithBindings {
        PhaseItemWithBindings {
            item: self,
            bind_groups,
        }
    }
}

impl PhaseItem {
    /// Returns the explicit bind-group identity carried by this item's draw
    /// function contract.
    #[must_use]
    pub fn bind_group_identity(&self, functions: &DrawFunctions) -> Option<BindGroupBindingId> {
        functions.bind_group_identity(self.draw)
    }

    /// Returns the complete bind-group contract carried by this item.
    #[must_use]
    pub fn bind_groups<'a>(&self, functions: &'a DrawFunctions) -> &'a [RenderBindGroupBinding] {
        functions.bind_groups(self.draw)
    }

    /// Returns the required per-item binding contracts for this phase item.
    #[must_use]
    pub fn dynamic_bind_group_contracts<'a>(
        &self,
        functions: &'a DrawFunctions,
    ) -> &'a [crate::draw::RenderBindGroupBindingContract] {
        functions.dynamic_bind_group_contracts(self.draw)
    }
}

impl PhaseItemWithBindings {
    /// Returns the actual per-draw groups carried by this item.
    #[must_use]
    pub fn dynamic_bind_groups(&self) -> &[RenderBindGroupBinding] {
        &self.bind_groups
    }
}

/// A range of packed indirect argument records in a declared buffer.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct IndirectArgs {
    /// The argument buffer, declared in the frame graph (so it is synchronized).
    pub args: prism_render_graph::BufferHandle,
    /// The byte offset of the first record.
    pub offset: u64,
    /// The number of records to draw.
    pub count: u32,
    /// The byte stride between records (tightly packed indexed args).
    pub stride: u32,
}

/// A producer-owned, non-indexed GPU argument record. Persistent queues hold
/// physical IDs and an allocation lease, NEVER handles from a previous graph.
/// Only the native consumer creates these after device/queue admission.
#[derive(Clone)]
pub(crate) struct ExternalIndirectDraw {
    pub args: BufferId,
    pub vertices: Vec<(BufferId, u64)>,
    pub owner: alloc::sync::Arc<dyn core::any::Any + Send + Sync>,
}

/// The frame's queued draws, one list per [`RenderPhase`].
///
/// Cloning is cheap (the items are plain value structs), which is what lets a
/// pass closure own the queue it records from instead of borrowing it: a pass
/// outlives the borrow, and the frame graph must not hold a reference into the
/// render world.
#[derive(Clone, Resource)]
pub struct RenderQueues {
    phases: [Vec<PhaseItem>; RenderPhase::COUNT],
    dynamic_bind_groups: [Vec<Vec<RenderBindGroupBinding>>; RenderPhase::COUNT],
    external_indirect: [Vec<Option<ExternalIndirectDraw>>; RenderPhase::COUNT],
    secondary_pipelines: BTreeMap<RenderPipelineId, RenderPipelineId>,
}

impl Default for RenderQueues {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderQueues {
    /// Empty queues for every phase.
    #[must_use]
    pub fn new() -> Self {
        Self {
            phases: core::array::from_fn(|_| Vec::new()),
            dynamic_bind_groups: core::array::from_fn(|_| Vec::new()),
            external_indirect: core::array::from_fn(|_| Vec::new()),
            secondary_pipelines: BTreeMap::new(),
        }
    }

    /// Drops every queued item, ready for the next frame.
    pub fn clear(&mut self) {
        for phase in &mut self.phases {
            phase.clear();
        }
        for phase in &mut self.dynamic_bind_groups {
            phase.clear();
        }
        for phase in &mut self.external_indirect {
            phase.clear();
        }
        self.secondary_pipelines.clear();
    }

    /// Appends `item` to `phase`'s list.
    pub fn push(&mut self, phase: RenderPhase, item: PhaseItem) {
        self.push_with_bind_groups(phase, item, Vec::new());
    }

    /// Appends one item and its dynamic bindings as an indivisible queue entry.
    pub fn push_with_bind_groups(
        &mut self,
        phase: RenderPhase,
        item: PhaseItem,
        bind_groups: Vec<RenderBindGroupBinding>,
    ) {
        self.push_with_external_indirect(phase, item, bind_groups, None);
    }

    pub(crate) fn push_with_external_indirect(
        &mut self,
        phase: RenderPhase,
        item: PhaseItem,
        bind_groups: Vec<RenderBindGroupBinding>,
        external: Option<ExternalIndirectDraw>,
    ) {
        self.phases[phase.index()].push(item);
        self.dynamic_bind_groups[phase.index()].push(bind_groups);
        self.external_indirect[phase.index()].push(external);
    }

    /// Copies an indivisible entry when frame routing splits a queue. Keeping
    /// both sidecars here avoids silently losing a producer lease in a split.
    pub(crate) fn copy_item_to(&self, phase: RenderPhase, index: usize, to: &mut Self) {
        let item = &self.phases[phase.index()][index];
        if let Some(&secondary) = self.secondary_pipelines.get(&item.pipeline) {
            to.secondary_pipelines.insert(item.pipeline, secondary);
        }
        to.push_with_external_indirect(
            phase,
            item.clone(),
            self.dynamic_bind_groups(phase, index).to_vec(),
            self.external_indirect(phase, index).cloned(),
        );
    }

    pub(crate) fn external_indirect(
        &self,
        phase: RenderPhase,
        index: usize,
    ) -> Option<&ExternalIndirectDraw> {
        self.external_indirect[phase.index()][index].as_ref()
    }

    /// Called only on a graph-local clone, including for every output graph.
    pub(crate) fn resolve_external_indirect(
        &mut self,
        mut import: impl FnMut(&ExternalIndirectDraw) -> IndirectArgs,
    ) {
        for phase in RenderPhase::ORDER {
            for (item, external) in self.phases[phase.index()]
                .iter_mut()
                .zip(&self.external_indirect[phase.index()])
            {
                if let Some(external) = external {
                    item.indirect = Some(import(external));
                }
            }
        }
    }

    /// Appends a pre-bound phase item.
    pub fn push_bound(&mut self, phase: RenderPhase, item: PhaseItemWithBindings) {
        self.push_with_bind_groups(phase, item.item, item.bind_groups);
    }

    /// The items queued for `phase`, in queue order.
    #[must_use]
    pub fn phase(&self, phase: RenderPhase) -> &[PhaseItem] {
        &self.phases[phase.index()]
    }

    /// Returns the dynamic binding list aligned with an item in `phase`.
    #[must_use]
    pub fn dynamic_bind_groups(
        &self,
        phase: RenderPhase,
        item_index: usize,
    ) -> &[RenderBindGroupBinding] {
        self.dynamic_bind_groups[phase.index()]
            .get(item_index)
            .unwrap_or_else(|| panic!("render queue binding sidecar is missing an item"))
    }

    /// Sorts every phase deterministically.
    ///
    /// The primary key is [`PhaseItem::order`]; ties break on
    /// `(pipeline, geometry, material, scene)`, so the recorded order does not
    /// depend on the order the visible set happened to be walked in.
    pub fn sort(&mut self) {
        for phase_index in 0..RenderPhase::COUNT {
            let phase = &mut self.phases[phase_index];
            let bindings = &mut self.dynamic_bind_groups[phase_index];
            let external = &mut self.external_indirect[phase_index];
            assert_eq!(
                phase.len(),
                bindings.len(),
                "render queue item and binding sidecar lengths diverged"
            );
            assert_eq!(
                phase.len(),
                external.len(),
                "external lease sidecar diverged"
            );
            let mut entries: Vec<_> = phase
                .drain(..)
                .zip(bindings.drain(..))
                .zip(external.drain(..))
                .collect();
            entries.sort_by_key(|((item, _), _)| {
                (
                    item.order,
                    item.pipeline,
                    item.geometry,
                    item.material,
                    item.scene,
                )
            });
            for ((item, dynamic_bind_groups), lease) in entries {
                phase.push(item);
                bindings.push(dynamic_bind_groups);
                external.push(lease);
            }
        }
    }

    /// Records every queued item of `phase` into `context`.
    ///
    /// Items are recorded in queue order (call [`sort`](Self::sort) first to
    /// make that deterministic), each through its own [`DrawFunctionId`].
    pub fn record(
        &self,
        phase: RenderPhase,
        context: &mut dyn crate::MeshRecorder,
        functions: &DrawFunctions,
    ) {
        for (index, item) in self.phase(phase).iter().enumerate() {
            functions.call_with_bindings(
                item.draw,
                item,
                self.dynamic_bind_groups(phase, index),
                context,
            );
        }
    }

    /// Records the paired surface-material raster recipe for a phase. Every
    /// item must have a registered secondary pipeline; silently falling back to
    /// the primary recipe would leave identity/UV attachments unwritten.
    pub(crate) fn record_secondary(
        &self,
        phase: RenderPhase,
        context: &mut dyn crate::MeshRecorder,
        functions: &DrawFunctions,
    ) {
        for (index, item) in self.phase(phase).iter().enumerate() {
            let secondary = self
                .secondary_pipelines
                .get(&item.pipeline)
                .copied()
                .unwrap_or_else(|| {
                    panic!(
                        "secondary raster pipeline missing for primary {:?}",
                        item.pipeline
                    )
                });
            let mut item = item.clone();
            item.pipeline = secondary;
            functions.call_with_bindings(
                item.draw,
                &item,
                self.dynamic_bind_groups(phase, index),
                context,
            );
        }
    }

    /// Associates the second pass pipeline with its primary pipeline.
    pub(crate) fn register_secondary_pipeline(
        &mut self,
        primary: RenderPipelineId,
        secondary: RenderPipelineId,
    ) {
        if let Some(existing) = self.secondary_pipelines.insert(primary, secondary) {
            assert_eq!(
                existing, secondary,
                "one primary pipeline cannot map to multiple secondary recipes"
            );
        }
    }

    /// Specializes and queues each resolved request.
    ///
    /// `build` supplies the pipeline descriptor for a variant (it needs the
    /// renderer's shader and target state, which the queue does not own). The
    /// specializer dedups per `(variant, layout)`, so two meshes that share a
    /// variant and layout share one pipeline.
    pub fn fill<D: RenderDevice>(
        &mut self,
        device: &D,
        specializer: &mut MeshPipelineSpecializer,
        requests: impl IntoIterator<Item = DrawRequest>,
        build: impl Fn(&MeshPipelineKey) -> RenderPipelineDescriptor,
    ) {
        for request in requests {
            let pipeline = specializer
                .specialize_with(device, request.variant, &request.layout, &build)
                .id();
            let dynamic_bind_groups = Vec::new();
            self.push_with_bind_groups(
                request.phase,
                PhaseItem {
                    pipeline,
                    draw: request.draw,
                    geometry: request.geometry,
                    vertices: request.vertices,
                    vertex_sizes: request.vertex_sizes,
                    indices: request.indices,
                    index_size: request.index_size,
                    first: request.first,
                    count: request.count,
                    base_vertex: request.base_vertex,
                    indirect: None,
                    lod_level: request.lod_level,
                    material: request.material,
                    scene: request.scene,
                    order: request.order,
                },
                dynamic_bind_groups,
            );
        }
    }

    /// Specializes and queues resolved requests with per-draw physical groups.
    pub fn fill_with_bindings<D: RenderDevice>(
        &mut self,
        device: &D,
        specializer: &mut MeshPipelineSpecializer,
        requests: impl IntoIterator<Item = DrawRequestWithBindings>,
        build: impl Fn(&MeshPipelineKey) -> RenderPipelineDescriptor,
    ) {
        for DrawRequestWithBindings {
            request,
            bind_groups,
        } in requests
        {
            let pipeline = specializer
                .specialize_with(device, request.variant, &request.layout, &build)
                .id();
            self.push_with_bind_groups(
                request.phase,
                PhaseItem {
                    pipeline,
                    draw: request.draw,
                    geometry: request.geometry,
                    vertices: request.vertices,
                    vertex_sizes: request.vertex_sizes,
                    indices: request.indices,
                    index_size: request.index_size,
                    first: request.first,
                    count: request.count,
                    base_vertex: request.base_vertex,
                    indirect: None,
                    lod_level: request.lod_level,
                    material: request.material,
                    scene: request.scene,
                    order: request.order,
                },
                bind_groups,
            );
        }
    }

    /// Resolves the frame's visible work and queues it.
    ///
    /// This is the queue stage end to end: for each visible
    /// [`GpuRenderWorkItem`](prism_render_visibility::GpuRenderWorkItem), the
    /// [`SceneRegistry`](crate::SceneRegistry) resolves it into a
    /// [`DrawRequest`](crate::DrawRequest), the [`MeshPipelineSpecializer`]
    /// specializes it, and the item lands in its phase. Returns the number of
    /// draws queued; a work item whose geometry/material is not resident is
    /// skipped.
    pub fn fill_from_visibility<D: RenderDevice, P: crate::MeshPipelineProvider>(
        &mut self,
        device: &D,
        specializer: &mut MeshPipelineSpecializer,
        registry: &crate::SceneRegistry,
        meshes: &crate::RenderAssets<crate::RenderMesh>,
        work: &[prism_render_visibility::GpuRenderWorkItem],
        depth_format: Option<prism_render_driver::TextureFormat>,
        sample_count: u32,
        draw: DrawFunctionId,
        provider: &P,
    ) -> usize {
        let mut queued = 0;
        for item in work {
            let Some(requests) =
                registry.resolve_passes(item, meshes, depth_format, sample_count, draw)
            else {
                continue;
            };
            for mut request in requests {
                if provider.uses_surface_uv1_fallback() {
                    request.layout = match surface_uv1_layout(&request.layout) {
                        Ok(layout) => layout,
                        Err(_) => continue,
                    };
                }
                let pipeline = specializer
                    .specialize_with_provider(device, request.variant, &request.layout, provider)
                    .id();
                if let Some(descriptor) = provider.secondary_descriptor(&request.variant) {
                    let secondary = specializer.specialize(
                        device,
                        request.variant,
                        &request.layout,
                        descriptor,
                    );
                    self.register_secondary_pipeline(pipeline, secondary);
                }
                self.push_with_bind_groups(
                    request.phase,
                    PhaseItem {
                        pipeline,
                        draw: request.draw,
                        geometry: request.geometry,
                        vertices: request.vertices,
                        vertex_sizes: request.vertex_sizes,
                        indices: request.indices,
                        index_size: request.index_size,
                        first: request.first,
                        count: request.count,
                        base_vertex: request.base_vertex,
                        indirect: None,
                        lod_level: request.lod_level,
                        material: request.material,
                        scene: request.scene,
                        order: request.order,
                    },
                    Vec::new(),
                );
                queued += 1;
            }
        }
        queued
    }

    /// The total number of queued items across every phase.
    #[must_use]
    pub fn len(&self) -> usize {
        self.phases.iter().map(Vec::len).sum()
    }

    /// Whether no items are queued.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
