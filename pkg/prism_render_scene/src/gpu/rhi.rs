//! RHI resource tables on the device that actually owns Bevy's Scene resources.
//!
//! Register [`init_scene_rhi`] explicitly in `RenderStartup`, before resource
//! allocation systems. This module does not install a plugin, use `PrismGpu`,
//! request a device, replace host callbacks, or submit commands. The host owns
//! submission, completion, device-loss handling, and recovery.
//!
//! Allocate here and clone the exported Bevy handles for native consumers;
//! arbitrary native resource import is deliberately absent. Keep leases alive
//! while recording/encoding their table-local IDs. Keep graph allocations until
//! the host confirms completion of the submission containing their commands;
//! encoding, callback registration, and callback disposal are not completion.
//!
//! A device `Arc` retains the table, not its individual entries. Share a lease
//! with `Arc` when several native worker threads use its ID; every ID user must
//! retain that lease until its last table lookup finishes. Table locking alone
//! cannot prevent a stale-ID lookup after another owner drops the lease.
//! Bevy's web-with-atomics wrappers still require their original wgpu thread;
//! the native worker-thread tests do not establish web thread compatibility.

use core::ops::Deref;
use std::sync::Arc;

use bevy_ecs::{resource::Resource, world::World};
use bevy_render::{
    render_resource::{Buffer, Texture, TextureView},
    renderer::{RenderAdapter, RenderDevice, RenderQueue},
};
use prism_render_driver::{self as rhi, RenderDevice as _};
use prism_render_driver_wgpu::{BackendError, WgpuDevice, WgpuQueue};

/// One RHI table and queue adopted from the render world's original host tuple.
///
/// The native device and queue are exact clones, not a second logical device.
/// Each construction creates a distinct resource table, even on the same host
/// device. Never mix IDs from different instances or the independent `PrismGpu`.
/// On recovery, retire old work before installing a new bridge; old leases keep
/// their original table/device alive and cannot remove replacement table slots.
#[derive(Resource)]
pub struct SceneRhi {
    device: Arc<WgpuDevice>,
    queue: WgpuQueue,
}

impl SceneRhi {
    /// Adopts the adapter/device/queue belonging to the same Bevy render world.
    ///
    /// The caller supplies the original host tuple, not handles collected from
    /// unrelated worlds. wgpu exposes no safe queue-owner query; this does not
    /// claim to authenticate arbitrary native tuples or cross-instance IDs.
    #[must_use]
    pub fn from_bevy(adapter: &RenderAdapter, device: &RenderDevice, queue: &RenderQueue) -> Self {
        // RenderAdapter/RenderQueue implement Deref to the native wgpu types.
        // Calling clone on the wrappers themselves would clone a Bevy wrapper.
        let (device, queue) = WgpuDevice::adopt(
            adapter.deref().clone(),
            device.wgpu_device().clone(),
            queue.deref().clone(),
        );
        Self {
            device: Arc::new(device),
            queue,
        }
    }

    /// The owning device/table, also retained by every resource lease.
    #[must_use]
    pub fn device(&self) -> &Arc<WgpuDevice> {
        &self.device
    }

    /// The adopted queue; use `encode` for record-only native handoff.
    ///
    /// The host submits the resulting command buffer on its `RenderQueue` and
    /// retains frame allocations through actual completion. This accessor does
    /// not submit, poll, or transfer submission ownership to Prism.
    /// Register completion only after that submission on the same host queue.
    /// The host must drive callbacks and distinguish callback execution from
    /// disposal; this bridge supplies no completion or retirement manager.
    #[must_use]
    pub fn queue(&self) -> &WgpuQueue {
        &self.queue
    }

    /// Allocates through RHI and exports the exact native buffer to Bevy.
    #[must_use]
    pub fn create_buffer(&self, descriptor: &rhi::BufferDescriptor) -> SceneRhiBuffer {
        let id = self.device.create_buffer(descriptor);
        SceneRhiBuffer {
            device: Arc::clone(&self.device),
            id,
            buffer: Buffer::from(self.device.wgpu_buffer(id)),
        }
    }

    /// Allocates through RHI and exports the exact native texture to Bevy.
    ///
    /// Backend conversion failures are returned; native validation retains the
    /// host device's existing error scopes/handler.
    pub fn create_texture(
        &self,
        descriptor: &rhi::TextureDescriptor,
    ) -> Result<SceneRhiTexture, BackendError> {
        let id = self.device.try_create_texture(descriptor)?;
        Ok(SceneRhiTexture {
            device: Arc::clone(&self.device),
            id,
            texture: Texture::from(self.device.wgpu_texture(id)),
        })
    }
}

/// Exclusive startup system: insert once, without deferred Commands or a plugin.
///
/// Requires the three Bevy renderer resources. Repeated initialization preserves
/// the table and its live IDs. Device replacement is an explicit host recovery
/// operation, not an implicit re-adoption during startup/resource allocation.
pub fn init_scene_rhi(world: &mut World) {
    if world.contains_resource::<SceneRhi>() {
        return;
    }
    let bridge = SceneRhi::from_bevy(
        world.resource::<RenderAdapter>(),
        world.resource::<RenderDevice>(),
        world.resource::<RenderQueue>(),
    );
    world.insert_resource(bridge);
}

/// Move-only table entry plus a Bevy handle to its exact native buffer.
///
/// Clone `buffer()` for native consumers; cloning it does not extend the RHI
/// ID's lifetime. Drop only removes the table entry, never calls native destroy,
/// submits, or waits. Do not manually destroy this ID or its native resource.
/// Native commands already encoded by wgpu keep their own resource references;
/// this does not make the old RHI ID reusable or retire pooled graph storage.
#[must_use = "retain the lease while its RHI ID is needed"]
pub struct SceneRhiBuffer {
    device: Arc<WgpuDevice>,
    id: rhi::BufferId,
    buffer: Buffer,
}

impl SceneRhiBuffer {
    /// ID in this lease's original resource table, not Bevy's BufferId.
    #[must_use]
    pub fn id(&self) -> rhi::BufferId {
        self.id
    }

    /// The creating device/table, retained independently of the Scene resource.
    #[must_use]
    pub fn device(&self) -> &Arc<WgpuDevice> {
        &self.device
    }

    /// Bevy handle to the exact native RHI allocation; cheap to clone.
    #[must_use]
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }
}

impl Drop for SceneRhiBuffer {
    fn drop(&mut self) {
        self.device.destroy_buffer(self.id);
    }
}

/// Move-only table entry plus a Bevy handle to its exact native texture.
///
/// Native clones/views can outlive this lease. Its ID cannot: keep this owner
/// through all recording/encoding that resolves the texture in the RHI table.
/// Do not manually destroy this ID or explicitly destroy the native texture.
#[must_use = "retain the lease while its RHI ID is needed"]
pub struct SceneRhiTexture {
    device: Arc<WgpuDevice>,
    id: rhi::TextureId,
    texture: Texture,
}

impl SceneRhiTexture {
    /// ID in this lease's original resource table, not Bevy's TextureId.
    #[must_use]
    pub fn id(&self) -> rhi::TextureId {
        self.id
    }

    /// The creating device/table.
    #[must_use]
    pub fn device(&self) -> &Arc<WgpuDevice> {
        &self.device
    }

    /// Bevy handle to the exact native RHI allocation; cheap to clone.
    #[must_use]
    pub fn texture(&self) -> &Texture {
        &self.texture
    }

    /// Creates a view through the same RHI table, with an exact Bevy clone.
    ///
    /// For a Bevy-only view, `texture().create_view(...)` also works, but does
    /// not register that view in the RHI table or provide an RHI view ID.
    pub fn create_view(
        &self,
        descriptor: &rhi::TextureViewDescriptor,
    ) -> Result<SceneRhiTextureView, BackendError> {
        let id = self.device.try_create_texture_view(self.id, descriptor)?;
        Ok(SceneRhiTextureView {
            device: Arc::clone(&self.device),
            id,
            view: TextureView::from(self.device.wgpu_texture_view(id)),
        })
    }
}

impl Drop for SceneRhiTexture {
    fn drop(&mut self) {
        self.device.destroy_texture(self.id);
    }
}

/// Move-only RHI view entry; its native Bevy clone retains the underlying texture.
#[must_use = "retain the lease while its RHI view ID is needed"]
pub struct SceneRhiTextureView {
    device: Arc<WgpuDevice>,
    id: rhi::TextureViewId,
    view: TextureView,
}

impl SceneRhiTextureView {
    /// ID in this lease's original resource table, not Bevy's TextureViewId.
    #[must_use]
    pub fn id(&self) -> rhi::TextureViewId {
        self.id
    }

    /// The creating device/table.
    #[must_use]
    pub fn device(&self) -> &Arc<WgpuDevice> {
        &self.device
    }

    /// Bevy handle to the exact native RHI view; cheap to clone.
    #[must_use]
    pub fn view(&self) -> &TextureView {
        &self.view
    }
}

impl Drop for SceneRhiTextureView {
    fn drop(&mut self) {
        self.device.destroy_texture_view(self.id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_platform::future::block_on;
    use prism_render_driver::Adapter as _;
    use prism_render_driver_wgpu::{
        require_hardware_gpu, AdapterEvidence, WgpuAdapter, WgpuInstance,
    };
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Mutex,
    };

    // These are bridge-boundary GPU tests, not evidence of Scene/Fabric
    // production registration, pass allocation, submission, or retirement.
    struct Host {
        _instance: Arc<WgpuInstance>,
        adapter: RenderAdapter,
        device: RenderDevice,
        queue: RenderQueue,
    }

    fn host_or_skip() -> Option<Host> {
        let instance = Arc::new(WgpuInstance::new());
        let Some(adapter) = instance.request_adapter(wgpu::PowerPreference::HighPerformance, false)
        else {
            assert!(!require_hardware_gpu(), "PRISM_REQUIRE_GPU=1: no adapter");
            eprintln!("SKIPPED SceneRhi bridge GPU test: no adapter");
            return None;
        };
        let evidence = AdapterEvidence::from_info(WgpuAdapter::new(adapter.clone()).info());
        if !evidence.is_hardware() {
            assert!(
                !require_hardware_gpu(),
                "PRISM_REQUIRE_GPU=1 requires physical hardware: {evidence}"
            );
            eprintln!("SKIPPED SceneRhi physical GPU test: {evidence}");
            return None;
        }
        eprintln!("SceneRhi bridge REAL HARDWARE GPU: {evidence}");
        // Only the test host requests a device. The production bridge receives
        // exactly the Bevy wrappers installed from this one original request.
        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("scene-rhi-test-host"),
            ..Default::default()
        }))
        .expect("hardware host device");
        Some(Host {
            _instance: instance,
            adapter: RenderAdapter::new(adapter),
            device: RenderDevice::new(device),
            queue: RenderQueue::new(queue),
        })
    }

    impl Host {
        fn bridge(&self) -> SceneRhi {
            SceneRhi::from_bevy(&self.adapter, &self.device, &self.queue)
        }
    }

    fn mapped_bytes(host: &Host, buffer: &wgpu::Buffer) -> Vec<u8> {
        let result = Arc::new(Mutex::new(None));
        let callback_result = Arc::clone(&result);
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |status| {
                *callback_result.lock().expect("map result") = Some(status);
            });
        host.device
            .wgpu_device()
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("host poll");
        assert_eq!(result.lock().expect("map result").take(), Some(Ok(())));
        let mapped = buffer.slice(..).get_mapped_range().expect("mapped bytes");
        let bytes = mapped.to_vec();
        drop(mapped);
        buffer.unmap();
        bytes
    }

    fn read_buffer(host: &Host, source: &Buffer) -> Vec<u8> {
        let staging = host.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scene-rhi-host-readback"),
            size: source.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = host
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(source, 0, &staging, 0, source.size());
        host.queue.submit([encoder.finish()]);
        mapped_bytes(host, &staging)
    }

    #[test]
    fn bridge_and_leases_are_send_sync() {
        fn send_sync<T: Send + Sync>() {}
        fn resource<T: Resource>() {}
        resource::<SceneRhi>();
        send_sync::<SceneRhi>();
        send_sync::<SceneRhiBuffer>();
        send_sync::<SceneRhiTexture>();
        send_sync::<SceneRhiTextureView>();
    }

    #[test]
    fn startup_adopts_exact_bevy_handles_once() {
        let Some(host) = host_or_skip() else { return };
        let mut world = World::new();
        world.insert_resource(host.adapter.clone());
        world.insert_resource(host.device.clone());
        world.insert_resource(host.queue.clone());
        let mut startup = bevy_ecs::schedule::Schedule::new(bevy_render::RenderStartup);
        startup.add_systems(init_scene_rhi);
        startup.run(&mut world);
        let device = Arc::clone(world.resource::<SceneRhi>().device());
        // Equality is checked only against clones from this same Instance.
        assert_eq!(device.wgpu_device(), host.device.wgpu_device());
        assert_eq!(device.wgpu_adapter(), host.adapter.deref());
        assert_eq!(
            world.resource::<SceneRhi>().queue().wgpu_queue(),
            host.queue.deref()
        );
        let lease = world
            .resource::<SceneRhi>()
            .create_buffer(&rhi::BufferDescriptor::new(4, rhi::BufferUsages::COPY_DST));
        startup.run(&mut world);
        assert!(Arc::ptr_eq(&device, world.resource::<SceneRhi>().device()));
        assert_eq!(device.wgpu_buffer(lease.id()), *lease.buffer().deref());
        drop(world);
        assert_eq!(lease.device().wgpu_buffer(lease.id()).size(), 4);
    }

    #[test]
    fn buffer_lease_retains_table_but_drop_does_not_destroy_native_clone() {
        let Some(host) = host_or_skip() else { return };
        let bridge = host.bridge();
        let lease = bridge.create_buffer(&rhi::BufferDescriptor::new(
            4,
            rhi::BufferUsages::COPY_SRC | rhi::BufferUsages::COPY_DST,
        ));
        let buffer = lease.buffer().clone();
        let id = lease.id();
        let weak = Arc::downgrade(lease.device());
        assert!(Arc::ptr_eq(bridge.device(), lease.device()));
        assert_eq!(lease.device().wgpu_buffer(id), *buffer.deref());
        drop(bridge);
        // No independent strong device Arc may mask a missing lease owner.
        assert_eq!(weak.strong_count(), 1, "only the lease retains this table");
        assert!(weak.upgrade().is_some(), "lease retains its original table");
        let device = Arc::clone(lease.device());
        drop(lease);
        assert!(catch_unwind(AssertUnwindSafe(|| device.wgpu_buffer(id))).is_err());
        drop(device);
        assert!(weak.upgrade().is_none());
        // Native use after every RHI owner disappeared catches an erroneous
        // native Buffer::destroy in the lease's Drop implementation.
        let payload = [1, 3, 5, 7];
        host.queue.write_buffer(&buffer, 0, &payload);
        assert_eq!(read_buffer(&host, &buffer), payload);
    }

    #[test]
    fn texture_and_view_leases_export_exact_handles_without_native_destruction() {
        let Some(host) = host_or_skip() else { return };
        let bridge = host.bridge();
        let lease = bridge
            .create_texture(&rhi::TextureDescriptor::new_2d(
                rhi::Extent3d::new_2d(1, 1),
                rhi::TextureFormat::Rgba8Unorm,
                rhi::TextureUsages::COPY_SRC
                    | rhi::TextureUsages::COPY_DST
                    | rhi::TextureUsages::TEXTURE_BINDING,
            ))
            .expect("RHI texture");
        let view_lease = lease
            .create_view(&rhi::TextureViewDescriptor::default())
            .expect("RHI view");
        let device = Arc::clone(lease.device());
        let texture_id = lease.id();
        let view_id = view_lease.id();
        let texture = lease.texture().clone();
        let view = view_lease.view().clone();
        assert!(Arc::ptr_eq(lease.device(), view_lease.device()));
        assert_eq!(device.wgpu_texture(texture_id), *texture.deref());
        assert_eq!(device.wgpu_texture_view(view_id), *view.deref());
        drop(lease);
        // A view remains resolvable after removal of the texture table entry.
        assert_eq!(device.wgpu_texture_view(view_id), *view.deref());
        drop(view_lease);
        assert!(catch_unwind(AssertUnwindSafe(|| device.wgpu_texture(texture_id))).is_err());
        assert!(catch_unwind(AssertUnwindSafe(|| device.wgpu_texture_view(view_id))).is_err());
        drop((device, bridge));
        let payload = [11, 22, 33, 44];
        host.queue.write_texture(
            texture.as_image_copy(),
            &payload,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let staging = host.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scene-rhi-texture-readback"),
            size: 256,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = host.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(1),
                },
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        host.queue.submit([encoder.finish()]);
        assert_eq!(&mapped_bytes(&host, &staging)[..4], &payload);
        // Validate that the surviving Bevy view can still be bound by the host.
        let layout =
            host.device
                .wgpu_device()
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    }],
                });
        let _group = host
            .device
            .wgpu_device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                }],
            });
    }

    #[test]
    fn record_only_copy_waits_for_host_submission_and_real_completion() {
        let Some(host) = host_or_skip() else { return };
        let bridge = host.bridge();
        let source = bridge.create_buffer(&rhi::BufferDescriptor {
            label: None,
            size: 4,
            usage: rhi::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        });
        let payload = [2, 4, 6, 8];
        source
            .buffer()
            .slice(..)
            .get_mapped_range_mut()
            .expect("mapped source")
            .copy_from_slice(&payload);
        source.buffer().unmap();
        let destination = bridge.create_buffer(&rhi::BufferDescriptor::new(
            4,
            rhi::BufferUsages::COPY_SRC | rhi::BufferUsages::COPY_DST,
        ));
        let output = destination.buffer().clone();
        let source_id = source.id();
        let destination_id = destination.id();
        let device = Arc::clone(bridge.device());
        let commands = rhi::CommandBuffer {
            label: Some("scene-rhi-host-owned-copy".into()),
            passes: vec![rhi::Pass::Transfer {
                label: None,
                commands: vec![rhi::CopyCommand::CopyBufferToBuffer {
                    src: source_id,
                    src_offset: 0,
                    dst: destination_id,
                    dst_offset: 0,
                    size: 4,
                }],
            }],
        };
        let encoded = bridge.queue().encode(&commands);
        assert_eq!(
            read_buffer(&host, &output),
            [0; 4],
            "encode must not submit"
        );
        let complete = Arc::new(AtomicBool::new(false));
        let callback_complete = Arc::clone(&complete);
        let submission = host.queue.submit([encoded]);
        // Register on the actual Bevy host queue AFTER its submission. No
        // synthetic completion signal or bridge-owned submit is substituted.
        host.queue.on_submitted_work_done(move || {
            drop((source, destination));
            callback_complete.store(true, Ordering::Release);
        });
        drop(bridge);
        assert!(!complete.load(Ordering::Acquire));
        assert_eq!(device.wgpu_buffer(source_id).size(), 4);
        assert_eq!(device.wgpu_buffer(destination_id).size(), 4);
        host.device
            .wgpu_device()
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .expect("host submission completion");
        assert!(
            complete.load(Ordering::Acquire),
            "real host callback executed"
        );
        assert!(catch_unwind(AssertUnwindSafe(|| device.wgpu_buffer(source_id))).is_err());
        assert!(catch_unwind(AssertUnwindSafe(|| device.wgpu_buffer(destination_id))).is_err());
        assert_eq!(read_buffer(&host, &output), payload);
    }

    #[test]
    fn old_lease_drop_cannot_remove_a_same_device_readoptions_slot() {
        let Some(host) = host_or_skip() else { return };
        let old = host.bridge();
        let replacement = host.bridge();
        let descriptor = rhi::BufferDescriptor::new(4, rhi::BufferUsages::COPY_DST);
        let old_buffer = old.create_buffer(&descriptor);
        let new_buffer = replacement.create_buffer(&descriptor);
        assert!(!Arc::ptr_eq(old_buffer.device(), new_buffer.device()));
        assert_eq!(
            old_buffer.id(),
            new_buffer.id(),
            "table-local IDs may collide"
        );
        assert_ne!(old_buffer.buffer().deref(), new_buffer.buffer().deref());
        drop((old, old_buffer));
        assert_eq!(
            replacement.device().wgpu_buffer(new_buffer.id()),
            *new_buffer.buffer().deref()
        );
    }

    #[test]
    fn native_worker_threads_allocate_and_retire_buffer_texture_and_view_leases() {
        let Some(host) = host_or_skip() else { return };
        let bridge = host.bridge();
        let leases = std::thread::scope(|scope| {
            let workers: Vec<_> = (0..4)
                .map(|_| {
                    let bridge = &bridge;
                    scope.spawn(move || {
                        let buffer = bridge.create_buffer(&rhi::BufferDescriptor::new(
                            4,
                            rhi::BufferUsages::COPY_SRC | rhi::BufferUsages::COPY_DST,
                        ));
                        let texture = bridge
                            .create_texture(&rhi::TextureDescriptor::new_2d(
                                rhi::Extent3d::new_2d(1, 1),
                                rhi::TextureFormat::Rgba8Unorm,
                                rhi::TextureUsages::RENDER_ATTACHMENT
                                    | rhi::TextureUsages::COPY_SRC,
                            ))
                            .expect("worker texture");
                        let view = texture
                            .create_view(&rhi::TextureViewDescriptor::default())
                            .expect("worker view");
                        (buffer, texture, view)
                    })
                })
                .collect();
            workers
                .into_iter()
                .map(|worker| worker.join().expect("worker allocation"))
                .collect::<Vec<_>>()
        });
        // Each table entry must be distinct despite simultaneous allocations.
        for (index, (buffer, texture, view)) in leases.iter().enumerate() {
            for (other_buffer, other_texture, other_view) in &leases[..index] {
                assert_ne!(buffer.id(), other_buffer.id());
                assert_ne!(texture.id(), other_texture.id());
                assert_ne!(view.id(), other_view.id());
            }
        }
        let weak = Arc::downgrade(bridge.device());
        drop(bridge);
        let native = std::thread::scope(|scope| {
            let workers: Vec<_> = leases
                .into_iter()
                .map(|(buffer, texture, view)| {
                    scope.spawn(move || {
                        let device = Arc::clone(buffer.device());
                        let ids = (buffer.id(), texture.id(), view.id());
                        let native = (
                            buffer.buffer().clone(),
                            texture.texture().clone(),
                            view.view().clone(),
                        );
                        assert_eq!(device.wgpu_buffer(ids.0), *native.0.deref());
                        assert_eq!(device.wgpu_texture(ids.1), *native.1.deref());
                        assert_eq!(device.wgpu_texture_view(ids.2), *native.2.deref());
                        // Drop from workers different from the allocating ones.
                        drop((buffer, texture, view));
                        assert!(
                            catch_unwind(AssertUnwindSafe(|| { device.wgpu_buffer(ids.0) }))
                                .is_err()
                        );
                        assert!(
                            catch_unwind(AssertUnwindSafe(|| { device.wgpu_texture(ids.1) }))
                                .is_err()
                        );
                        assert!(catch_unwind(AssertUnwindSafe(|| {
                            device.wgpu_texture_view(ids.2)
                        }))
                        .is_err());
                        native
                    })
                })
                .collect();
            workers
                .into_iter()
                .map(|worker| worker.join().expect("worker retirement"))
                .collect::<Vec<_>>()
        });
        assert!(weak.upgrade().is_none(), "all RHI table owners released");
        for (index, (buffer, texture, view)) in native.into_iter().enumerate() {
            let payload = [index as u8 + 1; 4];
            host.queue.write_buffer(&buffer, 0, &payload);
            assert_eq!(read_buffer(&host, &buffer), payload);
            let staging = host.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("scene-rhi-worker-texture-readback"),
                size: 256,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut encoder = host.device.create_command_encoder(&Default::default());
            {
                let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::RED),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    ..Default::default()
                });
            }
            encoder.copy_texture_to_buffer(
                texture.as_image_copy(),
                wgpu::TexelCopyBufferInfo {
                    buffer: &staging,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(256),
                        rows_per_image: Some(1),
                    },
                },
                wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
            host.queue.submit([encoder.finish()]);
            assert_eq!(&mapped_bytes(&host, &staging)[..4], &[255, 0, 0, 255]);
        }
    }

    #[test]
    fn encoded_native_commands_survive_worker_lease_drop_and_slot_reuse() {
        let Some(host) = host_or_skip() else { return };
        let bridge = host.bridge();
        let source = bridge.create_buffer(&rhi::BufferDescriptor {
            label: None,
            size: 4,
            usage: rhi::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        });
        let payload = [13, 21, 34, 55];
        source
            .buffer()
            .slice(..)
            .get_mapped_range_mut()
            .expect("mapped source")
            .copy_from_slice(&payload);
        source.buffer().unmap();
        let destination = bridge.create_buffer(&rhi::BufferDescriptor::new(
            4,
            rhi::BufferUsages::COPY_DST | rhi::BufferUsages::COPY_SRC,
        ));
        let output = destination.buffer().clone();
        let ids = (source.id(), destination.id());
        let encoded = bridge.queue().encode(&rhi::CommandBuffer {
            label: Some("scene-rhi-native-retention".into()),
            passes: vec![rhi::Pass::Transfer {
                label: None,
                commands: vec![rhi::CopyCommand::CopyBufferToBuffer {
                    src: ids.0,
                    src_offset: 0,
                    dst: ids.1,
                    dst_offset: 0,
                    size: 4,
                }],
            }],
        });
        std::thread::spawn(move || drop((source, destination)))
            .join()
            .expect("worker lease drop");
        assert!(catch_unwind(AssertUnwindSafe(|| bridge.device().wgpu_buffer(ids.0))).is_err());
        assert!(catch_unwind(AssertUnwindSafe(|| bridge.device().wgpu_buffer(ids.1))).is_err());
        // Force both freed slots to be reused. Old native commands must retain
        // their original allocations, not resolve new table occupants at submit.
        let descriptor = rhi::BufferDescriptor::new(4, rhi::BufferUsages::COPY_DST);
        let replacements = [
            bridge.create_buffer(&descriptor),
            bridge.create_buffer(&descriptor),
        ];
        for lease in &replacements {
            assert_ne!(lease.id(), ids.0);
            assert_ne!(lease.id(), ids.1);
            host.queue.write_buffer(lease.buffer(), 0, &[99; 4]);
        }
        let weak = Arc::downgrade(bridge.device());
        drop((replacements, bridge));
        assert!(weak.upgrade().is_none());
        // This is solely native command ownership, not a change to the rule
        // that reusable graph allocations need actual host-confirmed retirement.
        host.queue.submit([encoded]);
        assert_eq!(read_buffer(&host, &output), payload);
    }

    #[test]
    fn actual_replacement_device_is_isolated_from_old_lease_retirement() {
        let Some(host) = host_or_skip() else { return };
        // Only this recovery-boundary test requests a replacement device. The
        // production bridge still never requests devices or performs recovery.
        let (device, queue) = block_on(host.adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("scene-rhi-test-replacement-host"),
            ..Default::default()
        }))
        .expect("replacement host device");
        let replacement_host = Host {
            _instance: Arc::clone(&host._instance),
            adapter: host.adapter.clone(),
            device: RenderDevice::new(device),
            queue: RenderQueue::new(queue),
        };
        // Both requests came from the same Instance; this inequality therefore
        // does not rely on cross-Global native device ID comparisons.
        assert_ne!(
            host.device.wgpu_device(),
            replacement_host.device.wgpu_device()
        );
        let old = host.bridge();
        let replacement = replacement_host.bridge();
        let descriptor = rhi::BufferDescriptor::new(
            4,
            rhi::BufferUsages::COPY_SRC | rhi::BufferUsages::COPY_DST,
        );
        let old_buffer = old.create_buffer(&descriptor);
        let new_buffer = replacement.create_buffer(&descriptor);
        let old_native = old_buffer.buffer().clone();
        assert_eq!(old_buffer.id(), new_buffer.id(), "IDs are table local");
        let old_weak = Arc::downgrade(old.device());
        drop(old);
        std::thread::spawn(move || drop(old_buffer))
            .join()
            .expect("old device lease retirement");
        assert!(old_weak.upgrade().is_none());
        assert_eq!(
            replacement.device().wgpu_buffer(new_buffer.id()),
            *new_buffer.buffer().deref()
        );
        host.queue.write_buffer(&old_native, 0, &[1, 2, 3, 4]);
        replacement_host
            .queue
            .write_buffer(new_buffer.buffer(), 0, &[5, 6, 7, 8]);
        assert_eq!(read_buffer(&host, &old_native), [1, 2, 3, 4]);
        assert_eq!(
            read_buffer(&replacement_host, new_buffer.buffer()),
            [5, 6, 7, 8]
        );
    }
}
