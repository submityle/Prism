//! An RHI device factory sharing its original native tuple with resident decode.
//!
//! Enabled by `resident-page-streaming`. Device acquisition is authenticated
//! by construction through `GpuAdapter`, not by matching adapter descriptions
//! or device IDs from independently acquired instances.

use std::sync::{Arc, Mutex};

use prism_gdeflate_gpu::{GpuAdapter, GpuContext};
use prism_render_driver as rhi;
use prism_render_driver_wgpu::{WgpuAdapter, WgpuDevice, WgpuInstance, WgpuQueue, WgpuSurface};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

/// An adapter acquired directly from a [`ResidentWgpuInstance`]'s native instance.
///
/// The private token retains its acquisition ancestry; the RHI-facing metadata
/// is derived from that same adapter. There is no raw-parts constructor.
pub struct ResidentWgpuAdapter {
    token: GpuAdapter,
    metadata: WgpuAdapter,
}

impl ResidentWgpuAdapter {
    fn new(token: GpuAdapter) -> Self {
        let metadata = WgpuAdapter::new(token.adapter().clone());
        Self { token, metadata }
    }
}

impl rhi::Adapter for ResidentWgpuAdapter {
    fn info(&self) -> &rhi::AdapterInfo {
        rhi::Adapter::info(&self.metadata)
    }

    fn capabilities(&self) -> &rhi::DeviceCapabilities {
        rhi::Adapter::capabilities(&self.metadata)
    }
}

/// A snapshot of one original compute/render device, queue, and RHI table set.
///
/// Clones retain this exact tuple and the same RHI resource tables. Recovery
/// does not update existing snapshots: reacquire through
/// [`ResidentWgpuInstance::resident_device`] after successful runtime recovery.
/// Native resource IDs alone do not authenticate cross-instance ownership.
/// There is deliberately no constructor accepting caller-supplied raw parts.
#[derive(Clone)]
pub struct ResidentRenderDevice {
    context: Arc<GpuContext>,
    device: WgpuDevice,
    queue: WgpuQueue,
}

impl ResidentRenderDevice {
    /// Returns only a snapshot matching the runtime's committed native tuple.
    ///
    /// A factory can publish a candidate before runtime resource replay fails.
    /// This accessor rejects that candidate until the runtime commits it. The
    /// comparison is confined to the factory's original instance, never IDs
    /// from separately acquired instances. The factory retains the exact RHI
    /// tables returned to this runtime, not another raw-device adoption.
    #[must_use]
    pub fn from_runtime(
        runtime: &prism_render_runtime::RenderRuntime<ResidentWgpuInstance>,
    ) -> Option<Self> {
        let snapshot = runtime.instance().resident_device()?;
        (snapshot.context.instance() == runtime.instance().native.wgpu_instance()
            && snapshot.context.device() == runtime.device().wgpu_device()
            && snapshot.context.queue() == runtime.queue().wgpu_queue())
        .then_some(snapshot)
    }

    /// The original context on which resident decoding must run.
    #[must_use]
    pub fn context(&self) -> &GpuContext {
        &self.context
    }

    /// The renderer's device, retaining the original RHI resource tables.
    #[must_use]
    pub fn device(&self) -> &WgpuDevice {
        &self.device
    }

    /// The same native submission queue used by the renderer and decoder.
    #[must_use]
    pub fn queue(&self) -> &WgpuQueue {
        &self.queue
    }
}

/// A wgpu RHI instance that publishes its original renderer/decoder tuple.
///
/// Pass this factory directly to `RenderRuntime` or `PrismRenderPlugin`; they
/// create and recover devices through the ordinary RHI `Instance` contract.
/// Device creation installs Prism's owned-device loss callback, and creates
/// fresh RHI tables for each new native device. Adapter enumeration and
/// surface-compatible selection retain the original native instance.
///
/// Creation is serialized, including publication of the returned pair. The
/// snapshot lock is separate and held only while cloning/replacing a snapshot;
/// reading it does not wait for native device acquisition. No creation lock is
/// held during surface creation or runtime resource-replay callbacks.
///
/// Dedicate the factory to its runtime. Publication describes the latest
/// successful *factory device creation*, not a runtime recovery commit: surface
/// configuration or replay can still fail after creation. After any such
/// failure keep the old runtime snapshot; do not consume the new publication
/// until it matches the runtime's native device/queue on this original instance.
pub struct ResidentWgpuInstance {
    native: WgpuInstance,
    creation: Mutex<()>,
    resident: Mutex<Option<ResidentRenderDevice>>,
}

impl Default for ResidentWgpuInstance {
    fn default() -> Self {
        Self::new()
    }
}

impl ResidentWgpuInstance {
    /// Creates an instance with the native driver's default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::from_native(WgpuInstance::new())
    }

    /// Retains a configured native instance; all adapters are acquired from it.
    ///
    /// No existing device, queue, or resource tables are imported.
    #[must_use]
    pub fn from_wgpu(instance: wgpu::Instance) -> Self {
        Self::from_native(WgpuInstance::from_wgpu(instance))
    }

    fn from_native(native: WgpuInstance) -> Self {
        Self {
            native,
            creation: Mutex::new(()),
            resident: Mutex::new(None),
        }
    }

    /// Returns the most recently created tuple, or `None` before first success.
    ///
    /// Failed device creation leaves the previous publication unchanged. Each
    /// returned clone is an immutable snapshot; it retains the old device and
    /// tables across replacement without moving old resources to a new device.
    /// A runtime recovery failure *after* device creation can leave publication
    /// ahead of the runtime. Use only a snapshot matching that runtime's actual
    /// native tuple, not one inferred from metadata or table-local IDs.
    #[must_use]
    pub fn resident_device(&self) -> Option<ResidentRenderDevice> {
        self.resident
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    fn request(
        &self,
        options: &rhi::RequestAdapterOptions,
        compatible_surface: Option<&wgpu::Surface<'_>>,
    ) -> Option<ResidentWgpuAdapter> {
        let power_preference = match options.power_preference {
            rhi::PowerPreference::None => wgpu::PowerPreference::None,
            rhi::PowerPreference::LowPower => wgpu::PowerPreference::LowPower,
            rhi::PowerPreference::HighPerformance => wgpu::PowerPreference::HighPerformance,
        };
        GpuAdapter::request(
            self.native.wgpu_instance(),
            &wgpu::RequestAdapterOptions {
                power_preference,
                force_fallback_adapter: options.force_fallback,
                compatible_surface,
                apply_limit_buckets: false,
            },
        )
        .ok()
        .map(ResidentWgpuAdapter::new)
    }
}

impl rhi::Instance for ResidentWgpuInstance {
    type Adapter = ResidentWgpuAdapter;
    type Device = WgpuDevice;
    type Queue = WgpuQueue;
    type Surface = WgpuSurface;

    fn enumerate_adapters(&self) -> Vec<Self::Adapter> {
        GpuAdapter::enumerate(self.native.wgpu_instance(), wgpu::Backends::all())
            .into_iter()
            .map(ResidentWgpuAdapter::new)
            .collect()
    }

    fn request_adapter(&self, options: &rhi::RequestAdapterOptions) -> Option<Self::Adapter> {
        self.request(options, None)
    }

    fn request_adapter_for_surface(
        &self,
        options: &rhi::RequestAdapterOptions,
        surface: &Self::Surface,
    ) -> Option<Self::Adapter> {
        self.request(options, Some(surface.wgpu_surface()))
    }

    fn create_device(
        &self,
        adapter: &Self::Adapter,
        descriptor: &rhi::DeviceDescriptor,
    ) -> Result<(Self::Device, Self::Queue), rhi::DeviceError> {
        if adapter.token.instance() != self.native.wgpu_instance() {
            return Err(rhi::DeviceError::Request(
                "resident adapter belongs to a different native instance".into(),
            ));
        }

        let creation = self
            .creation
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let descriptor = WgpuInstance::device_descriptor(adapter.token.adapter(), descriptor);
        let context = Arc::new(
            adapter
                .token
                .request_device(&descriptor)
                .map_err(|error| rhi::DeviceError::Request(error.to_string()))?,
        );
        let (device, queue) = WgpuDevice::own_original(
            context.adapter().clone(),
            context.device().clone(),
            context.queue().clone(),
        );
        let snapshot = ResidentRenderDevice {
            context,
            device: device.clone(),
            queue: queue.clone(),
        };
        let previous = self
            .resident
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .replace(snapshot);
        // Dropping old GPU handles can invoke backend destruction. Do it after
        // releasing both locks, never while readers or another creation wait.
        drop(creation);
        drop(previous);
        Ok((device, queue))
    }

    fn create_surface<W>(
        &self,
        window: Arc<W>,
        device: &Self::Device,
        queue: &Self::Queue,
    ) -> Result<Self::Surface, rhi::SurfaceError>
    where
        W: HasWindowHandle + HasDisplayHandle + Send + Sync + 'static,
    {
        rhi::Instance::create_surface(&self.native, window, device, queue)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_driver::{Adapter, Instance, RenderDevice};
    use prism_render_runtime::{RenderInitError, RenderRuntime};

    #[test]
    fn public_contracts_are_send_sync_and_match_the_rhi() {
        fn send_sync<T: Send + Sync>() {}
        fn rhi_instance<
            T: rhi::Instance<
                Adapter = ResidentWgpuAdapter,
                Device = WgpuDevice,
                Queue = WgpuQueue,
                Surface = WgpuSurface,
            >,
        >() {
        }
        send_sync::<ResidentWgpuInstance>();
        send_sync::<ResidentWgpuAdapter>();
        send_sync::<ResidentRenderDevice>();
        rhi_instance::<ResidentWgpuInstance>();
        let _: fn() -> ResidentWgpuInstance = ResidentWgpuInstance::new;
        let _: fn() -> ResidentWgpuInstance = ResidentWgpuInstance::default;
        let _: fn(wgpu::Instance) -> ResidentWgpuInstance = ResidentWgpuInstance::from_wgpu;
        let _: fn(&ResidentWgpuInstance) -> Option<ResidentRenderDevice> =
            ResidentWgpuInstance::resident_device;
        let _: fn(&ResidentRenderDevice) -> &GpuContext = ResidentRenderDevice::context;
        let _: fn(&ResidentRenderDevice) -> &WgpuDevice = ResidentRenderDevice::device;
        let _: fn(&ResidentRenderDevice) -> &WgpuQueue = ResidentRenderDevice::queue;
        let _: fn(&wgpu::Instance, wgpu::Backends) -> Vec<GpuAdapter> = GpuAdapter::enumerate;
    }

    fn runtime() -> Option<RenderRuntime<ResidentWgpuInstance>> {
        let instance = ResidentWgpuInstance::new();
        assert!(instance.resident_device().is_none());
        match RenderRuntime::headless(
            instance,
            rhi::Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
        ) {
            Ok(runtime) => Some(runtime),
            Err(RenderInitError::NoAdapter) if std::env::var_os("PRISM_REQUIRE_GPU").is_none() => {
                eprintln!("skipping resident instance hardware test: no adapter");
                None
            }
            Err(error) => panic!("resident RHI runtime acquisition failed: {error}"),
        }
    }

    fn assert_runtime_tuple(runtime: &RenderRuntime<ResidentWgpuInstance>) -> ResidentRenderDevice {
        let snapshot = ResidentRenderDevice::from_runtime(runtime).unwrap();
        assert_eq!(
            snapshot.context().instance(),
            runtime.instance().native.wgpu_instance(),
        );
        assert_eq!(snapshot.context().device(), runtime.device().wgpu_device(),);
        assert_eq!(snapshot.context().queue(), runtime.queue().wgpu_queue());
        assert_eq!(
            snapshot.device().wgpu_device(),
            runtime.device().wgpu_device(),
        );
        assert_eq!(snapshot.queue().wgpu_queue(), runtime.queue().wgpu_queue());
        assert_eq!(
            snapshot.context().adapter(),
            runtime.adapter().token.adapter(),
        );
        snapshot
    }

    #[test]
    fn headless_runtime_uses_the_original_resident_device_and_queue() {
        let Some(runtime) = runtime() else { return };
        let snapshot = assert_runtime_tuple(&runtime);
        let buffer = snapshot
            .device()
            .create_buffer(&rhi::BufferDescriptor::new(4, rhi::BufferUsages::COPY_DST));
        // Native device identity is not enough: both handles must also share
        // the same RHI tables, rather than independently adopting that device.
        assert_eq!(
            snapshot.device().wgpu_buffer(buffer),
            runtime.device().wgpu_buffer(buffer),
        );
        runtime.device().destroy_buffer(buffer);
        let cloned = snapshot.clone();
        drop(runtime);
        drop(snapshot);
        assert_eq!(cloned.context().device(), cloned.device().wgpu_device());
        assert_eq!(cloned.context().queue(), cloned.queue().wgpu_queue());
        cloned.context().wait();
    }

    #[test]
    fn enumeration_retains_each_adapters_original_instance_and_metadata() {
        let instance = ResidentWgpuInstance::new();
        let enumerated = instance.enumerate_adapters();
        if enumerated.is_empty() && std::env::var_os("PRISM_REQUIRE_GPU").is_some() {
            panic!("PRISM_REQUIRE_GPU is set but native enumeration returned no adapters");
        }
        for adapter in enumerated {
            assert_eq!(adapter.token.instance(), instance.native.wgpu_instance());
            assert_eq!(adapter.token.adapter(), adapter.metadata.wgpu_adapter());
            assert_eq!(adapter.info(), adapter.metadata.info());
            assert_eq!(adapter.capabilities(), adapter.metadata.capabilities());
        }
        assert!(instance.resident_device().is_none());
        // An empty backend mask must not manufacture a default adapter.
        assert!(
            GpuAdapter::enumerate(instance.native.wgpu_instance(), wgpu::Backends::empty())
                .is_empty()
        );
    }

    #[test]
    fn foreign_adapter_is_rejected_without_replacing_a_publication() {
        let Some(runtime) = runtime() else { return };
        let instance = runtime.instance();
        let original = assert_runtime_tuple(&runtime);
        let foreign = ResidentWgpuInstance::new();
        let foreign_adapter = foreign
            .request_adapter(&rhi::RequestAdapterOptions::default())
            .expect("second instance must expose an adapter on the available GPU");
        assert_ne!(
            foreign_adapter.token.instance(),
            instance.native.wgpu_instance()
        );
        assert!(matches!(
            instance.create_device(&foreign_adapter, &rhi::DeviceDescriptor::default()),
            Err(rhi::DeviceError::Request(message)) if message.contains("different native instance")
        ));
        let after = assert_runtime_tuple(&runtime);
        assert_eq!(after.context().device(), original.context().device());
        assert!(foreign.resident_device().is_none());
    }

    #[test]
    fn failed_native_creation_keeps_the_previous_snapshot() {
        let Some(runtime) = runtime() else { return };
        let original = assert_runtime_tuple(&runtime);
        let mut descriptor = rhi::DeviceDescriptor::default();
        descriptor.required_limits.max_texture_dimension_2d = u32::MAX;
        assert!(runtime
            .instance()
            .create_device(runtime.adapter(), &descriptor)
            .is_err());
        let after = assert_runtime_tuple(&runtime);
        assert_eq!(after.context().device(), original.context().device());
        assert_eq!(after.context().queue(), original.context().queue());
    }

    #[test]
    fn runtime_recovery_publishes_a_new_tuple_without_mutating_old_snapshots() {
        let Some(mut runtime) = runtime() else { return };
        let old = assert_runtime_tuple(&runtime);
        let report = runtime
            .recover_device()
            .expect("headless resident recovery");
        assert_eq!(report.generation, 1);
        assert!(!report.surface_recreated);
        let current = assert_runtime_tuple(&runtime);
        assert_eq!(old.context().instance(), current.context().instance());
        assert_ne!(old.context().device(), current.context().device());
        assert_ne!(old.context().queue(), current.context().queue());
        assert_eq!(old.context().device(), old.device().wgpu_device());
        assert_eq!(old.context().queue(), old.queue().wgpu_queue());
        old.context().wait();
        current.context().wait();
    }

    #[test]
    fn failed_replay_does_not_make_the_new_publication_a_runtime_commit() {
        use prism_render_runtime::{RecoveryError, ReplayError};
        use std::sync::atomic::{AtomicBool, Ordering};

        let Some(mut runtime) = runtime() else { return };
        let old = assert_runtime_tuple(&runtime);
        let fail = Arc::new(AtomicBool::new(false));
        let recipe_fail = fail.clone();
        let replay = runtime
            .register_replay("resident publication boundary", move |_, _| {
                if recipe_fail.load(Ordering::SeqCst) {
                    Err(ReplayError::new(
                        "test replay failure after device creation",
                    ))
                } else {
                    Ok(())
                }
            })
            .expect("initial resource registration");
        fail.store(true, Ordering::SeqCst);
        assert!(matches!(
            runtime.recover_device(),
            Err(RecoveryError::Replay(_))
        ));
        assert_eq!(runtime.device_generation(), 0);
        assert_eq!(runtime.device().wgpu_device(), old.context().device());
        assert_eq!(runtime.queue().wgpu_queue(), old.context().queue());
        let staged = runtime.instance().resident_device().unwrap();
        assert_ne!(staged.context().device(), runtime.device().wgpu_device());
        assert!(ResidentRenderDevice::from_runtime(&runtime).is_none());
        // The failed runtime keeps the replay handle and old native tuple.
        // A subsequent successful recovery must publish/commit a matching pair.
        fail.store(false, Ordering::SeqCst);
        assert_eq!(runtime.recover_device().unwrap().generation, 1);
        let committed = assert_runtime_tuple(&runtime);
        assert_ne!(staged.context().device(), committed.context().device());
        drop(replay);
    }
}
