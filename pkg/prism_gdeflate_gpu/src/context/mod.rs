//! `GPU` device acquisition and submission helpers.
//!
//! [`GpuContext`] wraps the four long-lived `wgpu` handles (`Instance`,
//! `Adapter`, `Device`, `Queue`) that every kernel dispatch needs. Acquisition
//! is deliberately best-effort: [`GpuContext::try_headless`] returns [`None`]
//! on a host with no usable adapter so a test suite can skip gracefully.
//! Setting `PRISM_REQUIRE_GPU=1` turns that absence into a test failure.
//!
//! For renderer sharing, acquire the original tuple with
//! [`GpuContext::request_shared`], or use [`GpuAdapter::request`] followed by
//! [`GpuAdapter::request_device`] to inspect adapter capabilities first. Give
//! the renderer clones of the context's handles and retain or clone the context
//! for compute. There is deliberately no raw handles constructor: wgpu 30's
//! safe public API cannot authenticate an
//! arbitrary instance/adapter/device/queue tuple. Adapter information and
//! instance-local resource IDs are not proof of ownership.
//!
//! Provenance: standard `wgpu` initialisation; no Unreal Engine source or
//! derived code.

use alloc::string::String;
use core::fmt;

use wgpu::{
    Adapter, BackendOptions, Backends, Device, DeviceDescriptor, Instance, InstanceDescriptor,
    InstanceFlags, PollType, Queue, RequestAdapterOptions,
};

/// Blocks the current thread until `future` resolves.
///
/// A thin re-export of [`futures_lite::future::block_on`] so callers do not
/// need a direct dependency on an async executor just to drive the handful of
/// `wgpu` setup futures.
pub fn block_on<F: Future>(future: F) -> F::Output {
    futures_lite::future::block_on(future)
}

/// Why a real compute device could not be acquired.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GpuProbeError {
    /// No adapter was exposed by the enabled native backends.
    NoAdapter(String),
    /// An adapter was found, but its logical device could not be created.
    Device(String),
}

impl fmt::Display for GpuProbeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoAdapter(message) => write!(formatter, "no usable GPU adapter: {message}"),
            Self::Device(message) => write!(formatter, "GPU device creation failed: {message}"),
        }
    }
}

impl core::error::Error for GpuProbeError {}

/// An adapter and its original instance, acquired together by construction.
///
/// Inspect [`Self::adapter`] to choose device features and limits before calling
/// [`Self::request_device`]. Cloning preserves the same adapter and instance;
/// it neither selects another adapter nor creates a logical device.
/// There is no constructor accepting unrelated raw handles or metadata.
#[derive(Clone)]
pub struct GpuAdapter {
    instance: Instance,
    adapter: Adapter,
}

impl GpuAdapter {
    /// Enumerates all adapters selected by `backends` on the original instance.
    ///
    /// Each token retains the instance that directly produced its adapter.
    /// This blocks during native enumeration, without requesting any devices
    /// or reconstructing ancestry from adapter metadata.
    #[must_use]
    pub fn enumerate(instance: &Instance, backends: Backends) -> Vec<Self> {
        block_on(instance.enumerate_adapters(backends))
            .into_iter()
            .map(|adapter| Self {
                instance: instance.clone(),
                adapter,
            })
            .collect()
    }

    /// Requests an adapter from `instance` and retains both original handles.
    ///
    /// `options` is forwarded unchanged, including any compatible surface.
    /// This blocks during adapter acquisition but does not request a device.
    pub fn request(
        instance: &Instance,
        options: &RequestAdapterOptions<'_, '_>,
    ) -> Result<Self, GpuProbeError> {
        let adapter = block_on(instance.request_adapter(options))
            .map_err(|error| GpuProbeError::NoAdapter(error.to_string()))?;
        Ok(Self {
            instance: instance.clone(),
            adapter,
        })
    }

    /// Returns the original instance from which this adapter was requested.
    #[must_use]
    pub fn instance(&self) -> &Instance {
        &self.instance
    }

    /// Returns the original adapter for capability inspection and renderer use.
    #[must_use]
    pub fn adapter(&self) -> &Adapter {
        &self.adapter
    }

    /// Creates a device and queue directly on this retained adapter.
    ///
    /// `descriptor` is forwarded unchanged, without upgrading features or limits.
    /// No adapter selection is repeated. A failed request leaves this wrapper
    /// unchanged so the caller can retry with another descriptor.
    /// Each successful call creates a distinct logical device; cloning this
    /// wrapper does not make those devices or their resources interchangeable.
    /// Multiple requests follow the native backend's support and restrictions.
    ///
    /// This blocks during device acquisition. It does not install error/loss
    /// callbacks, submit commands, or wait for GPU work. Device ancestry follows
    /// from acquisition, not from metadata or instance-local device IDs.
    pub fn request_device(
        &self,
        descriptor: &DeviceDescriptor<'_>,
    ) -> Result<GpuContext, GpuProbeError> {
        let (device, queue) = block_on(self.adapter.request_device(descriptor))
            .map_err(|error| GpuProbeError::Device(error.to_string()))?;
        let adapter_info = device.adapter_info();
        Ok(GpuContext {
            instance: self.instance.clone(),
            adapter: self.adapter.clone(),
            adapter_info,
            device,
            queue,
        })
    }
}

/// The long-lived `wgpu` handles shared by every compute kernel in this crate.
/// Cloning retains the same instance, logical device, and submission queue.
#[derive(Clone)]
pub struct GpuContext {
    /// The `wgpu` instance the adapter was requested from.
    instance: Instance,
    /// The physical adapter backing [`GpuContext::device`].
    adapter: Adapter,
    /// Snapshot of the physical adapter identity used for diagnostics.
    adapter_info: wgpu::AdapterInfo,
    /// The logical device used to allocate resources and pipelines.
    device: Device,
    /// The queue used to submit command buffers.
    queue: Queue,
}

impl GpuContext {
    /// Requests the original device/queue tuple for both renderer and kernels.
    ///
    /// The adapter is requested from `instance` here, and its device and queue
    /// come from one native `request_device` call. This establishes ancestry
    /// by construction, not by inspecting caller-supplied adapter metadata.
    /// `options` can select the renderer's compatible surface; `descriptor`
    /// controls the enabled features and limits, without silently upgrading them.
    /// To choose that descriptor after inspecting capabilities, use
    /// [`GpuAdapter::request`] followed by [`GpuAdapter::request_device`].
    /// Give the renderer clones from [`Self::adapter`], [`Self::device`] and
    /// [`Self::queue`], and retain this context (or a clone) for compute.
    /// No extra logical device is needed for the compute kernels.
    ///
    /// # Ownership boundary
    ///
    /// The supplied instance is retained; the adapter, device and queue are
    /// acquired here rather than accepted as unrelated raw handles. Cloning
    /// this context preserves that tuple. This does not import or authenticate
    /// an existing renderer device: the renderer must adopt these handles.
    /// [`Self::adapter_info`] is diagnostic metadata, not an ownership token.
    /// Kernel/resource checks must compare both the retained instance and
    /// device because device IDs can alias across independent instances.
    ///
    /// This blocks during acquisition, just like [`Self::probe`]. It does not
    /// install error/loss callbacks, submit commands, or wait for GPU work.
    pub fn request_shared(
        instance: &Instance,
        options: &RequestAdapterOptions<'_, '_>,
        descriptor: &DeviceDescriptor<'_>,
    ) -> Result<Self, GpuProbeError> {
        GpuAdapter::request(instance, options)?.request_device(descriptor)
    }

    /// Attempts to acquire a headless compute device across the native backends.
    ///
    /// Returns [`None`] when no adapter or device can be obtained so callers can
    /// skip `GPU` work instead of panicking on a host without a usable adapter.
    #[must_use]
    pub fn try_headless() -> Option<GpuContext> {
        match Self::probe() {
            Ok(context) => Some(context),
            Err(error) => {
                if std::env::var_os("PRISM_REQUIRE_GPU").is_some() {
                    panic!("PRISM_REQUIRE_GPU is set but no usable GPU was found: {error}");
                }
                None
            }
        }
    }

    /// Acquires a headless compute device and preserves the concrete failure
    /// reason when the host cannot provide one.
    pub fn probe() -> Result<GpuContext, GpuProbeError> {
        let instance = Instance::new(InstanceDescriptor {
            // The crate enables Metal, Vulkan, and DX12 in its manifest. Keep
            // this list in one place so diagnostics describe the actual probe.
            backends: Backends::METAL | Backends::VULKAN | Backends::DX12,
            flags: InstanceFlags::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            display: None,
            backend_options: BackendOptions::default(),
        });
        let adapter = GpuAdapter::request(&instance, &RequestAdapterOptions::default())?;
        adapter.request_device(&DeviceDescriptor {
            required_limits: adapter.adapter().limits(),
            ..Default::default()
        })
    }

    /// Returns the logical device.
    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Returns the original instance, whose ancestry is known by construction.
    /// Compare it alongside `device()` when checking kernel/resource ownership:
    /// native wgpu device IDs alone can alias across independent instances.
    #[must_use]
    pub fn instance(&self) -> &Instance {
        &self.instance
    }

    /// Returns the original adapter for renderer adoption, not just its metadata.
    #[must_use]
    pub fn adapter(&self) -> &Adapter {
        &self.adapter
    }

    /// Returns the submission queue.
    #[must_use]
    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    /// Returns the adapter identity and backend selected by `wgpu`.
    #[must_use]
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.adapter_info
    }

    /// Blocks until all previously submitted work on this device completes.
    ///
    /// # Panics
    ///
    /// Panics if the device is lost while waiting, which indicates a broken
    /// adapter and cannot be recovered from within a single dispatch.
    pub fn wait(&self) {
        self.device
            .poll(PollType::wait_indefinitely())
            .expect("device poll should complete the submitted work");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DeflateBatchError, DeflateBatchLimits, DeflateDecodeError, DeflateTile, GpuDeflate,
        GpuFixedGDeflate, GROUP,
    };

    #[test]
    fn two_stage_acquisition_preserves_original_adapter_and_clone_lifetime() {
        let Some(probe) = GpuContext::try_headless() else {
            return;
        };
        // The convenience probe must continue to request the full adapter limits.
        assert_eq!(
            probe.device().limits().max_buffer_size,
            probe.adapter().limits().max_buffer_size,
        );
        assert_eq!(
            probe.device().limits().max_storage_buffer_binding_size,
            probe.adapter().limits().max_storage_buffer_binding_size,
        );
        let instance = probe.instance().clone();
        let acquired = GpuAdapter::request(&instance, &RequestAdapterOptions::default())
            .expect("two-stage adapter acquisition on an available GPU");
        let original_adapter = acquired.adapter().clone();
        let cloned = acquired.clone();
        assert_eq!(acquired.instance(), &instance);
        assert_eq!(cloned.instance(), &instance);
        assert_eq!(cloned.adapter(), &original_adapter);
        drop(acquired);
        drop(probe);

        // Descriptor selection happens after inspecting the retained adapter.
        let descriptor = DeviceDescriptor {
            required_limits: cloned.adapter().limits(),
            ..Default::default()
        };
        let ctx = cloned
            .request_device(&descriptor)
            .expect("device acquisition through an adapter clone");
        assert_eq!(ctx.instance(), &instance);
        assert_eq!(ctx.adapter(), &original_adapter);
        assert_eq!(ctx.device().features(), descriptor.required_features);
        assert_eq!(
            ctx.device().limits().max_buffer_size,
            descriptor.required_limits.max_buffer_size,
        );
        let shared = ctx.clone();
        let fixed = GpuFixedGDeflate::new(&ctx);
        let full = GpuDeflate::new(&ctx);
        drop(ctx);
        drop(cloned);
        drop(original_adapter);
        drop(instance);

        let mut source = [0u8; GROUP];
        source[..3].copy_from_slice(&[0x73, 0x04, 0x00]); // fixed block: "A"
        let scope = shared
            .device()
            .push_error_scope(wgpu::ErrorFilter::Validation);
        assert_eq!(fixed.decompress(&shared, &source, 3, 1).unwrap(), b"A");
        assert_eq!(full.decompress(&shared, &source, 3, 1).unwrap(), b"A");
        assert!(block_on(scope.pop()).is_none());
    }

    #[test]
    fn unsupported_features_do_not_poison_two_stage_device_retry() {
        let Some(probe) = GpuContext::try_headless() else {
            return;
        };
        let acquired = GpuAdapter::request(probe.instance(), &RequestAdapterOptions::default())
            .expect("two-stage adapter acquisition on an available GPU");
        drop(probe);
        let cloned = acquired.clone();
        let supported = acquired.adapter().features();
        let limits = acquired.adapter().limits();
        let unsupported = wgpu::Features::all() - supported;
        if unsupported.is_empty() {
            eprintln!("skipping unsupported-feature retry: adapter supports every known feature");
            return;
        }
        let error = acquired
            .request_device(&DeviceDescriptor {
                required_features: unsupported,
                ..Default::default()
            })
            .err()
            .expect("unsupported device features must fail");
        assert!(matches!(error, GpuProbeError::Device(message) if !message.is_empty()));
        assert_eq!(acquired.instance(), cloned.instance());
        assert_eq!(acquired.adapter(), cloned.adapter());
        assert_eq!(acquired.adapter().features(), supported);
        assert_eq!(acquired.adapter().limits(), limits);

        let descriptor = DeviceDescriptor::default();
        let ctx = acquired
            .request_device(&descriptor)
            .expect("valid retry on the same adapter after unsupported features");
        assert_eq!(ctx.instance(), acquired.instance());
        assert_eq!(ctx.adapter(), acquired.adapter());
        assert_eq!(ctx.device().features(), descriptor.required_features);
        assert_eq!(
            ctx.device().limits().max_buffer_size,
            descriptor.required_limits.max_buffer_size,
        );
        drop(acquired);
        drop(cloned);
        let mut source = [0u8; GROUP];
        source[..3].copy_from_slice(&[0x73, 0x04, 0x00]);
        let scope = ctx.device().push_error_scope(wgpu::ErrorFilter::Validation);
        assert_eq!(
            GpuDeflate::new(&ctx)
                .decompress(&ctx, &source, 3, 1)
                .unwrap(),
            b"A"
        );
        assert!(block_on(scope.pop()).is_none());
    }

    #[test]
    fn request_shared_preserves_handles_and_clone_decodes_on_the_same_device() {
        let Some(probe) = GpuContext::try_headless() else {
            return;
        };
        let instance = probe.instance().clone();
        let descriptor = DeviceDescriptor {
            label: Some("prism_gdeflate_shared_context_test"),
            ..Default::default()
        };
        let ctx =
            GpuContext::request_shared(&instance, &RequestAdapterOptions::default(), &descriptor)
                .expect("shared context on an available GPU");
        assert_eq!(ctx.instance(), &instance);
        assert_eq!(ctx.device().features(), descriptor.required_features);
        assert_eq!(
            ctx.device().limits().max_buffer_size,
            descriptor.required_limits.max_buffer_size,
        );
        assert_eq!(ctx.adapter_info().name, ctx.device().adapter_info().name);

        // These are the handles the renderer adopts, not a second device request.
        let renderer_instance = ctx.instance().clone();
        let renderer_adapter = ctx.adapter().clone();
        let renderer_device = ctx.device().clone();
        let renderer_queue = ctx.queue().clone();
        let shared = ctx.clone();
        assert_eq!(shared.instance(), &renderer_instance);
        assert_eq!(shared.adapter(), &renderer_adapter);
        assert_eq!(shared.device(), &renderer_device);
        assert_eq!(shared.queue(), &renderer_queue);

        let fixed = GpuFixedGDeflate::new(&ctx);
        let full = GpuDeflate::new(&ctx);
        drop(ctx);
        drop(instance);
        drop(probe);
        // Cloning retains the tuple even after its original wrappers are dropped.
        let mut source = [0u8; GROUP];
        source[..3].copy_from_slice(&[0x73, 0x04, 0x00]); // fixed block: "A"
        let requests = [DeflateTile {
            interleaved: &source,
            compressed_size: 3,
            expected_output_size: 1,
        }];
        let limits = DeflateBatchLimits {
            max_tiles: 1,
            max_device_bytes: 1024,
        };
        let scope = renderer_device.push_error_scope(wgpu::ErrorFilter::Validation);
        let fixed_tiles = fixed
            .decompress_batch_resident(&shared, &requests, limits)
            .expect("fixed resident decode with the shared device");
        let full_tiles = full
            .decompress_batch_resident(&shared, &requests, limits)
            .expect("full resident decode with the shared device");
        assert_eq!(fixed_tiles.len(), 1);
        assert_eq!(full_tiles.len(), 1);
        assert_eq!(fixed_tiles[0].byte_len(), 1);
        assert_eq!(full_tiles[0].byte_len(), 1);
        assert_eq!(fixed.decompress(&shared, &source, 3, 1).unwrap(), b"A");
        assert_eq!(full.decompress(&shared, &source, 3, 1).unwrap(), b"A");
        assert!(block_on(scope.pop()).is_none());
    }

    #[test]
    fn decode_device_guard_rejects_another_device_in_the_same_instance() {
        let Some(probe) = GpuContext::try_headless() else {
            return;
        };
        let acquired = GpuAdapter::request(probe.instance(), &RequestAdapterOptions::default())
            .expect("two-stage adapter acquisition on an available GPU");
        drop(probe);
        let cloned = acquired.clone();
        let descriptor = DeviceDescriptor {
            required_limits: acquired.adapter().limits(),
            ..Default::default()
        };
        let ctx = acquired
            .request_device(&descriptor)
            .expect("first logical device on the retained adapter");
        // Cloning the adapter retains it, but does not share a logical device.
        let other = cloned
            .request_device(&descriptor)
            .expect("second logical device on the retained adapter");
        assert_eq!(ctx.adapter(), acquired.adapter());
        assert_eq!(other.adapter(), acquired.adapter());
        assert_eq!(ctx.instance(), other.instance());
        assert_ne!(ctx.device(), other.device());
        assert_ne!(ctx.queue(), other.queue());
        drop(acquired);
        drop(cloned);
        assert_decode_device_mismatch(&ctx, &other);
        assert_decode_device_mismatch(&other, &ctx);
    }

    #[test]
    fn decode_device_guard_rejects_a_device_from_an_independent_instance() {
        let Some(ctx) = GpuContext::try_headless() else {
            return;
        };
        let independent =
            GpuContext::probe().expect("second independent instance on an available GPU");
        let acquired =
            GpuAdapter::request(independent.instance(), &RequestAdapterOptions::default())
                .expect("two-stage adapter acquisition on the independent instance");
        assert_ne!(ctx.instance(), acquired.instance());
        drop(independent);
        let other = acquired
            .request_device(&DeviceDescriptor {
                required_limits: acquired.adapter().limits(),
                ..Default::default()
            })
            .expect("device acquisition on the retained independent instance");
        assert_eq!(other.instance(), acquired.instance());
        assert_eq!(other.adapter(), acquired.adapter());
        drop(acquired);
        assert_ne!(ctx.instance(), other.instance());
        // Device/queue IDs may alias across instances, but their allocation
        // is backend-dependent. Require rejection whether or not they alias.
        assert_decode_device_mismatch(&ctx, &other);
    }

    fn assert_decode_device_mismatch(ctx: &GpuContext, other: &GpuContext) {
        let mut source = [0u8; GROUP];
        source[0] = 3; // final fixed block, EOB only: raw bytes [3, 0]
        let requests = [DeflateTile {
            interleaved: &source,
            compressed_size: 2,
            expected_output_size: 0,
        }];
        let limits = DeflateBatchLimits {
            max_tiles: 1,
            max_device_bytes: 1024,
        };
        let fixed = GpuFixedGDeflate::new(ctx);
        let full = GpuDeflate::new(ctx);
        let owner_scope = ctx.device().push_error_scope(wgpu::ErrorFilter::Validation);
        let scope = other
            .device()
            .push_error_scope(wgpu::ErrorFilter::Validation);
        assert_eq!(
            fixed
                .decompress_batch_resident(other, &requests, limits)
                .unwrap_err(),
            DeflateBatchError::Device(DeflateDecodeError::DeviceMismatch),
        );
        assert_eq!(
            full.decompress_batch_resident(other, &requests, limits)
                .unwrap_err(),
            DeflateBatchError::Device(DeflateDecodeError::DeviceMismatch),
        );
        assert_eq!(
            fixed.decompress(other, &source, 2, 0),
            Err(DeflateDecodeError::DeviceMismatch)
        );
        assert_eq!(
            full.decompress(other, &source, 2, 0),
            Err(DeflateDecodeError::DeviceMismatch)
        );
        assert!(block_on(scope.pop()).is_none());
        assert!(block_on(owner_scope.pop()).is_none());
        // The same handles cloned into another wrapper retain valid ownership.
        let cloned = ctx.clone();
        assert!(fixed.decompress(&cloned, &source, 2, 0).unwrap().is_empty());
        assert!(full.decompress(&cloned, &source, 2, 0).unwrap().is_empty());
    }
}
