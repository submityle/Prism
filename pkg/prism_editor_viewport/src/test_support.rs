//! Explicit GPU-test capability diagnostics; required runs may never skip.

use futures_lite::future::block_on;

#[expect(
    clippy::print_stderr,
    reason = "GPU tests must distinguish actual hardware exercise from an explicit skip"
)]
pub(crate) fn diagnostic(message: core::fmt::Arguments<'_>) {
    eprintln!("{message}");
}

pub(crate) fn device() -> Option<(wgpu::Adapter, wgpu::Device, wgpu::Queue)> {
    let required = std::env::var("PRISM_REQUIRE_GPU").as_deref() == Ok("1");
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::METAL | wgpu::Backends::VULKAN | wgpu::Backends::DX12,
        flags: wgpu::InstanceFlags::default(),
        memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
        display: None,
        backend_options: wgpu::BackendOptions::default(),
    });
    let adapter = match block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
    {
        Ok(adapter) => adapter,
        Err(error) => {
            assert!(
                !required,
                "PRISM_REQUIRE_GPU=1: GPU adapter unavailable: {error}"
            );
            diagnostic(format_args!(
                "SKIP viewport GPU test: adapter unavailable: {error}"
            ));
            return None;
        }
    };
    let adapter_info = adapter.get_info();
    diagnostic(format_args!("viewport GPU adapter: {adapter_info:?}"));
    let (device, queue) = match block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_limits: adapter.limits(),
        ..Default::default()
    })) {
        Ok(pair) => pair,
        Err(error) => {
            assert!(
                !required,
                "PRISM_REQUIRE_GPU=1: GPU device unavailable on {adapter_info:?}: {error}"
            );
            diagnostic(format_args!(
                "SKIP viewport GPU test: device unavailable on {adapter_info:?}: {error}"
            ));
            return None;
        }
    };
    Some((adapter, device, queue))
}
