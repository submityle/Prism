//! Maps wgpu's device-lost reason onto the RHI's backend-agnostic one.
//!
//! [`DeviceLostReason`](prism_render_driver::DeviceLostReason) itself lives in
//! the RHI crate, next to the
//! [`RenderDevice::take_device_lost`](prism_render_driver::RenderDevice::take_device_lost)
//! contract every backend serves; this module only translates wgpu's copy, which
//! is the one thing that depends on a specific `wgpu` version.

use prism_render_driver::DeviceLostReason;

/// Converts wgpu's reason into the RHI's backend-agnostic copy.
pub(crate) fn map_device_lost(reason: wgpu::DeviceLostReason) -> DeviceLostReason {
    match reason {
        wgpu::DeviceLostReason::Unknown => DeviceLostReason::Unknown,
        wgpu::DeviceLostReason::Destroyed => DeviceLostReason::Destroyed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_every_wgpu_reason() {
        assert_eq!(
            map_device_lost(wgpu::DeviceLostReason::Unknown),
            DeviceLostReason::Unknown
        );
        assert_eq!(
            map_device_lost(wgpu::DeviceLostReason::Destroyed),
            DeviceLostReason::Destroyed
        );
    }
}
