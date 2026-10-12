//! Device-loss reporting surfaced by [`crate::WgpuDevice`].
//!
//! wgpu signals a lost device (a driver reset or crash, or an explicit
//! `wgpu::Device::destroy`) through a callback rather than an error return, so
//! the backend records the reason and lets callers poll it. The public reason
//! is a backend-agnostic copy of wgpu's, so consumers of this crate classify a
//! loss without depending on a specific `wgpu` version.

/// Why a [`crate::WgpuDevice`] was lost.
///
/// Returned by [`crate::WgpuDevice::take_device_lost`]. Once a device is lost
/// its resources are unusable; a caller must rebuild the device rather than
/// keep submitting work.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DeviceLostReason {
    /// The device was lost for an unspecified reason, including driver resets
    /// and crashes.
    Unknown,
    /// The device's `destroy` method was called.
    Destroyed,
}

impl DeviceLostReason {
    /// Converts wgpu's reason into the backend-agnostic copy.
    pub(crate) fn from_wgpu(reason: wgpu::DeviceLostReason) -> Self {
        match reason {
            wgpu::DeviceLostReason::Unknown => Self::Unknown,
            wgpu::DeviceLostReason::Destroyed => Self::Destroyed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::DeviceLostReason;

    #[test]
    fn maps_every_wgpu_reason() {
        assert_eq!(
            DeviceLostReason::from_wgpu(wgpu::DeviceLostReason::Unknown),
            DeviceLostReason::Unknown
        );
        assert_eq!(
            DeviceLostReason::from_wgpu(wgpu::DeviceLostReason::Destroyed),
            DeviceLostReason::Destroyed
        );
    }
}
