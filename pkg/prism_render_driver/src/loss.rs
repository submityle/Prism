//! Device-loss reporting shared by every RHI backend.
//!
//! A backend signals a lost device (a driver reset or crash, or an explicit
//! device destroy) through
//! [`RenderDevice::take_device_lost`](crate::RenderDevice::take_device_lost), so
//! engine layers above the RHI can classify the loss without naming a graphics
//! API. This is the backend-agnostic reason returned there.

/// Why a [`RenderDevice`](crate::RenderDevice) was lost.
///
/// Returned by
/// [`RenderDevice::take_device_lost`](crate::RenderDevice::take_device_lost).
/// Once a device is lost its resources are unusable; a caller must stop
/// submitting work and rebuild the device rather than keep using it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DeviceLostReason {
    /// The device was lost for an unspecified reason, including driver resets
    /// and crashes.
    Unknown,
    /// The device's `destroy` method was called.
    Destroyed,
}
