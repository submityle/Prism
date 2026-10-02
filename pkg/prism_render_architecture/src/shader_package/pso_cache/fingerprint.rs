//! Device/driver fingerprint that gates reuse of a persisted PSO cache.
//!
//! A pipeline-state-object (PSO) cache compiled on one device is not portable:
//! a different GPU, vendor driver, backend, or render `ABI` can produce
//! incompatible pipeline binaries. Before a persisted cache is trusted on
//! startup, its stored [`DeviceFingerprint`] must match the current device's
//! fingerprint exactly; otherwise the cache is discarded and rebuilt. This
//! mirrors how `VkPipelineCache` / `ID3D12PipelineLibrary` headers embed device
//! identity so stale caches are rejected rather than silently misused.
//!
//! The comparison is deterministic and field-by-field so a mismatch yields an
//! actionable [`FingerprintMismatch`] reason for diagnostics and golden tests.

use crate::abi::AbiHash;

/// Graphics backend family a pipeline cache was compiled against.
///
/// Pipeline binaries never transfer across backends, so the backend is part of
/// the fingerprint. [`GraphicsBackend::Other`] carries an opaque discriminant
/// for backends not enumerated here, keeping the type forward-compatible.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum GraphicsBackend {
    /// Vulkan.
    Vulkan,
    /// Apple Metal.
    Metal,
    /// Direct3D 12.
    Dx12,
    /// WebGPU / `wgpu` default.
    WebGpu,
    /// Any other backend, identified by an opaque discriminant.
    Other(u32),
}

/// Immutable identity of the device + driver + `ABI` a PSO cache targets.
///
/// Two caches are interchangeable only when every field is identical. The
/// `ABI` hash is included because a changed render `ABI` invalidates the shader
/// inputs the pipelines were compiled against, independent of the hardware.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct DeviceFingerprint {
    /// Backend family the pipelines were compiled for.
    pub backend: GraphicsBackend,
    /// PCI (or platform) vendor identifier.
    pub vendor_id: u32,
    /// Device identifier within the vendor.
    pub device_id: u32,
    /// Opaque, monotonic driver version token reported by the backend.
    pub driver_version: u64,
    /// Render `ABI` hash the pipelines were compiled against.
    pub abi_hash: AbiHash,
}

/// First field in which two [`DeviceFingerprint`]s disagree.
///
/// Fields are checked in a fixed order (backend, vendor, device, driver, `ABI`)
/// so the reported reason is deterministic.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum FingerprintMismatch {
    /// The backend family differs.
    Backend,
    /// The vendor identifier differs.
    Vendor,
    /// The device identifier differs.
    Device,
    /// The driver version token differs.
    Driver,
    /// The render `ABI` hash differs.
    Abi,
}

impl DeviceFingerprint {
    /// Builds a fingerprint from its components.
    #[must_use]
    pub const fn new(
        backend: GraphicsBackend,
        vendor_id: u32,
        device_id: u32,
        driver_version: u64,
        abi_hash: AbiHash,
    ) -> Self {
        Self {
            backend,
            vendor_id,
            device_id,
            driver_version,
            abi_hash,
        }
    }

    /// Reports whether a persisted cache tagged with `self` may be reused on a
    /// device whose fingerprint is `current`.
    #[must_use]
    pub fn is_compatible_with(&self, current: &Self) -> bool {
        self.check_against(current).is_ok()
    }

    /// Compares `self` (the persisted tag) against the `current` device.
    ///
    /// # Errors
    ///
    /// Returns the first [`FingerprintMismatch`] in checked order when the
    /// cache must be discarded, or `Ok(())` when it is safe to reuse.
    pub fn check_against(&self, current: &Self) -> Result<(), FingerprintMismatch> {
        if self.backend != current.backend {
            return Err(FingerprintMismatch::Backend);
        }
        if self.vendor_id != current.vendor_id {
            return Err(FingerprintMismatch::Vendor);
        }
        if self.device_id != current.device_id {
            return Err(FingerprintMismatch::Device);
        }
        if self.driver_version != current.driver_version {
            return Err(FingerprintMismatch::Driver);
        }
        if self.abi_hash != current.abi_hash {
            return Err(FingerprintMismatch::Abi);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fp() -> DeviceFingerprint {
        DeviceFingerprint::new(GraphicsBackend::Vulkan, 0x10DE, 0x2204, 42, AbiHash([7u8; 32]))
    }

    #[test]
    fn identical_fingerprints_are_compatible() {
        assert!(fp().is_compatible_with(&fp()));
        assert_eq!(fp().check_against(&fp()), Ok(()));
    }

    #[test]
    fn backend_change_is_rejected_first() {
        let mut other = fp();
        other.backend = GraphicsBackend::Metal;
        other.vendor_id = 0; // also differs, but backend is checked first
        assert_eq!(fp().check_against(&other), Err(FingerprintMismatch::Backend));
        assert!(!fp().is_compatible_with(&other));
    }

    #[test]
    fn each_field_reports_its_own_reason() {
        let base = fp();

        let mut v = base.clone();
        v.vendor_id = 0x8086;
        assert_eq!(base.check_against(&v), Err(FingerprintMismatch::Vendor));

        let mut d = base.clone();
        d.device_id = 0x1;
        assert_eq!(base.check_against(&d), Err(FingerprintMismatch::Device));

        let mut dr = base.clone();
        dr.driver_version = 43;
        assert_eq!(base.check_against(&dr), Err(FingerprintMismatch::Driver));

        let mut a = base.clone();
        a.abi_hash = AbiHash([9u8; 32]);
        assert_eq!(base.check_against(&a), Err(FingerprintMismatch::Abi));
    }

    #[test]
    fn other_backend_discriminant_distinguishes() {
        let mut a = fp();
        a.backend = GraphicsBackend::Other(1);
        let mut b = fp();
        b.backend = GraphicsBackend::Other(2);
        assert_eq!(a.check_against(&b), Err(FingerprintMismatch::Backend));
    }
}
