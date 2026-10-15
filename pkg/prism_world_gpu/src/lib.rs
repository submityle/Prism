//! Deterministic world-proxy visibility contract and optional real-device twin.
//!
//! The CPU side is deliberately independent of an engine, ECS, renderer, or
//! scene implementation. It owns the data layout and the exact policy used by
//! the GPU kernel: sphere/frustum rejection, range rejection, screen-space
//! geometric error, LOD selection, and fixed-point error quantisation.
//!
//! The `gpu` feature adds a native `wgpu` compute implementation. Its output is
//! read back into the same four-word record produced by [`VisibilityOutput`],
//! so parity is checked at the contract boundary rather than by comparing
//! implementation-specific objects.

#![forbid(unsafe_code)]

mod contract;
mod packing;

pub use contract::{
    Cluster, FrustumPlane, ProxyInstance, VisibilityInput, VisibilityOutput, ERROR_QUANTUM,
    LOD_THRESHOLDS,
};
pub use packing::{ClusterSoa, ProxySoa, SoaBufferPack};

#[cfg(feature = "gpu")]
pub mod gpu;

#[cfg(feature = "gpu")]
pub use gpu::{AdapterEvidence, GpuAcquireError, GpuContext, GpuVisibility};

#[cfg(test)]
mod tests {
    use super::*;

    fn input() -> VisibilityInput {
        VisibilityInput::new(
            [0.0, 0.0, 0.0],
            [
                FrustumPlane::new([1.0, 0.0, 0.0], 10.0),
                FrustumPlane::new([-1.0, 0.0, 0.0], 10.0),
                FrustumPlane::new([0.0, 1.0, 0.0], 10.0),
                FrustumPlane::new([0.0, -1.0, 0.0], 10.0),
                FrustumPlane::new([0.0, 0.0, 1.0], 10.0),
                FrustumPlane::new([0.0, 0.0, -1.0], 10.0),
            ],
            1080,
            100.0,
        )
    }

    #[test]
    fn visible_and_rejected_proxies_have_stable_words() {
        let camera = input();
        let visible = ProxyInstance::new(7, 3, [0.0, 0.0, 2.0], 0.5, 1.0, 100.0);
        let outside = ProxyInstance::new(8, 3, [20.0, 0.0, 2.0], 0.5, 1.0, 100.0);

        let a = camera.evaluate(&visible);
        let b = camera.evaluate(&outside);

        assert!(a.visible);
        assert_eq!(a.proxy_id, 7);
        assert_eq!(a.words()[0], 1);
        assert!(!b.visible);
        assert_eq!(b.words(), [0, 0, 0, 8]);
    }

    #[test]
    fn lod_thresholds_and_error_quantisation_are_deterministic() {
        let camera = VisibilityInput::new(
            [0.0, 0.0, 0.0],
            [
                FrustumPlane::new([1.0, 0.0, 0.0], 1_000.0),
                FrustumPlane::new([-1.0, 0.0, 0.0], 1_000.0),
                FrustumPlane::new([0.0, 1.0, 0.0], 1_000.0),
                FrustumPlane::new([0.0, -1.0, 0.0], 1_000.0),
                FrustumPlane::new([0.0, 0.0, 1.0], 1_000.0),
                FrustumPlane::new([0.0, 0.0, -1.0], 1_000.0),
            ],
            256,
            1.0,
        );
        let proxy = ProxyInstance::new(9, 0, [0.0, 0.0, 1.0], 0.1, 1.0, 100.0);
        let output = camera.evaluate(&proxy);
        assert_eq!(output.lod, 0);
        assert_eq!(output.error_quantized, 32768);
        assert_eq!(output.words()[2], 32768);
        assert_eq!(LOD_THRESHOLDS, [64.0, 16.0, 4.0]);
    }

    #[test]
    fn soa_packing_preserves_order_and_uses_little_endian_words() {
        let proxies = [
            ProxyInstance::new(11, 2, [1.0, 2.0, 3.0], 4.0, 5.0, 6.0),
            ProxyInstance::new(12, 2, [-1.0, 0.0, 0.5], 0.25, 7.0, 8.0),
        ];
        let soa = ProxySoa::from_proxies(&proxies).expect("valid proxy array");
        let packed = soa.pack_le();

        assert_eq!(soa.len(), 2);
        assert_eq!(&packed.metadata[0..4], &11_u32.to_le_bytes());
        assert_eq!(&packed.metadata[16..20], &12_u32.to_le_bytes());
        assert_eq!(&packed.positions[0..4], &1.0_f32.to_bits().to_le_bytes());
        assert_eq!(&packed.errors[0..4], &5.0_f32.to_bits().to_le_bytes());
    }

    #[test]
    fn invalid_non_finite_proxy_is_rejected_before_upload() {
        let proxy = ProxyInstance::new(1, 0, [f32::NAN, 0.0, 0.0], 1.0, 1.0, 1.0);
        let error = ProxySoa::from_proxies(&[proxy]).expect_err("NaN is not a GPU contract value");
        assert!(error.contains("finite"));
    }
}
