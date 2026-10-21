//! Deterministic world-proxy visibility contract and optional real-device twin.
//!
//! The CPU side is deliberately independent of an engine, ECS, renderer, or
//! scene implementation. It owns the data layout and the exact policy used by
//! the GPU kernel: sphere/frustum rejection, range rejection, screen-space
//! geometric error, LOD selection, and fixed-point error quantisation.
//!
//! The `gpu` feature adds a native `wgpu` compute implementation. Its output is
//! read back as canonical four-word visibility plus the unquantized screen
//! error. Parity checks both representations at the contract boundary.

#![forbid(unsafe_code)]

mod contract;
mod density;
mod packing;
mod placement_render_bridge;
mod registry;
mod world_placement_graph;

pub use contract::{
    Cluster, FrustumPlane, ProxyInstance, VisibilityInput, VisibilityOutput, ERROR_QUANTUM,
    LOD_THRESHOLDS,
};
pub use density::{density_dither_visible, DensityPolicy, DensityPolicyError};
pub use packing::{ClusterSoa, ProxySoa, SoaBufferPack};
pub use placement_render_bridge::{
    build_placement_render_graph, PlacementComputeStage, PlacementDrawStage,
    PlacementImportedBuffer, PlacementRenderBridgeError, PlacementRenderStream,
    PlacementRenderWork, PlacementSceneTarget,
};
pub use registry::{PreparedSectorUpdate, ProxyHandle, ProxyRegistry, ProxyRegistryError};
pub use world_placement_graph::{
    PlacementCellManifest, PlacementGraphLayout, PlacementGraphPass, PlacementGraphResource,
    PlacementGraphStage, PlacementLifecycle, PlacementLifecycleError, PlacementRenderGraph,
    PlacementRhiUse, PlacementStream, PlacementStreamSnapshot,
};

/// Renderer-side ordered coverage test. Concatenate this WGSL with a draw
/// shader and call `world_density_survives(coverage, pixel, stable_phase)`.
/// Coverage is read from the resident dense source slot, not compact draw order.
pub const DENSITY_DITHER_WGSL: &str = include_str!("shaders/density_dither.wgsl");

#[cfg(feature = "gpu")]
pub mod gpu;

#[cfg(feature = "gpu")]
pub mod resident;

#[cfg(feature = "gpu")]
mod sdf;

#[cfg(feature = "gpu")]
mod resident_density;

#[cfg(feature = "gpu")]
mod sector_proxies;

#[cfg(feature = "gpu")]
pub use gpu::{AdapterEvidence, GpuAcquireError, GpuContext, GpuVisibility};

#[cfg(feature = "gpu")]
pub use resident::{GpuResidentError, GpuResidentVisibility};

#[cfg(feature = "gpu")]
pub use sdf::{
    GpuSdfError, GpuSdfEvaluation, GpuSdfLookup, GpuSdfQuery, GpuSdfQueryResult, GpuSdfTelemetry,
    GpuSdfToroidalLookup, GPU_SDF_ABI_VERSION, GPU_SDF_TOROIDAL_ABI_VERSION,
    GPU_SDF_TOROIDAL_VERSION, GPU_SDF_VERSION,
};

#[cfg(feature = "gpu")]
pub use sector_proxies::{GpuSectorProxies, GpuSectorProxyError, ProxyUploadStats};

#[cfg(feature = "gpu")]
mod scatter;

#[cfg(feature = "gpu")]
pub use scatter::{GpuScatter, GpuScatterError};

#[cfg(feature = "gpu")]
mod scatter_brush_index;
#[cfg(feature = "gpu")]
mod scatter_field;

#[cfg(feature = "gpu")]
pub use scatter_field::{GpuScatterField, GpuScatterFieldError};

#[cfg(feature = "gpu")]
mod scatter_deletion;

#[cfg(feature = "gpu")]
pub use scatter_deletion::{
    GpuScatterDeletionCacheKey, GpuScatterDeletionCapacity, GpuScatterDeletionError,
    GpuScatterDeletionUploadStats, GpuScatterDeletionUpstream, GpuScatterDeletionWorkBudget,
    GpuScatterDeletions, GpuScatterFieldRequest, GpuScatterSourceRequest,
    GPU_SCATTER_DELETION_KEY_ABI_VERSION, GPU_SCATTER_DELETION_VERSION,
};

#[cfg(feature = "terrain_scatter")]
pub use scatter_deletion::GpuScatterHeightfieldRequest;

#[cfg(feature = "terrain_scatter")]
mod scatter_heightfield;

#[cfg(feature = "terrain_scatter")]
pub use scatter_heightfield::{GpuScatterHeightfield, GpuScatterHeightfieldError};

#[cfg(feature = "terrain_scatter")]
mod heightfield_placement;

#[cfg(feature = "terrain_scatter")]
pub use heightfield_placement::{
    GpuHeightfieldPlacement, GpuHeightfieldPlacementActiveStagesSnapshot,
    GpuHeightfieldPlacementBufferSnapshot, GpuHeightfieldPlacementError,
    GpuHeightfieldPlacementRegistrationSnapshot, GpuHeightfieldPlacementStageSnapshot,
    GpuHeightfieldPlacementWorkBudget,
};

#[cfg(feature = "terrain_scatter")]
mod heightfield_placement_deletion;

#[cfg(feature = "terrain_scatter")]
mod planet_placement;

#[cfg(feature = "terrain_scatter")]
pub use planet_placement::{
    GpuPlanetPlacement, GpuPlanetPlacementCacheKey, GpuPlanetPlacementError,
    GpuPlanetPlacementWorkBudget,
};

#[cfg(feature = "terrain_scatter")]
mod planet_placement_deletion;

#[cfg(feature = "terrain_scatter")]
pub use planet_placement_deletion::{
    GpuPlanetPlacementDeletionCacheKey, GpuPlanetPlacementDeletionCapacity,
    GpuPlanetPlacementDeletionError, GpuPlanetPlacementDeletionUploadStats,
    GpuPlanetPlacementDeletionWorkBudget, GpuPlanetPlacementDeletions,
    GPU_PLANET_PLACEMENT_DELETION_KEY_ABI_VERSION, GPU_PLANET_PLACEMENT_DELETION_VERSION,
};

#[cfg(feature = "terrain_scatter")]
pub use heightfield_placement_deletion::{
    GpuHeightfieldPlacementDeletionCacheKey, GpuHeightfieldPlacementDeletionCapacity,
    GpuHeightfieldPlacementDeletionError, GpuHeightfieldPlacementDeletionUploadStats,
    GpuHeightfieldPlacementDeletionWorkBudget, GpuHeightfieldPlacementDeletions,
    GPU_HEIGHTFIELD_PLACEMENT_DELETION_KEY_ABI_VERSION, GPU_HEIGHTFIELD_PLACEMENT_DELETION_VERSION,
};

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

    #[test]
    fn cluster_columns_preserve_every_field_in_input_order() {
        let clusters = [
            Cluster {
                cluster_id: 9,
                first_proxy: 3,
                proxy_count: 5,
                center: [1.0, 2.0, 3.0],
                radius: 4.0,
                geometric_error: 7.25,
            },
            Cluster {
                cluster_id: 2,
                first_proxy: 8,
                proxy_count: 1,
                center: [-3.0, 0.0, 2.0],
                radius: 0.5,
                geometric_error: 0.125,
            },
        ];
        let soa = ClusterSoa::from_clusters(&clusters).unwrap();
        for (index, cluster) in clusters.iter().enumerate() {
            assert_eq!(
                soa.spans[index],
                [
                    cluster.cluster_id,
                    cluster.first_proxy,
                    cluster.proxy_count,
                    cluster.geometric_error.to_bits()
                ]
            );
            assert_eq!(
                soa.bounds[index],
                [
                    cluster.center[0],
                    cluster.center[1],
                    cluster.center[2],
                    cluster.radius
                ]
            );
        }
    }
}
