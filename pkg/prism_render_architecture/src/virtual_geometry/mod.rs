//! Paged cluster geometry feature boundary.
//!
//! This module owns the CPU-side decision layer for a Nanite-style virtualized
//! geometry pipeline: which discrete LOD of a cluster to display, which cluster
//! pages must be resident to cover the current and imminent view, and which GPU
//! rasterization path a cluster should take. The physical page store, the
//! vis-buffer raster and the streaming I/O all live in the backend; this layer
//! stays deterministic and GPU-independent so it can be unit-tested and shared
//! across backends.
//!
//! The three concerns are split into focused submodules:
//!
//! * [`cull`] — frustum and occlusion cluster culling.
//! * [`lod`] — screen-space-error LOD selection with hysteresis and
//!   velocity-scaled prefetch.
//! * [`page_table`] — page residency bookkeeping and budget-driven eviction.
//! * [`raster_path`] — software-vs-hardware raster path classification.
//! * [`pipeline`] — per-cluster composition of the decisions above.
//! * [`hierarchy`] — screen-space-error cut selection over the cluster DAG.
//! * [`bins`] — GPU-driven raster bin assignment for a selected cut.
//! * [`page_request`] — per-frame page-request coalescing across clusters.

pub mod bins;
pub mod cull;
pub mod hierarchy;
pub mod lod;
pub mod page_request;
pub mod page_table;
pub mod pipeline;
pub mod raster_path;

pub use bins::{bin_cut, RasterBins};
pub use cull::{cluster_cull, CullVerdict, Frustum, OcclusionProbe, Plane};
pub use hierarchy::{ClusterHierarchy, ClusterNode, CutCluster};
pub use lod::{select_lod, LodLevel, LodProjection, LodSelection};
pub use page_request::PageRequestBatch;
pub use page_table::{GeometryPageTable, PageEntry, PageResidency};
pub use pipeline::{ClusterDecision, ClusterRequest, ViewCullContext};
pub use raster_path::{select_raster_path, ClusterRasterStats, RasterCapability};

/// Version of the virtual-geometry contracts in this module.
pub const VIRTUAL_GEOMETRY_VERSION: u32 = 1;

/// Stable key addressing one page of a virtualized geometry asset.
///
/// `asset` identifies the source mesh/geometry stream; `page` identifies the
/// cluster page within that stream. The ordering is purely lexicographic and
/// only exists so pages can live in ordered containers with deterministic
/// iteration.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct GeometryPageKey {
    pub asset: u32,
    pub page: u32,
}

impl GeometryPageKey {
    /// Builds a page key from its asset and page indices.
    #[must_use]
    pub const fn new(asset: u32, page: u32) -> Self {
        Self { asset, page }
    }
}

/// Tunables that shape how aggressively LODs are refined and prefetched.
///
/// * `target_error_pixels` — the projected geometric error budget; the coarsest
///   LOD whose error stays within it is displayed.
/// * `hysteresis_pixels` — dead-band around the target that a level must cross
///   before the selector switches, suppressing LOD popping.
/// * `prefetch_velocity_scale` — world units of look-ahead applied per unit of
///   closing speed so finer pages stream in before the camera arrives.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GeometryLodPolicy {
    pub target_error_pixels: f32,
    pub hysteresis_pixels: f32,
    pub prefetch_velocity_scale: f32,
}

/// GPU rasterization strategy chosen for a cluster.
///
/// Tiny (near sub-pixel) triangles are cheapest through a compute software
/// rasterizer that avoids hardware setup overhead; larger clusters prefer mesh
/// shaders where available, then hardware indirect draws, and finally a plain
/// mesh fallback on capability-poor backends.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GeometryRasterPath {
    MeshShader,
    ComputeSoftware,
    IndirectHardware,
    FallbackMesh,
}
