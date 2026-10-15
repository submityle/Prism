//! Sector ownership and incremental uploads into fixed resident GPU buffers.

use crate::{
    DensityPolicy, GpuContext, GpuResidentError, GpuResidentVisibility, ProxyRegistry,
    ProxyRegistryError, ProxySoa, VisibilityInput,
};

/// Errors detected by the ownership/upload boundary.
#[derive(Debug)]
pub enum GpuSectorProxyError {
    /// Invalid registry capacity or CPU allocation failure.
    Registry(ProxyRegistryError),
    /// A resident buffer, camera, or device limit was rejected.
    Resident(GpuResidentError),
    /// The reusable CPU SoA staging allocation could not be reserved.
    StagingAllocation,
    /// Registry edits have not yet been uploaded; old GPU contents cannot render.
    PendingUpload,
    /// The borrowed registry was replaced rather than edited in place.
    RegistryReplaced,
}

impl std::fmt::Display for GpuSectorProxyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Registry(error) => write!(f, "sector proxy registry: {error}"),
            Self::Resident(error) => write!(f, "sector proxy resident GPU: {error}"),
            Self::StagingAllocation => f.write_str("could not reserve proxy SoA staging"),
            Self::PendingUpload => f.write_str("upload sector proxy edits before encoding"),
            Self::RegistryReplaced => {
                f.write_str("resident owner registry was replaced; recreate the GPU owner")
            }
        }
    }
}

impl std::error::Error for GpuSectorProxyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Registry(error) => Some(error),
            Self::Resident(error) => Some(error),
            Self::StagingAllocation | Self::PendingUpload | Self::RegistryReplaced => None,
        }
    }
}

impl From<ProxyRegistryError> for GpuSectorProxyError {
    fn from(error: ProxyRegistryError) -> Self {
        Self::Registry(error)
    }
}

impl From<GpuResidentError> for GpuSectorProxyError {
    fn from(error: GpuResidentError) -> Self {
        Self::Resident(error)
    }
}

/// Logical column writes queued by one upload, excluding camera and queue staging.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ProxyUploadStats {
    /// Number of coalesced source-slot ranges written to each SoA column.
    pub ranges: usize,
    /// Total rows transposed and uploaded, counting moved rows once.
    pub proxies: usize,
    /// Logical payload bytes (`48 * proxies`), not physical transfer or VRAM.
    pub column_bytes: u64,
    /// Active dense prefix after applying this batch.
    pub dense_count: usize,
}

/// Owns the sector registry, fixed GPU allocation, and reusable upload scratch.
///
/// Content remains derived: callers insert already decoded scene/scatter proxies
/// and remove their sector when the corresponding cell is evicted. This type
/// never owns author data, loads archives, or submits a frame. Proxy positions
/// must already use one consistent camera-relative coordinate space.
///
/// Edits invalidate encoding until `upload` succeeds. A shrink updates the draw
/// count even when it needs no column writes. Unchanged uploads perform no queue
/// writes or transposition. Scratch is reserved once (48 bytes per capacity row,
/// excluding allocator overhead); GPU staging remains managed by wgpu.
///
/// Submit each encoded frame before the next upload, camera write, or encode on
/// this object. Multiple sequential submissions need no host wait/readback.
pub struct GpuSectorProxies {
    registry: ProxyRegistry,
    registry_identity: std::sync::Arc<()>,
    resident: GpuResidentVisibility,
    scratch: ProxySoa,
}

impl GpuSectorProxies {
    /// Creates fixed-capacity ownership, staging, and resident GPU resources.
    pub fn new(
        ctx: &GpuContext,
        capacity: usize,
        vertex_count: u32,
    ) -> Result<Self, GpuSectorProxyError> {
        Self::with_sector_capacity(ctx, capacity, capacity.max(1), vertex_count)
    }

    /// Configures independent proxy and sector budgets, including empty sectors.
    /// A zero sector budget rejects every insertion. GPU limits are validated
    /// before any capacity-sized CPU allocation.
    pub fn with_sector_capacity(
        ctx: &GpuContext,
        capacity: usize,
        sector_capacity: usize,
        vertex_count: u32,
    ) -> Result<Self, GpuSectorProxyError> {
        Self::build(ctx, capacity, sector_capacity, vertex_count, None)
    }

    /// Adds fixed GPU density/coverage resources with independent sector budget.
    /// Renderers must consume the resident coverage column to display fades.
    pub fn with_sector_capacity_and_density(
        ctx: &GpuContext,
        capacity: usize,
        sector_capacity: usize,
        vertex_count: u32,
        policy: DensityPolicy,
    ) -> Result<Self, GpuSectorProxyError> {
        Self::build(ctx, capacity, sector_capacity, vertex_count, Some(policy))
    }

    fn build(
        ctx: &GpuContext,
        capacity: usize,
        sector_capacity: usize,
        vertex_count: u32,
        density: Option<DensityPolicy>,
    ) -> Result<Self, GpuSectorProxyError> {
        crate::resident::validate_capacity(capacity, &ctx.device().limits())?;
        if density.is_some() {
            crate::resident_density::validate_limits(&ctx.device().limits())?;
        }
        let registry = ProxyRegistry::with_sector_capacity(capacity, sector_capacity)?;
        let registry_identity = registry.identity().clone();
        let mut scratch = ProxySoa {
            positions: Vec::new(),
            errors: Vec::new(),
            metadata: Vec::new(),
        };
        scratch
            .positions
            .try_reserve_exact(capacity)
            .map_err(|_| GpuSectorProxyError::StagingAllocation)?;
        scratch
            .errors
            .try_reserve_exact(capacity)
            .map_err(|_| GpuSectorProxyError::StagingAllocation)?;
        scratch
            .metadata
            .try_reserve_exact(capacity)
            .map_err(|_| GpuSectorProxyError::StagingAllocation)?;
        let resident = match density {
            Some(policy) => {
                GpuResidentVisibility::new_with_density(ctx, capacity, vertex_count, policy)?
            }
            None => GpuResidentVisibility::new(ctx, capacity, vertex_count)?,
        };
        Ok(Self {
            registry,
            registry_identity,
            resident,
            scratch,
        })
    }

    /// Read-only ownership and handle-to-current-source-slot lookup.
    pub fn registry(&self) -> &ProxyRegistry {
        &self.registry
    }

    /// Edits are tracked by the registry; stale handles cannot mutate new rows.
    /// Replacing the entire registry invalidates this GPU owner, even if the new
    /// registry has the same capacity. Recreate the owner rather than replacing it.
    pub fn registry_mut(&mut self) -> &mut ProxyRegistry {
        &mut self.registry
    }

    /// Read-only render access, rejected while registry edits await upload.
    /// Compact output contains current dense slots.
    /// Resolve a slot with `registry().handle_at_dense`, not as a persistent ID.
    pub fn resident(&self) -> Result<&GpuResidentVisibility, GpuSectorProxyError> {
        self.ensure_uploaded()?;
        Ok(&self.resident)
    }

    /// Transposes dirty ranges into reserved scratch and queues column updates.
    ///
    /// No GPU resources are created, no queue submit or readback occurs. Dirty
    /// tracking is acknowledged only after every range and count are accepted;
    /// if validation returns an error, replaying all pending writes is safe.
    /// Device loss follows normal wgpu handling and requires recreating this
    /// derived state, not acknowledging it as a recovered GPU transaction.
    pub fn upload(&mut self) -> Result<ProxyUploadStats, GpuSectorProxyError> {
        self.ensure_identity()?;
        let mut stats = ProxyUploadStats {
            dense_count: self.registry.len(),
            ..ProxyUploadStats::default()
        };
        if !self.registry.has_pending_uploads() {
            return Ok(stats);
        }
        for range in self.registry.dirty_ranges() {
            self.scratch.positions.clear();
            self.scratch.errors.clear();
            self.scratch.metadata.clear();
            for proxy in &self.registry.proxies()[range.clone()] {
                self.scratch.positions.push([
                    proxy.position[0],
                    proxy.position[1],
                    proxy.position[2],
                    proxy.radius,
                ]);
                self.scratch
                    .errors
                    .push([proxy.geometric_error, proxy.max_distance, 0.0, 0.0]);
                self.scratch
                    .metadata
                    .push([proxy.proxy_id, proxy.cluster_id, proxy.flags, 0]);
            }
            self.resident.write_range(range.start, &self.scratch)?;
            stats.ranges += 1;
            stats.proxies += range.len();
        }
        self.resident.set_dense_count(stats.dense_count)?;
        self.registry.acknowledge_upload();
        stats.column_bytes = stats.proxies as u64 * 48;
        Ok(stats)
    }

    /// Uploads camera/count only after registry edits are synchronized.
    pub fn write_camera(&mut self, camera: VisibilityInput) -> Result<(), GpuSectorProxyError> {
        self.ensure_uploaded()?;
        self.resident.write_camera(camera)?;
        Ok(())
    }

    /// Updates only policy after ownership is synchronized; never rescans rows.
    pub fn write_density(&mut self, policy: DensityPolicy) -> Result<(), GpuSectorProxyError> {
        self.ensure_uploaded()?;
        self.resident.write_density(policy)?;
        Ok(())
    }

    /// Records culling/compact after upload and camera synchronization.
    /// Pending edits fail before any command is recorded, including full unload.
    pub fn encode(
        &self,
        camera: VisibilityInput,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<(), GpuSectorProxyError> {
        self.ensure_uploaded()?;
        self.resident.encode(camera, encoder)?;
        Ok(())
    }

    fn ensure_uploaded(&self) -> Result<(), GpuSectorProxyError> {
        self.ensure_identity()?;
        if self.registry.has_pending_uploads() {
            Err(GpuSectorProxyError::PendingUpload)
        } else {
            Ok(())
        }
    }

    fn ensure_identity(&self) -> Result<(), GpuSectorProxyError> {
        if std::sync::Arc::ptr_eq(&self.registry_identity, self.registry.identity()) {
            Ok(())
        } else {
            Err(GpuSectorProxyError::RegistryReplaced)
        }
    }
}
