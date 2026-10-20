//! Real `wgpu` proxy ownership coupled to the engine-neutral scene host.

use core::fmt;

use prism_world_gpu::{
    GpuSectorProxies, GpuSectorProxyError, ProxyRegistryError, ProxyUploadStats, VisibilityInput,
};
use prism_world_stream::{CellManifest, SectorId};

use crate::{CellProxyDecoder, SceneHost};

/// A failure at the decoder, derived registry, or authoritative scene seam.
#[derive(Debug)]
pub enum GpuProxyCellHostError<D, H> {
    /// The opaque cell payload could not produce derived proxies.
    Decoder(D),
    /// The derived proxy registry rejected the cell before the scene host ran.
    Registry(ProxyRegistryError),
    /// The authoritative scene host rejected the cell operation.
    Host(H),
    /// The GPU owner does not contain the sector, so authoritative eviction was not attempted.
    MissingGpuSector(SectorId),
}

impl<D: fmt::Display, H: fmt::Display> fmt::Display for GpuProxyCellHostError<D, H> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Decoder(error) => write!(f, "cell proxy decoder: {error}"),
            Self::Registry(error) => write!(f, "GPU proxy registry: {error}"),
            Self::Host(error) => write!(f, "cell scene host: {error}"),
            Self::MissingGpuSector(sector) => {
                write!(
                    f,
                    "GPU proxy sector is not registered for eviction: {sector:?}"
                )
            }
        }
    }
}

impl<D, H> std::error::Error for GpuProxyCellHostError<D, H>
where
    D: std::error::Error + 'static,
    H: std::error::Error + 'static,
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Decoder(error) => Some(error),
            Self::Registry(error) => Some(error),
            Self::Host(error) => Some(error),
            Self::MissingGpuSector(_) => None,
        }
    }
}

/// Couples one authoritative runtime scene host to one real GPU proxy owner.
///
/// The adapter owns no scene payload, ECS state, loaded-cell map, or second
/// residency truth. `GpuSectorProxies` stores only the derived proxy view and
/// its GPU resources. Loading or reloading decodes and fully prepares the
/// derived registry update before calling the authoritative host, then commits
/// it without a fallible step only after host success. A host error or panic
/// therefore leaves the previous derived sector unchanged. Eviction requires
/// the derived sector to be present, then calls the authoritative host and
/// removes the derived sector only after success.
///
/// The adapter never submits a command encoder. Call [`Self::stage_upload`]
/// after one or more cell changes, then [`Self::write_camera`] and
/// [`Self::encode_visibility`] at the frame boundary. The caller must submit
/// the encoder before the next upload or encode, as required by the GPU owner.
pub struct GpuProxyCellHost<'a, H: SceneHost, D: CellProxyDecoder> {
    host: &'a mut H,
    gpu: &'a mut GpuSectorProxies,
    decoder: D,
}

impl<'a, H: SceneHost, D: CellProxyDecoder> GpuProxyCellHost<'a, H, D> {
    /// Wraps the authoritative host and the single GPU proxy owner.
    #[must_use]
    pub fn new(host: &'a mut H, gpu: &'a mut GpuSectorProxies, decoder: D) -> Self {
        Self { host, gpu, decoder }
    }

    /// Returns the derived GPU owner without exposing a second scene state.
    #[must_use]
    pub fn gpu(&self) -> &GpuSectorProxies {
        self.gpu
    }

    /// Queues staged CPU-to-GPU column/count writes for all pending proxy edits.
    pub fn stage_upload(&mut self) -> Result<ProxyUploadStats, GpuSectorProxyError> {
        self.gpu.upload()
    }

    /// Uploads camera and current dense count after staged edits are synchronized.
    pub fn write_camera(&mut self, camera: VisibilityInput) -> Result<(), GpuSectorProxyError> {
        self.gpu.write_camera(camera)
    }

    /// Records visibility/compaction work into the caller-owned command encoder.
    pub fn encode_visibility(
        &self,
        camera: VisibilityInput,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<(), GpuSectorProxyError> {
        self.gpu.encode(camera, encoder)
    }
}

impl<H: SceneHost, D: CellProxyDecoder> SceneHost for GpuProxyCellHost<'_, H, D> {
    type Error = GpuProxyCellHostError<D::Error, H::Error>;

    fn load_cell(&mut self, manifest: &CellManifest, payload: &[u8]) -> Result<(), Self::Error> {
        let proxies = self
            .decoder
            .decode(manifest, payload)
            .map_err(GpuProxyCellHostError::Decoder)?;

        let pending = self
            .gpu
            .registry_mut()
            .prepare_sector_update(manifest.sector, proxies)
            .map_err(GpuProxyCellHostError::Registry)?;

        self.host
            .load_cell(manifest, payload)
            .map_err(GpuProxyCellHostError::Host)?;
        pending.commit();
        Ok(())
    }

    fn evict_cell(&mut self, sector: SectorId) -> Result<(), Self::Error> {
        if self.gpu.registry().sector_handles(sector).is_none() {
            return Err(GpuProxyCellHostError::MissingGpuSector(sector));
        }
        self.host
            .evict_cell(sector)
            .map_err(GpuProxyCellHostError::Host)?;
        let removed = self.gpu.registry_mut().remove_sector(sector);
        debug_assert!(removed, "sector was checked before authoritative eviction");
        Ok(())
    }
}
