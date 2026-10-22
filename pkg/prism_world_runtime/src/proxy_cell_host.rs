//! Derived CPU proxy ownership tied to the runtime scene-host lifecycle.

use alloc::vec::Vec;
use core::fmt;

use prism_world_gpu::{ProxyInstance, ProxyRegistry, ProxyRegistryError};
use prism_world_stream::{CellManifest, SectorId};

use crate::SceneHost;

/// Derives proxies from an opaque cell payload without prescribing its format.
///
/// The runtime coordinator validates and strips the manifest header before
/// calling the host. Positions must already use the registry consumer's render
/// coordinate space. The authoritative scene remains owned by [`SceneHost`].
pub trait CellProxyDecoder {
    /// Decoder-specific failure type.
    type Error;

    /// Decodes only the derived proxy representation of this cell.
    fn decode(
        &mut self,
        manifest: &CellManifest,
        payload: &[u8],
    ) -> Result<Vec<ProxyInstance>, Self::Error>;
}

impl<F, E> CellProxyDecoder for F
where
    F: FnMut(&CellManifest, &[u8]) -> Result<Vec<ProxyInstance>, E>,
{
    type Error = E;

    fn decode(
        &mut self,
        manifest: &CellManifest,
        payload: &[u8],
    ) -> Result<Vec<ProxyInstance>, Self::Error> {
        self(manifest, payload)
    }
}

/// A failure at one of the decoder, derived registry, or authoritative host seams.
#[derive(Debug)]
pub enum ProxyCellHostError<D, H> {
    /// Proxy decoding failed before either registry insertion or host loading.
    Decoder(D),
    /// Registry admission failed before calling the authoritative load method.
    Registry(ProxyRegistryError),
    /// The authoritative host rejected loading or eviction.
    Host(H),
}

impl<D: fmt::Display, H: fmt::Display> fmt::Display for ProxyCellHostError<D, H> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Decoder(error) => write!(f, "cell proxy decoder: {error}"),
            Self::Registry(error) => write!(f, "cell proxy registry: {error}"),
            Self::Host(error) => write!(f, "cell scene host: {error}"),
        }
    }
}

impl<D, H> std::error::Error for ProxyCellHostError<D, H>
where
    D: std::error::Error + 'static,
    H: std::error::Error + 'static,
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Decoder(error) => Some(error),
            Self::Registry(error) => Some(error),
            Self::Host(error) => Some(error),
        }
    }
}

/// Borrows an authoritative scene host and its derived CPU proxy registry.
///
/// Loading decodes and prepares an insertion or same-sector replacement before
/// calling `host.load_cell`. Invalid proxies and exhausted capacity never reach
/// the host. The prepared update changes no registry state until the host accepts
/// the authoritative cell; a host error or unwind therefore preserves the prior
/// sector generation. Eviction calls the host first and removes proxies only on
/// success; an error or panic leaves the registry untouched.
///
/// This is **not** an atomic transaction over the authoritative host. The
/// [`SceneHost`] contract does not promise failure atomicity: host errors and
/// panics can leave partial host changes, which this adapter cannot undo. Only
/// the derived sector insertion is compensated. Registry handle generations,
/// dense ordering, and upload dirtiness are not restored to an earlier snapshot.
/// Panic-abort builds do not run rollback destructors.
///
/// No second loaded-state truth is kept, no asset format is imposed, and no GPU
/// resources, uploads, or submissions are created. Dropping this adapter merely
/// ends its borrows; it does not evict successfully loaded cells.
pub struct ProxyCellHost<'a, H: SceneHost, D: CellProxyDecoder> {
    host: &'a mut H,
    registry: &'a mut ProxyRegistry,
    decoder: D,
}

impl<'a, H: SceneHost, D: CellProxyDecoder> ProxyCellHost<'a, H, D> {
    /// Wraps the borrowed host and registry without changing either one.
    #[must_use]
    pub fn new(host: &'a mut H, registry: &'a mut ProxyRegistry, decoder: D) -> Self {
        Self {
            host,
            registry,
            decoder,
        }
    }

    /// Observes derived state without allowing lifecycle-bypassing mutations.
    #[must_use]
    pub fn registry(&self) -> &ProxyRegistry {
        self.registry
    }
}

impl<H: SceneHost, D: CellProxyDecoder> SceneHost for ProxyCellHost<'_, H, D> {
    type Error = ProxyCellHostError<D::Error, H::Error>;

    fn load_cell(&mut self, manifest: &CellManifest, payload: &[u8]) -> Result<(), Self::Error> {
        let proxies = self
            .decoder
            .decode(manifest, payload)
            .map_err(ProxyCellHostError::Decoder)?;
        let pending = self
            .registry
            .prepare_sector_update(manifest.sector, proxies)
            .map_err(ProxyCellHostError::Registry)?;
        self.host
            .load_cell(manifest, payload)
            .map_err(ProxyCellHostError::Host)?;
        pending.commit();
        Ok(())
    }

    fn evict_cell(&mut self, sector: SectorId) -> Result<(), Self::Error> {
        self.host
            .evict_cell(sector)
            .map_err(ProxyCellHostError::Host)?;
        self.registry.remove_sector(sector);
        Ok(())
    }
}
