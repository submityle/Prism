//! Typed ECS residency transitions coupled to the persistent CPU proxy registry.

use core::fmt;

use prism_world::{ResidencyBatchError, ResidencyTransition, SceneHost as WorldSceneHost};
use prism_world_gpu::{ProxyInstance, ProxyRegistry, ProxyRegistryError, PROXY_FLAG_DISABLED};
use prism_world_residency::ResidencyState;
use prism_world_stream::SectorId;

/// Error produced by a typed world host or the derived proxy registry.
#[derive(Debug)]
pub enum ResidencyProxySceneHostError<H> {
    /// The authoritative ECS provider rejected a transition.
    Host(H),
    /// The residency identity has no current proxy binding.
    ResidencyProxyNotBound(u64),
    /// The proxy visibility bit does not agree with the committed source tier.
    ProxyStateMismatch {
        /// Stable world residency identity.
        residency_id: u64,
        /// Whether the current proxy should be visible for the source tier.
        expected_proxy: bool,
    },
    /// Updating the validated proxy handle failed.
    Registry(ProxyRegistryError),
}

impl<H: fmt::Display> fmt::Display for ResidencyProxySceneHostError<H> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Host(error) => write!(f, "world scene host: {error}"),
            Self::ResidencyProxyNotBound(id) => write!(f, "residency {id} has no current proxy binding"),
            Self::ProxyStateMismatch { residency_id, expected_proxy } => write!(f, "proxy state for residency {residency_id} does not match expected Tier C state {expected_proxy}"),
            Self::Registry(error) => write!(f, "proxy registry update failed: {error}"),
        }
    }
}

impl<H> std::error::Error for ResidencyProxySceneHostError<H>
where
    H: std::error::Error + 'static,
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Host(error) => Some(error),
            Self::Registry(error) => Some(error),
            Self::ResidencyProxyNotBound(_) | Self::ProxyStateMismatch { .. } => None,
        }
    }
}

/// Decorates the ECS-backed world host with derived proxy state.
///
/// Residency bindings live in ProxyRegistry, not this wrapper, so the wrapper
/// can be rebuilt for each coordinator call. Transition preflight resolves the
/// target before provider side effects. On success the provider commits first,
/// then the prevalidated CPU proxy record is updated without allocation.
pub struct ResidencyProxySceneHost<'a, H: WorldSceneHost> {
    host: &'a mut H,
    registry: &'a mut ProxyRegistry,
}

impl<'a, H: WorldSceneHost> ResidencyProxySceneHost<'a, H> {
    /// Creates the transition decorator around the durable registry.
    #[must_use]
    pub fn new(host: &'a mut H, registry: &'a mut ProxyRegistry) -> Self {
        Self { host, registry }
    }

    /// Binds a residency identity to a live sector-local proxy content ID.
    pub fn bind_residency_proxy(
        &mut self,
        residency_id: u64,
        sector: SectorId,
        proxy_id: u32,
    ) -> Result<(), ProxyRegistryError> {
        self.registry
            .bind_residency_proxy(residency_id, sector, proxy_id)
    }

    /// Returns the current derived proxy record for a residency identity.
    #[must_use]
    pub fn proxy_for_residency(&self, residency_id: u64) -> Option<&ProxyInstance> {
        self.registry
            .proxy_for_residency(residency_id)
            .map(|(_, proxy)| proxy)
    }

    fn prepare_proxy(
        &self,
        id: u64,
        from: ResidencyState,
        to: ResidencyState,
    ) -> Result<
        Option<(prism_world_gpu::ProxyHandle, ProxyInstance)>,
        ResidencyProxySceneHostError<H::Error>,
    > {
        if (from == ResidencyState::TierC) == (to == ResidencyState::TierC) {
            return Ok(None);
        }
        let (handle, current) = self
            .registry
            .proxy_for_residency(id)
            .ok_or(ResidencyProxySceneHostError::ResidencyProxyNotBound(id))?;
        let mut updated = *current;
        let is_visible = updated.flags & PROXY_FLAG_DISABLED == 0;
        if is_visible != (from == ResidencyState::TierC) {
            return Err(ResidencyProxySceneHostError::ProxyStateMismatch {
                residency_id: id,
                expected_proxy: from == ResidencyState::TierC,
            });
        }
        if to == ResidencyState::TierC {
            updated.flags &= !PROXY_FLAG_DISABLED;
        } else {
            updated.flags |= PROXY_FLAG_DISABLED;
        }
        Ok(Some((handle, updated)))
    }

    fn transition_one(
        &mut self,
        id: u64,
        from: ResidencyState,
        to: ResidencyState,
    ) -> Result<(), ResidencyProxySceneHostError<H::Error>> {
        let update = self.prepare_proxy(id, from, to)?;
        self.host
            .transition_residency(id, from, to)
            .map_err(ResidencyProxySceneHostError::Host)?;
        if let Some((handle, proxy)) = update {
            self.registry
                .update(handle, proxy)
                .map_err(ResidencyProxySceneHostError::Registry)?;
        }
        Ok(())
    }
}

impl<H: WorldSceneHost> WorldSceneHost for ResidencyProxySceneHost<'_, H> {
    type Error = ResidencyProxySceneHostError<H::Error>;

    fn load_cell(&mut self, sector: SectorId) -> Result<(), Self::Error> {
        self.host
            .load_cell(sector)
            .map_err(ResidencyProxySceneHostError::Host)
    }

    fn reload_cells_atomic(&mut self, sectors: &[SectorId]) -> Result<bool, Self::Error> {
        self.host
            .reload_cells_atomic(sectors)
            .map_err(ResidencyProxySceneHostError::Host)
    }

    fn unload_cell(&mut self, sector: SectorId) -> Result<(), Self::Error> {
        self.host
            .unload_cell(sector)
            .map_err(ResidencyProxySceneHostError::Host)?;
        self.registry.remove_sector(sector);
        Ok(())
    }

    fn transition_residency(
        &mut self,
        id: u64,
        from: ResidencyState,
        to: ResidencyState,
    ) -> Result<(), Self::Error> {
        self.transition_one(id, from, to)
    }

    fn transition_residency_batch(
        &mut self,
        transitions: &[ResidencyTransition],
    ) -> Result<(), ResidencyBatchError<Self::Error>> {
        for (completed, transition) in transitions.iter().enumerate() {
            if let Err(error) = self.transition_one(transition.id, transition.from, transition.to) {
                return Err(ResidencyBatchError { completed, error });
            }
        }
        Ok(())
    }
}
