//! A per-application scope reconciling stream eviction with tier work.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use super::{ResidencyBinding, SceneEcsHost, SceneEcsHostError, SceneEcsProvider, SectorId};
use crate::{ApplyError, SceneHost, WorldFramePlan, WorldRuntime};

impl WorldRuntime {
    /// Applies a frame plan through the scene/ECS ownership boundary.
    ///
    /// Delegates ordering and error propagation to [`Self::apply_plan`], while
    /// skipping tier work only for bindings removed by a successful cell unload
    /// in this application. Unknown ids, surviving targets and provider errors
    /// are not suppressed. No eviction exemptions survive this call, including
    /// on error or panic. Reloading a sector invalidates its exemptions.
    ///
    /// Like `apply_plan`, this is fail-fast, not transactional: completed scene
    /// changes remain committed if a later action fails. The caller still owns
    /// IO completion markers and committed residency records.
    pub fn apply_scene_plan<P: SceneEcsProvider>(
        &mut self,
        plan: &WorldFramePlan,
        host: &mut SceneEcsHost<'_, P>,
    ) -> Result<(), ApplyError<SceneEcsHostError<P::Error>>> {
        self.apply_plan(
            plan,
            &mut PlanApplication {
                host,
                evicted: BTreeMap::new(),
            },
        )
    }
}

struct PlanApplication<'host, 'world, P> {
    host: &'host mut SceneEcsHost<'world, P>,
    evicted: BTreeMap<u64, ResidencyBinding>,
}

impl<P: SceneEcsProvider> PlanApplication<'_, '_, P> {
    fn was_evicted(&self, id: u64) -> bool {
        self.host.residency_binding(id).is_none() && self.evicted.contains_key(&id)
    }
}

impl<P: SceneEcsProvider> SceneHost for PlanApplication<'_, '_, P> {
    type Error = SceneEcsHostError<P::Error>;

    fn load_cell(&mut self, sector: SectorId) -> Result<(), Self::Error> {
        SceneHost::load_cell(self.host, sector)?;
        self.evicted.retain(|_, binding| binding.sector != sector);
        Ok(())
    }

    fn unload_cell(&mut self, sector: SectorId) -> Result<(), Self::Error> {
        let removed: Vec<_> = self
            .host
            .ownership()
            .bindings
            .iter()
            .filter(|(_, binding)| binding.sector == sector)
            .map(|(&id, &binding)| (id, binding))
            .collect();
        SceneHost::unload_cell(self.host, sector)?;
        self.evicted.extend(removed);
        Ok(())
    }

    fn promote(&mut self, id: u64) -> Result<(), Self::Error> {
        if self.was_evicted(id) {
            return Ok(());
        }
        SceneHost::promote(self.host, id)
    }

    fn demote(&mut self, id: u64) -> Result<(), Self::Error> {
        if self.was_evicted(id) {
            return Ok(());
        }
        SceneHost::demote(self.host, id)
    }
}
