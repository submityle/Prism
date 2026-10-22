//! A per-application scope reconciling stream eviction with tier work.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use super::{ResidencyBinding, SceneEcsHost, SceneEcsHostError, SceneEcsProvider, SectorId};
use crate::{apply_plan_to, ApplyError, SceneHost, WorldFramePlan, WorldRuntime};
use prism_world_persist::FileJournal;

impl WorldRuntime {
    /// Applies a frame plan while requiring each scene eviction batch to be
    /// durably committed to `journal` before the scene is destroyed.
    ///
    /// Loads replay the same journal, so close/reopen plus this method forms
    /// the durable cell handoff. The journal must have been opened and its
    /// valid prefix recovered before application starts.
    pub fn apply_scene_plan_durable<P: SceneEcsProvider>(
        &mut self,
        plan: &WorldFramePlan,
        host: &mut SceneEcsHost<'_, P>,
        journal: &mut FileJournal,
    ) -> Result<(), ApplyError<SceneEcsHostError<P::Error>>> {
        apply_plan_to(
            plan,
            &mut DurablePlanApplication {
                host,
                journal,
                evicted: BTreeMap::new(),
            },
        )
    }

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
        let Self { persist, .. } = self;
        apply_plan_to(
            plan,
            &mut PlanApplication {
                host,
                log: persist,
                evicted: BTreeMap::new(),
            },
        )
    }
}

struct DurablePlanApplication<'host, 'world, 'journal, P> {
    host: &'host mut SceneEcsHost<'world, P>,
    journal: &'journal mut FileJournal,
    evicted: BTreeMap<u64, ResidencyBinding>,
}

impl<P: SceneEcsProvider> DurablePlanApplication<'_, '_, '_, P> {
    fn was_evicted(&self, id: u64) -> bool {
        self.host.residency_binding(id).is_none() && self.evicted.contains_key(&id)
    }
}

impl<P: SceneEcsProvider> SceneHost for DurablePlanApplication<'_, '_, '_, P> {
    type Error = SceneEcsHostError<P::Error>;

    fn load_cell(&mut self, sector: SectorId) -> Result<(), Self::Error> {
        let log = self.journal.log().clone();
        self.host.load_cell_with_persistence(sector, &log)?;
        self.evicted.retain(|_, binding| binding.sector != sector);
        Ok(())
    }

    fn reload_cells_atomic(&mut self, sectors: &[SectorId]) -> Result<bool, Self::Error> {
        self.host
            .reload_cells_atomic_with_persistence(sectors, self.journal.log())?;
        for sector in sectors {
            self.evicted.retain(|_, binding| binding.sector != *sector);
        }
        Ok(true)
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
        self.host.unload_cell_with_journal(sector, self.journal)?;
        self.evicted.extend(removed);
        Ok(())
    }

    fn transition_residency(
        &mut self,
        id: u64,
        from: prism_world_residency::ResidencyState,
        to: prism_world_residency::ResidencyState,
    ) -> Result<(), Self::Error> {
        if self.was_evicted(id) {
            return Ok(());
        }
        SceneHost::transition_residency(self.host, id, from, to)
    }
}

struct PlanApplication<'host, 'world, P> {
    host: &'host mut SceneEcsHost<'world, P>,
    log: &'host mut prism_world_persist::AppendOnlyLog,
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
        self.host.load_cell_with_persistence(sector, self.log)?;
        self.evicted.retain(|_, binding| binding.sector != sector);
        Ok(())
    }

    fn reload_cells_atomic(&mut self, sectors: &[SectorId]) -> Result<bool, Self::Error> {
        self.host
            .reload_cells_atomic_with_persistence(sectors, self.log)?;
        for sector in sectors {
            self.evicted.retain(|_, binding| binding.sector != *sector);
        }
        Ok(true)
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
        self.host.unload_cell_with_persistence(sector, self.log)?;
        self.evicted.extend(removed);
        Ok(())
    }

    fn transition_residency(
        &mut self,
        id: u64,
        from: prism_world_residency::ResidencyState,
        to: prism_world_residency::ResidencyState,
    ) -> Result<(), Self::Error> {
        if self.was_evicted(id) {
            return Ok(());
        }
        SceneHost::transition_residency(self.host, id, from, to)
    }
}
