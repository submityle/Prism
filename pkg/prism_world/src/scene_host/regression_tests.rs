extern crate std;

use self::std::panic::{catch_unwind, AssertUnwindSafe};
use super::tests::{Marker, Provider};
use super::*;
use crate::{ApplyError, SceneHost, WorldBudget, WorldRuntime};
use prism_ecs::component_hooks::{ComponentHooks, HookContext};
use prism_world_residency::{
    MaterializationBudget, ResidencyAction, ResidencyPolicy, ResidencyRecord, ResidencyScheduler,
    ResidencyState,
};
use prism_world_stream::{
    AabbI64, CellManifest, DemandSource, GlobalPos, StreamAction, StreamBudget,
};

fn runtime() -> WorldRuntime {
    WorldRuntime::new(ResidencyScheduler::new(
        ResidencyPolicy::new(1, 2, 3, 4, 0).unwrap(),
    ))
}

fn budget() -> WorldBudget {
    WorldBudget {
        stream: StreamBudget {
            io_bytes: 8,
            max_actions: 8,
        },
        materialization: MaterializationBudget::new(8, 8),
    }
}

fn manifest(sector: SectorId) -> CellManifest {
    CellManifest {
        sector,
        bounds: AabbI64::new(GlobalPos::new(0, 0, 0), GlobalPos::new(1, 1, 1)).unwrap(),
        scene_key: 1,
        content_version: 1,
        derived_hash: 2,
        io_bytes: 1,
    }
}

fn load_bound_cell(
    runtime: &mut WorldRuntime,
    host: &mut SceneEcsHost<'_, Provider>,
    sector: SectorId,
    id: u64,
) -> Entity {
    let plan = runtime.tick(
        &[DemandSource::new(sector, 0, 1, 10).unwrap()],
        &[manifest(sector)],
        &[],
        1,
        budget(),
    );
    assert!(matches!(
        plan.stream.actions.as_slice(),
        [StreamAction::Load(_)]
    ));
    runtime.apply_plan(&plan, host).unwrap();
    runtime.mark_resident(sector);
    host.bind_residency(
        id,
        ResidencyBinding {
            sector,
            local: LocalEntityId::new(0),
        },
    )
    .unwrap();
    host.loaded_cell(sector).unwrap().root.unwrap()
}

#[test]
fn runtime_tick_unload_and_demote_apply_successfully() {
    let mut world = World::new();
    let sector = SectorId::new(0, 0, 0, 0);
    let mut runtime = runtime();
    let mut host = SceneEcsHost::new(&mut world, Provider::default());
    let root = load_bound_cell(&mut runtime, &mut host, sector, 41);

    let plan = runtime.tick(
        &[],
        &[manifest(sector)],
        &[ResidencyRecord::new(41, ResidencyState::TierA, 10, 1, 1, 1)],
        2,
        budget(),
    );
    assert_eq!(
        plan.stream.actions,
        alloc::vec![StreamAction::Unload(sector)]
    );
    assert!(matches!(
        plan.residency.actions.as_slice(),
        [ResidencyAction::Demote { id: 41, .. }]
    ));
    runtime.apply_scene_plan(&plan, &mut host).unwrap();
    assert_eq!(host.loaded_cell_count(), 0);
    assert_eq!(host.residency_binding(41), None);
    assert!(!host.world().contains(root));
    assert_eq!(host.world().entity_count(), 0);
    assert!(host.provider().events.is_empty());
}

#[test]
fn dropping_host_preserves_world_owned_cells_and_bindings_for_takeover() {
    let mut world = World::new();
    let sector = SectorId::new(0, 0, 0, 0);
    let mut runtime = runtime();
    let root = {
        let mut host = SceneEcsHost::new(&mut world, Provider::default());
        load_bound_cell(&mut runtime, &mut host, sector, 41)
    };
    assert!(world.contains(root));

    let mut replacement = SceneEcsHost::new(&mut world, Provider::default());
    assert_eq!(replacement.loaded_cell_count(), 1);
    assert_eq!(replacement.loaded_cell(sector).unwrap().root, Some(root));
    assert_eq!(replacement.residency_binding(41).unwrap().sector, sector);
    SceneHost::demote(&mut replacement, 41).unwrap();
    assert_eq!(replacement.world().get::<Marker>(root), Some(&Marker(3)));
    SceneHost::unload_cell(&mut replacement, sector).unwrap();
    assert_eq!(replacement.world().entity_count(), 0);
}

#[test]
fn same_plan_eviction_does_not_hide_an_originally_unknown_id() {
    let mut world = World::new();
    let mut runtime = runtime();
    let mut host = SceneEcsHost::new(&mut world, Provider::default());
    let sector = SectorId::new(0, 0, 0, 0);
    let root = load_bound_cell(&mut runtime, &mut host, sector, 41);
    let plan = runtime.tick(
        &[],
        &[],
        &[
            ResidencyRecord::new(41, ResidencyState::TierA, 10, 1, 1, 1),
            ResidencyRecord::new(99, ResidencyState::TierA, 9, 1, 1, 1),
        ],
        2,
        budget(),
    );
    assert_eq!(plan.residency.actions[0].id(), 41);
    assert_eq!(
        runtime.apply_scene_plan(&plan, &mut host),
        Err(ApplyError::Host(SceneEcsHostError::ResidencyNotBound(99)))
    );
    // The failed application is intentionally not a transaction.
    assert!(!host.world().contains(root));
    assert_eq!(host.loaded_cell_count(), 0);
    assert!(host.provider().events.is_empty());
    // Even the previous eviction's id must fail in the next application.
    let next = runtime.tick(
        &[],
        &[],
        &[ResidencyRecord::new(41, ResidencyState::TierA, 10, 1, 1, 1)],
        3,
        budget(),
    );
    assert_eq!(
        runtime.apply_scene_plan(&next, &mut host),
        Err(ApplyError::Host(SceneEcsHostError::ResidencyNotBound(41)))
    );
}

#[test]
fn successful_plan_does_not_leave_eviction_exemptions() {
    let mut world = World::new();
    let mut runtime = runtime();
    let mut host = SceneEcsHost::new(&mut world, Provider::default());
    let sector = SectorId::new(0, 0, 0, 0);
    load_bound_cell(&mut runtime, &mut host, sector, 41);
    let record = ResidencyRecord::new(41, ResidencyState::TierA, 10, 1, 1, 1);
    let plan = runtime.tick(&[], &[], &[record], 2, budget());
    runtime.apply_scene_plan(&plan, &mut host).unwrap();
    assert_eq!(
        SceneHost::demote(&mut host, 41),
        Err(SceneEcsHostError::ResidencyNotBound(41))
    );
    let next = runtime.tick(&[], &[], &[record], 3, budget());
    assert!(next.stream.actions.is_empty());
    assert_eq!(
        runtime.apply_scene_plan(&next, &mut host),
        Err(ApplyError::Host(SceneEcsHostError::ResidencyNotBound(41)))
    );
    // After a real reload and explicit rebind, the same id is usable again.
    let root = load_bound_cell(&mut runtime, &mut host, sector, 41);
    let plan = runtime.tick(
        &[DemandSource::new(sector, 0, 1, 10).unwrap()],
        &[],
        &[record],
        4,
        budget(),
    );
    runtime.apply_scene_plan(&plan, &mut host).unwrap();
    assert_eq!(host.provider().events, alloc::vec![(2, 41, root)]);
}

#[test]
fn eviction_does_not_skip_surviving_cells_or_provider_errors() {
    let mut world = World::new();
    let mut runtime = runtime();
    let mut host = SceneEcsHost::new(&mut world, Provider::default());
    let evicted = SectorId::new(0, 0, 0, 0);
    let surviving = SectorId::new(0, 10, 0, 0);
    let removed_root = load_bound_cell(&mut runtime, &mut host, evicted, 41);
    // Keep the first cell in range while loading the second one.
    let load = runtime.tick(
        &[
            DemandSource::new(evicted, 0, 1, 10).unwrap(),
            DemandSource::new(surviving, 0, 1, 10).unwrap(),
        ],
        &[manifest(surviving)],
        &[],
        2,
        budget(),
    );
    runtime.apply_scene_plan(&load, &mut host).unwrap();
    runtime.mark_resident(surviving);
    host.bind_residency(
        42,
        ResidencyBinding {
            sector: surviving,
            local: LocalEntityId::new(0),
        },
    )
    .unwrap();
    let surviving_root = host.loaded_cell(surviving).unwrap().root.unwrap();
    let demands = [DemandSource::new(surviving, 0, 1, 10).unwrap()];
    let plan = runtime.tick(
        &demands,
        &[],
        &[
            ResidencyRecord::new(41, ResidencyState::TierA, 10, 1, 1, 1),
            ResidencyRecord::new(42, ResidencyState::TierA, 9, 1, 1, 1),
        ],
        3,
        budget(),
    );
    host.provider_mut().fail_dematerialize = true;
    assert_eq!(
        runtime.apply_scene_plan(&plan, &mut host),
        Err(ApplyError::Host(SceneEcsHostError::Provider(())))
    );
    assert!(!host.world().contains(removed_root));
    assert_eq!(host.world().get::<Marker>(surviving_root), Some(&Marker(7)));
    host.provider_mut().fail_dematerialize = false;
    let next = runtime.tick(
        &demands,
        &[],
        &[ResidencyRecord::new(42, ResidencyState::TierA, 9, 1, 1, 1)],
        4,
        budget(),
    );
    runtime.apply_scene_plan(&next, &mut host).unwrap();
    assert_eq!(host.provider().events, alloc::vec![(2, 42, surviving_root)]);
    assert_eq!(host.world().get::<Marker>(surviving_root), Some(&Marker(3)));
}

#[test]
fn bindings_to_never_loaded_cells_and_missing_locals_still_fail() {
    let mut world = World::new();
    let mut runtime = runtime();
    let mut host = SceneEcsHost::new(&mut world, Provider::default());
    let sector = SectorId::new(0, 0, 0, 0);
    host.bind_residency(
        41,
        ResidencyBinding {
            sector,
            local: LocalEntityId::new(1),
        },
    )
    .unwrap();
    let record = ResidencyRecord::new(41, ResidencyState::TierA, 10, 1, 1, 1);
    let plan = runtime.tick(&[], &[], &[record], 1, budget());
    assert_eq!(
        runtime.apply_scene_plan(&plan, &mut host),
        Err(ApplyError::Host(SceneEcsHostError::CellNotLoaded(sector)))
    );
    let plan = runtime.tick(
        &[DemandSource::new(sector, 0, 1, 10).unwrap()],
        &[manifest(sector)],
        &[record],
        2,
        budget(),
    );
    assert_eq!(
        runtime.apply_scene_plan(&plan, &mut host),
        Err(ApplyError::Host(
            SceneEcsHostError::LocalEntityUnavailable {
                residency_id: 41,
                sector,
                local: LocalEntityId::new(1),
            }
        ))
    );
}

#[test]
fn eviction_skips_promotions_but_rejects_unknown_promotions() {
    let mut world = World::new();
    let mut runtime = runtime();
    let mut host = SceneEcsHost::new(&mut world, Provider::default());
    let sector = SectorId::new(0, 0, 0, 0);
    let root = load_bound_cell(&mut runtime, &mut host, sector, 41);
    let plan = runtime.tick(
        &[],
        &[],
        &[
            ResidencyRecord::new(41, ResidencyState::TierC, 0, 1, 1, 1),
            ResidencyRecord::new(99, ResidencyState::TierC, 0, 1, 1, 1),
        ],
        2,
        budget(),
    );
    assert_eq!(
        plan.stream.actions,
        alloc::vec![StreamAction::Unload(sector)]
    );
    assert!(matches!(
        plan.residency.actions.as_slice(),
        [
            ResidencyAction::Promote { id: 41, .. },
            ResidencyAction::Promote { id: 99, .. }
        ]
    ));
    assert_eq!(
        runtime.apply_scene_plan(&plan, &mut host),
        Err(ApplyError::Host(SceneEcsHostError::ResidencyNotBound(99)))
    );
    assert!(!host.world().contains(root));
    assert!(host.provider().events.is_empty());
}

#[test]
fn out_of_plan_unload_and_failed_unload_never_exempt_tier_work() {
    let mut world = World::new();
    let mut runtime = runtime();
    let mut host = SceneEcsHost::new(&mut world, Provider::default());
    let sector = SectorId::new(0, 0, 0, 0);
    load_bound_cell(&mut runtime, &mut host, sector, 41);
    let record = ResidencyRecord::new(41, ResidencyState::TierA, 10, 1, 1, 1);
    let plan = runtime.tick(&[], &[], &[record], 2, budget());
    SceneHost::unload_cell(&mut host, sector).unwrap();
    // This unload did not succeed inside the application: fail before tier work.
    assert_eq!(
        runtime.apply_scene_plan(&plan, &mut host),
        Err(ApplyError::Host(SceneEcsHostError::CellNotLoaded(sector)))
    );
    let next = runtime.tick(&[], &[], &[record], 3, budget());
    assert!(next.stream.actions.is_empty());
    assert_eq!(
        runtime.apply_scene_plan(&next, &mut host),
        Err(ApplyError::Host(SceneEcsHostError::ResidencyNotBound(41)))
    );
    assert!(host.provider().events.is_empty());
}

#[test]
fn explicit_shutdown_preserves_unrelated_entities_and_allows_reuse() {
    let mut world = World::new();
    let unrelated = world.spawn(Marker(17));
    let sector = SectorId::new(0, 0, 0, 0);
    let mut runtime = runtime();
    let mut host = SceneEcsHost::new(&mut world, Provider::default());
    let root = load_bound_cell(&mut runtime, &mut host, sector, 41);
    // Shutdown also clears bindings without a corresponding loaded cell.
    host.bind_residency(
        99,
        ResidencyBinding {
            sector: SectorId::new(0, 1, 0, 0),
            local: LocalEntityId::new(0),
        },
    )
    .unwrap();
    assert_eq!(host.shutdown(), 1);
    assert_eq!(host.shutdown(), 0);
    assert_eq!(host.residency_binding(41), None);
    assert_eq!(host.residency_binding(99), None);
    assert!(!host.world().contains(root));
    assert_eq!(host.world().get::<Marker>(unrelated), Some(&Marker(17)));
    runtime.mark_unloaded(sector);
    let provider = host.into_provider();
    let mut replacement = SceneEcsHost::new(&mut world, provider);
    assert_eq!(replacement.loaded_cell_count(), 0);
    load_bound_cell(&mut runtime, &mut replacement, sector, 41);
    assert_eq!(replacement.shutdown(), 1);
    assert_eq!(replacement.world().entity_count(), 1);
}

#[test]
fn provider_handoff_preserves_state_without_unloading_or_overwriting_cells() {
    let mut world = World::new();
    let sector = SectorId::new(0, 0, 0, 0);
    let mut runtime = runtime();
    let mut host = SceneEcsHost::new(&mut world, Provider::default());
    let root = load_bound_cell(&mut runtime, &mut host, sector, 41);
    SceneHost::promote(&mut host, 41).unwrap();
    let provider = host.into_provider();
    let mut replacement = SceneEcsHost::new(&mut world, provider);
    assert_eq!(replacement.provider().events, alloc::vec![(1, 41, root)]);
    assert_eq!(
        SceneHost::load_cell(&mut replacement, sector),
        Err(SceneEcsHostError::CellAlreadyLoaded(sector))
    );
    assert_eq!(replacement.loaded_cell(sector).unwrap().root, Some(root));
    assert_eq!(replacement.shutdown(), 1);
}

struct DespawnPanicProbe {
    entity: Entity,
    id: u64,
    binding: ResidencyBinding,
    ownership_visible_in_hook: bool,
}

impl prism_ecs::resource::Resource for DespawnPanicProbe {}

fn panic_during_remove(ctx: HookContext<'_>) {
    let probe = ctx.world.resource::<DespawnPanicProbe>();
    if ctx.entity != probe.entity {
        return;
    }
    let ownership = ctx.world.resource::<SceneOwnership>();
    let visible = ownership.cells.contains_key(&probe.binding.sector)
        && ownership.bindings.get(&probe.id) == Some(&probe.binding);
    ctx.world
        .resource_mut::<DespawnPanicProbe>()
        .ownership_visible_in_hook = visible;
    panic!("scene despawn hook panic");
}

fn install_despawn_panic(host: &mut SceneEcsHost<'_, Provider>, entity: Entity, id: u64) {
    let binding = host.residency_binding(id).unwrap();
    host.world_mut().insert_resource(DespawnPanicProbe {
        entity,
        id,
        binding,
        ownership_visible_in_hook: false,
    });
    host.world_mut().register_component_hooks::<Marker>(
        ComponentHooks::new().with_on_remove(panic_during_remove),
    );
}

#[test]
fn unload_hook_panic_preserves_ownership_for_handoff_and_plan_retry() {
    let mut world = World::new();
    let mut runtime = runtime();
    let sector = SectorId::new(0, 0, 0, 0);
    let mut host = SceneEcsHost::new(&mut world, Provider::default());
    let root = load_bound_cell(&mut runtime, &mut host, sector, 41);
    let binding = host.residency_binding(41).unwrap();
    let plan = runtime.tick(
        &[],
        &[],
        &[ResidencyRecord::new(41, ResidencyState::TierA, 10, 1, 1, 1)],
        2,
        budget(),
    );
    assert_eq!(
        plan.stream.actions,
        alloc::vec![StreamAction::Unload(sector)]
    );
    assert!(matches!(
        plan.residency.actions.as_slice(),
        [ResidencyAction::Demote { id: 41, .. }]
    ));
    install_despawn_panic(&mut host, root, 41);
    let panic = catch_unwind(AssertUnwindSafe(|| {
        runtime.apply_scene_plan(&plan, &mut host)
    }))
    .expect_err("the component hook panic must propagate");
    assert_eq!(
        panic.downcast_ref::<&str>(),
        Some(&"scene despawn hook panic")
    );
    assert!(host.world().contains(root));
    assert!(host.loaded_cell(sector).is_some(), "unload lost its token");
    assert_eq!(host.residency_binding(41), Some(binding));
    assert!(
        host.world()
            .resource::<DespawnPanicProbe>()
            .ownership_visible_in_hook
    );

    let provider = host.into_provider();
    let mut replacement = SceneEcsHost::new(&mut world, provider);
    assert_eq!(replacement.loaded_cell(sector).unwrap().root, Some(root));
    assert_eq!(replacement.residency_binding(41), Some(binding));
    replacement
        .world_mut()
        .register_component_hooks::<Marker>(ComponentHooks::new());
    runtime.apply_scene_plan(&plan, &mut replacement).unwrap();
    assert!(!replacement.world().contains(root));
    assert_eq!(replacement.loaded_cell_count(), 0);
    assert_eq!(replacement.residency_binding(41), None);
    assert!(replacement.provider().events.is_empty());
}

#[test]
fn shutdown_hook_panic_preserves_failed_and_pending_cells_for_retry() {
    let mut world = World::new();
    let unrelated = world.spawn(Marker(17));
    let mut host = SceneEcsHost::new(&mut world, Provider::default());
    let mut sectors = [
        SectorId::new(0, 0, 0, 0),
        SectorId::new(0, 1, 0, 0),
        SectorId::new(0, 2, 0, 0),
    ];
    sectors.sort();
    let roots: alloc::vec::Vec<_> = sectors
        .iter()
        .enumerate()
        .map(|(index, &sector)| {
            SceneHost::load_cell(&mut host, sector).unwrap();
            host.bind_residency(
                index as u64,
                ResidencyBinding {
                    sector,
                    local: LocalEntityId::new(0),
                },
            )
            .unwrap();
            host.loaded_cell(sector).unwrap().root.unwrap()
        })
        .collect();
    let orphan = ResidencyBinding {
        sector: SectorId::new(0, 99, 0, 0),
        local: LocalEntityId::new(0),
    };
    host.bind_residency(99, orphan).unwrap();
    install_despawn_panic(&mut host, roots[1], 1);
    let panic = catch_unwind(AssertUnwindSafe(|| host.shutdown()))
        .expect_err("the component hook panic must propagate");
    assert_eq!(
        panic.downcast_ref::<&str>(),
        Some(&"scene despawn hook panic")
    );
    assert!(!host.world().contains(roots[0]));
    assert!(host.loaded_cell(sectors[0]).is_none());
    assert_eq!(host.residency_binding(0), None);
    for index in 1..3 {
        assert!(host.world().contains(roots[index]));
        assert!(
            host.loaded_cell(sectors[index]).is_some(),
            "shutdown lost pending token"
        );
        assert_eq!(
            host.residency_binding(index as u64).unwrap().sector,
            sectors[index]
        );
    }
    assert_eq!(host.loaded_cell_count(), 2);
    assert_eq!(host.residency_binding(99), Some(orphan));
    assert!(
        host.world()
            .resource::<DespawnPanicProbe>()
            .ownership_visible_in_hook
    );
    drop(host);

    let mut replacement = SceneEcsHost::new(&mut world, Provider::default());
    assert_eq!(replacement.loaded_cell_count(), 2);
    replacement
        .world_mut()
        .register_component_hooks::<Marker>(ComponentHooks::new());
    assert_eq!(replacement.shutdown(), 2);
    assert_eq!(replacement.shutdown(), 0);
    for id in [0, 1, 2, 99] {
        assert_eq!(replacement.residency_binding(id), None);
    }
    assert_eq!(replacement.world().entity_count(), 1);
    assert_eq!(
        replacement.world().get::<Marker>(unrelated),
        Some(&Marker(17))
    );
}
