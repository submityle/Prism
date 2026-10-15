//! The extraction contract: publishing a *projection* of the simulation world
//! into a secondary (render) world (design §23.5).
//!
//! The engine keeps two worlds: the authoritative simulation [`World`] and one
//! or more secondary worlds (the render world). They are bridged by an
//! **extract** step that runs once per frame, reads the simulation world, and
//! writes into the secondary world. Two properties make that seam safe to build
//! on:
//!
//! - **One-way and read-only.** Extraction never mutates simulation state — the
//!   main world is handed to these runners by shared reference and every
//!   extractor query is [`ReadOnlyQueryData`], so change detection does not even
//!   see a read as a write. Nothing here can feed back into the simulation.
//! - **Projection, not coupling.** A secondary world depends on the *extraction
//!   traits*, never on the simulation's internal layout: [`ExtractResource`]
//!   turns one main resource into one secondary resource, and
//!   [`ExtractComponent`] turns each matching main entity into one secondary
//!   entity carrying `Out`. A renderer that follows the traits keeps compiling
//!   when the simulation reorganises its own components.
//!
//! # Entity identity across worlds
//!
//! The secondary world has its own entity allocator, so the runner spawns a
//! **fresh** entity for each extracted row and returns the `(main, secondary)`
//! pairs in match order. A consumer that needs to find its secondary entity
//! again (resize a render buffer, despawn a removed object) stores that mapping
//! — typically as a marker component on the secondary entity, the way a
//! render-entity link works. Ids are deliberately *not* forced to match: doing
//! so would couple the two allocators for no benefit.
//!
//! [`ComponentExtraction`] owns the mapping for a continuing stream: it
//! updates existing projections, retires missing ones, and repairs externally
//! removed projections. Temporal update filters use a separate membership
//! filter so unchanged sources stay in the secondary world.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicUsize, Ordering};

use crate::change::Tick;
use crate::component::Component;
use crate::entity::Entity;
use crate::query::{ArchetypalFilter, QueryData, QueryFilter, QueryState, ReadOnlyQueryData};
use crate::resource::Resource;
use crate::world::{World, WorldId};

/// A resource whose value is published into a secondary world (design §23.5).
///
/// The extraction is a projection: `Out` may be a different type that carries
/// only what the secondary world needs (a renderer should not have to depend on
/// the simulation's authoring types).
pub trait ExtractResource: Resource {
    /// The value the secondary world receives.
    type Out: Resource;

    /// Produce the extracted value, or `None` to leave the secondary world
    /// untouched this frame (e.g. the main resource is absent or paused).
    fn extract(&self, main: &World) -> Option<Self::Out>;
}

/// A component whose value is published into a secondary world (design §23.5).
///
/// `QueryData`/`QueryFilter` name *what the extractor reads* — usually `&Self`
/// plus whatever neighbours the projection needs (a world transform, a mesh
/// handle) — so an extractor can be as narrow or as wide as it needs without
/// the secondary world ever naming a simulation component itself.
pub trait ExtractComponent: Component {
    /// The component the secondary world receives for each matching entity.
    type Out: Component;
    /// The components the extractor reads.
    type QueryData: ReadOnlyQueryData;
    /// The filter narrowing which rows are extracted.
    type QueryFilter: QueryFilter;

    /// Extract one matched row, or `None` to skip it this frame.
    fn extract(item: <Self::QueryData as QueryData>::Item<'_>) -> Option<Self::Out>;
}

/// Publish `R`'s projection into `secondary` (design §23.5).
///
/// Returns whether a value was written: `false` when the main world has no `R`,
/// or when the extractor declined this frame (so the secondary world keeps its
/// previous value).
pub fn extract_resource<R: ExtractResource>(main: &World, secondary: &mut World) -> bool {
    match main
        .get_resource::<R>()
        .and_then(|value| value.extract(main))
    {
        Some(out) => {
            secondary.insert_resource(out);
            true
        }
        None => false,
    }
}

/// Publish `C`'s projection for every matching entity into `secondary` (design
/// §23.5), returning the `(main, secondary)` entity pairs it created.
///
/// Rows are visited through the ordinary query iterator, so the per-row
/// `Added`/`Changed`/sparse gate applies exactly as it does anywhere else. Rows
/// whose extractor returns `None` are skipped and produce no secondary entity.
pub fn extract_components<C: ExtractComponent>(
    main: &mut World,
    secondary: &mut World,
) -> Vec<(Entity, Entity)> {
    let state = main.query_filtered::<C::QueryData, C::QueryFilter>();

    // Read the simulation world first, then write the secondary world: the two
    // borrows never overlap, and nothing extracted can observe a half-updated
    // secondary world.
    let mut extracted: Vec<(Entity, C::Out)> = Vec::new();
    {
        let mut iter = state.iter(main);
        while let Some((entity, item)) = iter.next_with_entity() {
            if let Some(out) = C::extract(item) {
                extracted.push((entity, out));
            }
        }
    }

    let mut mapping = Vec::with_capacity(extracted.len());
    for (main_entity, out) in extracted {
        let secondary_entity = secondary.spawn(out);
        mapping.push((main_entity, secondary_entity));
    }
    mapping
}

/// Identity of an entity managed by a continuing extraction stream.
///
/// Ownership is checked before updates/deletion, including after rollback or
/// entity-slot recycling. Independent streams can share a secondary world.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExtractedEntity {
    source: Entity,
    source_world: WorldId,
    stream: usize,
}

impl Component for ExtractedEntity {}

impl ExtractedEntity {
    /// The simulation entity projected into this secondary entity.
    pub fn source(&self) -> Entity {
        self.source
    }

    /// The simulation world the source entity belongs to.
    pub fn source_world(&self) -> WorldId {
        self.source_world
    }
}

/// Work performed by one successful extraction pass.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExtractionStats {
    /// New secondary entities, including repaired external deletions.
    pub spawned: usize,
    /// Existing secondary entities whose projection was overwritten.
    pub updated: usize,
    /// Owned secondary entities removed because their source left the stream.
    pub despawned: usize,
    /// Existing projections retained without invoking the extractor.
    pub retained: usize,
}

enum Projection<T> {
    Keep,
    Write(T),
    Remove,
}

/// Persistent, world-bound component extraction with stable secondary ids.
///
/// `M` describes continuing membership; `C::QueryFilter` selects updates among
/// those members. With a temporal update filter such as `Changed<T>`, use
/// [`with_membership`](Self::with_membership) and a non-temporal `M` (for
/// example `With<Visible>`). New members are projected even when they do not
/// pass the temporal filter. Returning `None` withdraws a projection.
///
/// Membership discovery visits matching rows; only updates/new members invoke
/// the extractor. This does not claim O(changes) membership discovery. Values
/// are staged before publishing, so an extractor panic leaves the secondary
/// world and successful-tick cursor untouched. Component hooks during
/// publication have ordinary World-operation panic semantics.
pub struct ComponentExtraction<
    C: ExtractComponent,
    M: QueryFilter = <C as ExtractComponent>::QueryFilter,
> {
    main_world: WorldId,
    secondary_world: WorldId,
    stream: usize,
    membership: QueryState<C::QueryData, M>,
    updates: QueryState<C::QueryData, (M, C::QueryFilter)>,
    owners: QueryState<&'static ExtractedEntity>,
    mapping: BTreeMap<Entity, Entity>,
    last_run: Tick,
    main_generation: u64,
    secondary_generation: u64,
}

impl<C: ExtractComponent> ComponentExtraction<C, C::QueryFilter>
where
    C::QueryFilter: ArchetypalFilter,
{
    /// Create a stream whose stable filter also defines its membership.
    /// Temporal filters require [`with_membership`](Self::with_membership).
    pub fn new(main: &mut World, secondary: &mut World) -> Self {
        Self::with_membership(main, secondary)
    }
}

impl<C: ExtractComponent, M: QueryFilter> ComponentExtraction<C, M> {
    /// Create a stream with explicit continuing membership filter `M`.
    ///
    /// `M` must express presence/visibility, rather than a transient change
    /// window. Sparse membership is supported. The owner marker is registered
    /// for snapshots so secondary-world rollback can recover its mappings.
    pub fn with_membership(main: &mut World, secondary: &mut World) -> Self {
        static NEXT_STREAM: AtomicUsize = AtomicUsize::new(1);
        let stream = NEXT_STREAM
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .expect("extraction stream identity exhausted");
        secondary.register_snapshot_component::<ExtractedEntity>();
        Self {
            main_world: main.id(),
            secondary_world: secondary.id(),
            stream,
            membership: main.query_filtered::<C::QueryData, M>(),
            updates: main.query_filtered::<C::QueryData, (M, C::QueryFilter)>(),
            owners: secondary.query::<&ExtractedEntity>(),
            mapping: BTreeMap::new(),
            last_run: Tick::ZERO,
            main_generation: main.archetype_generation(),
            secondary_generation: secondary.archetype_generation(),
        }
    }

    /// Find a mapped secondary entity. External deletions are repaired on sync.
    pub fn entity(&self, source: Entity) -> Option<Entity> {
        self.mapping.get(&source).copied()
    }

    /// Iterate the mapping in stable source-entity order.
    pub fn entities(&self) -> impl Iterator<Item = (Entity, Entity)> + '_ {
        self.mapping
            .iter()
            .map(|(&source, &target)| (source, target))
    }

    fn owns(&self, secondary: &World, source: Entity, target: Entity) -> bool {
        secondary
            .get::<ExtractedEntity>(target)
            .is_some_and(|owner| {
                owner.stream == self.stream
                    && owner.source_world == self.main_world
                    && owner.source == source
            })
    }

    /// Publish the next projection, preserving ids and retiring departed sources.
    ///
    /// Panics before publication when either world differs from the bound pair.
    /// Rollback forces reprojection; secondary rollback also recovers mappings
    /// from owner markers before retiring stale projections.
    pub fn sync(&mut self, main: &World, secondary: &mut World) -> ExtractionStats {
        assert_eq!(
            main.id(),
            self.main_world,
            "extraction source world mismatch"
        );
        assert_eq!(
            secondary.id(),
            self.secondary_world,
            "extraction secondary world mismatch"
        );
        let secondary_restored = secondary.archetype_generation() != self.secondary_generation;
        let force = main.archetype_generation() != self.main_generation || secondary_restored;
        // Keep recovered mappings local until every extractor call succeeds.
        let mut recovered = BTreeMap::new();
        let mapping = if secondary_restored {
            let mut owners = self.owners.iter(secondary);
            while let Some((target, owner)) = owners.next_with_entity() {
                if owner.stream == self.stream && owner.source_world == self.main_world {
                    recovered.insert(owner.source, target);
                }
            }
            &recovered
        } else {
            &self.mapping
        };
        let mut staged = BTreeMap::new();
        let mut members = self.membership.iter(main);
        while let Some((source, item)) = members.next_with_entity() {
            let valid = mapping.get(&source).is_some_and(|&target| {
                self.owns(secondary, source, target) && secondary.has::<C::Out>(target)
            });
            let projection = if force || !valid {
                C::extract(item).map_or(Projection::Remove, Projection::Write)
            } else {
                Projection::Keep
            };
            staged.insert(source, projection);
        }
        let mut updates = self
            .updates
            .iter_with_ticks(main, self.last_run, main.change_tick());
        while let Some((source, item)) = updates.next_with_entity() {
            if matches!(staged.get(&source), Some(Projection::Keep)) {
                staged.insert(
                    source,
                    C::extract(item).map_or(Projection::Remove, Projection::Write),
                );
            }
        }
        let mut stats = ExtractionStats::default();
        let mut next = BTreeMap::new();
        for (&source, &target) in mapping {
            if !matches!(
                staged.get(&source),
                Some(Projection::Keep | Projection::Write(_))
            ) && self.owns(secondary, source, target)
                && secondary.despawn(target)
            {
                stats.despawned += 1;
            }
        }
        for (source, projection) in staged {
            match projection {
                Projection::Keep => {
                    next.insert(source, mapping[&source]);
                    stats.retained += 1;
                }
                Projection::Write(out) => {
                    let target = match mapping
                        .get(&source)
                        .copied()
                        .filter(|&target| self.owns(secondary, source, target))
                    {
                        Some(target) => {
                            secondary.insert(target, out);
                            stats.updated += 1;
                            target
                        }
                        None => {
                            stats.spawned += 1;
                            secondary.spawn((
                                out,
                                ExtractedEntity {
                                    source,
                                    source_world: self.main_world,
                                    stream: self.stream,
                                },
                            ))
                        }
                    };
                    next.insert(source, target);
                }
                Projection::Remove => {}
            }
        }
        self.mapping = next;
        self.last_run = main.change_tick();
        self.main_generation = main.archetype_generation();
        self.secondary_generation = secondary.archetype_generation();
        stats
    }

    /// Despawn this stream's projections, preserving unrelated entities.
    /// The next sync bootstraps current members again.
    pub fn clear(&mut self, secondary: &mut World) -> usize {
        assert_eq!(
            secondary.id(),
            self.secondary_world,
            "extraction secondary world mismatch"
        );
        let mut iter = self.owners.iter(secondary);
        let mut owned = Vec::new();
        while let Some((entity, owner)) = iter.next_with_entity() {
            if owner.stream == self.stream && owner.source_world == self.main_world {
                owned.push(entity);
            }
        }
        let removed = owned
            .into_iter()
            .filter(|&entity| secondary.despawn(entity))
            .count();
        self.mapping.clear();
        self.last_run = Tick::ZERO;
        removed
    }
}
