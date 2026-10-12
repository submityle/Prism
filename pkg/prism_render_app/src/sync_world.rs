//! The render world's link back to the simulation: [`MainEntity`] and the sets
//! the renderer keeps of them.
//!
//! The two worlds have independent entity allocators — extraction spawns a fresh
//! entity per extracted row (`prism_ecs::extract::extract_components`), and ids
//! are deliberately *not* forced to agree, because coupling two allocators buys
//! nothing. So a render entity that must find its simulation counterpart again
//! — to resize its GPU buffer, drop a stale binding, or remove a queued phase
//! item — carries that counterpart as a component. Bevy spells this
//! `bevy_render::sync_world::MainEntity`; Prism keeps the name and the `id()`
//! accessor, so a ported system's body does not change.
//!
//! [`MainEntityHashSet`] is the same idea as a set, which is what the queue
//! stages actually need: they diff *this* frame's queued entities against the
//! previous frame's to find the items that stopped being queued, and they look up
//! membership by simulation entity. Hashing is on the entity's identity
//! `(index, generation)`, so a recycled slot is a different entity and never
//! aliases a stale one.

use std::collections::HashSet;

use prism_ecs::prelude::{Component, Entity};
use prism_ecs::world::World;

/// The simulation entity a render entity was extracted from.
///
/// Stored as a component on the render entity by [`tag_extracted_entities`].
/// Derefs to the [`Entity`], mirroring the source type so a ported
/// `*main_entity` keeps compiling.
#[repr(transparent)]
#[derive(Component, Clone, Copy, Debug, Eq, Hash, PartialEq, PartialOrd, Ord)]
pub struct MainEntity(Entity);

impl MainEntity {
    /// The simulation entity this render entity projects.
    #[must_use]
    pub const fn id(self) -> Entity {
        self.0
    }
}

impl From<Entity> for MainEntity {
    fn from(entity: Entity) -> Self {
        Self(entity)
    }
}

impl From<MainEntity> for Entity {
    fn from(main: MainEntity) -> Self {
        main.0
    }
}

impl core::ops::Deref for MainEntity {
    type Target = Entity;

    fn deref(&self) -> &Entity {
        &self.0
    }
}

/// A set of simulation entities, hashed by entity identity.
///
/// The renderer's per-frame bookkeeping: insert what is queued this frame,
/// compare against last frame's set, then keep this one. `drain` is how a stage
/// hands the previous frame's leftovers to the phases for removal.
#[derive(Clone, Debug, Default)]
pub struct MainEntityHashSet {
    entities: HashSet<MainEntity>,
}

impl MainEntityHashSet {
    /// An empty set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts `entity`, returning whether it was not already present.
    pub fn insert(&mut self, entity: MainEntity) -> bool {
        self.entities.insert(entity)
    }

    /// Removes `entity`, returning whether it was present.
    pub fn remove(&mut self, entity: &MainEntity) -> bool {
        self.entities.remove(entity)
    }

    /// Whether `entity` is in the set.
    #[must_use]
    pub fn contains(&self, entity: &MainEntity) -> bool {
        self.entities.contains(entity)
    }

    /// Empties the set.
    pub fn clear(&mut self) {
        self.entities.clear();
    }

    /// Removes every entity, handing them to the caller.
    pub fn drain(&mut self) -> impl Iterator<Item = MainEntity> + '_ {
        self.entities.drain()
    }

    /// Iterates the entities without removing them.
    pub fn iter(&self) -> impl Iterator<Item = MainEntity> + '_ {
        self.entities.iter().copied()
    }

    /// The number of entities in the set.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entities.len()
    }

    /// Whether the set is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }
}

impl FromIterator<MainEntity> for MainEntityHashSet {
    fn from_iter<I: IntoIterator<Item = MainEntity>>(iter: I) -> Self {
        Self {
            entities: iter.into_iter().collect(),
        }
    }
}

impl Extend<MainEntity> for MainEntityHashSet {
    fn extend<I: IntoIterator<Item = MainEntity>>(&mut self, iter: I) {
        self.entities.extend(iter);
    }
}

/// Tags each extracted render entity with the simulation entity it projects.
///
/// The renderer's half of the extract contract: `extract_components` returns the
/// `(main, secondary)` pairs it spawned, and this turns each pair into a
/// [`MainEntity`] component on the render entity — the link a later frame uses to
/// find that entity again. The returned set is the frame's membership, which a
/// queue stage diffs against the previous frame's to find what went away.
///
/// A pair whose render entity is no longer live is skipped (the set is the
/// authority on what was actually tagged), never panicked on.
pub fn tag_extracted_entities(
    render: &mut World,
    pairs: impl IntoIterator<Item = (Entity, Entity)>,
) -> MainEntityHashSet {
    let mut tagged = MainEntityHashSet::new();
    for (main, render_entity) in pairs {
        let main_entity = MainEntity::from(main);
        if render.insert(render_entity, main_entity) {
            tagged.insert(main_entity);
        }
    }
    tagged
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use prism_ecs::extract::{ExtractComponent, extract_components};
    use prism_ecs::prelude::{Component, Entity, With};
    use prism_ecs::world::World;

    use super::{MainEntity, MainEntityHashSet, tag_extracted_entities};

    #[test]
    fn main_entity_round_trips_the_simulation_entity() {
        let mut world = World::new();
        let entity = world.spawn(());
        let main = MainEntity::from(entity);
        assert_eq!(main.id(), entity);
        assert_eq!(Entity::from(main), entity);
        assert_eq!(*main, entity, "the component derefs to its entity");
    }

    #[test]
    fn membership_is_exact_across_generations() {
        let mut world = World::new();
        let live = world.spawn(());
        assert!(world.despawn(live), "the fixture entity is live");
        // The same slot comes back with a new generation: a different entity.
        let reused = world.spawn(());
        assert_eq!(
            live.index(),
            reused.index(),
            "the fixture recycles the slot, so the test really exercises generations"
        );
        assert_ne!(live, reused, "the recycled slot is a new entity");

        let mut set = MainEntityHashSet::new();
        assert!(set.insert(MainEntity::from(live)), "first insert is new");
        assert!(!set.insert(MainEntity::from(live)), "second insert is not");
        assert!(set.contains(&MainEntity::from(live)));
        assert!(
            !set.contains(&MainEntity::from(reused)),
            "a recycled slot is a different entity, never an alias"
        );
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn drain_hands_over_the_previous_frame_and_empties_the_set() {
        let mut world = World::new();
        let a = world.spawn(());
        let b = world.spawn(());
        let mut set = MainEntityHashSet::from_iter([MainEntity::from(a)]);
        set.insert(MainEntity::from(b));
        assert_eq!(set.len(), 2);

        let mut drained: Vec<MainEntity> = set.drain().collect();
        drained.sort();
        let mut expected = alloc::vec![MainEntity::from(a), MainEntity::from(b)];
        expected.sort();
        assert_eq!(drained, expected, "every entity was handed over");
        assert!(set.is_empty(), "draining empties the set");
    }

    /// The pair `extract_components` returns is exactly what the mapping needs:
    /// an extraction followed by tagging leaves every render entity pointing at
    /// its simulation entity, and the returned set is the frame's membership.
    #[test]
    fn extracted_rows_are_tagged_with_their_main_entity() {
        #[derive(Clone, Copy, Debug, PartialEq)]
        struct Health(i32);
        impl Component for Health {}

        #[derive(Clone, Copy, Debug, PartialEq)]
        struct HealthBar(i32);
        impl Component for HealthBar {}

        impl ExtractComponent for Health {
            type Out = HealthBar;
            type QueryData = &'static Health;
            type QueryFilter = With<HealthBar>;

            fn extract(health: &Health) -> Option<Self::Out> {
                Some(HealthBar(health.0))
            }
        }

        let mut main = World::new();
        let mut render = World::new();
        // The filter selects one of the two rows, so the mapping must follow the
        // rows that were actually extracted.
        let kept = main.spawn((Health(7), HealthBar(0)));
        let _skipped = main.spawn((Health(9),));

        let pairs = extract_components::<Health>(&mut main, &mut render);
        assert_eq!(pairs.len(), 1, "only the filtered row is extracted");
        assert_eq!(pairs[0].0, kept, "the pair names the simulation entity");
        let render_entity = pairs[0].1;

        let tagged = tag_extracted_entities(&mut render, pairs.clone());
        assert_eq!(tagged.len(), 1);
        assert!(
            tagged.contains(&MainEntity::from(kept)),
            "the extracted row's simulation entity is in the frame's set"
        );

        // The link is a real component on the render entity, not just a set
        // entry: a later frame can query it back.
        let state = render.query::<(&MainEntity, &HealthBar)>();
        let (main_entity, bar) = state
            .get(&render, render_entity)
            .expect("the render entity carries its main-entity link");
        assert_eq!(main_entity.id(), kept, "the render entity points home");
        assert_eq!(bar.0, 7, "the projected value rode along");
    }
}
