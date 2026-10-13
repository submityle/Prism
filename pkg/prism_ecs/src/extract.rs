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

use alloc::vec::Vec;

use crate::component::Component;
use crate::entity::Entity;
use crate::query::{QueryData, QueryFilter, ReadOnlyQueryData};
use crate::resource::Resource;
use crate::world::World;

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
