//! Render-world extraction of main-world water bodies.
//!
//! This system runs in [`bevy_render::ExtractSchedule`] and mirrors every
//! main-world [`WaterBody`] into the render-world [`ExtractedWater`] resource.
//! It follows the same clear-then-refill pattern the cloth and lighting
//! extracts use: the resource is emptied first and then repopulated from the
//! current main-world entities, so a body that despawns simply stops
//! contributing and the prepare stage always sees the exact set of live bodies
//! for the frame.
//!
//! The extract stage deliberately clones the authored `CPU` state rather than
//! borrowing across worlds: the render world outlives the main-world query for
//! the rest of the frame, and the owned snapshot lets the device-side prepare
//! stage borrow it again through [`WaterBody::as_upload`] and
//! [`WaterBody::as_extract`] without any cross-world lifetime.

use bevy_ecs::prelude::*;
use bevy_render::Extract;

use super::body::{ExtractedWater, WaterBody};

/// Snapshots every main-world [`WaterBody`] into [`ExtractedWater`].
///
/// Clears the resource and refills it from the current main-world query each
/// frame, mirroring the cloth extract's rebuild pattern so despawned bodies
/// drop out cleanly and no stale body survives into the prepare stage.
pub(crate) fn extract_water_bodies(
    mut extracted: ResMut<ExtractedWater>,
    bodies: Extract<Query<&WaterBody>>,
) {
    extracted.bodies.clear();
    for body in bodies.iter() {
        extracted.bodies.push(body.clone());
    }
}
