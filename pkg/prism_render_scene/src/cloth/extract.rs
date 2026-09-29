//! Render-world extraction of main-world cloth garments.
//!
//! This system runs in [`bevy_render::ExtractSchedule`] and mirrors every
//! main-world [`ClothGarment`] into the render-world [`ExtractedCloth`]
//! resource. It follows the same clear-then-refill pattern the lighting extract
//! uses ([`crate::lighting`]): the resource is emptied first and then
//! repopulated from the current main-world entities, so a garment that despawns
//! simply stops contributing and the prepare stage always sees the exact set of
//! live garments for the frame.
//!
//! The extract stage deliberately clones the authored `CPU` state rather than
//! borrowing across worlds: the render world outlives the main-world query for
//! the rest of the frame, and the owned snapshot lets the device-side prepare
//! stage borrow it again through [`ClothGarment::as_solve_input`] without any
//! cross-world lifetime.

use bevy_ecs::prelude::*;
use bevy_render::Extract;

use super::garment::{ClothGarment, ExtractedCloth};

/// Snapshots every main-world [`ClothGarment`] into [`ExtractedCloth`].
///
/// Clears the resource and refills it from the current main-world query each
/// frame, mirroring the lighting extract's rebuild pattern so despawned
/// garments drop out cleanly and no stale piece survives into the prepare
/// stage.
pub(crate) fn extract_cloth_garments(
    mut extracted: ResMut<ExtractedCloth>,
    garments: Extract<Query<&ClothGarment>>,
) {
    extracted.garments.clear();
    for garment in garments.iter() {
        extracted.garments.push(garment.clone());
    }
}
