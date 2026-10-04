//! Per-view resident visible-point list produced by the world-space `ReSTIR`
//! visible-point producer pass and consumed by the inject pass.
//!
//! The producer runs one compute invocation per screen *tile* (a
//! [`VISIBLE_POINTS_TILE_SIZE`] x [`VISIBLE_POINTS_TILE_SIZE`] framebuffer
//! block), reconstructs the tile-centre shading point from the SSR prepass
//! depth + packed normal and appends it to this list in the frozen
//! [`super::super::abi::GpuWorldRestirInjectPoint`] layout. The inject pass then
//! hashes each record into its `SHARC` world cell and claims a reservoir slot,
//! so the list is the screen-space bridge from the frame's visible geometry to
//! the resident world-space reservoir table.
//!
//! The buffer is sized to the derived tile grid of the SSR prepass framebuffer,
//! so the only reallocation trigger is a screen-size change; a steady-state
//! frame at the same resolution reuses the resident buffer untouched (the
//! producer overwrites every record each frame). The subsystem is opt-in and
//! shares the SSR prepass inputs, so the list exists exactly when both
//! [`PrismWorldRestirSettings::enabled`] holds and the view carries a resident
//! [`ViewSsrTextures`] prepass.

use bevy_ecs::prelude::*;
use bevy_math::UVec2;
use bevy_render::{
    camera::ExtractedCamera,
    render_resource::{Buffer, BufferDescriptor, BufferUsages},
    renderer::RenderDevice,
};

use super::super::super::ssr::ViewSsrTextures;
use super::super::abi::WORLD_RESTIR_INJECT_POINT_STRIDE;
use super::super::settings::PrismWorldRestirSettings;
use super::abi::VISIBLE_POINTS_TILE_SIZE;

/// Per-view resident visible-point list, present only while the producer pass
/// is enabled and the view carries a resident SSR prepass.
#[derive(Component)]
pub(crate) struct ViewWorldRestirVisiblePoints {
    /// The per-frame visible-point list: `point_count` records of the frozen
    /// [`WORLD_RESTIR_INJECT_POINT_STRIDE`]-byte inject-point layout. Bound
    /// read-write by the producer (`@binding(2)`) and read-only by the inject
    /// pass (`@binding(0)`).
    buffer: Buffer,
    /// Number of visible points (= tiles) produced this frame, the producer
    /// dispatch's per-invocation bounds check and the inject dispatch extent.
    point_count: u32,
    /// Derived tile grid `(ceil(width / tile), ceil(height / tile))`; the
    /// producer dispatches one workgroup block per tile.
    tiles: UVec2,
    /// SSR prepass framebuffer extent this list was sized for; the only
    /// reallocation trigger.
    screen: UVec2,
}

impl ViewWorldRestirVisiblePoints {
    /// The resident visible-point storage buffer.
    pub(crate) fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Number of visible points produced this frame (one per tile).
    pub(crate) fn point_count(&self) -> u32 {
        self.point_count
    }

    /// Derived tile grid the producer dispatches over.
    pub(crate) fn tiles(&self) -> UVec2 {
        self.tiles
    }

    /// SSR prepass framebuffer extent in texels.
    pub(crate) fn screen(&self) -> UVec2 {
        self.screen
    }
}

/// Derives the tile grid for a framebuffer of `screen` texels: one tile per
/// [`VISIBLE_POINTS_TILE_SIZE`] x [`VISIBLE_POINTS_TILE_SIZE`] block, rounded up
/// so a partial edge tile still produces a point.
fn tile_grid(screen: UVec2) -> UVec2 {
    UVec2::new(
        screen.x.div_ceil(VISIBLE_POINTS_TILE_SIZE),
        screen.y.div_ceil(VISIBLE_POINTS_TILE_SIZE),
    )
}

/// (Re)allocates [`ViewWorldRestirVisiblePoints`] for every camera view that
/// carries a resident SSR prepass while the producer is enabled, and removes it
/// otherwise.
///
/// Gated on [`PrismWorldRestirSettings::enabled`] and the presence of
/// [`ViewSsrTextures`] (the producer reconstructs its points from the SSR
/// prepass depth + packed normal). The buffer is sized to the SSR framebuffer's
/// tile grid; a steady-state frame at the same resolution reuses the resident
/// buffer untouched, and only a screen-size change triggers a realloc.
pub(crate) fn prepare_world_restir_visible_points(
    mut commands: Commands,
    settings: Res<PrismWorldRestirSettings>,
    device: Res<RenderDevice>,
    mut views: Query<
        (
            Entity,
            &ViewSsrTextures,
            Option<&mut ViewWorldRestirVisiblePoints>,
        ),
        With<ExtractedCamera>,
    >,
) {
    for (entity, ssr, existing) in &mut views {
        if !settings.enabled {
            if existing.is_some() {
                commands
                    .entity(entity)
                    .remove::<ViewWorldRestirVisiblePoints>();
            }
            continue;
        }

        let screen = ssr.size;
        // Steady state: an existing list already sized for this framebuffer is
        // reused as-is (the producer overwrites every record each frame).
        if let Some(points) = existing
            && points.screen == screen
        {
            continue;
        }

        let tiles = tile_grid(screen);
        let point_count = tiles.x * tiles.y;
        let size = u64::from(point_count.max(1)) * WORLD_RESTIR_INJECT_POINT_STRIDE;

        // STORAGE: written read-write by the producer, read read-only by the
        // inject pass. COPY_DST so a host-side clear can be scheduled if ever
        // needed; the producer writes every record each frame, so no clear is
        // required for correctness.
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("prism world-space ReSTIR visible points"),
            size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        commands
            .entity(entity)
            .insert(ViewWorldRestirVisiblePoints {
                buffer,
                point_count,
                tiles,
                screen,
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_grid_rounds_partial_edge_tiles_up() {
        // Exact multiples map one-to-one; a partial edge rounds up so the last
        // strip of pixels still produces a tile centre point.
        assert_eq!(
            tile_grid(UVec2::new(
                4 * VISIBLE_POINTS_TILE_SIZE,
                2 * VISIBLE_POINTS_TILE_SIZE
            )),
            UVec2::new(4, 2)
        );
        assert_eq!(
            tile_grid(UVec2::new(VISIBLE_POINTS_TILE_SIZE + 1, 1)),
            UVec2::new(2, 1)
        );
    }

    #[test]
    fn tile_grid_1080p_stays_under_the_default_capacity() {
        // ~129600 points @1080p for a tile edge of 4, under the 131072 default
        // reservoir-table capacity so the open-addressed hash grid never
        // thrashes.
        let tiles = tile_grid(UVec2::new(1920, 1080));
        assert_eq!(tiles, UVec2::new(480, 270));
        assert!(tiles.x * tiles.y <= 131_072);
    }
}
