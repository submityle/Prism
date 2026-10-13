//! The asset stage: registering a family, and the system that syncs it.

use core::marker::PhantomData;

use prism_app::App;
use prism_ecs::prelude::{Res, ResMut, Resource};
use prism_ecs::schedule::{IntoSystemConfigs, Schedules};
use prism_render_driver::{FrameSync, Instance};

use crate::label::{Render, RenderApp};
use crate::runtime_resource::RenderRuntimeResource;
use crate::sets::RenderSystems;

use super::extract::{extract_assets, ExtractedAssets};
use super::prepare::{prepare_extracted_assets, AssetSyncReport};
use super::table::RenderAssets;
use super::RenderAsset;

/// The last sync pass's report for one asset family.
///
/// Per-family, because "how many meshes did we prepare this frame" is a
/// different question from "how many textures", and a renderer that silently
/// stops preparing a family should be visible rather than inferred.
pub struct AssetSyncStats<A: RenderAsset> {
    /// What the most recent pass did.
    pub last: AssetSyncReport,
    family: PhantomData<fn() -> A>,
}

impl<A: RenderAsset> Default for AssetSyncStats<A> {
    fn default() -> Self {
        Self {
            last: AssetSyncReport::default(),
            family: PhantomData,
        }
    }
}

// Manual impl for the same reason `RenderRuntimeResource` has one: a generic
// resource cannot use the derive without adding bounds the family type does not
// need.
impl<A: RenderAsset> Resource for AssetSyncStats<A> {}

/// Prepares the frame's extracted assets of one family.
///
/// Registered in [`RenderSystems::PrepareAssets`], after the extract carried the
/// changed assets across and before anything queues against them. A frame with
/// nothing extracted does no work and leaves the last report standing, so the
/// stats stay meaningful across quiet frames.
pub fn prepare_assets<I, A>(
    mut extracted: ResMut<ExtractedAssets<A>>,
    mut prepared: ResMut<RenderAssets<A>>,
    mut stats: ResMut<AssetSyncStats<A>>,
    runtime: Res<RenderRuntimeResource<I>>,
) where
    I: Instance + Send + Sync + 'static,
    I::Adapter: Send + Sync + 'static,
    I::Device: Send + Sync + 'static,
    I::Queue: Send + Sync + FrameSync + 'static,
    I::Surface: Send + Sync + 'static,
    A: RenderAsset,
{
    if extracted.is_empty() {
        return;
    }
    stats.last = prepare_extracted_assets::<A>(
        &mut extracted,
        &mut prepared,
        runtime.runtime().device(),
        runtime.runtime().queue(),
    );
}

/// Registers one render-asset family with the app.
///
/// Adds the family's extract step (which copies changed assets into the render
/// world), its frame resources, and the prepare system in
/// [`RenderSystems::PrepareAssets`]. Call it for every `RenderAsset` the
/// renderer draws: this is the `RenderAssetPlugin<A>` of the Prism renderer.
///
/// Registering a family twice is a no-op: the spine registers the mesh family
/// itself (its frame draws meshes), and a caller that registers it again — or a
/// plugin added twice — must not end up with the prepare system running twice.
///
/// # Panics
///
/// Panics if the render sub-app is not registered yet, i.e. if this runs before
/// [`PrismRenderPlugin`](crate::PrismRenderPlugin). Adding a family is declaring
/// a dependency on the renderer, which cannot be satisfied in any other order.
pub fn register_render_assets<I, A>(app: &mut App)
where
    I: Instance + Send + Sync + 'static,
    I::Adapter: Send + Sync + 'static,
    I::Device: Send + Sync + 'static,
    I::Queue: Send + Sync + FrameSync + 'static,
    I::Surface: Send + Sync + 'static,
    A: RenderAsset,
{
    let already_registered = app
        .get_sub_app(RenderApp)
        .expect("register_render_assets: add PrismRenderPlugin first")
        .world
        .get_resource::<RenderAssets<A>>()
        .is_some();
    if already_registered {
        return;
    }
    app.add_extract(RenderApp, extract_assets::<A>);
    let render = app
        .sub_app_mut(RenderApp)
        .expect("the render sub-app was just checked");
    render.world.insert_resource(ExtractedAssets::<A>::new());
    render.world.insert_resource(RenderAssets::<A>::new());
    render.world.insert_resource(AssetSyncStats::<A>::default());
    render
        .world
        .resource_mut::<Schedules>()
        .get_mut(Render)
        .expect("the plugin registered the Render schedule")
        .add_systems(prepare_assets::<I, A>.in_set(RenderSystems::PrepareAssets));
}
