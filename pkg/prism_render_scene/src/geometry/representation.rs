//! Versioned geometry-representation selection shared by raster and ray consumers.
//!
//! Selection is data, not policy: visibility chooses a representation per view,
//! and consumers resolve that exact choice against geometry built from the same
//! generational identity and revision. A conservative RT representation is an
//! explicit fallback, never an implicit reinterpretation of a stale selection.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use bevy_ecs::prelude::Resource;
use prism_render_architecture::virtual_geometry::{CutCluster, GeometryPageKey};
use prism_render_architecture::{abi::GenerationalHandle, gpu_scene::GeometryHandle};

/// Current wire/data contract version for cross-pass geometry selection.
pub const GEOMETRY_REPRESENTATION_CONTRACT_VERSION: u32 = 1;

/// Stable logical identity of one view's geometry decision.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct GeometryRepresentationIdentity {
    /// Scene-instance slot and generation for which this choice was made.
    pub scene: GenerationalHandle,
    /// Geometry slot and generation selected by the scene instance.
    pub geometry: GeometryHandle,
    /// Geometry-table revision observed when the selection was made.
    pub geometry_revision: u32,
    /// Monotonic revision of this view's selection snapshot.
    pub selection_revision: u64,
    /// Stable view identity; `None` is reserved for view-independent consumers.
    pub view: Option<prism_render_architecture::abi::GenerationalHandle>,
}

/// Physical/logical representation selected for an instance.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum GeometryRepresentation {
    /// A raster-compatible resident level of detail.
    Lod { level: u32 },
    /// A virtual-geometry cluster, identified within its geometry generation.
    Cluster { cluster: u32 },
    /// A conservative, view-independent representation authored for ray tracing.
    ConservativeRayTracing,
}

/// Versioned selection exchanged between visibility, raster and ray consumers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GeometryRepresentationSelection {
    /// Contract version; unknown versions fail closed.
    pub version: u32,
    /// Full identity/revision observed by the selector.
    pub identity: GeometryRepresentationIdentity,
    /// Exact selected representation.
    pub selected: GeometryRepresentation,
    /// Publication identity required when `selected` is a virtual cluster.
    /// Ordinary mesh LOD selections must leave this unset.
    pub virtual_geometry: Option<VirtualGeometryRepresentationIdentity>,
    /// Whether the selector permits a conservative RT fallback if the exact
    /// representation is unavailable. Identity mismatches are never fallbackable.
    pub allow_conservative_rt_fallback: bool,
}

/// Immutable virtual-geometry publication identity captured by visibility.
/// Content revision identifies the artifact; generation prevents reuse across
/// retirement/replacement in the same runtime registry.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct VirtualGeometryRepresentationIdentity {
    /// Stable authored asset identity.
    pub asset_id: u32,
    /// Runtime registry generation.
    pub generation: u64,
    /// Canonical artifact digest supplied by the asset publisher.
    pub content_revision: [u8; 32],
}

impl GeometryRepresentationSelection {
    /// Constructs a selection using the current contract version.
    #[must_use]
    pub const fn new(
        identity: GeometryRepresentationIdentity,
        selected: GeometryRepresentation,
        allow_conservative_rt_fallback: bool,
    ) -> Self {
        Self {
            version: GEOMETRY_REPRESENTATION_CONTRACT_VERSION,
            identity,
            selected,
            virtual_geometry: None,
            allow_conservative_rt_fallback,
        }
    }

    /// Constructs a selection for one CLST hierarchy node.
    #[must_use]
    pub const fn virtual_cluster(
        identity: GeometryRepresentationIdentity,
        cluster: u32,
        virtual_geometry: VirtualGeometryRepresentationIdentity,
    ) -> Self {
        Self {
            version: GEOMETRY_REPRESENTATION_CONTRACT_VERSION,
            identity,
            selected: GeometryRepresentation::Cluster { cluster },
            virtual_geometry: Some(virtual_geometry),
            allow_conservative_rt_fallback: false,
        }
    }
}

/// Why a representation selection could not be consumed safely.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum GeometryRepresentationRejection {
    UnsupportedVersion { supplied: u32, supported: u32 },
    SceneGenerationMismatch,
    GeometryGenerationMismatch,
    GeometryRevisionMismatch { selected: u32, available: u32 },
    SelectionRevisionMismatch { selected: u64, available: u64 },
    ViewMismatch,
    RepresentationUnavailable,
    ConservativeFallbackUnavailable,
    VirtualGeometryIdentityMissing,
    UnexpectedVirtualGeometryIdentity,
}

/// Resolution result, preserving whether a consumer used the declared fallback.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResolvedGeometryRepresentation {
    pub representation: GeometryRepresentation,
    pub used_fallback: bool,
}

/// Authoritative per-view geometry choices emitted by unified visibility.
///
/// Entries are keyed by the scene-instance slot used by the render-scene and
/// ray-scene mirrors. The snapshot revision changes once per visibility build;
/// consumers must match both it and the embedded view/geometry identity before
/// reusing a selection. Only visible work items are present.
#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct GeometryRepresentationSelections {
    revision: u64,
    views:
        BTreeMap<GenerationalHandle, BTreeMap<GenerationalHandle, GeometryRepresentationSelection>>,
    cluster_cuts: BTreeMap<
        (GenerationalHandle, GenerationalHandle),
        (Vec<CutCluster>, Vec<(GeometryPageKey, f32)>),
    >,
    cluster_requests_by_view: BTreeMap<(GenerationalHandle, GeometryPageKey), f32>,
}

impl GeometryRepresentationSelections {
    /// Replaces the complete snapshot for one visibility frame.
    pub(crate) fn replace(
        &mut self,
        revision: u64,
        views: BTreeMap<
            GenerationalHandle,
            BTreeMap<GenerationalHandle, GeometryRepresentationSelection>,
        >,
    ) {
        self.revision = revision;
        self.views = views;
        self.cluster_cuts.clear();
        self.cluster_requests_by_view.clear();
    }

    /// Publishes one exact per-instance selection into the current snapshot.
    pub(crate) fn insert(
        &mut self,
        view: GenerationalHandle,
        scene_instance: GenerationalHandle,
        selection: GeometryRepresentationSelection,
    ) {
        self.views
            .entry(view)
            .or_default()
            .insert(scene_instance, selection);
    }

    /// Begins a new complete frame snapshot, discarding stale views/items.
    pub(crate) fn begin_frame(&mut self, revision: u64) {
        self.revision = revision;
        self.views.clear();
        self.cluster_cuts.clear();
        self.cluster_requests_by_view.clear();
    }

    /// Monotonic visibility-frame revision associated with this snapshot.
    #[must_use]
    pub const fn revision(&self) -> u64 {
        self.revision
    }

    /// Selections for one exact generational view identity.
    #[must_use]
    pub fn for_view(
        &self,
        view: GenerationalHandle,
    ) -> Option<&BTreeMap<GenerationalHandle, GeometryRepresentationSelection>> {
        self.views.get(&view)
    }

    /// Selection for one exact scene instance in one exact generational view.
    #[must_use]
    pub fn selection(
        &self,
        view: GenerationalHandle,
        scene: GenerationalHandle,
    ) -> Option<&GeometryRepresentationSelection> {
        self.views.get(&view)?.get(&scene)
    }

    /// Publishes a complete deterministic CLST cut for one visible instance.
    pub(crate) fn set_cluster_cut(
        &mut self,
        view: GenerationalHandle,
        scene: GenerationalHandle,
        cut: Vec<CutCluster>,
        requests: Vec<(GeometryPageKey, f32)>,
    ) {
        self.cluster_cuts.insert((view, scene), (cut, requests));
    }

    /// Publishes one exact CLST selection and its complete cut atomically.
    pub(crate) fn insert_cluster_selection(
        &mut self,
        view: GenerationalHandle,
        scene: GenerationalHandle,
        selection: GeometryRepresentationSelection,
        cut: Vec<CutCluster>,
        requests: Vec<(GeometryPageKey, f32)>,
    ) {
        self.insert(view, scene, selection);
        self.record_cluster_page_requests(view, &requests);
        self.set_cluster_cut(view, scene, cut, requests);
    }

    /// Records page demand even when no complete resident cut can currently be
    /// drawn. This lets streaming make progress while the renderer uses the
    /// instance's standard LOD fallback for the current frame.
    pub(crate) fn record_cluster_page_requests(
        &mut self,
        view: GenerationalHandle,
        requests: &[(GeometryPageKey, f32)],
    ) {
        for (page, priority) in requests {
            let current = self
                .cluster_requests_by_view
                .entry((view, *page))
                .or_insert(*priority);
            if *priority > *current {
                *current = *priority;
            }
        }
    }

    /// Exact CLST cut produced for one view/instance pair in this snapshot.
    #[must_use]
    pub fn cluster_cut(
        &self,
        view: GenerationalHandle,
        scene: GenerationalHandle,
    ) -> Option<&[CutCluster]> {
        self.cluster_cuts
            .get(&(view, scene))
            .map(|(cut, _)| cut.as_slice())
    }

    /// Deterministically coalesced pages referenced by the selected cut.
    #[must_use]
    pub fn cluster_page_requests(
        &self,
        view: GenerationalHandle,
        scene: GenerationalHandle,
    ) -> Option<Vec<GeometryPageKey>> {
        self.cluster_cuts
            .get(&(view, scene))
            .map(|(_, requests)| requests.iter().map(|(page, _)| *page).collect())
    }

    /// Per-instance page requests paired with maximum projected coverage
    /// priority. Results are lexicographically ordered by page key.
    #[must_use]
    pub fn weighted_cluster_page_requests(
        &self,
        view: GenerationalHandle,
        scene: GenerationalHandle,
    ) -> Option<&[(GeometryPageKey, f32)]> {
        self.cluster_cuts
            .get(&(view, scene))
            .map(|(_, requests)| requests.as_slice())
    }

    /// All page requests for a view, coalesced across scene instances. The
    /// lexicographic page order and maximum coverage priority are stable.
    #[must_use]
    pub fn page_requests_for_view(&self, view: GenerationalHandle) -> Vec<(GeometryPageKey, f32)> {
        self.cluster_requests_by_view
            .range(
                (view, GeometryPageKey::new(0, 0))
                    ..=(view, GeometryPageKey::new(u32::MAX, u32::MAX)),
            )
            .map(|((_, page), priority)| (*page, *priority))
            .collect()
    }

    /// Number of views represented by the current visibility snapshot.
    #[must_use]
    pub fn view_count(&self) -> usize {
        self.views.len()
    }

    /// Iterates the complete per-view snapshot in deterministic view order.
    pub(crate) fn iter(
        &self,
    ) -> impl Iterator<
        Item = (
            GenerationalHandle,
            &BTreeMap<GenerationalHandle, GeometryRepresentationSelection>,
        ),
    > {
        self.views
            .iter()
            .map(|(view, selections)| (*view, selections))
    }
}

/// Builds a versioned choice from one resolved visibility work item.
#[must_use]
pub(crate) fn visibility_lod_selection(
    geometry: GeometryHandle,
    geometry_revision: u32,
    scene: GenerationalHandle,
    view: GenerationalHandle,
    selection_revision: u64,
    resolved_lod: u32,
) -> GeometryRepresentationSelection {
    GeometryRepresentationSelection::new(
        GeometryRepresentationIdentity {
            scene,
            geometry,
            geometry_revision,
            selection_revision,
            view: Some(view),
        },
        GeometryRepresentation::Lod {
            level: resolved_lod,
        },
        true,
    )
}

/// Checks identity and resolves the exact representation or its explicit RT fallback.
///
/// `available` contains only representations whose payloads were built from
/// `available_geometry` and `available_geometry_revision`. Generation and
/// revision checks precede fallback, so an unrelated replacement can never be
/// mistaken for the selected mesh.
pub fn resolve_geometry_representation(
    selection: GeometryRepresentationSelection,
    available_scene: GenerationalHandle,
    available_geometry: GeometryHandle,
    available_geometry_revision: u32,
    available_view: Option<prism_render_architecture::abi::GenerationalHandle>,
    available_selection_revision: u64,
    available: &[GeometryRepresentation],
) -> Result<ResolvedGeometryRepresentation, GeometryRepresentationRejection> {
    if selection.version != GEOMETRY_REPRESENTATION_CONTRACT_VERSION {
        return Err(GeometryRepresentationRejection::UnsupportedVersion {
            supplied: selection.version,
            supported: GEOMETRY_REPRESENTATION_CONTRACT_VERSION,
        });
    }
    if selection.identity.scene != available_scene {
        return Err(GeometryRepresentationRejection::SceneGenerationMismatch);
    }
    if selection.identity.geometry.index != available_geometry.index
        || selection.identity.geometry.generation != available_geometry.generation
    {
        return Err(GeometryRepresentationRejection::GeometryGenerationMismatch);
    }
    if selection.identity.geometry_revision != available_geometry_revision {
        return Err(GeometryRepresentationRejection::GeometryRevisionMismatch {
            selected: selection.identity.geometry_revision,
            available: available_geometry_revision,
        });
    }
    if selection.identity.view != available_view {
        return Err(GeometryRepresentationRejection::ViewMismatch);
    }
    if selection.identity.selection_revision != available_selection_revision {
        return Err(GeometryRepresentationRejection::SelectionRevisionMismatch {
            selected: selection.identity.selection_revision,
            available: available_selection_revision,
        });
    }
    match (selection.selected, selection.virtual_geometry) {
        (GeometryRepresentation::Cluster { .. }, None) => {
            return Err(GeometryRepresentationRejection::VirtualGeometryIdentityMissing);
        }
        (GeometryRepresentation::Lod { .. }, Some(_))
        | (GeometryRepresentation::ConservativeRayTracing, Some(_)) => {
            return Err(GeometryRepresentationRejection::UnexpectedVirtualGeometryIdentity);
        }
        _ => {}
    }
    if available.contains(&selection.selected) {
        return Ok(ResolvedGeometryRepresentation {
            representation: selection.selected,
            used_fallback: false,
        });
    }
    if selection.allow_conservative_rt_fallback {
        if available.contains(&GeometryRepresentation::ConservativeRayTracing) {
            return Ok(ResolvedGeometryRepresentation {
                representation: GeometryRepresentation::ConservativeRayTracing,
                used_fallback: true,
            });
        }
        return Err(GeometryRepresentationRejection::ConservativeFallbackUnavailable);
    }
    Err(GeometryRepresentationRejection::RepresentationUnavailable)
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_architecture::abi::GenerationalHandle;

    fn selection() -> GeometryRepresentationSelection {
        GeometryRepresentationSelection::new(
            GeometryRepresentationIdentity {
                scene: GenerationalHandle::new(7, 2),
                geometry: GeometryHandle::new(5, 3),
                geometry_revision: 17,
                selection_revision: 29,
                view: Some(GenerationalHandle::new(2, 4)),
            },
            GeometryRepresentation::Lod { level: 2 },
            true,
        )
    }

    fn resolve(
        selection: GeometryRepresentationSelection,
        geometry: GeometryHandle,
        geometry_revision: u32,
        selection_revision: u64,
        available: &[GeometryRepresentation],
    ) -> Result<ResolvedGeometryRepresentation, GeometryRepresentationRejection> {
        resolve_geometry_representation(
            selection,
            GenerationalHandle::new(7, 2),
            geometry,
            geometry_revision,
            Some(GenerationalHandle::new(2, 4)),
            selection_revision,
            available,
        )
    }

    #[test]
    fn stale_geometry_generation_fails_closed_even_when_fallback_exists() {
        assert_eq!(
            resolve(
                selection(),
                GeometryHandle::new(5, 4),
                17,
                29,
                &[GeometryRepresentation::ConservativeRayTracing],
            ),
            Err(GeometryRepresentationRejection::GeometryGenerationMismatch)
        );
    }

    #[test]
    fn selected_lod_requires_matching_geometry_and_selection_revisions() {
        let selected = selection();
        assert_eq!(
            resolve(
                selected,
                GeometryHandle::new(5, 3),
                17,
                29,
                &[GeometryRepresentation::Lod { level: 2 }],
            ),
            Ok(ResolvedGeometryRepresentation {
                representation: GeometryRepresentation::Lod { level: 2 },
                used_fallback: false,
            })
        );
        assert!(matches!(
            resolve(selected, GeometryHandle::new(5, 3), 18, 29, &[]),
            Err(GeometryRepresentationRejection::GeometryRevisionMismatch { .. })
        ));
        assert!(matches!(
            resolve(selected, GeometryHandle::new(5, 3), 17, 30, &[]),
            Err(GeometryRepresentationRejection::SelectionRevisionMismatch { .. })
        ));
    }

    #[test]
    fn missing_lod_uses_only_an_explicit_conservative_rt_fallback() {
        assert_eq!(
            resolve(
                selection(),
                GeometryHandle::new(5, 3),
                17,
                29,
                &[GeometryRepresentation::ConservativeRayTracing],
            ),
            Ok(ResolvedGeometryRepresentation {
                representation: GeometryRepresentation::ConservativeRayTracing,
                used_fallback: true,
            })
        );
        let mut no_fallback = selection();
        no_fallback.allow_conservative_rt_fallback = false;
        assert_eq!(
            resolve(
                no_fallback,
                GeometryHandle::new(5, 3),
                17,
                29,
                &[GeometryRepresentation::ConservativeRayTracing],
            ),
            Err(GeometryRepresentationRejection::RepresentationUnavailable)
        );
    }

    #[test]
    fn scene_generation_mismatch_is_fail_closed_before_representation_fallback() {
        let selection = selection();
        assert_eq!(
            resolve_geometry_representation(
                selection,
                GenerationalHandle::new(7, 3),
                GeometryHandle::new(5, 3),
                17,
                Some(GenerationalHandle::new(2, 4)),
                29,
                &[GeometryRepresentation::ConservativeRayTracing],
            ),
            Err(GeometryRepresentationRejection::SceneGenerationMismatch)
        );
    }

    #[test]
    fn selection_snapshot_replaces_stale_views_and_preserves_generational_keys() {
        let mut snapshot = GeometryRepresentationSelections::default();
        let view = GenerationalHandle::new(2, 4);
        let scene = GenerationalHandle::new(7, 2);
        snapshot.begin_frame(29);
        snapshot.insert(view, scene, selection());
        assert_eq!(snapshot.revision(), 29);
        assert_eq!(snapshot.view_count(), 1);
        assert!(snapshot.for_view(view).unwrap().contains_key(&scene));
        snapshot.begin_frame(30);
        assert_eq!(snapshot.view_count(), 0);
        assert!(snapshot.for_view(view).is_none());
    }

    #[test]
    fn cluster_page_requests_expose_keys_and_weights_and_coalesce_by_max_priority() {
        let mut snapshot = GeometryRepresentationSelections::default();
        let view = GenerationalHandle::new(2, 4);
        let scene_a = GenerationalHandle::new(7, 2);
        let scene_b = GenerationalHandle::new(8, 2);
        let identity = GeometryRepresentationIdentity {
            scene: scene_a,
            geometry: GeometryHandle::new(5, 3),
            geometry_revision: 17,
            selection_revision: 29,
            view: Some(view),
        };
        let page_a = GeometryPageKey::new(9, 1);
        let page_b = GeometryPageKey::new(9, 2);
        snapshot.begin_frame(29);
        snapshot.insert_cluster_selection(
            view,
            scene_a,
            GeometryRepresentationSelection::virtual_cluster(
                identity,
                1,
                VirtualGeometryRepresentationIdentity {
                    asset_id: 9,
                    generation: 3,
                    content_revision: [4; 32],
                },
            ),
            vec![
                CutCluster {
                    node: 1,
                    page: page_a,
                },
                CutCluster {
                    node: 2,
                    page: page_b,
                },
            ],
            vec![(page_a, 2.0), (page_b, 4.0)],
        );
        let mut second = identity;
        second.scene = scene_b;
        snapshot.insert_cluster_selection(
            view,
            scene_b,
            GeometryRepresentationSelection::virtual_cluster(
                second,
                3,
                VirtualGeometryRepresentationIdentity {
                    asset_id: 9,
                    generation: 3,
                    content_revision: [4; 32],
                },
            ),
            vec![CutCluster {
                node: 3,
                page: page_a,
            }],
            vec![(page_a, 7.0)],
        );

        assert_eq!(
            snapshot.cluster_page_requests(view, scene_a),
            Some(vec![page_a, page_b])
        );
        assert_eq!(
            snapshot.weighted_cluster_page_requests(view, scene_a),
            Some(&[(page_a, 2.0), (page_b, 4.0)][..])
        );
        assert_eq!(
            snapshot.page_requests_for_view(view),
            vec![(page_a, 7.0), (page_b, 4.0)]
        );
        snapshot.begin_frame(30);
        assert!(snapshot.cluster_cut(view, scene_a).is_none());
        assert!(snapshot.page_requests_for_view(view).is_empty());
    }

    #[test]
    fn page_demand_is_retained_when_residency_snapshot_is_unavailable() {
        let mut snapshot = GeometryRepresentationSelections::default();
        let view = GenerationalHandle::new(2, 4);
        let requested = [(GeometryPageKey::new(12, 7), 3.5)];
        snapshot.begin_frame(41);
        snapshot.record_cluster_page_requests(view, &requested);
        assert_eq!(snapshot.page_requests_for_view(view), requested);
        assert!(snapshot.for_view(view).is_none());
        assert!(snapshot
            .cluster_cut(view, GenerationalHandle::new(5, 3))
            .is_none());
    }
}
