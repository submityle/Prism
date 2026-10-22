//! CPU contract for replacing unavailable cut clusters with resident ancestors.
//!
//! This guarantees only that emitted clusters are resident and unique. It does
//! not prove complete surface coverage or crack-free fallback: the hierarchy
//! representation does not encode enough region/transition metadata to prove
//! either property.

use alloc::collections::{BTreeMap, BTreeSet, VecDeque};
use alloc::vec::Vec;

use super::hierarchy::{ClusterHierarchy, ClusterNode, CutCluster};
use super::GeometryPageKey;

/// One requested region's result, in input-cut order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CutRegionResolution {
    /// Cluster selected before residency fallback.
    pub requested: CutCluster,
    /// Resident replacement, or `None` when no resident ancestor exists.
    pub resident: Option<CutCluster>,
}

/// Residency-resolved cut. The draw list is deterministic and deduplicated;
/// regions retains one mapping per input cluster, including unresolved regions
/// and regions that share a fallback.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResidentCut {
    /// Resident clusters ordered by the first region they resolve.
    pub clusters: Vec<CutCluster>,
    /// Per-input result in the same order as the selected cut.
    pub regions: Vec<CutRegionResolution>,
}

/// Immutable hierarchy index used to resolve residency without rebuilding
/// reverse edges for every view or frame.
///
/// Construct one resolver when an asset is registered and reuse it while the
/// hierarchy remains unchanged. The index costs O(nodes + edges) memory and
/// construction time; each resolution then visits only selected nodes and
/// their ancestor chains.
pub struct ResidentCutResolver<'a> {
    hierarchy: &'a ClusterHierarchy,
    parents: Vec<Vec<u32>>,
}

/// Why a selected cut cannot safely be resolved.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResidentCutError {
    /// The hierarchy failed structural validation.
    MalformedHierarchy,
    /// A selected node index is outside the hierarchy.
    InvalidNode { node: u32 },
    /// The selected page no longer matches its node's page identity.
    StalePage {
        node: u32,
        expected: GeometryPageKey,
        actual: GeometryPageKey,
    },
    /// An ancestor belongs to another asset and cannot safely replace a region.
    AncestorAssetMismatch {
        node: u32,
        expected: u32,
        actual: u32,
    },
    /// A node occurs more than once in the cut.
    DuplicateNode { node: u32 },
    /// The cut selects both a node and its descendant.
    OverlappingCut { ancestor: u32, descendant: u32 },
}

impl<'a> ResidentCutResolver<'a> {
    /// Builds a reusable reverse-edge index for one immutable hierarchy.
    pub fn new(hierarchy: &'a ClusterHierarchy) -> Result<Self, ResidentCutError> {
        if !hierarchy.is_well_formed() {
            return Err(ResidentCutError::MalformedHierarchy);
        }
        let nodes = hierarchy.nodes();
        let mut parents = alloc::vec![Vec::<u32>::new(); nodes.len()];
        for (parent_index, node) in nodes.iter().enumerate() {
            let first = node.first_child as usize;
            let end = first + node.child_count as usize;
            for child in first..end {
                parents[child].push(parent_index as u32);
            }
        }
        for list in &mut parents {
            list.sort_unstable();
            list.dedup();
        }
        Ok(Self { hierarchy, parents })
    }

    /// Resolves selected clusters using the cached hierarchy index.
    ///
    /// In a shared-child DAG, equally near resident ancestors are resolved by
    /// lowest node index. Breadth-first traversal and sorted parent edges make
    /// the choice stable. Missing-parent chains terminate as unresolved
    /// regions. Validation completes before `is_resident` is called.
    pub fn resolve(
        &self,
        cut: &[CutCluster],
        mut is_resident: impl FnMut(GeometryPageKey) -> bool,
    ) -> Result<ResidentCut, ResidentCutError> {
        let hierarchy = self.hierarchy;
        let nodes = hierarchy.nodes();
        let mut selected = BTreeSet::new();
        for cluster in cut {
            let Some(node) = nodes.get(cluster.node as usize) else {
                return Err(ResidentCutError::InvalidNode { node: cluster.node });
            };
            if node.page != cluster.page {
                return Err(ResidentCutError::StalePage {
                    node: cluster.node,
                    expected: node.page,
                    actual: cluster.page,
                });
            }
            if !selected.insert(cluster.node) {
                return Err(ResidentCutError::DuplicateNode { node: cluster.node });
            }
        }

        // Reject overlapping levels, including ancestry through either branch of
        // a shared-child DAG. This does not claim that a non-overlapping cut is
        // complete: completeness is not encoded by the current cut representation.
        for &descendant in &selected {
            let mut seen = BTreeSet::new();
            let mut queue = VecDeque::from([descendant]);
            while let Some(node) = queue.pop_front() {
                for &parent in &self.parents[node as usize] {
                    if selected.contains(&parent) {
                        return Err(ResidentCutError::OverlappingCut {
                            ancestor: parent,
                            descendant,
                        });
                    }
                    if seen.insert(parent) {
                        queue.push_back(parent);
                    }
                }
            }
        }

        let mut result = ResidentCut {
            clusters: Vec::with_capacity(cut.len()),
            regions: Vec::with_capacity(cut.len()),
        };
        let mut emitted = BTreeSet::new();
        let mut page_residency = BTreeMap::new();
        for &requested in cut {
            let node = nodes[requested.node as usize];
            let resolved =
                if cached_residency(requested.page, &mut page_residency, &mut is_resident) {
                    Some(requested)
                } else {
                    nearest_resident_ancestor(
                        requested.node,
                        node.page.asset,
                        nodes,
                        &self.parents,
                        &mut page_residency,
                        &mut is_resident,
                    )?
                };
            if let Some(cluster) = resolved {
                if emitted.insert(cluster.node) {
                    result.clusters.push(cluster);
                }
            }
            result.regions.push(CutRegionResolution {
                requested,
                resident: resolved,
            });
        }
        Ok(result)
    }
}

fn nearest_resident_ancestor(
    start: u32,
    asset: u32,
    nodes: &[ClusterNode],
    parents: &[Vec<u32>],
    page_residency: &mut BTreeMap<GeometryPageKey, bool>,
    is_resident: &mut impl FnMut(GeometryPageKey) -> bool,
) -> Result<Option<CutCluster>, ResidentCutError> {
    let mut visited = BTreeSet::from([start]);
    let mut frontier = parents[start as usize].clone();
    while !frontier.is_empty() {
        frontier.sort_unstable();
        frontier.dedup();
        let mut next = Vec::new();
        for node_index in frontier {
            let node = nodes[node_index as usize];
            if node.page.asset != asset {
                return Err(ResidentCutError::AncestorAssetMismatch {
                    node: node_index,
                    expected: asset,
                    actual: node.page.asset,
                });
            }
            if cached_residency(node.page, page_residency, is_resident) {
                // The sorted frontier makes this the lowest-index nearest hit.
                return Ok(Some(CutCluster {
                    node: node_index,
                    page: node.page,
                }));
            }
            if visited.insert(node_index) {
                next.extend_from_slice(&parents[node_index as usize]);
            }
        }
        frontier = next;
    }
    Ok(None)
}

fn cached_residency(
    page: GeometryPageKey,
    cache: &mut BTreeMap<GeometryPageKey, bool>,
    is_resident: &mut impl FnMut(GeometryPageKey) -> bool,
) -> bool {
    *cache.entry(page).or_insert_with(|| is_resident(page))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_scene::SceneBounds;

    fn bounds() -> SceneBounds {
        SceneBounds {
            center: [0.0; 3],
            radius: 1.0,
            half_extents: [1.0; 3],
            _padding: 0.0,
        }
    }
    fn page(index: u32) -> GeometryPageKey {
        GeometryPageKey::new(4, index)
    }
    fn leaf(node: u32, page_id: u32) -> CutCluster {
        CutCluster {
            node,
            page: page(page_id),
        }
    }
    fn hierarchy(nodes: Vec<ClusterNode>, roots: Vec<u32>) -> ClusterHierarchy {
        ClusterHierarchy::new(nodes, roots)
    }

    #[test]
    fn direct_resident_cluster_is_preserved() {
        let h = hierarchy(vec![ClusterNode::leaf(bounds(), 0.0, page(0))], vec![0]);
        let actual = ResidentCutResolver::new(&h)
            .unwrap()
            .resolve(&[leaf(0, 0)], |p| p == page(0))
            .unwrap();
        assert_eq!(actual.clusters, vec![leaf(0, 0)]);
        assert_eq!(actual.regions[0].resident, Some(leaf(0, 0)));
    }

    #[test]
    fn unavailable_cluster_uses_nearest_resident_ancestor() {
        let h = hierarchy(
            vec![
                ClusterNode::interior(bounds(), 4.0, page(0), 1, 1),
                ClusterNode::interior(bounds(), 2.0, page(1), 2, 1),
                ClusterNode::leaf(bounds(), 0.0, page(2)),
            ],
            vec![0],
        );
        let actual = ResidentCutResolver::new(&h)
            .unwrap()
            .resolve(&[leaf(2, 2)], |p| p == page(1))
            .unwrap();
        assert_eq!(actual.clusters, vec![leaf(1, 1)]);
    }

    #[test]
    fn unavailable_root_is_reported_unresolved() {
        let h = hierarchy(vec![ClusterNode::leaf(bounds(), 0.0, page(0))], vec![0]);
        let actual = ResidentCutResolver::new(&h)
            .unwrap()
            .resolve(&[leaf(0, 0)], |_| false)
            .unwrap();
        assert!(actual.clusters.is_empty());
        assert_eq!(actual.regions[0].resident, None);
    }

    #[test]
    fn mixed_regions_preserve_order_and_report_unresolved_region() {
        let h = hierarchy(
            vec![
                ClusterNode::interior(bounds(), 2.0, page(0), 1, 2),
                ClusterNode::leaf(bounds(), 0.0, page(1)),
                ClusterNode::leaf(bounds(), 0.0, page(2)),
            ],
            vec![0],
        );
        let actual = ResidentCutResolver::new(&h)
            .unwrap()
            .resolve(&[leaf(1, 1), leaf(2, 2)], |p| p == page(1))
            .unwrap();
        assert_eq!(actual.clusters, vec![leaf(1, 1)]);
        assert_eq!(actual.regions[0].resident, Some(leaf(1, 1)));
        assert_eq!(actual.regions[1].resident, None);
    }

    #[test]
    fn invalid_stale_duplicate_and_overlapping_cuts_are_rejected() {
        let h = hierarchy(
            vec![
                ClusterNode::interior(bounds(), 1.0, page(0), 1, 1),
                ClusterNode::leaf(bounds(), 0.0, page(1)),
            ],
            vec![0],
        );
        let mut queried = false;
        assert_eq!(
            ResidentCutResolver::new(&h)
                .unwrap()
                .resolve(&[leaf(2, 2)], |_| {
                    queried = true;
                    true
                }),
            Err(ResidentCutError::InvalidNode { node: 2 })
        );
        assert!(!queried);
        assert_eq!(
            ResidentCutResolver::new(&h)
                .unwrap()
                .resolve(&[leaf(1, 9)], |_| {
                    queried = true;
                    true
                }),
            Err(ResidentCutError::StalePage {
                node: 1,
                expected: page(1),
                actual: page(9)
            })
        );
        assert!(!queried);
        assert_eq!(
            ResidentCutResolver::new(&h)
                .unwrap()
                .resolve(&[leaf(1, 1), leaf(1, 1)], |_| true),
            Err(ResidentCutError::DuplicateNode { node: 1 })
        );
        assert_eq!(
            ResidentCutResolver::new(&h)
                .unwrap()
                .resolve(&[leaf(0, 0), leaf(1, 1)], |_| true),
            Err(ResidentCutError::OverlappingCut {
                ancestor: 0,
                descendant: 1
            })
        );
    }

    #[test]
    fn shared_dag_chooses_nearest_lowest_index_and_deduplicates_output() {
        let h = hierarchy(
            vec![
                ClusterNode::interior(bounds(), 8.0, page(0), 1, 2),
                ClusterNode::interior(bounds(), 4.0, page(1), 3, 1),
                ClusterNode::interior(bounds(), 4.0, page(2), 3, 1),
                ClusterNode::leaf(bounds(), 0.0, page(3)),
            ],
            vec![0],
        );
        let actual = ResidentCutResolver::new(&h)
            .unwrap()
            .resolve(&[leaf(1, 1), leaf(2, 2)], |p| p == page(0))
            .unwrap();
        assert_eq!(actual.clusters, vec![leaf(0, 0)]);
        assert_eq!(actual.regions[0].resident, Some(leaf(0, 0)));
        assert_eq!(actual.regions[1].resident, Some(leaf(0, 0)));
    }

    #[test]
    fn malformed_hierarchy_is_rejected_at_resolver_construction() {
        let malformed = hierarchy(
            vec![ClusterNode::interior(bounds(), 1.0, page(0), 2, 1)],
            vec![0],
        );
        assert_eq!(
            ResidentCutResolver::new(&malformed).err(),
            Some(ResidentCutError::MalformedHierarchy)
        );
    }

    #[test]
    fn cross_asset_ancestor_is_rejected() {
        let other_asset_page = GeometryPageKey::new(8, 0);
        let h = hierarchy(
            vec![
                ClusterNode::interior(bounds(), 1.0, other_asset_page, 1, 1),
                ClusterNode::leaf(bounds(), 0.0, page(1)),
            ],
            vec![0],
        );
        let resolver = ResidentCutResolver::new(&h).unwrap();
        assert_eq!(
            resolver.resolve(&[leaf(1, 1)], |_| false),
            Err(ResidentCutError::AncestorAssetMismatch {
                node: 0,
                expected: 4,
                actual: 8,
            })
        );
    }
}
