//! Deterministic editor HLOD preview planning.
//!
//! The planner owns declarations and show/hide state only. It does not bake,
//! decode, or render proxy assets. Proxy keys are opaque provider identities,
//! so the editor can use the same plan with local, remote, or offline stores.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
};

use prism_asset::StableGuid;
use prism_ecs::partition::cell::CellCoord;

use crate::CellKey;

/// One HLOD level and its distance band.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HlodLayer {
    level: u16,
    enter_distance: u32,
    exit_distance: u32,
}

impl HlodLayer {
    /// Creates a layer. Level zero must begin at distance zero; all bands are
    /// half-open `[enter_distance, exit_distance)` intervals.
    pub const fn new(
        level: u16,
        enter_distance: u32,
        exit_distance: u32,
    ) -> Result<Self, HlodError> {
        if exit_distance <= enter_distance {
            return Err(HlodError::InvalidBand);
        }
        if level == 0 && enter_distance != 0 {
            return Err(HlodError::FirstLayerDistance);
        }
        Ok(Self {
            level,
            enter_distance,
            exit_distance,
        })
    }

    /// Layer index, ordered from near to far.
    #[must_use]
    pub const fn level(self) -> u16 {
        self.level
    }

    /// Inclusive lower distance boundary.
    #[must_use]
    pub const fn enter_distance(self) -> u32 {
        self.enter_distance
    }

    /// Exclusive upper distance boundary.
    #[must_use]
    pub const fn exit_distance(self) -> u32 {
        self.exit_distance
    }
}

/// One source cluster and its opaque proxy asset keys.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HlodCluster {
    id: StableGuid,
    anchor: CellCoord,
    source_cells: BTreeSet<CellKey>,
    proxy_keys: BTreeMap<u16, String>,
}

impl HlodCluster {
    /// Creates a cluster with sorted source cells and one proxy key per layer.
    pub fn new(
        id: StableGuid,
        anchor: CellCoord,
        source_cells: impl IntoIterator<Item = CellCoord>,
        proxy_keys: impl IntoIterator<Item = (u16, String)>,
    ) -> Result<Self, HlodError> {
        if id.is_nil() {
            return Err(HlodError::NilCluster);
        }
        let source_cells: BTreeSet<_> = source_cells.into_iter().map(CellKey::new).collect();
        if source_cells.is_empty() {
            return Err(HlodError::EmptyCluster);
        }
        let proxy_keys: BTreeMap<_, _> = proxy_keys.into_iter().collect();
        if proxy_keys.values().any(String::is_empty) {
            return Err(HlodError::EmptyProxyKey);
        }
        Ok(Self {
            id,
            anchor,
            source_cells,
            proxy_keys,
        })
    }

    /// Stable cluster identity.
    #[must_use]
    pub const fn id(&self) -> StableGuid {
        self.id
    }

    /// Cell used for deterministic distance evaluation.
    #[must_use]
    pub const fn anchor(&self) -> CellCoord {
        self.anchor
    }

    /// Source cells covered by the cluster.
    pub fn source_cells(&self) -> impl Iterator<Item = CellCoord> + '_ {
        self.source_cells.iter().map(|cell| cell.coord())
    }

    /// Proxy key for one layer, if declared.
    #[must_use]
    pub fn proxy_key(&self, level: u16) -> Option<&str> {
        self.proxy_keys.get(&level).map(String::as_str)
    }
}

/// HLOD declaration failures.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HlodError {
    /// A layer's interval is empty or inverted.
    InvalidBand,
    /// The first layer must begin at distance zero.
    FirstLayerDistance,
    /// Layer indices or distance bands are not strictly increasing.
    NonDeterministicLayerOrder,
    /// A cluster identity is nil.
    NilCluster,
    /// A cluster has no source cells.
    EmptyCluster,
    /// A proxy key is empty.
    EmptyProxyKey,
    /// A cluster identity is already registered.
    DuplicateCluster(StableGuid),
    /// A planner capacity was reached.
    CapacityExceeded,
}

impl fmt::Display for HlodError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBand => formatter.write_str("HLOD layer distance band is invalid"),
            Self::FirstLayerDistance => formatter.write_str("first HLOD layer must begin at zero"),
            Self::NonDeterministicLayerOrder => {
                formatter.write_str("HLOD layers must have strictly increasing order")
            }
            Self::NilCluster => formatter.write_str("HLOD cluster identity must not be nil"),
            Self::EmptyCluster => formatter.write_str("HLOD cluster must contain a source cell"),
            Self::EmptyProxyKey => formatter.write_str("HLOD proxy key must not be empty"),
            Self::DuplicateCluster(id) => write!(formatter, "HLOD cluster {id} already exists"),
            Self::CapacityExceeded => formatter.write_str("HLOD planner capacity exceeded"),
        }
    }
}

impl std::error::Error for HlodError {}

/// Viewpoint and resident-detail state used for one HLOD plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HlodView {
    viewpoint: CellCoord,
    resident_cells: BTreeSet<CellKey>,
}

impl HlodView {
    /// Creates a view from a viewpoint and resident detail set.
    #[must_use]
    pub fn new(viewpoint: CellCoord, resident_cells: impl IntoIterator<Item = CellCoord>) -> Self {
        Self {
            viewpoint,
            resident_cells: resident_cells.into_iter().map(CellKey::new).collect(),
        }
    }

    /// Viewpoint cell.
    #[must_use]
    pub const fn viewpoint(&self) -> CellCoord {
        self.viewpoint
    }

    /// Returns whether detail for a source cell is resident.
    #[must_use]
    pub fn is_resident(&self, cell: CellCoord) -> bool {
        self.resident_cells.contains(&CellKey::new(cell))
    }
}

/// One proxy visibility change.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HlodPlanOperation {
    cluster: StableGuid,
    level: u16,
    proxy_key: String,
    show: bool,
}

impl HlodPlanOperation {
    /// Cluster identity.
    #[must_use]
    pub const fn cluster(&self) -> StableGuid {
        self.cluster
    }

    /// Selected HLOD level.
    #[must_use]
    pub const fn level(&self) -> u16 {
        self.level
    }

    /// Opaque proxy asset key.
    #[must_use]
    pub fn proxy_key(&self) -> &str {
        &self.proxy_key
    }

    /// Whether the operation shows rather than hides the proxy.
    #[must_use]
    pub const fn is_show(&self) -> bool {
        self.show
    }
}

/// Deterministic HLOD diff against the planner's previous state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HlodPlan {
    epoch: u64,
    operations: Vec<HlodPlanOperation>,
}

impl HlodPlan {
    /// Planner epoch captured by this plan.
    #[must_use]
    pub const fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Show/hide operations sorted by `(cluster, show, level)`. Hides are
    /// emitted before shows when a cluster switches proxy levels.
    #[must_use]
    pub fn operations(&self) -> &[HlodPlanOperation] {
        &self.operations
    }
}

/// Bounded stateful HLOD planner.
#[derive(Clone, Debug)]
pub struct HlodPlanner {
    layers: Vec<HlodLayer>,
    clusters: BTreeMap<StableGuid, HlodCluster>,
    shown: BTreeSet<(StableGuid, u16)>,
    max_layers: usize,
    max_clusters: usize,
    epoch: u64,
}

impl HlodPlanner {
    /// Creates an empty planner with explicit catalog bounds.
    #[must_use]
    pub const fn new(max_layers: usize, max_clusters: usize) -> Self {
        Self {
            layers: Vec::new(),
            clusters: BTreeMap::new(),
            shown: BTreeSet::new(),
            max_layers,
            max_clusters,
            epoch: 0,
        }
    }

    /// Adds a layer in near-to-far order.
    pub fn add_layer(&mut self, layer: HlodLayer) -> Result<(), HlodError> {
        if self.layers.len() >= self.max_layers {
            return Err(HlodError::CapacityExceeded);
        }
        if self.layers.last().is_some_and(|previous| {
            previous.level >= layer.level || previous.exit_distance > layer.enter_distance
        }) {
            return Err(HlodError::NonDeterministicLayerOrder);
        }
        if self.layers.is_empty() && layer.level != 0 {
            return Err(HlodError::NonDeterministicLayerOrder);
        }
        self.layers.push(layer);
        self.bump_epoch();
        Ok(())
    }

    /// Registers one source cluster.
    pub fn add_cluster(&mut self, cluster: HlodCluster) -> Result<(), HlodError> {
        if self.clusters.len() >= self.max_clusters {
            return Err(HlodError::CapacityExceeded);
        }
        if self.clusters.contains_key(&cluster.id()) {
            return Err(HlodError::DuplicateCluster(cluster.id()));
        }
        self.clusters.insert(cluster.id(), cluster);
        self.bump_epoch();
        Ok(())
    }

    /// Computes a minimal proxy visibility diff and commits its shown state.
    pub fn plan(&mut self, view: &HlodView) -> HlodPlan {
        let mut desired = BTreeSet::new();
        for cluster in self.clusters.values() {
            if cluster.source_cells().any(|cell| view.is_resident(cell)) {
                continue;
            }
            let anchor = cluster.anchor();
            let viewpoint = view.viewpoint();
            let distance = anchor
                .x
                .abs_diff(viewpoint.x)
                .max(anchor.y.abs_diff(viewpoint.y))
                .max(anchor.z.abs_diff(viewpoint.z));
            let Some(layer) = self
                .layers
                .iter()
                .find(|layer| distance >= layer.enter_distance && distance < layer.exit_distance)
            else {
                continue;
            };
            if cluster.proxy_key(layer.level).is_some() {
                desired.insert((cluster.id(), layer.level));
            }
        }

        let mut operations = Vec::new();
        for &(cluster, level) in self.shown.difference(&desired) {
            if let Some(proxy_key) = self
                .clusters
                .get(&cluster)
                .and_then(|cluster| cluster.proxy_key(level))
            {
                operations.push(HlodPlanOperation {
                    cluster,
                    level,
                    proxy_key: proxy_key.to_owned(),
                    show: false,
                });
            }
        }
        for &(cluster, level) in desired.difference(&self.shown) {
            if let Some(proxy_key) = self
                .clusters
                .get(&cluster)
                .and_then(|cluster| cluster.proxy_key(level))
            {
                operations.push(HlodPlanOperation {
                    cluster,
                    level,
                    proxy_key: proxy_key.to_owned(),
                    show: true,
                });
            }
        }
        operations
            .sort_unstable_by_key(|operation| (operation.cluster, operation.show, operation.level));
        self.shown = desired;
        self.bump_epoch();
        HlodPlan {
            epoch: self.epoch,
            operations,
        }
    }

    /// Returns the currently shown cluster/level pairs in stable order.
    pub fn shown(&self) -> impl Iterator<Item = (StableGuid, u16)> + '_ {
        self.shown.iter().copied()
    }

    fn bump_epoch(&mut self) {
        self.epoch = self.epoch.saturating_add(1);
    }
}
