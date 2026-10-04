//! Required-components closure diagnostic (design §16.1 / §16.6).
//!
//! Required components (design §16.1) let a component declare that inserting it
//! must auto-insert a set of other components (e.g. a `Mesh3d` requiring
//! `Transform`, `GlobalTransform`, `Visibility`). The registry stores, for every
//! component, the **flattened transitive** closure of everything it drags in —
//! breadth-first, nearest-requirer-wins (see
//! [`ComponentInfo::required`](crate::component::ComponentInfo::required)). That
//! closure is what the structural insert path actually materialises, so its
//! shape is a direct lever on spawn / insert cost and on how tightly component
//! types are coupled.
//!
//! This report takes a graph-shape view over that closure set and exposes two
//! complementary per-component numbers:
//!
//! * **closure size** — how many components inserting this one auto-pulls in.
//!   A large closure means one `insert` quietly performs many archetype moves'
//!   worth of structural work; it is the "insert blast radius" of a type.
//! * **fan-in** — how many *other* registered components transitively require
//!   this one. A high fan-in component is a **hot shared dependency**: it is
//!   auto-inserted by many unrelated bundles, is almost always present, and any
//!   change to its semantics or layout ripples across the whole content set.
//!   `Transform` / `GlobalTransform` are the archetypal high-fan-in types.
//!
//! Neither number is visible from the per-type occupancy or storage reports:
//! those describe *instances* and *bytes*, while this one describes the
//! *declared requirement graph* — well-defined even on an entity-less world,
//! because required edges are registration-time facts.
//!
//! # What each entry reports
//! For every component that participates in the requirement graph (it requires
//! something, is required by something, or both):
//!
//! * [`closure_size`](RequiredClosureEntry::closure_size) — size of its
//!   flattened transitive required set.
//! * [`fan_in`](RequiredClosureEntry::fan_in) — how many registered components
//!   transitively require it.
//! * Classifiers: [`is_shared_dependency`](RequiredClosureEntry::is_shared_dependency)
//!   (required by two or more types — a genuine hub) and
//!   [`is_root_requirer`](RequiredClosureEntry::is_root_requirer) (pulls others
//!   in but is required by nothing — a top-level bundle anchor).
//!
//! # Scope and determinism
//! Only the direct-edge graph is private to the registry; the closures this
//! report reads are the public flattened sets, so it cannot distinguish a
//! direct requirement from a transitive one, nor measure requirement *depth*.
//! It is a pure read of the component registry: `O(Σ closure size)`, allocates
//! no entities, and is deterministic. Entries are ranked hottest-dependency
//! first (fan-in desc, then closure size desc, then component id asc).
//!
//! A structural invariant backs the totals: every closure edge
//! `(requirer → required)` contributes exactly one to the requirer's closure
//! size and one to the required component's fan-in, so
//! `Σ closure_size == Σ fan_in ==`
//! [`total_closure_edges`](RequiredClosureReport::total_closure_edges).

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::component::{ComponentId, Components};
use crate::world::World;

/// One component's position in the required-components graph (design §16.1).
///
/// Produced as part of [`RequiredClosureReport`]; only components that take
/// part in the requirement graph (nonzero closure size or fan-in) are emitted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequiredClosureEntry {
    /// The component this entry describes.
    pub component: ComponentId,
    /// Human-readable component name (Rust type name, or the user-supplied name
    /// for dynamically-registered components).
    pub name: String,
    /// Number of components in this component's flattened transitive required
    /// closure — how many others inserting it auto-pulls in (design §16.1).
    pub closure_size: usize,
    /// How many registered components transitively require this one — its
    /// fan-in. High fan-in marks a hot shared dependency.
    pub fan_in: usize,
}

impl RequiredClosureEntry {
    /// Whether inserting this component auto-inserts at least one other
    /// (its closure is non-empty).
    #[inline]
    pub fn pulls_in_others(&self) -> bool {
        self.closure_size > 0
    }

    /// Whether at least one other component transitively requires this one.
    #[inline]
    pub fn is_required(&self) -> bool {
        self.fan_in > 0
    }

    /// Whether two or more distinct components transitively require this one — a
    /// genuine shared / hub dependency rather than a one-off requirement.
    #[inline]
    pub fn is_shared_dependency(&self) -> bool {
        self.fan_in >= 2
    }

    /// Whether this component pulls others in yet is required by nothing — a
    /// top-level bundle anchor at the root of a requirement chain.
    #[inline]
    pub fn is_root_requirer(&self) -> bool {
        self.closure_size > 0 && self.fan_in == 0
    }
}

/// A whole-registry view of the required-components graph (design §16.1 /
/// §16.6).
///
/// Produced by [`RequiredClosureReport::capture`]. [`entries`](Self::entries)
/// holds one record per participating component, ranked hottest-dependency
/// first (fan-in desc, then closure size desc, then component id asc).
#[derive(Debug, Clone, Default)]
pub struct RequiredClosureReport {
    /// Participating components (nonzero closure size or fan-in), ranked
    /// hottest-dependency first.
    pub entries: Vec<RequiredClosureEntry>,
    /// Total number of registered component types (participating or not).
    pub registered_components: usize,
    /// How many registered components take part in the requirement graph.
    pub participating_components: usize,
    /// How many registered components require at least one other (nonzero
    /// closure).
    pub components_with_requirements: usize,
    /// How many registered components are required by at least one other
    /// (nonzero fan-in).
    pub required_components: usize,
    /// Total number of closure edges `(requirer → required)`; equals both
    /// `Σ closure_size` and `Σ fan_in`.
    pub total_closure_edges: usize,
}

impl RequiredClosureReport {
    /// Capture the required-components graph of `world`'s component registry.
    #[inline]
    pub fn capture(world: &World) -> Self {
        Self::from_components(world.components())
    }

    /// Build the report from a component registry directly.
    pub fn from_components(components: &Components) -> Self {
        let len = components.len();

        // Closure size per component id, and fan-in accumulated across every
        // component's flattened closure. Dense ids 0..len index both vectors.
        let mut closure_size: Vec<usize> = alloc::vec![0; len];
        let mut fan_in: Vec<usize> = alloc::vec![0; len];

        for (i, cs_slot) in closure_size.iter_mut().enumerate() {
            let Some(info) = components.info(ComponentId::new(i as u32)) else {
                continue;
            };
            let required = info.required();
            *cs_slot = required.len();
            for rc in required {
                let idx = rc.id().index() as usize;
                if idx < len {
                    fan_in[idx] += 1;
                }
            }
        }

        let mut entries: Vec<RequiredClosureEntry> = Vec::new();
        let mut components_with_requirements = 0;
        let mut required_components = 0;
        let mut total_closure_edges = 0;

        for (i, (&cs, &fi)) in closure_size.iter().zip(fan_in.iter()).enumerate() {
            total_closure_edges += cs;
            if cs > 0 {
                components_with_requirements += 1;
            }
            if fi > 0 {
                required_components += 1;
            }
            if cs == 0 && fi == 0 {
                continue;
            }
            let id = ComponentId::new(i as u32);
            let name = components
                .info(id)
                .map(|info| info.name().to_string())
                .unwrap_or_default();
            entries.push(RequiredClosureEntry {
                component: id,
                name,
                closure_size: cs,
                fan_in: fi,
            });
        }

        // Rank hottest shared dependency first; deterministic total order via
        // the component id tiebreak.
        entries.sort_by(|a, b| {
            b.fan_in
                .cmp(&a.fan_in)
                .then_with(|| b.closure_size.cmp(&a.closure_size))
                .then_with(|| a.component.index().cmp(&b.component.index()))
        });

        let participating_components = entries.len();

        Self {
            entries,
            registered_components: len,
            participating_components,
            components_with_requirements,
            required_components,
            total_closure_edges,
        }
    }

    /// Whether no component participates in the requirement graph.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The entry for `component`, if it participates in the requirement graph.
    pub fn entry(&self, component: ComponentId) -> Option<&RequiredClosureEntry> {
        self.entries.iter().find(|e| e.component == component)
    }

    /// The highest-fan-in component — the hottest shared dependency — or `None`
    /// when nothing requires anything. Because entries are fan-in-ranked, this
    /// is the first entry whenever any fan-in is nonzero.
    pub fn hottest_dependency(&self) -> Option<&RequiredClosureEntry> {
        self.entries.first().filter(|e| e.fan_in > 0)
    }

    /// The component with the largest flattened closure — the heaviest insert
    /// blast radius — or `None` when nothing requires anything. Ties resolve to
    /// the lower component id.
    pub fn heaviest_requirer(&self) -> Option<&RequiredClosureEntry> {
        self.entries
            .iter()
            .filter(|e| e.closure_size > 0)
            .max_by(|a, b| {
                a.closure_size
                    .cmp(&b.closure_size)
                    .then_with(|| b.component.index().cmp(&a.component.index()))
            })
    }

    /// Mean closure size across the components that require at least one other,
    /// or `0.0` when none do.
    #[inline]
    pub fn average_requirer_closure(&self) -> f32 {
        if self.components_with_requirements == 0 {
            0.0
        } else {
            self.total_closure_edges as f32 / self.components_with_requirements as f32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::Component;
    use crate::world::World;

    #[derive(Default)]
    struct Transform;
    impl Component for Transform {}

    #[derive(Default)]
    struct GlobalTransform;
    impl Component for GlobalTransform {}

    #[derive(Default)]
    struct Visibility;
    impl Component for Visibility {}

    struct Mesh3d;
    impl Component for Mesh3d {}

    struct Sprite;
    impl Component for Sprite {}

    struct Lonely;
    impl Component for Lonely {}

    #[test]
    fn empty_registry_reports_nothing() {
        let world = World::new();
        let report = RequiredClosureReport::capture(&world);
        assert!(report.is_empty());
        assert_eq!(report.total_closure_edges, 0);
        assert!(report.hottest_dependency().is_none());
        assert!(report.heaviest_requirer().is_none());
    }

    #[test]
    fn transform_is_the_hot_shared_dependency() {
        let mut world = World::new();
        // Mesh3d and Sprite both require Transform; Transform requires nothing.
        world.register_required_component::<Mesh3d, Transform>();
        world.register_required_component::<Sprite, Transform>();
        // An unrelated component that never joins the graph.
        world.register_component::<Lonely>();

        let report = RequiredClosureReport::capture(&world);

        let transform = world.components().id_of::<Transform>().unwrap();
        let mesh = world.components().id_of::<Mesh3d>().unwrap();
        let lonely = world.components().id_of::<Lonely>().unwrap();

        // Transform is required by two types -> hottest shared dependency.
        let hottest = report.hottest_dependency().unwrap();
        assert_eq!(hottest.component, transform);
        assert_eq!(hottest.fan_in, 2);
        assert_eq!(hottest.closure_size, 0);
        assert!(hottest.is_shared_dependency());
        assert!(!hottest.pulls_in_others());

        // Mesh3d pulls Transform in but nothing requires Mesh3d -> root requirer.
        let mesh_entry = report.entry(mesh).unwrap();
        assert_eq!(mesh_entry.closure_size, 1);
        assert_eq!(mesh_entry.fan_in, 0);
        assert!(mesh_entry.is_root_requirer());

        // Lonely never participates, so it is filtered out of the entries.
        assert!(report.entry(lonely).is_none());

        assert_eq!(report.participating_components, 3); // Transform, Mesh3d, Sprite
        assert_eq!(report.components_with_requirements, 2); // Mesh3d, Sprite
        assert_eq!(report.required_components, 1); // Transform
        assert_eq!(report.total_closure_edges, 2);
    }

    #[test]
    fn transitive_closure_is_flattened_and_counted() {
        let mut world = World::new();
        // Mesh3d -> Transform -> GlobalTransform -> Visibility (a chain).
        world.register_required_component::<Mesh3d, Transform>();
        world.register_required_component::<Transform, GlobalTransform>();
        world.register_required_component::<GlobalTransform, Visibility>();

        let report = RequiredClosureReport::capture(&world);

        let mesh = world.components().id_of::<Mesh3d>().unwrap();
        let visibility = world.components().id_of::<Visibility>().unwrap();

        // Mesh3d's flattened closure is the whole chain: 3 components.
        assert_eq!(report.entry(mesh).unwrap().closure_size, 3);
        // The heaviest requirer is Mesh3d (longest closure).
        assert_eq!(report.heaviest_requirer().unwrap().component, mesh);

        // Visibility is the sink: required by Mesh3d, Transform, GlobalTransform.
        let vis = report.entry(visibility).unwrap();
        assert_eq!(vis.fan_in, 3);
        assert_eq!(vis.closure_size, 0);
        assert!(vis.is_shared_dependency());
    }

    #[test]
    fn closure_edges_balance_fan_in() {
        let mut world = World::new();
        world.register_required_component::<Mesh3d, Transform>();
        world.register_required_component::<Sprite, Transform>();
        world.register_required_component::<Transform, GlobalTransform>();

        let report = RequiredClosureReport::capture(&world);

        // The structural invariant: Σ closure_size == Σ fan_in == total edges.
        let sum_closure: usize = report.entries.iter().map(|e| e.closure_size).sum();
        let sum_fan_in: usize = report.entries.iter().map(|e| e.fan_in).sum();
        assert_eq!(sum_closure, report.total_closure_edges);
        assert_eq!(sum_fan_in, report.total_closure_edges);
        assert_eq!(sum_closure, sum_fan_in);
    }
}
