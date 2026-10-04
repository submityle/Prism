//! Prefab / `IsA` inheritance diagnostic (design §16.3 / §16.6).
//!
//! The kernel resolves component reads on an instance *through* its [`IsA`]
//! chain rather than by cloning (design §16.3): an instance stores only the
//! components it overrides, and every other component falls through to the
//! nearest ancestor that provides one. That resolve-through model is cheap to
//! spawn but turns two shapes into latent costs an editor wants surfaced:
//!
//! * **Deep chains** — every unresolved read walks the transitive `IsA`
//!   closure nearest-first (design §11), so a long `prefab → prefab → …` chain
//!   is a per-read resolution cost. This report records each instance's chain
//!   depth and the deepest instance in the world.
//! * **Hot templates** — a prefab reused by thousands of instances is a
//!   high-fan-in hub: editing it re-resolves everywhere, and it is the natural
//!   authoring unit. The per-template fan-in (`direct_instances`) ranks these.
//!
//! On top of the graph shape it runs a **resolved-component census** per
//! instance by comparing the instance's *own* archetype component set
//! (design §5.3 / §6) against the union its ancestors provide:
//!
//! * `overridden` — own components that also exist on an ancestor (the
//!   instance shadows the inherited value — a deliberate per-field override,
//!   design §16.3).
//! * `inherited` — components an ancestor provides that the instance does not
//!   store itself (resolved purely by fall-through).
//! * `resolved_components` — the distinct components resolvable on the
//!   instance (own ∪ ancestor-provided), i.e. its effective composition.
//!
//! Finally it flags **cyclic `IsA`** participation. Inheritance resolution is
//! cycle-safe by construction (design §16.3), but a cycle is still a modelling
//! bug — a prefab that transitively inherits from itself — and the resolution
//! order inside a cycle is an implementation detail nobody should depend on.
//! Surfacing cyclic instances turns a silent smell into a visible count.
//!
//! Relation edges are **non-fragmenting** (design §23.3): the `IsA` marker is
//! never a column in any archetype, so the own-component census reads real
//! stored components only and never miscounts the relation marker itself.
//!
//! Capture is read-only — it borrows the world immutably and mutates nothing —
//! and deterministic: instances and templates are reported in ascending
//! [`Entity`] order and every ranking tie breaks on the lowest [`Entity`].

use alloc::vec::Vec;

use crate::collections::HashMap;
use crate::component::ComponentId;
use crate::entity::Entity;
use crate::prefab::IsA;
use crate::world::World;

/// Per-instance inheritance accounting for one `IsA` instance (design §16.3).
///
/// An *instance* is any entity that holds at least one outgoing `IsA` edge.
/// All component figures count distinct [`ComponentId`]s; the own set is the
/// instance's archetype component set (design §6) and the ancestor set is the
/// union of every chain member's own set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefabInstanceEntry {
    /// The instance entity.
    pub instance: Entity,
    /// Immediate `IsA` targets (direct prefabs) this instance declares.
    pub direct_prefabs: usize,
    /// Transitive ancestor count — the length of the resolution walk (design
    /// §11 transitive closure, nearest-first). Zero only for a self-loop whose
    /// cycle-safe closure excludes the start.
    pub depth: usize,
    /// Components the instance stores itself (its archetype component set).
    pub own_components: usize,
    /// Own components that an ancestor also provides — deliberate per-field
    /// overrides shadowing the inherited value (design §16.3).
    pub overridden: usize,
    /// Components an ancestor provides that the instance does not store,
    /// resolved purely by `IsA` fall-through.
    pub inherited: usize,
    /// Distinct components resolvable on the instance (own ∪ ancestor-provided)
    /// — its effective composition.
    pub resolved_components: usize,
    /// Whether this instance participates in an `IsA` cycle (a modelling bug;
    /// resolution stays cycle-safe but order inside the cycle is unspecified).
    pub cyclic: bool,
}

impl PrefabInstanceEntry {
    /// Whether the instance shadows any inherited component with its own value.
    #[inline]
    pub fn has_overrides(&self) -> bool {
        self.overridden > 0
    }

    /// Whether the instance stores no components of its own — it inherits its
    /// entire composition and carries only the `IsA` edge(s).
    #[inline]
    pub fn is_pure_instance(&self) -> bool {
        self.own_components == 0
    }

    /// The share of the instance's own components that are overrides, in
    /// per-mille (integer, avoiding float; design §17). Returns `0` when the
    /// instance stores no components.
    #[inline]
    pub fn override_permille(&self) -> u32 {
        if self.own_components == 0 {
            return 0;
        }
        ((self.overridden as u64 * 1000) / self.own_components as u64) as u32
    }
}

/// Per-template accounting for one prefab targeted by `IsA` (design §16.3).
///
/// A *template* is any entity that at least one instance points at with an
/// `IsA` edge. A template may itself be an instance of a deeper prefab
/// (`is_intermediate`), forming a chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefabTemplateEntry {
    /// The prefab (template) entity.
    pub prefab: Entity,
    /// Fan-in: entities that declare a *direct* `IsA` edge to this prefab.
    pub direct_instances: usize,
    /// Whether this template is itself an instance of a deeper prefab (it holds
    /// its own outgoing `IsA` edge).
    pub is_intermediate: bool,
    /// Components this template defines (its own archetype component set).
    pub own_components: usize,
}

impl PrefabTemplateEntry {
    /// Whether this template is a root — a pure template that does not itself
    /// inherit from a deeper prefab.
    #[inline]
    pub fn is_root_template(&self) -> bool {
        !self.is_intermediate
    }
}

/// Read-only prefab / `IsA` inheritance report (design §16.3 / §16.6).
///
/// Captured from a [`World`] via [`capture`](PrefabInheritanceReport::capture).
/// Instances and templates are disjoint only when no template is itself an
/// instance; an intermediate prefab appears in both lists (once as the
/// instance it is, once as the template others inherit from).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefabInheritanceReport {
    instances: Vec<PrefabInstanceEntry>,
    templates: Vec<PrefabTemplateEntry>,
    instance_count: usize,
    template_count: usize,
    max_depth: usize,
    cyclic_count: usize,
    total_overridden: usize,
    total_inherited: usize,
}

impl PrefabInheritanceReport {
    /// Capture the prefab / `IsA` inheritance shape of `world` (design §16.3).
    ///
    /// Walks the non-fragmenting relation index once to collect every `IsA`
    /// edge, then resolves each instance's chain depth and component census and
    /// each template's fan-in. Read-only and deterministic.
    pub fn capture(world: &World) -> Self {
        let Some(isa) = world.components().id_of::<IsA>() else {
            // `IsA` was never registered: no prefab inheritance exists.
            return Self {
                instances: Vec::new(),
                templates: Vec::new(),
                instance_count: 0,
                template_count: 0,
                max_depth: 0,
                cyclic_count: 0,
                total_overridden: 0,
                total_inherited: 0,
            };
        };

        // Distinct instances (edge sources) and templates (edge targets).
        let mut instance_set: HashMap<Entity, ()> = HashMap::new();
        let mut template_set: HashMap<Entity, ()> = HashMap::new();
        for (relation, source, target) in world.relations().index().iter_edges() {
            if relation != isa {
                continue;
            }
            instance_set.insert(source, ());
            template_set.insert(target, ());
        }

        // ---- Instances: depth + component census + cycle flag. ----
        let mut instances: Vec<Entity> = instance_set.keys().copied().collect();
        instances.sort_unstable();
        let mut instance_entries = Vec::with_capacity(instances.len());
        let mut max_depth = 0usize;
        let mut cyclic_count = 0usize;
        let mut total_overridden = 0usize;
        let mut total_inherited = 0usize;
        for instance in instances {
            let chain = world.prefab_chain(instance);
            let depth = chain.len();
            if depth > max_depth {
                max_depth = depth;
            }

            let own = own_component_ids(world, instance);
            // Union of every ancestor's own component set.
            let mut ancestor_ids: Vec<ComponentId> = Vec::new();
            for &ancestor in &chain {
                ancestor_ids.extend_from_slice(own_component_ids(world, ancestor));
            }
            ancestor_ids.sort_unstable();
            ancestor_ids.dedup();

            let overridden = intersection_len(own, &ancestor_ids);
            let inherited = difference_len(&ancestor_ids, own);
            let resolved_components = union_len(own, &ancestor_ids);

            // Cyclic iff any direct prefab reaches this instance back (or is it).
            let direct = world.prefabs_of(instance);
            let cyclic = direct.iter().any(|&p| {
                p == instance || world.prefab_chain(p).contains(&instance)
            });
            if cyclic {
                cyclic_count += 1;
            }
            total_overridden += overridden;
            total_inherited += inherited;

            instance_entries.push(PrefabInstanceEntry {
                instance,
                direct_prefabs: direct.len(),
                depth,
                own_components: own.len(),
                overridden,
                inherited,
                resolved_components,
                cyclic,
            });
        }

        // ---- Templates: fan-in + intermediate flag + own components. ----
        let mut templates: Vec<Entity> = template_set.keys().copied().collect();
        templates.sort_unstable();
        let mut template_entries = Vec::with_capacity(templates.len());
        for prefab in templates {
            let direct_instances = world.relation_sources::<IsA>(prefab).len();
            let is_intermediate = instance_set.contains_key(&prefab);
            let own_components = own_component_ids(world, prefab).len();
            template_entries.push(PrefabTemplateEntry {
                prefab,
                direct_instances,
                is_intermediate,
                own_components,
            });
        }

        let instance_count = instance_entries.len();
        let template_count = template_entries.len();
        Self {
            instances: instance_entries,
            templates: template_entries,
            instance_count,
            template_count,
            max_depth,
            cyclic_count,
            total_overridden,
            total_inherited,
        }
    }

    /// Alias for [`capture`](PrefabInheritanceReport::capture).
    #[inline]
    pub fn from_world(world: &World) -> Self {
        Self::capture(world)
    }

    /// Whether the world holds no `IsA` instances at all.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }

    /// Every instance entry, in ascending [`Entity`] order.
    #[inline]
    pub fn instances(&self) -> &[PrefabInstanceEntry] {
        &self.instances
    }

    /// Every template entry, in ascending [`Entity`] order.
    #[inline]
    pub fn templates(&self) -> &[PrefabTemplateEntry] {
        &self.templates
    }

    /// Number of distinct instance entities (entities with an outgoing `IsA`).
    #[inline]
    pub fn instance_count(&self) -> usize {
        self.instance_count
    }

    /// Number of distinct template entities (prefabs targeted by `IsA`).
    #[inline]
    pub fn template_count(&self) -> usize {
        self.template_count
    }

    /// The deepest resolution chain depth over all instances.
    #[inline]
    pub fn max_depth(&self) -> usize {
        self.max_depth
    }

    /// Number of instances participating in an `IsA` cycle (modelling bugs).
    #[inline]
    pub fn cyclic_count(&self) -> usize {
        self.cyclic_count
    }

    /// Whether any instance participates in an `IsA` cycle.
    #[inline]
    pub fn has_cycles(&self) -> bool {
        self.cyclic_count > 0
    }

    /// Total overridden components summed over all instances.
    #[inline]
    pub fn total_overridden(&self) -> usize {
        self.total_overridden
    }

    /// Total inherited (fall-through) components summed over all instances.
    #[inline]
    pub fn total_inherited(&self) -> usize {
        self.total_inherited
    }

    /// The instance entry for `entity`, if it is an `IsA` instance.
    pub fn instance(&self, entity: Entity) -> Option<&PrefabInstanceEntry> {
        self.instances
            .binary_search_by(|e| e.instance.cmp(&entity))
            .ok()
            .map(|i| &self.instances[i])
    }

    /// The template entry for `entity`, if it is targeted by `IsA`.
    pub fn template(&self, entity: Entity) -> Option<&PrefabTemplateEntry> {
        self.templates
            .binary_search_by(|e| e.prefab.cmp(&entity))
            .ok()
            .map(|i| &self.templates[i])
    }

    /// The instance with the deepest resolution chain (ties break on the lowest
    /// [`Entity`]), or `None` when there are no instances.
    pub fn deepest_instance(&self) -> Option<&PrefabInstanceEntry> {
        self.instances
            .iter()
            .max_by(|a, b| a.depth.cmp(&b.depth).then_with(|| b.instance.cmp(&a.instance)))
    }

    /// The most-reused template by direct fan-in (ties break on the lowest
    /// [`Entity`]), or `None` when there are no templates.
    pub fn hottest_template(&self) -> Option<&PrefabTemplateEntry> {
        self.templates.iter().max_by(|a, b| {
            a.direct_instances
                .cmp(&b.direct_instances)
                .then_with(|| b.prefab.cmp(&a.prefab))
        })
    }

    /// Every instance participating in an `IsA` cycle, in ascending [`Entity`]
    /// order.
    pub fn cyclic_instances(&self) -> Vec<&PrefabInstanceEntry> {
        self.instances.iter().filter(|e| e.cyclic).collect()
    }

    /// Every instance that stores no components of its own (pure inheritors),
    /// in ascending [`Entity`] order.
    pub fn pure_instances(&self) -> Vec<&PrefabInstanceEntry> {
        self.instances.iter().filter(|e| e.is_pure_instance()).collect()
    }

    /// Every root template (a template that does not itself inherit), in
    /// ascending [`Entity`] order.
    pub fn root_templates(&self) -> Vec<&PrefabTemplateEntry> {
        self.templates.iter().filter(|e| e.is_root_template()).collect()
    }
}

/// The own (stored) component ids of `entity`: its archetype component set, or
/// an empty slice if the entity is dead or has no components (design §6). The
/// slice is sorted ascending by [`ComponentId`].
fn own_component_ids(world: &World, entity: Entity) -> &[ComponentId] {
    let Some(location) = world.entities().location(entity) else {
        return &[];
    };
    match world.archetypes().get(location.archetype_id) {
        Some(archetype) => archetype.components().ids(),
        None => &[],
    }
}

/// Count of ids present in both sorted-ascending slices.
fn intersection_len(a: &[ComponentId], b: &[ComponentId]) -> usize {
    let (mut i, mut j, mut n) = (0usize, 0usize, 0usize);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            core::cmp::Ordering::Less => i += 1,
            core::cmp::Ordering::Greater => j += 1,
            core::cmp::Ordering::Equal => {
                n += 1;
                i += 1;
                j += 1;
            }
        }
    }
    n
}

/// Count of ids in `a` (sorted ascending) absent from `b` (sorted ascending).
fn difference_len(a: &[ComponentId], b: &[ComponentId]) -> usize {
    a.len() - intersection_len(a, b)
}

/// Size of the union of two sorted-ascending id slices.
fn union_len(a: &[ComponentId], b: &[ComponentId]) -> usize {
    a.len() + b.len() - intersection_len(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::Component;
    use crate::world::World;

    struct Health(#[allow(dead_code)] u32);
    impl Component for Health {}
    struct Speed(#[allow(dead_code)] u32);
    impl Component for Speed {}
    struct Armor(#[allow(dead_code)] u32);
    impl Component for Armor {}

    #[test]
    fn empty_world_has_no_prefabs() {
        let world = World::new();
        let report = PrefabInheritanceReport::capture(&world);
        assert!(report.is_empty());
        assert_eq!(report.instance_count(), 0);
        assert_eq!(report.template_count(), 0);
        assert_eq!(report.max_depth(), 0);
        assert!(!report.has_cycles());
        assert!(report.deepest_instance().is_none());
        assert!(report.hottest_template().is_none());
    }

    #[test]
    fn single_instance_counts_inherited_and_overridden() {
        let mut world = World::new();
        let prefab = world.spawn((Health(100), Speed(5)));
        let instance = world.spawn_instance_of(prefab);
        world.insert(instance, Speed(9)); // override Speed, inherit Health

        let report = PrefabInheritanceReport::capture(&world);
        assert_eq!(report.instance_count(), 1);
        assert_eq!(report.template_count(), 1);

        let e = report.instance(instance).unwrap();
        assert_eq!(e.direct_prefabs, 1);
        assert_eq!(e.depth, 1);
        assert_eq!(e.own_components, 1); // only Speed stored
        assert_eq!(e.overridden, 1); // Speed shadows the prefab's Speed
        assert_eq!(e.inherited, 1); // Health falls through
        assert_eq!(e.resolved_components, 2); // Health + Speed
        assert!(e.has_overrides());
        assert!(!e.is_pure_instance());
        assert_eq!(e.override_permille(), 1000);
        assert!(!e.cyclic);

        let t = report.template(prefab).unwrap();
        assert_eq!(t.direct_instances, 1);
        assert!(t.is_root_template());
        assert_eq!(t.own_components, 2);
        assert_eq!(report.total_overridden(), 1);
        assert_eq!(report.total_inherited(), 1);
    }

    #[test]
    fn pure_instance_inherits_everything() {
        let mut world = World::new();
        let prefab = world.spawn((Health(30), Speed(5)));
        let instance = world.spawn_instance_of(prefab);

        let report = PrefabInheritanceReport::capture(&world);
        let e = report.instance(instance).unwrap();
        assert_eq!(e.own_components, 0);
        assert_eq!(e.overridden, 0);
        assert_eq!(e.inherited, 2);
        assert_eq!(e.resolved_components, 2);
        assert!(e.is_pure_instance());
        assert_eq!(e.override_permille(), 0);
        assert_eq!(report.pure_instances().len(), 1);
    }

    #[test]
    fn chain_depth_and_intermediate_template() {
        let mut world = World::new();
        let base = world.spawn((Health(100), Speed(5), Armor(3)));
        let mid = world.spawn_instance_of(base);
        world.insert(mid, Health(80)); // mid overrides Health
        let leaf = world.spawn_instance_of(mid);

        let report = PrefabInheritanceReport::capture(&world);
        assert_eq!(report.max_depth(), 2);

        // `leaf` resolves through mid then base.
        let leaf_e = report.instance(leaf).unwrap();
        assert_eq!(leaf_e.depth, 2);
        assert_eq!(leaf_e.own_components, 0);
        assert_eq!(leaf_e.inherited, 3); // Health, Speed, Armor via chain
        assert_eq!(leaf_e.resolved_components, 3);
        assert!(!leaf_e.cyclic);

        // `mid` is both an instance (of base) and a template (for leaf).
        let mid_i = report.instance(mid).unwrap();
        assert_eq!(mid_i.depth, 1);
        assert_eq!(mid_i.own_components, 1); // Health override
        assert_eq!(mid_i.overridden, 1);
        let mid_t = report.template(mid).unwrap();
        assert!(mid_t.is_intermediate);
        assert!(!mid_t.is_root_template());

        // Deepest instance is `leaf`.
        assert_eq!(report.deepest_instance().unwrap().instance, leaf);
    }

    #[test]
    fn hottest_template_ranks_by_fan_in() {
        let mut world = World::new();
        let shared = world.spawn(Health(10));
        let other = world.spawn(Speed(1));
        let _a = world.spawn_instance_of(shared);
        let _b = world.spawn_instance_of(shared);
        let _c = world.spawn_instance_of(shared);
        let _d = world.spawn_instance_of(other);

        let report = PrefabInheritanceReport::capture(&world);
        assert_eq!(report.template_count(), 2);
        let hot = report.hottest_template().unwrap();
        assert_eq!(hot.prefab, shared);
        assert_eq!(hot.direct_instances, 3);
        assert_eq!(report.instance_count(), 4);
    }

    #[test]
    fn cyclic_isa_is_flagged() {
        let mut world = World::new();
        let a = world.spawn(());
        let b = world.spawn(Armor(7));
        world.add_prefab(a, b);
        world.add_prefab(b, a);

        let report = PrefabInheritanceReport::capture(&world);
        assert!(report.has_cycles());
        assert_eq!(report.cyclic_count(), 2);
        assert_eq!(report.cyclic_instances().len(), 2);
        // Both are instances and templates of each other.
        assert!(report.instance(a).unwrap().cyclic);
        assert!(report.instance(b).unwrap().cyclic);
    }
}
