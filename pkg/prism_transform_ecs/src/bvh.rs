use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use prism_ecs::entity::Entity;
use prism_math::Aabb3;
use prism_physics_geometry::{Aabb, DynamicBvh, ProxyId};

/// Incremental adapter around the existing physics dynamic BVH.
#[derive(Default)]
pub struct BvhIndex {
    tree: DynamicBvh,
    proxies: BTreeMap<Entity, (ProxyId, Aabb3)>,
}

fn convert(bounds: Aabb3) -> Aabb {
    Aabb::new(
        [bounds.min.x, bounds.min.y, bounds.min.z].into(),
        [bounds.max.x, bounds.max.y, bounds.max.z].into(),
    )
}

impl BvhIndex {
    /// Existing backend for read-only geometry queries and structural validation.
    /// Raw BVH queries include its own internal fat margin.
    pub fn tree(&self) -> &DynamicBvh {
        &self.tree
    }

    /// Stable BVH handle for a full generational entity.
    pub fn proxy(&self, entity: Entity) -> Option<ProxyId> {
        self.proxies.get(&entity).map(|&(proxy, _)| proxy)
    }

    /// Exact overlap against committed transform bounds, in entity order.
    /// Filters the backend's extra fat-box slack; transform spatial margins,
    /// if configured, remain part of the committed bounds.
    pub fn query_aabb(&self, bounds: Aabb3) -> Vec<Entity> {
        if !super::valid(bounds) {
            return Vec::new();
        }
        let query = convert(bounds);
        let mut entities: Vec<_> = self
            .tree
            .query_aabb_collect(query)
            .into_iter()
            .filter_map(Entity::from_bits)
            .filter(|entity| {
                self.proxies
                    .get(entity)
                    .is_some_and(|&(_, bounds)| convert(bounds).intersects(&query))
            })
            .collect();
        entities.sort_unstable();
        entities
    }

    // Returns true only for an actual refit, not insertion or contained motion.
    pub(super) fn upsert(&mut self, entity: Entity, bounds: Aabb3) -> bool {
        if let Some((proxy, old)) = self.proxies.get_mut(&entity) {
            if *old == bounds {
                return false;
            }
            *old = bounds;
            self.tree.update(*proxy, convert(bounds))
        } else {
            let proxy = self.tree.insert(convert(bounds), entity.to_bits());
            self.proxies.insert(entity, (proxy, bounds));
            false
        }
    }

    pub(super) fn remove(&mut self, entity: Entity) {
        if let Some((proxy, _)) = self.proxies.remove(&entity) {
            self.tree.remove(proxy);
        }
    }
}
