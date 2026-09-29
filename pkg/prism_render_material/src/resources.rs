use alloc::collections::BTreeMap;
use prism_render_architecture::abi::GenerationalHandle;

#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum MaterialResourceKind {
    Texture2d = 1,
    TextureCube = 2,
    VirtualTexture = 3,
    Sampler = 4,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaterialResourceHandle {
    pub handle: GenerationalHandle,
    pub kind: MaterialResourceKind,
    pub flags: u32,
}

#[derive(Default)]
pub struct MaterialResourceTable {
    generations: BTreeMap<(MaterialResourceKind, u32), u32>,
}

impl MaterialResourceTable {
    pub fn publish(&mut self, resource: MaterialResourceHandle) -> bool {
        if !resource.handle.is_valid() || resource.handle.index == 0 {
            return false;
        }
        let key = (resource.kind, resource.handle.index);
        match self.generations.get(&key) {
            Some(generation) if *generation >= resource.handle.generation => false,
            _ => {
                self.generations.insert(key, resource.handle.generation);
                true
            }
        }
    }

    pub fn validate(&self, resource: MaterialResourceHandle) -> bool {
        self.generations
            .get(&(resource.kind, resource.handle.index))
            == Some(&resource.handle.generation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resource(kind: MaterialResourceKind, index: u32, generation: u32) -> MaterialResourceHandle {
        MaterialResourceHandle {
            handle: GenerationalHandle::new(index, generation),
            kind,
            flags: 0,
        }
    }

    #[test]
    fn publish_rejects_reserved_zero_index() {
        let mut table = MaterialResourceTable::default();
        assert!(!table.publish(resource(MaterialResourceKind::Texture2d, 0, 1)));
    }

    #[test]
    fn publish_rejects_invalid_handle() {
        let mut table = MaterialResourceTable::default();
        let invalid = MaterialResourceHandle {
            handle: GenerationalHandle::INVALID,
            kind: MaterialResourceKind::Sampler,
            flags: 0,
        };
        assert!(!table.publish(invalid));
    }

    #[test]
    fn publish_then_validate_round_trips() {
        let mut table = MaterialResourceTable::default();
        let handle = resource(MaterialResourceKind::Texture2d, 4, 1);
        assert!(table.publish(handle));
        assert!(table.validate(handle));
    }

    #[test]
    fn publish_rejects_stale_or_equal_generation() {
        let mut table = MaterialResourceTable::default();
        let current = resource(MaterialResourceKind::VirtualTexture, 7, 5);
        assert!(table.publish(current));
        // A strictly older generation is a stale republish.
        assert!(!table.publish(resource(MaterialResourceKind::VirtualTexture, 7, 4)));
        // Re-publishing the same generation is idempotent-reject, not a bump.
        assert!(!table.publish(resource(MaterialResourceKind::VirtualTexture, 7, 5)));
        // A newer generation supersedes and is accepted.
        assert!(table.publish(resource(MaterialResourceKind::VirtualTexture, 7, 6)));
    }

    #[test]
    fn validate_is_keyed_on_kind_and_generation() {
        let mut table = MaterialResourceTable::default();
        let published = resource(MaterialResourceKind::Texture2d, 3, 2);
        assert!(table.publish(published));
        // Same index, different kind is a distinct key.
        assert!(!table.validate(resource(MaterialResourceKind::TextureCube, 3, 2)));
        // Same key, mismatched generation does not validate.
        assert!(!table.validate(resource(MaterialResourceKind::Texture2d, 3, 1)));
    }
}
