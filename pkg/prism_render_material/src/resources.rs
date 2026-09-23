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
