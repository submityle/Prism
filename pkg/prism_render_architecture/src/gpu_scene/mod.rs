//! Stable, incrementally updated GPU scene.

use crate::abi::GenerationalHandle;

pub type SceneHandle = GenerationalHandle;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SceneBounds {
    pub center: [f32; 3],
    pub radius: f32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SceneChangeKind {
    Create,
    Destroy,
    Transform,
    Bounds,
    Geometry,
    Material,
    Flags,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SceneChange {
    pub handle: SceneHandle,
    pub kind: SceneChangeKind,
    pub sequence: u64,
}

#[derive(Default)]
pub struct SceneChangeJournal {
    changes: Vec<SceneChange>,
}

impl SceneChangeJournal {
    pub fn push(&mut self, change: SceneChange) {
        self.changes.push(change);
    }

    pub fn changes(&self) -> &[SceneChange] {
        &self.changes
    }

    pub fn clear(&mut self) {
        self.changes.clear();
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GpuSceneSnapshot {
    pub frame_epoch: u64,
    pub scene_epoch: u64,
    pub instance_count: u32,
}
