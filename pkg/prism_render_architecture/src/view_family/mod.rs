//! Main, shadow, reflection, stereo, capture, and offline views.

use crate::abi::GenerationalHandle;

pub type ViewHandle = GenerationalHandle;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ViewKind {
    Main,
    StereoEye,
    Shadow,
    Reflection,
    SceneCapture,
    Editor,
    OfflineTile,
}

#[derive(Clone, Copy, Debug)]
pub struct ViewImportance {
    pub streaming: f32,
    pub geometry_lod: f32,
    pub shadow_lod: f32,
}
