//! Retained indexed triangle ID drawing with camera and depth occlusion.

use alloc::{collections::BTreeMap, vec::Vec};
use core::{convert::Infallible, fmt};
use wgpu::util::DeviceExt;

use crate::{
    GpuPickingFrame, RenderTargetState, SceneIdPass, SceneIdPassEntry, SceneIdPassError,
    SceneIdPassSink, SceneObjectId, ID_BUFFER_FORMAT,
};

/// Column-major matrix, multiplied by column vectors in WGSL. View/projection
/// matrices must use WebGPU clip depth `[0, 1]` and target aspect ratio.
pub type SceneIdMatrix = [[f32; 4]; 4];

/// Identity model/view-projection matrix.
pub const SCENE_ID_IDENTITY_MATRIX: SceneIdMatrix = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

/// Depth convention for the camera projection. Equal-depth fragments retain
/// the first draw in deterministic scene sort order; both mesh sides are drawn.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SceneIdDepth {
    /// Near = 0, far = 1; strict `Less` test and clear to 1.
    Forward,
    /// Near = 1, far = 0; strict `Greater` test and clear to 0.
    Reverse,
}

impl SceneIdDepth {
    pub(crate) const fn clear_value(self) -> f32 {
        match self {
            Self::Forward => 1.0,
            Self::Reverse => 0.0,
        }
    }

    const fn compare(self) -> wgpu::CompareFunction {
        match self {
            Self::Forward => wgpu::CompareFunction::Less,
            Self::Reverse => wgpu::CompareFunction::Greater,
        }
    }
}

/// Host-computed camera tied to an exact target extent and camera generation.
/// Advance the viewport generation before changing this matrix or convention.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SceneIdCamera {
    /// Allocation, extent and viewport/camera generation used for this camera.
    pub target: RenderTargetState,
    /// World-to-clip matrix; object-to-world comes from the instance transform.
    pub view_projection: SceneIdMatrix,
    /// Depth convention matching the projection matrix.
    pub depth: SceneIdDepth,
}

/// Rejected geometry, resource mapping or frame/camera correlation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SceneIdGeometryError {
    /// Positions and triangle indices must both be nonempty.
    EmptyGeometry,
    /// Index count must be divisible by three.
    IncompleteTriangle,
    /// A position contains NaN or infinity.
    NonFiniteVertex {
        /// Position index.
        vertex: usize,
    },
    /// A model or camera matrix contains NaN or infinity.
    NonFiniteMatrix,
    /// An index refers beyond the uploaded position array.
    IndexOutOfBounds {
        /// Invalid index value.
        index: u32,
        /// Available positions.
        vertex_count: usize,
    },
    /// Geometry exceeds native buffer/draw count limits.
    GeometryCapacityExceeded,
    /// A visible draw has no uploaded geometry.
    MissingGeometry(u64),
    /// A visible draw has no uploaded model transform.
    MissingTransform(u64),
    /// Retained sink entries differ from the current scene. Apply commits in
    /// order or rebuild with `synchronize` before rendering.
    SceneOutOfSync,
    /// Frame was captured before a scene/resource change.
    SceneGenerationMismatch,
    /// Camera was captured for another viewport/generation/target/extent.
    CameraMismatch,
    /// Scene generation cannot advance for a resource change.
    Scene(SceneIdPassError),
}

impl fmt::Display for SceneIdGeometryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "scene ID geometry rejected: {self:?}")
    }
}

impl core::error::Error for SceneIdGeometryError {}

struct Geometry {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_count: u32,
}

struct Draw {
    entry: SceneIdPassEntry,
    identity: wgpu::BindGroup,
}

/// Optional concrete `SceneIdPassSink`: retained position/index buffers, model
/// transforms, full-identity tokens and real depth-tested triangle rasterization.
/// Create one per logical scene on the same device as its viewport targets.
/// No ECS, scene traversal, camera-math or material dependency is required.
pub struct SceneIdGeometryRenderer {
    device: wgpu::Device,
    matrix_layout: wgpu::BindGroupLayout,
    identity_layout: wgpu::BindGroupLayout,
    forward: wgpu::RenderPipeline,
    reverse: wgpu::RenderPipeline,
    geometry: BTreeMap<u64, Geometry>,
    transforms: BTreeMap<u64, wgpu::BindGroup>,
    draws: BTreeMap<SceneObjectId, Draw>,
}

impl SceneIdGeometryRenderer {
    /// Creates the two single-sample `Rgba32Uint`/`Depth32Float` pipelines.
    #[must_use]
    pub fn new(device: &wgpu::Device) -> Self {
        let matrix_layout = uniform_layout(device, 64, wgpu::ShaderStages::VERTEX);
        let identity_layout = uniform_layout(device, 16, wgpu::ShaderStages::FRAGMENT);
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("prism-editor-geometry-layout"),
            bind_group_layouts: &[
                Some(&matrix_layout),
                Some(&matrix_layout),
                Some(&identity_layout),
            ],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("prism-editor-geometry-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("geometry.wgsl").into()),
        });
        let forward = pipeline(device, &layout, &shader, SceneIdDepth::Forward);
        let reverse = pipeline(device, &layout, &shader, SceneIdDepth::Reverse);
        Self {
            device: device.clone(),
            matrix_layout,
            identity_layout,
            forward,
            reverse,
            geometry: BTreeMap::new(),
            transforms: BTreeMap::new(),
            draws: BTreeMap::new(),
        }
    }

    /// Uploads/replaces an indexed triangle-list mesh for `draw_key`. Positions
    /// are object-space Float32x3; indices are Uint32. Replacements affect every
    /// instance using this key. Success invalidates outstanding scene readbacks;
    /// invalid input leaves both the scene and retained resources unchanged.
    pub fn upload_geometry(
        &mut self,
        scene: &mut SceneIdPass,
        draw_key: u64,
        positions: &[[f32; 3]],
        indices: &[u32],
    ) -> Result<(), SceneIdGeometryError> {
        validate_geometry(positions, indices, self.device.limits().max_buffer_size)?;
        let vertices: Vec<_> = positions
            .iter()
            .flatten()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let index_bytes: Vec<_> = indices
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let geometry = Geometry {
            vertices: buffer(&self.device, &vertices, wgpu::BufferUsages::VERTEX),
            indices: buffer(&self.device, &index_bytes, wgpu::BufferUsages::INDEX),
            index_count: indices.len() as u32,
        };
        scene
            .invalidate_content()
            .map_err(SceneIdGeometryError::Scene)?;
        self.geometry.insert(draw_key, geometry);
        Ok(())
    }

    /// Uploads/replaces an object-to-world transform for `instance_key` and
    /// invalidates scene readbacks. A key may be shared by multiple subobjects.
    pub fn set_transform(
        &mut self,
        scene: &mut SceneIdPass,
        instance_key: u64,
        model: SceneIdMatrix,
    ) -> Result<(), SceneIdGeometryError> {
        validate_matrix(&model)?;
        let binding = uniform(&self.device, &self.matrix_layout, &matrix_bytes(&model));
        scene
            .invalidate_content()
            .map_err(SceneIdGeometryError::Scene)?;
        self.transforms.insert(instance_key, binding);
        Ok(())
    }

    /// Releases a mesh. Visible references will reject rendering until replaced.
    /// Returns false without advancing the generation if the key is absent.
    pub fn remove_geometry(
        &mut self,
        scene: &mut SceneIdPass,
        draw_key: u64,
    ) -> Result<bool, SceneIdGeometryError> {
        if !self.geometry.contains_key(&draw_key) {
            return Ok(false);
        }
        scene
            .invalidate_content()
            .map_err(SceneIdGeometryError::Scene)?;
        self.geometry.remove(&draw_key);
        Ok(true)
    }

    /// Releases a transform, with the same invalidation policy as mesh removal.
    pub fn remove_transform(
        &mut self,
        scene: &mut SceneIdPass,
        instance_key: u64,
    ) -> Result<bool, SceneIdGeometryError> {
        if !self.transforms.contains_key(&instance_key) {
            return Ok(false);
        }
        scene
            .invalidate_content()
            .map_err(SceneIdGeometryError::Scene)?;
        self.transforms.remove(&instance_key);
        Ok(true)
    }

    /// Rebuilds sink entries from a complete scene (initial attach/recovery).
    /// Mesh and transform uploads are retained; normal updates use `record_commit`.
    pub fn synchronize(&mut self, scene: &SceneIdPass) {
        self.draws.clear();
        for entry in scene.entries() {
            self.retain(*entry);
        }
    }

    /// Clears and draws all visible indexed geometry into an existing frame's
    /// integer/depth targets. Validates everything before modifying the encoder.
    /// Camera and scene generations must already match the current frame.
    /// Resource uploads are retained; only the camera uniform is created here.
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        frame: &GpuPickingFrame<'_>,
        scene: &SceneIdPass,
        camera: &SceneIdCamera,
    ) -> Result<(), SceneIdGeometryError> {
        if frame.scene_generation() != scene.generation() {
            return Err(SceneIdGeometryError::SceneGenerationMismatch);
        }
        if camera.target != frame.state() {
            return Err(SceneIdGeometryError::CameraMismatch);
        }
        validate_matrix(&camera.view_projection)?;
        if self.draws.len() != scene.entries().count()
            || scene.entries().any(|entry| {
                self.draws.get(&entry.item().id).map(|draw| draw.entry) != Some(*entry)
            })
        {
            return Err(SceneIdGeometryError::SceneOutOfSync);
        }
        let entries = scene.draw_entries();
        for entry in &entries {
            let item = entry.item();
            if !self.geometry.contains_key(&item.draw_key) {
                return Err(SceneIdGeometryError::MissingGeometry(item.draw_key));
            }
            if !self.transforms.contains_key(&item.instance_key) {
                return Err(SceneIdGeometryError::MissingTransform(item.instance_key));
            }
        }
        let camera_binding = uniform(
            &self.device,
            &self.matrix_layout,
            &matrix_bytes(&camera.view_projection),
        );
        let mut pass = frame.begin_geometry_render_pass(
            encoder,
            camera.depth,
            Some("prism-editor-scene-geometry-id"),
        );
        pass.set_pipeline(match camera.depth {
            SceneIdDepth::Forward => &self.forward,
            SceneIdDepth::Reverse => &self.reverse,
        });
        pass.set_bind_group(0, &camera_binding, &[]);
        for entry in entries {
            let item = entry.item();
            let geometry = &self.geometry[&item.draw_key];
            pass.set_bind_group(1, &self.transforms[&item.instance_key], &[]);
            pass.set_bind_group(2, &self.draws[&item.id].identity, &[]);
            pass.set_vertex_buffer(0, geometry.vertices.slice(..));
            pass.set_index_buffer(geometry.indices.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..geometry.index_count, 0, 0..1);
        }
        Ok(())
    }

    fn retain(&mut self, entry: SceneIdPassEntry) {
        if let Some(draw) = self.draws.get_mut(&entry.item().id)
            && draw.entry.encoded_id() == entry.encoded_id()
        {
            draw.entry = entry;
            return;
        }
        let bytes: Vec<_> = entry
            .encoded_id()
            .channels()
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect();
        let identity = uniform(&self.device, &self.identity_layout, &bytes);
        self.draws.insert(entry.item().id, Draw { entry, identity });
    }
}

impl SceneIdPassSink for SceneIdGeometryRenderer {
    type Error = Infallible;

    fn add(&mut self, entry: SceneIdPassEntry) -> Result<(), Self::Error> {
        self.retain(entry);
        Ok(())
    }

    fn update(&mut self, entry: SceneIdPassEntry) -> Result<(), Self::Error> {
        self.retain(entry);
        Ok(())
    }

    fn remove(&mut self, id: SceneObjectId) -> Result<(), Self::Error> {
        self.draws.remove(&id);
        Ok(())
    }
}

fn validate_geometry(
    positions: &[[f32; 3]],
    indices: &[u32],
    max_buffer_size: u64,
) -> Result<(), SceneIdGeometryError> {
    if positions.is_empty() || indices.is_empty() {
        return Err(SceneIdGeometryError::EmptyGeometry);
    }
    if !indices.len().is_multiple_of(3) {
        return Err(SceneIdGeometryError::IncompleteTriangle);
    }
    if indices.len() > u32::MAX as usize
        || positions.len() > u32::MAX as usize
        || positions.len() as u64 > max_buffer_size / 12
        || indices.len() as u64 > max_buffer_size / 4
    {
        return Err(SceneIdGeometryError::GeometryCapacityExceeded);
    }
    for (vertex, position) in positions.iter().enumerate() {
        if position.iter().any(|value| !value.is_finite()) {
            return Err(SceneIdGeometryError::NonFiniteVertex { vertex });
        }
    }
    for &index in indices {
        if index as usize >= positions.len() {
            return Err(SceneIdGeometryError::IndexOutOfBounds {
                index,
                vertex_count: positions.len(),
            });
        }
    }
    Ok(())
}

fn validate_matrix(matrix: &SceneIdMatrix) -> Result<(), SceneIdGeometryError> {
    if matrix.iter().flatten().any(|value| !value.is_finite()) {
        return Err(SceneIdGeometryError::NonFiniteMatrix);
    }
    Ok(())
}

fn matrix_bytes(matrix: &SceneIdMatrix) -> Vec<u8> {
    matrix
        .iter()
        .flatten()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn buffer(device: &wgpu::Device, bytes: &[u8], usage: wgpu::BufferUsages) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("prism-editor-geometry-buffer"),
        contents: bytes,
        usage,
    })
}

fn uniform_layout(
    device: &wgpu::Device,
    size: u64,
    visibility: wgpu::ShaderStages,
) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("prism-editor-geometry-uniform-layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(size),
            },
            count: None,
        }],
    })
}

fn uniform(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, bytes: &[u8]) -> wgpu::BindGroup {
    let buffer = buffer(device, bytes, wgpu::BufferUsages::UNIFORM);
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("prism-editor-geometry-uniform"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    })
}

fn pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    depth: SceneIdDepth,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("prism-editor-geometry-id-pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers: &[Some(wgpu::VertexBufferLayout {
                array_stride: 12,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                }],
            })],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(true),
            depth_compare: Some(depth.compare()),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: ID_BUFFER_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_validation_is_finite_indexed_triangle_list_and_bounded() {
        let positions = [[0.0; 3]; 3];
        assert_eq!(
            validate_geometry(&[], &[0, 0, 0], 64),
            Err(SceneIdGeometryError::EmptyGeometry)
        );
        assert_eq!(
            validate_geometry(&positions, &[0, 1], 64),
            Err(SceneIdGeometryError::IncompleteTriangle)
        );
        assert_eq!(
            validate_geometry(&positions, &[0, 1, 3], 64),
            Err(SceneIdGeometryError::IndexOutOfBounds {
                index: 3,
                vertex_count: 3
            })
        );
        assert_eq!(
            validate_geometry(&positions, &[0, 1, 2], 32),
            Err(SceneIdGeometryError::GeometryCapacityExceeded)
        );
        assert_eq!(
            validate_geometry(&[[f32::NAN; 3]], &[0, 0, 0], 64),
            Err(SceneIdGeometryError::NonFiniteVertex { vertex: 0 })
        );
        assert_eq!(validate_geometry(&positions, &[0, 1, 2], 64), Ok(()));
        assert_eq!(
            validate_matrix(&[[f32::INFINITY; 4]; 4]),
            Err(SceneIdGeometryError::NonFiniteMatrix)
        );
    }
}
