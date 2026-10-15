//! Validated, device-independent batch deformation and its storage-buffer ABI.
//!
//! Order is bind pose → morph → linear-blend skinning → external correction.
//! Morphs are transposed once into per-vertex CSR spans: sparse targets never
//! expand into a vertex × target allocation. Skin weights are normalized once;
//! an empty or zero-weight list preserves the morphed bind pose. CPU evaluation
//! is a reference/fallback, not a substitute for GPU execution evidence.

use alloc::{vec, vec::Vec};
use core::fmt;

use bytemuck::{Pod, Zeroable};

use crate::{MorphError, MorphTarget, MorphTargetData, SkinError, SkinningData};

/// Invalid authored data, profile, or animation input.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeformationError {
    /// A stream has the wrong number of elements.
    CountMismatch {
        /// Stream being checked.
        field: &'static str,
        /// Required element count.
        expected: usize,
        /// Supplied element count.
        actual: usize,
    },
    /// A component is NaN or infinity.
    NonFinite {
        /// Stream being checked.
        field: &'static str,
        /// Element containing the component.
        index: usize,
    },
    /// Only the explicit four- and eight-influence profiles are supported.
    InvalidProfile,
    /// A count or offset does not fit the u32 shader ABI.
    SizeOverflow,
    /// Invalid morph representation.
    Morph(MorphError),
    /// Invalid skin representation or too many influences (never truncated).
    Skin(SkinError),
    /// A joint has no finite inverse-transpose normal transform.
    SingularJoint,
    /// Deserialized joint rows disagree with their inverse-transpose rows.
    InvalidJoint,
    /// Reserved correction components are not zero.
    ReservedCorrection,
    /// Conservative pose bounds cannot guarantee finite f32 GPU arithmetic.
    ArithmeticRange,
}

impl fmt::Display for DeformationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid mesh deformation: {self:?}")
    }
}

impl std::error::Error for DeformationError {}

/// One bind-pose vertex and its morph/skin CSR spans (48-byte shader stride).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct DeformationVertex {
    /// Bind position; homogeneous component is one.
    pub position: [f32; 4],
    /// Bind normal, or zero when no normals were supplied.
    pub normal: [f32; 4],
    /// Morph offset/count followed by skin offset/count.
    pub spans: [u32; 4],
}

/// One nonzero morph delta and its dynamic weight index (16-byte stride).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct MorphContribution {
    /// Local position delta.
    pub delta: [f32; 3],
    /// Target weight index.
    pub target: u32,
}

/// One normalized linear-blend skin influence (8-byte stride).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct JointWeight {
    /// Palette index.
    pub joint: u32,
    /// Normalized nonnegative weight.
    pub weight: f32,
}

/// Affine position rows and inverse-transpose normal rows (96-byte stride).
///
/// Constructing the transform rejects singular/nonfinite matrices before any
/// GPU submission. Translation is ignored for normals; nonuniform scaling and
/// reflections use the same inverse-transpose rule as the CPU reference.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct JointTransform {
    rows: [[f32; 4]; 6],
}

impl JointTransform {
    /// Constructs from three row-major affine position rows (translation in w).
    pub fn new(rows: [[f32; 4]; 3]) -> Result<Self, DeformationError> {
        finite("joint", 0, rows.iter().flatten().copied())?;
        // f64 intermediates avoid rejecting a valid small/large uniform scale
        // just because its f32 determinant underflows/overflows.
        let a = rows[0].map(f64::from);
        let b = rows[1].map(f64::from);
        let c = rows[2].map(f64::from);
        let cofactor = [
            [
                b[1] * c[2] - b[2] * c[1],
                b[2] * c[0] - b[0] * c[2],
                b[0] * c[1] - b[1] * c[0],
            ],
            [
                c[1] * a[2] - c[2] * a[1],
                c[2] * a[0] - c[0] * a[2],
                c[0] * a[1] - c[1] * a[0],
            ],
            [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ],
        ];
        let det = (0..3).map(|i| a[i] * cofactor[0][i]).sum::<f64>();
        if det == 0.0 {
            return Err(DeformationError::SingularJoint);
        }
        let mut result = [[0.0; 4]; 6];
        result[..3].copy_from_slice(&rows);
        for i in 0..3 {
            for j in 0..3 {
                let value = (cofactor[i][j] / det) as f32;
                if !value.is_finite() {
                    return Err(DeformationError::SingularJoint);
                }
                result[i + 3][j] = value;
            }
        }
        Ok(Self { rows: result })
    }

    /// Identity palette entry.
    pub fn identity() -> Self {
        Self::new([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ])
        .expect("identity is invertible")
    }
}

/// Additive post-skin correction, e.g. from a cloth solver (32-byte stride).
/// The w components are reserved and must be zero.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct DeformationCorrection {
    /// XYZ position delta in the same local space as skinned output.
    pub position: [f32; 4],
    /// XYZ normal delta, applied before final normalization.
    pub normal: [f32; 4],
}

/// Shared compute/vertex ABI: xyz position and xyz normal in a 32-byte record.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct DeformedVertex {
    /// Position (w = 1).
    pub position: [f32; 4],
    /// Unit normal (w = 0), or zero for a missing/degenerate normal.
    pub normal: [f32; 4],
}

/// Borrowed animation inputs; validation precedes all GPU writes.
#[derive(Clone, Copy, Debug)]
pub struct DeformationPose<'a> {
    /// One finite weight for each authored morph target; signed weights allowed.
    pub morph_weights: &'a [f32],
    /// Palette including inverse bind transforms, in mesh-local output space.
    pub joints: &'a [JointTransform],
    /// Optional post-skin deltas, one per vertex when present.
    pub corrections: Option<&'a [DeformationCorrection]>,
}

/// Immutable, validated static inputs, reusable across poses and instances.
#[derive(Clone, Debug)]
pub struct DeformationMesh {
    vertices: Vec<DeformationVertex>,
    morphs: Vec<MorphContribution>,
    influences: Vec<JointWeight>,
    target_count: u32,
    joint_count: u32,
    position_extent: [f64; 3],
    normal_extent: [f64; 3],
    morph_extents: Vec<[f64; 3]>,
}

impl DeformationMesh {
    /// Prepares static streams without silently dropping attributes/influences.
    /// Normals are optional; their absence produces zero output normals.
    pub fn new(
        positions: &[[f32; 3]],
        normals: Option<&[[f32; 3]]>,
        targets: &[MorphTarget],
        skin: Option<&SkinningData>,
        max_influences: usize,
    ) -> Result<Self, DeformationError> {
        if !matches!(max_influences, 4 | 8) {
            return Err(DeformationError::InvalidProfile);
        }
        let count = u32_count(positions.len())?;
        let target_count = u32_count(targets.len())?;
        if let Some(normals) = normals {
            same_count("normals", positions.len(), normals.len())?;
        }
        let mut vertices = Vec::with_capacity(positions.len());
        for (i, &position) in positions.iter().enumerate() {
            let normal = normals.map_or([0.0; 3], |n| n[i]);
            finite("positions", i, position)?;
            finite("normals", i, normal)?;
            vertices.push(DeformationVertex {
                position: [position[0], position[1], position[2], 1.0],
                normal: [normal[0], normal[1], normal[2], 0.0],
                spans: [0; 4],
            });
        }
        for target in targets {
            target.validate().map_err(DeformationError::Morph)?;
            same_count(
                "morph vertices",
                positions.len(),
                target.vertex_count() as usize,
            )?;
            visit_target(target, |i, delta| {
                if delta != [0.0; 3] {
                    vertices[i].spans[1] += 1;
                }
            });
        }
        let mut total = 0_u32;
        for vertex in &mut vertices {
            vertex.spans[0] = total;
            total = total
                .checked_add(vertex.spans[1])
                .ok_or(DeformationError::SizeOverflow)?;
        }
        let mut morphs = vec![MorphContribution::zeroed(); total as usize];
        let mut cursors: Vec<u32> = vertices.iter().map(|v| v.spans[0]).collect();
        for (target_index, target) in targets.iter().enumerate() {
            visit_target(target, |i, delta| {
                if delta != [0.0; 3] {
                    morphs[cursors[i] as usize] = MorphContribution {
                        delta,
                        target: target_index as u32,
                    };
                    cursors[i] += 1;
                }
            });
        }
        let mut influences = Vec::new();
        let joint_count = skin.map_or(0, SkinningData::joint_count);
        if let Some(skin) = skin {
            skin.validate_with_max_influences(max_influences)
                .map_err(DeformationError::Skin)?;
            same_count(
                "skin vertices",
                count as usize,
                skin.vertex_count() as usize,
            )?;
            for (vertex, list) in vertices.iter_mut().zip(skin.iter()) {
                vertex.spans[2] = u32_count(influences.len())?;
                let sum = list.iter().map(|v| v.weight).sum::<f32>();
                if sum > 0.0 {
                    for value in list.iter().filter(|v| v.weight > 0.0) {
                        influences.push(JointWeight {
                            joint: value.joint_index,
                            weight: value.weight / sum,
                        });
                    }
                }
                vertex.spans[3] = u32_count(influences.len())? - vertex.spans[2];
            }
        }
        let mut position_extent = [0.0_f64; 3];
        let mut normal_extent = [0.0_f64; 3];
        for vertex in &vertices {
            for axis in 0..3 {
                position_extent[axis] =
                    position_extent[axis].max(f64::from(vertex.position[axis]).abs());
                normal_extent[axis] = normal_extent[axis].max(f64::from(vertex.normal[axis]).abs());
            }
        }
        let mut morph_extents = vec![[0.0_f64; 3]; targets.len()];
        for value in &morphs {
            for axis in 0..3 {
                let extent = &mut morph_extents[value.target as usize][axis];
                *extent = extent.max(f64::from(value.delta[axis]).abs());
            }
        }
        Ok(Self {
            vertices,
            morphs,
            influences,
            target_count,
            joint_count,
            position_extent,
            normal_extent,
            morph_extents,
        })
    }

    /// Static vertex records in the shared storage ABI.
    pub fn vertices(&self) -> &[DeformationVertex] {
        &self.vertices
    }
    /// CSR morph contributions; only nonzero deltas are stored.
    pub fn morphs(&self) -> &[MorphContribution] {
        &self.morphs
    }
    /// CSR skin influences, normalized and stripped of zero weights.
    pub fn influences(&self) -> &[JointWeight] {
        &self.influences
    }
    /// Number of output vertices.
    pub fn vertex_count(&self) -> u32 {
        self.vertices.len() as u32
    }
    /// Number of dynamic target weights required, including empty targets.
    pub fn target_count(&self) -> u32 {
        self.target_count
    }
    /// Number of palette entries required.
    pub fn joint_count(&self) -> u32 {
        self.joint_count
    }

    /// Checks the complete pose before a consumer mutates output/history.
    pub fn validate_pose(&self, pose: DeformationPose<'_>) -> Result<(), DeformationError> {
        same_count(
            "morph weights",
            self.target_count as usize,
            pose.morph_weights.len(),
        )?;
        same_count(
            "joint palette",
            self.joint_count as usize,
            pose.joints.len(),
        )?;
        for (i, &weight) in pose.morph_weights.iter().enumerate() {
            finite("morph weights", i, [weight])?;
        }
        // Pod permits all bit patterns, so validate even transforms obtained
        // via byte deserialization, not just those produced by `new`.
        for (i, joint) in pose.joints.iter().enumerate() {
            finite("joint palette", i, joint.rows.iter().flatten().copied())?;
            let expected = JointTransform::new([joint.rows[0], joint.rows[1], joint.rows[2]])?;
            if expected.rows != joint.rows {
                return Err(DeformationError::InvalidJoint);
            }
        }
        if let Some(corrections) = pose.corrections {
            same_count("corrections", self.vertices.len(), corrections.len())?;
            for (i, value) in corrections.iter().enumerate() {
                finite(
                    "corrections",
                    i,
                    value.position.into_iter().chain(value.normal),
                )?;
                if value.position[3] != 0.0 || value.normal[3] != 0.0 {
                    return Err(DeformationError::ReservedCorrection);
                }
            }
        }
        self.validate_arithmetic_range(pose)
    }

    // O(targets + joints + corrections), not a CPU skinning pass. Absolute
    // envelopes include intermediate products/sums, even when the final value
    // would cancel. Spare exponent range absorbs f32 rounding and normalized
    // weight accumulation. A rejected pose never reaches GPU queue writes.
    fn validate_arithmetic_range(&self, pose: DeformationPose<'_>) -> Result<(), DeformationError> {
        if self.vertices.is_empty() {
            return Ok(());
        }
        let limit = f64::from(f32::MAX) * 0.25;
        let check = |values: [f64; 3]| {
            if values.into_iter().all(|v| v <= limit) {
                Ok(())
            } else {
                Err(DeformationError::ArithmeticRange)
            }
        };
        let mut position = self.position_extent;
        for (extent, &weight) in self.morph_extents.iter().zip(pose.morph_weights) {
            for axis in 0..3 {
                position[axis] += extent[axis] * f64::from(weight).abs();
            }
        }
        check(position)?;
        let mut normal = self.normal_extent;
        let mut skinned_position = position;
        check(normal)?;
        // Include unskinned vertices as well as every palette entry. Palette
        // entries unused by a particular mesh can therefore cause rejection;
        // this is deliberate conservative validation, not silent saturation.
        for joint in pose.joints {
            for axis in 0..3 {
                let p = (0..3)
                    .map(|j| f64::from(joint.rows[axis][j]).abs() * position[j])
                    .sum::<f64>()
                    + f64::from(joint.rows[axis][3]).abs();
                let n = (0..3)
                    .map(|j| f64::from(joint.rows[axis + 3][j]).abs() * self.normal_extent[j])
                    .sum::<f64>();
                if p > limit || n > limit {
                    return Err(DeformationError::ArithmeticRange);
                }
                // Position's bind/morph envelope is kept unchanged while all
                // joints are checked, so palette order has no effect.
                normal[axis] = normal[axis].max(n);
                skinned_position[axis] = skinned_position[axis].max(p);
            }
        }
        if let Some(corrections) = pose.corrections {
            for value in corrections {
                check(core::array::from_fn(|axis| {
                    skinned_position[axis] + f64::from(value.position[axis]).abs()
                }))?;
                check(core::array::from_fn(|axis| {
                    normal[axis] + f64::from(value.normal[axis]).abs()
                }))?;
            }
        }
        Ok(())
    }

    /// Evaluates a whole mesh with the same order/ABI as the compute kernel.
    pub fn evaluate(
        &self,
        pose: DeformationPose<'_>,
    ) -> Result<Vec<DeformedVertex>, DeformationError> {
        self.validate_pose(pose)?;
        let mut output = Vec::with_capacity(self.vertices.len());
        for (i, vertex) in self.vertices.iter().enumerate() {
            let mut position = vertex.position;
            for value in &self.morphs[span(vertex.spans[0], vertex.spans[1])] {
                let weight = pose.morph_weights[value.target as usize];
                for axis in 0..3 {
                    position[axis] += value.delta[axis] * weight;
                }
            }
            let mut normal = vertex.normal;
            if vertex.spans[3] != 0 {
                let mut skinned = [0.0; 4];
                let mut skinned_normal = [0.0; 4];
                for influence in &self.influences[span(vertex.spans[2], vertex.spans[3])] {
                    let joint = &pose.joints[influence.joint as usize];
                    for axis in 0..3 {
                        skinned[axis] += dot(joint.rows[axis], position) * influence.weight;
                        skinned_normal[axis] +=
                            dot(joint.rows[axis + 3], normal) * influence.weight;
                    }
                }
                position = skinned;
                normal = skinned_normal;
            }
            if let Some(corrections) = pose.corrections {
                for axis in 0..3 {
                    position[axis] += corrections[i].position[axis];
                    normal[axis] += corrections[i].normal[axis];
                }
            }
            finite("deformed positions", i, position)?;
            finite("deformed normals", i, normal)?;
            position[3] = 1.0;
            normal = normalize(normal);
            output.push(DeformedVertex { position, normal });
        }
        Ok(output)
    }
}

fn span(start: u32, len: u32) -> core::ops::Range<usize> {
    start as usize..start as usize + len as usize
}
fn dot(a: [f32; 4], b: [f32; 4]) -> f32 {
    (0..4).map(|i| a[i] * b[i]).sum()
}
fn normalize(n: [f32; 4]) -> [f32; 4] {
    // Scale first to avoid overflow/underflow when normal lengths are extreme.
    let scale = n[..3].iter().fold(0.0_f32, |a, b| a.max(b.abs()));
    if scale == 0.0 {
        return [0.0; 4];
    }
    let v = [n[0] / scale, n[1] / scale, n[2] / scale, 0.0];
    let len = dot(v, v).sqrt();
    [v[0] / len, v[1] / len, v[2] / len, 0.0]
}
fn same_count(field: &'static str, expected: usize, actual: usize) -> Result<(), DeformationError> {
    if expected == actual {
        Ok(())
    } else {
        Err(DeformationError::CountMismatch {
            field,
            expected,
            actual,
        })
    }
}
fn u32_count(count: usize) -> Result<u32, DeformationError> {
    u32::try_from(count).map_err(|_| DeformationError::SizeOverflow)
}
fn finite(
    field: &'static str,
    index: usize,
    values: impl IntoIterator<Item = f32>,
) -> Result<(), DeformationError> {
    if values.into_iter().all(f32::is_finite) {
        Ok(())
    } else {
        Err(DeformationError::NonFinite { field, index })
    }
}
fn visit_target(target: &MorphTarget, mut visit: impl FnMut(usize, [f32; 3])) {
    match target.data() {
        MorphTargetData::Dense(deltas) => {
            for (i, delta) in deltas.iter().enumerate() {
                visit(i, delta.position);
            }
        }
        MorphTargetData::Sparse { indices, deltas } => {
            for (&i, delta) in indices.iter().zip(deltas) {
                visit(i as usize, delta.position);
            }
        }
    }
}

#[cfg(test)]
mod tests;
