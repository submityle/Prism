//! Versioned cluster payload codec with local `u16` position quantization.

use alloc::vec::Vec;

use prism_render_architecture::gpu_scene::SceneBounds;

/// Cluster payload signature.
pub const STATIC_CLUSTER_PAYLOAD_MAGIC: [u8; 4] = *b"CLST";
/// Quantized position payload schema.
pub const STATIC_CLUSTER_PAYLOAD_VERSION: u16 = 2;

/// Decoded quantized cluster payload.
#[derive(Clone, Debug, PartialEq)]
pub struct QuantizedClusterPayload {
    /// Positions reconstructed from local `u16` quantization.
    pub positions: Vec<[f32; 3]>,
    /// Local triangle indices.
    pub indices: Vec<u16>,
    /// Conservative upper bound on squared position quantization error.
    // The original unquantized positions are not stored in the payload, so the
    // decoder cannot measure the realized per-vertex error. This value is the
    // worst-case half-step bound for the three quantized axes.
    pub max_error_squared: f32,
}

/// Payload codec failures.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ClusterPayloadError {
    /// Signature or schema did not match this codec.
    InvalidHeader,
    /// Payload ended before a declared field.
    Truncated,
    /// Counts or values exceed the fixed cluster ABI.
    InvalidCounts,
    /// A decoded position or bound is non-finite.
    NonFinite,
    /// Payload bounds disagree with the caller's cluster bounds.
    BoundsMismatch,
}

impl core::fmt::Display for ClusterPayloadError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidHeader => f.write_str("invalid cluster payload header"),
            Self::Truncated => f.write_str("truncated cluster payload"),
            Self::InvalidCounts => f.write_str("invalid cluster payload counts"),
            Self::NonFinite => f.write_str("cluster payload contains non-finite data"),
            Self::BoundsMismatch => f.write_str("cluster payload bounds do not match metadata"),
        }
    }
}

/// Encodes local positions as normalized `u16x3` values within `bounds`.
#[must_use]
pub fn encode_cluster_payload(
    bounds: SceneBounds,
    positions: &[[f32; 3]],
    indices: &[u16],
) -> Result<Vec<u8>, ClusterPayloadError> {
    if !bounds_finite(bounds)
        || positions
            .iter()
            .any(|position| !position.iter().all(|value| value.is_finite()))
    {
        return Err(ClusterPayloadError::NonFinite);
    }
    if positions.is_empty()
        || positions.len() > 64
        || indices.is_empty()
        || indices.len() % 3 != 0
        || indices.len() / 3 > 128
        || indices
            .iter()
            .any(|index| *index as usize >= positions.len())
    {
        return Err(ClusterPayloadError::InvalidCounts);
    }
    let (min, max) = bounds_min_max(bounds);
    if positions.iter().any(|position| {
        position
            .iter()
            .enumerate()
            .any(|(axis, value)| *value < min[axis] || *value > max[axis])
    }) {
        return Err(ClusterPayloadError::BoundsMismatch);
    }
    let mut bytes = Vec::with_capacity(10 + 24 + positions.len() * 6 + indices.len() * 2);
    bytes.extend_from_slice(&STATIC_CLUSTER_PAYLOAD_MAGIC);
    bytes.extend_from_slice(&STATIC_CLUSTER_PAYLOAD_VERSION.to_le_bytes());
    bytes.extend_from_slice(&(positions.len() as u16).to_le_bytes());
    bytes.extend_from_slice(&((indices.len() / 3) as u16).to_le_bytes());
    for value in min.into_iter().chain(max) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    for position in positions {
        for axis in 0..3 {
            let extent = max[axis] - min[axis];
            let normalized = if extent > 0.0 {
                ((position[axis] - min[axis]) / extent).clamp(0.0, 1.0)
            } else {
                0.0
            };
            bytes.extend_from_slice(&((normalized * u16::MAX as f32).round() as u16).to_le_bytes());
        }
    }
    for index in indices {
        bytes.extend_from_slice(&index.to_le_bytes());
    }
    Ok(bytes)
}

/// Decodes and reconstructs a quantized cluster payload.
pub fn decode_cluster_payload(
    bounds: SceneBounds,
    bytes: &[u8],
) -> Result<QuantizedClusterPayload, ClusterPayloadError> {
    if bytes.len() < 10 || bytes[0..4] != STATIC_CLUSTER_PAYLOAD_MAGIC {
        return Err(ClusterPayloadError::InvalidHeader);
    }
    let version = u16::from_le_bytes([bytes[4], bytes[5]]);
    if version != STATIC_CLUSTER_PAYLOAD_VERSION || !bounds_finite(bounds) {
        return Err(if version == STATIC_CLUSTER_PAYLOAD_VERSION {
            ClusterPayloadError::NonFinite
        } else {
            ClusterPayloadError::InvalidHeader
        });
    }
    let vertex_count = u16::from_le_bytes([bytes[6], bytes[7]]) as usize;
    let triangle_count = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    if vertex_count == 0 || vertex_count > 64 || triangle_count == 0 || triangle_count > 128 {
        return Err(ClusterPayloadError::InvalidCounts);
    }
    let header_end = 10 + 24;
    let position_end = header_end + vertex_count * 6;
    let index_end = position_end + triangle_count * 3 * 2;
    if bytes.len() != index_end {
        return Err(ClusterPayloadError::Truncated);
    }
    let mut cursor = 10;
    let mut stored_min = [0.0; 3];
    let mut stored_max = [0.0; 3];
    for value in stored_min.iter_mut().chain(stored_max.iter_mut()) {
        *value = f32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap());
        cursor += 4;
    }
    if !stored_min
        .iter()
        .chain(stored_max.iter())
        .all(|value| value.is_finite())
    {
        return Err(ClusterPayloadError::NonFinite);
    }
    let (expected_min, expected_max) = bounds_min_max(bounds);
    if stored_min != expected_min || stored_max != expected_max {
        return Err(ClusterPayloadError::BoundsMismatch);
    }
    let mut positions = Vec::with_capacity(vertex_count);
    for _ in 0..vertex_count {
        let mut position = [0.0; 3];
        for axis in 0..3 {
            let quantized = u16::from_le_bytes([bytes[cursor], bytes[cursor + 1]]);
            cursor += 2;
            let normalized = quantized as f32 / u16::MAX as f32;
            position[axis] = stored_min[axis] + (stored_max[axis] - stored_min[axis]) * normalized;
        }
        positions.push(position);
    }
    let quantization_step = [
        (stored_max[0] - stored_min[0]) / u16::MAX as f32,
        (stored_max[1] - stored_min[1]) / u16::MAX as f32,
        (stored_max[2] - stored_min[2]) / u16::MAX as f32,
    ];
    let max_error_squared = 0.25
        * (quantization_step[0] * quantization_step[0]
            + quantization_step[1] * quantization_step[1]
            + quantization_step[2] * quantization_step[2]);
    let mut indices = Vec::with_capacity(triangle_count * 3);
    for _ in 0..triangle_count * 3 {
        let index = u16::from_le_bytes([bytes[cursor], bytes[cursor + 1]]);
        cursor += 2;
        if index as usize >= vertex_count {
            return Err(ClusterPayloadError::InvalidCounts);
        }
        indices.push(index);
    }
    Ok(QuantizedClusterPayload {
        positions,
        indices,
        max_error_squared,
    })
}

fn bounds_min_max(bounds: SceneBounds) -> ([f32; 3], [f32; 3]) {
    (
        [
            bounds.center[0] - bounds.half_extents[0],
            bounds.center[1] - bounds.half_extents[1],
            bounds.center[2] - bounds.half_extents[2],
        ],
        [
            bounds.center[0] + bounds.half_extents[0],
            bounds.center[1] + bounds.half_extents[1],
            bounds.center[2] + bounds.half_extents[2],
        ],
    )
}

fn bounds_finite(bounds: SceneBounds) -> bool {
    if !bounds
        .center
        .iter()
        .chain(bounds.half_extents.iter())
        .chain(core::iter::once(&bounds.radius))
        .all(|value| value.is_finite())
        || bounds.radius < 0.0
        || bounds.half_extents.iter().any(|value| *value < 0.0)
    {
        return false;
    }

    let (min, max) = bounds_min_max(bounds);
    min.iter().chain(max.iter()).all(|value| value.is_finite())
        && (0..3).all(|axis| (max[axis] - min[axis]).is_finite())
}

#[cfg(test)]
fn distance_squared(a: [f32; 3], b: [f32; 3]) -> f32 {
    let delta = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bounds() -> SceneBounds {
        SceneBounds {
            center: [1.0, 2.0, 3.0],
            radius: 2.0,
            half_extents: [1.0, 2.0, 3.0],
            _padding: 0.0,
        }
    }

    #[test]
    fn quantized_payload_round_trips_with_bounded_error() {
        let positions = [[0.0, 0.0, 0.0], [2.0, 2.0, 3.0], [1.0, 4.0, 6.0]];
        let bytes = encode_cluster_payload(bounds(), &positions, &[0, 1, 2]).expect("encodes");
        let decoded = decode_cluster_payload(bounds(), &bytes).expect("decodes");
        assert_eq!(decoded.indices, vec![0, 1, 2]);
        assert_eq!(decoded.positions.len(), positions.len());
        let maximum_error_squared = 0.25
            * ((2.0 / u16::MAX as f32).powi(2)
                + (4.0 / u16::MAX as f32).powi(2)
                + (6.0 / u16::MAX as f32).powi(2));
        assert!((decoded.max_error_squared - maximum_error_squared).abs() <= f32::EPSILON);
        for (expected, actual) in positions.into_iter().zip(decoded.positions) {
            assert!(distance_squared(expected, actual) <= decoded.max_error_squared);
        }
    }

    #[test]
    fn rejects_positions_outside_declared_bounds_instead_of_clamping() {
        for position in [
            [-0.001, 2.0, 3.0],
            [2.001, 2.0, 3.0],
            [1.0, 4.001, 3.0],
            [1.0, 2.0, 6.001],
        ] {
            let result = encode_cluster_payload(bounds(), &[position], &[0, 0, 0]);
            assert_eq!(result, Err(ClusterPayloadError::BoundsMismatch));
        }
    }

    #[test]
    fn non_finite_positions_report_non_finite_error() {
        let result = encode_cluster_payload(bounds(), &[[f32::NAN, 2.0, 3.0]], &[0, 0, 0]);
        assert_eq!(result, Err(ClusterPayloadError::NonFinite));
    }
}
