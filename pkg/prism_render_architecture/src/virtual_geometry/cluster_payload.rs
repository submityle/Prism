//! Versioned cluster payload codec with local `u16` position quantization.

use alloc::vec::Vec;

use crate::gpu_scene::SceneBounds;

mod attributes;
pub use attributes::{
    encode_cluster_payload_with_attributes, ClusterVertexAttribute, MAX_CLUSTER_ATTRIBUTES,
    MAX_CLUSTER_ATTRIBUTE_NAME_BYTES,
};

/// Cluster payload signature.
pub const STATIC_CLUSTER_PAYLOAD_MAGIC: [u8; 4] = *b"CLST";
/// Current quantized position and lossless attribute payload schema.
pub const STATIC_CLUSTER_PAYLOAD_VERSION: u16 = 3;
/// Legacy position-only schema accepted by the decoder and emitted by
/// [`encode_cluster_payload`].
pub const POSITION_ONLY_CLUSTER_PAYLOAD_VERSION: u16 = 2;

/// Decoded quantized cluster payload.
#[derive(Clone, Debug, PartialEq)]
pub struct QuantizedClusterPayload {
    /// Positions reconstructed from local `u16` quantization.
    pub positions: Vec<[f32; 3]>,
    /// Local triangle indices.
    pub indices: Vec<u16>,
    /// Authored shader location of `Vertex_Position` (zero in legacy v2).
    pub position_shader_location: u32,
    /// Losslessly preserved non-position attributes in shader-location order.
    pub attributes: Vec<ClusterVertexAttribute>,
    /// Conservative upper bound on squared position quantization error.
    // The original unquantized positions are not stored in the payload, so the
    // decoder cannot measure the realized per-vertex error. This value is the
    // worst-case half-step plus reconstruction rounding bound. Positive
    // infinity represents a finite coordinate range whose squared error cannot
    // fit in f32; it must never be interpreted as zero or as a passing budget.
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
    /// An attribute descriptor, semantic, format, or data length is invalid.
    InvalidAttribute,
    /// A shader location or semantic is repeated, including the position.
    DuplicateAttribute,
    /// Nonzero reserved/padding bytes, noncanonical ordering, or trailing data.
    InvalidFraming,
}

impl core::fmt::Display for ClusterPayloadError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidHeader => f.write_str("invalid cluster payload header"),
            Self::Truncated => f.write_str("truncated cluster payload"),
            Self::InvalidCounts => f.write_str("invalid cluster payload counts"),
            Self::NonFinite => f.write_str("cluster payload contains non-finite data"),
            Self::BoundsMismatch => f.write_str("cluster payload bounds do not match metadata"),
            Self::InvalidAttribute => f.write_str("invalid cluster vertex attribute"),
            Self::DuplicateAttribute => f.write_str("duplicate cluster vertex attribute"),
            Self::InvalidFraming => f.write_str("noncanonical cluster payload framing"),
        }
    }
}

/// Encodes a legacy v2 position-only payload as normalized `u16x3` values.
/// Use [`encode_cluster_payload_with_attributes`] for the current v3 schema.
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
    bytes.extend_from_slice(&POSITION_ONLY_CLUSTER_PAYLOAD_VERSION.to_le_bytes());
    bytes.extend_from_slice(&(positions.len() as u16).to_le_bytes());
    bytes.extend_from_slice(&((indices.len() / 3) as u16).to_le_bytes());
    for value in min.into_iter().chain(max) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    for position in positions {
        for axis in 0..3 {
            let extent = f64::from(max[axis]) - f64::from(min[axis]);
            let normalized = if extent > 0.0 {
                ((f64::from(position[axis]) - f64::from(min[axis])) / extent).clamp(0.0, 1.0)
            } else {
                0.0
            };
            bytes.extend_from_slice(
                &((normalized * f64::from(u16::MAX)).round() as u16).to_le_bytes(),
            );
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
    // The fixed header includes the magic/version/counts plus six f32 bounds
    // values. Check it before slicing any field so truncated untrusted input
    // returns an error instead of panicking.
    const FIXED_HEADER_SIZE: usize = 10 + 24;
    if bytes.len() < FIXED_HEADER_SIZE || bytes[0..4] != STATIC_CLUSTER_PAYLOAD_MAGIC {
        return Err(if bytes.len() < FIXED_HEADER_SIZE {
            ClusterPayloadError::Truncated
        } else {
            ClusterPayloadError::InvalidHeader
        });
    }
    let version = u16::from_le_bytes([bytes[4], bytes[5]]);
    if !matches!(
        version,
        POSITION_ONLY_CLUSTER_PAYLOAD_VERSION | STATIC_CLUSTER_PAYLOAD_VERSION
    ) || !bounds_finite(bounds)
    {
        return Err(
            if matches!(
                version,
                POSITION_ONLY_CLUSTER_PAYLOAD_VERSION | STATIC_CLUSTER_PAYLOAD_VERSION
            ) {
                ClusterPayloadError::NonFinite
            } else {
                ClusterPayloadError::InvalidHeader
            },
        );
    }
    let vertex_count = u16::from_le_bytes([bytes[6], bytes[7]]) as usize;
    let triangle_count = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    if vertex_count == 0 || vertex_count > 64 || triangle_count == 0 || triangle_count > 128 {
        return Err(ClusterPayloadError::InvalidCounts);
    }
    let header_end = FIXED_HEADER_SIZE;
    let position_end = header_end + vertex_count * 6;
    let index_end = position_end + triangle_count * 3 * 2;
    if bytes.len() < index_end {
        return Err(ClusterPayloadError::Truncated);
    }
    let (position_shader_location, attributes) = if version == POSITION_ONLY_CLUSTER_PAYLOAD_VERSION
    {
        if bytes.len() != index_end {
            return Err(ClusterPayloadError::InvalidFraming);
        }
        (0, Vec::new())
    } else {
        attributes::decode_attributes(vertex_count, bytes, index_end)?
    };
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
            let normalized = f64::from(quantized) / f64::from(u16::MAX);
            position[axis] = (f64::from(stored_min[axis])
                + (f64::from(stored_max[axis]) - f64::from(stored_min[axis])) * normalized)
                as f32;
        }
        positions.push(position);
    }
    let max_error_squared = cluster_position_error_squared(bounds)?;
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
        position_shader_location,
        attributes,
        max_error_squared,
    })
}

/// Conservative position displacement for the CLST local-u16 codec, including
/// f32 reconstruction rounding. Shared by offline error metadata and decoding.
pub fn cluster_position_error(bounds: SceneBounds) -> Result<f32, ClusterPayloadError> {
    Ok(round_up(position_error_squared_f64(bounds)?.sqrt()))
}

/// Squared displacement bound. Infinity denotes a valid coordinate range whose
/// squared error is not representable in f32; use [`cluster_position_error`]
/// when the unsquared distance is required.
pub fn cluster_position_error_squared(bounds: SceneBounds) -> Result<f32, ClusterPayloadError> {
    Ok(round_up(position_error_squared_f64(bounds)?))
}

fn position_error_squared_f64(bounds: SceneBounds) -> Result<f64, ClusterPayloadError> {
    if !bounds_finite(bounds) {
        return Err(ClusterPayloadError::NonFinite);
    }
    let (min, max) = bounds_min_max(bounds);
    Ok((0..3)
        .map(|axis| {
            let min = f64::from(min[axis]);
            let max = f64::from(max[axis]);
            if min == max {
                return 0.0;
            }
            let half_step = (max - min) / (2.0 * f64::from(u16::MAX));
            // Include conversion-to-f32 rounding, including subnormal coordinates.
            let rounding =
                min.abs().max(max.abs()) * f64::from(f32::EPSILON) + f64::from(f32::from_bits(1));
            (half_step + rounding).powi(2)
        })
        .sum())
}

fn round_up(value: f64) -> f32 {
    let rounded_error = value as f32;
    if f64::from(rounded_error) < value {
        f32::from_bits(rounded_error.to_bits() + 1)
    } else {
        rounded_error
    }
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

    #[test]
    fn truncated_fixed_header_is_rejected_without_panicking() {
        for length in 0..34 {
            let bytes = vec![0_u8; length];
            assert_eq!(
                decode_cluster_payload(bounds(), &bytes),
                Err(ClusterPayloadError::Truncated)
            );
        }
    }

    #[test]
    fn extreme_finite_coordinates_never_decode_to_nan_or_wrap_the_error_bound() {
        let bounds = SceneBounds {
            center: [0.0; 3],
            half_extents: [f32::MAX; 3],
            radius: f32::MAX,
            _padding: 0.0,
        };
        let source = [[-f32::MAX; 3], [f32::MAX; 3], [0.0; 3]];
        let bytes = encode_cluster_payload(bounds, &source, &[0, 1, 2]).unwrap();
        let decoded = decode_cluster_payload(bounds, &bytes).unwrap();
        assert_eq!(decoded.positions[0], source[0]);
        assert_eq!(decoded.positions[1], source[1]);
        assert!(decoded
            .positions
            .iter()
            .flatten()
            .all(|value| value.is_finite()));
        assert_eq!(
            decoded.max_error_squared,
            f32::INFINITY,
            "unrepresentable squared bounds remain conservative"
        );
        let distance = cluster_position_error(bounds).unwrap();
        assert!(distance.is_finite() && distance > 0.0);
        assert_eq!(cluster_position_error_squared(bounds), Ok(f32::INFINITY));
    }

    #[test]
    fn shared_position_bounds_cover_reconstruction_at_every_scale() {
        for extent in [f32::from_bits(8), 1.0e-30, 1.0, 1.0e30, f32::MAX * 0.5] {
            let bounds = SceneBounds {
                center: [0.0; 3],
                half_extents: [extent; 3],
                radius: extent,
                _padding: 0.0,
            };
            let source: Vec<_> = (0..64)
                .map(|i| {
                    let x = -1.0 + 2.0 * f64::from(i) / 63.0;
                    [
                        (f64::from(extent) * x) as f32,
                        (f64::from(extent) * x * 0.37) as f32,
                        (f64::from(extent) * x * -0.81) as f32,
                    ]
                })
                .collect();
            let payload = encode_cluster_payload(bounds, &source, &[0, 1, 2]).unwrap();
            let decoded = decode_cluster_payload(bounds, &payload).unwrap();
            assert_eq!(
                decoded.max_error_squared,
                cluster_position_error_squared(bounds).unwrap()
            );
            let limit = f64::from(cluster_position_error(bounds).unwrap()).powi(2);
            assert!(limit > 0.0 && limit.is_finite());
            for (expected, actual) in source.iter().zip(decoded.positions) {
                let actual_error: f64 = (0..3)
                    .map(|axis| (f64::from(expected[axis]) - f64::from(actual[axis])).powi(2))
                    .sum();
                assert!(actual_error <= limit, "{extent}: {actual_error} > {limit}");
            }
        }
        let point = SceneBounds {
            half_extents: [0.0; 3],
            ..bounds()
        };
        assert_eq!(cluster_position_error(point), Ok(0.0));
        assert_eq!(cluster_position_error_squared(point), Ok(0.0));
        for invalid in [
            SceneBounds {
                radius: f32::NAN,
                ..bounds()
            },
            SceneBounds {
                half_extents: [-1.0; 3],
                ..bounds()
            },
            SceneBounds {
                center: [f32::MAX; 3],
                half_extents: [f32::MAX; 3],
                ..bounds()
            },
        ] {
            assert_eq!(
                cluster_position_error(invalid),
                Err(ClusterPayloadError::NonFinite)
            );
            assert_eq!(
                cluster_position_error_squared(invalid),
                Err(ClusterPayloadError::NonFinite)
            );
        }
    }
}
