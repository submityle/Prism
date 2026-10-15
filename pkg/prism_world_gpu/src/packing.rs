//! Deterministic structure-of-arrays packing for world proxy uploads.

use crate::{Cluster, ProxyInstance};

/// Little-endian byte streams for the three proxy SoA columns.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SoaBufferPack {
    /// `x, y, z, radius` for each proxy.
    pub positions: Vec<u8>,
    /// `geometric_error, max_distance, 0, 0` for each proxy.
    pub errors: Vec<u8>,
    /// `proxy_id, cluster_id, flags, 0` for each proxy.
    pub metadata: Vec<u8>,
}

/// CPU-side proxy columns in stable input order.
#[derive(Clone, Debug, PartialEq)]
pub struct ProxySoa {
    /// Position and radius column.
    pub positions: Vec<[f32; 4]>,
    /// Error and range column.
    pub errors: Vec<[f32; 4]>,
    /// Identity and flags column.
    pub metadata: Vec<[u32; 4]>,
}

impl ProxySoa {
    /// Validates and transposes a proxy slice without reordering it.
    pub fn from_proxies(proxies: &[ProxyInstance]) -> Result<Self, String> {
        if proxies.iter().any(|proxy| !proxy.is_finite()) {
            return Err("proxy values must be finite and non-negative".to_owned());
        }

        let mut positions = Vec::with_capacity(proxies.len());
        let mut errors = Vec::with_capacity(proxies.len());
        let mut metadata = Vec::with_capacity(proxies.len());
        for proxy in proxies {
            positions.push([
                proxy.position[0],
                proxy.position[1],
                proxy.position[2],
                proxy.radius,
            ]);
            errors.push([proxy.geometric_error, proxy.max_distance, 0.0, 0.0]);
            metadata.push([proxy.proxy_id, proxy.cluster_id, proxy.flags, 0]);
        }
        Ok(Self {
            positions,
            errors,
            metadata,
        })
    }

    /// Number of proxy records in all columns.
    #[must_use]
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// Whether all columns contain no records.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Converts every `f32`/`u32` word to explicit little-endian bytes.
    #[must_use]
    pub fn pack_le(&self) -> SoaBufferPack {
        SoaBufferPack {
            positions: pack_f32x4(&self.positions),
            errors: pack_f32x4(&self.errors),
            metadata: pack_u32x4(&self.metadata),
        }
    }
}

/// Packs clusters into span/error-bit words and centre/radius floats,
/// using the same stable order as the caller supplied.
#[derive(Clone, Debug, PartialEq)]
pub struct ClusterSoa {
    /// `cluster_id, first_proxy, proxy_count, geometric_error.to_bits()`.
    pub spans: Vec<[u32; 4]>,
    /// `center.x, center.y, center.z, radius`.
    pub bounds: Vec<[f32; 4]>,
}

impl ClusterSoa {
    /// Validates and transposes cluster records.
    pub fn from_clusters(clusters: &[Cluster]) -> Result<Self, String> {
        if clusters.iter().any(|cluster| !cluster.is_finite()) {
            return Err("cluster values must be finite and non-negative".to_owned());
        }
        let mut spans = Vec::with_capacity(clusters.len());
        let mut bounds = Vec::with_capacity(clusters.len());
        for cluster in clusters {
            spans.push([
                cluster.cluster_id,
                cluster.first_proxy,
                cluster.proxy_count,
                cluster.geometric_error.to_bits(),
            ]);
            bounds.push([
                cluster.center[0],
                cluster.center[1],
                cluster.center[2],
                cluster.radius,
            ]);
        }
        Ok(Self { spans, bounds })
    }
}

fn pack_f32x4(values: &[[f32; 4]]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 16);
    for row in values {
        for value in row {
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
    }
    bytes
}

fn pack_u32x4(values: &[[u32; 4]]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 16);
    for row in values {
        for value in row {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    bytes
}
