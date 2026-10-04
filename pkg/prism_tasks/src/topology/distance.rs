//! Portable NUMA inter-node distance matrix (design §24.6).
//!
//! Real hardware describes the relative cost of reaching one NUMA node's
//! memory from another with a *distance matrix* (ACPI's SLIT table is the
//! canonical example). Entry `(a, b)` is a unit-less relative latency: the
//! self-distance (`a == b`, i.e. node-local access) is the smallest value, and
//! remote entries grow with topological distance.
//!
//! [`NumaDistanceMatrix`] is a pure, allocation-only, platform-independent
//! model of that table. It performs no probing of its own — `prism_platform`
//! injects the real values (see the module docs for `topology`) — but every
//! query on it is a deterministic function of the stored numbers, so the
//! distance-aware scheduling in [`super::schedule`] is fully unit-testable on
//! any machine.
//!
//! The convention follows ACPI SLIT: [`LOCAL_DISTANCE`] (`10`) is the standard
//! node-local value, and remote entries are typically `>= 20`. Only the
//! *ordering* of the numbers matters to the scheduler; absolute magnitudes are
//! never interpreted as real nanoseconds.

use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

use crate::numa::NumaNodeId;

/// The standard node-local distance (an access to a node's own memory), using
/// the ACPI SLIT convention where local is `10` and remote entries are larger.
pub const LOCAL_DISTANCE: u16 = 10;

/// A square matrix of relative NUMA inter-node distances, stored row-major.
///
/// Row `a`, column `b` holds the distance from node `a` to node `b`. The
/// matrix is not required to be symmetric (some platforms report asymmetric
/// latencies), but each node's self-distance is its row minimum.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NumaDistanceMatrix {
    node_count: usize,
    rows: Vec<u16>,
}

/// Why building a [`NumaDistanceMatrix`] from explicit rows failed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DistanceError {
    /// No rows were supplied; a matrix needs at least one node.
    Empty,
    /// Row `row` has `cols` entries but the matrix has `rows` nodes, so it is
    /// not square.
    NotSquare {
        /// Number of rows (the intended node count).
        rows: usize,
        /// Length of the offending row.
        cols: usize,
        /// Index of the offending row.
        row: usize,
    },
    /// Distance `from` -> `to` is zero; distances must be positive so that
    /// node-local access (the row minimum) is still a real, orderable cost.
    ZeroDistance {
        /// Source node index.
        from: usize,
        /// Destination node index.
        to: usize,
    },
    /// Node `node`'s self-distance is not the smallest entry in its row, which
    /// would mean a remote node is "closer" than the node itself.
    DiagonalNotMinimal {
        /// The offending node index.
        node: usize,
    },
}

impl fmt::Display for DistanceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DistanceError::Empty => f.write_str("distance matrix needs at least one node"),
            DistanceError::NotSquare { rows, cols, row } => write!(
                f,
                "distance matrix is not square: row {row} has {cols} entries, expected {rows}"
            ),
            DistanceError::ZeroDistance { from, to } => {
                write!(f, "distance {from}->{to} is zero; distances must be positive")
            }
            DistanceError::DiagonalNotMinimal { node } => write!(
                f,
                "node {node} self-distance is not the minimum of its row"
            ),
        }
    }
}

impl NumaDistanceMatrix {
    /// A single-node matrix: one node whose only (local) distance is
    /// [`LOCAL_DISTANCE`]. The non-NUMA fallback.
    pub fn single() -> Self {
        Self {
            node_count: 1,
            rows: vec![LOCAL_DISTANCE],
        }
    }

    /// A uniform matrix: every node is [`LOCAL_DISTANCE`] from itself and
    /// `remote` from every other node. `node_count` is clamped to at least `1`;
    /// `remote` is clamped to at least `LOCAL_DISTANCE` so a cross-node hop is
    /// never cheaper than a local one.
    pub fn uniform(node_count: usize, remote: u16) -> Self {
        let node_count = node_count.max(1);
        let remote = remote.max(LOCAL_DISTANCE);
        let mut rows = vec![remote; node_count * node_count];
        for n in 0..node_count {
            rows[n * node_count + n] = LOCAL_DISTANCE;
        }
        Self { node_count, rows }
    }

    /// Build a matrix from explicit rows (as a platform probe or a test would).
    ///
    /// # Errors
    /// Returns [`DistanceError`] if the input is empty, not square, contains a
    /// zero distance, or has a self-distance that is not its row's minimum.
    pub fn from_rows(rows: Vec<Vec<u16>>) -> Result<Self, DistanceError> {
        let node_count = rows.len();
        if node_count == 0 {
            return Err(DistanceError::Empty);
        }
        let mut flat = Vec::with_capacity(node_count * node_count);
        for (r, row) in rows.iter().enumerate() {
            if row.len() != node_count {
                return Err(DistanceError::NotSquare {
                    rows: node_count,
                    cols: row.len(),
                    row: r,
                });
            }
            let diagonal = row[r];
            for (c, &d) in row.iter().enumerate() {
                if d == 0 {
                    return Err(DistanceError::ZeroDistance { from: r, to: c });
                }
                if d < diagonal {
                    return Err(DistanceError::DiagonalNotMinimal { node: r });
                }
            }
            flat.extend_from_slice(row);
        }
        Ok(Self {
            node_count,
            rows: flat,
        })
    }

    /// Number of NUMA nodes described.
    pub fn node_count(&self) -> usize {
        self.node_count
    }

    #[inline]
    fn flat_index(&self, a: NumaNodeId, b: NumaNodeId) -> Option<usize> {
        let a = a.index() as usize;
        let b = b.index() as usize;
        if a < self.node_count && b < self.node_count {
            Some(a * self.node_count + b)
        } else {
            None
        }
    }

    /// The distance from node `a` to node `b`, or `None` if either index is out
    /// of range.
    pub fn get(&self, a: NumaNodeId, b: NumaNodeId) -> Option<u16> {
        self.flat_index(a, b).map(|i| self.rows[i])
    }

    /// The distance from node `a` to node `b`.
    ///
    /// # Panics
    /// Panics if either node index is out of range; use [`get`](Self::get) for
    /// a checked lookup.
    pub fn distance(&self, a: NumaNodeId, b: NumaNodeId) -> u16 {
        match self.get(a, b) {
            Some(d) => d,
            None => panic!(
                "NUMA node index out of range: a={}, b={}, node_count={}",
                a.index(),
                b.index(),
                self.node_count
            ),
        }
    }

    /// Whether the matrix is symmetric (`d(a, b) == d(b, a)` for all pairs).
    pub fn is_symmetric(&self) -> bool {
        for a in 0..self.node_count {
            for b in (a + 1)..self.node_count {
                if self.rows[a * self.node_count + b] != self.rows[b * self.node_count + a] {
                    return false;
                }
            }
        }
        true
    }
}
