//! Same-build deterministic diagnostic fingerprints of complete world state.
//!
//! Unlike the small render-state hash, this includes every field of the
//! world's recursively derived `Debug` representation: allocation generations
//! and free lists, previous poses, constraints, soft topology, pending forces,
//! fields and their ID allocator, articulation queues, contact history,
//! aerodynamic state, assembly merge recovery and breakage bookkeeping,
//! collision exclusion membership, per-body flags and the group ID allocator.
//! Joint history includes continuous revolute/gear angles, previous principal
//! samples, complete coordinate bindings and substep motor/gear multipliers.
//! Equal body poses therefore do not imply equal world fingerprints.
//!
//! Private state in other modules has no serialization/visitor contract.
//! Consequently this is deliberately a **same-build diagnostic**, not a stable
//! disk/network encoding. Changing a type's `Debug` schema changes the digest.
//! Brace-delimited records/maps/sets are recursively sorted so randomized
//! contact-map iteration cannot cause false desyncs. Sequence order is retained.
//! Finite floats use Rust's round-trip debug spelling (including signed zero);
//! infinities retain their sign. NaN is rejected because Debug loses its payload.
//! This allocates in proportion to world size; it is not a realtime hot path.

use crate::world::PhysicsWorld;

/// Non-cryptographic, schema-dependent fingerprint of the complete world.
///
/// Equal digests are useful evidence, not a proof of equality (collisions exist).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WorldHash(pub u64);

/// A complete, faithful diagnostic fingerprint could not be produced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorldHashError {
    /// Debug does not preserve NaN payloads; do not silently equate those states.
    Nan,
    /// An unbalanced or excessively nested structural representation was found.
    InvalidRepresentation,
}

/// Hashes all world-owned state, not only visible poses.
///
/// The caller must use the same build/schema on both sides. Solver/backend
/// objects and driver cursors are external to `PhysicsWorld` and not included.
pub fn hash_world(world: &PhysicsWorld) -> Result<WorldHash, WorldHashError> {
    let representation = format!("{world:?}");
    let mut parser = CanonicalDebug {
        input: representation.as_bytes(),
        cursor: 0,
    };
    let canonical = parser.group(None, 0)?;
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in b"prism-world-debug-v1\0".iter().chain(canonical.iter()) {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    Ok(WorldHash(hash))
}

// Only braces are unordered. In particular, slot/free-list/queue order in
// square brackets and vector/matrix/tuple component order must never be sorted.
struct CanonicalDebug<'a> {
    input: &'a [u8],
    cursor: usize,
}

impl CanonicalDebug<'_> {
    fn group(&mut self, end: Option<u8>, depth: usize) -> Result<Vec<u8>, WorldHashError> {
        if depth > 256 {
            return Err(WorldHashError::InvalidRepresentation);
        }
        let mut entries = Vec::new();
        let mut entry = Vec::new();
        while let Some(&byte) = self.input.get(self.cursor) {
            self.cursor += 1;
            match byte {
                b'{' | b'[' | b'(' => {
                    let close = match byte {
                        b'{' => b'}',
                        b'[' => b']',
                        _ => b')',
                    };
                    entry.push(byte);
                    entry.extend(self.group(Some(close), depth + 1)?);
                    entry.push(close);
                }
                b'}' | b']' | b')' => {
                    if end != Some(byte) {
                        return Err(WorldHashError::InvalidRepresentation);
                    }
                    return Ok(Self::join(entries, entry, end));
                }
                b',' => {
                    entries.push(core::mem::take(&mut entry));
                }
                b'"' | b'\'' => {
                    entry.push(byte);
                    let mut closed = false;
                    while let Some(&next) = self.input.get(self.cursor) {
                        self.cursor += 1;
                        entry.push(next);
                        if next == b'\\' {
                            let escaped = *self
                                .input
                                .get(self.cursor)
                                .ok_or(WorldHashError::InvalidRepresentation)?;
                            self.cursor += 1;
                            entry.push(escaped);
                        } else if next == byte {
                            closed = true;
                            break;
                        }
                    }
                    if !closed {
                        return Err(WorldHashError::InvalidRepresentation);
                    }
                }
                c if c.is_ascii_whitespace() => {}
                c if c.is_ascii_alphabetic() || c == b'_' => {
                    let start = self.cursor - 1;
                    while self
                        .input
                        .get(self.cursor)
                        .is_some_and(|c| c.is_ascii_alphanumeric() || *c == b'_')
                    {
                        self.cursor += 1;
                    }
                    let token = &self.input[start..self.cursor];
                    if token == b"NaN" {
                        return Err(WorldHashError::Nan);
                    }
                    entry.extend(token);
                }
                other => entry.push(other),
            }
        }
        if end.is_some() {
            return Err(WorldHashError::InvalidRepresentation);
        }
        Ok(Self::join(entries, entry, end))
    }

    fn join(mut entries: Vec<Vec<u8>>, entry: Vec<u8>, end: Option<u8>) -> Vec<u8> {
        if !entry.is_empty() {
            entries.push(entry);
        }
        if end == Some(b'}') {
            entries.sort_unstable();
        }
        entries.join(&[b','][..])
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod angular_tests;
