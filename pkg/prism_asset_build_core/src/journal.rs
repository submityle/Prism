//! Durable, append-only publication journal.
//!
//! The journal is deliberately small and local.  It records the publication
//! boundary, not every CAS write: CAS blobs are immutable and an interrupted
//! transaction may safely leave unreachable blobs for a later GC pass.  A
//! journal record is framed, checksummed, and synced before the next
//! publication step is allowed to proceed.

use crate::types::{BuildRevision, ContentHash};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

const FILE_MAGIC: &[u8] = b"PRISM-TXN-JNL\0\x02";
const FRAME_MAGIC: &[u8; 4] = b"TXN1";
pub(crate) const FRAME_HEADER_LEN: usize = 8;
const CHECKSUM_LEN: usize = 32;
const PAYLOAD_LEN: usize = 1 + 8 + 1 + 8 + 32 + 1 + 32;
const MAX_PAYLOAD_LEN: u32 = PAYLOAD_LEN as u32;
#[cfg(test)]
pub(crate) const FILE_HEADER_LEN: usize = FILE_MAGIC.len();
pub(crate) const FRAME_SIZE: usize = FRAME_HEADER_LEN + PAYLOAD_LEN + CHECKSUM_LEN;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum JournalRecord {
    Prepare {
        revision: BuildRevision,
        expected_current: Option<BuildRevision>,
        manifest_hash: ContentHash,
        publication_hash: Option<ContentHash>,
    },
    Commit {
        revision: BuildRevision,
        manifest_hash: ContentHash,
        publication_hash: Option<ContentHash>,
    },
    Abort {
        revision: BuildRevision,
        manifest_hash: ContentHash,
        publication_hash: Option<ContentHash>,
    },
}

#[derive(Clone, Debug)]
pub(crate) struct TransactionJournal {
    path: PathBuf,
}

impl TransactionJournal {
    pub(crate) fn open(path: impl Into<PathBuf>) -> Result<Self, io::Error> {
        let path = path.into();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        if !path.exists() {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)?;
            file.write_all(FILE_MAGIC)?;
            file.sync_all()?;
            sync_parent(&path)?;
        } else if fs::metadata(&path)?.len() == 0 {
            // A process can be killed after create_new succeeds but before the
            // header reaches stable storage.  An empty journal is still
            // uninitialized, not a valid empty history.
            let mut file = OpenOptions::new().write(true).open(&path)?;
            file.write_all(FILE_MAGIC)?;
            file.sync_all()?;
            sync_parent(&path)?;
        }
        Ok(Self { path })
    }

    pub(crate) fn scan(&self) -> Result<Vec<JournalRecord>, io::Error> {
        let bytes = fs::read(&self.path)?;
        if bytes.len() < FILE_MAGIC.len() || !bytes.starts_with(FILE_MAGIC) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "asset transaction journal has invalid header",
            ));
        }

        let mut offset = FILE_MAGIC.len();
        let mut records = Vec::new();
        loop {
            let frame_start = offset;
            let remaining = bytes.len().saturating_sub(offset);
            if remaining == 0 {
                break;
            }
            if remaining < FRAME_HEADER_LEN {
                self.truncate_tail(frame_start)?;
                break;
            }
            if &bytes[offset..offset + 4] != FRAME_MAGIC {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "asset transaction journal has an invalid frame",
                ));
            }
            let payload_len = u32::from_le_bytes(
                bytes[offset + 4..offset + FRAME_HEADER_LEN]
                    .try_into()
                    .expect("journal frame header has four length bytes"),
            );
            if payload_len != MAX_PAYLOAD_LEN {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "asset transaction journal has an invalid record length",
                ));
            }
            let payload_start = offset + FRAME_HEADER_LEN;
            let payload_end = payload_start + payload_len as usize;
            let checksum_end = payload_end + CHECKSUM_LEN;
            if checksum_end > bytes.len() {
                self.truncate_tail(frame_start)?;
                break;
            }
            let payload = &bytes[payload_start..payload_end];
            let checksum = blake3::hash(payload);
            if checksum.as_bytes() != &bytes[payload_end..checksum_end] {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "asset transaction journal checksum mismatch",
                ));
            }
            records.push(decode_payload(payload)?);
            offset = checksum_end;
        }
        Ok(records)
    }

    pub(crate) fn append(&self, record: JournalRecord) -> Result<(), io::Error> {
        let payload = encode_payload(record);
        debug_assert_eq!(payload.len(), PAYLOAD_LEN);
        let checksum = blake3::hash(&payload);
        let mut frame = Vec::with_capacity(FRAME_SIZE);
        frame.extend_from_slice(FRAME_MAGIC);
        frame.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        frame.extend_from_slice(&payload);
        frame.extend_from_slice(checksum.as_bytes());

        let mut file = OpenOptions::new().append(true).open(&self.path)?;
        file.write_all(&frame)?;
        file.sync_all()?;
        Ok(())
    }

    fn truncate_tail(&self, length: usize) -> Result<(), io::Error> {
        let file = OpenOptions::new().write(true).open(&self.path)?;
        file.set_len(length as u64)?;
        file.sync_all()?;
        Ok(())
    }
}

fn encode_payload(record: JournalRecord) -> Vec<u8> {
    let (kind, revision, expected_current, manifest_hash, publication_hash) = match record {
        JournalRecord::Prepare {
            revision,
            expected_current,
            manifest_hash,
            publication_hash,
        } => (
            1u8,
            revision,
            expected_current,
            manifest_hash,
            publication_hash,
        ),
        JournalRecord::Commit {
            revision,
            manifest_hash,
            publication_hash,
        } => (2u8, revision, None, manifest_hash, publication_hash),
        JournalRecord::Abort {
            revision,
            manifest_hash,
            publication_hash,
        } => (3u8, revision, None, manifest_hash, publication_hash),
    };
    let mut payload = Vec::with_capacity(PAYLOAD_LEN);
    payload.push(kind);
    payload.extend_from_slice(&revision.0.to_le_bytes());
    match expected_current {
        Some(current) => {
            payload.push(1);
            payload.extend_from_slice(&current.0.to_le_bytes());
        }
        None => {
            payload.push(0);
            payload.extend_from_slice(&0u64.to_le_bytes());
        }
    }
    payload.extend_from_slice(&manifest_hash.0);
    match publication_hash {
        Some(hash) => {
            payload.push(1);
            payload.extend_from_slice(&hash.0);
        }
        None => {
            payload.push(0);
            payload.extend_from_slice(&[0; 32]);
        }
    }
    payload
}

fn decode_payload(payload: &[u8]) -> Result<JournalRecord, io::Error> {
    if payload.len() != PAYLOAD_LEN {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "asset transaction journal record is truncated",
        ));
    }
    let kind = payload[0];
    let revision = BuildRevision(u64::from_le_bytes(
        payload[1..9]
            .try_into()
            .expect("journal revision has eight bytes"),
    ));
    let expected_current = match payload[9] {
        0 => None,
        1 => Some(BuildRevision(u64::from_le_bytes(
            payload[10..18]
                .try_into()
                .expect("journal expected revision has eight bytes"),
        ))),
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "asset transaction journal has an invalid optional revision",
            ))
        }
    };
    let manifest_hash = ContentHash(
        payload[18..50]
            .try_into()
            .expect("journal manifest hash has 32 bytes"),
    );
    let publication_hash = match payload[50] {
        0 => None,
        1 => Some(ContentHash(
            payload[51..83]
                .try_into()
                .expect("journal publication hash has 32 bytes"),
        )),
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "asset transaction journal has an invalid publication hash marker",
            ))
        }
    };
    match kind {
        1 => Ok(JournalRecord::Prepare {
            revision,
            expected_current,
            manifest_hash,
            publication_hash,
        }),
        2 if expected_current.is_none() => Ok(JournalRecord::Commit {
            revision,
            manifest_hash,
            publication_hash,
        }),
        3 if expected_current.is_none() => Ok(JournalRecord::Abort {
            revision,
            manifest_hash,
            publication_hash,
        }),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "asset transaction journal has an invalid record kind",
        )),
    }
}

fn sync_parent(path: &Path) -> Result<(), io::Error> {
    #[cfg(unix)]
    {
        if let Some(parent) = path.parent() {
            File::open(parent)?.sync_all()?;
        }
    }
    #[cfg(not(unix))]
    {
        let _ = path;
    }
    Ok(())
}
