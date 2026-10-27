//! Atomic durable records for checkpoints, inbox/outbox data, and execution events.

use crate::{ExecutionEvent, Value, VsError, VsResult};
use fs2::FileExt;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecordKey {
    pub namespace: String,
    pub id: String,
}

impl RecordKey {
    pub fn new(namespace: impl Into<String>, id: impl Into<String>) -> VsResult<Self> {
        let value = Self {
            namespace: namespace.into(),
            id: id.into(),
        };
        value.validate()?;
        Ok(value)
    }

    fn validate(&self) -> VsResult<()> {
        if self.namespace.is_empty()
            || self.id.is_empty()
            || !self
                .namespace
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
        {
            return Err(VsError::Validate("invalid durable record key".into()));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InboxEntry {
    pub id: String,
    pub payload_digest: [u8; 32],
}

impl InboxEntry {
    pub fn new(id: impl Into<String>, payload: &[u8]) -> Self {
        Self {
            id: id.into(),
            payload_digest: Sha256::digest(payload).into(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutboxEntry {
    pub id: String,
    pub topic: String,
    pub payload: Value,
    pub dispatched: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DurableRecord {
    pub revision: u64,
    pub payload: Vec<u8>,
    pub payload_digest: [u8; 32],
    pub inbox: BTreeMap<String, InboxEntry>,
    pub outbox: BTreeMap<String, OutboxEntry>,
    pub events: Vec<ExecutionEvent>,
}

impl DurableRecord {
    pub fn empty() -> Self {
        Self {
            revision: 0,
            payload: Vec::new(),
            payload_digest: Sha256::digest([]).into(),
            inbox: BTreeMap::new(),
            outbox: BTreeMap::new(),
            events: Vec::new(),
        }
    }

    pub fn decode<T: DeserializeOwned>(&self) -> VsResult<T> {
        let digest: [u8; 32] = Sha256::digest(&self.payload).into();
        if digest != self.payload_digest {
            return Err(VsError::Runtime("durable payload digest mismatch".into()));
        }
        serde_json::from_slice(&self.payload)
            .map_err(|error| VsError::Runtime(format!("decode durable payload: {error}")))
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RevisionGuard {
    Create,
    Match(u64),
    Any,
}

#[derive(Clone, Debug, Default)]
pub struct DurableMutation {
    pub payload: Option<Vec<u8>>,
    pub inbox: Vec<InboxEntry>,
    pub outbox: Vec<OutboxEntry>,
    pub acknowledge_outbox: Vec<String>,
    pub events: Vec<ExecutionEvent>,
}

impl DurableMutation {
    pub fn with_payload<T: Serialize>(mut self, payload: &T) -> VsResult<Self> {
        self.payload = Some(
            serde_json::to_vec(payload)
                .map_err(|error| VsError::Runtime(format!("encode durable payload: {error}")))?,
        );
        Ok(self)
    }
}

pub trait DurableStore: Send + Sync {
    fn load(&self, key: &RecordKey) -> VsResult<Option<DurableRecord>>;
    fn commit(
        &self,
        key: &RecordKey,
        guard: RevisionGuard,
        mutation: DurableMutation,
    ) -> VsResult<DurableRecord>;
}

fn apply_mutation(
    current: Option<DurableRecord>,
    guard: RevisionGuard,
    mutation: DurableMutation,
) -> VsResult<DurableRecord> {
    match (guard, current.as_ref().map(|record| record.revision)) {
        (RevisionGuard::Create, Some(_)) => {
            return Err(VsError::Runtime("durable record already exists".into()));
        }
        (RevisionGuard::Match(expected), actual) if actual != Some(expected) => {
            return Err(VsError::Runtime(format!(
                "durable revision conflict: expected {expected}, got {actual:?}"
            )));
        }
        _ => {}
    }
    let mut record = current.unwrap_or_else(DurableRecord::empty);
    if let Some(payload) = mutation.payload {
        record.payload_digest = Sha256::digest(&payload).into();
        record.payload = payload;
    }
    for incoming in mutation.inbox {
        if incoming.id.is_empty() {
            return Err(VsError::Validate("inbox identity must not be empty".into()));
        }
        if let Some(existing) = record.inbox.get(&incoming.id) {
            if existing.payload_digest != incoming.payload_digest {
                return Err(VsError::Runtime(format!(
                    "inbox identity {} was reused with different content",
                    incoming.id
                )));
            }
        } else {
            record.inbox.insert(incoming.id.clone(), incoming);
        }
    }
    for outgoing in mutation.outbox {
        if outgoing.id.is_empty() || outgoing.topic.is_empty() {
            return Err(VsError::Validate(
                "outbox identity and topic must not be empty".into(),
            ));
        }
        if let Some(existing) = record.outbox.get(&outgoing.id) {
            if existing.topic != outgoing.topic || existing.payload != outgoing.payload {
                return Err(VsError::Runtime(format!(
                    "outbox identity {} was reused with different content",
                    outgoing.id
                )));
            }
        } else {
            record.outbox.insert(outgoing.id.clone(), outgoing);
        }
    }
    for id in mutation.acknowledge_outbox {
        record
            .outbox
            .get_mut(&id)
            .ok_or_else(|| VsError::Runtime(format!("unknown outbox entry {id}")))?
            .dispatched = true;
    }
    for mut event in mutation.events {
        event.sequence = next_event_sequence(record.events.len())?;
        record.events.push(event);
    }
    record.revision = record
        .revision
        .checked_add(1)
        .ok_or_else(|| VsError::Runtime("durable revision exhausted".into()))?;
    Ok(record)
}

pub fn next_event_sequence(event_count: usize) -> VsResult<u64> {
    u64::try_from(event_count)
        .ok()
        .and_then(|sequence| sequence.checked_add(1))
        .ok_or_else(|| VsError::Runtime("durable event sequence exhausted".into()))
}

#[derive(Default)]
pub struct InMemoryDurableStore {
    records: Mutex<BTreeMap<RecordKey, DurableRecord>>,
}

impl DurableStore for InMemoryDurableStore {
    fn load(&self, key: &RecordKey) -> VsResult<Option<DurableRecord>> {
        key.validate()?;
        Ok(self
            .records
            .lock()
            .map_err(|_| VsError::Runtime("durable store poisoned".into()))?
            .get(key)
            .cloned())
    }

    fn commit(
        &self,
        key: &RecordKey,
        guard: RevisionGuard,
        mutation: DurableMutation,
    ) -> VsResult<DurableRecord> {
        key.validate()?;
        let mut records = self
            .records
            .lock()
            .map_err(|_| VsError::Runtime("durable store poisoned".into()))?;
        let record = apply_mutation(records.get(key).cloned(), guard, mutation)?;
        records.insert(key.clone(), record.clone());
        Ok(record)
    }
}

pub struct FileDurableStore {
    root: PathBuf,
    lock: Mutex<File>,
}

impl FileDurableStore {
    pub fn open(root: impl Into<PathBuf>) -> VsResult<Self> {
        let root = root.into();
        fs::create_dir_all(&root)
            .map_err(|error| VsError::Runtime(format!("create durable root: {error}")))?;
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(root.join(".store.lock"))
            .map_err(|error| VsError::Runtime(format!("open durable lock: {error}")))?;
        Ok(Self {
            root,
            lock: Mutex::new(lock),
        })
    }

    fn record_path(&self, key: &RecordKey) -> PathBuf {
        let digest = Sha256::digest(key.id.as_bytes());
        let name = digest
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        self.root.join(&key.namespace).join(format!("{name}.json"))
    }

    fn load_unlocked(&self, key: &RecordKey) -> VsResult<Option<DurableRecord>> {
        let path = self.record_path(key);
        match fs::read(path) {
            Ok(bytes) => serde_json::from_slice(&bytes)
                .map(Some)
                .map_err(|error| VsError::Runtime(format!("decode durable record: {error}"))),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(VsError::Runtime(format!("read durable record: {error}"))),
        }
    }

    fn write_unlocked(&self, key: &RecordKey, record: &DurableRecord) -> VsResult<()> {
        let path = self.record_path(key);
        let parent = path
            .parent()
            .ok_or_else(|| VsError::Runtime("durable record has no parent".into()))?;
        fs::create_dir_all(parent)
            .map_err(|error| VsError::Runtime(format!("create durable namespace: {error}")))?;
        let bytes = serde_json::to_vec(record)
            .map_err(|error| VsError::Runtime(format!("encode durable record: {error}")))?;
        let temp = path.with_extension(format!("tmp-{}", std::process::id()));
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&temp)
            .map_err(|error| VsError::Runtime(format!("create durable temp: {error}")))?;
        file.write_all(&bytes)
            .and_then(|()| file.sync_all())
            .map_err(|error| VsError::Runtime(format!("sync durable temp: {error}")))?;
        fs::rename(&temp, &path)
            .map_err(|error| VsError::Runtime(format!("publish durable record: {error}")))?;
        File::open(parent)
            .and_then(|directory| directory.sync_all())
            .map_err(|error| VsError::Runtime(format!("sync durable namespace: {error}")))?;
        Ok(())
    }
}

impl DurableStore for FileDurableStore {
    fn load(&self, key: &RecordKey) -> VsResult<Option<DurableRecord>> {
        key.validate()?;
        let lock = self
            .lock
            .lock()
            .map_err(|_| VsError::Runtime("durable file lock poisoned".into()))?;
        FileExt::lock_shared(&*lock)
            .map_err(|error| VsError::Runtime(format!("lock durable store: {error}")))?;
        let result = self.load_unlocked(key);
        let unlock = FileExt::unlock(&*lock);
        result.and_then(|value| {
            unlock
                .map(|()| value)
                .map_err(|error| VsError::Runtime(format!("unlock durable store: {error}")))
        })
    }

    fn commit(
        &self,
        key: &RecordKey,
        guard: RevisionGuard,
        mutation: DurableMutation,
    ) -> VsResult<DurableRecord> {
        key.validate()?;
        let lock = self
            .lock
            .lock()
            .map_err(|_| VsError::Runtime("durable file lock poisoned".into()))?;
        FileExt::lock_exclusive(&*lock)
            .map_err(|error| VsError::Runtime(format!("lock durable store: {error}")))?;
        let result = self
            .load_unlocked(key)
            .and_then(|current| apply_mutation(current, guard, mutation))
            .and_then(|record| self.write_unlocked(key, &record).map(|()| record));
        let unlock = FileExt::unlock(&*lock);
        result.and_then(|value| {
            unlock
                .map(|()| value)
                .map_err(|error| VsError::Runtime(format!("unlock durable store: {error}")))
        })
    }
}

pub fn durable_payload<T: Serialize>(value: &T) -> VsResult<Vec<u8>> {
    serde_json::to_vec(value)
        .map_err(|error| VsError::Runtime(format!("encode durable payload: {error}")))
}

pub fn verify_directory(path: &Path) -> VsResult<()> {
    if path.is_dir() {
        Ok(())
    } else {
        Err(VsError::Runtime("durable path is not a directory".into()))
    }
}
