//! Durable actor state and mailbox primitives.
//!
//! An actor is a long-lived keyed entity. Signals are accepted exactly once by `signal_id`,
//! delivered in mailbox order, and committed only with the current fencing token. The actual
//! workflow that handles a signal remains a normal UVS graph; this module owns the durable entity
//! boundary around it.

use crate::{
    DurableMutation, DurableStore, ExecutionClock, RecordKey, RevisionGuard, Value, VsError,
    VsResult,
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
#[serde(deny_unknown_fields)]
pub struct ActorId {
    pub kind: String,
    pub key: String,
}

impl ActorId {
    pub fn new(kind: impl Into<String>, key: impl Into<String>) -> Self {
        Self {
            kind: kind.into(),
            key: key.into(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActorSignal {
    pub signal_id: String,
    pub sequence: u64,
    pub topic: String,
    pub payload: Value,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActorMessage {
    pub signal_id: String,
    pub topic: String,
    pub payload: Value,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActorLease {
    pub owner: String,
    pub token: String,
    pub epoch: u64,
    pub expires_at_ms: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActorState {
    pub id: ActorId,
    pub revision: u64,
    pub value: Value,
    pub mailbox: VecDeque<ActorSignal>,
    pub in_flight: Option<ActorSignal>,
    pub accepted_signal_ids: BTreeSet<String>,
    pub next_sequence: u64,
    pub next_epoch: u64,
    pub lease: Option<ActorLease>,
}

impl ActorState {
    pub fn new(id: ActorId, value: Value) -> Self {
        Self {
            id,
            revision: 0,
            value,
            mailbox: VecDeque::new(),
            in_flight: None,
            accepted_signal_ids: BTreeSet::new(),
            next_sequence: 1,
            next_epoch: 0,
            lease: None,
        }
    }

    pub fn acquire_lease(
        &mut self,
        owner: impl Into<String>,
        now_ms: u64,
        ttl_ms: u64,
    ) -> VsResult<ActorLease> {
        if ttl_ms == 0 {
            return Err(VsError::Runtime(
                "actor lease ttl must be greater than zero".into(),
            ));
        }
        let owner = owner.into();
        if owner.is_empty() {
            return Err(VsError::Runtime(
                "actor lease owner must not be empty".into(),
            ));
        }
        if let Some(current) = &self.lease
            && current.expires_at_ms > now_ms
            && current.owner != owner
        {
            return Err(VsError::Runtime(format!(
                "actor {} is fenced by lease owner {}",
                self.id.key, current.owner
            )));
        }
        let expires_at_ms = now_ms
            .checked_add(ttl_ms)
            .ok_or_else(|| VsError::Runtime("actor lease expiry exhausted".into()))?;
        self.next_epoch = self
            .next_epoch
            .checked_add(1)
            .ok_or_else(|| VsError::Runtime("actor lease epoch exhausted".into()))?;
        let token = lease_token(&self.id, &owner, self.next_epoch);
        let lease = ActorLease {
            owner,
            token,
            epoch: self.next_epoch,
            expires_at_ms,
        };
        self.lease = Some(lease.clone());
        Ok(lease)
    }

    pub fn renew_lease(
        &mut self,
        owner: &str,
        token: &str,
        now_ms: u64,
        ttl_ms: u64,
    ) -> VsResult<ActorLease> {
        self.check_lease(owner, token, now_ms)?;
        if ttl_ms == 0 {
            return Err(VsError::Runtime(
                "actor lease ttl must be greater than zero".into(),
            ));
        }
        let expires_at_ms = now_ms
            .checked_add(ttl_ms)
            .ok_or_else(|| VsError::Runtime("actor lease expiry exhausted".into()))?;
        let Some(lease) = self.lease.as_mut() else {
            return Err(VsError::Runtime("actor lease state is inconsistent".into()));
        };
        lease.expires_at_ms = expires_at_ms;
        Ok(lease.clone())
    }

    pub fn enqueue(
        &mut self,
        owner: &str,
        token: &str,
        now_ms: u64,
        signal_id: impl Into<String>,
        topic: impl Into<String>,
        payload: Value,
    ) -> VsResult<bool> {
        self.check_lease(owner, token, now_ms)?;
        let signal_id = signal_id.into();
        if signal_id.is_empty() {
            return Err(VsError::Runtime("actor signal_id must not be empty".into()));
        }
        if !self.accepted_signal_ids.insert(signal_id.clone()) {
            return Ok(false);
        }
        let sequence = self.next_sequence;
        self.next_sequence = self
            .next_sequence
            .checked_add(1)
            .ok_or_else(|| VsError::Runtime("actor signal sequence exhausted".into()))?;
        self.mailbox.push_back(ActorSignal {
            signal_id,
            sequence,
            topic: topic.into(),
            payload,
        });
        Ok(true)
    }

    pub fn claim_next(
        &mut self,
        owner: &str,
        token: &str,
        now_ms: u64,
    ) -> VsResult<Option<ActorSignal>> {
        self.check_lease(owner, token, now_ms)?;
        if self.in_flight.is_none() {
            self.in_flight = self.mailbox.pop_front();
        }
        Ok(self.in_flight.clone())
    }

    pub fn commit(
        &mut self,
        owner: &str,
        token: &str,
        now_ms: u64,
        signal_id: &str,
        value: Value,
    ) -> VsResult<()> {
        self.check_lease(owner, token, now_ms)?;
        let Some(in_flight) = &self.in_flight else {
            return Err(VsError::Runtime("actor has no in-flight signal".into()));
        };
        if in_flight.signal_id != signal_id {
            return Err(VsError::Runtime(
                "actor signal commit identity mismatch".into(),
            ));
        }
        self.value = value;
        self.revision = self
            .revision
            .checked_add(1)
            .ok_or_else(|| VsError::Runtime("actor revision exhausted".into()))?;
        self.in_flight = None;
        Ok(())
    }

    fn check_lease(&self, owner: &str, token: &str, now_ms: u64) -> VsResult<()> {
        let Some(lease) = &self.lease else {
            return Err(VsError::Runtime("actor has no active lease".into()));
        };
        if lease.owner != owner || lease.token != token {
            return Err(VsError::Runtime("actor fencing token mismatch".into()));
        }
        if lease.expires_at_ms <= now_ms {
            return Err(VsError::Runtime("actor lease expired".into()));
        }
        Ok(())
    }
}

fn lease_token(id: &ActorId, owner: &str, epoch: u64) -> String {
    let mut digest = Sha256::new();
    digest.update(id.kind.as_bytes());
    digest.update([0]);
    digest.update(id.key.as_bytes());
    digest.update([0]);
    digest.update(owner.as_bytes());
    digest.update(epoch.to_le_bytes());
    digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

pub trait ActorStore: Send + Sync {
    fn load(&self, id: &ActorId) -> Option<ActorState>;
    fn save(&self, state: ActorState) -> VsResult<()>;
}

#[derive(Default)]
pub struct InMemoryActorStore {
    actors: Mutex<BTreeMap<ActorId, ActorState>>,
}

impl ActorStore for InMemoryActorStore {
    fn load(&self, id: &ActorId) -> Option<ActorState> {
        self.actors
            .lock()
            .expect("actor store poisoned")
            .get(id)
            .cloned()
    }

    fn save(&self, state: ActorState) -> VsResult<()> {
        self.actors
            .lock()
            .map_err(|_| VsError::Runtime("actor store poisoned".into()))?
            .insert(state.id.clone(), state);
        Ok(())
    }
}

#[async_trait]
pub trait ActorHandler: Send + Sync {
    async fn handle(&self, state: &ActorState, signal: &ActorSignal) -> VsResult<Value>;
}

/// Crash-recoverable actor dispatcher backed by [`DurableStore`]. The in-flight
/// signal is committed before user code runs and the resulting state is guarded
/// by both store revision CAS and the actor fencing token.
pub struct DurableActorRuntime {
    store: Arc<dyn DurableStore>,
}

impl DurableActorRuntime {
    pub fn new(store: Arc<dyn DurableStore>) -> Self {
        Self { store }
    }

    pub fn create(&self, id: ActorId, value: Value) -> VsResult<()> {
        let key = actor_record_key(&id)?;
        self.store.commit(
            &key,
            RevisionGuard::Create,
            DurableMutation::default().with_payload(&ActorState::new(id, value))?,
        )?;
        Ok(())
    }

    pub fn load(&self, id: &ActorId) -> VsResult<Option<ActorState>> {
        let Some(record) = self.store.load(&actor_record_key(id)?)? else {
            return Ok(None);
        };
        Ok(Some(record.decode()?))
    }

    pub fn enqueue(
        &self,
        id: &ActorId,
        owner: &str,
        now_ms: u64,
        ttl_ms: u64,
        message: ActorMessage,
    ) -> VsResult<bool> {
        let key = actor_record_key(id)?;
        let record = self
            .store
            .load(&key)?
            .ok_or_else(|| VsError::Runtime(format!("unknown actor {}:{}", id.kind, id.key)))?;
        let mut state: ActorState = record.decode()?;
        let lease = state.acquire_lease(owner, now_ms, ttl_ms)?;
        let accepted = state.enqueue(
            &lease.owner,
            &lease.token,
            now_ms,
            message.signal_id,
            message.topic,
            message.payload,
        )?;
        self.store.commit(
            &key,
            RevisionGuard::Match(record.revision),
            DurableMutation::default().with_payload(&state)?,
        )?;
        Ok(accepted)
    }

    pub async fn process_next(
        &self,
        id: &ActorId,
        owner: &str,
        ttl_ms: u64,
        clock: &dyn ExecutionClock,
        handler: &dyn ActorHandler,
    ) -> VsResult<Option<ActorSignal>> {
        let now_ms = clock.now_millis();
        let key = actor_record_key(id)?;
        let record = self
            .store
            .load(&key)?
            .ok_or_else(|| VsError::Runtime(format!("unknown actor {}:{}", id.kind, id.key)))?;
        let mut state: ActorState = record.decode()?;
        let lease = state.acquire_lease(owner, now_ms, ttl_ms)?;
        let Some(signal) = state.claim_next(&lease.owner, &lease.token, now_ms)? else {
            self.store.commit(
                &key,
                RevisionGuard::Match(record.revision),
                DurableMutation::default().with_payload(&state)?,
            )?;
            return Ok(None);
        };
        let claimed = self.store.commit(
            &key,
            RevisionGuard::Match(record.revision),
            DurableMutation::default().with_payload(&state)?,
        )?;

        let value = handler.handle(&state, &signal).await?;
        let latest = self
            .store
            .load(&key)?
            .ok_or_else(|| VsError::Runtime("actor disappeared during dispatch".into()))?;
        if latest.revision != claimed.revision {
            return Err(VsError::Runtime(
                "actor changed while its signal handler was running".into(),
            ));
        }
        let mut state: ActorState = latest.decode()?;
        state.commit(
            &lease.owner,
            &lease.token,
            clock.now_millis(),
            &signal.signal_id,
            value,
        )?;
        self.store.commit(
            &key,
            RevisionGuard::Match(latest.revision),
            DurableMutation::default().with_payload(&state)?,
        )?;
        Ok(Some(signal))
    }
}

fn actor_record_key(id: &ActorId) -> VsResult<RecordKey> {
    let encoded = serde_json::to_string(id)
        .map_err(|error| VsError::Runtime(format!("encode actor identity: {error}")))?;
    RecordKey::new("actors", encoded)
}
