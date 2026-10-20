//! Transport-neutral synchronization for a Prism editor and runtime.
//!
//! The protocol is intentionally independent from a wire transport and from
//! any engine or ECS. A future BRP adapter may translate its own requests into
//! [`RequestEnvelope`] values, call an [`Endpoint`], and translate the result
//! back. This crate does not implement BRP and does not depend on JSON, Bevy,
//! or any `bevy_*` crate.
//!
//! The core data flow is:
//!
//! ```text
//! transport adapter -> RequestEnvelope -> Endpoint -> ResponseEnvelope
//!                                           |
//!                                      SyncStore
//! ```
//!
//! [`encode_request`] and [`encode_response`] provide a deterministic binary
//! representation for adapters that need a compact framing-independent codec.

#![forbid(unsafe_code)]

mod codec;
mod endpoint;
mod model;
mod replica;

pub use codec::{
    decode_request, decode_response, encode_request, encode_response, CodecError, CodecLimits,
};
pub use endpoint::{Endpoint, InMemoryEndpoint, InMemoryStore, SyncEndpoint, SyncStore};
pub use model::{
    Capability, Change, ChangeSet, ComponentId, EndpointCapabilities, EndpointDescriptor, EntityId,
    EntityRecord, ErrorCode, PropertyPath, PropertySegment, ProtocolVersion, Request,
    RequestEnvelope, RequestId, Response, ResponseEnvelope, ResponsePayload, Revision, SyncError,
    TargetId, Value, WorldSnapshot,
};
pub use replica::{PendingRequest, Replica, SyncClient};
