//! Transport-neutral synchronization for a Prism editor and runtime.
//!
//! The protocol is intentionally independent from a wire transport and from
//! any engine or ECS. Binary synchronization and the JSON-RPC adapter both
//! translate into [`RequestEnvelope`] values, call an [`Endpoint`], and
//! translate the result back. The JSON-RPC layer is renderer/runtime-neutral
//! and uses BRP-style method naming without depending on a particular engine.
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
//! [`JsonRpcServer`] provides bounded JSON-RPC 2.0 requests, responses,
//! notifications, initialization/capability negotiation, and incremental
//! subscriptions. [`PieSession`] keeps runtime play state separate from an
//! editor snapshot and makes commit/rollback explicit.

#![forbid(unsafe_code)]

mod codec;
#[cfg(feature = "ecs")]
pub mod ecs;
mod endpoint;
mod jsonrpc;
mod model;
mod pie;
mod replica;
mod runtime;
mod transport;

pub use codec::{
    decode_request, decode_response, encode_request, encode_response, CodecError, CodecLimits,
};
pub use endpoint::{
    CommitResult, Endpoint, InMemoryEndpoint, InMemoryStore, SyncEndpoint, SyncStore,
};
pub use jsonrpc::{
    decode_json_rpc, decode_json_rpc_with_limits, encode_json_rpc, request, JsonNumber,
    JsonRpcCodecError, JsonRpcDispatchError, JsonRpcErrorObject, JsonRpcId, JsonRpcLimits,
    JsonRpcMessage, JsonRpcNotification, JsonRpcRequest, JsonRpcResponse, JsonRpcServer, JsonValue,
    NegotiatedCapabilities, SubscriptionId,
};
pub use model::{
    Capability, Change, ChangeSet, ComponentId, EndpointCapabilities, EndpointDescriptor, EntityId,
    EntityRecord, ErrorCode, PropertyPath, PropertySegment, ProtocolVersion, Request,
    RequestEnvelope, RequestId, Response, ResponseEnvelope, ResponsePayload, Revision, SyncError,
    TargetId, Value, WorldSnapshot,
};
pub use pie::{PieRuntimeRecord, PieSession, PieSessionId, PieState, PieStopPolicy, PieStopResult};
pub use replica::{PendingRequest, Replica, SyncClient};
pub use runtime::{
    EngineRuntimeAdapter, LoopbackRuntimeAdapter, RuntimeBridge, RuntimeError, RuntimeLaunchSpec,
    RuntimeLimits, RuntimeOperation, RuntimeRequest, RuntimeSession, RuntimeState, RuntimeStatus,
    RuntimeTransport,
};
pub use transport::{
    connect_tcp, FrameStage, JsonRpcClient, JsonRpcExchange, JsonRpcServerSession,
    LengthPrefixedStream, TcpJsonRpcListener, TransportError, TransportLimits,
};
