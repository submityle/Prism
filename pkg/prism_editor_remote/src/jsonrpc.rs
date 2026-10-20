//! Runtime-neutral JSON-RPC 2.0 and BRP-style synchronization adapter.
//!
//! The adapter intentionally owns only protocol concerns. Runtime state stays
//! behind [`crate::Endpoint`], while JSON is used only at the boundary. The
//! implementation has a bounded parser, deterministic object encoding, strict
//! request correlation, explicit initialization/capability negotiation, and
//! incremental target subscriptions.

use crate::model::{
    Capability, Change, ChangeSet, ComponentId, EndpointCapabilities, EndpointDescriptor, EntityId,
    PropertyPath, PropertySegment, ProtocolVersion, Request, RequestEnvelope, RequestId, Response,
    ResponsePayload, Revision, SyncError, TargetId, Value, WorldSnapshot,
};
use crate::Endpoint;
use std::collections::BTreeMap;
use std::fmt;

const JSONRPC_VERSION: &str = "2.0";
const METHOD_INITIALIZE: &str = "prism/initialize";
const METHOD_DISCOVER: &str = "rpc.discover";
const METHOD_DESCRIBE: &str = "prism/describe";
const METHOD_PING: &str = "prism/ping";
const METHOD_READ_SNAPSHOT: &str = "prism/world/readSnapshot";
const METHOD_READ_CHANGES: &str = "prism/world/readChanges";
const METHOD_APPLY_CHANGES: &str = "prism/world/applyChanges";
const METHOD_SUBSCRIBE: &str = "prism/world/subscribe";
const METHOD_UNSUBSCRIBE: &str = "prism/world/unsubscribe";
const METHOD_CHANGED: &str = "prism/world/changed";
const METHOD_RESYNC: &str = "prism/world/resyncRequired";

/// A JSON number retains integer width until a fractional/exponent form is
/// encountered. This avoids losing large revision and entity identifiers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum JsonNumber {
    /// A signed integer.
    Signed(i64),
    /// An unsigned integer.
    Unsigned(u64),
    /// A finite JSON floating-point number.
    Float(f64),
}

/// A bounded, deterministic JSON value used by the RPC boundary.
#[derive(Clone, Debug, PartialEq)]
pub enum JsonValue {
    /// JSON null.
    Null,
    /// JSON boolean.
    Bool(bool),
    /// JSON number.
    Number(JsonNumber),
    /// JSON string.
    String(String),
    /// JSON array.
    Array(Vec<Self>),
    /// JSON object with deterministic key ordering.
    Object(BTreeMap<String, Self>),
}

impl JsonValue {
    fn object(entries: impl IntoIterator<Item = (String, Self)>) -> Self {
        Self::Object(entries.into_iter().collect())
    }

    fn as_object(&self) -> Result<&BTreeMap<String, Self>, JsonRpcDispatchError> {
        match self {
            Self::Object(value) => Ok(value),
            _ => Err(JsonRpcDispatchError::invalid_params(
                "params must be an object",
            )),
        }
    }

    fn as_string(&self, field: &str) -> Result<&str, JsonRpcDispatchError> {
        match self {
            Self::String(value) => Ok(value),
            _ => Err(JsonRpcDispatchError::invalid_params(format!(
                "{field} must be a string"
            ))),
        }
    }

    fn as_u64(&self, field: &str) -> Result<u64, JsonRpcDispatchError> {
        match self {
            Self::Number(JsonNumber::Unsigned(value)) => Ok(*value),
            Self::Number(JsonNumber::Signed(value)) if *value >= 0 => Ok(*value as u64),
            _ => Err(JsonRpcDispatchError::invalid_params(format!(
                "{field} must be a non-negative integer"
            ))),
        }
    }

    fn as_array(&self, field: &str) -> Result<&[Self], JsonRpcDispatchError> {
        match self {
            Self::Array(value) => Ok(value),
            _ => Err(JsonRpcDispatchError::invalid_params(format!(
                "{field} must be an array"
            ))),
        }
    }
}

/// Limits applied to every JSON-RPC message before allocating nested data.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct JsonRpcLimits {
    /// Maximum UTF-8 input message size.
    pub max_message_bytes: usize,
    /// Maximum string size in bytes.
    pub max_string_bytes: usize,
    /// Maximum number of items in one array or object.
    pub max_collection_items: usize,
    /// Maximum nesting depth.
    pub max_nesting_depth: usize,
    /// Maximum active subscriptions per server.
    pub max_subscriptions: usize,
}

impl Default for JsonRpcLimits {
    fn default() -> Self {
        Self {
            max_message_bytes: 16 * 1024 * 1024,
            max_string_bytes: 1024 * 1024,
            max_collection_items: 100_000,
            max_nesting_depth: 64,
            max_subscriptions: 1024,
        }
    }
}

/// JSON-RPC request/notification correlation ID.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum JsonRpcId {
    /// An unsigned JSON integer ID.
    Number(u64),
    /// A non-empty string ID.
    String(String),
}

/// A JSON-RPC request with an ID and therefore a response obligation.
#[derive(Clone, Debug, PartialEq)]
pub struct JsonRpcRequest {
    /// Correlation ID.
    pub id: JsonRpcId,
    /// Method name.
    pub method: String,
    /// Optional method parameters.
    pub params: Option<JsonValue>,
}

/// A JSON-RPC notification without an ID or response obligation.
#[derive(Clone, Debug, PartialEq)]
pub struct JsonRpcNotification {
    /// Method name.
    pub method: String,
    /// Optional method parameters.
    pub params: Option<JsonValue>,
}

/// A JSON-RPC error object.
#[derive(Clone, Debug, PartialEq)]
pub struct JsonRpcErrorObject {
    /// Standard or implementation-defined error code.
    pub code: i32,
    /// Human-readable diagnostic.
    pub message: String,
    /// Optional structured diagnostic data.
    pub data: Option<JsonValue>,
}

/// A JSON-RPC response.
#[derive(Clone, Debug, PartialEq)]
pub struct JsonRpcResponse {
    /// Correlation ID copied from the request.
    pub id: JsonRpcId,
    /// Successful result, mutually exclusive with [`Self::error`].
    pub result: Option<JsonValue>,
    /// Error result, mutually exclusive with [`Self::result`].
    pub error: Option<JsonRpcErrorObject>,
}

impl JsonRpcResponse {
    fn success(id: JsonRpcId, result: JsonValue) -> Self {
        Self {
            id,
            result: Some(result),
            error: None,
        }
    }

    fn error(id: JsonRpcId, error: JsonRpcErrorObject) -> Self {
        Self {
            id,
            result: None,
            error: Some(error),
        }
    }
}

/// A parsed JSON-RPC message. Batch requests are represented explicitly so a
/// caller cannot accidentally treat a batch as a single correlated request.
#[derive(Clone, Debug, PartialEq)]
pub enum JsonRpcMessage {
    /// A request that requires one response.
    Request(JsonRpcRequest),
    /// A notification that requires no response.
    Notification(JsonRpcNotification),
    /// A response received from a peer.
    Response(JsonRpcResponse),
    /// A JSON-RPC batch.
    Batch(Vec<Self>),
}

/// Wire-level parse, encoding, or limit failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum JsonRpcCodecError {
    /// Input was not valid UTF-8.
    InvalidUtf8,
    /// The parser found invalid JSON at a byte offset.
    InvalidJson {
        /// Byte offset where parsing failed.
        offset: usize,
        /// Parser diagnostic.
        detail: String,
    },
    /// The message exceeded a configured limit.
    LimitExceeded {
        /// Limit category.
        kind: &'static str,
        /// Configured maximum.
        limit: usize,
    },
    /// The message had trailing non-whitespace data.
    TrailingData {
        /// Byte offset of the trailing data.
        offset: usize,
    },
    /// The JSON value is not a valid JSON-RPC message.
    InvalidMessage(String),
}

impl fmt::Display for JsonRpcCodecError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidUtf8 => formatter.write_str("JSON-RPC input is not UTF-8"),
            Self::InvalidJson { offset, detail } => {
                write!(formatter, "invalid JSON at byte {offset}: {detail}")
            }
            Self::LimitExceeded { kind, limit } => {
                write!(formatter, "JSON-RPC {kind} exceeds limit {limit}")
            }
            Self::TrailingData { offset } => {
                write!(formatter, "JSON-RPC trailing data at {offset}")
            }
            Self::InvalidMessage(detail) => write!(formatter, "invalid JSON-RPC message: {detail}"),
        }
    }
}

impl std::error::Error for JsonRpcCodecError {}

/// Parses one JSON-RPC message using default limits.
pub fn decode_json_rpc(bytes: &[u8]) -> Result<JsonRpcMessage, JsonRpcCodecError> {
    decode_json_rpc_with_limits(bytes, JsonRpcLimits::default())
}

/// Parses one JSON-RPC message with explicit boundary limits.
pub fn decode_json_rpc_with_limits(
    bytes: &[u8],
    limits: JsonRpcLimits,
) -> Result<JsonRpcMessage, JsonRpcCodecError> {
    if bytes.len() > limits.max_message_bytes {
        return Err(JsonRpcCodecError::LimitExceeded {
            kind: "message",
            limit: limits.max_message_bytes,
        });
    }
    let text = std::str::from_utf8(bytes).map_err(|_| JsonRpcCodecError::InvalidUtf8)?;
    let mut parser = JsonParser::new(text, limits);
    let value = parser
        .parse_value(0)
        .map_err(|detail| JsonRpcCodecError::InvalidJson {
            offset: parser.offset,
            detail,
        })?;
    parser.skip_whitespace();
    if parser.offset != text.len() {
        return Err(JsonRpcCodecError::TrailingData {
            offset: parser.offset,
        });
    }
    parse_message_value(value)
}

/// Encodes a JSON-RPC message deterministically.
pub fn encode_json_rpc(message: &JsonRpcMessage) -> Vec<u8> {
    let value = message_to_value(message);
    let mut output = String::new();
    write_json(&value, &mut output);
    output.into_bytes()
}

/// Encodes a single JSON-RPC request with a numeric ID.
#[must_use]
pub fn request(id: u64, method: impl Into<String>, params: Option<JsonValue>) -> JsonRpcMessage {
    JsonRpcMessage::Request(JsonRpcRequest {
        id: JsonRpcId::Number(id),
        method: method.into(),
        params,
    })
}

/// The result of a successful capability negotiation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NegotiatedCapabilities {
    /// Common protocol version selected by both sides.
    pub protocol: ProtocolVersion,
    /// Capabilities available to this session.
    pub capabilities: EndpointCapabilities,
}

/// Stable subscription identifier allocated by the server.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct SubscriptionId(String);

impl SubscriptionId {
    /// Returns the wire identifier.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug)]
struct Subscription {
    id: SubscriptionId,
    target: TargetId,
    next_revision: Revision,
}

/// Dispatch error returned for a valid call that cannot be executed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JsonRpcDispatchError {
    /// JSON-RPC error code.
    pub code: i32,
    /// Human-readable message.
    pub message: String,
}

impl JsonRpcDispatchError {
    fn invalid_params(message: impl Into<String>) -> Self {
        Self {
            code: -32602,
            message: message.into(),
        }
    }

    fn method_not_found(method: &str) -> Self {
        Self {
            code: -32601,
            message: format!("method `{method}` was not found"),
        }
    }

    fn invalid_request(message: impl Into<String>) -> Self {
        Self {
            code: -32600,
            message: message.into(),
        }
    }
}

/// A JSON-RPC server backed by any renderer/runtime-neutral endpoint.
pub struct JsonRpcServer<E> {
    endpoint: E,
    limits: JsonRpcLimits,
    negotiated: Option<NegotiatedCapabilities>,
    next_internal_request_id: u64,
    next_subscription_id: u64,
    subscriptions: BTreeMap<SubscriptionId, Subscription>,
}

impl<E: Endpoint> JsonRpcServer<E> {
    /// Creates an uninitialized JSON-RPC server.
    #[must_use]
    pub fn new(endpoint: E) -> Self {
        Self::with_limits(endpoint, JsonRpcLimits::default())
    }

    /// Creates a server with explicit protocol and allocation limits.
    #[must_use]
    pub fn with_limits(endpoint: E, limits: JsonRpcLimits) -> Self {
        Self {
            endpoint,
            limits,
            negotiated: None,
            next_internal_request_id: 1,
            next_subscription_id: 1,
            subscriptions: BTreeMap::new(),
        }
    }

    /// Returns the configured limits.
    #[must_use]
    pub const fn limits(&self) -> JsonRpcLimits {
        self.limits
    }

    /// Returns the negotiated session, if initialization succeeded.
    #[must_use]
    pub fn negotiated(&self) -> Option<&NegotiatedCapabilities> {
        self.negotiated.as_ref()
    }

    /// Returns the wrapped endpoint.
    #[must_use]
    pub fn endpoint(&self) -> &E {
        &self.endpoint
    }

    /// Returns the wrapped endpoint mutably.
    #[must_use]
    pub fn endpoint_mut(&mut self) -> &mut E {
        &mut self.endpoint
    }

    /// Decodes and dispatches one JSON-RPC message, returning zero or more
    /// encoded messages. A notification normally returns no response; a
    /// mutation may still produce subscription notifications.
    pub fn handle_bytes(&mut self, bytes: &[u8]) -> Result<Vec<Vec<u8>>, JsonRpcCodecError> {
        let message = decode_json_rpc_with_limits(bytes, self.limits)?;
        let messages = self
            .handle_message(message)
            .map_err(|error| JsonRpcCodecError::InvalidMessage(error.message))?;
        Ok(messages.iter().map(encode_json_rpc).collect())
    }

    /// Dispatches one parsed message.
    pub fn handle_message(
        &mut self,
        message: JsonRpcMessage,
    ) -> Result<Vec<JsonRpcMessage>, JsonRpcDispatchError> {
        match message {
            JsonRpcMessage::Request(call) => self.dispatch_call(call, true),
            JsonRpcMessage::Notification(call) => self.dispatch_notification(call),
            JsonRpcMessage::Batch(_) => Err(JsonRpcDispatchError::invalid_request(
                "batch messages are not accepted by this bounded session API",
            )),
            JsonRpcMessage::Response(_) => Err(JsonRpcDispatchError::invalid_request(
                "a server cannot dispatch a peer response",
            )),
        }
    }

    /// Publishes an externally observed contiguous change set to matching
    /// subscribers. A revision mismatch produces a deterministic resync
    /// notification instead of silently dropping data.
    pub fn publish_changes(
        &mut self,
        target: TargetId,
        changes: &ChangeSet,
    ) -> Result<Vec<JsonRpcMessage>, JsonRpcDispatchError> {
        let mut messages = Vec::new();
        for subscription in self.subscriptions.values_mut() {
            if subscription.target != target {
                continue;
            }
            messages.push(subscription_message(subscription, changes));
        }
        Ok(messages)
    }

    fn dispatch_notification(
        &mut self,
        call: JsonRpcNotification,
    ) -> Result<Vec<JsonRpcMessage>, JsonRpcDispatchError> {
        let request = JsonRpcRequest {
            id: JsonRpcId::Number(0),
            method: call.method,
            params: call.params,
        };
        let mut messages = self.dispatch_call(request, false)?;
        messages.retain(|message| matches!(message, JsonRpcMessage::Notification(_)));
        Ok(messages)
    }

    fn dispatch_call(
        &mut self,
        call: JsonRpcRequest,
        with_response: bool,
    ) -> Result<Vec<JsonRpcMessage>, JsonRpcDispatchError> {
        let id = call.id.clone();
        let result = self.dispatch_method(&call.method, call.params);
        match result {
            Ok(mut messages) => {
                if with_response {
                    let response_index = messages
                        .iter()
                        .position(|message| matches!(message, JsonRpcMessage::Response(_)))
                        .ok_or_else(|| {
                            JsonRpcDispatchError::invalid_request(
                                "method did not produce a response",
                            )
                        })?;
                    let response = messages.remove(response_index);
                    let result = match response {
                        JsonRpcMessage::Response(response) => {
                            response.result.unwrap_or(JsonValue::Null)
                        }
                        _ => JsonValue::Null,
                    };
                    let mut output = vec![JsonRpcMessage::Response(JsonRpcResponse::success(
                        id, result,
                    ))];
                    output.extend(messages);
                    Ok(output)
                } else {
                    Ok(messages)
                }
            }
            Err(error) if with_response => {
                Ok(vec![JsonRpcMessage::Response(JsonRpcResponse::error(
                    id,
                    JsonRpcErrorObject {
                        code: error.code,
                        message: error.message,
                        data: None,
                    },
                ))])
            }
            Err(error) => Err(error),
        }
    }

    fn dispatch_method(
        &mut self,
        method: &str,
        params: Option<JsonValue>,
    ) -> Result<Vec<JsonRpcMessage>, JsonRpcDispatchError> {
        if method == METHOD_INITIALIZE {
            return self.initialize(params).map(|result| {
                vec![JsonRpcMessage::Response(JsonRpcResponse::success(
                    JsonRpcId::Number(0),
                    result,
                ))]
            });
        }
        if method == METHOD_DISCOVER {
            return Ok(vec![JsonRpcMessage::Response(JsonRpcResponse::success(
                JsonRpcId::Number(0),
                discovery_value(),
            ))]);
        }
        if self.negotiated.is_none() {
            return Err(JsonRpcDispatchError::invalid_request(
                "prism/initialize must succeed before this method",
            ));
        }
        match method {
            METHOD_DESCRIBE => self.call_endpoint(Request::Describe),
            METHOD_PING => self.call_endpoint(Request::Ping),
            METHOD_READ_SNAPSHOT => {
                let object = params_object(params)?;
                let target = parse_target(required(&object, "target")?)?;
                self.call_endpoint(Request::ReadSnapshot { target })
            }
            METHOD_READ_CHANGES => {
                let object = params_object(params)?;
                let target = parse_target(required(&object, "target")?)?;
                let from_revision =
                    Revision::from_raw(required(&object, "fromRevision")?.as_u64("fromRevision")?);
                self.call_endpoint(Request::ReadChanges {
                    target,
                    from_revision,
                })
            }
            METHOD_APPLY_CHANGES => {
                let object = params_object(params)?;
                let target = parse_target(required(&object, "target")?)?;
                let base_revision =
                    Revision::from_raw(required(&object, "baseRevision")?.as_u64("baseRevision")?);
                let changes = parse_changes(required(&object, "changes")?)?;
                let messages = self.call_endpoint(Request::ApplyChanges {
                    target,
                    base_revision,
                    changes,
                })?;
                let notifications = messages
                    .iter()
                    .find_map(commit_changeset)
                    .map_or_else(Vec::new, |(target, change_set)| {
                        self.notifications_for_change_set(target, &change_set)
                    });
                Ok(messages.into_iter().chain(notifications).collect())
            }
            METHOD_SUBSCRIBE => self.subscribe(params),
            METHOD_UNSUBSCRIBE => self.unsubscribe(params),
            _ => Err(JsonRpcDispatchError::method_not_found(method)),
        }
    }

    fn initialize(&mut self, params: Option<JsonValue>) -> Result<JsonValue, JsonRpcDispatchError> {
        if self.negotiated.is_some() {
            return Err(JsonRpcDispatchError::invalid_request(
                "the session is already initialized",
            ));
        }
        let object = params_object(params)?;
        let protocol = parse_protocol(required(&object, "protocol")?)?;
        if protocol.major != ProtocolVersion::CURRENT.major {
            return Err(JsonRpcDispatchError {
                code: -32602,
                message: format!(
                    "unsupported protocol major {}; expected {}",
                    protocol.major,
                    ProtocolVersion::CURRENT.major
                ),
            });
        }
        let requested = parse_capabilities(required(&object, "capabilities")?)?;
        let endpoint = self.endpoint.descriptor();
        let selected =
            EndpointCapabilities::from_bits(endpoint.capabilities.bits() & requested.bits())
                .map_err(sync_dispatch_error)?;
        self.negotiated = Some(NegotiatedCapabilities {
            protocol: ProtocolVersion {
                major: protocol.major,
                minor: protocol.minor.min(endpoint.protocol.minor),
            },
            capabilities: selected,
        });
        Ok(JsonValue::object([
            (
                String::from("protocol"),
                protocol_value(self.negotiated.as_ref().expect("set").protocol),
            ),
            (String::from("capabilities"), capabilities_value(selected)),
            (String::from("endpoint"), descriptor_value(endpoint)),
        ]))
    }

    fn call_endpoint(
        &mut self,
        request: Request,
    ) -> Result<Vec<JsonRpcMessage>, JsonRpcDispatchError> {
        let request_id = RequestId::from_raw(self.next_internal_request_id);
        self.next_internal_request_id = self.next_internal_request_id.saturating_add(1).max(1);
        let response = self.endpoint.handle(RequestEnvelope {
            request_id,
            request,
        });
        Ok(vec![
            response_message(response).map_err(sync_dispatch_error)?
        ])
    }

    fn subscribe(
        &mut self,
        params: Option<JsonValue>,
    ) -> Result<Vec<JsonRpcMessage>, JsonRpcDispatchError> {
        if !self
            .negotiated
            .as_ref()
            .is_some_and(|value| value.capabilities.contains(Capability::IncrementalRead))
        {
            return Err(sync_dispatch_error(SyncError::UnsupportedCapability {
                capability: Capability::IncrementalRead,
            }));
        }
        if self.subscriptions.len() >= self.limits.max_subscriptions {
            return Err(JsonRpcDispatchError {
                code: -32010,
                message: String::from("subscription limit exceeded"),
            });
        }
        let object = params_object(params)?;
        let target = parse_target(required(&object, "target")?)?;
        let from_revision =
            Revision::from_raw(required(&object, "fromRevision")?.as_u64("fromRevision")?);
        if !self.endpoint.descriptor().targets.contains(&target) {
            return Err(sync_dispatch_error(SyncError::UnknownTarget { target }));
        }
        let id = SubscriptionId(format!("sub-{}", self.next_subscription_id));
        self.next_subscription_id = self.next_subscription_id.saturating_add(1).max(1);
        self.subscriptions.insert(
            id.clone(),
            Subscription {
                id: id.clone(),
                target,
                next_revision: from_revision,
            },
        );
        Ok(vec![JsonRpcMessage::Response(JsonRpcResponse::success(
            JsonRpcId::Number(0),
            JsonValue::object([
                (String::from("subscription"), JsonValue::String(id.0)),
                (String::from("target"), target_value(target)),
                (String::from("fromRevision"), revision_value(from_revision)),
            ]),
        ))])
    }

    fn unsubscribe(
        &mut self,
        params: Option<JsonValue>,
    ) -> Result<Vec<JsonRpcMessage>, JsonRpcDispatchError> {
        let object = params_object(params)?;
        let id = required(&object, "subscription")?.as_string("subscription")?;
        if self
            .subscriptions
            .remove(&SubscriptionId(id.to_owned()))
            .is_none()
        {
            return Err(JsonRpcDispatchError {
                code: -32011,
                message: format!("unknown subscription `{id}`"),
            });
        }
        Ok(vec![JsonRpcMessage::Response(JsonRpcResponse::success(
            JsonRpcId::Number(0),
            JsonValue::Bool(true),
        ))])
    }

    fn notifications_for_change_set(
        &mut self,
        target: TargetId,
        changes: &ChangeSet,
    ) -> Vec<JsonRpcMessage> {
        self.subscriptions
            .values_mut()
            .filter(|subscription| subscription.target == target)
            .map(|subscription| subscription_message(subscription, changes))
            .collect()
    }
}

fn subscription_message(subscription: &mut Subscription, changes: &ChangeSet) -> JsonRpcMessage {
    let expected_revision = subscription.next_revision;
    let notification = if subscription.next_revision == changes.base_revision {
        subscription.next_revision = changes.target_revision;
        JsonRpcNotification {
            method: String::from(METHOD_CHANGED),
            params: Some(JsonValue::object([
                (
                    String::from("subscription"),
                    JsonValue::String(subscription.id.0.clone()),
                ),
                (String::from("target"), target_value(subscription.target)),
                (String::from("changes"), changeset_value(changes)),
            ])),
        }
    } else {
        subscription.next_revision = changes.target_revision;
        JsonRpcNotification {
            method: String::from(METHOD_RESYNC),
            params: Some(JsonValue::object([
                (
                    String::from("subscription"),
                    JsonValue::String(subscription.id.0.clone()),
                ),
                (String::from("target"), target_value(subscription.target)),
                (
                    String::from("expectedRevision"),
                    revision_value(expected_revision),
                ),
                (
                    String::from("receivedBaseRevision"),
                    revision_value(changes.base_revision),
                ),
                (
                    String::from("receivedTargetRevision"),
                    revision_value(changes.target_revision),
                ),
            ])),
        }
    };
    JsonRpcMessage::Notification(notification)
}

fn commit_changeset(message: &JsonRpcMessage) -> Option<(TargetId, ChangeSet)> {
    let JsonRpcMessage::Response(response) = message else {
        return None;
    };
    let result = response.result.as_ref()?;
    let object = match result {
        JsonValue::Object(object) => object,
        _ => return None,
    };
    let target = parse_target(object.get("target")?).ok()?;
    let revision = Revision::from_raw(object.get("revision")?.as_u64("revision").ok()?);
    let changes = parse_changes(object.get("changes")?).ok()?;
    let base_revision = revision.prev();
    Some((
        target,
        ChangeSet::new(base_revision, revision, changes).ok()?,
    ))
}

fn response_message(response: crate::ResponseEnvelope) -> Result<JsonRpcMessage, SyncError> {
    match response.response {
        Response::Success(payload) => Ok(JsonRpcMessage::Response(JsonRpcResponse::success(
            JsonRpcId::Number(response.request_id.raw()),
            payload_value(&payload),
        ))),
        Response::Error(error) => Err(error),
    }
}

fn sync_dispatch_error(error: SyncError) -> JsonRpcDispatchError {
    JsonRpcDispatchError {
        code: -32000 - error.code() as i32,
        message: error.to_string(),
    }
}

fn required<'a>(
    object: &'a BTreeMap<String, JsonValue>,
    key: &str,
) -> Result<&'a JsonValue, JsonRpcDispatchError> {
    object
        .get(key)
        .ok_or_else(|| JsonRpcDispatchError::invalid_params(format!("missing parameter `{key}`")))
}

fn params_object(
    params: Option<JsonValue>,
) -> Result<BTreeMap<String, JsonValue>, JsonRpcDispatchError> {
    params.unwrap_or(JsonValue::Null).as_object().cloned()
}

fn parse_target(value: &JsonValue) -> Result<TargetId, JsonRpcDispatchError> {
    value
        .as_string("target")?
        .parse::<u128>()
        .map(TargetId::from_raw)
        .map_err(|_| JsonRpcDispatchError::invalid_params("target must be a decimal u128 string"))
}

fn parse_entity(value: &JsonValue) -> Result<EntityId, JsonRpcDispatchError> {
    value
        .as_string("entity")?
        .parse::<u128>()
        .map(EntityId::from_raw)
        .map_err(|_| JsonRpcDispatchError::invalid_params("entity must be a decimal u128 string"))
}

fn parse_protocol(value: &JsonValue) -> Result<ProtocolVersion, JsonRpcDispatchError> {
    let object = value.as_object()?;
    Ok(ProtocolVersion {
        major: u16::try_from(required(object, "major")?.as_u64("major")?)
            .map_err(|_| JsonRpcDispatchError::invalid_params("protocol.major is too large"))?,
        minor: u16::try_from(required(object, "minor")?.as_u64("minor")?)
            .map_err(|_| JsonRpcDispatchError::invalid_params("protocol.minor is too large"))?,
    })
}

fn parse_capabilities(value: &JsonValue) -> Result<EndpointCapabilities, JsonRpcDispatchError> {
    let values = value.as_array("capabilities")?;
    let mut capabilities = EndpointCapabilities::empty();
    for value in values {
        let name = value.as_string("capability")?;
        capabilities.insert(capability_from_name(name).ok_or_else(|| {
            JsonRpcDispatchError::invalid_params(format!("unknown capability `{name}`"))
        })?);
    }
    Ok(capabilities)
}

fn parse_changes(value: &JsonValue) -> Result<Vec<Change>, JsonRpcDispatchError> {
    value
        .as_array("changes")?
        .iter()
        .map(parse_change)
        .collect()
}

fn parse_change(value: &JsonValue) -> Result<Change, JsonRpcDispatchError> {
    let object = value.as_object()?;
    let operation = required(object, "op")?.as_string("op")?;
    let entity = || parse_entity(required(object, "entity")?);
    match operation {
        "createEntity" => Ok(Change::CreateEntity { entity: entity()? }),
        "deleteEntity" => Ok(Change::DeleteEntity { entity: entity()? }),
        "setComponent" => Ok(Change::SetComponent {
            entity: entity()?,
            component: ComponentId::new(required(object, "component")?.as_string("component")?)
                .map_err(sync_dispatch_error)?,
            value: json_to_value(required(object, "value")?)?,
        }),
        "removeComponent" => Ok(Change::RemoveComponent {
            entity: entity()?,
            component: ComponentId::new(required(object, "component")?.as_string("component")?)
                .map_err(sync_dispatch_error)?,
        }),
        "setProperty" => Ok(Change::SetProperty {
            entity: entity()?,
            component: ComponentId::new(required(object, "component")?.as_string("component")?)
                .map_err(sync_dispatch_error)?,
            path: parse_path(required(object, "path")?)?,
            value: json_to_value(required(object, "value")?)?,
        }),
        "removeProperty" => Ok(Change::RemoveProperty {
            entity: entity()?,
            component: ComponentId::new(required(object, "component")?.as_string("component")?)
                .map_err(sync_dispatch_error)?,
            path: parse_path(required(object, "path")?)?,
        }),
        _ => Err(JsonRpcDispatchError::invalid_params(format!(
            "unknown change operation `{operation}`"
        ))),
    }
}

fn parse_path(value: &JsonValue) -> Result<PropertyPath, JsonRpcDispatchError> {
    let segments = value
        .as_array("path")?
        .iter()
        .map(|segment| {
            let object = segment.as_object()?;
            let kind = required(object, "kind")?.as_string("path.kind")?;
            match kind {
                "field" => Ok(PropertySegment::Field(
                    required(object, "value")?
                        .as_string("path.value")?
                        .to_owned(),
                )),
                "key" => Ok(PropertySegment::Key(
                    required(object, "value")?
                        .as_string("path.value")?
                        .to_owned(),
                )),
                "index" => Ok(PropertySegment::Index(
                    u32::try_from(required(object, "value")?.as_u64("path.value")?).map_err(
                        |_| JsonRpcDispatchError::invalid_params("path index is too large"),
                    )?,
                )),
                _ => Err(JsonRpcDispatchError::invalid_params(format!(
                    "unknown path segment `{kind}`"
                ))),
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    PropertyPath::new(segments).map_err(sync_dispatch_error)
}

fn json_to_value(value: &JsonValue) -> Result<Value, JsonRpcDispatchError> {
    match value {
        JsonValue::Null => Ok(Value::Null),
        JsonValue::Bool(value) => Ok(Value::Bool(*value)),
        JsonValue::Number(JsonNumber::Signed(value)) => Ok(Value::Signed(*value)),
        JsonValue::Number(JsonNumber::Unsigned(value)) => Ok(Value::Unsigned(*value)),
        JsonValue::Number(JsonNumber::Float(value)) => Ok(Value::float(*value)),
        JsonValue::String(value) => Ok(Value::String(value.clone())),
        JsonValue::Array(values) => values
            .iter()
            .map(json_to_value)
            .collect::<Result<Vec<_>, _>>()
            .map(Value::Array),
        JsonValue::Object(object) => {
            if let Some(JsonValue::String(bits)) = object.get("$prism.floatBits") {
                let bits = u64::from_str_radix(bits, 16).map_err(|_| {
                    JsonRpcDispatchError::invalid_params("$prism.floatBits must be hexadecimal")
                })?;
                return Ok(Value::Float(bits));
            }
            if let Some(JsonValue::Array(bytes)) = object.get("$prism.bytes") {
                let bytes = bytes
                    .iter()
                    .map(|value| {
                        u8::try_from(value.as_u64("$prism.bytes item")?).map_err(|_| {
                            JsonRpcDispatchError::invalid_params("byte value must be 0..255")
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                return Ok(Value::Bytes(bytes));
            }
            object
                .iter()
                .map(|(key, value)| Ok((key.clone(), json_to_value(value)?)))
                .collect::<Result<BTreeMap<_, _>, JsonRpcDispatchError>>()
                .map(Value::Object)
        }
    }
}

fn value_to_json(value: &Value) -> JsonValue {
    match value {
        Value::Null => JsonValue::Null,
        Value::Bool(value) => JsonValue::Bool(*value),
        Value::Signed(value) => JsonValue::Number(JsonNumber::Signed(*value)),
        Value::Unsigned(value) => JsonValue::Number(JsonNumber::Unsigned(*value)),
        Value::Float(bits) => JsonValue::object([(
            String::from("$prism.floatBits"),
            JsonValue::String(format!("{bits:016x}")),
        )]),
        Value::String(value) => JsonValue::String(value.clone()),
        Value::Bytes(values) => JsonValue::object([(
            String::from("$prism.bytes"),
            JsonValue::Array(
                values
                    .iter()
                    .map(|value| JsonValue::Number(JsonNumber::Unsigned(u64::from(*value))))
                    .collect(),
            ),
        )]),
        Value::Array(values) => JsonValue::Array(values.iter().map(value_to_json).collect()),
        Value::Object(values) => JsonValue::Object(
            values
                .iter()
                .map(|(key, value)| (key.clone(), value_to_json(value)))
                .collect(),
        ),
    }
}

fn payload_value(payload: &ResponsePayload) -> JsonValue {
    match payload {
        ResponsePayload::Descriptor(descriptor) => descriptor_value(descriptor),
        ResponsePayload::Snapshot(snapshot) => snapshot_value(snapshot),
        ResponsePayload::Changes(changes) => changeset_value(changes),
        ResponsePayload::Commit {
            target,
            revision,
            changes,
        } => JsonValue::object([
            (String::from("target"), target_value(*target)),
            (String::from("revision"), revision_value(*revision)),
            (
                String::from("changes"),
                JsonValue::Array(changes.iter().map(change_value).collect()),
            ),
        ]),
        ResponsePayload::Pong => JsonValue::object([(String::from("pong"), JsonValue::Bool(true))]),
    }
}

fn descriptor_value(descriptor: &EndpointDescriptor) -> JsonValue {
    JsonValue::object([
        (
            String::from("protocol"),
            protocol_value(descriptor.protocol),
        ),
        (
            String::from("name"),
            JsonValue::String(descriptor.name.clone()),
        ),
        (
            String::from("capabilities"),
            capabilities_value(descriptor.capabilities),
        ),
        (
            String::from("targets"),
            JsonValue::Array(
                descriptor
                    .targets
                    .iter()
                    .copied()
                    .map(target_value)
                    .collect(),
            ),
        ),
    ])
}

fn snapshot_value(snapshot: &WorldSnapshot) -> JsonValue {
    let entities = snapshot
        .entities
        .iter()
        .map(|(entity, record)| {
            (
                entity.raw().to_string(),
                JsonValue::Object(
                    record
                        .components
                        .iter()
                        .map(|(component, value)| {
                            (component.as_str().to_owned(), value_to_json(value))
                        })
                        .collect(),
                ),
            )
        })
        .collect();
    JsonValue::object([
        (String::from("target"), target_value(snapshot.target)),
        (String::from("revision"), revision_value(snapshot.revision)),
        (String::from("entities"), JsonValue::Object(entities)),
        (
            String::from("metadata"),
            JsonValue::Object(
                snapshot
                    .metadata
                    .iter()
                    .map(|(key, value)| (key.clone(), value_to_json(value)))
                    .collect(),
            ),
        ),
    ])
}

fn changeset_value(changes: &ChangeSet) -> JsonValue {
    JsonValue::object([
        (
            String::from("baseRevision"),
            revision_value(changes.base_revision),
        ),
        (
            String::from("targetRevision"),
            revision_value(changes.target_revision),
        ),
        (
            String::from("changes"),
            JsonValue::Array(changes.changes.iter().map(change_value).collect()),
        ),
    ])
}

fn change_value(change: &Change) -> JsonValue {
    match change {
        Change::CreateEntity { entity } => JsonValue::object([
            (
                String::from("op"),
                JsonValue::String(String::from("createEntity")),
            ),
            (String::from("entity"), entity_value(*entity)),
        ]),
        Change::DeleteEntity { entity } => JsonValue::object([
            (
                String::from("op"),
                JsonValue::String(String::from("deleteEntity")),
            ),
            (String::from("entity"), entity_value(*entity)),
        ]),
        Change::SetComponent {
            entity,
            component,
            value,
        } => JsonValue::object([
            (
                String::from("op"),
                JsonValue::String(String::from("setComponent")),
            ),
            (String::from("entity"), entity_value(*entity)),
            (
                String::from("component"),
                JsonValue::String(component.as_str().to_owned()),
            ),
            (String::from("value"), value_to_json(value)),
        ]),
        Change::RemoveComponent { entity, component } => JsonValue::object([
            (
                String::from("op"),
                JsonValue::String(String::from("removeComponent")),
            ),
            (String::from("entity"), entity_value(*entity)),
            (
                String::from("component"),
                JsonValue::String(component.as_str().to_owned()),
            ),
        ]),
        Change::SetProperty {
            entity,
            component,
            path,
            value,
        } => JsonValue::object([
            (
                String::from("op"),
                JsonValue::String(String::from("setProperty")),
            ),
            (String::from("entity"), entity_value(*entity)),
            (
                String::from("component"),
                JsonValue::String(component.as_str().to_owned()),
            ),
            (String::from("path"), path_value(path)),
            (String::from("value"), value_to_json(value)),
        ]),
        Change::RemoveProperty {
            entity,
            component,
            path,
        } => JsonValue::object([
            (
                String::from("op"),
                JsonValue::String(String::from("removeProperty")),
            ),
            (String::from("entity"), entity_value(*entity)),
            (
                String::from("component"),
                JsonValue::String(component.as_str().to_owned()),
            ),
            (String::from("path"), path_value(path)),
        ]),
    }
}

fn path_value(path: &PropertyPath) -> JsonValue {
    JsonValue::Array(
        path.segments()
            .iter()
            .map(|segment| match segment {
                PropertySegment::Field(value) => JsonValue::object([
                    (
                        String::from("kind"),
                        JsonValue::String(String::from("field")),
                    ),
                    (String::from("value"), JsonValue::String(value.clone())),
                ]),
                PropertySegment::Index(value) => JsonValue::object([
                    (
                        String::from("kind"),
                        JsonValue::String(String::from("index")),
                    ),
                    (
                        String::from("value"),
                        JsonValue::Number(JsonNumber::Unsigned(u64::from(*value))),
                    ),
                ]),
                PropertySegment::Key(value) => JsonValue::object([
                    (String::from("kind"), JsonValue::String(String::from("key"))),
                    (String::from("value"), JsonValue::String(value.clone())),
                ]),
            })
            .collect(),
    )
}

fn target_value(target: TargetId) -> JsonValue {
    JsonValue::String(target.raw().to_string())
}

fn entity_value(entity: EntityId) -> JsonValue {
    JsonValue::String(entity.raw().to_string())
}

fn revision_value(revision: Revision) -> JsonValue {
    JsonValue::Number(JsonNumber::Unsigned(revision.raw()))
}

fn protocol_value(protocol: ProtocolVersion) -> JsonValue {
    JsonValue::object([
        (
            String::from("major"),
            JsonValue::Number(JsonNumber::Unsigned(u64::from(protocol.major))),
        ),
        (
            String::from("minor"),
            JsonValue::Number(JsonNumber::Unsigned(u64::from(protocol.minor))),
        ),
    ])
}

fn capability_name(capability: Capability) -> &'static str {
    match capability {
        Capability::SnapshotRead => "snapshotRead",
        Capability::IncrementalRead => "incrementalRead",
        Capability::ComponentWrite => "componentWrite",
        Capability::PropertyWrite => "propertyWrite",
        Capability::RevisionChecks => "revisionChecks",
        Capability::DeterministicCodec => "deterministicCodec",
    }
}

fn capability_from_name(name: &str) -> Option<Capability> {
    [
        Capability::SnapshotRead,
        Capability::IncrementalRead,
        Capability::ComponentWrite,
        Capability::PropertyWrite,
        Capability::RevisionChecks,
        Capability::DeterministicCodec,
    ]
    .into_iter()
    .find(|capability| capability_name(*capability) == name)
}

fn capabilities_value(capabilities: EndpointCapabilities) -> JsonValue {
    JsonValue::Array(
        [
            Capability::SnapshotRead,
            Capability::IncrementalRead,
            Capability::ComponentWrite,
            Capability::PropertyWrite,
            Capability::RevisionChecks,
            Capability::DeterministicCodec,
        ]
        .into_iter()
        .filter(|capability| capabilities.contains(*capability))
        .map(|capability| JsonValue::String(String::from(capability_name(capability))))
        .collect(),
    )
}

fn protocol_error_value(error: &JsonRpcErrorObject) -> JsonValue {
    let mut object = BTreeMap::new();
    object.insert(
        String::from("code"),
        JsonValue::Number(JsonNumber::Signed(i64::from(error.code))),
    );
    object.insert(
        String::from("message"),
        JsonValue::String(error.message.clone()),
    );
    if let Some(data) = &error.data {
        object.insert(String::from("data"), data.clone());
    }
    JsonValue::Object(object)
}

fn discovery_value() -> JsonValue {
    JsonValue::object([
        (
            String::from("jsonrpc"),
            JsonValue::String(String::from(JSONRPC_VERSION)),
        ),
        (
            String::from("methods"),
            JsonValue::Array(
                [
                    METHOD_INITIALIZE,
                    METHOD_DISCOVER,
                    METHOD_DESCRIBE,
                    METHOD_PING,
                    METHOD_READ_SNAPSHOT,
                    METHOD_READ_CHANGES,
                    METHOD_APPLY_CHANGES,
                    METHOD_SUBSCRIBE,
                    METHOD_UNSUBSCRIBE,
                    METHOD_CHANGED,
                    METHOD_RESYNC,
                ]
                .into_iter()
                .map(|value| JsonValue::String(String::from(value)))
                .collect(),
            ),
        ),
    ])
}

fn parse_message_value(value: JsonValue) -> Result<JsonRpcMessage, JsonRpcCodecError> {
    if let JsonValue::Array(values) = value {
        if values.is_empty() {
            return Err(JsonRpcCodecError::InvalidMessage(String::from(
                "a JSON-RPC batch must not be empty",
            )));
        }
        return values
            .into_iter()
            .map(parse_message_value)
            .collect::<Result<Vec<_>, _>>()
            .map(JsonRpcMessage::Batch);
    }
    let object = match value {
        JsonValue::Object(object) => object,
        _ => {
            return Err(JsonRpcCodecError::InvalidMessage(String::from(
                "message must be an object or batch",
            )))
        }
    };
    match object.get("jsonrpc") {
        Some(JsonValue::String(version)) if version == JSONRPC_VERSION => {}
        _ => {
            return Err(JsonRpcCodecError::InvalidMessage(String::from(
                "jsonrpc must be the string `2.0`",
            )))
        }
    }
    if let Some(method) = object.get("method") {
        let method = match method {
            JsonValue::String(method) if !method.is_empty() => method.clone(),
            _ => {
                return Err(JsonRpcCodecError::InvalidMessage(String::from(
                    "method must be a non-empty string",
                )))
            }
        };
        let params = object.get("params").cloned();
        return match object.get("id") {
            Some(id) => Ok(JsonRpcMessage::Request(JsonRpcRequest {
                id: parse_id(id)?,
                method,
                params,
            })),
            None => Ok(JsonRpcMessage::Notification(JsonRpcNotification {
                method,
                params,
            })),
        };
    }
    let id =
        parse_id(object.get("id").ok_or_else(|| {
            JsonRpcCodecError::InvalidMessage(String::from("response missing id"))
        })?)?;
    let result = object.get("result").cloned();
    let error = object.get("error").map(parse_error).transpose()?;
    if result.is_some() == error.is_some() {
        return Err(JsonRpcCodecError::InvalidMessage(String::from(
            "response must contain exactly one of result or error",
        )));
    }
    Ok(JsonRpcMessage::Response(JsonRpcResponse {
        id,
        result,
        error,
    }))
}

fn parse_id(value: &JsonValue) -> Result<JsonRpcId, JsonRpcCodecError> {
    match value {
        JsonValue::Number(JsonNumber::Unsigned(value)) => Ok(JsonRpcId::Number(*value)),
        JsonValue::Number(JsonNumber::Signed(value)) if *value >= 0 => {
            Ok(JsonRpcId::Number(*value as u64))
        }
        JsonValue::String(value) if !value.is_empty() => Ok(JsonRpcId::String(value.clone())),
        _ => Err(JsonRpcCodecError::InvalidMessage(String::from(
            "id must be a non-negative integer or non-empty string",
        ))),
    }
}

fn parse_error(value: &JsonValue) -> Result<JsonRpcErrorObject, JsonRpcCodecError> {
    let JsonValue::Object(object) = value else {
        return Err(JsonRpcCodecError::InvalidMessage(String::from(
            "error must be an object",
        )));
    };
    let code = match object.get("code") {
        Some(JsonValue::Number(JsonNumber::Signed(code))) => i32::try_from(*code).map_err(|_| ()),
        Some(JsonValue::Number(JsonNumber::Unsigned(code))) => i32::try_from(*code).map_err(|_| ()),
        _ => Err(()),
    }
    .map_err(|_| JsonRpcCodecError::InvalidMessage(String::from("error.code must be an i32")))?;
    let message = match object.get("message") {
        Some(JsonValue::String(message)) => message.clone(),
        _ => {
            return Err(JsonRpcCodecError::InvalidMessage(String::from(
                "error.message must be a string",
            )))
        }
    };
    Ok(JsonRpcErrorObject {
        code,
        message,
        data: object.get("data").cloned(),
    })
}

fn message_to_value(message: &JsonRpcMessage) -> JsonValue {
    match message {
        JsonRpcMessage::Request(request) => {
            let mut object = BTreeMap::new();
            object.insert(String::from("id"), id_value(&request.id));
            object.insert(
                String::from("jsonrpc"),
                JsonValue::String(String::from(JSONRPC_VERSION)),
            );
            object.insert(
                String::from("method"),
                JsonValue::String(request.method.clone()),
            );
            if let Some(params) = &request.params {
                object.insert(String::from("params"), params.clone());
            }
            JsonValue::Object(object)
        }
        JsonRpcMessage::Notification(notification) => {
            let mut object = BTreeMap::new();
            object.insert(
                String::from("jsonrpc"),
                JsonValue::String(String::from(JSONRPC_VERSION)),
            );
            object.insert(
                String::from("method"),
                JsonValue::String(notification.method.clone()),
            );
            if let Some(params) = &notification.params {
                object.insert(String::from("params"), params.clone());
            }
            JsonValue::Object(object)
        }
        JsonRpcMessage::Response(response) => {
            let mut object = BTreeMap::new();
            object.insert(String::from("id"), id_value(&response.id));
            object.insert(
                String::from("jsonrpc"),
                JsonValue::String(String::from(JSONRPC_VERSION)),
            );
            if let Some(result) = &response.result {
                object.insert(String::from("result"), result.clone());
            }
            if let Some(error) = &response.error {
                object.insert(String::from("error"), protocol_error_value(error));
            }
            JsonValue::Object(object)
        }
        JsonRpcMessage::Batch(messages) => {
            JsonValue::Array(messages.iter().map(message_to_value).collect())
        }
    }
}

fn id_value(id: &JsonRpcId) -> JsonValue {
    match id {
        JsonRpcId::Number(value) => JsonValue::Number(JsonNumber::Unsigned(*value)),
        JsonRpcId::String(value) => JsonValue::String(value.clone()),
    }
}

fn write_json(value: &JsonValue, output: &mut String) {
    match value {
        JsonValue::Null => output.push_str("null"),
        JsonValue::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
        JsonValue::Number(JsonNumber::Signed(value)) => output.push_str(&value.to_string()),
        JsonValue::Number(JsonNumber::Unsigned(value)) => output.push_str(&value.to_string()),
        JsonValue::Number(JsonNumber::Float(value)) => output.push_str(&value.to_string()),
        JsonValue::String(value) => write_string(value, output),
        JsonValue::Array(values) => {
            output.push('[');
            for (index, value) in values.iter().enumerate() {
                if index != 0 {
                    output.push(',');
                }
                write_json(value, output);
            }
            output.push(']');
        }
        JsonValue::Object(values) => {
            output.push('{');
            for (index, (key, value)) in values.iter().enumerate() {
                if index != 0 {
                    output.push(',');
                }
                write_string(key, output);
                output.push(':');
                write_json(value, output);
            }
            output.push('}');
        }
    }
}

fn write_string(value: &str, output: &mut String) {
    output.push('"');
    for character in value.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\u{08}' => output.push_str("\\b"),
            '\u{0c}' => output.push_str("\\f"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            character if character <= '\u{1f}' => {
                output.push_str(&format!("\\u{:04x}", character as u32));
            }
            character => output.push(character),
        }
    }
    output.push('"');
}

struct JsonParser<'a> {
    text: &'a str,
    offset: usize,
    limits: JsonRpcLimits,
}

impl<'a> JsonParser<'a> {
    fn new(text: &'a str, limits: JsonRpcLimits) -> Self {
        Self {
            text,
            offset: 0,
            limits,
        }
    }

    fn parse_value(&mut self, depth: usize) -> Result<JsonValue, String> {
        if depth > self.limits.max_nesting_depth {
            return Err(format!(
                "nesting exceeds limit {}",
                self.limits.max_nesting_depth
            ));
        }
        self.skip_whitespace();
        match self.peek() {
            Some('n') => self.parse_literal("null", JsonValue::Null),
            Some('t') => self.parse_literal("true", JsonValue::Bool(true)),
            Some('f') => self.parse_literal("false", JsonValue::Bool(false)),
            Some('"') => self.parse_string().map(JsonValue::String),
            Some('[') => self.parse_array(depth),
            Some('{') => self.parse_object(depth),
            Some('-' | '0'..='9') => self.parse_number(),
            Some(character) => Err(format!("unexpected character `{character}`")),
            None => Err(String::from("unexpected end of input")),
        }
    }

    fn parse_literal<T>(&mut self, literal: &str, value: T) -> Result<T, String> {
        if self.text[self.offset..].starts_with(literal) {
            self.offset += literal.len();
            Ok(value)
        } else {
            Err(format!("expected {literal}"))
        }
    }

    fn parse_array(&mut self, depth: usize) -> Result<JsonValue, String> {
        self.offset += 1;
        let mut values = Vec::new();
        self.skip_whitespace();
        if self.consume(']') {
            return Ok(JsonValue::Array(values));
        }
        loop {
            if values.len() >= self.limits.max_collection_items {
                return Err(format!(
                    "array exceeds limit {}",
                    self.limits.max_collection_items
                ));
            }
            values.push(self.parse_value(depth + 1)?);
            self.skip_whitespace();
            if self.consume(']') {
                return Ok(JsonValue::Array(values));
            }
            if !self.consume(',') {
                return Err(String::from("expected `,` or `]`"));
            }
        }
    }

    fn parse_object(&mut self, depth: usize) -> Result<JsonValue, String> {
        self.offset += 1;
        let mut values = BTreeMap::new();
        self.skip_whitespace();
        if self.consume('}') {
            return Ok(JsonValue::Object(values));
        }
        loop {
            if values.len() >= self.limits.max_collection_items {
                return Err(format!(
                    "object exceeds limit {}",
                    self.limits.max_collection_items
                ));
            }
            self.skip_whitespace();
            if self.peek() != Some('"') {
                return Err(String::from("object key must be a string"));
            }
            let key = self.parse_string()?;
            if key.len() > self.limits.max_string_bytes {
                return Err(format!(
                    "string exceeds limit {}",
                    self.limits.max_string_bytes
                ));
            }
            self.skip_whitespace();
            if !self.consume(':') {
                return Err(String::from("expected `:` after object key"));
            }
            let value = self.parse_value(depth + 1)?;
            if values.insert(key, value).is_some() {
                return Err(String::from("duplicate object key"));
            }
            self.skip_whitespace();
            if self.consume('}') {
                return Ok(JsonValue::Object(values));
            }
            if !self.consume(',') {
                return Err(String::from("expected `,` or `}`"));
            }
        }
    }

    fn parse_string(&mut self) -> Result<String, String> {
        if !self.consume('"') {
            return Err(String::from("expected string"));
        }
        let mut output = String::new();
        loop {
            let character = self
                .next()
                .ok_or_else(|| String::from("unterminated string"))?;
            match character {
                '"' => {
                    if output.len() > self.limits.max_string_bytes {
                        return Err(format!(
                            "string exceeds limit {}",
                            self.limits.max_string_bytes
                        ));
                    }
                    return Ok(output);
                }
                '\\' => {
                    let escaped = self
                        .next()
                        .ok_or_else(|| String::from("unterminated escape"))?;
                    match escaped {
                        '"' => output.push('"'),
                        '\\' => output.push('\\'),
                        '/' => output.push('/'),
                        'b' => output.push('\u{08}'),
                        'f' => output.push('\u{0c}'),
                        'n' => output.push('\n'),
                        'r' => output.push('\r'),
                        't' => output.push('\t'),
                        'u' => output.push(self.parse_unicode_escape()?),
                        _ => return Err(String::from("invalid string escape")),
                    }
                }
                character if character <= '\u{1f}' => {
                    return Err(String::from("unescaped control character in string"));
                }
                character => output.push(character),
            }
            if output.len() > self.limits.max_string_bytes {
                return Err(format!(
                    "string exceeds limit {}",
                    self.limits.max_string_bytes
                ));
            }
        }
    }

    fn parse_unicode_escape(&mut self) -> Result<char, String> {
        let first = self.parse_hex_quad()?;
        if (0xd800..=0xdbff).contains(&first) {
            if self.next() != Some('\\') || self.next() != Some('u') {
                return Err(String::from("high surrogate is missing a low surrogate"));
            }
            let second = self.parse_hex_quad()?;
            if !(0xdc00..=0xdfff).contains(&second) {
                return Err(String::from("invalid low surrogate"));
            }
            let code_point =
                0x1_0000 + ((u32::from(first) - 0xd800) << 10) + (u32::from(second) - 0xdc00);
            char::from_u32(code_point).ok_or_else(|| String::from("invalid unicode code point"))
        } else if (0xdc00..=0xdfff).contains(&first) {
            Err(String::from("unexpected low surrogate"))
        } else {
            char::from_u32(u32::from(first))
                .ok_or_else(|| String::from("invalid unicode code point"))
        }
    }

    fn parse_hex_quad(&mut self) -> Result<u16, String> {
        let mut value = 0_u16;
        for _ in 0..4 {
            let character = self
                .next()
                .ok_or_else(|| String::from("truncated unicode escape"))?;
            value = value
                .checked_mul(16)
                .and_then(|value| {
                    character
                        .to_digit(16)
                        .and_then(|digit| value.checked_add(digit as u16))
                })
                .ok_or_else(|| String::from("invalid unicode escape"))?;
        }
        Ok(value)
    }

    fn parse_number(&mut self) -> Result<JsonValue, String> {
        let start = self.offset;
        if self.consume('-') {
            if self.peek() == Some('0') {
                self.offset += 1;
            } else if !self.consume_digits() {
                return Err(String::from("negative number requires digits"));
            }
        } else if self.peek() == Some('0') {
            self.offset += 1;
        } else {
            if !self.consume_digits() {
                return Err(String::from("number requires digits"));
            }
        }
        if self.peek() == Some('.') {
            self.offset += 1;
            if !self.consume_digits() {
                return Err(String::from("fraction requires digits"));
            }
        }
        if matches!(self.peek(), Some('e' | 'E')) {
            self.offset += 1;
            if matches!(self.peek(), Some('+' | '-')) {
                self.offset += 1;
            }
            if !self.consume_digits() {
                return Err(String::from("exponent requires digits"));
            }
        }
        let text = &self.text[start..self.offset];
        if text.contains('.') || text.contains('e') || text.contains('E') {
            let value = text
                .parse::<f64>()
                .map_err(|_| String::from("invalid number"))?;
            if !value.is_finite() {
                return Err(String::from("number must be finite"));
            }
            Ok(JsonValue::Number(JsonNumber::Float(value)))
        } else if text.starts_with('-') {
            text.parse::<i64>()
                .map(JsonNumber::Signed)
                .map(JsonValue::Number)
                .map_err(|_| String::from("signed number is out of range"))
        } else {
            text.parse::<u64>()
                .map(JsonNumber::Unsigned)
                .map(JsonValue::Number)
                .map_err(|_| String::from("unsigned number is out of range"))
        }
    }

    fn consume_digits(&mut self) -> bool {
        let start = self.offset;
        while matches!(self.peek(), Some('0'..='9')) {
            self.offset += 1;
        }
        self.offset != start
    }

    fn skip_whitespace(&mut self) {
        while matches!(self.peek(), Some(' ' | '\n' | '\r' | '\t')) {
            self.offset += 1;
        }
    }

    fn consume(&mut self, expected: char) -> bool {
        if self.peek() == Some(expected) {
            self.offset += expected.len_utf8();
            true
        } else {
            false
        }
    }

    fn peek(&self) -> Option<char> {
        self.text[self.offset..].chars().next()
    }

    fn next(&mut self) -> Option<char> {
        let character = self.peek()?;
        self.offset += character.len_utf8();
        Some(character)
    }
}

impl Revision {
    fn prev(self) -> Self {
        Self::from_raw(self.raw().saturating_sub(1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EndpointCapabilities, EntityRecord, InMemoryEndpoint, InMemoryStore};

    fn server() -> JsonRpcServer<InMemoryEndpoint> {
        let capabilities = EndpointCapabilities::from_capabilities([
            Capability::SnapshotRead,
            Capability::IncrementalRead,
            Capability::ComponentWrite,
            Capability::RevisionChecks,
        ]);
        let mut store = InMemoryStore::new("json", capabilities, 8).expect("store");
        let target = TargetId::from_raw(7);
        let mut snapshot = WorldSnapshot::empty(target);
        snapshot
            .entities
            .insert(EntityId::from_raw(1), EntityRecord::new());
        store.register_target(snapshot).expect("target");
        JsonRpcServer::new(crate::SyncEndpoint::new(store))
    }

    fn initialize() -> JsonRpcMessage {
        request(
            1,
            METHOD_INITIALIZE,
            Some(JsonValue::object([
                (
                    String::from("protocol"),
                    protocol_value(ProtocolVersion::CURRENT),
                ),
                (
                    String::from("capabilities"),
                    capabilities_value(EndpointCapabilities::from_capabilities([
                        Capability::SnapshotRead,
                        Capability::IncrementalRead,
                        Capability::ComponentWrite,
                        Capability::RevisionChecks,
                    ])),
                ),
            ])),
        )
    }

    #[test]
    fn json_rpc_round_trip_escapes_and_preserves_correlation() {
        let message = request(9, "method", Some(JsonValue::String(String::from("a\n\"b"))));
        let bytes = encode_json_rpc(&message);
        assert_eq!(decode_json_rpc(&bytes).expect("decode"), message);
        assert!(String::from_utf8(bytes).expect("utf8").contains("\\n"));
    }

    #[test]
    fn parser_rejects_duplicate_keys_and_limits() {
        assert!(matches!(
            decode_json_rpc(br#"{"jsonrpc":"2.0","id":1,"method":"x","method":"y"}"#),
            Err(JsonRpcCodecError::InvalidJson { .. })
        ));
        let limits = JsonRpcLimits {
            max_message_bytes: 1,
            ..JsonRpcLimits::default()
        };
        assert!(matches!(
            decode_json_rpc_with_limits(br#"{}"#, limits),
            Err(JsonRpcCodecError::LimitExceeded { .. })
        ));
    }

    #[test]
    fn initialize_negotiates_intersection() {
        let mut server = server();
        let response = server.handle_message(initialize()).expect("initialize");
        let JsonRpcMessage::Response(response) = &response[0] else {
            panic!("response")
        };
        assert!(response.error.is_none());
        assert!(server
            .negotiated()
            .expect("negotiated")
            .capabilities
            .contains(Capability::SnapshotRead));
    }

    #[test]
    fn endpoint_request_response_keeps_external_id() {
        let mut server = server();
        server.handle_message(initialize()).expect("initialize");
        let response = server
            .handle_message(request(44, METHOD_PING, None))
            .expect("ping");
        let JsonRpcMessage::Response(response) = &response[0] else {
            panic!("response")
        };
        assert_eq!(response.id, JsonRpcId::Number(44));
        assert!(response.result.is_some());
    }

    #[test]
    fn notification_has_no_response() {
        let mut server = server();
        server.handle_message(initialize()).expect("initialize");
        let notification = JsonRpcMessage::Notification(JsonRpcNotification {
            method: String::from(METHOD_PING),
            params: None,
        });
        assert!(server
            .handle_message(notification)
            .expect("notification")
            .is_empty());
    }

    #[test]
    fn subscription_emits_contiguous_incremental_notification() {
        let mut server = server();
        server.handle_message(initialize()).expect("initialize");
        let subscribe = request(
            2,
            METHOD_SUBSCRIBE,
            Some(JsonValue::object([
                (String::from("target"), target_value(TargetId::from_raw(7))),
                (String::from("fromRevision"), revision_value(Revision::ZERO)),
            ])),
        );
        let response = server.handle_message(subscribe).expect("subscribe");
        assert!(matches!(response[0], JsonRpcMessage::Response(_)));
        let changes = ChangeSet::new(
            Revision::ZERO,
            Revision::from_raw(1),
            vec![Change::CreateEntity {
                entity: EntityId::from_raw(2),
            }],
        )
        .expect("changes");
        let notifications = server
            .publish_changes(TargetId::from_raw(7), &changes)
            .expect("publish");
        assert!(matches!(notifications[0], JsonRpcMessage::Notification(_)));
        let bytes = encode_json_rpc(&notifications[0]);
        assert!(String::from_utf8(bytes)
            .expect("utf8")
            .contains(METHOD_CHANGED));
    }

    #[test]
    fn subscription_gap_requests_resync() {
        let mut server = server();
        server.handle_message(initialize()).expect("initialize");
        server
            .handle_message(request(
                2,
                METHOD_SUBSCRIBE,
                Some(JsonValue::object([
                    (String::from("target"), target_value(TargetId::from_raw(7))),
                    (String::from("fromRevision"), revision_value(Revision::ZERO)),
                ])),
            ))
            .expect("subscribe");
        let changes = ChangeSet::new(
            Revision::from_raw(2),
            Revision::from_raw(3),
            vec![Change::CreateEntity {
                entity: EntityId::from_raw(2),
            }],
        )
        .expect("changes");
        let notifications = server
            .publish_changes(TargetId::from_raw(7), &changes)
            .expect("publish");
        assert!(String::from_utf8(encode_json_rpc(&notifications[0]))
            .expect("utf8")
            .contains(METHOD_RESYNC));
    }

    #[test]
    fn tagged_float_and_bytes_round_trip_through_changes() {
        let value = Value::Object(BTreeMap::from([
            (String::from("float"), Value::Float(f64::NAN.to_bits())),
            (String::from("bytes"), Value::Bytes(vec![1, 2, 3])),
        ]));
        assert_eq!(json_to_value(&value_to_json(&value)).expect("value"), value);
    }

    #[test]
    fn batches_are_parsed_but_not_dispatched_as_single_call() {
        let batch = decode_json_rpc(br#"[{"jsonrpc":"2.0","id":1,"method":"x"}]"#).expect("batch");
        assert!(matches!(batch, JsonRpcMessage::Batch(_)));
        let mut server = server();
        assert!(server.handle_message(batch).is_err());
    }
}
