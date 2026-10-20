//! Canonical little-endian v1 command stream. Lengths are checked before copy.
use super::*;

type Result<T> = core::result::Result<T, GraphAuthoringError>;
const HEADER: &[u8] = b"PGRF\x01\x00";
struct Writer {
    bytes: Vec<u8>,
    limits: GraphAuthoringLimits,
}
impl Writer {
    fn put(&mut self, bytes: &[u8]) -> Result<()> {
        if bytes.len() > self.limits.command_bytes.saturating_sub(self.bytes.len()) {
            return Err(GraphAuthoringError::Limit);
        }
        self.bytes.extend_from_slice(bytes);
        Ok(())
    }
    fn tag(&mut self, n: u8) -> Result<()> {
        self.put(&[n])
    }
    fn number(&mut self, n: u64) -> Result<()> {
        id(n, 1)?;
        self.put(&n.to_le_bytes())
    }
    fn count(&mut self, n: usize, max: usize) -> Result<()> {
        if n > max {
            return Err(GraphAuthoringError::Limit);
        }
        self.put(&(n as u32).to_le_bytes())
    }
    fn blob(&mut self, b: &[u8], max: usize) -> Result<()> {
        self.count(b.len(), max)?;
        self.put(b)
    }
    fn text(&mut self, s: &str) -> Result<()> {
        if s.is_empty() {
            return Err(GraphAuthoringError::Codec);
        }
        self.blob(s.as_bytes(), self.limits.text_bytes)
    }
    fn point(&mut self, p: GraphPoint) -> Result<()> {
        if p.x.unsigned_abs() > 1000000 || p.y.unsigned_abs() > 1000000 {
            return Err(GraphAuthoringError::Limit);
        }
        self.put(&p.x.to_le_bytes())?;
        self.put(&p.y.to_le_bytes())
    }
}
impl GraphCommand {
    /// Encode canonical v1 bytes, bounded before each append. Port/edit order is
    /// significant; no map iteration or floating-point representation is stored.
    pub fn encode(&self, limits: GraphAuthoringLimits) -> Result<Vec<u8>> {
        limits.validate()?;
        let mut w = Writer {
            bytes: Vec::new(),
            limits,
        };
        w.put(HEADER)?;
        match self {
            Self::Undo => w.tag(1)?,
            Self::Redo => w.tag(2)?,
            Self::Edit(edits) => {
                if edits.is_empty() {
                    return Err(GraphAuthoringError::Codec);
                }
                w.tag(0)?;
                w.count(edits.len(), limits.edits)?;
                for e in edits {
                    match e {
                        GraphEdit::CreateNode(n) => {
                            w.tag(0)?;
                            w.number(n.id.0)?;
                            w.text(&n.name)?;
                            w.blob(&n.payload, limits.payload_bytes)?;
                            w.point(n.position)?;
                            w.count(n.ports.len(), limits.ports_per_node)?;
                            let mut next = 1;
                            for p in &n.ports {
                                id(p.id.0, next)?;
                                next = p.id.0 + 1;
                                w.number(p.id.0)?;
                                w.text(&p.spec.name)?;
                                w.text(p.spec.port_type.key())?;
                                w.tag(match p.spec.direction {
                                    PortDirection::Input => 0,
                                    PortDirection::Output => 1,
                                })?;
                                w.tag(match p.spec.connection_policy {
                                    ConnectionPolicy::Single => 0,
                                    ConnectionPolicy::Multiple => 1,
                                })?;
                            }
                        }
                        GraphEdit::DeleteNode(n) => {
                            w.tag(1)?;
                            w.number(n.0)?;
                        }
                        GraphEdit::MoveNode { node, position } => {
                            w.tag(2)?;
                            w.number(node.0)?;
                            w.point(*position)?;
                        }
                        GraphEdit::SetPayload { node, payload } => {
                            w.tag(3)?;
                            w.number(node.0)?;
                            w.blob(payload, limits.payload_bytes)?;
                        }
                        GraphEdit::Connect {
                            edge,
                            output,
                            input,
                        } => {
                            w.tag(4)?;
                            w.number(edge.0)?;
                            w.number(output.0)?;
                            w.number(input.0)?;
                        }
                        GraphEdit::Disconnect(e) => {
                            w.tag(5)?;
                            w.number(e.0)?;
                        }
                    }
                }
            }
        }
        Ok(w.bytes)
    }
    /// Decode exactly one command; reject unknown versions/tags, invalid UTF-8,
    /// excessive counts/lengths, truncation and trailing data before execution.
    pub fn decode(bytes: &[u8], limits: GraphAuthoringLimits) -> Result<Self> {
        limits.validate()?;
        if bytes.len() > limits.command_bytes {
            return Err(GraphAuthoringError::Limit);
        }
        let mut r = Reader { bytes, offset: 0 };
        if r.take(HEADER.len())? != HEADER {
            return Err(GraphAuthoringError::Codec);
        }
        let command = match r.tag()? {
            1 => Self::Undo,
            2 => Self::Redo,
            0 => {
                let count = r.count(limits.edits)?;
                if count == 0 || count > r.remaining() {
                    return Err(GraphAuthoringError::Codec);
                }
                let mut edits = Vec::new();
                for _ in 0..count {
                    edits.push(match r.tag()? {
                        0 => {
                            let id = NodeId(r.number()?);
                            let name = r.text(limits.text_bytes)?;
                            let payload = r.blob(limits.payload_bytes)?.to_vec();
                            let position = r.point()?;
                            let count = r.count(limits.ports_per_node)?;
                            if count > r.remaining() / 18 {
                                return Err(GraphAuthoringError::Codec);
                            }
                            let mut ports = Vec::new();
                            for _ in 0..count {
                                let id = PortId(r.number()?);
                                let name = r.text(limits.text_bytes)?;
                                let port_type = PortType::named(r.text(limits.text_bytes)?);
                                let direction = match r.tag()? {
                                    0 => PortDirection::Input,
                                    1 => PortDirection::Output,
                                    _ => return Err(GraphAuthoringError::Codec),
                                };
                                let connection_policy = match r.tag()? {
                                    0 => ConnectionPolicy::Single,
                                    1 => ConnectionPolicy::Multiple,
                                    _ => return Err(GraphAuthoringError::Codec),
                                };
                                ports.push(GraphPortDraft {
                                    id,
                                    spec: PortSpec {
                                        name,
                                        port_type,
                                        direction,
                                        connection_policy,
                                    },
                                });
                            }
                            GraphEdit::CreateNode(GraphNodeDraft {
                                id,
                                name,
                                payload,
                                ports,
                                position,
                            })
                        }
                        1 => GraphEdit::DeleteNode(NodeId(r.number()?)),
                        2 => GraphEdit::MoveNode {
                            node: NodeId(r.number()?),
                            position: r.point()?,
                        },
                        3 => GraphEdit::SetPayload {
                            node: NodeId(r.number()?),
                            payload: r.blob(limits.payload_bytes)?.to_vec(),
                        },
                        4 => GraphEdit::Connect {
                            edge: EdgeId(r.number()?),
                            output: PortId(r.number()?),
                            input: PortId(r.number()?),
                        },
                        5 => GraphEdit::Disconnect(EdgeId(r.number()?)),
                        _ => return Err(GraphAuthoringError::Codec),
                    });
                }
                Self::Edit(edits)
            }
            _ => return Err(GraphAuthoringError::Codec),
        };
        if r.remaining() != 0 {
            return Err(GraphAuthoringError::Codec);
        }
        // Share all canonical field validation with the encoder, including IDs.
        command.encode(limits)?;
        Ok(command)
    }
}
struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}
impl<'a> Reader<'a> {
    fn remaining(&self) -> usize {
        self.bytes.len() - self.offset
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if n > self.remaining() {
            return Err(GraphAuthoringError::Codec);
        }
        let bytes = &self.bytes[self.offset..self.offset + n];
        self.offset += n;
        Ok(bytes)
    }
    fn tag(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn number(&mut self) -> Result<u64> {
        let n = u64::from_le_bytes(self.take(8)?.try_into().expect("eight bytes"));
        id(n, 1)?;
        Ok(n)
    }
    fn count(&mut self, max: usize) -> Result<usize> {
        let n = u32::from_le_bytes(self.take(4)?.try_into().expect("four bytes")) as usize;
        if n > max {
            Err(GraphAuthoringError::Limit)
        } else {
            Ok(n)
        }
    }
    fn blob(&mut self, max: usize) -> Result<&'a [u8]> {
        let n = self.count(max)?;
        self.take(n)
    }
    fn text(&mut self, max: usize) -> Result<String> {
        let s = core::str::from_utf8(self.blob(max)?).map_err(|_| GraphAuthoringError::Codec)?;
        if s.is_empty() {
            return Err(GraphAuthoringError::Codec);
        }
        Ok(s.to_owned())
    }
    fn point(&mut self) -> Result<GraphPoint> {
        let x = i32::from_le_bytes(self.take(4)?.try_into().expect("four bytes"));
        let y = i32::from_le_bytes(self.take(4)?.try_into().expect("four bytes"));
        if x.unsigned_abs() > 1000000 || y.unsigned_abs() > 1000000 {
            return Err(GraphAuthoringError::Limit);
        }
        Ok(GraphPoint { x, y })
    }
}
