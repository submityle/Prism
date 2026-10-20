//! Minimal child executable used by runtime bridge integration tests.

use prism_editor_remote::{
    EndpointCapabilities, InMemoryStore, JsonRpcServer, JsonRpcServerSession, SyncEndpoint,
    TransportLimits,
};
use std::env;
use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpListener};
use std::process;
use std::thread;
use std::time::Duration;

const MODE: &str = "PRISM_EDITOR_REMOTE_CHILD_MODE";
const ADDRESS: &str = "PRISM_EDITOR_REMOTE_CHILD_ADDRESS";

fn main() {
    let mode = env::var(MODE).unwrap_or_else(|_| String::from("serve"));
    match mode.as_str() {
        "exit" => process::exit(23),
        "block" => thread::sleep(Duration::from_secs(300)),
        "tcp" => {
            let address: SocketAddr = env::var(ADDRESS)
                .expect("PRISM_EDITOR_REMOTE_CHILD_ADDRESS")
                .parse()
                .expect("valid child TCP address");
            let listener = TcpListener::bind(address).expect("child TCP bind");
            let (stream, _) = listener.accept().expect("child TCP accept");
            serve(stream);
        }
        "block-after-discover" => {
            serve_one_then_block(StdioStream::new(io::stdin(), io::stdout()));
        }
        "exit-after-discover" => {
            serve_one(StdioStream::new(io::stdin(), io::stdout()));
            process::exit(23);
        }
        _ => serve(StdioStream::new(io::stdin(), io::stdout())),
    }
}

struct StdioStream<R, W> {
    reader: R,
    writer: W,
}

impl<R, W> StdioStream<R, W> {
    fn new(reader: R, writer: W) -> Self {
        Self { reader, writer }
    }
}

impl<R: Read, W> Read for StdioStream<R, W> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        self.reader.read(buffer)
    }
}

impl<R, W: Write> Write for StdioStream<R, W> {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        self.writer.write(buffer)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()
    }
}

fn serve<S: Read + Write>(stream: S) {
    let mut session = session(stream);
    session.serve().expect("runtime session");
}

fn serve_one<S: Read + Write>(stream: S) {
    let mut session = session(stream);
    session.serve_one().expect("runtime startup request");
}

fn serve_one_then_block<S: Read + Write>(stream: S) {
    let mut session = session(stream);
    session.serve_one().expect("runtime startup request");
    thread::sleep(Duration::from_secs(300));
}

fn session<S: Read + Write>(stream: S) -> JsonRpcServerSession<SyncEndpoint<InMemoryStore>, S> {
    let store = InMemoryStore::new("loopback-runtime", EndpointCapabilities::empty(), 1)
        .expect("runtime store");
    JsonRpcServerSession::new(
        JsonRpcServer::new(SyncEndpoint::new(store)),
        stream,
        TransportLimits::default(),
    )
}
