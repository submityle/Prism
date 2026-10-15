//! Real archive-to-host latency; bounded memory, no GPU or RSS claims.
use prism_world_io::{checksum, ThreadedFileConfig};
use prism_world_runtime::{
    CellFrameCoordinator, CellReadRequest, IoByteLimits, ManifestHeader, SceneHost, ThreadedCellIo,
    CELL_HEADER_SIZE,
};
use prism_world_stream::{AabbI64, CellManifest, DemandSource, GlobalPos, SectorId, StreamBudget};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::{mpsc, Arc};
use std::task::{Context, Poll, Wake, Waker};
use std::time::{Duration, Instant};

const CELLS: usize = 32;
const CELL_BYTES: usize = 64 * 1024;
const WARMUP: usize = 8;
const SAMPLES: usize = 64;

struct Archive(PathBuf);
impl Drop for Archive {
    fn drop(&mut self) {
        fs::remove_file(&self.0).expect("remove benchmark archive");
    }
}
struct Signal(mpsc::SyncSender<()>);
impl Wake for Signal {
    fn wake(self: Arc<Self>) {
        let _ = self.0.try_send(());
    }
    fn wake_by_ref(self: &Arc<Self>) {
        let _ = self.0.try_send(());
    }
}
#[derive(Default)]
struct Host {
    loaded: usize,
    bytes: usize,
}
impl SceneHost for Host {
    type Error = ();
    fn load_cell(&mut self, m: &CellManifest, payload: &[u8]) -> Result<(), ()> {
        assert_eq!(payload.len(), CELL_BYTES - CELL_HEADER_SIZE);
        assert!(payload.iter().all(|&byte| byte == m.sector.x as u8));
        self.loaded += 1;
        self.bytes += payload.len();
        Ok(())
    }
    fn evict_cell(&mut self, _: SectorId) -> Result<(), ()> {
        Ok(())
    }
}

fn percentile(values: &mut [u128], percent: usize) -> u128 {
    values.sort_unstable();
    values[(values.len() * percent).div_ceil(100) - 1]
}

fn main() {
    let path = std::env::temp_dir().join(format!(
        "prism-world-archive-bench-{}.bin",
        std::process::id()
    ));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .expect("unique archive");
    let archive = Archive(path);
    let mut sources = Vec::with_capacity(CELLS);
    for x in 0..CELLS {
        let m = CellManifest {
            sector: SectorId::new(0, x as i64, 0, 0),
            bounds: AabbI64::new(
                GlobalPos::new(x as i64, 0, 0),
                GlobalPos::new(x as i64 + 1, 1, 1),
            )
            .unwrap(),
            scene_key: x as u128,
            content_version: 1,
            derived_hash: 42,
            io_bytes: CELL_BYTES as u64,
        };
        let mut bytes = ManifestHeader::from_manifest(m).encode();
        bytes.resize(CELL_BYTES, x as u8);
        file.write_all(&bytes).unwrap();
        sources.push(
            CellReadRequest::new(m, 0)
                .with_byte_offset((x * CELL_BYTES) as u64)
                .with_expected_checksum(checksum(&bytes)),
        );
    }
    drop(file);
    let manifests: Vec<_> = sources.iter().map(|s| s.manifest()).collect();
    let demands: Vec<_> = manifests
        .iter()
        .map(|m| DemandSource::new(m.sector, 0, 1, 10).unwrap())
        .collect();
    let mut elapsed = Vec::with_capacity(SAMPLES);
    let mut poll_ns = Vec::new();
    for sample in 0..WARMUP + SAMPLES {
        let mut io = ThreadedCellIo::threaded(
            &archive.0,
            ThreadedFileConfig {
                workers: 2,
                capacity: 4,
                max_chunk_size: 16 * 1024,
            },
            4,
            CELLS,
            IoByteLimits {
                max_request_bytes: CELL_BYTES as u64,
                max_total_bytes: (CELLS * CELL_BYTES) as u64,
            },
        )
        .unwrap();
        for source in &sources {
            io.register(*source).unwrap();
        }
        let mut coordinator = CellFrameCoordinator::with_io(io);
        let mut host = Host::default();
        let (sender, receiver) = mpsc::sync_channel(1);
        let waker = Waker::from(Arc::new(Signal(sender)));
        let mut cx = Context::from_waker(&waker);
        let start = Instant::now();
        let report = coordinator.plan_frame(
            &demands,
            &manifests,
            &[],
            StreamBudget {
                io_bytes: (CELLS * CELL_BYTES) as u64,
                max_actions: CELLS,
            },
            &mut host,
        );
        assert_eq!(report.submitted, CELLS);
        while coordinator.in_flight() != 0 {
            let poll_start = Instant::now();
            let result = coordinator.poll_io_with_context(&mut cx, &mut host);
            if sample >= WARMUP {
                poll_ns.push(poll_start.elapsed().as_nanos());
            }
            match result {
                Poll::Ready(result) => result.unwrap(),
                Poll::Pending => receiver
                    .recv_timeout(Duration::from_secs(5))
                    .expect("IO wake"),
            }
        }
        if sample >= WARMUP {
            elapsed.push(start.elapsed().as_nanos());
        }
        assert_eq!(host.loaded, CELLS);
        assert_eq!(host.bytes, CELLS * (CELL_BYTES - CELL_HEADER_SIZE));
    }
    let p50 = percentile(&mut elapsed, 50);
    let p95 = percentile(&mut elapsed, 95);
    let p99 = percentile(&mut elapsed, 99);
    println!("archive bytes={} cells={} workers=2 active=4 chunk=16384 warmup={} samples={} cached_file=true", CELLS * CELL_BYTES, CELLS, WARMUP, SAMPLES);
    println!(
        "end_to_end_ns p50={p50} p95={p95} p99={p99}; median_MiB_per_s={:.2}",
        2.0e9 / p50 as f64
    );
    println!(
        "poll_including_validation_and_host_ns p50={} p95={} p99={} polls={}",
        percentile(&mut poll_ns, 50),
        percentile(&mut poll_ns, 95),
        percentile(&mut poll_ns, 99),
        poll_ns.len()
    );
}
