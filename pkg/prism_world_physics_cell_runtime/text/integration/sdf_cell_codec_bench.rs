//! Complete cell encode/decode costs; construction and destruction are not timed.

use core::hint::black_box;
use prism_world_physics_bridge::{WorldSdfSnapshot, WorldSdfTransform};
use prism_world_physics_cell_runtime::{SdfCellCodec, SdfCellCodecLimits, SdfCellSnapshot};
use prism_world_residency::{BrickCoord, SparseSdfBrickGrid};
use prism_world_stream::{AabbI64, CellManifest, DataLayerMask, GlobalPos, SectorId};
use std::{
    io::{self, Write},
    time::Instant,
};

fn measure(bricks: usize) {
    let sector = SectorId::new(0, 0, 0, 0);
    let mut grid = SparseSdfBrickGrid::new(8, 1).expect("grid");
    for i in 0..bricks {
        let samples = (0..512)
            .map(|j| ((i * 31 + j * 17) as i16).wrapping_sub(12000))
            .collect();
        grid.insert_brick(BrickCoord::new(i as i32 - 512, 0, 0, 0), samples)
            .expect("brick");
    }
    let source = SdfCellSnapshot::new(
        sector,
        WorldSdfSnapshot::new(
            WorldSdfTransform::new([0; 3], 100, 10).expect("transform"),
            1,
            grid,
        ),
    );
    let codec = SdfCellCodec::new(SdfCellCodecLimits::new(2 << 20, 1024, 524288, 2 << 20));
    let manifest = CellManifest {
        sector,
        bounds: AabbI64::new(GlobalPos::new(0, 0, 0), GlobalPos::new(1, 1, 1)).expect("bounds"),
        scene_key: 1,
        content_version: 1,
        derived_hash: 1,
        io_bytes: 0,
        data_layers: DataLayerMask::EMPTY,
    };
    let bytes = codec.encode(&source).expect("encode");
    assert_eq!(
        codec.decode_payload(&manifest, &bytes).expect("decode"),
        source
    );
    for _ in 0..8 {
        black_box(codec.encode(&source).expect("warm encode"));
        black_box(
            codec
                .decode_payload(&manifest, &bytes)
                .expect("warm decode"),
        );
    }
    let mut encode_times = Vec::with_capacity(128);
    let mut decode_times = Vec::with_capacity(128);
    for _ in 0..128 {
        let start = Instant::now();
        let encoded = codec.encode(black_box(&source)).expect("encode");
        encode_times.push(start.elapsed().as_nanos());
        assert_eq!(encoded, bytes);
        let start = Instant::now();
        let decoded = codec
            .decode_payload(black_box(&manifest), black_box(&bytes))
            .expect("decode");
        decode_times.push(start.elapsed().as_nanos());
        assert_eq!(decoded, source);
        black_box((encoded, decoded));
    }
    encode_times.sort_unstable();
    decode_times.sort_unstable();
    writeln!(io::stdout().lock(),"bricks={bricks} edge=8 samples={} frame_bytes={} logical_bytes={} trials=128 warmups=8 encode P50/P95/P99 ns={}/{}/{} decode P50/P95/P99 ns={}/{}/{}",bricks*512,bytes.len(),48+bricks*16+bricks*1024,encode_times[63],encode_times[121],encode_times[126],decode_times[63],decode_times[121],decode_times[126]).expect("benchmark report");
}

fn main() {
    measure(64);
    measure(1024);
}
