//! Real PNG I/O, filtering, resource limits and asynchronous publication.
extern crate alloc;
use alloc::sync::Arc;
use prism_editor_content::{
    thumbnail::{
        Thumbnail, ThumbnailError, ThumbnailKey, ThumbnailLimits, ThumbnailRequest,
        ThumbnailService,
    },
    AssetId,
};
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
    task::Poll,
    time::{Duration, Instant},
};

static NEXT: AtomicU64 = AtomicU64::new(0);
struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "prism-thumbnail-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn key(&self, revision: u128, width: u32, height: u32) -> ThumbnailKey {
        ThumbnailKey {
            asset: AssetId::from_raw(1),
            revision,
            source: self.0.join("image.png"),
            width,
            height,
        }
    }
    fn png(
        &self,
        width: u32,
        height: u32,
        color: png::ColorType,
        depth: png::BitDepth,
        data: &[u8],
    ) {
        let mut encoder = png::Encoder::new(
            fs::File::create(self.0.join("image.png")).unwrap(),
            width,
            height,
        );
        encoder.set_color(color);
        encoder.set_depth(depth);
        encoder
            .write_header()
            .unwrap()
            .write_image_data(data)
            .unwrap();
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
fn ready(request: &mut ThumbnailRequest) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !request.is_finished() {
        assert!(Instant::now() < deadline);
        std::thread::yield_now();
    }
}
fn finish(request: &mut ThumbnailRequest) -> Result<Arc<Thumbnail>, ThumbnailError> {
    ready(request);
    match request.poll() {
        Poll::Ready(result) => result,
        Poll::Pending => panic!("finished worker was pending"),
    }
}
fn service(limits: ThumbnailLimits) -> ThumbnailService {
    ThumbnailService::new(limits).unwrap()
}
fn load(fixture: &Fixture, width: u32, height: u32) -> Arc<Thumbnail> {
    finish(
        &mut service(ThumbnailLimits::default())
            .request(fixture.key(1, width, height))
            .unwrap(),
    )
    .unwrap()
}

#[test]
fn actual_png_normalizes_rgb_gray_alpha_palette_and_sixteen_bits() {
    let fixture = Fixture::new();
    for (color, depth, input, expected) in [
        (
            png::ColorType::Rgba,
            png::BitDepth::Eight,
            vec![30, 60, 90, 128],
            [30, 60, 90, 128],
        ),
        (
            png::ColorType::Rgb,
            png::BitDepth::Eight,
            vec![30, 60, 90],
            [30, 60, 90, 255],
        ),
        (
            png::ColorType::Grayscale,
            png::BitDepth::Eight,
            vec![80],
            [80, 80, 80, 255],
        ),
        (
            png::ColorType::GrayscaleAlpha,
            png::BitDepth::Eight,
            vec![80, 128],
            [80, 80, 80, 128],
        ),
        (
            png::ColorType::Rgb,
            png::BitDepth::Sixteen,
            vec![30, 255, 60, 255, 90, 255],
            [30, 60, 90, 255],
        ),
        (
            png::ColorType::Grayscale,
            png::BitDepth::One,
            vec![128],
            [255, 255, 255, 255],
        ),
    ] {
        fixture.png(1, 1, color, depth, &input);
        let thumbnail = load(&fixture, 8, 8);
        assert_eq!((thumbnail.width(), thumbnail.height()), (1, 1));
        assert_eq!(thumbnail.rgba8(), expected);
    }
    let mut encoder =
        png::Encoder::new(fs::File::create(fixture.0.join("image.png")).unwrap(), 1, 1);
    encoder.set_color(png::ColorType::Indexed);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.set_palette(vec![9, 30, 70]);
    encoder.set_trns(vec![128]);
    encoder
        .write_header()
        .unwrap()
        .write_image_data(&[0])
        .unwrap();
    assert_eq!(load(&fixture, 1, 1).rgba8(), [9, 30, 70, 128]);
}

#[test]
fn area_filter_is_linear_light_and_premultiplied_alpha() {
    let fixture = Fixture::new();
    fixture.png(
        2,
        1,
        png::ColorType::Rgba,
        png::BitDepth::Eight,
        &[0, 0, 0, 255, 255, 255, 255, 255],
    );
    assert_eq!(load(&fixture, 1, 1).rgba8(), [188, 188, 188, 255]);
    fixture.png(
        2,
        1,
        png::ColorType::Rgba,
        png::BitDepth::Eight,
        &[255, 0, 0, 255, 0, 0, 255, 0],
    );
    assert_eq!(load(&fixture, 1, 1).rgba8(), [255, 0, 0, 128]);
    fixture.png(
        2,
        1,
        png::ColorType::Rgba,
        png::BitDepth::Eight,
        &[255, 0, 0, 0, 0, 255, 255, 0],
    );
    assert_eq!(load(&fixture, 1, 1).rgba8(), [0, 0, 0, 0]);
}

#[test]
fn declared_gamma_is_converted_to_srgb_and_aspect_fit_never_upscales() {
    let fixture = Fixture::new();
    let mut encoder =
        png::Encoder::new(fs::File::create(fixture.0.join("image.png")).unwrap(), 4, 2);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.set_source_gamma(png::ScaledFloat::new(1.0));
    encoder
        .write_header()
        .unwrap()
        .write_image_data(&[128; 24])
        .unwrap();
    let thumbnail = load(&fixture, 3, 3);
    assert_eq!((thumbnail.width(), thumbnail.height()), (3, 2));
    assert_eq!(&thumbnail.rgba8()[..4], [188, 188, 188, 255]);
    let thumbnail = load(&fixture, 40, 40);
    assert_eq!((thumbnail.width(), thumbnail.height()), (4, 2));
}

#[test]
fn malformed_crc_and_truncated_png_never_enter_cache() {
    let fixture = Fixture::new();
    fixture.png(1, 1, png::ColorType::Rgb, png::BitDepth::Eight, &[1, 2, 3]);
    let valid = fs::read(fixture.0.join("image.png")).unwrap();
    let mut bad_crc = valid.clone();
    bad_crc[29] ^= 1;
    for bytes in [
        b"not a png".to_vec(),
        valid[..valid.len() / 2].to_vec(),
        valid[..valid.len() - 12].to_vec(),
        bad_crc,
    ] {
        fs::write(fixture.0.join("image.png"), bytes).unwrap();
        let service = service(ThumbnailLimits::default());
        assert!(matches!(
            finish(&mut service.request(fixture.key(1, 1, 1)).unwrap()),
            Err(ThumbnailError::Decode(_))
        ));
        assert_eq!(service.cache_usage().unwrap(), (0, 0));
        assert_eq!(service.in_flight(), 0);
    }
}

#[test]
fn encoded_source_decoder_and_output_allocations_are_bounded() {
    let fixture = Fixture::new();
    fixture.png(2, 2, png::ColorType::Rgb, png::BitDepth::Eight, &[1; 12]);
    for limits in [
        ThumbnailLimits {
            max_encoded_bytes: 8,
            ..ThumbnailLimits::default()
        },
        ThumbnailLimits {
            max_source_pixels: 3,
            ..ThumbnailLimits::default()
        },
        ThumbnailLimits {
            max_decoder_bytes: 1,
            ..ThumbnailLimits::default()
        },
    ] {
        let service = service(limits);
        assert!(finish(&mut service.request(fixture.key(1, 2, 2)).unwrap()).is_err());
        assert_eq!(service.cache_usage().unwrap(), (0, 0));
    }
    let service = service(ThumbnailLimits {
        max_output_bytes: 15,
        ..ThumbnailLimits::default()
    });
    assert!(matches!(
        service.request(fixture.key(1, 2, 2)),
        Err(ThumbnailError::Limit("output bytes"))
    ));
    assert_eq!(service.in_flight(), 0);
}

#[test]
fn invalid_configuration_and_paths_fail_before_reserving_capacity() {
    for limits in [
        ThumbnailLimits {
            max_in_flight: 0,
            ..ThumbnailLimits::default()
        },
        ThumbnailLimits {
            max_output_bytes: 0,
            ..ThumbnailLimits::default()
        },
        ThumbnailLimits {
            max_encoded_bytes: usize::MAX,
            ..ThumbnailLimits::default()
        },
        ThumbnailLimits {
            timeout: Duration::MAX,
            ..ThumbnailLimits::default()
        },
    ] {
        assert!(matches!(
            ThumbnailService::new(limits),
            Err(ThumbnailError::InvalidConfiguration)
        ));
    }
    let fixture = Fixture::new();
    let service = service(ThumbnailLimits::default());
    let mut relative = fixture.key(1, 1, 1);
    relative.source = "image.png".into();
    for key in [relative, fixture.key(1, 0, 1), fixture.key(1, 513, 1)] {
        assert!(matches!(
            service.request(key),
            Err(ThumbnailError::InvalidConfiguration)
        ));
    }
    let short = ThumbnailService::new(ThumbnailLimits {
        max_path_bytes: 1,
        ..ThumbnailLimits::default()
    })
    .unwrap();
    assert!(matches!(
        short.request(fixture.key(1, 1, 1)),
        Err(ThumbnailError::Limit("source path bytes"))
    ));
    assert_eq!(service.in_flight(), 0);
}

#[test]
fn ready_results_and_cached_hits_hold_slots_until_consumed_or_dropped() {
    let fixture = Fixture::new();
    fixture.png(1, 1, png::ColorType::Rgb, png::BitDepth::Eight, &[1, 2, 3]);
    let service = service(ThumbnailLimits {
        max_in_flight: 1,
        ..ThumbnailLimits::default()
    });
    let mut request = service.request(fixture.key(1, 1, 1)).unwrap();
    ready(&mut request);
    assert_eq!(service.in_flight(), 1);
    assert!(matches!(
        service.clone().request(fixture.key(1, 1, 1)),
        Err(ThumbnailError::Busy)
    ));
    let image = finish(&mut request).unwrap();
    assert_eq!(service.in_flight(), 0);
    assert!(matches!(
        request.poll(),
        Poll::Ready(Err(ThumbnailError::AlreadyConsumed))
    ));
    fs::remove_file(fixture.0.join("image.png")).unwrap();
    let mut cached = service.request(fixture.key(1, 1, 1)).unwrap();
    assert!(cached.is_finished());
    assert_eq!(service.in_flight(), 1);
    assert!(Arc::ptr_eq(&image, &finish(&mut cached).unwrap()));
    let cached = service.request(fixture.key(1, 1, 1)).unwrap();
    drop(cached);
    assert_eq!(service.in_flight(), 0);
}

#[test]
fn cancellation_and_project_clear_revoke_unconsumed_results_including_cache_hits() {
    let fixture = Fixture::new();
    fixture.png(1, 1, png::ColorType::Rgb, png::BitDepth::Eight, &[1, 2, 3]);
    let service = service(ThumbnailLimits::default());
    let mut cancelled = service.request(fixture.key(1, 1, 1)).unwrap();
    ready(&mut cancelled);
    cancelled.cancel();
    assert!(matches!(
        finish(&mut cancelled),
        Err(ThumbnailError::Cancelled)
    ));
    assert_eq!(service.cache_usage().unwrap(), (0, 0));
    let mut stale = service.request(fixture.key(1, 1, 1)).unwrap();
    ready(&mut stale);
    service.clear().unwrap();
    assert!(matches!(finish(&mut stale), Err(ThumbnailError::Stale)));
    finish(&mut service.request(fixture.key(1, 1, 1)).unwrap()).unwrap();
    let mut cached = service.request(fixture.key(1, 1, 1)).unwrap();
    service.clear().unwrap();
    assert!(matches!(finish(&mut cached), Err(ThumbnailError::Stale)));
    assert_eq!(service.cache_usage().unwrap(), (0, 0));
}

#[test]
fn revision_source_path_and_dimensions_are_all_part_of_cache_identity() {
    let fixture = Fixture::new();
    fixture.png(
        2,
        1,
        png::ColorType::Rgb,
        png::BitDepth::Eight,
        &[255, 0, 0, 255, 0, 0],
    );
    let service = service(ThumbnailLimits::default());
    let first = finish(&mut service.request(fixture.key(1, 2, 2)).unwrap()).unwrap();
    fixture.png(
        2,
        1,
        png::ColorType::Rgb,
        png::BitDepth::Eight,
        &[0, 255, 0, 0, 255, 0],
    );
    let second = finish(&mut service.request(fixture.key(2, 2, 2)).unwrap()).unwrap();
    assert_eq!(&first.rgba8()[..4], [255, 0, 0, 255]);
    assert_eq!(&second.rgba8()[..4], [0, 255, 0, 255]);
    let smaller = finish(&mut service.request(fixture.key(2, 1, 1)).unwrap()).unwrap();
    assert_eq!((smaller.width(), smaller.height()), (1, 1));
    fs::copy(fixture.0.join("image.png"), fixture.0.join("other.png")).unwrap();
    let mut key = fixture.key(2, 2, 2);
    key.source = fixture.0.join("other.png");
    let other = finish(&mut service.request(key).unwrap()).unwrap();
    assert!(!Arc::ptr_eq(&second, &other));
    assert_eq!(service.cache_usage().unwrap(), (4, 28));
}

#[test]
fn lru_entries_and_bytes_evict_without_invalidating_caller_pixels() {
    let fixture = Fixture::new();
    fixture.png(1, 1, png::ColorType::Rgb, png::BitDepth::Eight, &[1, 2, 3]);
    for limits in [
        ThumbnailLimits {
            max_cache_entries: 2,
            ..ThumbnailLimits::default()
        },
        ThumbnailLimits {
            max_cache_bytes: 8,
            ..ThumbnailLimits::default()
        },
    ] {
        let service = service(limits);
        let first = finish(&mut service.request(fixture.key(1, 1, 1)).unwrap()).unwrap();
        finish(&mut service.request(fixture.key(2, 1, 1)).unwrap()).unwrap();
        finish(&mut service.request(fixture.key(1, 1, 1)).unwrap()).unwrap();
        finish(&mut service.request(fixture.key(3, 1, 1)).unwrap()).unwrap();
        assert_eq!(service.cache_usage().unwrap(), (2, 8));
        assert_eq!(first.rgba8(), [1, 2, 3, 255]);
        fs::rename(fixture.0.join("image.png"), fixture.0.join("held.png")).unwrap();
        assert!(finish(&mut service.request(fixture.key(2, 1, 1)).unwrap()).is_err());
        assert!(Arc::ptr_eq(
            &first,
            &finish(&mut service.request(fixture.key(1, 1, 1)).unwrap()).unwrap()
        ));
        fs::rename(fixture.0.join("held.png"), fixture.0.join("image.png")).unwrap();
    }
    for limits in [
        ThumbnailLimits {
            max_cache_entries: 0,
            ..ThumbnailLimits::default()
        },
        ThumbnailLimits {
            max_cache_bytes: 3,
            ..ThumbnailLimits::default()
        },
    ] {
        let service = service(limits);
        finish(&mut service.request(fixture.key(1, 1, 1)).unwrap()).unwrap();
        assert_eq!(service.cache_usage().unwrap(), (0, 0));
    }
}

#[test]
fn zero_deadline_fails_without_publishing_or_leaking_a_slot() {
    let fixture = Fixture::new();
    let service = service(ThumbnailLimits {
        timeout: Duration::ZERO,
        ..ThumbnailLimits::default()
    });
    assert!(matches!(
        finish(&mut service.request(fixture.key(1, 1, 1)).unwrap()),
        Err(ThumbnailError::TimedOut)
    ));
    assert_eq!(service.cache_usage().unwrap(), (0, 0));
    assert_eq!(service.in_flight(), 0);
}

#[test]
fn animated_and_explicit_alternate_primaries_fail_without_wrong_color_output() {
    let fixture = Fixture::new();
    for animated in [true, false] {
        let mut encoder =
            png::Encoder::new(fs::File::create(fixture.0.join("image.png")).unwrap(), 1, 1);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        if animated {
            encoder.set_animated(1, 0).unwrap();
        } else {
            encoder.set_source_chromaticities(png::SourceChromaticities::new(
                (0.3127, 0.329),
                (0.64, 0.33),
                (0.30, 0.60),
                (0.15, 0.06),
            ));
        }
        encoder
            .write_header()
            .unwrap()
            .write_image_data(&[1, 2, 3])
            .unwrap();
        let service = service(ThumbnailLimits::default());
        assert!(matches!(
            finish(&mut service.request(fixture.key(1, 1, 1)).unwrap()),
            Err(ThumbnailError::UnsupportedFormat)
        ));
        assert_eq!(service.cache_usage().unwrap(), (0, 0));
    }
}

#[test]
fn dropped_workers_release_capacity_after_cleanup_and_never_cache() {
    let fixture = Fixture::new();
    fixture.png(
        512,
        512,
        png::ColorType::Rgb,
        png::BitDepth::Eight,
        &vec![128; 512 * 512 * 3],
    );
    let service = service(ThumbnailLimits {
        max_in_flight: 1,
        ..ThumbnailLimits::default()
    });
    let request = service.request(fixture.key(1, 1, 1)).unwrap();
    drop(request);
    let deadline = Instant::now() + Duration::from_secs(10);
    while service.in_flight() != 0 {
        assert!(Instant::now() < deadline);
        std::thread::yield_now();
    }
    assert_eq!(service.cache_usage().unwrap(), (0, 0));
    finish(&mut service.request(fixture.key(2, 1, 1)).unwrap()).unwrap();
}

#[cfg(unix)]
#[test]
fn directories_and_leaf_symlinks_are_rejected_as_sources() {
    let fixture = Fixture::new();
    fs::create_dir(fixture.0.join("image.png")).unwrap();
    let service = service(ThumbnailLimits::default());
    assert!(matches!(
        finish(&mut service.request(fixture.key(1, 1, 1)).unwrap()),
        Err(ThumbnailError::NotRegularFile)
    ));
    fs::remove_dir(fixture.0.join("image.png")).unwrap();
    fs::write(fixture.0.join("target.png"), [1, 2, 3]).unwrap();
    std::os::unix::fs::symlink(fixture.0.join("target.png"), fixture.0.join("image.png")).unwrap();
    assert!(matches!(
        finish(&mut service.request(fixture.key(1, 1, 1)).unwrap()),
        Err(ThumbnailError::NotRegularFile)
    ));
}
