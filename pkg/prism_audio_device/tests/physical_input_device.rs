//! Opt-in physical input-device lifecycle smoke test.
//!
//! Default CI does not claim microphone coverage. Set
//! `PRISM_AUDIO_REQUIRE_DEVICE=1` to require a real input endpoint and verify
//! enumeration, opening, callback frame arrival, start, and stop. A missing endpoint or unsupported
//! host configuration is a test failure with the concrete CPAL error.

#[cfg(feature = "cpal-backend")]
#[test]
fn physical_default_input_enumerates_starts_and_stops() {
    use prism_audio_device::{
        capture_ring, open_default_input_paused, CaptureEvent, CpalInputCatalog, DeviceError,
        DeviceState,
    };

    if std::env::var_os("PRISM_AUDIO_REQUIRE_DEVICE").as_deref() != Some(std::ffi::OsStr::new("1"))
    {
        eprintln!("physical input test skipped; set PRISM_AUDIO_REQUIRE_DEVICE=1 to require it");
        return;
    }

    let mut catalog = CpalInputCatalog::new();
    let devices = catalog
        .devices()
        .unwrap_or_else(|error| panic!("input enumeration is required: {error}"));
    eprintln!("enumerated input devices: {devices:#?}");
    assert!(
        !devices.is_empty(),
        "strict hardware evidence: CPAL enumerated no input devices"
    );
    assert!(catalog
        .poll_changes()
        .expect("initial catalog poll")
        .is_none());

    let mut last_mismatch = None;
    for channels in 1..=16 {
        let (sink, consumer) = capture_ring(channels, 4096);
        match open_default_input_paused(sink) {
            Ok(mut input) => {
                eprintln!("opened input stream: {:?}", input.info());
                assert_eq!(input.state(), DeviceState::Stopped);
                input.start().expect("physical input stream must start");
                assert_eq!(input.state(), DeviceState::Running);
                assert_eq!(input.poll_event(), Some(CaptureEvent::Started));
                assert!(matches!(
                    input.start(),
                    Err(DeviceError::InvalidState { .. })
                ));
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
                while consumer.buffered_frames() == 0 && std::time::Instant::now() < deadline {
                    assert_eq!(
                        input.state(),
                        DeviceState::Running,
                        "physical stream failed while awaiting callback frames: {input:?}"
                    );
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
                assert!(
                    consumer.buffered_frames() > 0,
                    "strict hardware evidence: start succeeded but no callback frames arrived within 3s; input={input:?}"
                );
                eprintln!(
                    "physical callback delivered {} frames",
                    consumer.buffered_frames()
                );
                while let Some(event) = input.poll_event() {
                    assert!(matches!(event, CaptureEvent::Overflow { .. }), "{event:?}");
                }
                input.stop().expect("physical input stream must stop");
                assert_eq!(input.state(), DeviceState::Stopped);
                assert_eq!(input.poll_event(), Some(CaptureEvent::Stopped));
                assert!(matches!(
                    input.stop(),
                    Err(DeviceError::InvalidState { .. })
                ));
                return;
            }
            Err(DeviceError::CaptureChannelCountMismatch { .. }) => {
                last_mismatch = Some(channels);
            }
            Err(error) => panic!("physical input open failed after enumeration: {error}"),
        }
    }

    panic!(
        "strict hardware evidence: no supported default input channel count in 1..=16; last mismatch={last_mismatch:?}"
    );
}
