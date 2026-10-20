//! Opt-in physical output-device lifecycle smoke test.
//!
//! Default CI does not claim hardware coverage. Set
//! `PRISM_AUDIO_REQUIRE_DEVICE=1` to require a real default endpoint and verify
//! that CPAL can open it paused, start callback delivery, and stop it again.

#[cfg(feature = "cpal-backend")]
#[test]
fn physical_default_output_starts_and_stops() {
    use prism_audio_device::{open_default_output_paused, AudioDeviceBackend, DeviceState};
    use prism_audio_rt::{runtime, AudioRuntimeConfig};

    if std::env::var_os("PRISM_AUDIO_REQUIRE_DEVICE").as_deref() != Some(std::ffi::OsStr::new("1"))
    {
        eprintln!("physical audio test skipped; set PRISM_AUDIO_REQUIRE_DEVICE=1 to require it");
        return;
    }

    let (runtime, _client, _collector) = runtime(AudioRuntimeConfig::default());
    let mut output = open_default_output_paused(runtime, 256)
        .expect("a physical default output endpoint is required");
    assert_eq!(output.state(), DeviceState::Stopped);
    output.start().expect("physical stream must start");
    assert_eq!(output.state(), DeviceState::Running);
    output.stop().expect("physical stream must stop");
    assert_eq!(output.state(), DeviceState::Stopped);
}
