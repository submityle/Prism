//! Cross-thread integration tests that exercise the lock-free bridge with real
//! OS threads: a task thread submitting commands and graphs, the audio thread
//! rendering blocks, and a collector reclaiming retired graphs.

extern crate alloc;

use alloc::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
use prism_audio_core::graph::{AudioGraph, AudioNode, PortRef, ProcessIo, RenderContext};
use prism_audio_core::voice::{VoiceGroup, VoiceRequest};
use prism_audio_rt::{AudioCommand, AudioRuntimeConfig, runtime};

/// A node that writes a constant `1.0` to its single stereo output.
struct Dc;

impl AudioNode for Dc {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let out = io.output(0);
        for channel in 0..out.channels() {
            for sample in out.channel_mut(channel) {
                *sample = 1.0;
            }
        }
    }
}

/// Builds a compiled graph whose stereo master output is a constant `1.0`.
fn build_dc_graph() -> Box<AudioGraph> {
    let mut graph = AudioGraph::new(48_000, 512);
    let dc = graph.add_node(Box::new(Dc), vec![], vec![ChannelLayout::Stereo]);
    graph
        .set_master(PortRef::new(dc, 0))
        .expect("master port is valid");
    graph.compile().expect("graph compiles");
    Box::new(graph)
}

#[test]
fn commands_graphs_and_telemetry_cross_threads() {
    let (mut rt, client, collector) = runtime(AudioRuntimeConfig::default());
    let stop = Arc::new(AtomicBool::new(false));
    let stop_audio = Arc::clone(&stop);

    let audio = thread::spawn(move || {
        let mut out = AudioBuffer::new(ChannelLayout::Stereo, 256);
        let mut blocks = 0u64;
        while !stop_audio.load(Ordering::Relaxed) {
            rt.process_block(&mut out);
            blocks += 1;
            thread::sleep(Duration::from_micros(200));
        }
        blocks
    });

    // Task thread: install a signal graph and spawn an audible voice.
    assert!(client.publish_graph(build_dc_graph()).is_none());
    client
        .send(AudioCommand::SpawnVoice {
            request: VoiceRequest::new(VoiceGroup(1), 0.9),
        })
        .expect("command ring has room");
    thread::sleep(Duration::from_millis(50));

    // Publish a second graph, forcing the first to be retired to the collector.
    client.publish_graph(build_dc_graph());
    thread::sleep(Duration::from_millis(50));

    // The collector must reclaim the retired first graph off the audio thread.
    let mut reclaimed = 0;
    for _ in 0..200 {
        reclaimed += collector.collect();
        if reclaimed >= 1 {
            break;
        }
        thread::sleep(Duration::from_millis(1));
    }

    // Drain telemetry accumulated by the audio thread.
    let mut saw_voice = false;
    let mut saw_signal = false;
    while let Some(frame) = client.recv_telemetry() {
        if frame.physical_voices >= 1 {
            saw_voice = true;
        }
        if frame.master_peak > 0.0 {
            saw_signal = true;
        }
    }

    stop.store(true, Ordering::Relaxed);
    let blocks = audio.join().expect("audio thread joins");

    assert!(blocks > 0, "audio thread rendered at least one block");
    assert!(saw_voice, "telemetry should report the spawned voice");
    assert!(saw_signal, "the DC graph should produce non-silent master output");
    assert!(reclaimed >= 1, "collector should reclaim the retired graph");
}
