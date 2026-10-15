//! Bounded, allocation-free network telemetry for replication connections.
//!
//! The counters are numeric and channel-indexed. A caller can sample
//! [`MuxTelemetrySnapshot`] into its own diagnostics backend without making
//! the replication hot path depend on strings, ECS, or a metrics framework.

use crate::multiplex::NetChannel;

/// Number of logical channels in the replication multiplex.
pub const NET_CHANNEL_COUNT: usize = 5;

/// Cumulative counters for one logical network channel.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ChannelTelemetry {
    /// Messages admitted by the application.
    pub messages_accepted: u64,
    /// Physical packets emitted for this channel, including ACKs and retries.
    pub packets_sent: u64,
    /// Bytes emitted for this channel, including protocol headers.
    pub bytes_sent: u64,
    /// First transmissions delivered to the datagram transport.
    pub messages_sent: u64,
    /// Physical packets received for this channel, including ACKs.
    pub packets_received: u64,
    /// Bytes received for this channel, including protocol headers.
    pub bytes_received: u64,
    /// Messages released to the application.
    pub messages_received: u64,
    /// Messages discarded, replaced, or refused after admission.
    pub dropped_messages: u64,
    /// Reliable sends refused by the bounded queue.
    pub backpressure_events: u64,
    /// Data packets emitted as retransmissions.
    pub retransmissions: u64,
}

/// A copyable point-in-time view of one multiplexed connection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MuxTelemetrySnapshot {
    /// Monotonic logical tick at which the snapshot was taken.
    pub tick: u64,
    /// Physical datagrams emitted, including ACKs.
    pub datagrams_sent: u64,
    /// Physical datagrams received with a valid channel header.
    pub datagrams_received: u64,
    /// Malformed or unsupported datagrams rejected before delivery.
    pub malformed_datagrams: u64,
    /// Reliable bytes currently charged against the shared congestion window.
    pub bytes_in_flight: usize,
    /// Current shared reliable congestion window.
    pub congestion_window: usize,
    /// Current expendable per-tick byte budget.
    pub unreliable_budget_bytes: usize,
    /// Smoothed minimum RTT, when a non-retransmitted flight has been ACKed.
    pub min_rtt_ticks: Option<u64>,
    /// Current queued/unacknowledged message count for each channel.
    pub queued_messages: [usize; NET_CHANNEL_COUNT],
    /// Per-channel counters in [`NetChannel`] discriminant order.
    pub channels: [ChannelTelemetry; NET_CHANNEL_COUNT],
}

impl MuxTelemetrySnapshot {
    /// Read one channel's counters without requiring an external mapping.
    #[must_use]
    pub const fn channel(self, channel: NetChannel) -> ChannelTelemetry {
        self.channels[channel_index(channel)]
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct NetTelemetry {
    datagrams_sent: u64,
    datagrams_received: u64,
    malformed_datagrams: u64,
    channels: [ChannelTelemetry; NET_CHANNEL_COUNT],
}

impl NetTelemetry {
    pub(crate) fn record_accepted(&mut self, channel: NetChannel) {
        let counters = &mut self.channels[channel_index(channel)];
        counters.messages_accepted = counters.messages_accepted.saturating_add(1);
    }

    pub(crate) fn record_sent(
        &mut self,
        channel: NetChannel,
        bytes: usize,
        first_transmission: bool,
        retransmission: bool,
    ) {
        self.datagrams_sent = self.datagrams_sent.saturating_add(1);
        let counters = &mut self.channels[channel_index(channel)];
        counters.packets_sent = counters.packets_sent.saturating_add(1);
        counters.bytes_sent = counters.bytes_sent.saturating_add(bytes as u64);
        if first_transmission {
            counters.messages_sent = counters.messages_sent.saturating_add(1);
        }
        if retransmission {
            counters.retransmissions = counters.retransmissions.saturating_add(1);
        }
    }

    pub(crate) fn record_received(&mut self, channel: NetChannel, bytes: usize) {
        self.datagrams_received = self.datagrams_received.saturating_add(1);
        let counters = &mut self.channels[channel_index(channel)];
        counters.packets_received = counters.packets_received.saturating_add(1);
        counters.bytes_received = counters.bytes_received.saturating_add(bytes as u64);
    }

    pub(crate) fn record_delivered(&mut self, channel: NetChannel, count: usize) {
        let counters = &mut self.channels[channel_index(channel)];
        counters.messages_received = counters.messages_received.saturating_add(count as u64);
    }

    pub(crate) fn record_dropped(&mut self, channel: NetChannel, count: usize) {
        let counters = &mut self.channels[channel_index(channel)];
        counters.dropped_messages = counters.dropped_messages.saturating_add(count as u64);
    }

    pub(crate) fn record_backpressure(&mut self, channel: NetChannel) {
        let counters = &mut self.channels[channel_index(channel)];
        counters.backpressure_events = counters.backpressure_events.saturating_add(1);
    }

    pub(crate) fn record_malformed(&mut self) {
        self.malformed_datagrams = self.malformed_datagrams.saturating_add(1);
    }

    pub(crate) fn snapshot(
        &self,
        tick: u64,
        bytes_in_flight: usize,
        congestion_window: usize,
        unreliable_budget_bytes: usize,
        min_rtt_ticks: Option<u64>,
        queued_messages: [usize; NET_CHANNEL_COUNT],
    ) -> MuxTelemetrySnapshot {
        MuxTelemetrySnapshot {
            tick,
            datagrams_sent: self.datagrams_sent,
            datagrams_received: self.datagrams_received,
            malformed_datagrams: self.malformed_datagrams,
            bytes_in_flight,
            congestion_window,
            unreliable_budget_bytes,
            min_rtt_ticks,
            queued_messages,
            channels: self.channels,
        }
    }
}

const fn channel_index(channel: NetChannel) -> usize {
    match channel {
        NetChannel::EventsReliable => 0,
        NetChannel::StateUnreliable => 1,
        NetChannel::InputUnreliable => 2,
        NetChannel::RpcReliable => 3,
        NetChannel::BulkReliable => 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_keeps_channels_in_a_stable_wire_order() {
        let mut telemetry = NetTelemetry::default();
        telemetry.record_accepted(NetChannel::EventsReliable);
        telemetry.record_accepted(NetChannel::StateUnreliable);
        telemetry.record_sent(NetChannel::RpcReliable, 12, true, false);
        let snapshot = telemetry.snapshot(7, 10, 20, 30, Some(4), [1, 2, 3, 4, 5]);
        assert_eq!(snapshot.tick, 7);
        assert_eq!(
            snapshot
                .channel(NetChannel::EventsReliable)
                .messages_accepted,
            1
        );
        assert_eq!(snapshot.channel(NetChannel::RpcReliable).bytes_sent, 12);
        assert_eq!(snapshot.queued_messages, [1, 2, 3, 4, 5]);
    }
}
