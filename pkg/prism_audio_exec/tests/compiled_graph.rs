//! End-to-end integration coverage for the sections 29/30 compiled-graph
//! execution model of `prism_audio_exec`.
//!
//! The per-module unit tests check topological ordering, level assignment,
//! island partitioning, buffer liveness, PDC, and scheduling in isolation. This
//! test drives the whole planner the way `ExecPlan::compile` does at load time:
//! a realistic diamond graph with latency plus a second disconnected island,
//! then asserts the cross-module invariants that only hold when the stages
//! agree — topo order respects every edge, wavefronts match longest-path
//! levels and carry no intra-level edge, PDC aligns parallel paths, in-place
//! aliasing shrinks the buffer pool, and `ExecPlan` re-exposes each stage
//! consistently.
//!
//! # Provenance
//! Original work authored for Prism. Contains no Unreal Engine, Unity, Godot,
//! Wwise, FMOD, Steam Audio, Dolby, MPEG, Google Resonance Audio, Web Audio, or
//! Microsoft Project Acoustics source or derived code. No AI/ML is used; these
//! are pure deterministic integer graph algorithms.
//!
//! # Relationship
//! Validates sections 29 (compiled graph execution: topo plan, buffer liveness,
//! PDC) and 30 (deterministic parallel scheduling: wavefront islands) of the
//! Prism audio engine design, over the public API of `prism_audio_exec`.

use prism_audio_exec::{
    allocate_slots, assign_levels, build_schedule, compute_pdc, partition_islands, position_map,
    topological_order, EdgeDesc, ExecError, ExecPlan, GraphDesc, NodeDesc,
};

/// Builds a diamond with latency (`src` -> {`eq`, `rev`} -> `mix` master) and a
/// second disconnected `gnode` -> `sink` island. Returns the graph and its node
/// ids so individual tests can reference them.
fn diamond_plus_island() -> (GraphDesc, [prism_audio_exec::NodeId; 6]) {
    let mut g = GraphDesc::new();
    let src = g.add_node(NodeDesc::new(0, 1));
    let eq = g.add_node(NodeDesc::new(1, 1).with_latency(32).with_in_place(true));
    let rev = g.add_node(NodeDesc::new(1, 1).with_latency(64));
    let mix = g.add_node(NodeDesc::new(2, 1));
    let gnode = g.add_node(NodeDesc::new(0, 1));
    let sink = g.add_node(NodeDesc::new(1, 1));

    g.connect(EdgeDesc::new(src, 0, eq, 0));
    g.connect(EdgeDesc::new(src, 0, rev, 0));
    g.connect(EdgeDesc::new(eq, 0, mix, 0));
    g.connect(EdgeDesc::new(rev, 0, mix, 1));
    g.connect(EdgeDesc::new(gnode, 0, sink, 0));
    g.set_master(mix);

    (g, [src, eq, rev, mix, gnode, sink])
}

/// The topological order must place every producer strictly before each of its
/// consumers, across both islands.
#[test]
fn topo_order_respects_every_edge() {
    let (g, _) = diamond_plus_island();
    g.validate().expect("graph is structurally valid");

    let order = topological_order(&g).expect("acyclic graph orders");
    assert_eq!(order.len(), g.node_count());
    let pos = position_map(&order, g.node_count());
    for edge in g.edges() {
        assert!(
            pos[edge.from_node.0] < pos[edge.to_node.0],
            "producer {:?} must precede consumer {:?}",
            edge.from_node,
            edge.to_node
        );
    }
}

/// Longest-path levels must increase by at least one across every edge, and the
/// parallel schedule must bucket nodes exactly by level with no edge inside a
/// single wavefront.
#[test]
fn levels_and_wavefronts_agree_and_are_edge_free() {
    let (g, ids) = diamond_plus_island();
    let [src, eq, rev, mix, gnode, sink] = ids;
    let order = topological_order(&g).unwrap();

    let levels = assign_levels(&g, &order);
    assert_eq!(levels.level_of(src), 0);
    assert_eq!(levels.level_of(eq), 1);
    assert_eq!(levels.level_of(rev), 1);
    assert_eq!(levels.level_of(mix), 2, "mix sits one past its deepest parent");
    assert_eq!(levels.level_of(gnode), 0);
    assert_eq!(levels.level_of(sink), 1);

    for edge in g.edges() {
        assert!(
            levels.level_of(edge.to_node) > levels.level_of(edge.from_node),
            "level must strictly increase along an edge"
        );
    }

    let schedule = build_schedule(&g, &order);
    assert_eq!(schedule.node_count(), g.node_count());
    assert_eq!(schedule.depth(), levels.depth() as usize);

    // Every node appears in exactly the wavefront matching its level.
    for (wf_index, wave) in schedule.wavefronts().iter().enumerate() {
        for &node in wave {
            assert_eq!(
                levels.level_of(node) as usize,
                wf_index,
                "node landed in the wavefront of its level"
            );
        }
    }
    // No edge connects two nodes within one wavefront (data hazard freedom).
    for edge in g.edges() {
        assert_ne!(
            levels.level_of(edge.from_node),
            levels.level_of(edge.to_node),
            "a dependency must cross a wavefront boundary"
        );
    }
    // Widest wavefront holds the three mid-level nodes {eq, rev, sink}.
    assert_eq!(schedule.max_width(), 3);
}

/// PDC propagates per-path latency and compensates the shorter path so parallel
/// branches re-align at the summing node.
#[test]
fn pdc_aligns_parallel_branches() {
    let (g, ids) = diamond_plus_island();
    let [src, eq, rev, mix, _gen, _sink] = ids;
    let order = topological_order(&g).unwrap();

    let pdc = compute_pdc(&g, &order);
    assert_eq!(pdc.output_latency_of(src), 0);
    assert_eq!(pdc.output_latency_of(eq), 32);
    assert_eq!(pdc.output_latency_of(rev), 64);
    // mix inherits the deeper (64) path and adds its own zero latency.
    assert_eq!(pdc.output_latency_of(mix), 64);
    assert_eq!(pdc.total_latency(), 64);

    // The eq->mix edge is delayed by 32 frames so it lands with the rev->mix
    // branch; all other edges need no compensation.
    let delays = pdc.edge_delay();
    assert_eq!(delays.len(), g.edge_count());
    // Edge order matches insertion: src->eq, src->rev, eq->mix, rev->mix, gnode->sink.
    assert_eq!(delays[0], 0, "src->eq");
    assert_eq!(delays[1], 0, "src->rev");
    assert_eq!(delays[2], 32, "eq->mix delayed to align with the reverb path");
    assert_eq!(delays[3], 0, "rev->mix is the critical path");
    assert_eq!(delays[4], 0, "gnode->sink standalone island");
}

/// Weakly-connected components become separate islands; the diamond and the
/// standalone generator chain must not share one.
#[test]
fn islands_separate_disconnected_components() {
    let (g, ids) = diamond_plus_island();
    let [src, eq, rev, mix, gnode, sink] = ids;

    let part = partition_islands(&g);
    assert_eq!(part.island_count(), 2);
    // Diamond nodes share one island.
    let diamond = part.island_of(src);
    assert_eq!(part.island_of(eq), diamond);
    assert_eq!(part.island_of(rev), diamond);
    assert_eq!(part.island_of(mix), diamond);
    // The generator chain is the other island.
    let standalone = part.island_of(gnode);
    assert_eq!(part.island_of(sink), standalone);
    assert_ne!(diamond, standalone);
}

/// A linear in-place chain must alias its single-consumer input buffers, so the
/// slot pool is smaller than the number of distinct output buffers.
#[test]
fn inplace_chain_shrinks_buffer_pool() {
    let mut g = GraphDesc::new();
    let src = g.add_node(NodeDesc::new(0, 1));
    let eq = g.add_node(NodeDesc::new(1, 1).with_in_place(true));
    let comp = g.add_node(NodeDesc::new(1, 1).with_in_place(true));
    let out = g.add_node(NodeDesc::new(1, 1).with_in_place(true));
    g.connect(EdgeDesc::new(src, 0, eq, 0));
    g.connect(EdgeDesc::new(eq, 0, comp, 0));
    g.connect(EdgeDesc::new(comp, 0, out, 0));
    g.set_master(out);

    let order = topological_order(&g).unwrap();
    let alloc = allocate_slots(&g, &order);

    // Four output buffers, but every stage aliases its sole-consumer input, so
    // the whole chain collapses onto a single physical slot.
    assert_eq!(alloc.buffers().len(), 4);
    assert_eq!(alloc.pool_size(), 1, "a pure in-place chain needs one slot");
    assert_eq!(alloc.inplace_pairs().len(), 3, "each stage aliases its input");

    // Every output buffer resolves to the same physical slot.
    for node in [src, eq, comp, out] {
        let buf = alloc.buffer_of(node, 0).expect("each node has an output buffer");
        assert_eq!(buf.slot, 0);
        assert!(buf.death >= buf.birth, "a buffer dies no earlier than it is born");
    }
}

/// A node fanning its single output to two consumers cannot be aliased in place
/// by the first consumer, because the data is still needed by the second.
#[test]
fn shared_buffer_blocks_premature_inplace_reuse() {
    let mut g = GraphDesc::new();
    let src = g.add_node(NodeDesc::new(0, 1));
    let a = g.add_node(NodeDesc::new(1, 1).with_in_place(true));
    let b = g.add_node(NodeDesc::new(1, 1).with_in_place(true));
    let mix = g.add_node(NodeDesc::new(2, 1));
    g.connect(EdgeDesc::new(src, 0, a, 0));
    g.connect(EdgeDesc::new(src, 0, b, 0));
    g.connect(EdgeDesc::new(a, 0, mix, 0));
    g.connect(EdgeDesc::new(b, 0, mix, 1));
    g.set_master(mix);

    let order = topological_order(&g).unwrap();
    let alloc = allocate_slots(&g, &order);

    // src's buffer feeds both a and b. The earlier consumer a must not overwrite
    // it in place, because b still reads the same data afterwards; the last
    // consumer b may legitimately reuse the slot once no one else needs it.
    let src_buf = alloc.buffer_of(src, 0).unwrap();
    let a_buf = alloc.buffer_of(a, 0).unwrap();
    let b_buf = alloc.buffer_of(b, 0).unwrap();
    assert_ne!(src_buf.slot, a_buf.slot, "earlier consumer a cannot alias the shared buffer");
    assert_eq!(src_buf.slot, b_buf.slot, "last consumer b reuses the freed source slot");
    assert!(alloc.pool_size() >= 2);
}

/// `ExecPlan::compile` must re-expose each stage consistently with running the
/// stages directly.
#[test]
fn exec_plan_bundles_stages_consistently() {
    let (g, ids) = diamond_plus_island();
    let [_src, _eq, _rev, mix, _gen, _sink] = ids;

    let plan = ExecPlan::compile(&g).expect("valid graph compiles");
    let order = topological_order(&g).unwrap();

    assert_eq!(plan.order(), order.as_slice());
    assert_eq!(plan.master(), mix);
    assert_eq!(plan.islands().island_count(), 2);
    assert_eq!(plan.schedule().node_count(), g.node_count());
    assert_eq!(plan.buffer_pool_size(), plan.allocation().pool_size());
    assert_eq!(plan.total_latency(), plan.pdc().total_latency());
    assert_eq!(plan.total_latency(), 64);

    // master_slot is the physical slot of the master node's output buffer.
    let master_buf = plan.allocation().buffer_of(mix, 0).expect("master output buffer");
    assert_eq!(plan.master_slot(), master_buf.slot);
}

/// Structural errors must surface as recoverable `ExecError`s, never panics.
#[test]
fn invalid_graphs_are_rejected() {
    // Cycle: a -> b -> a.
    let mut cyclic = GraphDesc::new();
    let a = cyclic.add_node(NodeDesc::new(1, 1));
    let b = cyclic.add_node(NodeDesc::new(1, 1));
    cyclic.connect(EdgeDesc::new(a, 0, b, 0));
    cyclic.connect(EdgeDesc::new(b, 0, a, 0));
    cyclic.set_master(b);
    match ExecPlan::compile(&cyclic) {
        Err(ExecError::Cycle) => {}
        other => panic!("expected Cycle, got {other:?}"),
    }

    // No master designated.
    let mut no_master = GraphDesc::new();
    no_master.add_node(NodeDesc::new(0, 1));
    match ExecPlan::compile(&no_master) {
        Err(ExecError::NoMaster) => {}
        other => panic!("expected NoMaster, got {other:?}"),
    }

    // Edge referencing a non-existent node.
    let mut unknown = GraphDesc::new();
    let only = unknown.add_node(NodeDesc::new(0, 1));
    unknown.connect(EdgeDesc::new(only, 0, prism_audio_exec::NodeId(9), 0));
    unknown.set_master(only);
    match ExecPlan::compile(&unknown) {
        Err(ExecError::UnknownNode(prism_audio_exec::NodeId(9))) => {}
        other => panic!("expected UnknownNode(9), got {other:?}"),
    }

    // Port index beyond the declared port count.
    let mut bad_port = GraphDesc::new();
    let p = bad_port.add_node(NodeDesc::new(0, 1));
    let q = bad_port.add_node(NodeDesc::new(1, 1));
    bad_port.connect(EdgeDesc::new(p, 3, q, 0)); // p only has 1 output port
    bad_port.set_master(q);
    match ExecPlan::compile(&bad_port) {
        Err(ExecError::PortOutOfRange { node, port, .. }) => {
            assert_eq!(node, p);
            assert_eq!(port, 3);
        }
        other => panic!("expected PortOutOfRange, got {other:?}"),
    }
}
