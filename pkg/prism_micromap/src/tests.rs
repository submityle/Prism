//! Unit tests for the device-free `OMM` baker.

use alloc::vec;
use alloc::vec::Vec;

use crate::omm::{
    bake_triangle, micro_triangle_at, micro_triangles, pack, packed_len, unpack, AlphaMask,
    OmmBakeInput, OmmBuilder, OmmFormat, OpacityState, SampleStrategy, SubdivisionLevel,
    TextureAlphaMask, WrapMode,
};

/// Approximate float equality for deterministic scalar checks.
fn approx(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-6
}

/// A canonical unit base triangle covering the lower-left `UV` corner.
const BASE_TRI: [[f32; 2]; 3] = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];

/// Builds a fully opaque or fully transparent flat mask.
fn flat_mask(opaque: bool) -> TextureAlphaMask {
    let value = if opaque { 1.0 } else { 0.0 };
    TextureAlphaMask::new(4, 4, vec![value; 16], 0.5)
}

/// Builds a mask whose left half is transparent and right half opaque.
fn left_right_mask() -> TextureAlphaMask {
    // 4x4: columns 0,1 transparent (0.0); columns 2,3 opaque (1.0).
    let mut alpha = Vec::with_capacity(16);
    for _y in 0..4 {
        for x in 0..4u32 {
            alpha.push(if x < 2 { 0.0 } else { 1.0 });
        }
    }
    TextureAlphaMask::new(4, 4, alpha, 0.5)
}

fn bake(mask: &TextureAlphaMask, level: u8, format: OmmFormat) -> crate::omm::BakedOmm {
    let input = OmmBakeInput {
        uv: BASE_TRI,
        level: SubdivisionLevel::new(level).unwrap(),
        format,
        strategy: SampleStrategy::Uniform { samples_per_edge: 4 },
    };
    bake_triangle(&input, mask)
}

#[test]
fn all_opaque_mask_bakes_all_opaque() {
    let baked = bake(&flat_mask(true), 2, OmmFormat::FourState);
    assert_eq!(baked.micro_triangle_count(), 16);
    assert!(baked.states().iter().all(|s| *s == OpacityState::Opaque));
    assert_eq!(baked.stats().opaque, 16);
    assert_eq!(baked.stats().transparent, 0);
    assert_eq!(baked.stats().unknown, 0);
    assert!(approx(baked.stats().unknown_ratio(), 0.0));
}

#[test]
fn all_transparent_mask_bakes_all_transparent() {
    let baked = bake(&flat_mask(false), 2, OmmFormat::FourState);
    assert!(baked
        .states()
        .iter()
        .all(|s| *s == OpacityState::Transparent));
    assert_eq!(baked.stats().transparent, 16);
    assert_eq!(baked.stats().total(), 16);
}

#[test]
fn half_half_mask_produces_unknown_slivers() {
    let baked = bake(&left_right_mask(), 3, OmmFormat::FourState);
    assert!(baked.stats().unknown > 0, "edge should straddle micro-tris");
    assert!(baked.stats().opaque > 0);
    assert!(baked.stats().transparent > 0);
    let ratio = baked.stats().unknown_ratio();
    assert!((0.0..=1.0).contains(&ratio));
}

#[test]
fn micro_triangle_count_is_four_to_the_level() {
    for level in 0..=6u8 {
        let lvl = SubdivisionLevel::new(level).unwrap();
        let expected = 1u32 << (2 * u32::from(level));
        assert_eq!(lvl.micro_triangle_count(), expected);
        assert_eq!(micro_triangles(lvl).len() as u32, expected);
    }
}

#[test]
fn random_access_matches_iteration_order() {
    let lvl = SubdivisionLevel::new(4).unwrap();
    let n = lvl.segments();
    let all = micro_triangles(lvl);
    for (index, tri) in all.iter().enumerate() {
        let at = micro_triangle_at(lvl, index as u32).expect("in range");
        assert_eq!(&at, tri);
        // Every vertex is a lattice point summing to `n`.
        for vertex in &tri.vertices {
            assert_eq!(vertex[0] + vertex[1] + vertex[2], n);
        }
    }
    assert!(micro_triangle_at(lvl, lvl.micro_triangle_count()).is_none());
}

#[test]
fn subdivision_level_rejects_out_of_range() {
    assert!(SubdivisionLevel::new(12).is_some());
    assert!(SubdivisionLevel::new(13).is_none());
}

#[test]
fn pack_unpack_round_trips_four_state() {
    let states = [
        OpacityState::Transparent,
        OpacityState::Opaque,
        OpacityState::UnknownTransparent,
        OpacityState::UnknownOpaque,
        OpacityState::Opaque,
    ];
    let bytes = pack(&states, OmmFormat::FourState);
    assert_eq!(bytes.len(), packed_len(states.len() as u32, OmmFormat::FourState));
    let back = unpack(&bytes, states.len() as u32, OmmFormat::FourState).unwrap();
    assert_eq!(back.as_slice(), states.as_slice());
}

#[test]
fn pack_unpack_round_trips_two_state() {
    // Two-state packing collapses the unknown variants deterministically.
    let states = [
        OpacityState::Transparent,
        OpacityState::Opaque,
        OpacityState::UnknownTransparent,
        OpacityState::UnknownOpaque,
    ];
    let bytes = pack(&states, OmmFormat::TwoState);
    assert_eq!(bytes.len(), packed_len(states.len() as u32, OmmFormat::TwoState));
    let back = unpack(&bytes, states.len() as u32, OmmFormat::TwoState).unwrap();
    assert_eq!(
        back,
        vec![
            OpacityState::Transparent,
            OpacityState::Opaque,
            OpacityState::Transparent,
            OpacityState::Opaque,
        ]
    );
}

#[test]
fn pack_uses_little_endian_within_byte() {
    // Four-state: entries 0..=3 pack into one byte, low bits first.
    let states = [
        OpacityState::Opaque,            // 0b01 at shift 0
        OpacityState::UnknownTransparent, // 0b10 at shift 2
        OpacityState::UnknownOpaque,     // 0b11 at shift 4
        OpacityState::Transparent,       // 0b00 at shift 6
    ];
    let bytes = pack(&states, OmmFormat::FourState);
    assert_eq!(bytes.len(), 1);
    assert_eq!(bytes[0], 0b00_11_10_01);
}

#[test]
fn two_state_resolution_matches_to_2state() {
    assert_eq!(
        OpacityState::UnknownOpaque.to_2state(),
        OpacityState::Opaque
    );
    assert_eq!(
        OpacityState::UnknownTransparent.to_2state(),
        OpacityState::Transparent
    );
    assert_eq!(OpacityState::Opaque.to_2state(), OpacityState::Opaque);
    assert_eq!(
        OpacityState::Transparent.to_2state(),
        OpacityState::Transparent
    );
}

#[test]
fn opacity_state_round_trips_through_u8() {
    for raw in 0u8..=3 {
        let state = OpacityState::from_u8(raw).unwrap();
        assert_eq!(state.as_u8(), raw);
    }
    assert!(OpacityState::from_u8(4).is_none());
}

#[test]
fn unpack_rejects_short_buffer() {
    let short = [0u8; 1];
    assert!(unpack(&short, 100, OmmFormat::FourState).is_none());
}

#[test]
fn builder_deduplicates_identical_triangles() {
    let mask = flat_mask(true);
    let level = SubdivisionLevel::new(2).unwrap();
    let mut builder = OmmBuilder::new(OmmFormat::FourState, SampleStrategy::default());
    // Three identical all-opaque triangles should collapse to one micromap.
    for _ in 0..3 {
        builder.add_triangle(BASE_TRI, level, &mask);
    }
    let out = builder.finish();
    assert_eq!(out.triangle_count(), 3);
    assert_eq!(out.unique_count(), 1);
    assert_eq!(out.indices, vec![0, 0, 0]);
}

#[test]
fn builder_keeps_distinct_triangles_separate() {
    let opaque = flat_mask(true);
    let transparent = flat_mask(false);
    let level = SubdivisionLevel::new(2).unwrap();
    let mut builder = OmmBuilder::new(OmmFormat::FourState, SampleStrategy::default());
    let a = builder.add_triangle(BASE_TRI, level, &opaque);
    let b = builder.add_triangle(BASE_TRI, level, &transparent);
    assert_ne!(a, b);
    let out = builder.finish();
    assert_eq!(out.unique_count(), 2);
    assert_eq!(out.triangle_count(), 2);
}

#[test]
fn wrap_mode_repeat_and_clamp() {
    assert!(approx(WrapMode::Repeat.apply(1.5), 0.5));
    assert!(approx(WrapMode::Repeat.apply(-0.25), 0.75));
    assert!(approx(WrapMode::Clamp.apply(1.5), 1.0));
    assert!(approx(WrapMode::Clamp.apply(-0.5), 0.0));
    assert!(approx(WrapMode::Clamp.apply(0.3), 0.3));
}

#[test]
fn texture_mask_texel_bounds() {
    let mask = left_right_mask();
    assert_eq!(mask.width(), 4);
    assert_eq!(mask.height(), 4);
    assert_eq!(mask.wrap(), WrapMode::Clamp);
    assert_eq!(mask.texel(0, 0), Some(0.0));
    assert_eq!(mask.texel(3, 0), Some(1.0));
    assert!(mask.texel(4, 0).is_none());
    assert!(mask.is_opaque(0.9, 0.1));
    assert!(!mask.is_opaque(0.1, 0.1));
}

#[test]
fn format_bit_widths_and_packed_len() {
    assert_eq!(OmmFormat::TwoState.bits_per_micro_triangle(), 1);
    assert_eq!(OmmFormat::FourState.bits_per_micro_triangle(), 2);
    assert_eq!(packed_len(8, OmmFormat::TwoState), 1);
    assert_eq!(packed_len(9, OmmFormat::TwoState), 2);
    assert_eq!(packed_len(4, OmmFormat::FourState), 1);
    assert_eq!(packed_len(5, OmmFormat::FourState), 2);
    assert_eq!(packed_len(0, OmmFormat::FourState), 0);
}

#[test]
fn texel_conservative_strategy_classifies_flat_mask() {
    let input = OmmBakeInput {
        uv: BASE_TRI,
        level: SubdivisionLevel::new(2).unwrap(),
        format: OmmFormat::FourState,
        strategy: SampleStrategy::TexelConservative,
    };
    let baked = bake_triangle(&input, &flat_mask(true));
    assert!(baked.states().iter().all(|s| *s == OpacityState::Opaque));
}

#[test]
fn baked_data_matches_repacked_states() {
    let baked = bake(&left_right_mask(), 3, OmmFormat::FourState);
    let repacked = pack(baked.states(), OmmFormat::FourState);
    assert_eq!(baked.data(), repacked.as_slice());
}
