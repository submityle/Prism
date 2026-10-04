//! Implicit level resolution: UAX#9 rules I1 and I2.

use super::class::BidiClass::{self, *};

/// Applies rules I1 and I2 to one isolating run sequence, raising embedding
/// levels according to each character's resolved class.
///
/// `classes` and `levels` are parallel slices for the sequence; `levels` is
/// modified in place.
pub fn resolve(classes: &[BidiClass], levels: &mut [u8]) {
    for (c, level) in classes.iter().zip(levels.iter_mut()) {
        if *level % 2 == 0 {
            // I1: even (L) level.
            match c {
                R => *level += 1,
                AN | EN => *level += 2,
                _ => {}
            }
        } else {
            // I2: odd (R) level.
            match c {
                L | EN | AN => *level += 1,
                _ => {}
            }
        }
    }
}
