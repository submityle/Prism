//! Weak type resolution: UAX#9 rules W1–W7.
//!
//! Each rule operates on the class slice of a single isolating run sequence,
//! using `sos` as the implicit strong type preceding the first character.

use super::class::BidiClass::{self, *};

/// Applies rules W1–W7 in order to one isolating run sequence.
///
/// `classes` is modified in place; `sos` is the start-of-sequence boundary
/// type (X10) used wherever a rule looks "before" the first character.
pub fn resolve(classes: &mut [BidiClass], sos: BidiClass) {
    w1(classes, sos);
    w2(classes, sos);
    w3(classes);
    w4(classes);
    w5(classes);
    w6(classes);
    w7(classes, sos);
}

/// W1: each NSM takes the type of the previous character, or `sos` at the start;
/// an NSM after an isolate initiator or PDI becomes `ON`.
fn w1(classes: &mut [BidiClass], sos: BidiClass) {
    let mut prev = sos;
    for c in classes.iter_mut() {
        if *c == NSM {
            *c = if prev.is_isolate_initiator() || prev == PDI { ON } else { prev };
        }
        prev = *c;
    }
}

/// W2: each EN becomes AN when the preceding strong type (or `sos`) is AL.
fn w2(classes: &mut [BidiClass], sos: BidiClass) {
    let mut strong = sos;
    for c in classes.iter_mut() {
        match *c {
            L | R | AL => strong = *c,
            EN if strong == AL => *c = AN,
            _ => {}
        }
    }
}

/// W3: every AL becomes R.
fn w3(classes: &mut [BidiClass]) {
    for c in classes.iter_mut() {
        if *c == AL {
            *c = R;
        }
    }
}

/// W4: a single ES between two ENs becomes EN; a single CS between two numbers
/// of the same type becomes that type.
fn w4(classes: &mut [BidiClass]) {
    for i in 1..classes.len().saturating_sub(1) {
        let prev = classes[i - 1];
        let next = classes[i + 1];
        match classes[i] {
            ES | CS if prev == EN && next == EN => classes[i] = EN,
            CS if prev == AN && next == AN => classes[i] = AN,
            _ => {}
        }
    }
}

/// W5: a sequence of ETs adjacent to an EN takes the type EN.
fn w5(classes: &mut [BidiClass]) {
    let n = classes.len();
    let mut i = 0;
    while i < n {
        if classes[i] == ET {
            let start = i;
            while i < n && classes[i] == ET {
                i += 1;
            }
            let before_en = start > 0 && classes[start - 1] == EN;
            let after_en = i < n && classes[i] == EN;
            if before_en || after_en {
                for c in classes.iter_mut().take(i).skip(start) {
                    *c = EN;
                }
            }
        } else {
            i += 1;
        }
    }
}

/// W6: any remaining ES, ET or CS becomes ON.
fn w6(classes: &mut [BidiClass]) {
    for c in classes.iter_mut() {
        if matches!(*c, ES | ET | CS) {
            *c = ON;
        }
    }
}

/// W7: each EN becomes L when the preceding strong type (or `sos`) is L.
fn w7(classes: &mut [BidiClass], sos: BidiClass) {
    let mut strong = sos;
    for c in classes.iter_mut() {
        match *c {
            L | R => strong = *c,
            EN if strong == L => *c = L,
            _ => {}
        }
    }
}
