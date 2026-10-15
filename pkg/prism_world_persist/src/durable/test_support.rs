//! Inspect journal bytes through the original locked handle, including on Windows.

use std::io::{Read, Seek, SeekFrom};

use super::FileJournal;

pub(crate) fn read_journal_bytes(journal: &mut FileJournal) -> Vec<u8> {
    let file = &mut journal.file;
    let position = file.stream_position().expect("save journal cursor");
    let mut bytes = Vec::new();
    let read_result = (|| {
        file.seek(SeekFrom::Start(0))?;
        file.read_to_end(&mut bytes)
    })();
    // Restore even when inspection failed, before reporting either error.
    let restore_result = file.seek(SeekFrom::Start(position));
    assert_eq!(restore_result.expect("restore journal cursor"), position);
    read_result.expect("read journal through locked handle");
    bytes
}

#[test]
fn inspecting_locked_journal_preserves_non_eof_cursor_and_lock() {
    use super::{encode_record, journal_header, JournalError};
    use crate::durable_tests::{temporary_path, transaction};

    let path = temporary_path("locked-inspection");
    let first = transaction(104, 1, 0, 10);
    let mut expected = journal_header().to_vec();
    expected.extend_from_slice(&encode_record(&first).unwrap());
    let mut journal = FileJournal::open(&path).unwrap();
    journal.append(first).unwrap();

    for position in [1, expected.len() as u64 + 17] {
        journal.file.seek(SeekFrom::Start(position)).unwrap();
        assert_eq!(read_journal_bytes(&mut journal), expected);
        assert_eq!(journal.file.stream_position().unwrap(), position);
        assert!(matches!(
            FileJournal::open(&path),
            Err(JournalError::Locked)
        ));
    }
    journal.append(transaction(105, 2, 1, 20)).unwrap();
    assert_eq!(journal.log().len(), 2);
}
