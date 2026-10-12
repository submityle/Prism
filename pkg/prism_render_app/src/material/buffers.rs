//! The material ABI in GPU memory: header rows and the parameter word heap.
//!
//! Two buffers hold every material: a row per material in the header buffer, and
//! a shared word heap holding each material's packed surface block at the
//! offset its row names. A shader reads the row for the material it is shading
//! and decodes `parameter_size / 4` words from the heap with `lobe_mask`, which
//! is why the heap is *variable* length — a plain dielectric costs 12 words, a
//! fully-loaded surface up to 36 — and why it needs a real sub-allocator
//! ([`ParameterHeap`]) rather than a fixed stride.
//!
//! This layer places and uploads the ABI; what a header contains and how a
//! surface packs stays in `prism_render_material`.

use core::mem::size_of;

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use prism_render_driver::{BufferDescriptor, BufferId, BufferUsages, RenderDevice, RenderQueue};
use prism_render_material::{
    GpuMaterialHeader, GpuMaterialTexture, MaterialRecord, inactive_material_header,
};

use super::ParameterHeap;

/// The usages the header buffer is created with: a shader reads the rows.
pub(crate) const HEADER_USAGES: BufferUsages =
    BufferUsages::STORAGE.union(BufferUsages::COPY_DST);
/// The usages the parameter word heap is created with: a shader reads the words.
pub(crate) const PARAMETER_USAGES: BufferUsages =
    BufferUsages::STORAGE.union(BufferUsages::COPY_DST);
/// The usages the texture row buffer is created with: a shader reads the rows.
pub(crate) const TEXTURE_USAGES: BufferUsages =
    BufferUsages::STORAGE.union(BufferUsages::COPY_DST);

/// One material's run in the word heap.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Placement {
    /// The word offset of the packed surface block.
    offset: u32,
    /// The block's length in words.
    words: u32,
}

/// What one sync did, so a caller can tell a quiet frame from a rebuild.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct MaterialSyncReport {
    /// Materials whose header row was written this pass.
    pub headers_written: u32,
    /// Materials whose packed parameters were written this pass.
    pub parameters_written: u32,
    /// Materials whose texture rows were written this pass.
    pub textures_written: u32,
    /// Materials that gave up their previous run because their block changed
    /// size.
    pub relocated: u32,
    /// Whether a buffer was reallocated (and therefore fully re-uploaded).
    pub grew: bool,
}

/// The material ABI's two buffers, with a CPU mirror of each.
///
/// A render-world resource: the render world owns the GPU buffers, and the
/// prepare stage is the only thing that writes them.
///
/// The mirror is not a cache: it is how the buffers can *grow*. A reallocated
/// buffer keeps nothing, so the new one has to be filled from something that
/// survives the reallocation — the mirror — without asking the caller to hand
/// back every material it ever synced. It is also what keeps a quiet frame quiet:
/// only the rows and words that changed are uploaded.
#[derive(prism_ecs::prelude::Resource)]
pub struct MaterialParameterBuffers {
    headers: Option<BufferId>,
    header_capacity: u32,
    parameters: Option<BufferId>,
    parameter_capacity_words: u32,
    heap: ParameterHeap,
    textures: Option<BufferId>,
    texture_capacity_rows: u32,
    /// The texture table is the same kind of address space as the parameter heap
    /// — a material owns a variable-length run of rows — so it uses the same
    /// allocator, with rows for words.
    texture_heap: ParameterHeap,
    /// Row mirror, one entry per header slot.
    rows: Vec<GpuMaterialHeader>,
    /// Word mirror, one entry per parameter word slot.
    words: Vec<u32>,
    /// Texture row mirror, one entry per texture row slot.
    texture_rows: Vec<GpuMaterialTexture>,
    placements: BTreeMap<u32, Placement>,
    /// Each material's run in the texture table.
    texture_placements: BTreeMap<u32, Placement>,
    /// Extra usages every created buffer carries — `COPY_SRC` when a caller wants
    /// a diagnostic readback; production passes none.
    extra_usage: BufferUsages,
}

impl MaterialParameterBuffers {
    /// Allocates the ABI buffers, sized for `header_capacity` materials and
    /// `parameter_words` of surface data.
    #[must_use]
    pub fn new<D: RenderDevice>(
        device: &D,
        header_capacity: u32,
        parameter_words: u32,
        texture_rows: u32,
        extra_usage: BufferUsages,
    ) -> Self {
        let mut buffers = Self {
            headers: None,
            header_capacity: 0,
            parameters: None,
            parameter_capacity_words: 0,
            heap: ParameterHeap::new(),
            textures: None,
            texture_capacity_rows: 0,
            texture_heap: ParameterHeap::new(),
            rows: Vec::new(),
            words: Vec::new(),
            texture_rows: Vec::new(),
            placements: BTreeMap::new(),
            texture_placements: BTreeMap::new(),
            extra_usage,
        };
        buffers.allocate(
            device,
            header_capacity.max(1),
            parameter_words.max(1),
            texture_rows.max(1),
            extra_usage,
        );
        buffers
    }

    /// The header buffer (one [`GpuMaterialHeader`] per material slot).
    #[must_use]
    pub const fn headers(&self) -> Option<BufferId> {
        self.headers
    }

    /// The parameter word heap.
    #[must_use]
    pub const fn parameters(&self) -> Option<BufferId> {
        self.parameters
    }

    /// The texture row buffer.
    #[must_use]
    pub const fn textures(&self) -> Option<BufferId> {
        self.textures
    }

    /// How many header slots the buffer holds.
    #[must_use]
    pub const fn header_capacity(&self) -> u32 {
        self.header_capacity
    }

    /// How many parameter words the heap buffer holds.
    #[must_use]
    pub const fn parameter_capacity_words(&self) -> u32 {
        self.parameter_capacity_words
    }

    /// How many words the heap currently uses (one past the last allocated
    /// word), i.e. how far into the buffer a shader may read.
    #[must_use]
    pub fn used_words(&self) -> u32 {
        self.heap.word_capacity()
    }

    /// How many texture rows the table currently uses.
    #[must_use]
    pub fn used_texture_rows(&self) -> u32 {
        self.texture_heap.word_capacity()
    }

    /// How many texture rows the buffer holds.
    #[must_use]
    pub const fn texture_capacity_rows(&self) -> u32 {
        self.texture_capacity_rows
    }

    /// The texture rows of `index`, as last uploaded.
    #[must_use]
    pub fn textures_of(&self, index: u32) -> Option<&[GpuMaterialTexture]> {
        let placement = self.texture_placements.get(&index)?;
        let start = placement.offset as usize;
        self.texture_rows
            .get(start..start + placement.words as usize)
    }

    /// The header row of `index`, as last uploaded.
    #[must_use]
    pub fn header(&self, index: u32) -> Option<&GpuMaterialHeader> {
        self.rows.get(index as usize)
    }

    /// The packed word run of `index`, as last uploaded.
    #[must_use]
    pub fn parameters_of(&self, index: u32) -> Option<&[u32]> {
        let placement = self.placements.get(&index)?;
        let start = placement.offset as usize;
        self.words.get(start..start + placement.words as usize)
    }

    /// Packs `records` into the buffers, reusing each material's run when its
    /// block length is unchanged and relocating it when the lobe set changed.
    ///
    /// `records` is `(material index, record)`; the index is the row a shader
    /// addresses, so it must be the material handle's index. Only changed rows
    /// and words are uploaded, unless a buffer had to grow — then the mirrors
    /// refill it in full.
    pub fn sync<D: RenderDevice, Q: RenderQueue>(
        &mut self,
        device: &D,
        queue: &Q,
        records: impl IntoIterator<Item = (u32, MaterialRecord)>,
        epoch: u64,
    ) -> MaterialSyncReport {
        let records: Vec<(u32, MaterialRecord)> = records.into_iter().collect();
        let mut report = MaterialSyncReport::default();

        let mut dirty_headers = Vec::new();
        let mut dirty_words: Vec<(u32, u32)> = Vec::new();
        let mut dirty_textures: Vec<(u32, u32)> = Vec::new();

        for (index, record) in records {
            // A material past the current row count grows the header buffer; the
            // mirrors carry what was already there into the new allocation.
            if self.ensure_header(device, index + 1) {
                report.grew = true;
            }

            let packed = record.packed_parameters();
            let block = packed.pack();
            let len = block.len() as u32;

            let placement = match self.placements.get(&index) {
                // Same width: the run already fits. A lobe *value* changes far
                // more often than the lobe *set*, so this is the common case.
                Some(placement) if placement.words == len => *placement,
                previous => {
                    if let Some(previous) = previous {
                        self.heap.free(previous.offset, previous.words);
                        report.relocated += 1;
                    }
                    let placement = Placement {
                        offset: self.heap.alloc(len),
                        words: len,
                    };
                    self.placements.insert(index, placement);
                    placement
                }
            };

            // The heap may have bumped its high-water mark past what the buffer
            // covers, which is what grows the word heap.
            if self.ensure_words(device, self.heap.word_capacity()) {
                report.grew = true;
            }

            let start = placement.offset as usize;
            let slot = &mut self.words[start..start + len as usize];
            if slot != block.as_slice() {
                slot.copy_from_slice(&block);
                dirty_words.push((placement.offset, placement.words));
                report.parameters_written += 1;
            }

            // The material's texture entries own their own variable-length run
            // in the texture table, addressed by the header's `texture_offset` /
            // `texture_count` exactly as parameters are by `parameter_offset`.
            let textures = &record.textures;
            let texture_len = textures.len() as u32;
            let texture_placement = match self.texture_placements.get(&index) {
                Some(placement) if placement.words == texture_len => *placement,
                previous => {
                    if let Some(previous) = previous {
                        self.texture_heap.free(previous.offset, previous.words);
                    }
                    let placement = Placement {
                        offset: self.texture_heap.alloc(texture_len),
                        words: texture_len,
                    };
                    self.texture_placements.insert(index, placement);
                    placement
                }
            };
            if self.ensure_texture_rows(device, self.texture_heap.word_capacity()) {
                report.grew = true;
            }
            if texture_len != 0 {
                let start = texture_placement.offset as usize;
                let slot = &mut self.texture_rows[start..start + texture_len as usize];
                if slot != textures.as_slice() {
                    slot.copy_from_slice(textures);
                    dirty_textures.push((texture_placement.offset, texture_len));
                    report.textures_written += 1;
                }
            }

            let row = record.header(
                placement.offset,
                texture_placement.offset,
                epoch,
            );
            let slot = &mut self.rows[index as usize];
            if *slot != row {
                *slot = row;
                dirty_headers.push(index);
                report.headers_written += 1;
            }
        }

        if report.grew {
            self.upload_all(queue);
        } else {
            self.upload_rows(queue, &dirty_headers);
            self.upload_words(queue, &dirty_words);
            self.upload_textures(queue, &dirty_textures);
        }
        report
    }

    /// Retires a material: its run goes back to the heap and its row becomes an
    /// inactive placeholder carrying the slot's generation, so a shader holding
    /// a stale handle rejects the row instead of reading freed words.
    pub fn remove<Q: RenderQueue>(&mut self, queue: &Q, index: u32, generation: u32) {
        let Some(placement) = self.placements.remove(&index) else {
            return;
        };
        self.heap.free(placement.offset, placement.words);
        if let Some(textures) = self.texture_placements.remove(&index) {
            self.texture_heap.free(textures.offset, textures.words);
        }
        let Some(slot) = self.rows.get_mut(index as usize) else {
            return;
        };
        let row = inactive_material_header(generation);
        if *slot != row {
            *slot = row;
            self.upload_rows(queue, &[index]);
        }
    }

    /// Releases every buffer.
    pub fn release<D: RenderDevice>(&mut self, device: &D) {
        if let (Some(headers), Some(parameters), Some(textures)) =
            (self.headers, self.parameters, self.textures)
        {
            device.destroy_buffer(headers);
            device.destroy_buffer(parameters);
            device.destroy_buffer(textures);
        }
        self.headers = None;
        self.parameters = None;
        self.textures = None;
    }

    /// Grows the header buffer so it has at least `needed` rows, preserving the
    /// mirror. Returns whether it reallocated.
    fn ensure_header<D: RenderDevice>(&mut self, device: &D, needed: u32) -> bool {
        if needed <= self.header_capacity {
            return false;
        }
        let capacity = needed
            .next_power_of_two()
            .max(self.header_capacity.saturating_mul(2));
        self.allocate(
            device,
            capacity,
            self.parameter_capacity_words,
            self.texture_capacity_rows,
            self.extra_usage,
        );
        true
    }

    /// Grows the word heap buffer so it covers `needed` words, preserving the
    /// mirror. Returns whether it reallocated.
    fn ensure_words<D: RenderDevice>(&mut self, device: &D, needed: u32) -> bool {
        if needed <= self.parameter_capacity_words {
            return false;
        }
        let capacity = needed
            .next_power_of_two()
            .max(self.parameter_capacity_words.saturating_mul(2));
        self.allocate(
            device,
            self.header_capacity,
            capacity,
            self.texture_capacity_rows,
            self.extra_usage,
        );
        true
    }

    /// (Re)allocates both buffers, refilling them from the mirrors.
    fn allocate<D: RenderDevice>(
        &mut self,
        device: &D,
        header_capacity: u32,
        words: u32,
        texture_rows: u32,
        extra_usage: BufferUsages,
    ) {
        if let (Some(headers), Some(parameters), Some(textures)) =
            (self.headers, self.parameters, self.textures)
        {
            device.destroy_buffer(headers);
            device.destroy_buffer(parameters);
            device.destroy_buffer(textures);
        }
        self.header_capacity = header_capacity;
        self.parameter_capacity_words = words;
        self.texture_capacity_rows = texture_rows;
        self.headers = Some(device.create_buffer(&BufferDescriptor {
            label: Some("prism material headers".into()),
            size: u64::from(header_capacity) * header_row_bytes(),
            usage: HEADER_USAGES.union(extra_usage),
            mapped_at_creation: false,
        }));
        self.parameters = Some(device.create_buffer(&BufferDescriptor {
            label: Some("prism material parameters".into()),
            size: u64::from(words) * 4,
            usage: PARAMETER_USAGES.union(extra_usage),
            mapped_at_creation: false,
        }));
        self.textures = Some(device.create_buffer(&BufferDescriptor {
            label: Some("prism material textures".into()),
            size: u64::from(texture_rows) * texture_row_bytes(),
            usage: TEXTURE_USAGES.union(extra_usage),
            mapped_at_creation: false,
        }));
        // The mirrors are the buffers' contents, so growth *extends* them: the
        // rows and words already written stay, and only the slack a reallocated
        // buffer gained starts as explicit values rather than whatever the driver
        // handed back.
        self.rows
            .resize(header_capacity as usize, inactive_material_header(0));
        self.words.resize(words as usize, 0);
        self.texture_rows
            .resize(texture_rows as usize, GpuMaterialTexture::default());
    }

    /// Uploads the whole mirror (after a growth).
    fn upload_all<Q: RenderQueue>(&self, queue: &Q) {
        let (Some(headers), Some(parameters), Some(textures)) =
            (self.headers, self.parameters, self.textures)
        else {
            return;
        };
        queue.write_buffer(headers, 0, &header_bytes(&self.rows));
        queue.write_buffer(parameters, 0, &word_bytes(&self.words));
        queue.write_buffer(textures, 0, &texture_bytes(&self.texture_rows));
    }

    /// Uploads the given texture runs, batched into the span they cover.
    fn upload_textures<Q: RenderQueue>(&self, queue: &Q, dirty: &[(u32, u32)]) {
        let Some(first) = dirty.iter().map(|(offset, _)| *offset).min() else {
            return;
        };
        let last = dirty
            .iter()
            .map(|(offset, rows)| offset + rows)
            .max()
            .expect("a dirty list is non-empty");
        let Some(textures) = self.textures else {
            return;
        };
        queue.write_buffer(
            textures,
            u64::from(first) * texture_row_bytes(),
            &texture_bytes(&self.texture_rows[first as usize..last as usize]),
        );
    }

    /// Grows the texture row buffer so it holds `needed` rows, preserving the
    /// mirror. Returns whether it reallocated.
    fn ensure_texture_rows<D: RenderDevice>(&mut self, device: &D, needed: u32) -> bool {
        if needed <= self.texture_capacity_rows {
            return false;
        }
        let capacity = needed
            .next_power_of_two()
            .max(self.texture_capacity_rows.saturating_mul(2));
        self.allocate(
            device,
            self.header_capacity,
            self.parameter_capacity_words,
            capacity,
            self.extra_usage,
        );
        true
    }

    /// Uploads the given header rows, batched into the span they cover.
    fn upload_rows<Q: RenderQueue>(&self, queue: &Q, dirty: &[u32]) {
        let Some((first, last)) = span(dirty) else {
            return;
        };
        let Some(headers) = self.headers else {
            return;
        };
        let rows = &self.rows[first as usize..=last as usize];
        queue.write_buffer(
            headers,
            u64::from(first) * header_row_bytes(),
            &header_bytes(rows),
        );
    }

    /// Uploads the given word runs, batched into the span they cover.
    fn upload_words<Q: RenderQueue>(&self, queue: &Q, dirty: &[(u32, u32)]) {
        let Some(first) = dirty.iter().map(|(offset, _)| *offset).min() else {
            return;
        };
        let last = dirty
            .iter()
            .map(|(offset, words)| offset + words)
            .max()
            .expect("a dirty list is non-empty");
        let Some(parameters) = self.parameters else {
            return;
        };
        queue.write_buffer(
            parameters,
            u64::from(first) * 4,
            &word_bytes(&self.words[first as usize..last as usize]),
        );
    }
}

/// The byte size of one header row.
#[must_use]
pub(crate) fn header_row_bytes() -> u64 {
    size_of::<GpuMaterialHeader>() as u64
}

/// The byte size of one texture row.
#[must_use]
pub(crate) fn texture_row_bytes() -> u64 {
    size_of::<GpuMaterialTexture>() as u64
}

/// Texture rows as bytes, in the layout the ABI defines.
fn texture_bytes(rows: &[GpuMaterialTexture]) -> Vec<u8> {
    bytemuck::cast_slice(rows).to_vec()
}

/// Rows as bytes, in the layout the ABI defines.
fn header_bytes(rows: &[GpuMaterialHeader]) -> Vec<u8> {
    bytemuck::cast_slice(rows).to_vec()
}

/// Words as little-endian bytes.
fn word_bytes(words: &[u32]) -> Vec<u8> {
    bytemuck::cast_slice(words).to_vec()
}

/// The inclusive `(first, last)` indices a dirty list covers.
fn span(dirty: &[u32]) -> Option<(u32, u32)> {
    Some((
        dirty.iter().copied().min()?,
        dirty.iter().copied().max()?,
    ))
}

