use candle_core::{DType, Device, Result, Tensor};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};

/// Shape of a tensor from a safetensor file header.
/// Tensors in GLM-5.2 FP8 are either 1D (e.g. layernorm scales) or 2D (weight matrices).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TensorShape {
    D1(usize),
    D2(usize, usize), // (rows, cols) — row-major
}

/// How a tensor is distributed across GPUs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Dist {
    /// Each rank has a full copy.
    Replicated,
    /// Output dim (dim 0) is split: each rank gets `global_rows / nproc` rows.
    ColumnSharded,
    /// Input dim (dim 1) is split: each rank gets `global_cols / nproc` cols.
    RowSharded,
    /// Vocab-parallel linear: output dim split, followed by an all-gather.
    VocabParallel,
}

/// Metadata for a single tensor from a safetensor file header.
/// `data_offsets` are absolute file offsets (converted from safetensor
/// data-section-relative offsets during index construction).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorMeta {
    pub src_dtype: String,
    pub shape: TensorShape,
    /// Absolute byte range `[start, end)` within the safetensor file.
    pub data_offsets: (usize, usize),
    pub dist: Dist,
    /// Handle to the allocated GPU buffer (only set after `_alloc`).
    #[serde(skip)]
    pub buffer: Option<Tensor>,
    /// Desired GPU dtype for allocation and loading, when different from the
    /// on-disk safetensor dtype (`src_dtype`).  E.g. RMS norm weights stored
    /// as BF16 but used as F32.  Set by the `_alloc` constructor before
    /// allocating.  `None` means `src_dtype` is the target.
    pub dst_dtype: Option<String>,
}

#[derive(Deserialize)]
struct SafetensorTensorEntry {
    dtype: String,
    #[serde(deserialize_with = "deserialize_shape")]
    shape: TensorShape,
    #[serde(deserialize_with = "deserialize_pair")]
    data_offsets: (usize, usize),
}

fn deserialize_shape<'de, D>(deserializer: D) -> std::result::Result<TensorShape, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let arr: Vec<usize> = Vec::deserialize(deserializer)?;
    match arr.len() {
        1 => Ok(TensorShape::D1(arr[0])),
        2 => Ok(TensorShape::D2(arr[0], arr[1])),
        _ => panic!("Tensor has {} dimensions, expected 1 or 2", arr.len()),
    }
}

fn deserialize_pair<'de, D>(deserializer: D) -> std::result::Result<(usize, usize), D::Error>
where
    D: serde::Deserializer<'de>,
{
    let arr: Vec<usize> = Vec::deserialize(deserializer)?;
    Ok((arr[0], arr[1]))
}

/// A representation of the entire model weight set.
///
/// Holds a map from every tensor name (e.g. `"model.layers.0.self_attn.q_a_proj.weight"`)
/// to its metadata as declared in the safetensor files, together with the list of
/// sharded weight files and a file-to-tensors index for efficient loading.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorIndex {
    pub tensors: HashMap<String, TensorMeta>,
    pub weight_files: Vec<PathBuf>,
    /// Maps each weight file to the list of tensor names it contains.
    pub file_tensors: HashMap<PathBuf, Vec<String>>,
    /// Largest weight file size in bytes (computed by root, broadcast to all ranks).
    pub max_file_size: usize,
}

impl TensorIndex {
    /// Build a `TensorIndex` from the model weight directory.
    ///
    /// Reads `model.safetensors.index.json` to discover all weight files,
    /// then reads the JSON header of each safetensor file and builds the
    /// tensor map.
    ///
    /// `weight_dir` must contain both `model.safetensors.index.json` and the
    /// sharded `.safetensors` files.
    pub fn new(weight_dir: &Path) -> candle_core::Result<Self> {
        let index_path = weight_dir.join("model.safetensors.index.json");

        // --- 1. Read the index JSON ---
        tracing::info!("[root] Reading index: {}", index_path.display());
        let index_file = std::fs::File::open(&index_path)?;
        let reader = std::io::BufReader::new(index_file);

        #[derive(Deserialize)]
        struct IndexFile {
            weight_map: HashMap<String, String>,
        }

        let index: IndexFile = serde_json::from_reader(reader).map_err(candle_core::Error::wrap)?;

        // Collect unique weight files and sort for deterministic ordering.
        let mut file_set: Vec<PathBuf> = index
            .weight_map
            .values()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .map(|f| weight_dir.join(f))
            .collect();
        file_set.sort();

        // Build reverse mapping: filename → list of tensor names in that file.
        // We resolve relative filenames to full paths so they match weight_files.
        let mut file_tensors: HashMap<PathBuf, Vec<String>> = HashMap::new();
        for (name, file_rel) in &index.weight_map {
            let full_path = weight_dir.join(file_rel);
            file_tensors
                .entry(full_path)
                .or_default()
                .push(name.clone());
        }

        let mut tensors: HashMap<String, TensorMeta> =
            HashMap::with_capacity(index.weight_map.len());

        for (i, file_path) in file_set.iter().enumerate() {
            tracing::info!(
                "[root] Parsing header  {}/{}  {}",
                i + 1,
                file_set.len(),
                file_path.file_name().unwrap_or_default().to_string_lossy()
            );
            let file = std::fs::File::open(file_path)?;
            let mut reader = std::io::BufReader::new(file);

            // Safetensor format: 8-byte little-endian header size, then JSON header.
            let mut header_size_buf = [0u8; 8];
            reader.read_exact(&mut header_size_buf)?;
            let header_size = u64::from_le_bytes(header_size_buf) as usize;

            let mut header_bytes = vec![0u8; header_size];
            reader.read_exact(&mut header_bytes)?;

            let file_tensors_entries: HashMap<String, SafetensorTensorEntry> =
                serde_json::from_slice(&header_bytes).map_err(candle_core::Error::wrap)?;

            // Convert data_offsets to absolute file offsets so we never
            // need to re-parse the header during weight loading.
            let data_section_start = 8 + header_size;
            for (name, entry) in file_tensors_entries {
                tensors.insert(
                    name,
                    TensorMeta {
                        src_dtype: entry.dtype,
                        shape: entry.shape,
                        data_offsets: (
                            entry.data_offsets.0 + data_section_start,
                            entry.data_offsets.1 + data_section_start,
                        ),
                        dist: Dist::Replicated,
                        buffer: None,
                        dst_dtype: None,
                    },
                );
            }
        }

        tracing::info!(
            "[root] Parsed {} tensors across {} files",
            tensors.len(),
            file_set.len(),
        );

        let max_file_size = file_set
            .iter()
            .filter_map(|p| std::fs::metadata(p).ok())
            .map(|m| m.len() as usize)
            .max()
            .unwrap_or(0);

        Ok(Self {
            tensors,
            weight_files: file_set,
            file_tensors,
            max_file_size,
        })
    }

    /// Broadcast the TensorIndex from root to all ranks via NCCL.
    ///
    /// Root serialises the index with bincode and broadcasts it.
    /// Non-root ranks allocate a GPU buffer, receive the broadcast,
    /// copy back to CPU and deserialise.
    ///
    /// `comm` is the NCCL communicator, `device` the CUDA device.
    #[cfg(feature = "nccl")]
    pub fn bcast(
        comm: &candle_core::cuda_backend::cudarc::nccl::safe::Comm,
        is_root: bool,
    ) -> candle_core::Result<Self> {
        let root_rank = 0;

        if is_root {
            panic!("bcast() called on root — use TensorIndex::new() instead");
        }

        let device = comm.device();

        // --- Phase 1: broadcast the serialised size ---
        let mut size_buf: Vec<u8> = vec![0u8; 8];
        let mut size_gpu = unsafe { device.alloc::<u8>(8) }
            .map_err(|e| candle_core::Error::Msg(format!("CUDA alloc(size): {e:?}")))?;

        comm.broadcast_in_place(&mut size_gpu, root_rank)
            .map_err(|e| candle_core::Error::Msg(format!("NCCL broadcast size: {e:?}")))?;

        device
            .dtoh_sync_copy_into(&size_gpu, &mut size_buf)
            .map_err(|e| candle_core::Error::Msg(format!("CUDA DtoH(size): {e:?}")))?;

        let buf_len = u64::from_le_bytes(size_buf.try_into().unwrap()) as usize;

        // --- Phase 2: broadcast the serialised data ---
        let mut data_host = vec![0u8; buf_len];
        let mut data_gpu = unsafe { device.alloc::<u8>(buf_len) }
            .map_err(|e| candle_core::Error::Msg(format!("CUDA alloc(data): {e:?}")))?;

        comm.broadcast_in_place(&mut data_gpu, root_rank)
            .map_err(|e| candle_core::Error::Msg(format!("NCCL broadcast data: {e:?}")))?;

        device
            .dtoh_sync_copy_into(&data_gpu, &mut data_host)
            .map_err(|e| candle_core::Error::Msg(format!("CUDA DtoH(data): {e:?}")))?;

        let idx: TensorIndex = bincode::deserialize(&data_host)
            .map_err(|e| candle_core::Error::Msg(format!("bincode deserialize: {e}")))?;

        Ok(idx)
    }

    /// Root-side companion: serialises `self` and broadcasts to all ranks.
    #[cfg(feature = "nccl")]
    pub fn bcast_from_root(
        &self,
        comm: &candle_core::cuda_backend::cudarc::nccl::safe::Comm,
    ) -> candle_core::Result<()> {
        tracing::info!("[root] Broadcasting TensorIndex to all ranks ...");

        let root_rank = 0;
        let device = comm.device();
        let data_host = bincode::serialize(self)
            .map_err(|e| candle_core::Error::Msg(format!("bincode serialize: {e}")))?;
        let buf_len = data_host.len();

        // --- Phase 1: broadcast the serialised size ---
        let size_bytes = (buf_len as u64).to_le_bytes().to_vec();
        let mut size_gpu = unsafe { device.alloc::<u8>(8) }
            .map_err(|e| candle_core::Error::Msg(format!("CUDA alloc(size): {e:?}")))?;

        device
            .htod_sync_copy_into(&size_bytes, &mut size_gpu)
            .map_err(|e| candle_core::Error::Msg(format!("CUDA HtoD(size): {e:?}")))?;

        comm.broadcast_in_place(&mut size_gpu, root_rank)
            .map_err(|e| candle_core::Error::Msg(format!("NCCL broadcast size: {e:?}")))?;

        // --- Phase 2: broadcast the serialised data ---
        let mut data_gpu = unsafe { device.alloc::<u8>(buf_len) }
            .map_err(|e| candle_core::Error::Msg(format!("CUDA alloc(data): {e:?}")))?;

        device
            .htod_sync_copy_into(&data_host, &mut data_gpu)
            .map_err(|e| candle_core::Error::Msg(format!("CUDA HtoD(data): {e:?}")))?;

        comm.broadcast_in_place(&mut data_gpu, root_rank)
            .map_err(|e| candle_core::Error::Msg(format!("NCCL broadcast data: {e:?}")))?;

        tracing::info!("[root] TensorIndex broadcast complete.");

        Ok(())
    }

    /// Print all tensor entries to stderr for inspection.
    pub fn print_tensors(&self) {
        let mut names: Vec<&String> = self.tensors.keys().collect();
        names.sort();
        for name in names {
            let meta = &self.tensors[name];
            match meta.shape {
                TensorShape::D1(d) => {
                    eprintln!("  {}  {:6}  [{}]", name, meta.src_dtype, d);
                }
                TensorShape::D2(r, c) => {
                    eprintln!("  {}  {:6}  [{}, {}]", name, meta.src_dtype, r, c);
                }
            }
        }
    }

    // ------------------------------------------------------------------
    //  Helpers for the new alloc constructors
    // ------------------------------------------------------------------

    /// Parse a safetensor dtype string into a candle `DType`.
    pub fn parse_dtype(s: &str) -> Option<DType> {
        match s {
            "F8_E4M3" | "F8E4M3" => Some(DType::F8E4M3),
            "BF16" => Some(DType::BF16),
            "F16" => Some(DType::F16),
            "F32" => Some(DType::F32),
            "F64" => Some(DType::F64),
            "F8_E8M0" | "F8E8M0" => Some(DType::F8E8M0),
            "U8" => Some(DType::U8),
            "U32" => Some(DType::U32),
            "I64" => Some(DType::I64),
            _ => None,
        }
    }

    /// Convert a candle `DType` back to a safetensor-style string (inverse of `parse_dtype`).
    pub fn dtype_to_string(dtype: DType) -> &'static str {
        match dtype {
            DType::F8E4M3 => "F8_E4M3",
            DType::BF16 => "BF16",
            DType::F16 => "F16",
            DType::F32 => "F32",
            DType::F64 => "F64",
            DType::F8E8M0 => "F8_E8M0",
            DType::U8 => "U8",
            DType::U32 => "U32",
            DType::I64 => "I64",
        }
    }

    /// Look up the metadata for a tensor by its full name.
    pub fn meta(&self, name: &str) -> Option<&TensorMeta> {
        self.tensors.get(name)
    }

    /// Register an allocated tensor buffer into the index.
    /// The data-filling pass uses `buffer` to know which GPU tensor to write into.
    pub fn register(&mut self, name: &str, tensor: Tensor, dist: Dist) {
        if let Some(meta) = self.tensors.get_mut(name) {
            meta.dist = dist;
            meta.buffer = Some(tensor);
        }
    }

    /// Register a weight + optional scale tensor pair (for FP8).
    pub fn register_weight_scale(
        &mut self,
        name: &str,
        weight: Tensor,
        scale: Option<Tensor>,
        dist: Dist,
    ) {
        self.register(name, weight, dist);
        if let Some(s) = scale {
            let s_name = format!("{}.weight_scale_inv", name);
            self.register(&s_name, s, Dist::Replicated);
        }
    }

    /// Allocate a zero-initialised tensor whose dtype is taken from the index
    /// metadata, or from `dst_dtype` if set on the TensorMeta.
    /// Does NOT register the tensor — callers should `register()` with the
    /// correct `Dist` after this.
    pub fn alloc_zeros(
        &mut self,
        name: &str,
        local_shape: impl Into<candle_core::Shape>,
        device: &Device,
    ) -> Result<Tensor> {
        let meta = self
            .tensors
            .get(name)
            .ok_or_else(|| candle_core::Error::Msg(format!("TensorIndex: '{}' not found", name)))?;
        let dtype = meta
            .dst_dtype
            .as_deref()
            .and_then(Self::parse_dtype)
            .unwrap_or_else(|| Self::parse_dtype(&meta.src_dtype).expect("unsupported dtype"));
        Tensor::zeros(local_shape, dtype, device)
    }

    /// Allocate a replicated (full-size) weight with an optional bias.
    pub fn alloc_replicated_linear(
        &mut self,
        prefix: &str,
        out_dim: usize,
        in_dim: usize,
        has_bias: bool,
        device: &Device,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let w_name = format!("{}.weight", prefix);
        let weight = self.alloc_zeros(&w_name, (out_dim, in_dim), device)?;
        self.register(&w_name, weight.clone(), Dist::Replicated);
        let bias = if has_bias {
            let b_name = format!("{}.bias", prefix);
            if self.tensors.contains_key(&b_name) {
                let b = self.alloc_zeros(&b_name, out_dim, device)?;
                self.register(&b_name, b.clone(), Dist::Replicated);
                Some(b)
            } else {
                None
            }
        } else {
            None
        };
        Ok((weight, bias))
    }

    /// Allocate a column-sharded weight (shard dim 0).
    pub fn alloc_column_sharded(
        &mut self,
        prefix: &str,
        global_out: usize,
        global_in: usize,
        _iproc: usize,
        nproc: usize,
        device: &Device,
    ) -> Result<Tensor> {
        let w_name = format!("{}.weight", prefix);
        let local_out = global_out / nproc;
        let t = self.alloc_zeros(&w_name, (local_out, global_in), device)?;
        self.register(&w_name, t.clone(), Dist::ColumnSharded);
        Ok(t)
    }

    /// Allocate a row-sharded weight (shard dim 1).
    pub fn alloc_row_sharded(
        &mut self,
        prefix: &str,
        global_out: usize,
        global_in: usize,
        _iproc: usize,
        nproc: usize,
        device: &Device,
    ) -> Result<Tensor> {
        let w_name = format!("{}.weight", prefix);
        let local_in = global_in / nproc;
        let t = self.alloc_zeros(&w_name, (global_out, local_in), device)?;
        self.register(&w_name, t.clone(), Dist::RowSharded);
        Ok(t)
    }

    /// Allocate a weight + optional FP8 scale tensor pair.
    pub fn alloc_weight_scale(
        &mut self,
        name: &str,
        local_shape: impl Into<candle_core::Shape> + Clone,
        block_size: &[usize],
        dist: Dist,
        device: &Device,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let w_name = format!("{}.weight", name);
        let weight = self.alloc_zeros(&w_name, local_shape.clone(), device)?;
        self.register(&w_name, weight.clone(), dist);

        let s_name = format!("{}.weight_scale_inv", name);
        if self.tensors.contains_key(&s_name) {
            let scale = if dist == Dist::Replicated {
                // Replicated: allocate at the global shape from metadata
                // so the full file data fits into the buffer.
                let global_shape = match &self.tensors[&s_name].shape {
                    TensorShape::D2(r, c) => vec![*r, *c],
                    _ => {
                        let s = local_shape.clone().into();
                        s.dims().to_vec()
                    }
                };
                Tensor::zeros(global_shape.as_slice(), DType::F32, device)?
            } else {
                // Sharded: compute local scale dimensions from the local shape.
                assert!(
                    block_size.len() == 2,
                    "alloc_weight_scale '{}': block_size must have 2 elements, got {:?}",
                    name,
                    block_size
                );
                let (local_rows, local_cols) = if block_size.is_empty() {
                    return Ok((weight, None));
                } else {
                    let s = local_shape.into();
                    let d = s.dims();
                    (d[0], d[1])
                };
                let by = block_size[0];
                let bx = block_size[1];
                let sn = (local_rows + by - 1) / by;
                let sk = (local_cols + bx - 1) / bx;
                Tensor::zeros((sn, sk), DType::F32, device)?
            };
            self.register(&s_name, scale.clone(), dist);
            Ok((weight, Some(scale)))
        } else {
            Ok((weight, None))
        }
    }
}

/// Root-only inspection: parses all safetensor headers, prints progress and summary.
/// Non-root ranks never touch the filesystem.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inspect_glm5_fp8() {
        let dir = Path::new("glm-5.2-fp8-config");
        assert!(dir.exists(), "glm-5.2-fp8-config directory not found");
        let idx = TensorIndex::new(dir).unwrap();
        eprintln!("\nTotal tensors: {}", idx.tensors.len());
        eprintln!("Total files: {}\n", idx.weight_files.len());
        idx.print_tensors();
    }
}
