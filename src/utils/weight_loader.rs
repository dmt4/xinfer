use crate::utils::tensor_index::{Dist, TensorIndex, TensorShape};
use candle_core::cuda_backend::cudarc::driver::sys::{CUmemorytype_enum, CUstream, CUDA_MEMCPY2D};
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use candle_core::cuda_backend::cudarc::nccl::safe::Comm;
use candle_core::cuda_backend::CudaStorageSlice;
use candle_core::{DType, Device, Result, Storage, Tensor};
use std::ffi::c_void;
use std::io::Read;
use std::rc::Rc;

/// Pinned (page-locked) host memory buffer, directly accessible by GPU via NCCL.
///
/// Allocated via `cuMemHostAlloc`, freed via `cuMemFreeHost` on drop.
struct PinnedBuffer {
    ptr: *mut u8,
    capacity: usize,
}

// SAFETY: The buffer is owned by this struct.  Ownership transfer between threads
// and shared `&` access are safe because:
// - Send: only one owner at a time.
// - Sync: immutable slices from `&self` are safe to share.
unsafe impl Send for PinnedBuffer {}
unsafe impl Sync for PinnedBuffer {}

impl PinnedBuffer {
    fn new(size: usize) -> Result<Self> {
        if size == 0 {
            return Ok(Self {
                ptr: std::ptr::null_mut(),
                capacity: 0,
            });
        }
        let lib = unsafe { candle_core::cuda_backend::cudarc::driver::sys::lib() };
        let mut host_ptr = std::mem::MaybeUninit::uninit();
        unsafe {
            let err = lib.cuMemHostAlloc(
                host_ptr.as_mut_ptr(),
                size,
                0, // CU_MEMHOSTALLOC_DEFAULT
            );
            use candle_core::cuda_backend::cudarc::driver::sys::cudaError_enum;
            if err != cudaError_enum::CUDA_SUCCESS {
                return Err(candle_core::Error::Msg(format!(
                    "cuMemHostAlloc({}): err={err:?}",
                    size,
                )));
            }
            let ptr = host_ptr.assume_init() as *mut u8;
            Ok(Self {
                ptr,
                capacity: size,
            })
        }
    }

    fn as_slice(&self) -> &[u8] {
        if self.ptr.is_null() {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(self.ptr, self.capacity) }
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        if self.ptr.is_null() {
            return &mut [];
        }
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.capacity) }
    }

    fn as_ptr(&self) -> *const u8 {
        self.ptr as *const u8
    }

    fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }
}

impl Drop for PinnedBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                let lib = candle_core::cuda_backend::cudarc::driver::sys::lib();
                let _ = lib.cuMemFreeHost(self.ptr as *mut std::ffi::c_void);
            }
        }
    }
}

/// Fill pre-allocated GPU buffers with data from safetensor files.
pub fn load_weights(
    ti: &TensorIndex,
    comm: Rc<Comm>,
    device: &Device,
    iproc: usize,
    nproc: usize,
) -> Result<()> {
    let is_root = iproc == 0;
    let root_rank = 0;
    debug_assert_eq!(
        nproc,
        comm.world_size() as usize,
        "load_weights iproc/nproc mismatch with NCCL communicator"
    );
    let _dev = device.as_cuda_device()?;
    let cuda_stream: CUstream = *_dev.cu_stream();

    let max_file_size = ti.max_file_size;

    if max_file_size == 0 {
        return Ok(());
    }

    if is_root {
        tracing::info!(
            "[root] Weight loading: {} files, pinned buffer {} bytes",
            ti.weight_files.len(),
            max_file_size,
        );
    }

    let mut data_buf = PinnedBuffer::new(max_file_size)?;

    for (file_idx, file_path) in ti.weight_files.iter().enumerate() {
        let Some(names) = ti.file_tensors.get(file_path) else {
            continue;
        };
        if names.is_empty() {
            continue;
        }

        let file_label = file_path.file_name().unwrap_or_default().to_string_lossy();
        if is_root {
            tracing::info!(
                "[root] Weights  {}/{}  {}",
                file_idx + 1,
                ti.weight_files.len(),
                file_label,
            );
        }

        {
            let file_len = std::fs::metadata(file_path)?.len() as usize;
            let pinned_slice = &mut data_buf.as_mut_slice()[..file_len];
            if is_root {
                let mut file = std::fs::File::open(file_path).map_err(|e| {
                    candle_core::Error::Msg(format!("open {}: {e}", file_path.display()))
                })?;
                file.read_exact(pinned_slice).map_err(|e| {
                    candle_core::Error::Msg(format!("read {}: {e}", file_path.display()))
                })?;
            }

            use candle_core::cuda_backend::cudarc::nccl::result as nccl_result;
            let send_ptr = if is_root {
                data_buf.as_ptr() as *const std::ffi::c_void
            } else {
                std::ptr::null()
            };
            let recv_ptr = data_buf.as_mut_ptr() as *mut std::ffi::c_void;
            let raw_comm = comm.raw_comm();
            let nccl_stream = *comm.device().cu_stream() as *mut _;
            unsafe {
                nccl_result::broadcast(
                    send_ptr,
                    recv_ptr,
                    max_file_size,
                    candle_core::cuda_backend::cudarc::nccl::sys::ncclDataType_t::ncclUint8,
                    root_rank,
                    raw_comm,
                    nccl_stream,
                )
                .map_err(|e| candle_core::Error::Msg(format!("bcast {}: {e:?}", file_label)))?;
                device.synchronize()?;
            }
        }

        for name in names {
            let Some(meta) = ti.tensors.get(name) else {
                continue;
            };
            let Some(ref gpu_buf) = meta.buffer else {
                continue;
            };

            let dst_shape = gpu_buf.shape();
            let dst_dtype = gpu_buf.dtype();

            let src_dtype = match TensorIndex::parse_dtype(&meta.src_dtype) {
                Some(dt) => dt,
                None => dst_dtype,
            };

            let full_bytes = &data_buf.as_slice()[meta.data_offsets.0..meta.data_offsets.1];

            let fill_result = try_fill_tensor(
                full_bytes,
                gpu_buf,
                dst_shape,
                dst_dtype,
                src_dtype,
                device,
                cuda_stream,
                iproc,
                nproc,
                name,
                &meta,
            );
            if let Err(e) = fill_result {
                let msg = format!(
                    "Error filling tensor '{}' from {} (offset {}..{}): {e}",
                    name, file_label, meta.data_offsets.0, meta.data_offsets.1,
                );
                if is_root {
                    tracing::error!("{msg}");
                }
                return Err(candle_core::Error::Msg(msg).bt());
            }
        }
    }

    if is_root {
        tracing::info!("[root] Weight loading complete.");
    }

    device.synchronize()?;
    Ok(())
}

fn try_fill_tensor(
    full_bytes: &[u8],
    gpu_buf: &Tensor,
    dst_shape: &candle_core::Shape,
    dst_dtype: DType,
    src_dtype: DType,
    device: &Device,
    cuda_stream: CUstream,
    iproc: usize,
    nproc: usize,
    _name: &str,
    meta: &crate::utils::tensor_index::TensorMeta,
) -> Result<()> {
    if src_dtype == dst_dtype {
        return pinned_copy_to_gpu(full_bytes, gpu_buf, meta, iproc, nproc, cuda_stream);
    }

    let maybe_convert = |bytes: &[u8], shape: &[usize]| -> Result<Tensor> {
        let src = Tensor::from_raw_buffer(bytes, src_dtype, shape, device)?;
        src.to_dtype(dst_dtype)
    };

    let src_t = match meta.dist {
        Dist::Replicated => maybe_convert(full_bytes, dst_shape.dims())?,

        Dist::ColumnSharded | Dist::VocabParallel => {
            let dims = dst_shape.dims();
            let local_rows = dims[0];
            let global_dims = match meta.shape {
                TensorShape::D2(r, c) => vec![r, c],
                _ => dims.to_vec(),
            };
            let global_rows = global_dims[0];
            assert!(
                global_rows % nproc == 0,
                "ColumnSharded/VocabParallel '{}': global_rows={} not divisible by nproc={}",
                _name,
                global_rows,
                nproc,
            );
            let src = Tensor::from_raw_buffer(full_bytes, src_dtype, &global_dims, device)?;
            let full = src.to_dtype(dst_dtype)?;

            let rows_per_rank = global_rows / nproc;
            let offset = iproc * rows_per_rank;
            let rows_to_copy = local_rows.min(global_rows - offset);

            if rows_to_copy == 0 {
                Tensor::zeros((local_rows, dims[1]), dst_dtype, device)?
            } else {
                let slice = full.narrow(0, offset, rows_to_copy)?.contiguous()?;
                if rows_to_copy < local_rows {
                    let padding =
                        Tensor::zeros((local_rows - rows_to_copy, dims[1]), dst_dtype, device)?;
                    Tensor::cat(&[&slice, &padding], 0)?
                } else {
                    slice
                }
            }
        }

        Dist::RowSharded => {
            let dims = dst_shape.dims();
            let global_dims = match meta.shape {
                TensorShape::D2(r, c) => vec![r, c],
                _ => dims.to_vec(),
            };
            let local_cols = if dims.len() > 1 { dims[1] } else { 1 };
            let global_cols = global_dims[1];
            assert!(
                global_cols % nproc == 0,
                "RowSharded '{}': global_cols={} not divisible by nproc={}",
                _name,
                global_cols,
                nproc,
            );
            let src = Tensor::from_raw_buffer(full_bytes, src_dtype, &global_dims, device)?;
            let full = src.to_dtype(dst_dtype)?;

            let cols_per_rank = global_cols / nproc;
            let offset = iproc * cols_per_rank;
            let cols_to_copy = local_cols.min(global_cols - offset);

            if cols_to_copy == 0 {
                Tensor::zeros((dims[0], local_cols), dst_dtype, device)?
            } else {
                let slice = full.narrow(1, offset, cols_to_copy)?.contiguous()?;
                if cols_to_copy < local_cols {
                    let padding =
                        Tensor::zeros((dims[0], local_cols - cols_to_copy), dst_dtype, device)?;
                    Tensor::cat(&[&slice, &padding], 1)?
                } else {
                    slice
                }
            }
        }
    };

    copy_device_to_device(&src_t, gpu_buf, cuda_stream).map_err(|e| {
        candle_core::Error::Msg(format!(
            "dtod copy failed for '{}' (src_t shape={:?} dtype={:?}, dst shape={:?} dtype={:?}): {e}",
            _name, src_t.shape(), src_t.dtype(), gpu_buf.shape(), gpu_buf.dtype(),
        ))
    })?;

    device.synchronize()?;
    Ok(())
}

fn pinned_copy_to_gpu(
    full_bytes: &[u8],
    gpu_buf: &Tensor,
    meta: &crate::utils::tensor_index::TensorMeta,
    iproc: usize,
    nproc: usize,
    cuda_stream: CUstream,
) -> Result<()> {
    let dims = gpu_buf.dims();
    let elem_size = gpu_buf.dtype().size_in_bytes();

    let (dst_storage, dst_layout) = gpu_buf.storage_and_layout();
    let Storage::Cuda(cs) = &*dst_storage else {
        return Err(candle_core::Error::Msg(
            "pinned_copy_to_gpu: dst not on CUDA device".into(),
        ));
    };
    let dst_stride = dst_layout.stride();
    let dst_base = cuda_storage_slice_device_ptr(&cs.slice);
    let dst_dev_ptr = dst_base + (dst_layout.start_offset() * elem_size) as u64;

    if dims.len() == 2 {
        let (src_byte_off, height, width_bytes, src_pitch) = match meta.dist {
            Dist::ColumnSharded | Dist::VocabParallel => {
                let rows = dims[0];
                let cols = dims[1];
                let global_rows = match meta.shape {
                    TensorShape::D2(r, _) => r,
                    TensorShape::D1(r) => r,
                };
                assert!(
                    global_rows % nproc == 0,
                    "pinned_copy_to_gpu ColumnSharded/VocabParallel: global_rows={} not divisible by nproc={}",
                    global_rows, nproc,
                );
                let rows_per_rank = global_rows / nproc;
                let src_byte_off = iproc * rows_per_rank * cols * elem_size;
                (src_byte_off, rows, cols * elem_size, cols * elem_size)
            }
            Dist::RowSharded => {
                let rows = dims[0];
                let local_cols = dims[1];
                let global_cols = match meta.shape {
                    TensorShape::D2(_, c) => c,
                    TensorShape::D1(c) => c,
                };
                assert!(
                    global_cols % nproc == 0,
                    "pinned_copy_to_gpu RowSharded: global_cols={} not divisible by nproc={}",
                    global_cols,
                    nproc,
                );
                let cols_per_rank = global_cols / nproc;
                let src_byte_off = iproc * cols_per_rank * elem_size;
                (
                    src_byte_off,
                    rows,
                    local_cols * elem_size,
                    global_cols * elem_size,
                )
            }
            Dist::Replicated => {
                let cols = dims[1];
                (0, dims[0], cols * elem_size, cols * elem_size)
            }
        };

        let src_ptr = unsafe { full_bytes.as_ptr().add(src_byte_off) };

        let copy_params = CUDA_MEMCPY2D {
            srcXInBytes: 0,
            srcY: 0,
            srcMemoryType: CUmemorytype_enum::CU_MEMORYTYPE_HOST,
            srcHost: src_ptr as *const c_void,
            srcDevice: 0,
            srcArray: std::ptr::null_mut(),
            srcPitch: src_pitch,
            dstXInBytes: 0,
            dstY: 0,
            dstMemoryType: CUmemorytype_enum::CU_MEMORYTYPE_DEVICE,
            dstHost: std::ptr::null_mut(),
            dstDevice: dst_dev_ptr,
            dstArray: std::ptr::null_mut(),
            dstPitch: dst_stride[0] * elem_size,
            WidthInBytes: width_bytes,
            Height: height,
        };

        unsafe {
            use candle_core::cuda_backend::cudarc::driver::sys::cudaError_enum;
            let lib = candle_core::cuda_backend::cudarc::driver::sys::lib();
            let err = lib.cuMemcpy2DAsync_v2(&copy_params, cuda_stream);
            if err != cudaError_enum::CUDA_SUCCESS {
                return Err(candle_core::Error::Msg(format!(
                    "cuMemcpy2DAsync_v2 failed (rows={height}, cols={width_bytes}): err={err:?}"
                )));
            }
        }
    } else if dims.len() == 1 {
        let local_count = dims[0];
        let byte_count = local_count * elem_size;
        let offset = match meta.dist {
            Dist::ColumnSharded | Dist::VocabParallel => {
                let global_count = match meta.shape {
                    TensorShape::D1(n) => n,
                    _ => local_count * nproc,
                };
                assert!(
                    global_count % nproc == 0,
                    "pinned_copy_to_gpu 1D ColumnSharded/VocabParallel: global_count={} not divisible by nproc={}",
                    global_count, nproc,
                );
                let per_rank = global_count / nproc;
                iproc * per_rank * elem_size
            }
            Dist::RowSharded => {
                let global_count = match meta.shape {
                    TensorShape::D1(n) => n,
                    _ => local_count * nproc,
                };
                assert!(
                    global_count % nproc == 0,
                    "pinned_copy_to_gpu 1D RowSharded: global_count={} not divisible by nproc={}",
                    global_count,
                    nproc,
                );
                let per_rank = global_count / nproc;
                iproc * per_rank * elem_size
            }
            _ => 0,
        };

        let src_ptr = unsafe { full_bytes.as_ptr().add(offset) };
        unsafe {
            use candle_core::cuda_backend::cudarc::driver::sys::cudaError_enum;
            let lib = candle_core::cuda_backend::cudarc::driver::sys::lib();
            let err = lib.cuMemcpyHtoDAsync_v2(
                dst_dev_ptr,
                src_ptr as *const c_void,
                byte_count,
                cuda_stream,
            );
            if err != cudaError_enum::CUDA_SUCCESS {
                return Err(candle_core::Error::Msg(format!(
                    "cuMemcpyHtoDAsync_v2 failed (count={local_count}): err={err:?}"
                )));
            }
        }
    } else {
        return Err(candle_core::Error::Msg(format!(
            "pinned_copy_to_gpu: unexpected {}D tensor",
            dims.len()
        )));
    }

    Ok(())
}

fn copy_device_to_device(src: &Tensor, dst: &Tensor, cuda_stream: CUstream) -> Result<()> {
    let (src_storage, src_layout) = src.storage_and_layout();
    let Storage::Cuda(src_cs) = &*src_storage else {
        return Err(candle_core::Error::Msg(
            "copy_device_to_device: src not on CUDA".into(),
        ));
    };
    let src_base = cuda_storage_slice_device_ptr(&src_cs.slice);
    let src_ptr = src_base + (src_layout.start_offset() * src.dtype().size_in_bytes()) as u64;

    let (dst_storage, dst_layout) = dst.storage_and_layout();
    let Storage::Cuda(dst_cs) = &*dst_storage else {
        return Err(candle_core::Error::Msg(
            "copy_device_to_device: dst not on CUDA".into(),
        ));
    };
    let dst_base = cuda_storage_slice_device_ptr(&dst_cs.slice);
    let dst_ptr = dst_base + (dst_layout.start_offset() * dst.dtype().size_in_bytes()) as u64;

    let byte_count = src.elem_count() * src.dtype().size_in_bytes();

    unsafe {
        let lib = candle_core::cuda_backend::cudarc::driver::sys::lib();
        let err = lib.cuMemcpyDtoDAsync_v2(dst_ptr, src_ptr, byte_count, cuda_stream);
        if err != candle_core::cuda_backend::cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
            return Err(candle_core::Error::Msg(format!(
                "cuMemcpyDtoDAsync_v2 failed: {err:?}"
            )));
        }
    }
    Ok(())
}

fn cuda_storage_slice_device_ptr(slice: &CudaStorageSlice) -> u64 {
    match slice {
        CudaStorageSlice::U8(s) => *s.device_ptr(),
        CudaStorageSlice::U32(s) => *s.device_ptr(),
        CudaStorageSlice::I64(s) => *s.device_ptr(),
        CudaStorageSlice::BF16(s) => *s.device_ptr(),
        CudaStorageSlice::F16(s) => *s.device_ptr(),
        CudaStorageSlice::F32(s) => *s.device_ptr(),
        CudaStorageSlice::F64(s) => *s.device_ptr(),
    }
}
