//! Safetensors weight loading and RoPE precomputation.

use anyhow::Result;
use cudarc::driver::CudaSlice;
use half::bf16;
use log::info;
use memmap2::Mmap;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::fs;
use std::time::Instant;

use crate::tensor::{DeviceContext, DeviceMatrix, DeviceVec};

/// Load shard metadata. Returns (shard_file_paths, weight_map: tensor_name -> shard_index)
pub fn load_shard_info(model_path: &str) -> Result<(Vec<String>, HashMap<String, usize>)> {
    let single_path = format!("{}/model.safetensors", model_path);
    if std::path::Path::new(&single_path).exists() {
        return Ok((vec![single_path], HashMap::new()));
    }

    let index_path = format!("{}/model.safetensors.index.json", model_path);
    let index_content = fs::read_to_string(&index_path)?;
    let index: serde_json::Value = serde_json::from_str(&index_content)?;

    let weight_map_json = index["weight_map"]
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("Invalid index.json: missing weight_map"))?;

    let mut shard_files: Vec<String> = Vec::new();
    let mut file_to_idx: HashMap<String, usize> = HashMap::new();
    let mut weight_map: HashMap<String, usize> = HashMap::new();

    for (tensor_name, shard_file_val) in weight_map_json {
        let shard_file = shard_file_val.as_str().unwrap().to_string();
        let idx = if let Some(&idx) = file_to_idx.get(&shard_file) {
            idx
        } else {
            let idx = shard_files.len();
            shard_files.push(format!("{}/{}", model_path, &shard_file));
            file_to_idx.insert(shard_file, idx);
            idx
        };
        weight_map.insert(tensor_name.clone(), idx);
    }

    Ok((shard_files, weight_map))
}

/// Memory-map shard files and return the mmaps.
///
/// Typically chained with [`deserialize_shards`] to get `SafeTensors` views:
/// ```ignore
/// let mmaps = mmap_shards(&paths)?;
/// let shards = deserialize_shards(&mmaps)?;
/// ```
pub(crate) fn mmap_shards(shard_paths: &[String]) -> Result<Vec<Mmap>> {
    let t0 = Instant::now();
    let mmaps: Vec<Mmap> = shard_paths
        .iter()
        .map(|p| {
            let file = fs::File::open(p)?;
            // SAFETY: we keep the Mmap alive for the duration of model loading,
            // and the file is not modified concurrently.
            unsafe { Mmap::map(&file) }
        })
        .collect::<std::io::Result<_>>()?;

    let total_bytes: usize = mmaps.iter().map(|m| m.len()).sum();
    info!(
        "Memory-mapped {} shard(s) ({:.1} MB) in {:.0}ms",
        mmaps.len(),
        total_bytes as f64 / 1e6,
        t0.elapsed().as_secs_f64() * 1e3
    );
    Ok(mmaps)
}

/// Deserialize memory-mapped shard data into `SafeTensors` views.
pub(crate) fn deserialize_shards(mmaps: &[Mmap]) -> Result<Vec<SafeTensors<'_>>> {
    mmaps
        .iter()
        .map(|m| {
            SafeTensors::deserialize(m).map_err(|e| anyhow::anyhow!("Deserialize error: {}", e))
        })
        .collect()
}

fn find_tensor<'a>(
    shards: &'a [SafeTensors<'a>],
    weight_map: &HashMap<String, usize>,
    name: &str,
) -> Result<safetensors::tensor::TensorView<'a>> {
    if let Some(&idx) = weight_map.get(name) {
        shards[idx]
            .tensor(name)
            .map_err(|e| anyhow::anyhow!("Failed to load tensor '{}': {}", name, e))
    } else {
        // Fallback: try all shards (single-file case)
        for shard in shards {
            if let Ok(t) = shard.tensor(name) {
                return Ok(t);
            }
        }
        Err(anyhow::anyhow!("Tensor '{}' not found in any shard", name))
    }
}

pub(crate) fn load_tensor_1d(
    ctx: &DeviceContext,
    shards: &[SafeTensors],
    weight_map: &HashMap<String, usize>,
    name: &str,
) -> Result<DeviceVec> {
    let tensor = find_tensor(shards, weight_map, name)?;
    DeviceVec::from_safetensors(ctx, tensor.data())
}

pub(crate) fn load_tensor_2d(
    ctx: &DeviceContext,
    shards: &[SafeTensors],
    weight_map: &HashMap<String, usize>,
    name: &str,
) -> Result<DeviceMatrix> {
    let tensor = find_tensor(shards, weight_map, name)?;
    let shape = tensor.shape();
    DeviceMatrix::from_safetensors(ctx, tensor.data(), shape[0], shape[1])
}

#[allow(clippy::cast_ptr_alignment)]
pub(crate) fn load_tensor_2d_row_shard(
    ctx: &DeviceContext,
    shards: &[SafeTensors],
    weight_map: &HashMap<String, usize>,
    name: &str,
    row_offset: usize,
    rows: usize,
) -> Result<DeviceMatrix> {
    let tensor = find_tensor(shards, weight_map, name)?;
    let shape = tensor.shape();
    if shape.len() != 2 {
        return Err(anyhow::anyhow!(
            "Tensor '{}' expected 2D, got shape {:?}",
            name,
            shape
        ));
    }
    let total_rows = shape[0];
    let cols = shape[1];
    if row_offset + rows > total_rows {
        return Err(anyhow::anyhow!(
            "2D row shard out of bounds for '{}': row_offset={} rows={} total_rows={}",
            name,
            row_offset,
            rows,
            total_rows
        ));
    }
    let data = tensor.data();
    let elems =
        unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<bf16>(), total_rows * cols) };
    let start = row_offset * cols;
    let end = (row_offset + rows) * cols;
    DeviceMatrix::from_host(ctx, &elems[start..end], rows, cols)
}

#[allow(clippy::cast_ptr_alignment)]
pub(crate) fn load_tensor_2d_col_shard(
    ctx: &DeviceContext,
    shards: &[SafeTensors],
    weight_map: &HashMap<String, usize>,
    name: &str,
    col_offset: usize,
    cols: usize,
) -> Result<DeviceMatrix> {
    let tensor = find_tensor(shards, weight_map, name)?;
    let shape = tensor.shape();
    if shape.len() != 2 {
        return Err(anyhow::anyhow!(
            "Tensor '{}' expected 2D, got shape {:?}",
            name,
            shape
        ));
    }
    let rows = shape[0];
    let total_cols = shape[1];
    if col_offset + cols > total_cols {
        return Err(anyhow::anyhow!(
            "2D col shard out of bounds for '{}': col_offset={} cols={} total_cols={}",
            name,
            col_offset,
            cols,
            total_cols
        ));
    }
    let data = tensor.data();
    let elems =
        unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<bf16>(), rows * total_cols) };
    let mut host = vec![bf16::ZERO; rows * cols];
    for row in 0..rows {
        let src = row * total_cols + col_offset;
        let dst = row * cols;
        host[dst..dst + cols].copy_from_slice(&elems[src..src + cols]);
    }
    DeviceMatrix::from_host(ctx, &host, rows, cols)
}

/// Precompute RoPE cos/sin cache as contiguous GPU buffers.
/// Layout: [max_seq_len * head_dim] — position `pos` at offset `pos * head_dim`.
pub(crate) fn precompute_rope(
    ctx: &DeviceContext,
    head_dim: usize,
    max_seq_len: usize,
    theta: f32,
) -> Result<(DeviceVec, DeviceVec)> {
    let half_dim = head_dim / 2;

    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / theta.powf(i as f32 * 2.0 / head_dim as f32))
        .collect();

    let total = max_seq_len * head_dim;
    let mut cos_host = vec![bf16::ZERO; total];
    let mut sin_host = vec![bf16::ZERO; total];

    for pos in 0..max_seq_len {
        let base = pos * head_dim;
        for i in 0..half_dim {
            let freq = pos as f32 * inv_freq[i];
            let cos_val = bf16::from_f32(freq.cos());
            let sin_val = bf16::from_f32(freq.sin());
            // Half-split layout: [cos(0)..cos(63), cos(0)..cos(63)]
            cos_host[base + i] = cos_val;
            cos_host[base + i + half_dim] = cos_val;
            sin_host[base + i] = sin_val;
            sin_host[base + i + half_dim] = sin_val;
        }
    }

    let cos_cache = DeviceVec::from_host(ctx, &cos_host)?;
    let sin_cache = DeviceVec::from_host(ctx, &sin_host)?;

    Ok((cos_cache, sin_cache))
}

/// Precompute YaRN RoPE cos/sin cache as contiguous GPU buffers.
///
/// Matches the transformers `_compute_yarn_parameters` implementation:
///   - Low-frequency dimensions use interpolation (scaled by `factor`).
///   - High-frequency dimensions use extrapolation (original frequencies).
///   - Smooth linear ramp between `beta_fast` and `beta_slow` correction bounds.
///
/// Layout: [max_seq_len * head_dim] — half-split pairs, same as `precompute_rope`.
pub(crate) fn precompute_yarn_rope(
    ctx: &DeviceContext,
    head_dim: usize,
    max_seq_len: usize,
    theta: f32,
    beta_fast: f32,
    beta_slow: f32,
    factor: f32,
    original_max_position_embeddings: usize,
) -> Result<(DeviceVec, DeviceVec)> {
    let half_dim = head_dim / 2;

    // pos_freqs = theta ^ (arange(0, head_dim, 2) / head_dim)
    let pos_freqs: Vec<f32> = (0..half_dim)
        .map(|i| theta.powf((i as f32 * 2.0) / head_dim as f32))
        .collect();

    let inv_freq_extrapolation: Vec<f32> = pos_freqs.iter().map(|&f| 1.0 / f).collect();
    let inv_freq_interpolation: Vec<f32> = pos_freqs.iter().map(|&f| 1.0 / (factor * f)).collect();

    // find_correction_dim from transformers
    let find_correction_dim = |num_rotations: f32| -> f32 {
        let numerator = head_dim as f32
            * ((original_max_position_embeddings as f32)
                / (num_rotations * 2.0 * std::f32::consts::PI))
                .ln();
        let denominator = 2.0 * theta.ln();
        numerator / denominator
    };

    let low_f = find_correction_dim(beta_fast);
    let high_f = find_correction_dim(beta_slow);
    let low = low_f.floor().max(0.0) as usize;
    let high = high_f.ceil().min((head_dim - 1) as f32) as usize;

    // Build blended inv_freq
    let mut inv_freq = vec![0.0f32; half_dim];
    for i in 0..half_dim {
        let ramp = if low == high {
            if i < low { 0.0 } else { 1.0 }
        } else {
            let t = (i as f32 - low as f32) / (high as f32 - low as f32);
            t.clamp(0.0, 1.0)
        };
        // inv_freq_extrapolation_factor = 1 - ramp
        inv_freq[i] = inv_freq_interpolation[i] * ramp + inv_freq_extrapolation[i] * (1.0 - ramp);
    }

    // Build cos/sin cache with half-split layout
    let total = max_seq_len * head_dim;
    let mut cos_host = vec![bf16::ZERO; total];
    let mut sin_host = vec![bf16::ZERO; total];

    for pos in 0..max_seq_len {
        let base = pos * head_dim;
        for i in 0..half_dim {
            let freq = pos as f32 * inv_freq[i];
            let cos_val = bf16::from_f32(freq.cos());
            let sin_val = bf16::from_f32(freq.sin());
            cos_host[base + i] = cos_val;
            cos_host[base + i + half_dim] = cos_val;
            sin_host[base + i] = sin_val;
            sin_host[base + i + half_dim] = sin_val;
        }
    }

    let cos_cache = DeviceVec::from_host(ctx, &cos_host)?;
    let sin_cache = DeviceVec::from_host(ctx, &sin_host)?;
    Ok((cos_cache, sin_cache))
}

#[allow(clippy::cast_ptr_alignment)]
/// Load a 1D F32 tensor to GPU as CudaSlice<f32>.
/// For weights stored in float32 (e.g., A_log, norm.weight in linear attention).
pub(crate) fn load_tensor_1d_f32(
    ctx: &DeviceContext,
    shards: &[SafeTensors],
    weight_map: &HashMap<String, usize>,
    name: &str,
) -> Result<CudaSlice<f32>> {
    let tensor = find_tensor(shards, weight_map, name)?;
    let data = tensor.data();
    if data.len() % 4 != 0 {
        return Err(anyhow::anyhow!(
            "F32 tensor '{}': data length {} not multiple of 4",
            name,
            data.len()
        ));
    }
    let len = data.len() / 4;
    let slice = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<f32>(), len) };
    let gpu_data = ctx
        .stream
        .clone_htod(slice)
        .map_err(|e| anyhow::anyhow!("H2D copy failed for '{}': {}", name, e))?;
    Ok(gpu_data)
}

/// Load shard info with fixup for mismatched shard filenames in index.json.
///
/// Some models (e.g., Qwen3.5) have index.json with shard filenames like
/// `model.safetensors-00001-of-00002.safetensors` while actual files are
/// `model-00001-of-00002.safetensors`. This function detects and fixes that.
pub(crate) fn load_shard_info_fixed(
    model_path: &str,
) -> Result<(Vec<String>, HashMap<String, usize>)> {
    let (mut shard_files, weight_map) = load_shard_info(model_path)?;

    for path in &mut shard_files {
        if !std::path::Path::new(path).exists() {
            // Try replacing "model.safetensors-" with "model-" in filename
            let filename = std::path::Path::new(path)
                .file_name()
                .unwrap()
                .to_str()
                .unwrap();
            if let Some(rest) = filename.strip_prefix("model.safetensors-") {
                let fixed = format!("{}/model-{}", model_path, rest);
                if std::path::Path::new(&fixed).exists() {
                    log::info!(
                        "Fixed shard path: {} -> {}",
                        filename,
                        std::path::Path::new(&fixed)
                            .file_name()
                            .unwrap()
                            .to_str()
                            .unwrap()
                    );
                    *path = fixed;
                    continue;
                }
            }
            return Err(anyhow::anyhow!("Shard file not found: {}", path));
        }
    }

    Ok((shard_files, weight_map))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precompute_yarn_rope_against_transformers() {
        // DSV3.2-like params
        let head_dim = 64;
        let theta = 10000.0f32;
        let beta_fast = 32.0f32;
        let beta_slow = 1.0f32;
        let factor = 40.0f32;
        let orig_max = 4096usize;
        let max_seq_len = 163840usize;

        // We can't run GPU code without a device, but we can test the host math
        // by replicating the frequency computation here.
        let half_dim = head_dim / 2;

        let pos_freqs: Vec<f32> = (0..half_dim)
            .map(|i| theta.powf((i as f32 * 2.0) / head_dim as f32))
            .collect();

        let inv_freq_extrapolation: Vec<f32> = pos_freqs.iter().map(|&f| 1.0 / f).collect();
        let inv_freq_interpolation: Vec<f32> =
            pos_freqs.iter().map(|&f| 1.0 / (factor * f)).collect();

        let find_correction_dim = |num_rotations: f32| -> f32 {
            let numerator = head_dim as f32
                * ((orig_max as f32) / (num_rotations * 2.0 * std::f32::consts::PI)).ln();
            let denominator = 2.0 * theta.ln();
            numerator / denominator
        };

        let low_f = find_correction_dim(beta_fast);
        let high_f = find_correction_dim(beta_slow);
        let low = low_f.floor().max(0.0) as usize;
        let high = high_f.ceil().min((head_dim - 1) as f32) as usize;

        assert_eq!(low, 10);
        assert_eq!(high, 23);

        let mut inv_freq = vec![0.0f32; half_dim];
        for i in 0..half_dim {
            let ramp = if low == high {
                if i < low { 0.0 } else { 1.0 }
            } else {
                let t = (i as f32 - low as f32) / (high as f32 - low as f32);
                t.clamp(0.0, 1.0)
            };
            inv_freq[i] =
                inv_freq_interpolation[i] * ramp + inv_freq_extrapolation[i] * (1.0 - ramp);
        }

        // Expected values from Python/transformers reference
        let expected_inv_freq_first5 = [1.0f32, 0.7498942, 0.56234133, 0.42169651, 0.31622776];
        for (i, &exp) in expected_inv_freq_first5.iter().enumerate() {
            assert!(
                (inv_freq[i] - exp).abs() < 1e-6,
                "inv_freq[{}] = {} != {}",
                i,
                inv_freq[i],
                exp
            );
        }

        let expected_inv_freq_30_31 = [4.4456983e-06f32, 3.3338035e-06];
        assert!((inv_freq[30] - expected_inv_freq_30_31[0]).abs() < 1e-12);
        assert!((inv_freq[31] - expected_inv_freq_30_31[1]).abs() < 1e-12);

        // cos/sin at pos=100 for first 5 elements
        let pos = 100usize;
        let mut cos_host = vec![bf16::ZERO; max_seq_len * head_dim];
        let mut sin_host = vec![bf16::ZERO; max_seq_len * head_dim];
        for p in 0..max_seq_len {
            let base = p * head_dim;
            for i in 0..half_dim {
                let freq = p as f32 * inv_freq[i];
                let cos_val = bf16::from_f32(freq.cos());
                let sin_val = bf16::from_f32(freq.sin());
                cos_host[base + i] = cos_val;
                cos_host[base + i + half_dim] = cos_val;
                sin_host[base + i] = sin_val;
                sin_host[base + i + half_dim] = sin_val;
            }
        }

        let base = pos * head_dim;
        let expected_cos_100 = [0.8623189f32, 0.9175962, 0.9509410, -0.2394990, 0.9786828];
        let expected_sin_100 = [-0.5063657f32, -0.3975137, -0.3093725, -0.9708966, 0.2053776];

        for i in 0..5 {
            let cos_f = cos_host[base + i].to_f32();
            let sin_f = sin_host[base + i].to_f32();
            // bf16 rounding tolerance
            assert!(
                (cos_f - expected_cos_100[i]).abs() < 1e-2,
                "cos[100, {}] = {} != {}",
                i,
                cos_f,
                expected_cos_100[i]
            );
            assert!(
                (sin_f - expected_sin_100[i]).abs() < 1e-2,
                "sin[100, {}] = {} != {}",
                i,
                sin_f,
                expected_sin_100[i]
            );
        }

        // Check half-split symmetry
        for i in 0..half_dim {
            assert_eq!(
                cos_host[base + i],
                cos_host[base + i + half_dim],
                "cos half-split mismatch at i={}",
                i
            );
            assert_eq!(
                sin_host[base + i],
                sin_host[base + i + half_dim],
                "sin half-split mismatch at i={}",
                i
            );
        }
    }
}
