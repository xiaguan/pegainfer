use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

struct TritonKernelSpec {
    artifact_dir: &'static str,
    kernel_path: &'static str,
    kernel_name: &'static str,
    signature: &'static str,
    grid: &'static str,
    out_name: &'static str,
    num_warps: u32,
    num_stages: u32,
}

fn parse_sm_token(raw: &str) -> Option<String> {
    let token = raw.trim().trim_matches('"');
    if token.is_empty() {
        return None;
    }

    let token = token
        .strip_prefix("sm_")
        .or_else(|| token.strip_prefix("compute_"))
        .unwrap_or(token);

    if let Some((major, minor)) = token.split_once('.') {
        if major.chars().all(|c| c.is_ascii_digit()) && minor.chars().all(|c| c.is_ascii_digit()) {
            return Some(format!("{}{}", major, minor));
        }
        return None;
    }

    if token.chars().all(|c| c.is_ascii_digit()) {
        if token.len() == 1 {
            return Some(format!("{}0", token));
        }
        return Some(token.to_string());
    }

    None
}

fn sm_targets_from_nvidia_smi() -> Option<Vec<String>> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut sms = BTreeSet::new();
    for line in stdout.lines() {
        let cap = line.split(',').next().unwrap_or(line).trim();
        if let Some(sm) = parse_sm_token(cap) {
            sms.insert(sm);
        }
    }

    if sms.is_empty() {
        None
    } else {
        Some(sms.into_iter().collect())
    }
}

fn detect_sm_targets() -> Vec<String> {
    if let Ok(env) = std::env::var("PEGAINFER_CUDA_SM").or_else(|_| std::env::var("CUDA_SM")) {
        let mut sms = Vec::new();
        for token in env.split(',') {
            if let Some(sm) = parse_sm_token(token) {
                sms.push(sm);
            } else {
                print!(
                    "cargo:warning=Invalid SM token '{}' in CUDA_SM environment variable, skipping.",
                    token
                );
            }
        }
        if !sms.is_empty() {
            return sms;
        }
        print!(
            "cargo:warning=No valid SM tokens found in CUDA_SM environment variable '{}', falling back to auto-detection.",
            env
        );
    }

    if let Some(sms) = sm_targets_from_nvidia_smi() {
        return sms;
    }

    print!(
        "cargo:warning=Failed to detect GPU SMs via nvidia-smi. Set PEGAINFER_CUDA_SM/CUDA_SM environment variable to override."
    );
    panic!("GPU detection failed");
}

fn nvcc_arch_args(sm_targets: &[String]) -> Vec<String> {
    let mut args = Vec::new();
    for sm in sm_targets {
        args.push("-gencode".to_string());
        args.push(format!("arch=compute_{sm},code=sm_{sm}"));
    }

    if let Some(max_sm) = sm_targets
        .iter()
        .filter_map(|sm| sm.parse::<u32>().ok())
        .max()
    {
        args.push("-gencode".to_string());
        args.push(format!("arch=compute_{max_sm},code=compute_{max_sm}"));
    }

    args
}

fn probe_triton_python(candidate: &str) -> Result<String, String> {
    let output = Command::new(candidate)
        .args(["-c", "import triton"])
        .output()
        .map_err(|err| format!("{candidate}: {err}"))?;

    if output.status.success() {
        Ok(candidate.to_string())
    } else {
        Err(format!(
            "{candidate}: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ))
    }
}

fn find_triton_python() -> Result<String, String> {
    if let Ok(candidate) = std::env::var("PEGAINFER_TRITON_PYTHON") {
        let candidate = candidate.trim();
        if candidate.is_empty() {
            return Err(
                "PEGAINFER_TRITON_PYTHON is set but empty. See tools/triton/README.md.".to_string(),
            );
        }
        return probe_triton_python(candidate).map_err(|message| {
            format!(
                "PEGAINFER_TRITON_PYTHON=`{candidate}` could not import Triton. {message}. See tools/triton/README.md."
            )
        });
    }

    let local_venv = PathBuf::from(".venv/bin/python");
    let mut diagnostics = Vec::new();
    let mut candidates = Vec::new();
    if local_venv.exists() {
        candidates.push(local_venv.to_string_lossy().to_string());
    }
    candidates.extend(["python3".to_string(), "python".to_string()]);

    for candidate in candidates {
        match probe_triton_python(&candidate) {
            Ok(path) => return Ok(path),
            Err(message) => diagnostics.push(message),
        }
    }

    Err(format!(
        "Could not find a Python interpreter with Triton installed. Set PEGAINFER_TRITON_PYTHON, bootstrap .venv, or ensure `python3 -c 'import triton'` works. Probe results: {}.",
        diagnostics.join(" | ")
    ))
}

fn triton_target(sm_targets: &[String]) -> String {
    let max_sm = sm_targets
        .iter()
        .filter_map(|sm| sm.parse::<u32>().ok())
        .max()
        .expect("expected at least one CUDA SM target for Triton AOT");

    if sm_targets.len() > 1 {
        println!(
            "cargo:warning=Triton AOT currently emits one cubin per kernel spec; using highest detected target sm_{max_sm}. Set PEGAINFER_CUDA_SM to pin one target explicitly."
        );
    }

    format!("cuda:{max_sm}:32")
}

fn generate_triton_artifacts(
    python: &str,
    out_dir: &Path,
    triton_target: &str,
    spec: &TritonKernelSpec,
) -> (String, PathBuf) {
    let generator_path = PathBuf::from("tools/triton/gen_triton_aot.py");
    let artifact_dir = out_dir.join("triton_aot").join(spec.artifact_dir);

    let output = Command::new(python)
        .arg(&generator_path)
        .arg("--kernel-path")
        .arg(spec.kernel_path)
        .arg("--kernel-name")
        .arg(spec.kernel_name)
        .arg("--signature")
        .arg(spec.signature)
        .arg("--grid")
        .arg(spec.grid)
        .arg("--out-name")
        .arg(spec.out_name)
        .arg("--out-dir")
        .arg(&artifact_dir)
        .arg("--target")
        .arg(triton_target)
        .arg("--num-warps")
        .arg(spec.num_warps.to_string())
        .arg("--num-stages")
        .arg(spec.num_stages.to_string())
        .output()
        .unwrap_or_else(|err| {
            panic!(
                "failed to run Triton AOT generator for {}: {err}",
                spec.kernel_name
            )
        });

    if !output.status.success() {
        panic!(
            "Triton AOT generator failed for {}. stdout: {} stderr: {}",
            spec.kernel_name,
            String::from_utf8_lossy(&output.stdout).trim(),
            String::from_utf8_lossy(&output.stderr).trim(),
        );
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut func_name = None;
    let mut c_path = None;
    for line in stdout.lines() {
        if let Some(value) = line.strip_prefix("FUNC_NAME=") {
            func_name = Some(value.trim().to_string());
        } else if let Some(value) = line.strip_prefix("C_PATH=") {
            c_path = Some(PathBuf::from(value.trim()));
        }
    }

    let func_name = func_name.expect("Triton generator did not print FUNC_NAME");
    let c_path = c_path.expect("Triton generator did not print C_PATH");
    (func_name, c_path)
}

fn write_wrapper(generated_c: &Path, file_name: &str, wrapper_src: String) -> PathBuf {
    let wrapper_path = generated_c
        .parent()
        .expect("generated Triton source should have a parent directory")
        .join(file_name);
    std::fs::write(&wrapper_path, wrapper_src).expect("failed to write Triton wrapper source");
    wrapper_path
}

fn compile_triton_aot_kernels(cuda_path: &str, out_dir: &Path, sm_targets: &[String]) {
    let python = find_triton_python().unwrap_or_else(|message| panic!("{message}"));
    let triton_target = triton_target(sm_targets);
    let mut generated_sources = Vec::new();

    let silu_spec = TritonKernelSpec {
        artifact_dir: "silu_mul",
        kernel_path: "tools/triton/silu_mul_kernel.py",
        kernel_name: "silu_mul_kernel",
        signature: "*bf16,*bf16,*bf16,i32,256",
        grid: "(n_elements + 255) / 256,1,1",
        out_name: "triton_silu_mul",
        num_warps: 4,
        num_stages: 2,
    };
    let (silu_func, silu_c) =
        generate_triton_artifacts(&python, out_dir, &triton_target, &silu_spec);
    let silu_wrapper = write_wrapper(
        &silu_c,
        "triton_silu_mul_wrapper.c",
        format!(
            "#include <cuda.h>\n#include <stdint.h>\n\nCUresult {func}(CUstream stream, CUdeviceptr gate, CUdeviceptr up, CUdeviceptr out, int32_t n_elements);\n\nCUresult silu_mul_triton_aot_cuda(const uint16_t* gate, const uint16_t* up, uint16_t* out, int n, CUstream stream) {{\n    return {func}(stream, (CUdeviceptr)gate, (CUdeviceptr)up, (CUdeviceptr)out, (int32_t)n);\n}}\n",
            func = silu_func
        ),
    );
    generated_sources.push(silu_c);
    generated_sources.push(silu_wrapper);

    let add_spec = TritonKernelSpec {
        artifact_dir: "add",
        kernel_path: "tools/triton/basic_kernels.py",
        kernel_name: "add_kernel",
        signature: "*bf16,*bf16,*bf16,i32,256",
        grid: "(n_elements + 255) / 256,1,1",
        out_name: "triton_add",
        num_warps: 4,
        num_stages: 2,
    };
    let (add_func, add_c) = generate_triton_artifacts(&python, out_dir, &triton_target, &add_spec);
    let add_wrapper = write_wrapper(
        &add_c,
        "triton_add_wrapper.c",
        format!(
            "#include <cuda.h>\n#include <stdint.h>\n\nCUresult {func}(CUstream stream, CUdeviceptr a, CUdeviceptr b, CUdeviceptr out, int32_t n_elements);\n\nCUresult add_cuda(const uint16_t* a, const uint16_t* b, uint16_t* out, int n, CUstream stream) {{\n    return {func}(stream, (CUdeviceptr)a, (CUdeviceptr)b, (CUdeviceptr)out, (int32_t)n);\n}}\n",
            func = add_func
        ),
    );
    generated_sources.push(add_c);
    generated_sources.push(add_wrapper);

    let embedding_spec = TritonKernelSpec {
        artifact_dir: "embedding",
        kernel_path: "tools/triton/basic_kernels.py",
        kernel_name: "embedding_kernel",
        signature: "*bf16,i32,*bf16,i32,256",
        grid: "(hidden_size + 255) / 256,1,1",
        out_name: "triton_embedding",
        num_warps: 4,
        num_stages: 2,
    };
    let (embedding_func, embedding_c) =
        generate_triton_artifacts(&python, out_dir, &triton_target, &embedding_spec);
    let embedding_wrapper = write_wrapper(
        &embedding_c,
        "triton_embedding_wrapper.c",
        format!(
            "#include <cuda.h>\n#include <stdint.h>\n\nCUresult {func}(CUstream stream, CUdeviceptr embed, int32_t token_id, CUdeviceptr out, int32_t hidden_size);\n\nCUresult embedding_cuda(const uint16_t* embed, int token_id, uint16_t* out, int hidden_size, CUstream stream) {{\n    return {func}(stream, (CUdeviceptr)embed, (int32_t)token_id, (CUdeviceptr)out, (int32_t)hidden_size);\n}}\n",
            func = embedding_func
        ),
    );
    generated_sources.push(embedding_c);
    generated_sources.push(embedding_wrapper);

    let embedding_decode_spec = TritonKernelSpec {
        artifact_dir: "embedding_decode",
        kernel_path: "tools/triton/basic_kernels.py",
        kernel_name: "embedding_decode_kernel",
        signature: "*bf16,*i32,*bf16,i32,256",
        grid: "(hidden_size + 255) / 256,1,1",
        out_name: "triton_embedding_decode",
        num_warps: 4,
        num_stages: 2,
    };
    let (embedding_decode_func, embedding_decode_c) =
        generate_triton_artifacts(&python, out_dir, &triton_target, &embedding_decode_spec);
    let embedding_decode_wrapper = write_wrapper(
        &embedding_decode_c,
        "triton_embedding_decode_wrapper.c",
        format!(
            "#include <cuda.h>\n#include <stdint.h>\n\nCUresult {func}(CUstream stream, CUdeviceptr embed, CUdeviceptr decode_meta, CUdeviceptr out, int32_t hidden_size);\n\nCUresult embedding_decode_cuda(const uint16_t* embed, const int* decode_meta, uint16_t* out, int hidden_size, CUstream stream) {{\n    return {func}(stream, (CUdeviceptr)embed, (CUdeviceptr)decode_meta, (CUdeviceptr)out, (int32_t)hidden_size);\n}}\n",
            func = embedding_decode_func
        ),
    );
    generated_sources.push(embedding_decode_c);
    generated_sources.push(embedding_decode_wrapper);

    let embedding_batched_spec = TritonKernelSpec {
        artifact_dir: "embedding_batched",
        kernel_path: "tools/triton/basic_kernels.py",
        kernel_name: "embedding_batched_kernel",
        signature: "*bf16,*i32,*bf16,i32,i32,256",
        grid: "(hidden_size * seq_len + 255) / 256,1,1",
        out_name: "triton_embedding_batched",
        num_warps: 4,
        num_stages: 2,
    };
    let (embedding_batched_func, embedding_batched_c) =
        generate_triton_artifacts(&python, out_dir, &triton_target, &embedding_batched_spec);
    let embedding_batched_wrapper = write_wrapper(
        &embedding_batched_c,
        "triton_embedding_batched_wrapper.c",
        format!(
            "#include <cuda.h>\n#include <stdint.h>\n\nCUresult {func}(CUstream stream, CUdeviceptr embed, CUdeviceptr token_ids, CUdeviceptr out, int32_t hidden_size, int32_t seq_len);\n\nCUresult embedding_batched_cuda(const uint16_t* embed, const int* token_ids, uint16_t* out, int hidden_size, int seq_len, CUstream stream) {{\n    return {func}(stream, (CUdeviceptr)embed, (CUdeviceptr)token_ids, (CUdeviceptr)out, (int32_t)hidden_size, (int32_t)seq_len);\n}}\n",
            func = embedding_batched_func
        ),
    );
    generated_sources.push(embedding_batched_c);
    generated_sources.push(embedding_batched_wrapper);

    // Split-KV attention decode: grid = (num_qheads, NUM_KV_SPLITS=4, 1)
    // Signature: pointers..., scalars..., constexprs: NUM_KV_SPLITS=4, BLOCK_N=64, HEAD_DIM=128
    let attention_decode_spec = TritonKernelSpec {
        artifact_dir: "attention_decode",
        kernel_path: "tools/triton/attention_decode_kernel.py",
        kernel_name: "fused_attention_decode_kernel",
        signature: "*bf16,*bf16,*bf16,*bf16,*bf16,*bf16,*bf16,*i32,*bf16,*bf16,*fp32,*fp32,*fp32,i32,i32,i32,4,64,128",
        grid: "num_qheads,4,1",
        out_name: "triton_attention_decode",
        num_warps: 4,
        num_stages: 2,
    };
    let (attention_decode_func, attention_decode_c) =
        generate_triton_artifacts(&python, out_dir, &triton_target, &attention_decode_spec);
    let attention_decode_wrapper = write_wrapper(
        &attention_decode_c,
        "triton_attention_decode_wrapper.c",
        format!(
            "#include <cuda.h>\n#include <stdint.h>\n\nCUresult {func}(CUstream stream, CUdeviceptr q_full, CUdeviceptr k_full, CUdeviceptr v_full, CUdeviceptr q_norm_weight, CUdeviceptr k_norm_weight, CUdeviceptr cos_cache_base, CUdeviceptr sin_cache_base, CUdeviceptr decode_meta, CUdeviceptr k_cache, CUdeviceptr v_cache, CUdeviceptr partial_out, CUdeviceptr partial_m, CUdeviceptr partial_l, int32_t num_qheads, int32_t num_kvheads, int32_t gqa_ratio);\n\nCUresult fused_gqa_attention_decode(const uint16_t* q_full, const uint16_t* k_full, const uint16_t* v_full, const uint16_t* q_norm_weight, const uint16_t* k_norm_weight, const uint16_t* cos_cache_base, const uint16_t* sin_cache_base, const int32_t* decode_meta, uint16_t* k_cache, uint16_t* v_cache, float* partial_out, float* partial_m, float* partial_l, int32_t num_qheads, int32_t num_kvheads, int32_t gqa_ratio, CUstream stream) {{\n    return {func}(stream, (CUdeviceptr)q_full, (CUdeviceptr)k_full, (CUdeviceptr)v_full, (CUdeviceptr)q_norm_weight, (CUdeviceptr)k_norm_weight, (CUdeviceptr)cos_cache_base, (CUdeviceptr)sin_cache_base, (CUdeviceptr)decode_meta, (CUdeviceptr)k_cache, (CUdeviceptr)v_cache, (CUdeviceptr)partial_out, (CUdeviceptr)partial_m, (CUdeviceptr)partial_l, num_qheads, num_kvheads, gqa_ratio);\n}}\n",
            func = attention_decode_func
        ),
    );
    generated_sources.push(attention_decode_c);
    generated_sources.push(attention_decode_wrapper);

    // FlashAttention-2 prefill kernel: fused QK + softmax + V for all query tokens
    // Grid: (cdiv(seq_len, BLOCK_M=128), num_q_heads, 1)
    // Signature: Q(*bf16), K_cache(*bf16), V_cache(*bf16), Output(*bf16),
    //   num_q_heads(i32), num_kv_heads(i32), gqa_ratio(i32), seq_len(i32), start_pos(i32), q_dim(i32),
    //   constexprs: BLOCK_M=128, BLOCK_N=64, HEAD_DIM=128
    let flash_attn_prefill_spec = TritonKernelSpec {
        artifact_dir: "flash_attention_prefill",
        kernel_path: "tools/triton/flash_attention_prefill_kernel.py",
        kernel_name: "flash_attention_prefill_kernel",
        signature: "*bf16,*bf16,*bf16,*bf16,i32,i32,i32,i32,i32,i32,128,64,128",
        grid: "(seq_len + 127) / 128,num_q_heads,1",
        out_name: "triton_flash_attention_prefill",
        num_warps: 4,
        num_stages: 2,
    };
    let (flash_attn_func, flash_attn_c) =
        generate_triton_artifacts(&python, out_dir, &triton_target, &flash_attn_prefill_spec);
    let flash_attn_wrapper = write_wrapper(
        &flash_attn_c,
        "triton_flash_attention_prefill_wrapper.c",
        format!(
            "#include <cuda.h>\n#include <stdint.h>\n\nCUresult {func}(CUstream stream, CUdeviceptr Q, CUdeviceptr K_cache, CUdeviceptr V_cache, CUdeviceptr Output, int32_t num_q_heads, int32_t num_kv_heads, int32_t gqa_ratio, int32_t seq_len, int32_t start_pos, int32_t q_dim);\n\nCUresult flash_attention_prefill_cuda(const uint16_t* Q, const uint16_t* K_cache, const uint16_t* V_cache, uint16_t* Output, int32_t num_q_heads, int32_t num_kv_heads, int32_t gqa_ratio, int32_t seq_len, int32_t start_pos, int32_t q_dim, CUstream stream) {{\n    return {func}(stream, (CUdeviceptr)Q, (CUdeviceptr)K_cache, (CUdeviceptr)V_cache, (CUdeviceptr)Output, num_q_heads, num_kv_heads, gqa_ratio, seq_len, start_pos, q_dim);\n}}\n",
            func = flash_attn_func
        ),
    );
    generated_sources.push(flash_attn_c);
    generated_sources.push(flash_attn_wrapper);

    // Attention reduce kernel: merges split-KV partials into final output
    let attention_reduce_spec = TritonKernelSpec {
        artifact_dir: "attention_reduce",
        kernel_path: "tools/triton/attention_reduce_kernel.py",
        kernel_name: "attention_reduce_kernel",
        signature: "*fp32,*fp32,*fp32,*bf16,i32,4,128",
        grid: "num_qheads,1,1",
        out_name: "triton_attention_reduce",
        num_warps: 1,
        num_stages: 1,
    };
    let (attention_reduce_func, attention_reduce_c) =
        generate_triton_artifacts(&python, out_dir, &triton_target, &attention_reduce_spec);
    let attention_reduce_wrapper = write_wrapper(
        &attention_reduce_c,
        "triton_attention_reduce_wrapper.c",
        format!(
            "#include <cuda.h>\n#include <stdint.h>\n\nCUresult {func}(CUstream stream, CUdeviceptr partial_out, CUdeviceptr partial_m, CUdeviceptr partial_l, CUdeviceptr output, int32_t num_qheads);\n\nCUresult attention_decode_reduce(float* partial_out, float* partial_m, float* partial_l, uint16_t* output, int32_t num_qheads, CUstream stream) {{\n    return {func}(stream, (CUdeviceptr)partial_out, (CUdeviceptr)partial_m, (CUdeviceptr)partial_l, (CUdeviceptr)output, num_qheads);\n}}\n",
            func = attention_reduce_func
        ),
    );
    generated_sources.push(attention_reduce_c);
    generated_sources.push(attention_reduce_wrapper);

    let mut build = cc::Build::new();
    build
        .cuda(false)
        .include(format!("{}/include", cuda_path))
        .flag("-std=c11")
        .warnings(false);
    for source in &generated_sources {
        build.file(source);
    }
    build.compile("triton_kernels_aot");

    println!("cargo:rustc-link-lib=cuda");
    println!(
        "cargo:warning=Using Triton AOT as the default path for silu_mul, add, embedding, and Qwen3 decode attention; extract/write vector copies now use cudarc device memcpy"
    );
    println!("cargo:rerun-if-changed=tools/triton/attention_decode_kernel.py");
    println!("cargo:rerun-if-changed=tools/triton/attention_reduce_kernel.py");
    println!("cargo:rerun-if-changed=tools/triton/flash_attention_prefill_kernel.py");
    println!("cargo:rerun-if-changed=tools/triton/basic_kernels.py");
    println!("cargo:rerun-if-changed=tools/triton/gen_triton_aot.py");
    println!("cargo:rerun-if-changed=tools/triton/silu_mul_kernel.py");
    println!("cargo:rerun-if-env-changed=PEGAINFER_TRITON_PYTHON");
}

fn main() {
    let cuda_path = std::env::var("CUDA_HOME")
        .or_else(|_| std::env::var("CUDA_PATH"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());

    let nvcc = format!("{}/bin/nvcc", cuda_path);
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let sm_targets = detect_sm_targets();
    let arch_args = nvcc_arch_args(&sm_targets);
    println!(
        "cargo:warning=Compiling CUDA kernels for targets: {}",
        sm_targets
            .iter()
            .map(|sm| format!("sm_{sm}"))
            .collect::<Vec<_>>()
            .join(",")
    );

    let replaced_cuda_files = BTreeSet::from(["activation.cu", "elementwise.cu", "embedding.cu"]);

    let csrc_dir = Path::new("csrc");
    let cu_files: Vec<_> = std::fs::read_dir(csrc_dir)
        .expect("Failed to read csrc/ directory")
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            let file_name = path.file_name()?.to_str()?;
            if path.extension().and_then(|e| e.to_str()) == Some("cu")
                && !replaced_cuda_files.contains(file_name)
            {
                Some(path)
            } else {
                None
            }
        })
        .collect();

    println!(
        "cargo:warning=Legacy CUDA translation units retired from the runtime build: {}",
        replaced_cuda_files
            .iter()
            .copied()
            .collect::<Vec<_>>()
            .join(", ")
    );

    let mut obj_files = Vec::new();
    for cu_file in &cu_files {
        let stem = cu_file.file_stem().unwrap().to_str().unwrap();
        let obj_file = out_dir.join(format!("{}_cuda.o", stem));

        let mut nvcc_args = vec![
            "-c".to_string(),
            cu_file.to_string_lossy().to_string(),
            "-o".to_string(),
            obj_file.to_string_lossy().to_string(),
            "-O3".to_string(),
        ];
        nvcc_args.extend(arch_args.clone());
        nvcc_args.extend(["--compiler-options".to_string(), "-fPIC".to_string()]);

        let status = Command::new(&nvcc)
            .args(&nvcc_args)
            .status()
            .unwrap_or_else(|_| panic!("Failed to run nvcc for {}", cu_file.display()));

        if !status.success() {
            panic!("nvcc compilation failed for {}", cu_file.display());
        }

        obj_files.push(obj_file);
    }

    let cuda_lib = out_dir.join("libkernels_cuda.a");
    let mut ar_args = vec!["rcs".to_string(), cuda_lib.to_string_lossy().to_string()];
    ar_args.extend(
        obj_files
            .into_iter()
            .map(|path| path.to_string_lossy().to_string()),
    );

    let status = Command::new("ar")
        .args(&ar_args)
        .status()
        .expect("Failed to run ar");

    if !status.success() {
        panic!("ar failed");
    }

    compile_triton_aot_kernels(&cuda_path, &out_dir, &sm_targets);

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    if cfg!(target_os = "windows") {
        println!("cargo:rustc-link-search=native={}/lib/x64", cuda_path);
    } else {
        println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    }
    println!("cargo:rustc-link-lib=static=kernels_cuda");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cublas");

    println!("cargo:rerun-if-changed=csrc/");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=PEGAINFER_CUDA_SM");
    println!("cargo:rerun-if-env-changed=CUDA_SM");
}
