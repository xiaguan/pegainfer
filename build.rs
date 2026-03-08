use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

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
    // To support multi-GPU systems with different SMs, we can build for all detected SMs.
    // Use a BTreeSet to deduplicate and sort SM targets.
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
    // read the CUDA_SM or PEGAINFER_CUDA_SM environment variable for manual override
    if let Ok(env) = std::env::var("PEGAINFER_CUDA_SM").or_else(|_| std::env::var("CUDA_SM")) {
        // Support comma-separated list of SMs in the environment variable.
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

    // Fallback: build for Ampere (bf16 baseline) when GPU detection is unavailable.
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

fn find_triton_python() -> Result<String, String> {
    if let Ok(candidate) = std::env::var("PEGAINFER_TRITON_PYTHON") {
        let candidate = candidate.trim();
        if candidate.is_empty() {
            return Err(
                "PEGAINFER_TRITON_PYTHON is set but empty. See tools/triton/README.md.".to_string(),
            );
        }

        let output = Command::new(candidate)
            .args(["-c", "import triton"])
            .output()
            .map_err(|err| {
                format!(
                    "failed to spawn PEGAINFER_TRITON_PYTHON=`{candidate}`: {err}. See tools/triton/README.md."
                )
            })?;

        if output.status.success() {
            return Ok(candidate.to_string());
        }

        return Err(format!(
            "PEGAINFER_TRITON_PYTHON=`{candidate}` could not import Triton. stdout: {} stderr: {} See tools/triton/README.md.",
            String::from_utf8_lossy(&output.stdout).trim(),
            String::from_utf8_lossy(&output.stderr).trim(),
        ));
    }

    let mut diagnostics = Vec::new();
    for candidate in ["python3", "python"] {
        match Command::new(candidate)
            .args(["-c", "import triton"])
            .output()
        {
            Ok(output) if output.status.success() => return Ok(candidate.to_string()),
            Ok(output) => diagnostics.push(format!(
                "{candidate}: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            )),
            Err(err) => diagnostics.push(format!("{candidate}: {err}")),
        }
    }

    Err(format!(
        "Could not find a Python interpreter with Triton installed. Set PEGAINFER_TRITON_PYTHON or ensure `python3 -c 'import triton'` works. Probe results: {}. See tools/triton/README.md.",
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
            "cargo:warning=Triton AOT currently emits a single cubin for silu_mul; using highest detected target sm_{max_sm}. Set PEGAINFER_CUDA_SM to pin one target explicitly."
        );
    }

    format!("cuda:{max_sm}:32")
}

fn generate_triton_silu_artifacts(out_dir: &Path, triton_target: &str) -> (String, PathBuf) {
    let python = find_triton_python().unwrap_or_else(|message| panic!("{message}"));
    let kernel_path = PathBuf::from("tools/triton/silu_mul_kernel.py");
    let generator_path = PathBuf::from("tools/triton/gen_silu_mul_aot.py");
    let artifact_dir = out_dir.join("triton_aot").join("silu_mul");

    let output = Command::new(&python)
        .arg(&generator_path)
        .arg("--kernel-path")
        .arg(&kernel_path)
        .arg("--out-dir")
        .arg(&artifact_dir)
        .arg("--target")
        .arg(triton_target)
        .output()
        .unwrap_or_else(|err| panic!("failed to run Triton AOT generator for silu_mul: {err}"));

    if !output.status.success() {
        panic!(
            "Triton AOT generator failed. stdout: {} stderr: {}",
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

fn compile_triton_aot_silu(cuda_path: &str, out_dir: &Path, sm_targets: &[String]) {
    let triton_target = triton_target(sm_targets);
    let (func_name, generated_c) = generate_triton_silu_artifacts(out_dir, &triton_target);
    let wrapper_path = generated_c
        .parent()
        .expect("generated Triton source should have a parent directory")
        .join("triton_silu_mul_wrapper.c");
    let wrapper_src = format!(
        "#include <cuda.h>\n#include <stdint.h>\n\nCUresult {func}(CUstream stream, CUdeviceptr gate, CUdeviceptr up, CUdeviceptr out, int32_t n_elements);\n\nCUresult silu_mul_triton_aot_cuda(const uint16_t* gate, const uint16_t* up, uint16_t* out, int n, CUstream stream) {{\n    return {func}(stream, (CUdeviceptr)gate, (CUdeviceptr)up, (CUdeviceptr)out, (int32_t)n);\n}}\n",
        func = func_name
    );
    std::fs::write(&wrapper_path, wrapper_src).expect("failed to write Triton wrapper source");

    cc::Build::new()
        .cuda(false)
        .include(format!("{}/include", cuda_path))
        .flag("-std=c11")
        .warnings(false)
        .file(&generated_c)
        .file(&wrapper_path)
        .compile("triton_kernels_aot");

    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:warning=Using Triton AOT as the default silu_mul path");
    println!("cargo:rerun-if-changed=tools/triton/silu_mul_kernel.py");
    println!("cargo:rerun-if-changed=tools/triton/gen_silu_mul_aot.py");
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

    // Discover all .cu files in csrc/
    let csrc_dir = Path::new("csrc");
    let cu_files: Vec<_> = std::fs::read_dir(csrc_dir)
        .expect("Failed to read csrc/ directory")
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("cu") {
                Some(path)
            } else {
                None
            }
        })
        .collect();

    // Compile each .cu file into a .o
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

    compile_triton_aot_silu(&cuda_path, &out_dir, &sm_targets);

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
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
