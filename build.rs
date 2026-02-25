use std::collections::BTreeSet;
use std::path::Path;
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

fn main() {
    let cuda_path = std::env::var("CUDA_HOME")
        .or_else(|_| std::env::var("CUDA_PATH"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());

    let nvcc = format!("{}/bin/nvcc", cuda_path);
    let out_dir = std::env::var("OUT_DIR").unwrap();
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
        let obj_file = format!("{}/{}_cuda.o", out_dir, stem);

        let mut nvcc_args = vec![
            "-c".to_string(),
            cu_file.to_string_lossy().to_string(),
            "-o".to_string(),
            obj_file.clone(),
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

    // Create static library from all object files
    let cuda_lib = format!("{}/libkernels_cuda.a", out_dir);
    let mut ar_args = vec!["rcs".to_string(), cuda_lib];
    ar_args.extend(obj_files);

    let status = Command::new("ar")
        .args(&ar_args)
        .status()
        .expect("Failed to run ar");

    if !status.success() {
        panic!("ar failed");
    }

    // Link
    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    println!("cargo:rustc-link-lib=static=kernels_cuda");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cublas");

    // Rerun if any CUDA source changes
    println!("cargo:rerun-if-changed=csrc/");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=PEGAINFER_CUDA_SM");
    println!("cargo:rerun-if-env-changed=CUDA_SM");
}
