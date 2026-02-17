use std::path::Path;
use std::process::Command;

fn main() {
    let cuda_path = std::env::var("CUDA_HOME")
        .or_else(|_| std::env::var("CUDA_PATH"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());

    let nvcc = format!("{}/bin/nvcc", cuda_path);
    let out_dir = std::env::var("OUT_DIR").unwrap();

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

        let status = Command::new(&nvcc)
            .args([
                "-c",
                cu_file.to_str().unwrap(),
                "-o",
                &obj_file,
                "-O3",
                "-arch=sm_120",
                "--compiler-options",
                "-fPIC",
            ])
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
}
