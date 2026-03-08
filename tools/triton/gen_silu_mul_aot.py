import argparse
from pathlib import Path

import triton
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.driver import ty_to_cpp
from triton.tools.compile import CompileArgs, compile_kernel


class OfflineCudaDriver:
    def __init__(self, target: GPUTarget):
        self._target = target

    def get_current_target(self) -> GPUTarget:
        return self._target

    def map_python_to_cpp_type(self, ty: str) -> str:
        return ty_to_cpp(ty)


def activate_target_driver(target: str | None) -> None:
    if target is None:
        return

    backend, arch, warp_size = target.split(":")
    if backend != "cuda":
        raise ValueError(f"unsupported Triton AOT target backend: {backend}")

    triton.runtime.driver.set_active(
        OfflineCudaDriver(GPUTarget(backend=backend, arch=int(arch), warp_size=int(warp_size)))
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--target", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    activate_target_driver(args.target)

    func_name, output_files = compile_kernel(
        CompileArgs(
            path=args.kernel_path,
            kernel_name="silu_mul_kernel",
            signature="*bf16,*bf16,*bf16,i32,256",
            grid="(n_elements + 255) / 256,1,1",
            num_warps=4,
            num_stages=2,
            out_name="triton_silu_mul",
            out_path=out_dir / "triton_silu_mul",
        )
    )

    c_path = next(path for path in output_files if path.suffix == ".c")

    print(f"FUNC_NAME={func_name}")
    print(f"C_PATH={c_path}")


if __name__ == "__main__":
    main()
