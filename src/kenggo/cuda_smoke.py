"""Require a working host driver and execute a small synchronized CUDA check."""

import json
import platform
import subprocess
import time

import torch


def main() -> int:
    started, cpu = time.monotonic(), time.process_time()
    result = {
        "status": "failed",
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_cuda_build": torch.version.cuda,
        "gpu_allocation_seconds": 0.0,
        "scope": "Installation readiness only; no model, dataset or throughput measurement.",
        "cost_scope": "Check body after imports; GPU allocation starts at the first CUDA availability probe.",
    }
    allocation_started = None
    try:
        host = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid,name,driver_version,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        result["host"] = {"exit_code": host.returncode, "stdout": host.stdout.strip(),
                          "stderr": host.stderr.strip()}
        if host.returncode:
            raise RuntimeError("Host NVIDIA check failed; repair the driver before CUDA validation")
        if torch.version.cuda is None:
            raise RuntimeError("This PyTorch build has no CUDA support")
        allocation_started = time.monotonic()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; no CPU fallback is permitted")
        torch.cuda.set_device(0)
        torch.cuda.synchronize()
        properties = torch.cuda.get_device_properties(0)
        result["device"] = {"index": 0, "name": properties.name,
                            "capability": list(torch.cuda.get_device_capability(0)),
                            "total_memory_bytes": properties.total_memory}
        inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda", requires_grad=True)
        weights = torch.tensor([[2.0], [-1.0]], device="cuda")
        loss = (inputs @ weights).square().sum()
        loss.backward()
        torch.cuda.synchronize()
        torch.testing.assert_close(loss.detach().cpu(), torch.tensor(4.0))
        torch.testing.assert_close(inputs.grad.cpu(), torch.tensor([[0.0, 0.0], [8.0, -4.0]]))
        result.update(status="ok", checks=["host_driver", "cuda_tensor_autograd", "cuda_synchronize"])
    except (RuntimeError, OSError, subprocess.TimeoutExpired, AssertionError) as error:
        result.update(error_type=type(error).__name__, error=str(error))
    finally:
        if allocation_started is not None:
            result["gpu_allocation_seconds"] = time.monotonic() - allocation_started
        result.update(cpu_seconds=time.process_time() - cpu, wall_seconds=time.monotonic() - started)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
