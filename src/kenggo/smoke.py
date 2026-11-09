"""Check the installed CPU research dependencies without a dataset or GPU."""

import json
import platform
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch


def main() -> None:
    values = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    inputs = torch.from_numpy(values).requires_grad_()
    weights = torch.tensor([[2.0], [-1.0]], device="cpu")
    loss = (inputs @ weights).square().sum()
    torch.testing.assert_close(loss, torch.tensor(4.0, device="cpu"))
    loss.backward()
    torch.testing.assert_close(
        inputs.grad, torch.tensor([[0.0, 0.0], [8.0, -4.0]], device="cpu")
    )
    np.testing.assert_array_equal(inputs.detach().numpy(), values)

    table = pa.table({"sample_id": [1, 2], "measurement": [1.5, None]})
    with TemporaryDirectory(prefix="kenggo-smoke-") as directory:
        path = Path(directory) / "sample.parquet"
        pq.write_table(table, path)
        restored = pq.read_table(path)
        if not restored.equals(table, check_metadata=True):
            raise RuntimeError("Parquet round trip changed values, nulls or schema")

    print(
        json.dumps(
            {
                "status": "ok",
                "device": "cpu",
                "python": platform.python_version(),
                "kenggo": version("kenggo"),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "pyarrow": pa.__version__,
                "torch_cuda_build": torch.version.cuda,
                "checks": ["tensor_autograd", "numpy_interop", "parquet_null_roundtrip"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
