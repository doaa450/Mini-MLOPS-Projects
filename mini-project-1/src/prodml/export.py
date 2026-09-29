import pickle
from pathlib import Path

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from prodml.config import settings


def export_to_onnx(model_path: str | None = None, onnx_path: str | None = None) -> str:
    """Load the pickled (dv, model), export just the regression model to ONNX."""
    model_path = model_path or settings.model_path
    onnx_path = onnx_path or settings.onnx_path

    with open(model_path, "rb") as f_in:
        dv, model = pickle.load(f_in)

    n_features = len(dv.get_feature_names_out())
    initial_type = [("float_input", FloatTensorType([None, n_features]))]

    onnx_model = convert_sklearn(model, initial_types=initial_type)

    Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)
    with open(onnx_path, "wb") as f_out:
        f_out.write(onnx_model.SerializeToString())

    return onnx_path


if __name__ == "__main__":
    path = export_to_onnx()
    print(f"Exported ONNX model to {path}")
