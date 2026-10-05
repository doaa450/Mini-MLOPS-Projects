import numpy as np
import onnxruntime as ort

from prodml.config import settings
from prodml.data import load_data, split_data
from prodml.features import to_dicts


def test_pickle_onnx_parity():
    """Predictions from the pickle model and the ONNX model must match closely."""
    _, val_df = split_data(load_data())
    sample = val_df.sample(n=500, random_state=settings.random_state)

    import pickle

    with open(settings.model_path, "rb") as f_in:
        dv, model = pickle.load(f_in)

    dicts = to_dicts(sample)
    X = dv.transform(dicts).toarray().astype(np.float32)

    pred_pkl = model.predict(X)

    session = ort.InferenceSession(settings.onnx_path)
    input_name = session.get_inputs()[0].name
    pred_onnx = session.run(None, {input_name: X})[0].flatten()

    assert np.allclose(pred_pkl, pred_onnx, atol=1e-4)
