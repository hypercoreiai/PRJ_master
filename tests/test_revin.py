import pandas as pd
import numpy as np

from normalize.revin import RevinTransform


def test_revin_transform_roundtrip():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
    rt = RevinTransform(feature_columns=["a", "b"])
    out = rt.fit_transform(df)
    assert "a_normalized" in out.columns
    inv = rt.inverse_transform(out)
    # denormalized columns should equal original (approximately)
    assert np.allclose(inv["a_denormalized"].values, df["a"].values)
    assert np.allclose(inv["b_denormalized"].values, df["b"].values)
