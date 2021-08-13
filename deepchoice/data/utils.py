from typing import Union, List

import pandas as pd
import torch


def pivot3d(df: pd.DataFrame, dim0: str, dim1: str, values: Union[str, List[str]]) -> torch.Tensor:
    """
    Creates a tensor of shape (df[dim0].nunique(), df[dim1].nunique(), len(values)) from the
    provided data frame.

    Example, if dim0 is the column of session ID, dim1 is the column of alternative names, then
        out[t, i, k] is the feature values[k] of item i in session t. The returned tensor
        has shape (num_sessions, num_items, num_params), which fits the purpose of conditioanl
        logit models.
    """
    if not isinstance(values, list):
        values = [values]
    
    dim1_list = sorted(df[dim1].unique())
    
    tensor_slice = list()
    for value in values:
        layer = df.pivot(index=dim0, columns=dim1, values=value)
        tensor_slice.append(torch.Tensor(layer[dim1_list].values))
    
    tensor = torch.stack(tensor_slice, dim=-1)
    assert tensor.shape == (df[dim0].nunique(), df[dim1].nunique(), len(values))
    return tensor
