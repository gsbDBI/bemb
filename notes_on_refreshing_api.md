# Towards BEMB `v1.0`.
We are planning to refine and expand the current API of `BEMBFlex`.
## The `pred_item` and multiple class prediction.
- [ ] Currently the model supports predicting *binary* `batch.label` or multi-class `batch.item_index`. We plan to support arbitrary multi-class classifications.
    * In particular, you don't need to change anything if `pred_item=True`, the model will know the number of classes is exactly the `num_items` parameter. Also, in this case, your `ChoiceDataset` object does **not** need to have a `label` attribute, since the model will look for the `item_index` as the ground truth for training.
    * In contrast, if `pred_item=False`, now you need to supply a `num_classes` to the `BEMBFlex.__init__()` method. Also, you would need a `label` attribute in the `ChoiceDataset` object. The `label` attribute should be a `LongTensor` with values from `{0, 1, ..., num_classes}`.

## Post-Estimation
- [ ] Thanks to feedbacks from our valued users, we are planning to reorganize our post-estimation prediction methods for better user experience.
    - [ ] We will implement a method called `predict_proba()`, the same name as inference methods of scikit-learn models.
    - [ ] This method will have `@torch.no_grad()` as a decorator, so you can use it however you want without being worried about gradient tracking.
    - [ ] With `pred_items = True`, the `batch` needs `item_index` attribute only if it's involved in the utility computation (e.g., within-category computation).
    - [ ] With `pred_items = False,` the `batch` does **not** need to have a `label` attribute.
    - [ ] The preliminary API of `predict_proba()` is used as the following:

```{python}
batch = ChoiceDataset(...)
bemb = BEMBFlex(..., pred_item=True, ...)
proba = bemb.predict_proba(batch)  # shape = (len(batch), num_items)

batch = ChoiceDataset(...)
# not that batch doesn't need to have a label attribute.
bemb = BEMBFlex(..., pred_item=False, num_classes=..., ...)
proba = bemb.predict_proba(batch)  # shape = (len(batch), num_classes)
```


## Renaming Variables.
- [ ] We received feedbacks that the naming of `price`-variation is ambiguous, we propose to change it to `sessionitem`-variation instead (this is precisely the definition of such variables).
