"""
Constructors for joining multiple datasets together.
"""
from typing import Union
import torch


class JointDataset(torch.utils.data.Dataset):
    """A helper class for joinining several pytorch datasets, using JointDataset
    and pytorch data loader allows for sampling the same batch index from several
    datasets.
    """
    def __init__(self, **datasets):
        super(JointDataset, self).__init__()
        self.datasets = datasets
        # check the length of sub-datasets are the same.
        assert len(set([len(d) for d in self.datasets.values()])) == 1

    def __len__(self) -> int:
        return len(self.dataset_list[0])

    def __getitem__(self, indices: Union[int, torch.LongTensor]):
        return tuple(d[indices] for d in self.datasets.values())

    def __repr__(self) -> str:
        out = [f'JointDataset with {len(self.datasets)} sub-datasets: (']
        for name, dataset in self.datasets.items():
            out.append(f'\t{name}: {str(dataset)}')
        out.append(')')
        return '\n'.join(out)
