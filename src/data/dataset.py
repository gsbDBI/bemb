"""
Default data structure for a consumer choice problem dataset.
"""
from typing import List, Union, Optional, Dict

import torch

# VAR_TYPES = ['u', 'i', 'ui', 't', 'ut', 'it', 'uit']


# def collate_fn(sample_list):
#     """Collate function for the special (X, y) tuple returned by CMDataset.__getitem__()
#     Expected input format:
#     [... ((X_dict, user_onehot, A, C), Y) ...]
#     """
#     x_cat = dict()
#     for key in VAR_TYPES:
#         if sample_list[0][0][0][key] is None:
#             x_cat[key] = None
#         else:
#             x_cat[key] = torch.cat([sample[0][0][key] for sample in sample_list], dim=0)
    
#     user_onehot_cat = torch.cat([sample[0][1] for sample in sample_list], dim=0)
#     A_cat = torch.cat([sample[0][2] for sample in sample_list], dim=0)
#     # list of 0 dimensional tensors.
#     C_cat = torch.stack([sample[0][3] for sample in sample_list])
#     Y_cat = torch.stack([sample[1] for sample in sample_list])
#     return (x_cat, user_onehot_cat, A_cat, C_cat), Y_cat


class CMDataset(torch.utils.data.Dataset):
    """Choice modelling dataset"""
    def __init__(self,
                 path: Optional[str] = None,
                 X: Optional[Dict[str, torch.Tensor]] = None,
                 user_onehot: Optional[torch.LongTensor] = None,
                 A: Optional[torch.BoolTensor] = None,
                 Y: Optional[torch.LongTensor] = None,
                 C: Optional[torch.LongTensor] = None,
                 device: str = 'cpu'
                 ) -> None:
        """Constructs the dataloader, reads dataset from disk directly if `path` is provided.
        Otherwise, the provided X, user_onehot, etc, tensors will be used.

        Args:
            path (Optional[str], optional): the path of dataset. Defaults to None.

            X (Optional[Dict[str, torch.Tensor]]): a dictionary with keys from variable types, i.e.,
                ['u', 'i', 'ui', 't', 'ut', 'it', 'uit']
                and have tensors with shape (num_sessions, num_items, num_params).
                num_sessions and num_items are consistent over all keys and values, but num_params
                may vary.
                For example, X['u'][session_id, item_id, :] contains user-specific features of the
                user involved in session `session_id`, which will conribute to the utility of
                item `item_id`. Since the feature is user-specific and does not vary across items,
                X['u'][session_id, item_id_1, :] == X['i'][session_id, item_id_2, :].
                However, for user-item-specific feature in X['ui'], this equality does not hold.
                Defaults to None.

            user_onehot (Optional[torch.LongTensor], optional): A long tensor with shape
                (num_sessions, num_users), in which each row is a one-hot encoding of the ID of the
                user who was making purchasing decision in this session.
                Defaults to None.

            A (Optional[torch.BoolTensor], optional): A boolean tensor with shape
                (num_sessions, num_items) indicating the aviliability of each item in each session.
                Defaults to None.

            Y (Optional[torch.LongTensor], optional): A long tensor with shape (num_sessions,) and
                takes values from {0, 1, ..., num_items-1} indicating which item was purchased in
                each shopping session.
                Defaults to None.

            C (Optional[torch.LongTensor], optional): A long tensor with shape (num_sessions,) and
                takes values from {0, 1, ..., num_categories-1} indicating which category the item
                bought in that session belongs to.
                Defaults to None.
            
            device (str): location to store the entire dataset. Defaults to 'cpu'.
        """
        self.device = device
        if path is not None:
            self._load_from_path(path)
        else:
            self.X = X
            for k, v in self.X.items():
                if v is not None:
                    self.X[k] = v.to(self.device)
            
            self.user_onehot = user_onehot.to(self.device)
            self.A = A.to(self.device)
            self.Y = Y.to(self.device)
            self.C = C.to(self.device)

            self.num_sessions = len(self.Y)
            self.num_users = self.user_onehot.shape[1]
            self.num_items = self.A.shape[1]
            assert torch.max(self.Y) + 1 == self.num_items
            self.num_categories = torch.max(self.C) + 1

    def __getitem__(self, idx: Union[int, List[int]]):
        batch_size = 1 if isinstance(idx, int) else len(idx)
        x_row = dict()
        for key, val in self.X.items():
            # iterating through different types of variables.
            if val is None:
                x_row[key] = None
            else:
                x_row[key] = val[idx, :, :].view(batch_size, self.num_items, -1)  # (batch_size, num_items, num_params)
        # user onehot, raw show (num_sessions, num_users)
        U = self.user_onehot[idx, :].view(batch_size, self.num_users)
        # item aviliability, raw shape (num_sessions, num_items)
        A = self.A[idx, :].view(batch_size, self.num_items)
        # item category, raw shape (num_sessions,)
        C = self.C[idx]
        # purchase choice, raw shape (num_sessions,)
        Y = self.Y[idx]
        
        return (x_row, U, A, C), Y

    def __len__(self) -> int:
        return self.num_sessions

    def _load_from_path(self, path: str):
        """Load data from disk directly."""
        user_obs = ...  # (num_users, *)
        item_obs = ...  # (num_items, *)
        raise NotImplementedError
