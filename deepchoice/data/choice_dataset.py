"""
The dataset class for consumer choice datasets.
"""
import copy
from typing import List, Dict, Optional, Union
import torch


class ChoiceDataset(torch.utils.data.Dataset):
    def __init__(self,
                 label: torch.LongTensor,
                 user_onehot: Optional[torch.LongTensor] = None,
                 item_availability: Optional[torch.BoolTensor] = None,
                 **kwargs) -> None:
        """
        reserved classes: item (i), user (u), session (t)
        """
        super(ChoiceDataset, self).__init__()
        self.label = label
        self.user_onehot = user_onehot
        self.item_availability = item_availability

        self.variable_types = ['user', 'item', 'session', 'taste', 'price']
        for key, item in kwargs.items():
            setattr(self, key, item)

    @staticmethod
    def _dict_index(d, indices) -> dict:
        # subset values of dictionary using the provided index, this method only subsets tensors and
        # keeps other values unchanged.
        subset = dict()
        for key, val in d.items():
            if torch.is_tensor(val):
                subset[key] = val[indices, ...]
            else:
                subset[key] = val
        return subset

    # def __setattr__(self, key, value):
    #     raise NotImplementErorr()
    #     self.__dict__[key] = value
    #     self._is_valid()
    #     # do some sanity check before setting attribute.
    #     # might cause performance issue

    def __getitem__(self, indices: Union[int, torch.LongTensor]):
        # TODO: Do we really need to initialize a new ChoiceDataset object?
        new_dict = dict()

        new_dict['label'] = self.label[indices]

        if self.user_onehot is None:
            new_dict['user_onehot'] = None
        else:
            new_dict['user_onehot'] = self.user_onehot[indices, :]

        if self.item_availability is None:
            new_dict['item_availability'] = None
        else:
            new_dict['item_availability'] = self.item_availability[indices, :]

        for key, val in self.__dict__.items():
            # ignore 'label', 'user_onehot' and 'item_availability' keys, already added.
            if key in new_dict.keys():
                continue
            # for tensors that has the session dimension, subset them.
            if torch.is_tensor(val) and (self._is_session_attribute(key) or self._is_price_attribute(key)):
                new_dict[key] = val[indices]
            else:
                new_dict[key] = val
        return self._from_dict(new_dict)

    def __len__(self) -> int:
        return len(self.label)

    def __contains__(self, key: str) -> bool:
        return key in self.keys

    @property
    def device(self) -> str:
        return self.label.device

    @property
    def num_users(self) -> int:
        for key, val in self.__dict__.items():
            if torch.is_tensor(val):
                if self._is_user_attribute(key) or self._is_taste_attribute(key):
                    return val.shape[0]
        return 1

    @property
    def num_items(self) -> int:
        for key, val in self.__dict__.items():
            if torch.is_tensor(val):
                if self._is_item_attribute(key):
                    return val.shape[0]
                elif self._is_taste_attribute(key) or self._is_price_attribute(key):
                    return val.shape[1]
        return 1

    @property
    def num_sessions(self) -> int:
        return len(self.label)

    def apply_tensor(self, func):
        for key, item in self.__dict__.items():
            if torch.is_tensor(item):
                setattr(self, key, func(item))
            elif isinstance(getattr(self, key), dict):
                for obj_key, obj_item in getattr(self, key).items():
                    if torch.is_tensor(obj_item):
                        setattr(getattr(self, key), obj_key, func(obj_item))
        return self

    def to(self, device):
        return self.apply_tensor(lambda x: x.to(device))

    def _check_device_consistency(self):
        # assert all tensors are on the same device.
        devices = list()
        for val in self.__dict__.values():
            if torch.is_tensor(val):
                devices.append(val.device)
        if len(set(devices)) > 1:
            raise Exception(f'Found tensors on different devices: {set(devices)}.',
                            'Use dataset.to() method to align devices.')

    def clone(self):
        dictionary = {}
        for k, v in self.__dict__.items():
            if torch.is_tensor(v):
                dictionary[k] = v.clone()
            else:
                dictionary[k] = copy.deepcopy(v)
        return self.__class__._from_dict(dictionary)

    @classmethod
    def _from_dict(cls, dictionary: Dict[str, torch.tensor]):
        dataset = cls(**dictionary)
        for key, item in dictionary.items():
            setattr(dataset, key, item)
        return dataset

    def _size_repr(self, value) -> List[int]:
        if torch.is_tensor(value):
            return list(value.size())
        elif isinstance(value, int) or isinstance(value, float):
            return [1]
        elif isinstance(value, list) or isinstance(value, tuple):
            return [len(value)]
        else:
            return []

    def __repr__(self):
        info = [
            f'{key}={self._size_repr(item)}' for key, item in self.__dict__.items()]
        return f"{self.__class__.__name__}({', '.join(info)}, device={self.device})"

    @staticmethod
    def _is_item_attribute(key: str) -> bool:
        return key.startswith('item_') and 'availability' not in key

    @staticmethod
    def _is_user_attribute(key: str) -> bool:
        return key.startswith('user_') and 'onehot' not in key

    @staticmethod
    def _is_session_attribute(key: str) -> bool:
        return key.startswith('session_')

    @staticmethod
    def _is_taste_attribute(key: str) -> bool:
        return key.startswith('taste_')

    @staticmethod
    def _is_price_attribute(key: str) -> bool:
        return key.startswith('price_')

    def _is_attribute(self, key: str) -> bool:
        return self._is_item_attribute(key) \
            or self._is_user_attribute(key) \
            or self._is_session_attribute(key) \
            or self._is_taste_attribute(key) \
            or self._is_price_attribute(key)

    def _is_valid(self):
        r"""
        Check validity.
        """
        raise NotImplementedError
        for key in self.keys:
            if self._is_node_attribute(key):
                if self.num_nodes != self[key].shape[0]:
                    raise ValueError(
                        f"key {key} is not valid, num nodes must equal "
                        "num nodes w/ features."
                    )

    def split(
        self,
        task: str = "node",
        split_ratio: List[float] = None,
        shuffle: bool = True
    ):
        raise NotImplementedError

    def _expand_tensor(self, key: str, val: torch.Tensor) -> torch.Tensor:
        # convert raw tensors into (num_sessions, num_items/num_category, num_params).
        if not self._is_attribute(key):
            # don't expand non-attribute tensors, if any.
            return val

        num_params = val.shape[-1]
        if self._is_user_attribute(key):
            # user_attribute (num_users, *)
            user_idx = torch.nonzero(self.user_onehot, as_tuple=True)[1]
            out = val[user_idx].view(
                self.num_sessions, 1, num_params).expand(-1, self.num_items, -1)
        elif self._is_item_attribute(key):
            # item_attribute (num_items, *)
            out = val.view(1, self.num_items, num_params).expand(
                self.num_sessions, -1, -1)
        elif self._is_session_attribute(key):
            # session_attribute (num_sessions, *)
            out = val.view(self.num_sessions, 1,
                           num_params).expand(-1, self.num_items, -1)
        elif self._is_taste_attribute(key):
            # taste_attribute (num_users, num_items, *)
            user_idx = torch.nonzero(self.user_onehot, as_tuple=True)[1]
            out = val[user_idx, :, :]
        elif self._is_price_attribute(key):
            # price_attribute (num_sessions, num_items, *)
            out = val

        assert out.shape == (self.num_sessions, self.num_items, num_params)
        return out

    @property
    def x_dict(self) -> Dict[object, torch.Tensor]:
        """Get the x_dict object for used in model's forward function."""
        # reshape raw tensors into (num_sessions, num_items/num_category, num_params).
        out = dict()
        for key, val in self.__dict__.items():
            if self._is_attribute(key):
                out[key] = self._expand_tensor(key, val)
        # ENHANCEMENT(Tianyu): cache results, check performance.
        return out

    def variable_group():
        pass
