import numpy as np
import torch
from termcolor import cprint
from choice_dataset import ChoiceDataset
from joint_dataset import JointDataset


def print_dict_shape(d):
    for key, val in d.items():
        if torch.is_tensor(val):
            print(f'dict.{key}.shape={val.shape}')


# Creates some fake input features, feel free to modify it as you want.
num_users = 10
num_user_features = 128
num_items = 4
num_item_features = 64
num_sessions = 10000

# create observables/features/covariates, the number of parameters are
# arbitrarily chosen.
user_obs = torch.randn(num_users, 128)  # generate 128 features for each user.
item_obs = torch.randn(num_items, 64)
session_obs = torch.randn(num_sessions, 234)
taste_obs = torch.randn(num_users, num_items, 567)
price_obs = torch.randn(num_sessions, num_items, 12)
label = torch.LongTensor(np.random.choice(num_items, size=num_sessions))

user_onehot = torch.zeros(num_sessions, num_users)
user_idx = torch.LongTensor(np.random.choice(num_users, size=num_sessions))
user_onehot[torch.arange(num_sessions), user_idx] = 1

item_availability = torch.ones(num_sessions, num_items).bool()

dataset = ChoiceDataset(
    # pre-specified keywords of __init__
    label=label.long(),  # required.
    # optional:
    user_onehot=user_onehot.long(),
    item_availability=item_availability.bool(),
    # additional keywords of __init__
    user_obs=user_obs,
    item_obs=item_obs,
    session_obs=session_obs,
    taste_obs=taste_obs,
    price_obs=price_obs)


# print(f'{dataset.num_users=:}')
# print(f'{dataset.num_items=:}')
# print(f'{dataset.num_sessions=:}')
# print(f'{len(dataset)=:}')

# clone
# print(dataset.label[:10])
# dataset_cloned = dataset.clone()
# dataset_cloned.label = 99 * torch.ones(num_sessions)
# print(dataset_cloned.label[:10])
# print(dataset.label[:10])  # does not change the original dataset.

# move to device
# print(f'{dataset.device=:}')
# print(f'{dataset.label.device=:}')
# print(f'{dataset.taste_obs.device=:}')
# print(f'{dataset.user_onehot.device=:}')

dataset = dataset.to('cuda')

# print(f'{dataset.device=:}')
# print(f'{dataset.label.device=:}')
# print(f'{dataset.taste_obs.device=:}')
# print(f'{dataset.user_onehot.device=:}')

# create dictionary inputs for model.forward()
print_dict_shape(dataset.x_dict)
cprint('Subsetting test starts', 'yellow')
indices = torch.LongTensor(np.arange(len(dataset)))
# breakpoint()
subset = dataset[indices]

print(subset.label)
print(dataset.label[indices])

subset.label += 1  # modifying the batch does not change the original dataset.

print(subset.label)
print(dataset.label[indices])


print(id(subset.label))
print(id(dataset.label[indices]))

from torch.utils.data.sampler import BatchSampler, SequentialSampler, RandomSampler
shuffle = False
batch_size = 32

sampler = BatchSampler(
    RandomSampler(dataset) if shuffle else SequentialSampler(dataset),
    batch_size=batch_size,
    drop_last=False)

dataloader = torch.utils.data.DataLoader(dataset,
                                         sampler=sampler,
                                         num_workers=0,  # 0 if dataset.device == 'cuda' else os.cpu_count(),
                                         collate_fn=lambda x: x[0],
                                         pin_memory=(dataset.device == 'cpu'))

print(f'{item_obs.shape=:}')
item_obs_all = item_obs.view(1, num_items, -1).expand(num_sessions, -1, -1)
item_obs_all = item_obs_all.to(dataset.device)
label_all = label.to(dataset.device)
print(f'{item_obs_all.shape=:}')

for i, batch in enumerate(dataloader):
    # check consistency.
    first, last = i * batch_size, min(len(dataset), (i + 1) * batch_size)
    idx = torch.arange(first, last)
    assert torch.all(item_obs_all[idx, :, :] == batch.x_dict['item_obs'])
    assert torch.all(label_all[idx] == batch.label)

dataset1 = dataset.clone()
dataset2 = dataset.clone()
joint_dataset = JointDataset(the_dataset=dataset1, another_dataset=dataset2)

cprint('Done', 'green')
