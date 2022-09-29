# Tutorial for Bayesian Embedding (BEMB) with Educational Data
Author: Tianyu Du

Date: May. 7, 2022
Update: Sept. 28, 2022

This tutorial helps lab members to deploy the BEMB model on educational question-answering (QA) datasets. We will be using the 17Zuoye data, which is available on Sherlock, throughout this tutorial.

However, this tutorial generalizes to any QA datasets in which each row of the dataset corresponds to a triple (student, question, label). Equivalently, each row of these QA datasets is about a student answering a question correctly/incorrectly.

You can find the executable Jupyter notebook for this tutorial [here](https://github.com/gsbDBI/deepchoice/blob/main/tutorials/education_data/education_data.ipynb)


```python
import os
import numpy as np
import pandas as pd
import torch
from bemb.model import LitBEMBFlex
from bemb.utils.run_helper import run
from sklearn import preprocessing
from torch_choice.data import ChoiceDataset
```

## Load Data

We build some helper functions especially for the Zuoye data for demonstration, you can skip this part if you have your own data ready.
Please see below for the formats data.


```python
def get_all_unique_fields(column, field = 'id'):
    unique_fields = set()
    for tag_list in column:
        for entry in tag_list:
            unique_fields.add(entry[field])
    return list(unique_fields)

def convert_tag_list_into_binary_vector(tag_list, encoder, vec_len):
    index = encoder.transform([x['id'] for x in tag_list])
    out = torch.zeros([vec_len], dtype = torch.float64)
    out[index] = 1
    return out

def convert_column_to_binary_vectors(column):
    all_elements = get_all_unique_fields(column)
    my_encoder = preprocessing.LabelEncoder()
    my_encoder.fit(all_elements)
    out = column.apply(lambda x: convert_tag_list_into_binary_vector(x, my_encoder, len(all_elements)))
    return out
```

If you wish to try this tutorial on the 17Zuoye dataset, which is located at `data_path` on Sherlock.
Please make sure the `data_path` is correct if you are running on your local machine.

Henry prepared these datasets in the `feather` format.
Feather is a portable file format for storing Arrow tables or data frames (from languages like Python or R) that utilizes the Arrow IPC format internally. Feather was created early in the Arrow project as a proof of concept for fast, language-agnostic data frame storage for Python (pandas) and R (see [here](https://arrow.apache.org/docs/python/feather.html) for more information about Feather data format).
You can easily load the data using pandas.


```python
data_path = '/oak/stanford/groups/athey/17Zuoye/bayesian_measurement_17zy/bayes'
# data_path = '/media/tianyudu/Data/Athey/bayes'
```


```python
response_path = os.path.join(data_path, 'exam_response_with_attrib.feather')
attribute_path = os.path.join(data_path, 'exam_response_ques_attrib.feather')
```

### The User-Item and Label Dataset (i.e., The **Response** Dataset)
For the student response use case, the **response** dataset contains at least three columns: `{user_id, item_id, label}`.

Where `user_id` is typically the student's ID, `item_id` is the question's ID, and `label` is the student's response to the question, which is a binary variable indicating whether the student answered the question correctly.

In the `df_resp` dataset loaded below, the `student_id` column corresponds to the `user_id`, the `question_id` column corresponds to the `item_id`, and the `correct` column corresponds to the `label`.

The length of the `df_resp` dataset is the total number of times students answer questions, this corresponds to the number of **purchasing records** following our terminology in the data management tutorial.


```python
df_resp = pd.read_feather(response_path)
print('Number of student-question response pairs:', len(df_resp))
df_resp
```

    Number of student-question response pairs: 8621720





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>student_id</th>
      <th>question_id</th>
      <th>correct</th>
      <th>subject</th>
      <th>grade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>90368</td>
      <td>409</td>
      <td>0</td>
      <td>CHINESE</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>90368</td>
      <td>409</td>
      <td>0</td>
      <td>CHINESE</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>90368</td>
      <td>409</td>
      <td>0</td>
      <td>CHINESE</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>93193</td>
      <td>409</td>
      <td>0</td>
      <td>CHINESE</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>93193</td>
      <td>409</td>
      <td>0</td>
      <td>CHINESE</td>
      <td>2</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>8621715</th>
      <td>115131</td>
      <td>2080</td>
      <td>0</td>
      <td>MATH</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8621716</th>
      <td>83680</td>
      <td>2561</td>
      <td>1</td>
      <td>ENGLISH</td>
      <td>3</td>
    </tr>
    <tr>
      <th>8621717</th>
      <td>83680</td>
      <td>2564</td>
      <td>1</td>
      <td>ENGLISH</td>
      <td>3</td>
    </tr>
    <tr>
      <th>8621718</th>
      <td>83680</td>
      <td>2563</td>
      <td>1</td>
      <td>ENGLISH</td>
      <td>3</td>
    </tr>
    <tr>
      <th>8621719</th>
      <td>83680</td>
      <td>2562</td>
      <td>1</td>
      <td>ENGLISH</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>8621720 rows × 5 columns</p>
</div>



The dataset contains `261,756` students and `3,604` questions. Student IDs are already encoded as integers ranging from  `0` to `261,755`, and question IDs are already encoded as integers ranging from `0` to `3,603`.


```python
print(df_resp['student_id'].nunique())
print(df_resp['question_id'].nunique())
```

    261756
    3604



```python
print(df_resp['student_id'].max())
print(df_resp['question_id'].max())
```

    261755
    3603


### The Attribute Dataset
Researchers can optionally supply a separate **attribute** dataset including observables of users (i.e., students) and items (i.e., questions).

Here we load the `df_attr` dataset, which has length equal to the number of questions. Each row of `df_attr` contains attributes/observables of each question.

Specifically, `df_attr` contains a column called `question_id` and several other columns of attributes.
For each question, we have two attribute as known as `capability` and `knowledge`.


```python
df_attr = pd.read_feather(attribute_path).sort_values('question_id').reset_index(drop=True)
df_attr
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>question_id</th>
      <th>capability</th>
      <th>knowledge</th>
      <th>kp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>[{'id': 'TAG_10100001553832', 'type': 0}, {'id...</td>
      <td>[{'id': '0101001', 'type': 0.0}, {'id': '01020...</td>
      <td>[{'id': 'KP_10100071064944'}, {'id': 'KP_10100...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>[{'id': 'TAG_10100001553832', 'type': 0}, {'id...</td>
      <td>[{'id': '0101001', 'type': 0.0}, {'id': '01020...</td>
      <td>[{'id': 'KP_10100050863402'}, {'id': 'KP_10100...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>[{'id': 'TAG_10100001553832', 'type': 0}, {'id...</td>
      <td>[{'id': '0101001', 'type': 0.0}, {'id': '01020...</td>
      <td>[{'id': 'KP_10100050866393'}]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>[{'id': 'TAG_10100001553832', 'type': 0}, {'id...</td>
      <td>[{'id': '0101001', 'type': 0.0}, {'id': '01020...</td>
      <td>[{'id': 'KP_10100125674593'}, {'id': 'KP_10100...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>[{'id': 'TAG_10100001553832', 'type': 0}, {'id...</td>
      <td>[{'id': '0101001', 'type': 0.0}, {'id': '01020...</td>
      <td>[{'id': 'KP_10100077305590'}, {'id': 'KP_10100...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3599</th>
      <td>3599</td>
      <td>[{'id': 'TAG_10300000827653', 'type': 0}, {'id...</td>
      <td>[{'id': '0301001', 'type': 0.0}, {'id': '03020...</td>
      <td>[{'id': 'KP_10300117105040'}]</td>
    </tr>
    <tr>
      <th>3600</th>
      <td>3600</td>
      <td>[{'id': 'TAG_10300000827653', 'type': 0}, {'id...</td>
      <td>[{'id': '0301001', 'type': 0.0}, {'id': '03020...</td>
      <td>[{'id': 'KP_10300212870515'}]</td>
    </tr>
    <tr>
      <th>3601</th>
      <td>3601</td>
      <td>[{'id': 'TAG_10300000827653', 'type': 0}, {'id...</td>
      <td>[{'id': '0301001', 'type': 0.0}, {'id': '03020...</td>
      <td>[{'id': 'KP_10300111435423'}]</td>
    </tr>
    <tr>
      <th>3602</th>
      <td>3602</td>
      <td>[{'id': 'TAG_10300000827653', 'type': 0}, {'id...</td>
      <td>[{'id': '0301001', 'type': 0.0}, {'id': '03020...</td>
      <td>[{'id': 'KP_10300213265389'}]</td>
    </tr>
    <tr>
      <th>3603</th>
      <td>3603</td>
      <td>[{'id': 'TAG_10300000827653', 'type': 0}, {'id...</td>
      <td>[{'id': '0301001', 'type': 0.0}, {'id': '03020...</td>
      <td>[{'id': 'KP_10300111316448'}]</td>
    </tr>
  </tbody>
</table>
<p>3604 rows × 4 columns</p>
</div>



There are 90 types of capabilities and 34 types of knowledge required by different questions in ths dataset.
We convert these attributes into two binary vectors named `capability_vec` and `knowledge_vec`.

The `capability_vec` vector has shape `(number_of_questions, 90)` and the `knowledge_vec` vector has shape `(number_of_questions, 34)`.
For example, `knowledge_vec[i, j] = 1`  indicates answering question `i` correctly requires type `j` of knowledge.


```python
def f(z):
    # extract knowledge domain.
    return z[-1]['id']

knowledge_domain = [f(x) for x in df_attr['knowledge'].values]

df_attr['capability_vec'] = convert_column_to_binary_vectors(df_attr['capability'])
df_attr['knowledge_vec'] = convert_column_to_binary_vectors(df_attr['knowledge'])

capability_vec = torch.stack(df_attr['capability_vec'].to_list(), dim = 0).float()
knowledge_vec = torch.stack(df_attr['knowledge_vec'].to_list(), dim = 0).float()
```


```python
print(f"{knowledge_vec.shape=:}")
print(f"{capability_vec.shape=:}")
```

    knowledge_vec.shape=torch.Size([3604, 34])
    capability_vec.shape=torch.Size([3604, 90])


Lastly, we concatenate the `capability_vec` and `knowledge_vec` vectors into a single vector called `item_obs` with shape `(number_of_questions, 124)`. This vector encompasses all attributes/observables of items (i.e., questions in this context).


```python
item_obs = torch.cat([capability_vec, knowledge_vec], dim=1)
print(f"{item_obs.shape=:}")
```

    item_obs.shape=torch.Size([3604, 124])


### Construct the `ChoiceDataset` Object
The last step is to construct the `ChoiceDataset` object. The `item_index`(`user_index`) keyword argument holds the identify of question answered (student answering the question) in each student-question response pair respectively. The `label` argument is a binary tensor indicating whether the student answered the question correctly.
Lastly, we put the `item_obs` to capture observables of questions to the dataset.
In this tutorial, we don't have any user observables (i.e., observables of students).


```python
choice_dataset = ChoiceDataset(item_index=torch.LongTensor(df_resp['question_id'].values),
                               user_index=torch.LongTensor(df_resp['student_id'].values),
                               label=torch.LongTensor(df_resp['correct'].values),
                               item_obs=item_obs)
```

    No `session_index` is provided, assume each choice instance is in its own session.


You can print the `choice_dataset` to see information about tensors encompassed.


```python
print(choice_dataset)
```

    ChoiceDataset(label=[8621720], item_index=[8621720], user_index=[8621720], session_index=[8621720], item_availability=[], item_obs=[3604, 124], device=cpu)



```python
num_users = len(torch.unique(choice_dataset.user_index))
num_items = len(torch.unique(choice_dataset.item_index))
num_item_obs = choice_dataset.item_obs.shape[-1]
```

### Splitting Data into Training, Validation, and Testing Sets
To test the generalizability of the model, we split the data into training, validation, and testing sets.
Specifically, we randomly take 80% of student-question pairs as the training set, 10% as the validation set, and the rest 10% as the testing set. 


```python
# randomly permutate the index ranging from (0, 1, ..., len(choice_Dataset) - 1).
idx = np.random.permutation(len(choice_dataset))
# take the first 80% from the random permutation as indices for the training set.
train_size = int(0.8 * len(choice_dataset))
val_size = int(0.1 * len(choice_dataset))
train_idx = idx[:train_size]
val_idx = idx[train_size: train_size + val_size]
test_idx = idx[train_size + val_size:]

# we put train/validation/test datasets into a list.
dataset_list = [choice_dataset[train_idx], choice_dataset[val_idx], choice_dataset[test_idx]]
```


```python
print('[Training dataset]', dataset_list[0])
print('[Validation dataset]', dataset_list[1])
print('[Testing dataset]', dataset_list[2])
```

    [Training dataset] ChoiceDataset(label=[6897376], item_index=[6897376], user_index=[6897376], session_index=[6897376], item_availability=[], item_obs=[3604, 124], device=cpu)
    [Validation dataset] ChoiceDataset(label=[862172], item_index=[862172], user_index=[862172], session_index=[862172], item_availability=[], item_obs=[3604, 124], device=cpu)
    [Testing dataset] ChoiceDataset(label=[862172], item_index=[862172], user_index=[862172], session_index=[862172], item_availability=[], item_obs=[3604, 124], device=cpu)


## Fitting the Model
### One Basic Model
Now let's fit a basic BEMB model to the data. Recall that an `user` $u$ corresponds to a student and an `item` $i$ corresponds to question in this tutorial.

The basic model we will be fitting has utility representation

$$
U_{ui} = \lambda_i + \theta_u^\top \alpha_i
$$

where

$$
\theta_u, \alpha_i \in \mathbb{R}^{10}
$$

The predicted probability for student $u$ to correctly answer question $i$ is

$$
\frac{1}{1 + e^{-U_{ui}}}
$$

**Important**: be sure to set `pred_item=False` below since the model is predicting `choice_dataset.label` instead of `choice_dataset.item` as in traditional consumer choice modeling.


```python
obs2prior_dict = {'lambda_item': False, 'theta_user': False, 'alpha_item': False}
LATENT_DIM = 10
coef_dim_dict = {'lambda_item': 1, 'theta_user': LATENT_DIM, 'alpha_item': LATENT_DIM}

bemb = LitBEMBFlex(
    learning_rate=0.1,
    pred_item=False,
    num_seeds=4,
    utility_formula='lambda_item + theta_user * alpha_item',
    num_users=num_users,
    num_items=num_items,
    obs2prior_dict=obs2prior_dict,
    coef_dim_dict=coef_dim_dict,
    trace_log_q=True,
    num_item_obs=num_item_obs,
    prior_variance=1
)

if torch.cuda.is_available():
    bemb = bemb.to('cuda')

bemb = run(bemb, dataset_list, batch_size=len(choice_dataset) // 20, num_epochs=10)
```

    BEMB: utility formula parsed:
    [{'coefficient': ['lambda_item'], 'observable': None},
     {'coefficient': ['theta_user', 'alpha_item'], 'observable': None}]
    ==================== model received ====================
    Bayesian EMBedding Model with U[user, item, session] = lambda_item + theta_user * alpha_item
    Total number of parameters: 5314408.
    With the following coefficients:
    ModuleDict(
      (lambda_item): BayesianCoefficient(num_classes=3604, dimension=1, prior=N(0, I))
      (theta_user): BayesianCoefficient(num_classes=261756, dimension=10, prior=N(0, I))
      (alpha_item): BayesianCoefficient(num_classes=3604, dimension=10, prior=N(0, I))
    )
    []
    ==================== data set received ====================
    [Training dataset] ChoiceDataset(label=[6897376], item_index=[6897376], user_index=[6897376], session_index=[6897376], item_availability=[], item_obs=[3604, 124], device=cpu)
    [Validation dataset] ChoiceDataset(label=[862172], item_index=[862172], user_index=[862172], session_index=[862172], item_availability=[], item_obs=[3604, 124], device=cpu)
    [Testing dataset] ChoiceDataset(label=[862172], item_index=[862172], user_index=[862172], session_index=[862172], item_availability=[], item_obs=[3604, 124], device=cpu)
    ==================== train the model ====================


    /home/tianyudu/anaconda3/envs/development/lib/python3.10/site-packages/pytorch_lightning/trainer/connectors/accelerator_connector.py:447: LightningDeprecationWarning: Setting `Trainer(gpus=1)` is deprecated in v1.7 and will be removed in v2.0. Please use `Trainer(accelerator='gpu', devices=1)` instead.
      rank_zero_deprecation(
    GPU available: True (cuda), used: True
    TPU available: False, using: 0 TPU cores
    IPU available: False, using: 0 IPUs
    HPU available: False, using: 0 HPUs
    LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    
      | Name  | Type     | Params
    -----------------------------------
    0 | model | BEMBFlex | 5.3 M 
    -----------------------------------
    5.3 M     Trainable params
    0         Non-trainable params
    5.3 M     Total params
    21.258    Total estimated model params size (MB)



    Sanity Checking: 0it [00:00, ?it/s]



    Training: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]


    `Trainer.fit` stopped: `max_epochs=10` reached.


    time taken: 54.69776630401611
    ==================== test performance ====================


    LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]



    Testing: 0it [00:00, ?it/s]


    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
           Test metric             DataLoader 0
    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
            test_acc            0.8299005302886199
            test_auc             0.855947006804546
             test_f1             0.897520421366994
             test_ll           -0.36410953061998036
            test_mse            0.1176305070796812
           test_mse_se         0.0009547248708845476
         test_precision         0.8541084351886367
           test_recall          0.9455835324033649
    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────


### Leveraging More Complex Utility Representations
Let's add the item-observable measuring capacities and knowledge required by answering each question to the utility representation.

$$
U_{ui} = \lambda_i + \theta_u^\top \alpha_i + \eta_u^\top X^{(item\_obs)}_i
$$

where

$$
\theta_u, \alpha_i \in \mathbb{R}^{10}
$$

and

$$
\eta_u, X^{(item\_obs)}_i \in \mathbb{R}^{124}
$$


```python
obs2prior_dict = {'lambda_item': False, 'theta_user': False, 'alpha_item': False, 'eta_user': False}
LATENT_DIM = 10
coef_dim_dict = {'lambda_item': 1, 'theta_user': LATENT_DIM, 'alpha_item': LATENT_DIM, 'eta_user': num_item_obs}

bemb = LitBEMBFlex(
    # trainings args.
    learning_rate=0.1,
    pred_item=False,
    num_seeds=4,
    # model args, will be passed to BEMB constructor.
    utility_formula='lambda_item + theta_user * alpha_item + eta_user * item_obs',
    num_users=num_users,
    num_items=num_items,
    obs2prior_dict=obs2prior_dict,
    coef_dim_dict=coef_dim_dict,
    trace_log_q=True,
    num_item_obs=num_item_obs,
    prior_variance=1
)


if torch.cuda.is_available():
    bemb = bemb.to('cuda')

bemb = run(bemb, dataset_list, batch_size=len(choice_dataset) // 20, num_epochs=10)
```

    BEMB: utility formula parsed:
    [{'coefficient': ['lambda_item'], 'observable': None},
     {'coefficient': ['theta_user', 'alpha_item'], 'observable': None},
     {'coefficient': ['eta_user'], 'observable': 'item_obs'}]
    ==================== model received ====================
    Bayesian EMBedding Model with U[user, item, session] = lambda_item + theta_user * alpha_item + eta_user * item_obs
    Total number of parameters: 70229896.
    With the following coefficients:
    ModuleDict(
      (lambda_item): BayesianCoefficient(num_classes=3604, dimension=1, prior=N(0, I))
      (theta_user): BayesianCoefficient(num_classes=261756, dimension=10, prior=N(0, I))
      (alpha_item): BayesianCoefficient(num_classes=3604, dimension=10, prior=N(0, I))
      (eta_user): BayesianCoefficient(num_classes=261756, dimension=124, prior=N(0, I))
    )
    []
    ==================== data set received ====================
    [Training dataset] ChoiceDataset(label=[6897376], item_index=[6897376], user_index=[6897376], session_index=[6897376], item_availability=[], item_obs=[3604, 124], device=cpu)
    [Validation dataset] ChoiceDataset(label=[862172], item_index=[862172], user_index=[862172], session_index=[862172], item_availability=[], item_obs=[3604, 124], device=cpu)
    [Testing dataset] ChoiceDataset(label=[862172], item_index=[862172], user_index=[862172], session_index=[862172], item_availability=[], item_obs=[3604, 124], device=cpu)
    ==================== train the model ====================


    /home/tianyudu/anaconda3/envs/development/lib/python3.10/site-packages/pytorch_lightning/trainer/connectors/accelerator_connector.py:447: LightningDeprecationWarning: Setting `Trainer(gpus=1)` is deprecated in v1.7 and will be removed in v2.0. Please use `Trainer(accelerator='gpu', devices=1)` instead.
      rank_zero_deprecation(
    GPU available: True (cuda), used: True
    TPU available: False, using: 0 TPU cores
    IPU available: False, using: 0 IPUs
    HPU available: False, using: 0 HPUs
    LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    
      | Name  | Type     | Params
    -----------------------------------
    0 | model | BEMBFlex | 70.2 M
    -----------------------------------
    70.2 M    Trainable params
    0         Non-trainable params
    70.2 M    Total params
    280.920   Total estimated model params size (MB)



    Sanity Checking: 0it [00:00, ?it/s]



    Training: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]


    `Trainer.fit` stopped: `max_epochs=10` reached.


    time taken: 112.26553511619568
    ==================== test performance ====================


    LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]



    Testing: 0it [00:00, ?it/s]


    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
           Test metric             DataLoader 0
    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
            test_acc            0.8517094036920707
            test_auc             0.885204845799722
             test_f1            0.9099669643997097
             test_ll            -0.3426648870818855
            test_mse            0.10709758663938111
           test_mse_se         0.0010785000066356716
         test_precision          0.872055564772127
           test_recall          0.9513266710617021
    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────


### Leveraging `obs2prior`
In both examples above, the prior of all coefficients were standard Gaussian distributions.

We can improve the model by incorporating the `obs2prior` option and let the mean of prior distribution of item-specific coefficients (i.e., $\lambda_i$ and $\alpha_i$) depend on item observables.

One can turn on the `obs2prior` option easily by setting `obs2prior_dict['lambda_item'] = True` and `obs2prior_dict['alpha_item'] = True`.

**Important**: we recommend to set a small `prior_variance` to make `obs2prior` more effective. For example, if one set `prior_variance=`$\infty$, prior distributions do not matter at all to the optimization, and the `obs2prior` will be ineffectively as a result.

$$
U_{ui} = \lambda_i + \theta_u^\top \alpha_i
$$

where

$$
\theta_u, \alpha_i \in \mathbb{R}^{10}
$$


```python
obs2prior_dict = {'lambda_item': True, 'theta_user': False, 'alpha_item': True, 'eta_user': False}
LATENT_DIM = 10
coef_dim_dict = {'lambda_user': 1, 'lambda_item': 1, 'theta_user': LATENT_DIM, 'alpha_item': LATENT_DIM, 'eta_user': num_item_obs}

bemb = LitBEMBFlex(
    # trainings args.
    learning_rate=0.1,
    pred_item=False,
    num_seeds=4,
    # model args, will be passed to BEMB constructor.
    utility_formula='lambda_item + theta_user * alpha_item + eta_user * item_obs',
    num_users=num_users,
    num_items=num_items,
    obs2prior_dict=obs2prior_dict,
    coef_dim_dict=coef_dim_dict,
    trace_log_q=True,
    num_item_obs=num_item_obs,
    prior_variance=0.01
)

if torch.cuda.is_available():
    bemb = bemb.to('cuda')
   
bemb = run(bemb, dataset_list, batch_size=len(choice_dataset) // 20, num_epochs=50)
```

    BEMB: utility formula parsed:
    [{'coefficient': ['lambda_item'], 'observable': None},
     {'coefficient': ['theta_user', 'alpha_item'], 'observable': None},
     {'coefficient': ['eta_user'], 'observable': 'item_obs'}]
    ==================== model received ====================
    Bayesian EMBedding Model with U[user, item, session] = lambda_item + theta_user * alpha_item + eta_user * item_obs
    Total number of parameters: 70232624.
    With the following coefficients:
    ModuleDict(
      (lambda_item): BayesianCoefficient(num_classes=3604, dimension=1, prior=N(H*X_obs(H shape=torch.Size([1, 124]), X_obs shape=124), Ix0.01))
      (theta_user): BayesianCoefficient(num_classes=261756, dimension=10, prior=N(0, I))
      (alpha_item): BayesianCoefficient(num_classes=3604, dimension=10, prior=N(H*X_obs(H shape=torch.Size([10, 124]), X_obs shape=124), Ix0.01))
      (eta_user): BayesianCoefficient(num_classes=261756, dimension=124, prior=N(0, I))
    )
    []
    ==================== data set received ====================
    [Training dataset] ChoiceDataset(label=[6897376], item_index=[6897376], user_index=[6897376], session_index=[6897376], item_availability=[], item_obs=[3604, 124], device=cpu)
    [Validation dataset] ChoiceDataset(label=[862172], item_index=[862172], user_index=[862172], session_index=[862172], item_availability=[], item_obs=[3604, 124], device=cpu)
    [Testing dataset] ChoiceDataset(label=[862172], item_index=[862172], user_index=[862172], session_index=[862172], item_availability=[], item_obs=[3604, 124], device=cpu)
    ==================== train the model ====================


    /home/tianyudu/anaconda3/envs/development/lib/python3.10/site-packages/pytorch_lightning/trainer/connectors/accelerator_connector.py:447: LightningDeprecationWarning: Setting `Trainer(gpus=1)` is deprecated in v1.7 and will be removed in v2.0. Please use `Trainer(accelerator='gpu', devices=1)` instead.
      rank_zero_deprecation(
    GPU available: True (cuda), used: True
    TPU available: False, using: 0 TPU cores
    IPU available: False, using: 0 IPUs
    HPU available: False, using: 0 HPUs
    LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    
      | Name  | Type     | Params
    -----------------------------------
    0 | model | BEMBFlex | 70.2 M
    -----------------------------------
    70.2 M    Trainable params
    0         Non-trainable params
    70.2 M    Total params
    280.930   Total estimated model params size (MB)



    Sanity Checking: 0it [00:00, ?it/s]



    Training: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]


    `Trainer.fit` stopped: `max_epochs=50` reached.


    time taken: 562.2096726894379
    ==================== test performance ====================


    LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]



    Testing: 0it [00:00, ?it/s]


    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
           Test metric             DataLoader 0
    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
            test_acc            0.8202945583943807
            test_auc            0.8306907476885476
             test_f1            0.8939502167060785
             test_ll           -0.39504207835551197
            test_mse            0.12611882235908273
           test_mse_se         0.0009444419977889863
         test_precision         0.8352596597377366
           test_recall          0.9615144020585291
    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────


### Tuning the Model
There are tons of parameters in models above, for example, we choose `LATENT_DIM = 10` based on our own experience. However, these choices of hyper-parameters can be sub-optimal.

We recommend researchers to try out different combinations of hyper-parameters before sticking with a particular hyper-parameter configuration.

We will be providing a script for effectively parameter tuning though the `learning-tool-competition` project.
