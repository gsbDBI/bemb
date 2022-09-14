"""
This script converts the `feather` format 17Zuoye data to the `ChoiceDataset` format.
"""
import pandas as pd
import torch
from sklearn import preprocessing
from torch_choice.data import ChoiceDataset


def get_all_unique_fields(column, field='id'):
    unique_fields = set()
    for tag_list in column:
        for entry in tag_list:
            unique_fields.add(entry[field])
    return list(unique_fields)


def convert_tag_list_into_binary_vector(tag_list, encoder, vec_len):
    index = encoder.transform([x['id'] for x in tag_list])
    out = torch.zeros([vec_len], dtype=torch.float64)
    out[index] = 1
    return out


def convert_column_to_binary_vectors(column):
    all_elements = get_all_unique_fields(column)
    my_encoder = preprocessing.LabelEncoder()
    my_encoder.fit(all_elements)
    out = column.apply(
        lambda x: convert_tag_list_into_binary_vector(
            x, my_encoder, len(all_elements)))
    return out


def load_data(
        response_path: str = '/home/tianyudu/Data/Zuoye/bayes/exam_response_with_attrib.feather',
        attribute_path: str = '/home/tianyudu/Data/Zuoye/bayes/exam_response_ques_attrib.feather'):
    df_resp = pd.read_feather(response_path)
    print('Number of student-question response pairs:', len(df_resp))

    # TODO: figure out how to use observables.
    df_attr = pd.read_feather(attribute_path).sort_values(
        'question_id').reset_index(drop=True)

    def f(z):
        # extract knowledge domain.
        return z[-1]['id']

    knowledge_domain = [f(x) for x in df_attr['knowledge'].values]

    df_attr['capability_vec'] = convert_column_to_binary_vectors(
        df_attr['capability'])
    df_attr['knowledge_vec'] = convert_column_to_binary_vectors(
        df_attr['knowledge'])

    ques_attrib_list = [
        torch.stack(df_attr['capability_vec'].to_list(), dim=0).float(),
        torch.stack(df_attr['knowledge_vec'].to_list(), dim=0).float()
    ]

    # NOTE: modify this to decide which one to use.
    item_obs = torch.cat(ques_attrib_list, dim=1)

    choice_dataset = ChoiceDataset(
        item_index=torch.LongTensor(
            df_resp['question_id'].values), user_index=torch.LongTensor(
            df_resp['student_id'].values), label=torch.LongTensor(
                df_resp['correct'].values), item_obs=item_obs)

    return choice_dataset
