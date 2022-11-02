"""
Generates simulated students (c.f. users) and questions (c.f. items) for the student-question pairs following
Henry Shi's thesis.

Author: Tianyu Du
Date: 2022-03-10
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch_choice.data import ChoiceDataset

NUM_RESPONSES = 500000


def simulate_dense_dataset():
    # Following section "3.5.1 Simulated dense data" of Henry's paper.
    NUM_STUDENTS = 50  # i
    NUM_QUESTIONS = 30  # j
    theta = np.random.randn(NUM_STUDENTS)
    beta = np.random.randn(NUM_QUESTIONS)
    alpha = np.random.randn(NUM_QUESTIONS) * 0.5 + 1.2
    alpha[alpha < 0] = 0.05

    P = np.zeros((NUM_STUDENTS, NUM_QUESTIONS))
    for i in range(NUM_STUDENTS):
        for j in range(NUM_QUESTIONS):
            P[i, j] = 1 / (1 + np.exp(- alpha[j] * (theta[i] - beta[j])))

    # generate student-question responses.
    student_idx = np.random.choice(NUM_STUDENTS, size=NUM_RESPONSES)
    question_idx = np.random.choice(NUM_QUESTIONS, size=NUM_RESPONSES)
    p_correct = P[student_idx, question_idx]
    y = np.random.binomial(1, p_correct)
    return theta, alpha, beta, student_idx, question_idx, y


def simulate_overlap_data_with_missings():
    # Following section "3.5.2 Simulated overlap data with missings." of
    # Henry's paper.
    NUM_STUDENTS = 100  # i
    NUM_QUESTIONS = 50  # j

    theta = np.random.randn(NUM_STUDENTS)
    alpha = np.random.randn(NUM_QUESTIONS) * 0.5 + 1.2
    alpha[alpha < 0] = 0.05

    beta_1 = np.random.randn(NUM_QUESTIONS // 2)
    beta_2 = np.random.randn(NUM_QUESTIONS // 2) * 0.7 + 0.5

    beta = np.concatenate([beta_1, beta_2])

    P = np.zeros((NUM_STUDENTS, NUM_QUESTIONS))
    for i in range(NUM_STUDENTS):
        for j in range(NUM_QUESTIONS):
            P[i, j] = 1 / (1 + np.exp(- alpha[j] * (theta[i] - beta[j])))

    student_idx = np.random.choice(NUM_STUDENTS, size=NUM_RESPONSES)
    question_idx = np.empty(NUM_RESPONSES)
    group_1_mask = (student_idx <= 50)
    # different students are answering different questions.
    question_idx[group_1_mask] = np.random.choice(
        np.arange(NUM_QUESTIONS // 2), size=np.sum(group_1_mask))
    question_idx[~group_1_mask] = np.random.choice(np.arange(
        NUM_QUESTIONS // 2, NUM_QUESTIONS), size=np.sum(~group_1_mask))
    question_idx = question_idx.astype(int)

    p_correct = P[student_idx, question_idx]
    y = np.random.binomial(1, p_correct)
    return theta, alpha, beta, student_idx, question_idx, y


# def visualize_data():
#     fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
#     axes[0].hist(alpha, label='alpha')
#     axes[1].hist(beta, label='beta (question)')
#     axes[2].hist(theta, label='theta (student)')
#     for ax in axes:
#         ax.legend()
#     fig.savefig('./out/params.png')

#     fig, ax = plt.subplots()
#     sns.heatmap(P, ax=ax)
#     fig.savefig('./out/prob.png')


def simulate_dataset():
    theta, alpha, beta, student_idx, question_idx, y = simulate_overlap_data_with_missings()
    item_obs = torch.eye(len(np.unique(question_idx)))

    choice_dataset = ChoiceDataset(item_index=torch.LongTensor(question_idx),
                                   user_index=torch.LongTensor(student_idx),
                                   label=torch.LongTensor(y),
                                   item_obs=item_obs)

    return choice_dataset


# if __name__ == '__main__':
#     student_latent_hat = bemb.model.coef_dict['theta_user'].variational_mean.squeeze()
#     question_latent_hat = bemb.model.coef_dict['alpha_item'].variational_mean.squeeze()

#     # plot.
#     order = np.argsort(beta)
#     fig, ax = plt.subplots()
#     ax.scatter(np.arange(len(beta)), question_latent_hat.detach().numpy()[order], label='hat')
#     ax.scatter(np.arange(len(beta)), beta[order], label='true')
#     ax.legend()
#     fig.savefig('./out/question_latent.png')
