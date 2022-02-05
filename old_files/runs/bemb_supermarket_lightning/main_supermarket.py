import argparse
import os
import sys
import time
from pprint import pprint

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from deepchoice.data import ChoiceDataset
from deepchoice.data.utils import create_data_loader
from deepchoice.model import LitBEMB
from sklearn.preprocessing import LabelEncoder
from termcolor import cprint
from tqdm import tqdm

STOP_CODES = {
    1 : 'validation set llh increase per supplied config'
}

def write_bemb_cpp_format(configs, model, user_encoder, item_encoder, it=""):
    users = user_encoder.classes_
    items = item_encoder.classes_
    users_df = pd.DataFrame(users, columns=['user_id'])
    items_df = pd.DataFrame(items, columns=['item_id'])

    if 'lambda_item' in model.coef_dict:
        # Write lambda and theta
        lambdas = model.coef_dict['lambda_item'].variational_mean
        lambda_df = pd.DataFrame(lambdas.detach().cpu().numpy())
        lambda_df = pd.concat((items_df, lambda_df), axis=1)
        lambda_df.to_csv('%s/param_lambda%s_mean.tsv' % (configs.out_dir, it), sep='\t', header=False)
        lambdas = torch.exp(model.coef_dict['lambda_item'].variational_logstd)
        lambda_df = pd.DataFrame(lambdas.detach().cpu().numpy())
        lambda_df = pd.concat((items_df, lambda_df), axis=1)
        lambda_df.to_csv('%s/param_lambda%s_std.tsv' % (configs.out_dir, it), sep='\t', header=False)

    if 'alpha_item' in model.coef_dict:
        # Write alpha and theta
        alphas = model.coef_dict['alpha_item'].variational_mean
        alpha_df = pd.DataFrame(alphas.detach().cpu().numpy())
        alpha_df = pd.concat((items_df, alpha_df), axis=1)
        alpha_df.to_csv('%s/param_alpha%s_mean.tsv' % (configs.out_dir, it), sep='\t', header=False)
        alphas = torch.exp(model.coef_dict['alpha_item'].variational_logstd)
        alpha_df = pd.DataFrame(alphas.detach().cpu().numpy())
        alpha_df = pd.concat((items_df, alpha_df), axis=1)
        alpha_df.to_csv('%s/param_alpha%s_std.tsv' % (configs.out_dir, it), sep='\t', header=False)

        thetas = model.coef_dict['theta_user'].variational_mean
        theta_df = pd.DataFrame(thetas.detach().cpu().numpy())
        theta_df = pd.concat((users_df, theta_df), axis=1)
        theta_df.to_csv('%s/param_theta%s_mean.tsv' % (configs.out_dir, it), sep='\t', header=False)
        thetas = torch.exp(model.coef_dict['theta_user'].variational_logstd)
        theta_df = pd.DataFrame(thetas.detach().cpu().numpy())
        theta_df = pd.concat((users_df, theta_df), axis=1)
        theta_df.to_csv('%s/param_theta%s_std.tsv' % (configs.out_dir, it), sep='\t', header=False)

    if 'beta_item' in model.coef_dict:
        # write beta and gamma
        betas = model.coef_dict['beta_item'].variational_mean
        beta_df = pd.DataFrame(betas.detach().cpu().numpy())
        beta_df = pd.concat((items_df, beta_df), axis=1)
        beta_df.to_csv('%s/param_beta%s_mean.tsv' % (configs.out_dir, it), sep='\t', header=False)
        betas = torch.exp(model.coef_dict['beta_item'].variational_logstd)
        beta_df = pd.DataFrame(betas.detach().cpu().numpy())
        beta_df = pd.concat((items_df, beta_df), axis=1)
        beta_df.to_csv('%s/param_beta%s_std.tsv' % (configs.out_dir, it), sep='\t', header=False)

        gammas = model.coef_dict['gamma_user'].variational_mean
        gamma_df = pd.DataFrame(gammas.detach().cpu().numpy())
        gamma_df = pd.concat((users_df, gamma_df), axis=1)
        gamma_df.to_csv('%s/param_gamma%s_mean.tsv' % (configs.out_dir, it), sep='\t', header=False)
        gammas = torch.exp(model.coef_dict['gamma_user'].variational_logstd)
        gamma_df = pd.DataFrame(gammas.detach().cpu().numpy())
        gamma_df = pd.concat((users_df, gamma_df), axis=1)
        gamma_df.to_csv('%s/param_gamma%s_std.tsv' % (configs.out_dir, it), sep='\t', header=False)

    if 'obsuser_item' in model.coef_dict:
        # write beta and gamma
        obsuser_item = model.coef_dict['obsuser_item'].variational_mean
        obsuser_item_df = pd.DataFrame(obsuser_item.detach().cpu().numpy())
        obsuser_item_df= pd.concat((items_df, obsuser_item_df), axis=1)
        obsuser_item_df.to_csv('%s/param_obsUsers%s_mean.tsv' % (configs.out_dir, it), sep='\t', header=False)
        obsuser_item = torch.exp(model.coef_dict['obsuser_item'].variational_logstd)
        obsuser_item_df = pd.DataFrame(obsuser_item.detach().cpu().numpy())
        obsuser_item_df = pd.concat((items_df, obsuser_item_df), axis=1)
        obsuser_item_df.to_csv('%s/param_obsUsers%s_std.tsv' % (configs.out_dir, it), sep='\t', header=False)

    if 'obsitem_user' in model.coef_dict:
        # write beta and gamma
        obsitem_user = model.coef_dict['obsitem_user'].variational_mean
        obsitem_user_df = pd.DataFrame(obsitem_user.detach().cpu().numpy())
        obsitem_user_df= pd.concat((users_df, obsitem_user_df), axis=1)
        obsitem_user_df.to_csv('%s/param_obsItems%s_mean.tsv' % (configs.out_dir, it), sep='\t', header=False)
        obsitem_user = torch.exp(model.coef_dict['obsitem_user'].variational_logstd)
        obsitem_user_df = pd.DataFrame(obsitem_user.detach().cpu().numpy())
        obsitem_user_df = pd.concat((users_df, obsitem_user_df), axis=1)
        obsitem_user_df.to_csv('%s/param_obsItems%s_std.tsv' % (configs.out_dir, it), sep='\t', header=False)

def load_configs(yaml_file: str):
    with open(yaml_file, 'r') as file:
        data_loaded = yaml.safe_load(file)
    # Add defaults
    defaults = {
        'num_verify_val' : 10,
        'early_stopping' : {'validation_llh_flat' : -1},
        'write_best_model' : True,
        'val_llh_series_out' : '',
        'alt_val_dir' : '',
        'alt_val_llh_series_out' : ''
    }
    defaults.update(data_loaded)
    configs = argparse.Namespace(**defaults)
    return configs


def load_params_to_model(model, path) -> None:
    def load_cpp_tsv(file):
        df = pd.read_csv(os.path.join(path, file), sep='\t', index_col=0, header=None)
        return torch.Tensor(df.values[:, 1:])

    cpp_theta_mean = load_cpp_tsv('param_theta_mean.tsv')
    cpp_theta_std = load_cpp_tsv('param_theta_std.tsv')

    cpp_alpha_mean = load_cpp_tsv('param_alpha_mean.tsv')
    cpp_alpha_std = load_cpp_tsv('param_alpha_std.tsv')

    # theta user
    model.coef_dict['theta_user'].mean.data = cpp_theta_mean.to(model.device)
    model.coef_dict['theta_user'].logstd.data = torch.log(cpp_theta_std).to(model.device)
    # alpha item
    model.coef_dict['alpha_item'].mean.data = cpp_alpha_mean.to(model.device)
    model.coef_dict['alpha_item'].logstd.data = torch.log(cpp_alpha_std).to(model.device)


def is_sorted(x):
    return all(x == np.sort(x))


def load_tsv(file_name, data_dir):
    return pd.read_csv(os.path.join(data_dir, file_name),
                       sep='\t',
                       index_col=None,
                       names=['user_id', 'item_id', 'session_id', 'quantity'])


if __name__ == '__main__':
    cprint('Your are running an example script.', 'green')
    # sys.argv[1] should be the yaml file.
    configs = load_configs(sys.argv[1])

    # ==============================================================================================
    # Load standard BEMB inputs.
    # ==============================================================================================
    train = load_tsv('train.tsv', configs.data_dir)
    # read standard BEMB input files.
    validation = load_tsv('validation.tsv', configs.data_dir)
    test = load_tsv('test.tsv', configs.data_dir)

    print(f'{train.shape=:}, {validation.shape=:}, {test.shape=:}')

    # ==============================================================================================
    # Encode users and items to {0, 1, ..., num-1}.
    # ==============================================================================================
    # combine data for encoding.
    data_all = pd.concat([train, validation, test], axis=0)
    # encode user.
    user_encoder = LabelEncoder().fit(data_all['user_id'].values)
    configs.num_users = len(user_encoder.classes_)
    assert is_sorted(user_encoder.classes_)
    # encode items.
    item_encoder = LabelEncoder().fit(data_all['item_id'].values)
    configs.num_items = len(item_encoder.classes_)
    assert is_sorted(item_encoder.classes_)

    # ==============================================================================================
    # user observables
    # ==============================================================================================
    user_obs = pd.read_csv(os.path.join(configs.data_dir, 'obsUser.tsv'),
                           sep='\t',
                           index_col=0,
                           header=None)
    # do we need to catch it in some check process?
    user_obs = user_obs.groupby(user_obs.index).first().sort_index()
    user_obs = torch.Tensor(user_obs.values)
    configs.num_user_obs = user_obs.shape[1]

    # ==============================================================================================
    # item observables
    # ==============================================================================================
    item_obs = pd.read_csv(os.path.join(configs.data_dir, 'obsItem.tsv'),
                           sep='\t',
                           index_col=0,
                           header=None)
    item_obs = item_obs.groupby(item_obs.index).first().sort_index()
    item_obs = torch.Tensor(item_obs.values)
    configs.num_item_obs = item_obs.shape[1]

    # ==============================================================================================
    # item availability
    # ==============================================================================================
    # parse item availability.
    # Try and catch? Optionally specify full availability?
    a_tsv = pd.read_csv(os.path.join(configs.data_dir, 'availabilityList.tsv'),
                        sep='\t',
                        index_col=None,
                        header=None,
                        names=['session_id', 'item_id'])

    # availability ties session as well.
    session_encoder = LabelEncoder().fit(a_tsv['session_id'].values)
    configs.num_sessions = len(session_encoder.classes_)
    assert is_sorted(session_encoder.classes_)
    # this loop could be slow, depends on # sessions.
    item_availability = torch.zeros(configs.num_sessions, configs.num_items).bool()

    a_tsv['item_id'] = item_encoder.transform(a_tsv['item_id'].values)
    a_tsv['session_id'] = session_encoder.transform(a_tsv['session_id'].values)

    for session_id, df_group in a_tsv.groupby('session_id'):
        # get IDs of items available at this date.
        a_item_ids = df_group['item_id'].unique()  # this unique is not necessary if the dataset is well-prepared.
        item_availability[session_id, a_item_ids] = True

    # ==============================================================================================
    # price observables
    # ==============================================================================================
    df_price = pd.read_csv(os.path.join(configs.data_dir, 'item_sess_price.tsv'),
                           sep='\t',
                           names=['item_id', 'session_id', 'price'])

    # only keep prices of relevant items.
    mask = df_price['item_id'].isin(item_encoder.classes_)
    df_price = df_price[mask]

    df_price['item_id'] = item_encoder.transform(df_price['item_id'].values)
    df_price['session_id'] = session_encoder.transform(df_price['session_id'].values)
    df_price = df_price.pivot(index='session_id', columns='item_id')
    # NAN prices.
    df_price.fillna(0.0, inplace=True)
    price_obs = torch.Tensor(df_price.values).view(configs.num_sessions, configs.num_items, 1)
    configs.num_price_obs = 1

    # ==============================================================================================
    # create datasets
    # ==============================================================================================
    dataset_list = list()
    for d in (train, validation, test):
        user_index = user_encoder.transform(d['user_id'].values)
        label = torch.LongTensor(item_encoder.transform(d['item_id'].values))
        session_index = torch.LongTensor(session_encoder.transform(d['session_id'].values))
        # get the date (aka session_id in the raw dataset) of each row in the dataset, retrieve
        # the item availability information from that date.

        choice_dataset = ChoiceDataset(label=label,
                                       user_index=user_index,
                                       session_index=session_index,
                                       item_availability=item_availability,
                                       user_obs=user_obs,
                                       item_obs=item_obs,
                                       price_obs=price_obs)

        dataset_list.append(choice_dataset)

    # ==============================================================================================
    # category information
    # ==============================================================================================
    item_groups = pd.read_csv(os.path.join(configs.data_dir, 'itemGroup.tsv'),
                              sep='\t',
                              index_col=None,
                              names=['item_id', 'category_id'])

    # TODO(Tianyu): handle duplicate group information.
    item_groups = item_groups.groupby('item_id').first().reset_index()
    # filter out items never purchased.
    mask = item_groups['item_id'].isin(item_encoder.classes_)
    item_groups = item_groups[mask].reset_index(drop=True)
    item_groups = item_groups.sort_values(by='item_id')

    category_encoder = LabelEncoder().fit(item_groups['category_id'])
    configs.num_categories = len(category_encoder.classes_)

    # encode them to consecutive integers {0, ..., num_items-1}.
    item_groups['item_id'] = item_encoder.transform(
        item_groups['item_id'].values)
    item_groups['category_id'] = category_encoder.transform(
        item_groups['category_id'].values)

    print('Category sizes:')
    print(item_groups.groupby('category_id').size().describe())
    item_groups = item_groups.groupby('category_id')['item_id'].apply(list)
    category_to_item = dict(zip(item_groups.index, item_groups.values))

    # ==============================================================================================
    # create data loaders.
    # ==============================================================================================

    dataloaders = dict()
    for dataset, partition in zip(dataset_list, ('train', 'validation', 'test')):
        # dataset = dataset.to(configs.device)
        dataloader = create_data_loader(dataset, configs, num_workers=8)
        dataloaders[partition] = dataloader

    # ==============================================================================================
    # create model
    # ==============================================================================================
    # model = BEMB(num_users=configs.num_users,
    #              num_items=configs.num_items,
    #              num_sessions=configs.num_sessions,
    #              obs2prior_dict=configs.obs2prior_dict,
    #              latent_dim=configs.latent_dim,
    #              latent_dim_price=configs.latent_dim_price,
    #              trace_log_q=configs.trace_log_q,
    #              category_to_item=category_to_item,
    #              num_user_obs=configs.num_user_obs,
    #              num_item_obs=configs.num_item_obs,
    #              num_price_obs=configs.num_price_obs
    #              ).to(configs.device)

    # best_val_llh_model = BEMB(num_users=configs.num_users,
    #                           num_items=configs.num_items,
    #                           num_sessions=configs.num_sessions,
    #                           obs2prior_dict=configs.obs2prior_dict,
    #                           latent_dim=configs.latent_dim,
    #                           latent_dim_price=configs.latent_dim_price,
    #                           trace_log_q=configs.trace_log_q,
    #                           category_to_item=category_to_item,
    #                           num_user_obs=configs.num_user_obs,
    #                           num_item_obs=configs.num_item_obs,
    #                           num_price_obs=configs.num_price_obs
    #                           ).to(configs.device)

    # print(model.variational_dict['theta_user'].mean)
    # # breakpoint()
    # load_params_to_model(model, '/home/tianyudu/Data/MoreSupermarket/t79338-n2123-m1109-k20-users3-lik3-shuffle0-eta0.005-zF0.1-nS10-batch100000-run_K20_1000_rmsprop_bs100000')
    # print(model.variational_dict['theta_user'].mean)

    bemb = LitBEMB(
        learning_rate=configs.learning_rate,
        num_seeds=configs.num_mc_seeds,
        num_users=configs.num_users,
        num_items=configs.num_items,
        num_sessions=configs.num_sessions,
        obs2prior_dict=configs.obs2prior_dict,
        latent_dim=configs.latent_dim,
        latent_dim_price=configs.latent_dim_price,
        trace_log_q=configs.trace_log_q,
        category_to_item=category_to_item,
        num_user_obs=configs.num_user_obs,
        num_item_obs=configs.num_item_obs,
        num_price_obs=configs.num_price_obs)

    print(80 * '=')
    print(bemb.model)
    print(80 * '=')

    trainer = pl.Trainer(gpus=1,
                         max_epochs=configs.num_epochs,
                         check_val_every_n_epoch=3,
                         log_every_n_steps=1)

    trainer.fit(bemb, dataloaders['train'], dataloaders['validation'])

    # old training pipeline below.
    quit()
    # ==============================================================================================
    # training
    # ==============================================================================================
    # breakpoint()
    performance_by_epoch = list()
    best_val_llh = np.NINF
    last_val_llh = np.NINF
    val_llh_decrease_count = 0
    # stop > 0 leads to a stopping criteria different from num epochs having run
    # we use stop = 1 when we perform an early stop based on validation log likelihood
    stop = 0

    for i in tqdm(range(configs.num_epochs), desc='epoch'):
        total_loss = torch.scalar_tensor(0.0).to(configs.device)

        for batch in tqdm(dataloaders['train'], desc='batch', leave=False):
            # maximize the ELBO.
            loss = - model.elbo(batch.to(configs.device), num_seeds=1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.detach().item()
        print(loss)
        scheduler.step()
        if i % (configs.num_epochs // configs.num_verify_val) == 0:
            # report training progress, report 10 times in total.
            with torch.no_grad():
                performance = {'iteration': i,
                               'duration_seconds': time.time() - start_time}
                # compute performance for each data partition.
                for partition in ('train', 'validation', 'test'):
                    metrics = ['log_likelihood', 'accuracy', 'precision', 'recall', 'f1score']
                    for m in metrics:
                        performance[partition + '_' + m] = list()
                    # compute performance for each batch.
                    for batch in dataloaders[partition]:
                        pred = model(batch.to(configs.device))  # (num_sessions, num_items) log-likelihood.
                        LL_all = pred[torch.arange(len(batch)), batch.label].detach().cpu().numpy()
                        LL = LL_all.mean().item()
                        performance[partition + '_log_likelihood'].append(LL)
                        accuracy_metrics = model.get_within_category_accuracy(pred, batch.label)
                        for key, val in accuracy_metrics.items():
                            performance[partition + '_' + key].append(val)

                for key, val in performance.items():
                    performance[key] = np.mean(val)

                pprint(performance)

                # Early Stopping
                val_llh = performance['validation_log_likelihood']
                if val_llh <= best_val_llh:
                    best_val_llh_model.load_state_dict(model.state_dict())

                if val_llh < last_val_llh:
                    val_llh_decrease_count += 1
                    early_stop = configs.early_stopping['validation_llh_flat']
                    if early_stop > 0 and val_llh_decrease_count >= early_stop:
                        stop = 1
                else:
                    val_llh_decrease_count = 0
                last_val_llh = val_llh
                performance_by_epoch.append(performance)
                pprint(performance)
                print(f'Epoch [{i}] negative elbo (the lower the better)={total_loss}')
        if stop > 0:
            print(f'EARLY STOPPING due to {STOP_CODES[stop]}')
            break
    print(f'Time taken: {time.time() - start_time: 0.1f} seconds.')
    log = pd.DataFrame(performance_by_epoch)

    # ==============================================================================================
    # save results
    # ==============================================================================================

    os.system(f'mkdir {configs.out_dir}')
    log.to_csv(os.path.join(configs.out_dir, 'performance_log_by_epoch.csv'))
    write_bemb_cpp_format(configs, model, user_encoder, item_encoder)

    # save best_val_llh_model weights
    if (configs.write_best_model):
        write_val_partitions = [('validation', configs.val_llh_series_out, validation)]
        if configs.alt_val_dir != '':
            alt_validation = load_tsv("validation.tsv", configs.alt_val_dir)
            user_index = user_encoder.transform(alt_validation['user_id'].values)
            label = torch.LongTensor(item_encoder.transform(alt_validation['item_id'].values))
            session_index = torch.LongTensor(session_encoder.transform(alt_validation['session_id'].values))
            # get the date (aka session_id in the raw dataset) of each row in the dataset, retrieve
            # the item availability information from that date.
            choice_dataset = ChoiceDataset(label=label,
                                           user_index=user_index,
                                           session_index=session_index,
                                           item_availability=item_availability,
                                           user_obs=user_obs,
                                           item_obs=item_obs,
                                           price_obs=price_obs)
            dataloaders['alt_validation'] = create_data_loader(choice_dataset, configs)
            write_val_partitions.append(('alt_validation', configs.alt_val_llh_series_out, alt_validation))
        with torch.no_grad():
            # compute performance for each data partition.
            for partition in write_val_partitions:
                # compute performance for each batch.
                LL_all = list()
                for batch in dataloaders[partition[0]]:
                    pred = best_val_llh_model(batch.to(configs.device))  # (num_sessions, num_items) log-likelihood.
                    LL_all.append(pd.Series(pred[torch.arange(len(batch)), batch.label].detach().cpu().numpy()))
                    # accuracy_metrics = model.get_within_category_accuracy(pred, batch.label)
                    # for key, val in accuracy_metrics.items():
                    #     performance[partition + '_' + key].append(val)
                LL_df = pd.concat(LL_all)
                partition[2]['log_likelihood_part'] = list(LL_df)
                partition[2].to_csv(partition[1], index=False)


        torch.save(best_val_llh_model, os.path.join(configs.out_dir, 'best_val_llh_model.pt'))
        torch.save(best_val_llh_model.state_dict(), os.path.join(configs.out_dir, 'best_val_llh_model_state_dict.pt'))
        write_bemb_cpp_format(configs, best_val_llh_model, user_encoder, item_encoder)

    # save model weights
    torch.save(model, os.path.join(configs.out_dir, 'model.pt'))
    torch.save(model.state_dict(), os.path.join(configs.out_dir, 'state_dict.pt'))
