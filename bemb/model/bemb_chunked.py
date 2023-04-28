"""
A chunked version of BEMB.

We divide users, items(categories) and sessions into u, i and s chunks.
Then for each user, there are i*s parameters, for each item there are u*s parameters and for each session there are u*i parameters.

Author: Ayush Kanodia
Update: Dec 04, 2022
"""
import warnings
from pprint import pprint
import warnings
from typing import Dict, List, Optional, Tuple, Union
from pprint import pprint
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch_choice.data import ChoiceDataset
from torch_scatter import scatter_logsumexp, scatter_max
from torch_scatter.composite import scatter_log_softmax

from bemb.model.bayesian_coefficient import BayesianCoefficient

# ======================================================================================================================
# helper functions.
# ======================================================================================================================

from bemb.model.bemb import PositiveInteger, parse_utility

positive_integer = PositiveInteger()

# ======================================================================================================================
# core class of the BEMB model.
# ======================================================================================================================


class BEMBFlexChunked(nn.Module):
    # ==================================================================================================================
    # core function as a PyTorch module.
    # ==================================================================================================================
    def __init__(self,
                 utility_formula: str,
                 obs2prior_dict: Dict[str, bool],
                 coef_dim_dict: Dict[str, int],
                 num_items: int,
                 pred_item: bool,
                 num_classes: int = 2,
                 H_zero_mask_dict: Optional[Dict[str, torch.BoolTensor]] = None,
                 prior_mean: Union[float, Dict[str, float]] = 0.0,
                 prior_variance: Union[float, Dict[str, float]] = 1.0,
                 num_users: Optional[int] = None,
                 num_sessions: Optional[int] = None,
                 trace_log_q: bool = False,
                 category_to_item: Dict[int, List[int]] = None,
                 # number of observables.
                 num_user_obs: Optional[int] = None,
                 num_item_obs: Optional[int] = None,
                 num_session_obs: Optional[int] = None,
                 num_price_obs: Optional[int] = None,
                 num_taste_obs: Optional[int] = None,
                 # additional modules.
                 additional_modules: Optional[List[nn.Module]] = None,
                 deterministic_variational: bool = False,
                 chunk_info: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] = None,
                 ) -> None:
        """
        Args:
            utility_formula (str): a string representing the utility function U[user, item, session].
                See documentation for more details in the documentation for the format of formula.
                Examples:
                    lambda_item
                    lambda_item + theta_user * alpha_item + zeta_user * item_obs
                    lambda_item + theta_user * alpha_item + gamma_user * beta_item * price_obs
                See the doc-string of parse_utility for an example.

            obs2prior_dict (Dict[str, bool]): a dictionary maps coefficient name (e.g., 'lambda_item')
                to a boolean indicating if observable (e.g., item_obs) enters the prior of the coefficient.

            coef_dim_dict (Dict[str, int]): a dictionary maps coefficient name (e.g., 'lambda_item')
                to an integer indicating the dimension of coefficient.
                For standalone coefficients like U = lambda_item, the dim should be 1.
                For factorized coefficients like U = theta_user * alpha_item, the dim should be the
                    latent dimension of theta and alpha.
                For coefficients multiplied with observables like U = zeta_user * item_obs, the dim
                    should be the number of observables in item_obs.
                For factorized coefficient multiplied with observables like U = gamma_user * beta_item * price_obs,
                    the dim should be the latent dim multiplied by number of observables in price_obs.

            H_zero_mask_dict (Dict[str, torch.BoolTensor]): A dictionary maps coefficient names to a boolean tensor,
                you should only specify the H_zero_mask for coefficients with obs2prior turned on.
                Recall that with obs2prior on, the prior of coefficient looks like N(H*X_obs, sigma * I), the H_zero_mask
                the mask for this coefficient should have the same shape as H, and H[H_zero_mask] will be set to zeros
                and non-learnable during the training.
                Defaults to None.

            num_items (int): number of items.

            pred_item (bool): there are two use cases of this model, suppose we have `user_index[i]` and `item_index[i]`
                for the i-th observation in the dataset.
                Case 1: which item among all items user `user_index[i]` is going to purchase, the prediction label
                    is therefore `item_index[i]`. Equivalently, we can ask what's the likelihood for user `user_index[i]`
                    to purchase `item_index[i]`.
                Case 2: what rating would user `user_index[i]` assign to item `item_index[i]`? In this case, the dataset
                    object needs to contain a separate label.
                    NOTE: for now, we only support binary labels.

            prior_mean (Union[float, Dict[str, float]]): the mean of prior
                distribution for coefficients. If a float is provided, all prior
                mean will be diagonal matrix with the provided value.  If a
                dictionary is provided, keys of prior_mean should be coefficient
                names, and the mean of prior of coef_name would the provided
                value Defaults to 0.0, which means all prior means are
                initialized to 0.0

                If a dictionary prior_mean is supplied, for coefficient names not in the prior_mean.keys(), the
                user can add a `prior_mean['default']` value to specify the mean for those coefficients.
                If no `prior_mean['default']` is provided, the default prior mean will be 0.0 for those coefficients
                not in the prior_mean.keys().

                Defaults to 0.0.

            prior_variance (Union[float, Dict[str, float]], Dict[str, torch. Tensor]): the variance of prior distribution
                for coefficients.
                If a float is provided, all priors will be diagonal matrix with prior_variance along the diagonal.
                If a float-valued dictionary is provided, keys of prior_variance should be coefficient names, and the
                variance of prior of coef_name would be a diagonal matrix with prior_variance[coef_name] along the diagonal.
                If a tensor-valued dictionary is provided, keys of prior_variance should be coefficient names, and the
                values need to be tensor with shape (num_classes, coef_dim_dict[coef_name]). For example, for `beta_user` in
                `U = beta_user * item_obs`, the prior_variance should be a tensor with shape (num_classes, dimension_of_item_obs).
                In this case, every single entry in the coefficient has its own prior variance.
                Following the `beta_user` example, for every `i` and `j`, `beta_user[i, j]` is a scalar with prior variance
                `prior_variance['beta_user'][i, j]`. Moreover, `beta_user[i, j]`'s are independent for different `i, j`.

                If a dictionary prior_variance is supplied, for coefficient names not in the prior_variance.keys(), the
                user can add a `prior_variance['default']` value to specify the variance for those coefficients.
                If no `prior_variance['default']` is provided, the default prior variance will be 1.0 for those coefficients
                not in the prior_variance.keys().

                Defaults to 1.0, which means all priors have identity matrix as the covariance matrix.

            num_users (int, optional): number of users, required only if coefficient or observable
                depending on user is in utility. Defaults to None.
            num_sessions (int, optional): number of sessions, required only if coefficient or
                observable depending on session is in utility. Defaults to None.

            trace_log_q (bool, optional): whether to trace the derivative of variational likelihood logQ
                with respect to variational parameters in the ELBO while conducting gradient update.
                Defaults to False.

            category_to_item (Dict[str, List[int]], optional): a dictionary with category id or name
                as keys, and category_to_item[C] contains the list of item ids belonging to category C.
                If None is provided, all items are assumed to be in the same category.
                Defaults to None.

            num_{user, item, session, price, taste}_obs (int, optional): number of observables of
                each type of features, only required if observable enters prior.
                NOTE: currently we only allow coefficient to depend on either user or item, thus only
                user and item observables can enter the prior of coefficient. Hence session, price,
                and taste observables are never required, we include it here for completeness.

            deterministic_variational (bool, optional): if True, the variational posterior is equivalent to frequentist MLE estimates of parameters

            chunk_info (Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor], optional): a tuple of four tensors
                The first tensor specifies a chunk id for each user
                The second tensor specifies a chunk id for each item
                The third tensor specifies a chunk id for each category
                The fourth tensor specifies a chunk id for each session
        """
        super(BEMBFlexChunked, self).__init__()
        self.utility_formula = utility_formula
        self.obs2prior_dict = obs2prior_dict
        self.coef_dim_dict = coef_dim_dict
        if H_zero_mask_dict is not None:
            self.H_zero_mask_dict = H_zero_mask_dict
        else:
            self.H_zero_mask_dict = dict()
        self.prior_variance = prior_variance
        self.prior_mean = prior_mean
        self.pred_item = pred_item
        if not self.pred_item:
            assert isinstance(num_classes, int) and num_classes > 0, \
                f"With pred_item being False, the num_classes should be a positive integer, received {num_classes} instead."
            self.num_classes = num_classes
            if self.num_classes != 2:
                raise NotImplementedError('Multi-class classification is not supported yet.')
            # we don't set the num_classes attribute when pred_item == False to avoid calling it accidentally.

        self.num_items = num_items
        self.num_users = num_users
        self.num_sessions = num_sessions
        self.deterministic_variational = deterministic_variational

        self.trace_log_q = trace_log_q
        self.category_to_item = category_to_item

        # ==============================================================================================================
        # Category ID to Item ID mapping.
        # Category ID to Category Size mapping.
        # Item ID to Category ID mapping.
        # ==============================================================================================================
        if self.category_to_item is None:
            if self.pred_item:
                # assign all items to the same category if predicting items.
                self.category_to_item = {0: list(np.arange(self.num_items))}
            else:
                # otherwise, for the j-th observation in the dataset, the label[j]
                # only depends on user_index[j] and item_index[j], so we put each
                # item to its own category.
                self.category_to_item = {i: [i] for i in range(self.num_items)}

        self.num_categories = len(self.category_to_item)

        max_category_size = max(len(x) for x in self.category_to_item.values())
        category_to_item_tensor = torch.full(
            (self.num_categories, max_category_size), -1)
        category_to_size_tensor = torch.empty(self.num_categories)

        for c, item_in_c in self.category_to_item.items():
            category_to_item_tensor[c, :len(
                item_in_c)] = torch.LongTensor(item_in_c)
            category_to_size_tensor[c] = torch.scalar_tensor(len(item_in_c))

        self.register_buffer('category_to_item_tensor',
                             category_to_item_tensor.long())
        self.register_buffer('category_to_size_tensor',
                             category_to_size_tensor.long())

        item_to_category_tensor = torch.zeros(self.num_items)
        for c, items_in_c in self.category_to_item.items():
            item_to_category_tensor[items_in_c] = c
        self.register_buffer('item_to_category_tensor',
                             item_to_category_tensor.long())

        # ==============================================================================================================
        # Chunk Information
        self.num_user_chunks = chunk_info[0].max().item() + 1
        self.num_item_chunks = chunk_info[1].max().item() + 1
        self.num_category_chunks = chunk_info[2].max().item() + 1
        self.num_session_chunks = chunk_info[3].max().item() + 1
        self.register_buffer('user_chunk_ids', chunk_info[0])
        self.register_buffer('item_chunk_ids', chunk_info[1])
        self.register_buffer('category_chunk_ids', chunk_info[2])
        self.register_buffer('session_chunk_ids', chunk_info[3])
        # ==============================================================================================================
        # Create Bayesian Coefficient Objects
        # ==============================================================================================================
        # model configuration.
        self.formula = parse_utility(utility_formula)
        print('BEMB: utility formula parsed:')
        pprint(self.formula)
        self.raw_formula = utility_formula
        self.obs2prior_dict = obs2prior_dict

        # dimension of each observable, this one is used only for obs2prior.
        self.num_obs_dict = {
            'user': num_user_obs,
            'item': num_item_obs,
            'category' : 0,
            'session': num_session_obs,
            'price': num_price_obs,
            'taste': num_taste_obs,
            'constant': 1  # not really used, for dummy variables.
        }

        # how many classes for the variational distribution.
        # for example, beta_item would be `num_items` 10-dimensional gaussian if latent dim = 10.
        variation_to_num_classes = {
            'user': self.num_users,
            'item': self.num_items,
            'constant': 1,
            'category' : self.num_categories,
        }

        variation_to_num_chunks = {
            'user':(self.num_category_chunks, self.num_session_chunks),
            'item':(self.num_session_chunks, self.num_user_chunks),
            'category':(self.num_session_chunks, self.num_user_chunks),
            'session':(self.num_user_chunks, self.num_category_chunks),
            'constant': (1, 1),
            }

        coef_dict = dict()
        for additive_term in self.formula:
            for coef_name in additive_term['coefficient']:
                variation = coef_name.split('_')[-1]
                if isinstance(self.prior_mean, dict):
                    # the user didn't specify prior mean for this coefficient.
                    if coef_name not in self.prior_mean.keys():
                        # the user may specify 'default' prior variance through the prior_variance dictionary.
                        if 'default' in self.prior_mean.keys():
                            # warnings.warn(f"You provided a dictionary of prior mean, but coefficient {coef_name} is not a key in it. We found a key 'default' in the dictionary, so we use the value of 'default' as the prior mean for coefficient {coef_name}.")
                            self.prior_mean[coef_name] = self.prior_mean['default']
                        else:
                            # warnings.warn(f"You provided a dictionary of prior mean, but coefficient {coef_name} is not a key in it. Supply a value for 'default' in the prior_mean dictionary to use that as default value (e.g., prior_mean['default'] = 0.1); now using mean=0.0 since this is not supplied.")
                            self.prior_mean[coef_name] = 0.0

                mean = self.prior_mean[coef_name] if isinstance(
                    self.prior_mean, dict) else self.prior_mean

                if isinstance(self.prior_variance, dict):
                    # the user didn't specify prior variance for this coefficient.
                    if coef_name not in self.prior_variance.keys():
                        # the user may specify 'default' prior variance through the prior_variance dictionary.
                        if 'default' in self.prior_variance.keys():
                            # warnings.warn(f"You provided a dictionary of prior variance, but coefficient {coef_name} is not a key in it. We found a key 'default' in the dictionary, so we use the value of 'default' as the prior variance for coefficient {coef_name}.")
                            self.prior_variance[coef_name] = self.prior_variance['default']
                        else:
                            # warnings.warn(f"You provided a dictionary of prior variance, but coefficient {coef_name} is not a key in it. Supply a value for 'default' in the prior_variance dictionary to use that as default value (e.g., prior_variance['default'] = 0.3); now using variance=1.0 since this is not supplied.")
                            self.prior_variance[coef_name] = 1.0

                s2 = self.prior_variance[coef_name] if isinstance(
                    self.prior_variance, dict) else self.prior_variance

                if coef_name in self.H_zero_mask_dict.keys():
                    H_zero_mask = self.H_zero_mask_dict[coef_name]
                else:
                    H_zero_mask = None

                if (not self.obs2prior_dict[coef_name]) and (H_zero_mask is not None):
                    raise ValueError(f'You specified H_zero_mask for {coef_name}, but obs2prior is False for this coefficient.')

                chunk_sizes = variation_to_num_chunks[variation]
                bayesian_coefs = [] * chunk_sizes[0]
                for ii in range(chunk_sizes[0]):
                    bayesian_coefs_inner = []
                    for jj in range(chunk_sizes[1]):
                        bayesian_coefs_inner.append(BayesianCoefficient(variation=variation,
                                                                        num_classes=variation_to_num_classes[variation],
                                                                        obs2prior=self.obs2prior_dict[coef_name],
                                                                        num_obs=self.num_obs_dict[variation],
                                                                        dim=self.coef_dim_dict[coef_name],
                                                                        prior_mean=mean,
                                                                        prior_variance=s2,
                                                                        H_zero_mask=H_zero_mask,
                                                                        is_H=False
                                                                        )
                                                    )
                    bayesian_coefs_inner = nn.ModuleList(bayesian_coefs_inner)
                    bayesian_coefs.append(bayesian_coefs_inner)
                coef_dict[coef_name] = nn.ModuleList(bayesian_coefs)

        self.coef_dict = nn.ModuleDict(coef_dict)

        # ==============================================================================================================
        # Optional: register additional modules.
        # ==============================================================================================================
        if additional_modules is None:
            self.additional_modules = []
        else:
            raise NotImplementedError(
                'Additional modules are temporarily disabled for further development.')
            self.additional_modules = nn.ModuleList(additional_modules)

    def __str__(self):
        return f'Bayesian EMBedding Model with U[user, item, session] = {self.raw_formula}\n' \
               + f'Total number of parameters: {self.num_params}.\n' \
               + 'With the following coefficients:\n' \
               + str(self.coef_dict) + '\n' \
               + str(self.additional_modules)

    def posterior_mean(self, coef_name: str) -> torch.Tensor:
        """Returns the mean of estimated posterior distribution of coefficient `coef_name`.

        Args:
            coef_name (str): name of the coefficient to query.

        Returns:
            torch.Tensor: mean of the estimated posterior distribution of `coef_name`.
        """
        if coef_name in self.coef_dict.keys():
            return self.coef_dict[coef_name].variational_mean
        else:
            raise KeyError(f'{coef_name} is not a valid coefficient name in {self.utility_formula}.')

    def posterior_distribution(self, coef_name: str) -> torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal:
        """Returns the posterior distribution of coefficient `coef_name`.

        Args:
            coef_name (str): name of the coefficient to query.

        Returns:
            torch.Tensor: variance of the estimated posterior distribution of `coef_name`.
        """
        if coef_name in self.coef_dict.keys():
            return self.coef_dict[coef_name].variational_distribution
        else:
            raise KeyError(f'{coef_name} is not a valid coefficient name in {self.utility_formula}.')

    def ivs(self, batch) -> torch.Tensor:
        """The combined method of computing utilities and log probability.

            Args:
                batch (dict): a batch of data.

            Returns:
                torch.Tensor: the combined utility and log probability.
            """
        # Use the means of variational distributions as the sole MC sample.
        sample_dict = self.sample_coefficient_dictionary(1, deterministic=True)
        # there is 1 random seed in this case.
        # (num_seeds=1, len(batch), num_items)
        out = self.log_likelihood_all_items(batch, return_logit=True, sample_dict=sample_dict)
        out = out.squeeze(0)
        # import pdb; pdb.set_trace()
        out = out.view(-1, self.num_items)
        ivs = scatter_logsumexp(out, self.item_to_category_tensor, dim=-1)
        return ivs # (len(batch), num_categories)

    def sample_choices(self, batch:ChoiceDataset, debug: bool = False, num_seeds: int = 1, **kwargs) -> Tuple[torch.Tensor]:
        """Samples choices given model paramaters and trips

        Args:
        batch(ChoiceDataset): batch data containing trip information; item choice information is discarded
        debug(bool): whether to print debug information

        Returns:
        Tuple[torch.Tensor]: sampled choices; shape: (batch_size, num_categories)
        """
        # Use the means of variational distributions as the sole MC sample.
        sample_dict = dict()
        for coef_name, coef in self.coef_dict.items():
            sample_dict[coef_name] = coef.variational_distribution.mean.unsqueeze(dim=0)  # (1, num_*, dim)
        # sample_dict = self.sample_coefficient_dictionary(num_seeds)
        maxes, out = self.sample_log_likelihoods(batch, sample_dict)
        return maxes.squeeze(), out.squeeze()

    def sample_log_likelihoods(self, batch:ChoiceDataset, sample_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples log likelihoods given model parameters and trips

        Args:
        batch(ChoiceDataset): batch data containing trip information; item choice information is discarded
        sample_dict(Dict[str, torch.Tensor]): sampled coefficient values

        Returns:
        Tuple[torch.Tensor]: sampled log likelihoods; shape: (batch_size, num_categories)
        """
        # TODO(akanodia): disallow this for now
        raise NotImplementedError()
        # get the log likelihoods for all items for all categories
        utility = self.log_likelihood_all_items(batch, return_logit=True, sample_dict=sample_dict)
        mu_gumbel = 0.0
        beta_gumbel = 1.0
        EUL_MAS_CONST = 0.5772156649
        mean_gumbel = torch.tensor([mu_gumbel + beta_gumbel * EUL_MAS_CONST], device=self.device)
        m = torch.distributions.gumbel.Gumbel(torch.tensor([0.0], device=self.device), torch.tensor([1.0], device=self.device))
        # m = torch.distributions.gumbel.Gumbel(0.0, 1.0)
        gumbel_samples = m.sample(utility.shape).squeeze(-1)
        gumbel_samples -= mean_gumbel
        utility += gumbel_samples
        max_by_category, argmax_by_category = scatter_max(utility, self.item_to_category_tensor, dim=-1)
        return max_by_category, argmax_by_category
        log_likelihoods = self.sample_log_likelihoods_per_category(batch, sample_dict)

        # sum over all categories.
        log_likelihoods = log_likelihoods.sum(dim=1)

        return log_likelihoods, log_likelihoods

    @torch.no_grad()
    def predict_proba(self, batch: ChoiceDataset) -> torch.Tensor:
        """
        Draw prediction on a given batch of dataset.

        Args:
        batch (ChoiceDataset): the dataset to draw inference on.

        Returns:
        torch.Tensor: the predicted probabilities for each class, the behavior varies by self.pred_item.
        (1: pred_item == True) While predicting items, the return tensor has shape (len(batch), num_items), out[i, j] is the predicted probability for choosing item j AMONG ALL ITEMS IN ITS CATEGORY in observation i. Please note that since probabilities are computed from within-category normalization, hence out.sum(dim=0) can be greater than 1 if there are multiple categories.
        (2: pred_item == False) While predicting external labels for each observations, out[i, 0] is the predicted probability for label == 0 on the i-th observation, out[i, 1] is the predicted probability for label == 1 on the i-th observation. Generally, out[i, 0] + out[i, 1] = 1.0. However, this could be false if under-flowing/over-flowing issue is encountered.

        We highly recommend users to get log-probs as those are less prone to overflow/underflow; those can be accessed using the forward() function.
        """
        if self.pred_item:
            # (len(batch), num_items)
            log_p = self.forward(batch, return_type='log_prob', return_scope='all_items', deterministic=True)
            p = log_p.exp()
        else:
            # (len(batch), num_items)
            # probability of getting label = 1.
            p_1 = torch.nn.functional.sigmoid(self.forward(batch, return_type='utility', return_scope='all_items', deterministic=True))
            # (len(batch), 1)
            p_1 = p_1[torch.arange(len(batch)), batch.item_index].view(len(batch), 1)
            p_0 = 1 - p_1
            # (len(batch), 2)
            p = torch.cat([p_0, p_1], dim=1)

        if self.pred_item:
            assert p.shape == (len(batch), self.num_items)
        else:
            assert p.shape == (len(batch), self.num_classes)

        return p

    def forward(self, batch: ChoiceDataset,
                return_type: str,
                return_scope: str,
                deterministic: bool = True,
                sample_dict: Optional[Dict[str, torch.Tensor]] = None,
                num_seeds: Optional[int] = None
                ) -> torch.Tensor:
        """A combined method for inference with the model.

        Args:
            batch (ChoiceDataset): batch data containing choice information.
            return_type (str): either 'log_prob' or 'utility'.
                'log_prob': return the log-probability (by within-category log-softmax) for items
                'utility': return the utility value of items.
            return_scope (str): either 'item_index' or 'all_items'.
                'item_index': for each observation i, return log-prob/utility for the chosen item batch.item_index[i] only.
                'all_items': for each observation i, return log-prob/utility for all items.
            deterministic (bool, optional):
                True: expectations of parameter variational distributions are used for inference.
                False: the user needs to supply a dictionary of sampled parameters for inference.
                Defaults to True.
            sample_dict (Optional[Dict[str, torch.Tensor]], optional): sampled parameters for inference task.
                This is not needed when `deterministic` is True.
                When `deterministic` is False, the user can supply a `sample_dict`. If `sample_dict` is not provided,
                this method will create `num_seeds` samples.
                Defaults to None.
            num_seeds (Optional[int]): the number of random samples of parameters to construct. This is only required
                if `deterministic` is False (i.e., stochastic mode) and `sample_dict` is not provided.
                Defaults to None.
        Returns:
            torch.Tensor: a tensor of log-probabilities or utilities, depending on `return_type`.
                The shape of the returned tensor depends on `return_scope` and `deterministic`.
                -------------------------------------------------------------------------
                | `return_scope` | `deterministic` |         Output shape               |
                -------------------------------------------------------------------------
                |   'item_index` |      True       | (len(batch),)                      |
                -------------------------------------------------------------------------
                |   'all_items'  |      True       | (len(batch), num_items)            |
                -------------------------------------------------------------------------
                |   'item_index' |      False      | (num_seeds, len(batch))            |
                -------------------------------------------------------------------------
                |   'all_items'  |      False      | (num_seeds, len(batch), num_items) |
                -------------------------------------------------------------------------
        """
        # ==============================================================================================================
        # check arguments.
        # ==============================================================================================================
        assert return_type in [
            'log_prob', 'utility'], "return_type must be either 'log_prob' or 'utility'."
        assert return_scope in [
            'item_index', 'all_items'], "return_scope must be either 'item_index' or 'all_items'."
        assert deterministic in [True, False]
        if (not deterministic) and (sample_dict is None):
            assert num_seeds >= 1, "A positive interger `num_seeds` is required if `deterministic` is False and no `sample_dict` is provided."

        # when pred_item is true, the model is predicting which item is bought (specified by item_index).
        if self.pred_item:
            batch.label = batch.item_index

        # ==============================================================================================================
        # get sample_dict ready.
        # ==============================================================================================================
        if deterministic:
            sample_dict = self.sample_coefficient_dictionary(num_seeds, deterministic=True)
            '''
            num_seeds = 1
            # Use the means of variational distributions as the sole deterministic MC sample.
            # NOTE: here we don't need to sample the obs2prior weight H since we only compute the log-likelihood.
            # TODO: is this correct?
            sample_dict = dict()
            for coef_name, bayesian_coeffs in self.coef_dict.items():
                num_classes = bayesian_coeffs[0][0].num_classes
                dim = bayesian_coeffs[0][0].dim
                this_sample = torch.FloatTensor(num_seeds, num_classes, dim, len(bayesian_coeffs), len(bayesian_coeffs[0])).to(self.device)
                # outer_list = []
                for ii, bayesian_coeffs_inner in enumerate(bayesian_coeffs):
                    # inner_list = []
                    for jj, coef in enumerate(bayesian_coeffs_inner):
                        this_sample[:, :, :, ii, jj] = coef.variational_distribution.mean.unsqueeze(dim=0) # (1, num_*, dim)
                        # inner_list.append(coef.variational_distribution.mean.unsqueeze(dim=0)) # (1, num_*, dim)
                        # inner_list.append(coef.variational_distribution.mean.unsqueeze(dim=0)) # (1, num_*, dim)
                    # outer_list.append(inner_list)
                sample_dict[coef_name] = this_sample
            '''
        else:
            if sample_dict is None:
                # sample stochastic parameters.
                sample_dict = self.sample_coefficient_dictionary(num_seeds)
            else:
                # use the provided sample_dict.
                num_seeds = list(sample_dict.values())[0].shape[0]

        # ==============================================================================================================
        # call the sampling method of additional modules.
        # ==============================================================================================================
        for module in self.additional_modules:
            # deterministic sample.
            if deterministic:
                module.dsample()
            else:
                module.rsample(num_seeds=num_seeds)

        # if utility is requested, don't run log-softmax, simply return logit.
        return_logit = (return_type == 'utility')
        if return_scope == 'all_items':
            # (num_seeds, len(batch), num_items)
            # TODO: (akanodia) disallow this for now.
            raise NotImplementedError()
            out = self.log_likelihood_all_items(
                batch=batch, sample_dict=sample_dict, return_logit=return_logit)
        elif return_scope == 'item_index':
            # (num_seeds, len(batch))
            out = self.log_likelihood_item_index(
                batch=batch, sample_dict=sample_dict, return_logit=return_logit)

        if deterministic:
            # drop the first dimension, which has size of `num_seeds` (equals 1 in the deterministic case).
            # (len(batch), num_items) or (len(batch),)
            return out.squeeze(dim=0)

        return out

    @property
    def num_params(self) -> int:
        return sum([p.numel() for p in self.parameters()])

    @property
    def device(self) -> torch.device:
        for coef in self.coef_dict.values():
            return coef[0][0].device

    # ==================================================================================================================
    # helper functions.
    # ==================================================================================================================
    def sample_coefficient_dictionary(self, num_seeds: int, deterministic: bool=False) -> Dict[str, torch.Tensor]:
        """A helper function to sample parameters from coefficients.

        Args:
            num_seeds (int): number of random samples.

        Returns:
            Dict[str, torch.Tensor]: a dictionary maps coefficient names to tensor of sampled coefficient parameters,
                where the first dimension of the sampled tensor has size `num_seeds`.
                Each sample tensor has shape (num_seeds, num_classes, dim).
        """
        sample_dict = dict()
        if deterministic:
            num_seeds = 1
            # Use the means of variational distributions as the sole deterministic MC sample.
            # NOTE: here we don't need to sample the obs2prior weight H since we only compute the log-likelihood.
            # TODO: is this correct?
            sample_dict = dict()
            for coef_name, bayesian_coeffs in self.coef_dict.items():
                num_classes = bayesian_coeffs[0][0].num_classes
                dim = bayesian_coeffs[0][0].dim
                this_sample = torch.FloatTensor(num_seeds, num_classes, dim, len(bayesian_coeffs), len(bayesian_coeffs[0])).to(self.device)
                # outer_list = []
                for ii, bayesian_coeffs_inner in enumerate(bayesian_coeffs):
                    # inner_list = []
                    for jj, coef in enumerate(bayesian_coeffs_inner):
                        this_sample[:, :, :, ii, jj] = coef.variational_distribution.mean.unsqueeze(dim=0) # (1, num_*, dim)
                        # inner_list.append(coef.variational_distribution.mean.unsqueeze(dim=0)) # (1, num_*, dim)
                        # inner_list.append(coef.variational_distribution.mean.unsqueeze(dim=0)) # (1, num_*, dim)
                    # outer_list.append(inner_list)
                sample_dict[coef_name] = this_sample
        else:
            for coef_name, bayesian_coeffs in self.coef_dict.items():
                # outer_list = []
                num_classes = bayesian_coeffs[0][0].num_classes
                dim = bayesian_coeffs[0][0].dim
                this_sample = torch.FloatTensor(num_seeds, num_classes, dim, len(bayesian_coeffs), len(bayesian_coeffs[0])).to(self.device)
                obs2prior = self.obs2prior_dict[coef_name]
                if obs2prior:
                    num_obs = bayesian_coeffs[0][0].num_obs
                    this_sample_H = torch.FloatTensor(num_seeds, dim, num_obs, len(bayesian_coeffs), len(bayesian_coeffs[0])).to(self.device)
                    # outer_list_H = []
                for ii, bayesian_coeffs_inner in enumerate(bayesian_coeffs):
                    # inner_list = []
                    # if obs2prior:
                        # inner_list_H = []
                    for jj, coef in enumerate(bayesian_coeffs_inner):
                        s = coef.rsample(num_seeds)
                        if coef.obs2prior:
                            # sample both obs2prior weight and realization of variable.
                            assert isinstance(s, tuple) and len(s) == 2
                            this_sample[:, :, :, ii, jj] = s[0]
                            this_sample_H[:, :, :, ii, jj] = s[1]
                            # inner_list.append(s[0])
                            # inner_list_H.append(s[1])
                        else:
                            # only sample the realization of variable.
                            assert torch.is_tensor(s)
                            this_sample[:, :, :, ii, jj] = s
                            # inner_list.append(s)
                    # outer_list.append(inner_list)
                    # if obs2prior:
                            # outer_list_H.append(inner_list_H)
                sample_dict[coef_name] = this_sample
                # sample_dict[coef_name] = outer_list
                if obs2prior:
                    sample_dict[coef_name + '.H'] = this_sample_H
                    # sample_dict[coef_name + '.H'] = outer_list_H

        return sample_dict

    @torch.no_grad()
    def get_within_category_accuracy(self, log_p_all_items: torch.Tensor, label: torch.LongTensor) -> Dict[str, float]:
        """A helper function for computing prediction accuracy (i.e., all non-differential metrics)
        within category.
        In particular, this method calculates the accuracy, precision, recall and F1 score.


        This method has the same functionality as the following peusodcode:
        for C in categories:
            # get sessions in which item in category C was purchased.
            T <- (t for t in {0,1,..., len(label)-1} if label[t] is in C)
            Y <- label[T]

            predictions = list()
            for t in T:
                # get the prediction within category for this session.
                y_pred = argmax_{items in C} log prob computed before.
                predictions.append(y_pred)

            accuracy = mean(Y == predictions)

        Similarly, this function computes precision, recall and f1score as well.

        Args:
            log_p_all_items (torch.Tensor): shape (num_sessions, num_items) the log probability of
                choosing each item in each session.
            label (torch.LongTensor): shape (num_sessions,), the IDs of items purchased in each session.

        Returns:
            [Dict[str, float]]: A dictionary containing performance metrics.
        """
        # argmax: (num_sessions, num_categories), within category argmax.
        # item IDs are consecutive, thus argmax is the same as IDs of the item with highest P.
        _, argmax_by_category = scatter_max(
            log_p_all_items, self.item_to_category_tensor, dim=-1)

        # category_purchased[t] = the category of item label[t].
        # (num_sessions,)
        category_purchased = self.item_to_category_tensor[label]

        # pred[t] = the item with highest utility from the category item label[t] belongs to.
        # (num_sessions,)
        pred_from_category = argmax_by_category[torch.arange(
            len(label)), category_purchased]

        within_category_accuracy = (
            pred_from_category == label).float().mean().item()

        # precision
        precision = list()

        recall = list()
        for i in range(self.num_items):
            correct_i = torch.sum(
                (torch.logical_and(pred_from_category == i, label == i)).float())
            precision_i = correct_i / \
                torch.sum((pred_from_category == i).float())
            recall_i = correct_i / torch.sum((label == i).float())

            # do not add if divided by zero.
            if torch.any(pred_from_category == i):
                precision.append(precision_i.cpu().item())
            if torch.any(label == i):
                recall.append(recall_i.cpu().item())

        precision = float(np.mean(precision))
        recall = float(np.mean(recall))

        if precision == recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        return {'accuracy': within_category_accuracy,
                'precision': precision,
                'recall': recall,
                'f1score': f1}

    # ==================================================================================================================
    # Methods for terms in the ELBO: prior, likelihood, and variational.
    # ==================================================================================================================
    def log_likelihood_all_items(self, batch: ChoiceDataset, return_logit: bool, sample_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        NOTE to developers:
        NOTE (akanodia to tianyudu): Is this really slow; even with log_likelihood you need log_prob which depends on logits of all items?
        This method computes utilities for all items available, which is a relatively slow operation. For
        training the model, you only need the utility/log-prob for the chosen/relevant item (i.e., item_index[i] for each i-th observation).
        Use this method for inference only.
        Use self.log_likelihood_item_index() for training instead.

        Computes the log probability of choosing `each` item in each session based on current model parameters.
        NOTE (akanodiadu to tianyudu): What does the next line mean? I think it just says its allowing for samples instead of posterior mean.
        This method allows for specifying {user, item}_latent_value for Monte Carlo estimation in ELBO.
        For actual prediction tasks, use the forward() function, which will use means of variational
        distributions for user and item latents.

        Args:
            batch (ChoiceDataset): a ChoiceDataset object containing relevant information.
            return_logit(bool): if set to True, return the log-probability, otherwise return the logit/utility.
            sample_dict(Dict[str, torch.Tensor]): Monte Carlo samples for model coefficients
                (i.e., those Greek letters).
                sample_dict.keys() should be the same as keys of self.obs2prior_dict, i.e., those
                greek letters actually enter the functional form of utility.
                The value of sample_dict should be tensors of shape (num_seeds, num_classes, dim)
                where num_classes in {num_users, num_items, 1}
                and dim in {latent_dim(K), num_item_obs, num_user_obs, 1}.

        Returns:
            torch.Tensor: a tensor of shape (num_seeds, len(batch), self.num_items), where
                out[x, y, z] is the probability of choosing item z in session y conditioned on
                latents to be the x-th Monte Carlo sample.
        """
        batch.item_index = torch.arange(self.num_items, device=batch.device)
        batch.item_index = batch.item_index.repeat(batch.user_index.shape[0])
        batch.user_index = batch.user_index.repeat_interleave(self.num_items)
        batch.session_index = batch.session_index.repeat_interleave(self.num_items)
        return self.log_likelihood_item_index(batch, return_logit, sample_dict, all_items=True)

    def log_likelihood_item_index(self, batch: ChoiceDataset, return_logit: bool, sample_dict: Dict[str, torch.Tensor], all_items: bool=False) -> torch.Tensor:
        """
        NOTE for developers:
        This method is more efficient and only computes log-likelihood/logit(utility) for item in item_index[i] for each
        i-th observation.
        Developers should use use `log_likelihood_all_items` for inference purpose and to computes log-likelihoods/utilities
        for ALL items for the i-th observation.

        Computes the log probability of choosing item_index[i] in each session based on current model parameters.
        This method allows for specifying {user, item}_latent_value for Monte Carlo estimation in ELBO.
        For actual prediction tasks, use the forward() function, which will use means of variational
        distributions for user and item latents.

        Args:
            batch (ChoiceDataset): a ChoiceDataset object containing relevant information.
            return_logit(bool): if set to True, return the logit/utility, otherwise return the log-probability.
            sample_dict(Dict[str, torch.Tensor]): Monte Carlo samples for model coefficients
                (i.e., those Greek letters).
                sample_dict.keys() should be the same as keys of self.obs2prior_dict, i.e., those
                greek letters actually enter the functional form of utility.
                The value of sample_dict should be tensors of shape (num_seeds, num_classes, dim)
                where num_classes in {num_users, num_items, 1}
                and dim in {latent_dim(K), num_item_obs, num_user_obs, 1}.

        Returns:
            torch.Tensor: a tensor of shape (num_seeds, len(batch)), where
                out[x, y] is the probabilities of choosing item batch.item[y] in session y
                conditioned on latents to be the x-th Monte Carlo sample.
        """
        num_seeds = list(sample_dict.values())[0].shape[0]

        # get category id of the item bought in each row of batch.
        cate_index = self.item_to_category_tensor[batch.item_index]

        # get item ids of all items from the same category of each item bought.
        relevant_item_index = self.category_to_item_tensor[cate_index, :]
        relevant_item_index = relevant_item_index.view(-1,)
        # index were padded with -1's, drop those dummy entries.
        relevant_item_index = relevant_item_index[relevant_item_index != -1]

        # the first repeats[0] entries in relevant_item_index are for the category of item_index[0]
        repeats = self.category_to_size_tensor[cate_index]
        # argwhere(reverse_indices == k) are positions in relevant_item_index for the category of item_index[k].
        reverse_indices = torch.repeat_interleave(
            torch.arange(len(batch), device=self.device), repeats)
        # expand the user_index and session_index.
        # if all_items:
        #     breakpoint()
        user_index = torch.repeat_interleave(batch.user_index, repeats)
        repeat_category_index = torch.repeat_interleave(cate_index, repeats)
        session_index = torch.repeat_interleave(batch.session_index, repeats)
        # duplicate the item focused to match.
        item_index_expanded = torch.repeat_interleave(
            batch.item_index, repeats)

        # short-hands for easier shape check.
        R = num_seeds
        # total number of relevant items.
        total_computation = len(session_index)
        S = self.num_sessions
        U = self.num_users
        I = self.num_items
        NC = self.num_categories

        user_chunk_ids = torch.repeat_interleave(self.user_chunk_ids[batch.user_index], repeats)
        item_chunk_ids = torch.repeat_interleave(self.item_chunk_ids[batch.item_index], repeats)
        session_chunk_ids = torch.repeat_interleave(self.session_chunk_ids[batch.session_index], repeats)
        category_chunk_ids = torch.repeat_interleave(self.category_chunk_ids[cate_index], repeats)

        # ==========================================================================================
        # Helper Functions for Reshaping.
        # ==========================================================================================

        def reshape_coef_sample(sample, name):
            # reshape the monte carlo sample of coefficients to (R, P, I, *).
            if name.endswith('_user'):
                # (R, total_computation, dim, chunk_size_1, chunk_size_2)
                all_chunks_sample = sample[:, user_index, :, :, :]
                # (total_computation) --> (1, total_computation, 1, 1, 1)
                second_chunk_index = session_chunk_ids.reshape(1, -1, 1, 1, 1)
                # (1, total_computation, 1, 1, 1) --> (R, total_computation, dim, chunk_size_1, 1)
                second_chunk_index = second_chunk_index.repeat(R, 1, all_chunks_sample.shape[2], all_chunks_sample.shape[3], 1)
                # (total_computation) --> (1, total_computation, 1, 1)
                first_chunk_index = category_chunk_ids.reshape(1, -1, 1, 1)
                # (1, total_computation, 1, 1) --> (R, total_computation, dim, 1)
                first_chunk_index = first_chunk_index.repeat(R, 1, all_chunks_sample.shape[2], 1)
                # select the first chunk.
                second_chunk_selected = torch.gather(all_chunks_sample, -1, second_chunk_index).squeeze(-1)
                # select the second chunk.
                first_chunk_selected = torch.gather(second_chunk_selected, -1, first_chunk_index).squeeze(-1)
                return first_chunk_selected
            elif name.endswith('_item'):
                # (R, total_computation, dim, chunk_size_1, chunk_size_2)
                all_chunks_sample = sample[:, relevant_item_index, :, :, :]
                # (total_computation) --> (1, total_computation, 1, 1, 1)
                second_chunk_index = user_chunk_ids.reshape(1, -1, 1, 1, 1)
                # (1, total_computation, 1, 1, 1) --> (R, total_computation, dim, chunk_size_1, 1)
                second_chunk_index = second_chunk_index.repeat(R, 1, all_chunks_sample.shape[2], all_chunks_sample.shape[3], 1)
                # (total_computation) --> (1, total_computation, 1, 1)
                first_chunk_index = session_chunk_ids.reshape(1, -1, 1, 1)
                # (1, total_computation, 1, 1) --> (R, total_computation, dim, 1)
                first_chunk_index = first_chunk_index.repeat(R, 1, all_chunks_sample.shape[2], 1)
                # select the first chunk.
                second_chunk_selected = torch.gather(all_chunks_sample, -1, second_chunk_index).squeeze(-1)
                # select the second chunk.
                first_chunk_selected = torch.gather(second_chunk_selected, -1, first_chunk_index).squeeze(-1)
                return first_chunk_selected
            elif name.endswith('_category'):
                # (R, total_computation, dim, chunk_size_1, chunk_size_2)
                all_chunks_sample = sample[:, repeat_category_index, :, :, :]
                # (total_computation) --> (1, total_computation, 1, 1, 1)
                second_chunk_index = user_chunk_ids.reshape(1, -1, 1, 1, 1)
                # (1, total_computation, 1, 1, 1) --> (R, total_computation, dim, chunk_size_1, 1)
                second_chunk_index = second_chunk_index.repeat(R, 1, all_chunks_sample.shape[2], all_chunks_sample.shape[3], 1)
                # (total_computation) --> (1, total_computation, 1, 1)
                first_chunk_index = session_chunk_ids.reshape(1, -1, 1, 1)
                # (1, total_computation, 1, 1) --> (R, total_computation, dim, 1)
                first_chunk_index = first_chunk_index.repeat(R, 1, all_chunks_sample.shape[2], 1)
                # select the first chunk.
                second_chunk_selected = torch.gather(all_chunks_sample, -1, second_chunk_index).squeeze(-1)
                # select the second chunk.
                first_chunk_selected = torch.gather(second_chunk_selected, -1, first_chunk_index).squeeze(-1)
                return first_chunk_selected
            elif name.endswith('_constant'):
                # (R, *) --> (R, total_computation, *)
                return sample[:, 0, 0].view(R, 1, -1).expand(-1, total_computation, -1)
            else:
                raise ValueError

        def reshape_observable(obs, name):
            # reshape observable to (R, P, I, *) so that it can be multiplied with monte carlo
            # samples of coefficients.
            O = obs.shape[-1]  # number of observables.
            assert O == positive_integer
            if name.startswith('item_'):
                assert obs.shape == (I, O)
                obs = obs[relevant_item_index, :]
            elif name.startswith('user_'):
                assert obs.shape == (U, O)
                obs = obs[user_index, :]
            elif name.startswith('session_'):
                assert obs.shape == (S, O)
                obs = obs[session_index, :]
            elif name.startswith('price_'):
                assert obs.shape == (S, I, O)
                obs = obs[session_index, relevant_item_index, :]
            elif name.startswith('taste_'):
                assert obs.shape == (U, I, O)
                obs = obs[user_index, relevant_item_index, :]
            else:
                raise ValueError
            assert obs.shape == (total_computation, O)
            return obs.unsqueeze(dim=0).expand(R, -1, -1)

        # ==========================================================================================
        # Compute Components related to users and items only.
        # ==========================================================================================
        utility = torch.zeros(R, total_computation, device=self.device)

        # loop over additive term to utility
        for term in self.formula:
            # Type I: single coefficient, e.g., lambda_item or lambda_user.
            if len(term['coefficient']) == 1 and term['observable'] is None:
                # E.g., lambda_item or lambda_user
                coef_name = term['coefficient'][0]
                coef_sample = reshape_coef_sample(
                    sample_dict[coef_name], coef_name)
                assert coef_sample.shape == (R, total_computation, 1)
                additive_term = coef_sample.view(R, total_computation)

            # Type II: factorized coefficient, e.g., <theta_user, lambda_item>.
            elif len(term['coefficient']) == 2 and term['observable'] is None:
                coef_name_0 = term['coefficient'][0]
                coef_name_1 = term['coefficient'][1]

                coef_sample_0 = reshape_coef_sample(
                    sample_dict[coef_name_0], coef_name_0)
                coef_sample_1 = reshape_coef_sample(
                    sample_dict[coef_name_1], coef_name_1)

                assert coef_sample_0.shape == coef_sample_1.shape == (
                    R, total_computation, positive_integer)

                additive_term = (coef_sample_0 * coef_sample_1).sum(dim=-1)

            # Type III: single coefficient multiplied by observable, e.g., theta_user * x_obs_item.
            elif len(term['coefficient']) == 1 and term['observable'] is not None:
                coef_name = term['coefficient'][0]
                coef_sample = reshape_coef_sample(
                    sample_dict[coef_name], coef_name)
                # breakpoint()
                assert coef_sample.shape == (
                    R, total_computation, positive_integer)

                obs_name = term['observable']
                obs = reshape_observable(getattr(batch, obs_name), obs_name)
                assert obs.shape == (R, total_computation, positive_integer)

                additive_term = (coef_sample * obs).sum(dim=-1)

            # Type IV: factorized coefficient multiplied by observable.
            # e.g., gamma_user * beta_item * price_obs.
            elif len(term['coefficient']) == 2 and term['observable'] is not None:
                coef_name_0, coef_name_1 = term['coefficient'][0], term['coefficient'][1]
                coef_sample_0 = reshape_coef_sample(
                    sample_dict[coef_name_0], coef_name_0)
                coef_sample_1 = reshape_coef_sample(
                    sample_dict[coef_name_1], coef_name_1)
                assert coef_sample_0.shape == coef_sample_1.shape == (
                    R, total_computation, positive_integer)
                num_obs_times_latent_dim = coef_sample_0.shape[-1]

                obs_name = term['observable']
                obs = reshape_observable(getattr(batch, obs_name), obs_name)
                assert obs.shape == (R, total_computation, positive_integer)
                num_obs = obs.shape[-1]  # number of observables.

                assert (num_obs_times_latent_dim % num_obs) == 0
                latent_dim = num_obs_times_latent_dim // num_obs

                coef_sample_0 = coef_sample_0.view(
                    R, total_computation, num_obs, latent_dim)
                coef_sample_1 = coef_sample_1.view(
                    R, total_computation, num_obs, latent_dim)
                # compute the factorized coefficient with shape (R, P, I, O).
                coef = (coef_sample_0 * coef_sample_1).sum(dim=-1)

                additive_term = (coef * obs).sum(dim=-1)

            else:
                raise ValueError(f'Undefined term type: {term}')

            assert additive_term.shape == (R, total_computation)
            utility += additive_term

        # ==========================================================================================
        # Mask Out Unavailable Items in Each Session.
        # ==========================================================================================

        if batch.item_availability is not None:
            # expand to the Monte Carlo sample dimension.
            A = batch.item_availability[session_index, relevant_item_index].unsqueeze(
                dim=0).expand(R, -1)
            utility[~A] = - (torch.finfo(utility.dtype).max / 2)

        for module in self.additional_modules:
            # current utility shape: (R, total_computation)
            additive_term = module(batch)
            assert additive_term.shape == (
                R, len(batch)) or additive_term.shape == (R, len(batch), 1)
            if additive_term.shape == (R, len(batch), 1):
                # TODO: need to make this consistent with log_likelihood_all.
                # be tolerant for some customized module with BayesianLinear that returns (R, len(batch), 1).
                additive_term = additive_term.view(R, len(batch))
            # expand to total number of computation, query by reverse_indices.
            # reverse_indices has length total_computation, and reverse_indices[i] correspond to the row-id that this
            # computation is responsible for.
            additive_term = additive_term[:, reverse_indices]
            assert additive_term.shape == (R, total_computation)

        if return_logit:
            # (num_seeds, len(batch))
            u = utility[:, item_index_expanded == relevant_item_index]
            assert u.shape == (R, len(batch))
            return u

        if self.pred_item:
            # compute log likelihood log p(choosing item i | user, item latents)
            # compute the log probability from logits/utilities.
            # output shape: (num_seeds, len(batch), num_items)
            log_p = scatter_log_softmax(utility, reverse_indices, dim=-1)
            # select the log-P of the item actually bought.
            log_p = log_p[:, item_index_expanded == relevant_item_index]
            assert log_p.shape == (R, len(batch))
            return log_p
        else:
            # This is the binomial choice situation in which case we just report sigmoid log likelihood
            utility = utility[:, item_index_expanded == relevant_item_index]
            assert utility.shape == (R, len(batch))
            bce = nn.BCELoss(reduction='none')
            # make num_seeds copies of the label, expand to (R, len(batch))
            label_expanded = batch.label.to(torch.float32).view(1, len(batch)).expand(R, -1)
            assert label_expanded.shape == (R, len(batch))
            log_p = - bce(torch.sigmoid(utility), label_expanded)
            assert log_p.shape == (R, len(batch))
            return log_p

    def log_prior(self, batch: ChoiceDataset, sample_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculates the log-likelihood of Monte Carlo samples of Bayesian coefficients under their
        prior distribution. This method assume coefficients are statistically independent.

        Args:
            batch (ChoiceDataset): a dataset object contains observables for computing the prior distribution
                if obs2prior is True.
            sample_dict (Dict[str, torch.Tensor]): a dictionary coefficient names to Monte Carlo samples.

        Raises:
            ValueError: [description]

        Returns:
            torch.scalar_tensor: a tensor with shape (num_seeds,) of [ log P_{prior_distribution}(param[i]) ],
                where param[i] is the i-th Monte Carlo sample.
        """
        # assert sample_dict.keys() == self.coef_dict.keys()
        num_seeds = list(sample_dict.values())[0].shape[0]
        cate_index = self.item_to_category_tensor[batch.item_index]
        user_chunk_ids = self.user_chunk_ids[batch.user_index]
        item_chunk_ids = self.item_chunk_ids[batch.item_index]
        session_chunk_ids = self.session_chunk_ids[batch.session_index]
        category_chunk_ids = self.category_chunk_ids[cate_index]

        total = torch.zeros(num_seeds, device=self.device)

        def reshape_coef_sample(sample, name):
            # reshape the monte carlo sample of coefficients to (R, P, I, *).
            if name.endswith('_user'):
                # (R, U, *) --> (R, total_computation, *)
                temp = sample[:, :, :, :, :]
                stemp = session_chunk_ids.reshape(1, -1, 1, 1, 1)
                stemp = stemp.repeat(1, 1, temp.shape[2], temp.shape[3], 1)
                ctemp = category_chunk_ids.reshape(1, -1, 1, 1)
                ctemp = ctemp.repeat(1, 1, temp.shape[2], 1)
                gathered1 = torch.gather(temp, 4, stemp).squeeze(4)
                gathered2 = torch.gather(gathered1, 3, ctemp).squeeze(3)
                return gathered2
                # return sample[:, user_index, :, category_chunk_ids, session_chunk_ids]
            elif name.endswith('_item'):
                # (R, I, *) --> (R, total_computation, *)
                temp = sample[:, :, :, :, :]
                utemp = user_chunk_ids.reshape(1, -1, 1, 1, 1)
                utemp = utemp.repeat(1, 1, temp.shape[2], temp.shape[3], 1)
                stemp = session_chunk_ids.reshape(1, -1, 1, 1)
                stemp = stemp.repeat(1, 1, temp.shape[2], 1)
                gathered1 = torch.gather(temp, 4, utemp).squeeze(4)
                gathered2 = torch.gather(gathered1, 3, stemp).squeeze(3)
                return gathered2
                # return sample[:, relevant_item_index, :, session_chunk_ids, user_chunk_ids]
            elif name.endswith('_category'):
                # (R, NC, *) --> (R, total_computation, *)
                return sample[:, repeat_category_index, :, session_chunk_ids, user_chunk_ids]
            elif name.endswith('_constant'):
                # (R, *) --> (R, total_computation, *)
                return sample[:, 0, 0].view(R, 1, -1).expand(-1, total_computation, -1)
            else:
                raise ValueError

        # for coef_name, coef in self.coef_dict.items():
        for coef_name, bayesian_coeffs in self.coef_dict.items():
            for ii, bayesian_coeffs_inner in enumerate(bayesian_coeffs):
                for jj, coef in enumerate(bayesian_coeffs_inner):
                    if self.obs2prior_dict[coef_name]:
                        if coef_name.endswith('_item'):
                            x_obs = batch.item_obs
                        elif coef_name.endswith('_user'):
                            x_obs = batch.user_obs
                        else:
                            raise ValueError(
                                f'No observable found to support obs2prior for {coef_name}.')

                        total += coef.log_prior(sample=sample_dict[coef_name][:, :, :, ii, jj],
                                                H_sample=sample_dict[coef_name + '.H'][:, :, :, ii, jj],
                                                x_obs=x_obs).sum(dim=-1)
                    else:
                        # log_prob outputs (num_seeds, num_{items, users}), sum to (num_seeds).
                        total += coef.log_prior(
                            sample=sample_dict[coef_name][:, :, :, ii, jj], H_sample=None, x_obs=None).sum(dim=-1)
                    # break
                # break

        for module in self.additional_modules:
            raise NotImplementedError()
            total += module.log_prior()

        return total

    def log_variational(self, sample_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate the log-likelihood of samples in sample_dict under the current variational
        distribution.

        Args:
            sample_dict (Dict[str, torch.Tensor]):  a dictionary coefficient names to Monte Carlo
                samples.

        Returns:
            torch.Tensor: a tensor of shape (num_seeds) of [ log P_{variational_distribution}(param[i]) ],
                where param[i] is the i-th Monte Carlo sample.
        """
        num_seeds = list(sample_dict.values())[0].shape[0]
        total = torch.zeros(num_seeds, device=self.device)

        for coef_name, coef in self.coef_dict.items():
            # log_prob outputs (num_seeds, num_{items, users}), sum to (num_seeds).
            total += coef.log_variational(sample_dict[coef_name]).sum(dim=-1)

        for module in self.additional_modules:
            raise NotImplementedError()
            # with shape (num_seeds,)
            total += module.log_variational().sum()

        return total

    def elbo(self, batch: ChoiceDataset, num_seeds: int = 1) -> torch.Tensor:
        """A combined method to computes the current ELBO given a batch, this method is used for training the model.

        Args:
            batch (ChoiceDataset): a ChoiceDataset containing necessary information.
            num_seeds (int, optional): the number of Monte Carlo samples from variational distributions
                to evaluate the expectation in ELBO.
                Defaults to 1.

        Returns:
            torch.Tensor: a scalar tensor of the ELBO estimated from num_seeds Monte Carlo samples.
        """
        # ==============================================================================================================
        # 1. sample latent variables from their variational distributions.
        # ==============================================================================================================
        if self.deterministic_variational:
            num_seeds = 1
            # Use the means of variational distributions as the sole deterministic MC sample.
            # NOTE: here we don't need to sample the obs2prior weight H since we only compute the log-likelihood.
            # TODO: is this correct?
            sample_dict = dict()
            for coef_name, coef in self.coef_dict.items():
                sample_dict[coef_name] = coef.variational_distribution.mean.unsqueeze(
                    dim=0)  # (1, num_*, dim)
        else:
            sample_dict = self.sample_coefficient_dictionary(num_seeds)

        # ==============================================================================================================
        # 2. compute log p(latent) prior.
        # (num_seeds,) --mean--> scalar.
        # with torch.no_grad():
        #     while True:
        #         elbo = self.log_prior(batch, sample_dict).mean(dim=0)
        # elbo = torch.tensor(0.0, device=self.device)
        elbo = self.log_prior(batch, sample_dict).mean(dim=0)
        # ==============================================================================================================

        # ==============================================================================================================
        # 3. compute the log likelihood log p(obs|latent).
        # sum over independent purchase decision for individual observations, mean over MC seeds.
        # the forward() function calls module.rsample(num_seeds) for module in self.additional_modules.
        # ==============================================================================================================
        if self.pred_item:
            # the prediction target is item_index.
            elbo_expanded = self.forward(batch,
                                 return_type='log_prob',
                                 return_scope='item_index',
                                 deterministic=self.deterministic_variational,
                                 sample_dict=sample_dict)
            if self.deterministic_variational:
                elbo_expanded = elbo_expanded.unsqueeze(dim=0)
            elbo += elbo_expanded.sum(dim=1).mean(dim=0)  # (num_seeds, len(batch)) --> scalar.
        else:
            # the prediction target is binary.
            # TODO: update the prediction function.
            utility = self.forward(batch,
                                   return_type='utility',
                                   return_scope='item_index',
                                   deterministic=self.deterministic_variational,
                                   sample_dict=sample_dict)  # (num_seeds, len(batch))

            # compute the log-likelihood for binary label.
            # (num_seeds, len(batch))
            y_stacked = torch.stack([batch.label] * num_seeds).float()
            assert y_stacked.shape == utility.shape
            bce = nn.BCELoss(reduction='none')
            # scalar.
            ll = - bce(torch.sigmoid(utility),
                       y_stacked).sum(dim=1).mean(dim=0)
            elbo += ll

        # ==============================================================================================================
        # 4. optionally add log likelihood under variational distributions q(latent).
        # ==============================================================================================================
        if self.trace_log_q:
            #TODO(akanodia): do not allow at this time
            raise NotImplementedError()
            assert not self.deterministic_variational, "deterministic_variational is not compatible with trace_log_q."
            elbo -= self.log_variational(sample_dict).mean(dim=0)

        return elbo
