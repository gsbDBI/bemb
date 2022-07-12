"""
The core class of the Bayesian EMBedding (BEMB) model.

Author: Tianyu Du
Update: Apr. 28, 2022
"""
from pprint import pprint
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch_choice.data import ChoiceDataset
from torch_scatter import scatter_max, scatter_logsumexp
from torch_scatter.composite import scatter_log_softmax

from bemb.model.bayesian_coefficient import BayesianCoefficient

# ======================================================================================================================
# helper functions.
# ======================================================================================================================


class PositiveInteger(object):
    # A helper wildcard class for shape matching.
    def __eq__(self, other):
        return isinstance(other, int) and other > 0


positive_integer = PositiveInteger()


def parse_utility(utility_string: str) -> List[Dict[str, Union[List[str], None]]]:
    """
    A helper function parse utility string into a list of additive terms.

    Example:
        utility_string = 'lambda_item + theta_user * alpha_item + gamma_user * beta_item * price_obs'
        output = [
            {
                'coefficient': ['lambda_item'],
                'observable': None
            },
            {
                'coefficient': ['theta_user', 'alpha_item'],
                'observable': None
            },
            {
                'coefficient': ['gamma_user', 'beta_item'],
                'observable': 'price_obs'
            }
            ]
    """
    # split additive terms
    coefficient_suffix = ('_item', '_user', '_constant', '_category')
    observable_prefix = ('item_', 'user_', 'session_', 'price_', 'taste_')

    def is_coefficient(name: str) -> bool:
        return any(name.endswith(suffix) for suffix in coefficient_suffix)

    def is_observable(name: str) -> bool:
        return any(name.startswith(prefix) for prefix in observable_prefix)

    additive_terms = utility_string.split(' + ')
    additive_decomposition = list()
    for term in additive_terms:
        atom = {'coefficient': [], 'observable': None}
        # split multiplicative terms.
        for x in term.split(' * '):
            if is_coefficient(x):
                atom['coefficient'].append(x)
            elif is_observable(x):
                atom['observable'] = x
            else:
                raise ValueError(f'{x} term cannot be classified.')
        additive_decomposition.append(atom)
    return additive_decomposition

# ======================================================================================================================
# core class of the BEMB model.
# ======================================================================================================================


class BEMBFlex(nn.Module):
    # ==================================================================================================================
    # core function as a PyTorch module.
    # ==================================================================================================================
    def __init__(self,
                 utility_formula: str,
                 obs2prior_dict: Dict[str, bool],
                 coef_dim_dict: Dict[str, int],
                 num_items: int,
                 pred_item: bool,
                 prior_mean: Union[float, Dict[str, float]] = 0.0,
                 default_prior_mean: float = 0.0,
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
                 additional_modules: Optional[List[nn.Module]] = None
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

            num_items (int): number of items.

            pred_item (bool): there are two use cases of this model, suppose we have `user_index[i]` and `item_index[i]`
                for the i-th observation in the dataset.
                Case 1: which item among all items user `user_index[i]` is going to purchase, the prediction label
                    is therefore `item_index[i]`. Equivalently, we can ask what's the likelihood for user `user_index[i]`
                    to purchase `item_index[i]`.
                Case 2: what rating would user `user_index[i]` assign to item `item_index[i]`? In this case, the dataset
                    object needs to contain a separate label.
                    NOTE: for now, we only support binary labels.

            default_prior_mean (float): the default prior mean for coefficients,
            if it is not specified in the prior_mean; defaults to 0.0.

            prior_mean (Union[float, Dict[str, float]]): the mean of prior
                distribution for coefficients. If a float is provided, all prior
                mean will be diagonal matrix with the provided value.  If a
                dictionary is provided, keys of prior_mean should be coefficient
                names, and the mean of prior of coef_name would the provided
                value Defaults to 0.0, which means all prior means are
                initalized to 0.0

            prior_variance (Union[float, Dict[str, float]]): the variance of prior distribution for
                coefficients. If a float is provided, all priors will be diagonal matrix with
                prior_variance along the diagonal. If a dictionary is provided, keys of prior_variance
                should be coefficient names, and the variance of prior of coef_name would be a diagonal
                matrix with prior_variance[coef_name] along the diagonal.
                Defaults to 1.0, which means all prior have identity matrix as the covariance matrix.

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
        """
        super(BEMBFlex, self).__init__()
        self.utility_formula = utility_formula
        self.obs2prior_dict = obs2prior_dict
        self.coef_dim_dict = coef_dim_dict
        self.prior_variance = prior_variance
        self.default_prior_mean = default_prior_mean
        self.prior_mean = prior_mean

        self.pred_item = pred_item

        self.num_items = num_items
        self.num_users = num_users
        self.num_sessions = num_sessions

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

        coef_dict = dict()
        for additive_term in self.formula:
            for coef_name in additive_term['coefficient']:
                variation = coef_name.split('_')[-1]
                mean = self.prior_mean[coef_name] if isinstance(
                    self.prior_mean, dict) else self.default_prior_mean
                s2 = self.prior_variance[coef_name] if isinstance(
                    self.prior_variance, dict) else self.prior_variance
                coef_dict[coef_name] = BayesianCoefficient(variation=variation,
                                                           num_classes=variation_to_num_classes[variation],
                                                           obs2prior=self.obs2prior_dict[coef_name],
                                                           num_obs=self.num_obs_dict[variation],
                                                           dim=self.coef_dim_dict[coef_name],
                                                           prior_mean=mean,
                                                           prior_variance=s2)
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

    def ivs(self, batch) -> torch.Tensor:
        """The combined method of computing utilities and log probability.

            Args:
                batch (dict): a batch of data.

            Returns:
                torch.Tensor: the combined utility and log probability.
            """
        # Use the means of variational distributions as the sole MC sample.
        sample_dict = dict()
        for coef_name, coef in self.coef_dict.items():
            sample_dict[coef_name] = coef.variational_distribution.mean.unsqueeze(dim=0)  # (1, num_*, dim)

        # there is 1 random seed in this case.
        # (num_seeds=1, len(batch), num_items)
        out = self.log_likelihood_all_items(batch, return_logit=True, sample_dict=sample_dict)
        out = out.squeeze(0)
        # import pdb; pdb.set_trace()
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
        """Samples log likelihoods given model paramaters and trips

        Args:
        batch(ChoiceDataset): batch data containing trip information; item choice information is discarded
        sample_dict(Dict[str, torch.Tensor]): sampled coefficient values

        Returns:
        Tuple[torch.Tensor]: sampled log likelihoods; shape: (batch_size, num_categories)
        """
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
            num_seeds = 1
            # Use the means of variational distributions as the sole deterministic MC sample.
            # NOTE: here we don't need to sample the obs2prior weight H since we only compute the log-likelihood.
            # TODO: is this correct?
            sample_dict = dict()
            for coef_name, coef in self.coef_dict.items():
                sample_dict[coef_name] = coef.variational_distribution.mean.unsqueeze(
                    dim=0)  # (1, num_*, dim)
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
            return coef.device

    # ==================================================================================================================
    # helper functions.
    # ==================================================================================================================
    def sample_coefficient_dictionary(self, num_seeds: int) -> Dict[str, torch.Tensor]:
        """A helper function to sample parameters from coefficients.

        Args:
            num_seeds (int): number of random samples.

        Returns:
            Dict[str, torch.Tensor]: a dictionary maps coefficient names to tensor of sampled coefficient parameters,
                where the first dimension of the sampled tensor has size `num_seeds`.
                Each sample tensor has shape (num_seeds, num_classes, dim).
        """
        sample_dict = dict()
        for coef_name, coef in self.coef_dict.items():
            s = coef.rsample(num_seeds)
            if coef.obs2prior:
                # sample both obs2prior weight and realization of variable.
                assert isinstance(s, tuple) and len(s) == 2
                sample_dict[coef_name] = s[0]
                sample_dict[coef_name + '.H'] = s[1]
            else:
                # only sample the realization of variable.
                assert torch.is_tensor(s)
                sample_dict[coef_name] = s
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
        num_seeds = next(iter(sample_dict.values())).shape[0]

        # avoid repeated work when user purchased several items in the same session.
        user_session_index = torch.stack(
            [batch.user_index, batch.session_index])
        assert user_session_index.shape == (2, len(batch))
        unique_user_sess, inverse_indices = torch.unique(
            user_session_index, dim=1, return_inverse=True)

        user_index = unique_user_sess[0, :]
        session_index = unique_user_sess[1, :]
        assert len(user_index) == len(session_index)

        # short-hands for easier shape check.
        R = num_seeds
        # P = len(batch)  # num_purchases.
        P = unique_user_sess.shape[1]
        S = self.num_sessions
        U = self.num_users
        I = self.num_items
        NC = self.num_categories

        # ==============================================================================================================
        # Helper Functions for Reshaping.
        # ==============================================================================================================
        def reshape_user_coef_sample(C):
            # input shape (R, U, *)
            C = C.view(R, U, 1, -1).expand(-1, -1, I, -1)  # (R, U, I, *)
            C = C[:, user_index, :, :]
            assert C.shape == (R, P, I, positive_integer)
            return C

        def reshape_item_coef_sample(C):
            # input shape (R, I, *)
            C = C.view(R, 1, I, -1).expand(-1, P, -1, -1)
            assert C.shape == (R, P, I, positive_integer)
            return C

        def reshape_category_coef_sample(C):
            # input shape (R, NC, *)
            C = torch.repeat_interleave(C, self.category_to_size_tensor, dim=1)
            # input shape (R, I, *)
            C = C.view(R, 1, I, -1).expand(-1, P, -1, -1)
            assert C.shape == (R, P, I, positive_integer)
            return C

        def reshape_constant_coef_sample(C):
            # input shape (R, *)
            C = C.view(R, 1, 1, -1).expand(-1, P, I, -1)
            assert C.shape == (R, P, I, positive_integer)
            return C

        def reshape_coef_sample(sample, name):
            # reshape the monte carlo sample of coefficients to (R, P, I, *).
            if name.endswith('_user'):
                # (R, U, *) --> (R, P, I, *)
                return reshape_user_coef_sample(sample)
            elif name.endswith('_item'):
                # (R, I, *) --> (R, P, I, *)
                return reshape_item_coef_sample(sample)
            elif name.endswith('_category'):
                # (R, NC, *) --> (R, P, NC, *)
                return reshape_category_coef_sample(sample)
            elif name.endswith('_constant'):
                # (R, *) --> (R, P, I, *)
                return reshape_constant_coef_sample(sample)
            else:
                raise ValueError

        def reshape_observable(obs, name):
            # reshape observable to (R, P, I, *) so that it can be multiplied with monte carlo
            # samples of coefficients.
            O = obs.shape[-1]  # number of observables.
            assert O == positive_integer
            if name.startswith('item_'):
                assert obs.shape == (I, O)
                obs = obs.view(1, 1, I, O).expand(R, P, -1, -1)
            elif name.startswith('user_'):
                assert obs.shape == (U, O)
                obs = obs[user_index, :]  # (P, O)
                obs = obs.view(1, P, 1, O).expand(R, -1, I, -1)
            elif name.startswith('session_'):
                assert obs.shape == (S, O)
                obs = obs[session_index, :]  # (P, O)
                return obs.view(1, P, 1, O).expand(R, -1, I, -1)
            elif name.startswith('price_'):
                assert obs.shape == (S, I, O)
                obs = obs[session_index, :, :]  # (P, I, O)
                return obs.view(1, P, I, O).expand(R, -1, -1, -1)
            elif name.startswith('taste_'):
                assert obs.shape == (U, I, O)
                obs = obs[user_index, :, :]  # (P, I, O)
                return obs.view(1, P, I, O).expand(R, -1, -1, -1)
            else:
                raise ValueError
            assert obs.shape == (R, P, I, O)
            return obs

        # ==============================================================================================================
        # Copmute the Utility Term by Term.
        # ==============================================================================================================
        # P is the number of unique (user, session) pairs.
        # (random_seeds, P, num_items).
        utility = torch.zeros(R, P, I, device=self.device)

        # loop over additive term to utility
        for term in self.formula:
            # Type I: single coefficient, e.g., lambda_item or lambda_user.
            if len(term['coefficient']) == 1 and term['observable'] is None:
                # E.g., lambda_item or lambda_user
                coef_name = term['coefficient'][0]
                coef_sample = reshape_coef_sample(
                    sample_dict[coef_name], coef_name)
                assert coef_sample.shape == (R, P, I, 1)
                additive_term = coef_sample.view(R, P, I)

            # Type II: factorized coefficient, e.g., <theta_user, lambda_item>.
            elif len(term['coefficient']) == 2 and term['observable'] is None:
                coef_name_0 = term['coefficient'][0]
                coef_name_1 = term['coefficient'][1]

                coef_sample_0 = reshape_coef_sample(
                    sample_dict[coef_name_0], coef_name_0)
                coef_sample_1 = reshape_coef_sample(
                    sample_dict[coef_name_1], coef_name_1)

                assert coef_sample_0.shape == coef_sample_1.shape == (
                    R, P, I, positive_integer)

                additive_term = (coef_sample_0 * coef_sample_1).sum(dim=-1)

            # Type III: single coefficient multiplied by observable, e.g., theta_user * x_obs_item.
            elif len(term['coefficient']) == 1 and term['observable'] is not None:
                coef_name = term['coefficient'][0]
                coef_sample = reshape_coef_sample(
                    sample_dict[coef_name], coef_name)
                assert coef_sample.shape == (R, P, I, positive_integer)

                obs_name = term['observable']
                obs = reshape_observable(getattr(batch, obs_name), obs_name)
                assert obs.shape == (R, P, I, positive_integer)

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
                    R, P, I, positive_integer)
                num_obs_times_latent_dim = coef_sample_0.shape[-1]

                obs_name = term['observable']
                obs = reshape_observable(getattr(batch, obs_name), obs_name)
                assert obs.shape == (R, P, I, positive_integer)
                num_obs = obs.shape[-1]  # number of observables.

                assert (num_obs_times_latent_dim % num_obs) == 0
                latent_dim = num_obs_times_latent_dim // num_obs

                coef_sample_0 = coef_sample_0.view(
                    R, P, I, num_obs, latent_dim)
                coef_sample_1 = coef_sample_1.view(
                    R, P, I, num_obs, latent_dim)
                # compute the factorized coefficient with shape (R, P, I, O).
                coef = (coef_sample_0 * coef_sample_1).sum(dim=-1)

                additive_term = (coef * obs).sum(dim=-1)

            else:
                raise ValueError(f'Undefined term type: {term}')

            assert additive_term.shape == (R, P, I)
            utility += additive_term

        # ==============================================================================================================
        # Mask Out Unavailable Items in Each Session.
        # ==============================================================================================================

        if batch.item_availability is not None:
            # expand to the Monte Carlo sample dimension.
            # (S, I) -> (P, I) -> (1, P, I) -> (R, P, I)
            A = batch.item_availability[session_index, :].unsqueeze(
                dim=0).expand(R, -1, -1)
            utility[~A] = - (torch.finfo(utility.dtype).max / 2)

        utility = utility[:, inverse_indices, :]
        assert utility.shape == (R, len(batch), I)

        for module in self.additional_modules:
            additive_term = module(batch)
            assert additive_term.shape == (R, len(batch), 1)
            utility += additive_term.expand(-1, -1, I)

        if return_logit:
            # output shape: (num_seeds, len(batch), num_items)
            return utility
        else:
            # compute log likelihood log p(choosing item i | user, item latents)
            # compute log softmax separately within each category.
            if self.pred_item:
                # output shape: (num_seeds, len(batch), num_items)
                log_p = scatter_log_softmax(
                    utility, self.item_to_category_tensor, dim=-1)
            else:
                log_p = torch.nn.functional.logsigmoid(utility)
            return log_p

    def log_likelihood_item_index(self, batch: ChoiceDataset, return_logit: bool, sample_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
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
            return_logit(bool): if set to True, return the log-probability, otherwise return the logit/utility.
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
        num_seeds = next(iter(sample_dict.values())).shape[0]

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
        # ==========================================================================================
        # Helper Functions for Reshaping.
        # ==========================================================================================

        def reshape_coef_sample(sample, name):
            # reshape the monte carlo sample of coefficients to (R, P, I, *).
            if name.endswith('_user'):
                # (R, U, *) --> (R, total_computation, *)
                return sample[:, user_index, :]
            elif name.endswith('_item'):
                # (R, I, *) --> (R, total_computation, *)
                return sample[:, relevant_item_index, :]
            elif name.endswith('_category'):
                # (R, NC, *) --> (R, total_computation, *)
                return sample[:, repeat_category_index, :]
            elif name.endswith('_constant'):
                # (R, *) --> (R, total_computation, *)
                return sample.view(R, 1, -1).expand(-1, total_computation, -1)
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

        # compute log likelihood log p(choosing item i | user, item latents)
        if return_logit:
            log_p = utility
        else:
            if self.pred_item:
                # compute the log probability from logits/utilities.
                # output shape: (num_seeds, len(batch), num_items)
                log_p = scatter_log_softmax(utility, reverse_indices, dim=-1)
                # select the log-P of the item actually bought.
                log_p = log_p[:, item_index_expanded == relevant_item_index]
            else:
                # This is the binomial choice situation in which case we just report sigmoid log likelihood
                bce = nn.BCELoss(reduction='none')
                log_p = - bce(torch.sigmoid(utility.view(-1)), batch.label.to(torch.float32))
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
        num_seeds = next(iter(sample_dict.values())).shape[0]

        total = torch.zeros(num_seeds, device=self.device)

        for coef_name, coef in self.coef_dict.items():
            if self.obs2prior_dict[coef_name]:
                if coef_name.endswith('_item'):
                    x_obs = batch.item_obs
                elif coef_name.endswith('_user'):
                    x_obs = batch.user_obs
                else:
                    raise ValueError(
                        f'No observable found to support obs2prior for {coef_name}.')

                total += coef.log_prior(sample=sample_dict[coef_name],
                                        H_sample=sample_dict[coef_name + '.H'],
                                        x_obs=x_obs).sum(dim=-1)
            else:
                # log_prob outputs (num_seeds, num_{items, users}), sum to (num_seeds).
                total += coef.log_prior(
                    sample=sample_dict[coef_name], H_sample=None, x_obs=None).sum(dim=-1)

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
        sample_dict = self.sample_coefficient_dictionary(num_seeds)

        # ==============================================================================================================
        # 2. compute log p(latent) prior.
        # (num_seeds,) --mean--> scalar.
        elbo = self.log_prior(batch, sample_dict).mean(dim=0)
        # ==============================================================================================================

        # ==============================================================================================================
        # 3. compute the log likelihood log p(obs|latent).
        # sum over independent purchase decision for individual observations, mean over MC seeds.
        # the forward() function calls module.rsample(num_seeds) for module in self.additional_modules.
        # ==============================================================================================================
        if self.pred_item:
            # the prediction target is item_index.
            elbo += self.forward(batch,
                                 return_type='log_prob',
                                 return_scope='item_index',
                                 deterministic=False,
                                 sample_dict=sample_dict).sum(dim=1).mean(dim=0)  # (num_seeds, len(batch)) --> scalar.
        else:
            # the prediction target is binary.
            # TODO: update the prediction function.
            utility = self.forward(batch,
                                   return_type='utility',
                                   return_scope='item_index',
                                   deterministic=False,
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
            elbo -= self.log_variational(sample_dict).mean(dim=0)

        return elbo
