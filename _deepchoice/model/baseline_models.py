"""
Collection of baseline models.
"""
import torch
import torch.nn as nn
import math
from torch.nn import functional as F


class BaseMNLNoMask(nn.Module):
    """
        Simplest multinomial logit model. Considers all possible output classes, regardless of availability. Features
        used are user-specific and the coefficients are item-specific.
    """

    def __init__(self, feature_dim, class_dim):
        super(BaseMNLNoMask, self).__init__()
        self.feature_dim = feature_dim
        self.class_dim = class_dim
        # The linear module allows item-specific coefficients for user features
        self.linear = nn.Linear(feature_dim, class_dim)

    def forward(self, x):
        utility = self.linear(x)
        return utility


class BaseMNL(nn.Module):
    """
        Simplest multinomial logit model with masking. In other words, only allows predictions for those classes for
        which a particular session has availability. Features used are user-specific and the coefficients are
        item-specific.
    """

    def __init__(self, feature_dim, class_dim):
        super(BaseMNL, self).__init__()
        self.feature_dim = feature_dim
        self.class_dim = class_dim
        # The linear module allows item-specific coefficients for user features
        self.linear = nn.Linear(feature_dim, class_dim)

    def forward(self, x, mask):
        utility = self.linear(x)
        # Set utility to large negative value if not in availability list
        utility[~mask] = -1.0e20
        return utility


class DeepMNL(nn.Module):
    """
        A "deep" multinomial logit model. This one has num_layers number of hidden layers between input and output to
        allow for a deeper representation of the response surface. Besides the hidden layers, this is otherwise
        identical to BaseMNL.
    """

    def __init__(self, feature_dim, class_dim, size_layer, num_layers):
        super(DeepMNL, self).__init__()
        self.feature_dim = feature_dim
        self.class_dim = class_dim
        self.size_layer = size_layer
        self.num_layers = num_layers

        # The linear module allows item-specific coefficients for user features
        self.linear_in = nn.Linear(feature_dim, size_layer)
        self.hidden_layers = nn.ModuleList()
        # Add num_layers number of hidden layers of the same width size_layer
        for _ in range(num_layers):
            self.hidden_layers.append(nn.Linear(size_layer, size_layer))
        self.linear_out = nn.Linear(size_layer, class_dim)

    def forward(self, x, mask):
        # All layers use the ReLU activation function
        hidden = F.relu(self.linear_in(x))
        for layer in self.hidden_layers:
            hidden = F.relu(layer(hidden))
        utility = self.linear_out(hidden)
        # Set utility to large negative value if not in availability list
        utility[~mask] = -1.0e20
        return utility


class InterceptMNL(nn.Module):
    """
        This is an intercept MNL model. That means the input features are one-hots for persona_1 and the outputs
        are probabilities of reaching persona_2. Since this could easily become intractible with millions of
        parameters to train, we mask out most of them using a persona_1 to persona_2 availability list.

        __init__ arguments:
        coef_mask -- a tensor of dimension feature_dim (number of persona_1) by class_dim (number of persona_2).
        It is an availability list at the persona_1 to persona_2 transition level.

        forward arguments:
        mask -- a tensor of dimension x.size(1) by class_dim. It is an availability list at the session to
        persona_2 transition level.
    """

    def __init__(self, feature_dim, class_dim, coef_mask):
        super(InterceptMNL, self).__init__()
        self.feature_dim = feature_dim
        self.class_dim = class_dim
        self.coef_mask = coef_mask
        # The linear module allows item-specific coefficients for user features
        self.linear = nn.Linear(feature_dim, class_dim)
        self.linear.weight.data.mul_(coef_mask.t())  # Inplace mask weights

    def forward(self, x, mask):
        utility = self.linear(x)
        # Set utility to large negative value if not in availability list
        utility[~mask] = -1.0e20
        return utility


class InertiaMNL(nn.Module):
    """
        A baseline multinomial logit model with inertia.
    """

    def __init__(self):
        super(InertiaMNL, self).__init__()
        self.inertia_coef = nn.Parameter(torch.randn(1).requires_grad_(True))
        # The linear module allows item-specific coefficients for user features

    def forward(self, x_inertia, mask):
        utility = self.inertia_coef * x_inertia
        # Set utility to large negative value if not in availability list
        utility[~mask] = -1.0e20
        return utility


class ConditionalLogit(nn.Module):
    """
        A baseline conditional logit model. Here we use features of _choices_ (x_choice), the conditional part, and it
        uses the same coefficient for all choices within the same feature. We also use features of user (x_user), the
        multinomial logit part, and they have item-specific coefficients. This model also allows for varying choice
        sets through mask.
    """

    def __init__(self, user_feature_dim, choice_feature_dim, class_dim):
        super(ConditionalLogit, self).__init__()
        self.user_feature_dim = user_feature_dim
        self.choice_feature_dim = choice_feature_dim
        self.class_dim = class_dim
        # The linear module allows item-specific coefficients for user features
        self.linear = nn.Linear(user_feature_dim, class_dim)
        # We use choice_feature_dim number of conditional logit parameters. Each one of them applies to a single
        # feature.
        # TODO(Tianyu): No price term here? already included in the x_choice?
        self.conditional_coef = nn.Parameter(torch.randn(choice_feature_dim, 1).requires_grad_(True))

    def forward(self, x_user, x_choice, mask):
        mnl_utility = self.linear(x_user)
        # Just in case NaNs pop up, set them to zero
        # x_choice[torch.isnan(x_choice)] = 0.0
        # All possible choices share the same conditional logit coefficient for that feature.
        # That is, coefficients vary by feature but not by choice.
        # We achieve this by broadcasting the element-wise multiplication.
        conditional_utility = (self.conditional_coef * x_choice).sum(dim=1)  # dim=1 corresponds to choice_feature_dim
        utility = mnl_utility + conditional_utility
        # Set utility to large negative value if not in availability list
        utility[~mask] = -1.0e20
        return utility


class InterceptConditionalLogit(nn.Module):
    """
        An intercept conditional logit model. This includes 3 components:
        1) an MNL part using persona_1 one-hots with a coefficient mask to zero out coefficients
        not in the availability list,
        2) an MNL part for user features,
        3) a conditional logit part with persona_2 and price features.

        __init__ arguments:
        user_intercept_dim -- number of persona_1. The categorical variable persona_1 is represented
        as one-hots.
        user_feature_dim -- number of persona_1 level features
        choice_feature_dim -- number of persona_2 level features
        class_dim -- number of persona_2
        coef_mask -- a tensor of dimension feature_dim (number of persona_1) by class_dim (number of persona_2).
        It is an availability list at the persona_1 to persona_2 transition level.

        forward arguments:
        x_intercept -- tensor whose size is num_sessions x user_intercept_dim. It is a one-hot expansion of
        the cateogorical variable, persona_1.
        x_user -- tensor of size num_sessions x user_feature_dim. These are the persona_1 features
        x_choice -- tensor of size num_sessions x choice_feature_dim x class_dim. These are the "widened"
        persona_2 features.
        mask -- tensor of size num_sessions x class_dim. This represents the session to persona_2 availability list.
    """

    def __init__(self, user_intercept_dim, user_feature_dim, choice_feature_dim, class_dim, coef_mask):
        super(InterceptConditionalLogit, self).__init__()
        self.user_intercept_dim = user_intercept_dim
        self.user_feature_dim = user_feature_dim
        self.choice_feature_dim = choice_feature_dim
        self.class_dim = class_dim
        self.coef_mask = coef_mask
        # The linear module allows item-specific coefficients for user features
        self.linear1 = nn.Linear(user_intercept_dim, class_dim)
        self.linear1.weight.data.mul_(coef_mask.t())  # Inplace mask weights
        self.linear2 = nn.Linear(user_feature_dim, class_dim)
        # We use choice_feature_dim number of conditional logit parameters. Each one of them applies to a single
        # feature.
        self.conditional_coef = nn.Parameter(torch.randn(choice_feature_dim, 1).requires_grad_(True))

    def forward(self, x_intercept, x_user, x_choice, mask):
        mnl_utility = self.linear1(x_intercept) + self.linear2(x_user)
        # Just in case NaNs pop up, set them to zero
        # x_choice[torch.isnan(x_choice)] = 0.0
        # All possible choices share the same conditional logit coefficient for that feature.
        # That is, coefficients vary by feature but not by choice.
        # We achieve this by broadcasting the element-wise multiplication.
        conditional_utility = (self.conditional_coef * x_choice).sum(dim=1)  # dim=1 corresponds to choice_feature_dim
        utility = mnl_utility + conditional_utility
        # Set utility to large negative value if not in availability list
        utility[~mask] = -1.0e20
        return utility


class InertiaConditionalLogit(nn.Module):
    """
        A conditional logit model with inertia.
    """

    def __init__(self, user_feature_dim, choice_feature_dim, class_dim):
        super(InertiaConditionalLogit, self).__init__()
        self.user_feature_dim = user_feature_dim
        self.choice_feature_dim = choice_feature_dim
        self.class_dim = class_dim
        self.inertia_coef = nn.Parameter(torch.randn(1).requires_grad_(True))
        # The linear module allows item-specific coefficients for user features
        self.linear = nn.Linear(user_feature_dim, class_dim)
        # We use choice_feature_dim number of conditional logit parameters. Each one of them applies to a single
        # feature.
        self.conditional_coef = nn.Parameter(torch.randn(choice_feature_dim, 1).requires_grad_(True))

    def forward(self, x_inertia, x_user, x_choice, mask):
        inertia_utility = self.inertia_coef * x_inertia
        mnl_utility = self.linear(x_user)
        # All possible choices share the same conditional logit coefficient for that feature.
        # That is, coefficients vary by feature but not by choice.
        # We achieve this by broadcasting the element-wise multiplication.
        conditional_utility = (self.conditional_coef * x_choice).sum(dim=1)  # dim=1 corresponds to choice_feature_dim
        utility = inertia_utility + mnl_utility + conditional_utility
        # Set utility to large negative value if not in availability list
        utility[~mask] = -1.0e20
        return utility


class LeanConditionalLogit(nn.Module):
    """
        Similar to ConditionalLogit but reduces memory consumption by separating clogit features that depend only on
        persona_2 (and therefore don't need to be copied for every session) and those that depend both on persona_2 and
        session (which will be treated as before).

        __init__ arguments:
        user_feature_dim -- number of persona_1 level features
        price_dim -- number of price features, i.e. those that vary by both session and persona_2
        choice_feature_dim -- number of persona_2 level features
        class_dim -- number of persona_2

        forward arguments:
        x_user -- tensor of size num_sessions x user_feature_dim. These are the persona_1 features
        x_price -- tensor of size num_sessions x price_dim x class_dim. These are the "widened" price features.
        x_choice -- tensor of size choice_feature_dim x class_dim. These are the persona_2 features independent of
        sessions.
        mask -- tensor of size num_sessions x class_dim. This represents the session to persona_2 availability list.
    """

    def __init__(self, user_feature_dim, price_dim, choice_feature_dim, class_dim):
        super().__init__()
        self.user_feature_dim = user_feature_dim
        self.price_dim = price_dim
        self.choice_feature_dim = choice_feature_dim
        self.class_dim = class_dim
        # The linear module allows item-specific coefficients for user features
        self.linear = nn.Linear(user_feature_dim, class_dim)
        # We use choice_feature_dim number of conditional logit parameters. Each one of them applies to a single
        # feature.
        # TODO(Tianyu): Shape should be (1, price_dim, 1) for broadcasting.
        self.price_coef = nn.Parameter(torch.randn(price_dim, 1).requires_grad_(True))
        self.conditional_coef = nn.Parameter(torch.randn(choice_feature_dim, 1).requires_grad_(True))

    def forward(self, x_user, x_price, x_choice, mask):
        mnl_utility = self.linear(x_user)
        # All possible choices share the same conditional logit coefficient for that feature.
        # That is, coefficients vary by feature but not by choice.
        # We achieve this by broadcasting the element-wise multiplication.
        price_utility = (self.price_coef * x_price).sum(dim=1)  # dim=1 corresponds to price_dim
        # dim=0 corresponds to choice_feature_dim
        conditional_utility = (self.conditional_coef * x_choice).sum(dim=0).unsqueeze(dim=0)

        utility = mnl_utility + price_utility + conditional_utility
        # Set utility to large negative value if not in availability list
        utility[~mask] = -1.0e20
        return utility
