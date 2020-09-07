import tensorflow.compat.v1 as tf1
import tensorflow_probability as tfp
from tensorflow.compat.v1.initializers import constant
from tensorflow.compat.v1.layers import Dense


def rename_tensor(tensor, name):
    # tricksery from https://medium.com/@jsflo.dev/saving-and-loading-a-tensorflow-model-using-the-savedmodel-api-17645576527
    return tf1.identity(input=tensor, name=name)


class PolicyHeader:
    def construct(self, latent_policy):
        self.dist_params = rename_tensor(self.setup_dist_params(latent_policy), "dist_params")
        self._distribution = self.setup_distribution()
        self.action = rename_tensor(self._distribution.sample(), "action")
        self.det_action = rename_tensor(tf1.cast(self._distribution.mode(), tf1.float32), "det_action")
        self.neg_logits = rename_tensor(self.neg_logits_from_actions(self.action), "neg_logits")
        self.entropy = rename_tensor(self._distribution.entropy(), "entropy")

    def neg_logits_from_actions(self, action_tensor):
        return - self._distribution.log_prob(action_tensor)

    def setup_dist_params(self, latent_policy):
        raise NotImplementedError

    def setup_distribution(self):
        raise NotImplementedError


class Gauss(PolicyHeader):
    def setup_dist_params(self, latent_policy):
        raise NotImplementedError

    def setup_distribution(self):
        mean, log_std = tf1.split(axis=1, num_or_size_splits=2, value=self.dist_params)
        return tfp.distributions.Normal(loc=mean, scale=tf1.exp(log_std))


class GaussFull(Gauss):
    def setup_dist_params(self, latent_policy):
        return Dense(units=2, activation=None, kernel_initializer="he_uniform")(latent_policy)


class GaussConstrained(Gauss):
    def setup_dist_params(self, latent_policy):
        mean = Dense(units=1, activation="sigmoid", kernel_initializer="he_uniform")(latent_policy)
        neg_log_std = Dense(units=1, activation="sigmoid", kernel_initializer="he_uniform")(latent_policy)
        return tf1.concat([mean, -neg_log_std], axis=1)


class GaussConstVariance(Gauss):
    def setup_dist_params(self, latent_policy):
        mean = Dense(units=1, activation=None, kernel_initializer="he_uniform", name="gaussian_mean")(latent_policy)
        log_std = tf1.get_variable("log_std", initializer=[[0.]], trainable=True)
        # broadcast and concat
        dist_params = tf1.concat([mean, mean * 0 + log_std], axis=1)
        return dist_params


class Binomial(PolicyHeader):
    def setup_dist_params(self, latent_policy):
        proba = Dense(units=1, activation=tf1.sigmoid, kernel_initializer="zeros")(
                latent_policy)
        return proba

    def setup_distribution(self):
        return tfp.distributions.Bernoulli(probs=self.dist_params)


class Beta(PolicyHeader):
    def setup_dist_params(self, latent_policy):
        log_params = Dense(units=2, activation=None, kernel_initializer=constant(value=1.0e-10))(latent_policy)
        return tf1.add(tf1.exp(log_params), 1)

    def setup_distribution(self):
        a, b = tf1.split(axis=1, num_or_size_splits=2, value=self.dist_params)
        return tfp.distributions.Beta(concentration0=a, concentration1=b, name="action_dist")


class Gamma(PolicyHeader):
    def setup_dist_params(self, latent_policy):
        log_params = Dense(units=2, activation=None, kernel_initializer=constant(value=1.0e-10))(latent_policy)
        return tf1.exp(log_params)

    def setup_distribution(self):
        a, b = tf1.split(axis=1, num_or_size_splits=2, value=self.dist_params)
        return tfp.distributions.Gamma(concentration=a, rate=b, name="action_dist")


class LogNormal(PolicyHeader):
    def setup_dist_params(self, latent_policy):
        mean = Dense(units=1, activation=None, kernel_initializer="he_uniform", name="gaussian_mean")(latent_policy)
        log_std = tf1.get_variable("log_std", initializer=[[-1.0]], trainable=True)
        # broadcast and concat
        dist_params = tf1.concat([mean, mean * 0 + log_std], axis=1)
        return dist_params

    def setup_distribution(self):
        mean, log_std = tf1.split(axis=1, num_or_size_splits=2, value=self.dist_params)
        return tfp.distributions.LogNormal(loc=mean, scale=tf1.exp(log_std))
