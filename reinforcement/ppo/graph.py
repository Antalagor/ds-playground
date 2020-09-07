from tensorflow.compat.v1 import reduce_mean, Graph, Session, placeholder, placeholder_with_default, identity, exp, clip_by_value, maximum, global_variables_initializer, \
    where, ones_like, is_nan, float32, string, ensure_shape, squeeze, regex_replace, cast, trainable_variables
from tensorflow.compat.v1.layers import Dense
from tensorflow.compat.v1.losses import mean_squared_error
from tensorflow.compat.v1.math import reduce_std
from tensorflow.compat.v1.saved_model import simple_save
from tensorflow.compat.v1.summary import scalar, merge, histogram
from tensorflow.compat.v1.train import Saver, AdamOptimizer, GradientDescentOptimizer
from tensorflow.io import decode_base64, decode_raw

from models.architecture.policy_header import GaussFull


def net_core(observation_ph, net_arch, initializer, activation):
    input_layer = observation_ph
    for layer_size in net_arch:
        input_layer = Dense(units=layer_size, activation=activation, kernel_initializer=initializer)(input_layer)
    return input_layer


def replace_nan(tensor, default):
    return where(is_nan(tensor), ones_like(tensor) * default, tensor)


class PpoGraph:
    """
    Proximal Policy Implementation in tensorflow. https://arxiv.org/abs/1707.06347 ("Proximal Policy Optimization Algorithms", J. Schulman et al, 2017)
    This class encapsulates all tensorflow interactions
    """

    def __init__(self, observation_size, net_arch, initializer, activation, clip_range, value_coef, entropy_coef, learning_rate, pre_training_learning_rate, action_bounds, policy):

        """
        :param observation_size:
        :param net_arch:
        :param initializer:
        :param activation:
        :param clip_range:
        :param value_coef:
        :param entropy_coef:
        :param learning_rate:
        :param pre_training_learning_rate:
        :param action_bounds:
        :param policy:
        """

        """Set class constants"""
        self.observation_size = observation_size
        self.net_arch = net_arch
        self.initializer = initializer
        self.activation = activation
        self.clip_range = clip_range
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        if action_bounds is None:
            action_bounds = [0.0, 1.5]
        self.action_bounds = action_bounds
        self.learning_rate = learning_rate
        self.pre_training_learning_rate = pre_training_learning_rate

        if policy is None:
            policy = GaussFull()
        self.policy = policy

        """Set up the tensorflow graph"""
        self.graph = Graph()

        with self.graph.as_default():
            self.sess = Session(graph=self.graph)

            """ core """
            # place holders
            self.observation_string_ph = placeholder(shape=(None, 1), dtype=string, name="observation_string_ph")
            self.action_ph = placeholder(dtype=float32, shape=(None, 1), name="action_ph")
            self.old_neg_logits = placeholder(dtype=float32, shape=(None, 1), name="old_neg_logits")
            self.advantage_ph = placeholder(dtype=float32, shape=(None, 1), name="advantage_ph")
            self.value_target_ph = placeholder(dtype=float32, shape=(None, 1), name="value_target_ph")
            # learning rate tensors
            self.learning_rate_ph = placeholder_with_default(input=self.learning_rate, shape=())
            self.pre_training_learning_rate_ph = placeholder_with_default(input=self.pre_training_learning_rate, shape=())

            # observation tensor
            replaced1 = regex_replace(self.observation_string_ph, "/", "_")
            replaced2 = regex_replace(replaced1, r"\+", "-")
            byte_tensor = decode_base64(replaced2)
            decoded = decode_raw(byte_tensor, out_type=float32)
            squeezed = squeeze(decoded, axis=1)
            self.observation_input = ensure_shape(squeezed, shape=(None, self.observation_size), name="observation_input")

            # policy net
            latent_policy = net_core(self.observation_input, self.net_arch, self.initializer, self.activation)
            self.policy.construct(latent_policy=latent_policy)

            self.clipped_action = clip_by_value(cast(self.policy.action, float32), self.action_bounds[0], self.action_bounds[1], "clipped_action")

            # value net
            latent_value = net_core(self.observation_input, self.net_arch, self.initializer, self.activation)
            self.value = identity(input=Dense(units=1, activation=None, kernel_initializer=self.initializer)(latent_value), name="value")

            """loss calculation"""
            # policy loss
            self.neg_logits = self.policy.neg_logits_from_actions(self.action_ph)
            ratio = exp(self.old_neg_logits - self.neg_logits)

            standardized_adv = (self.advantage_ph - reduce_mean(self.advantage_ph)) / (reduce_std(self.advantage_ph) + 1e-8)
            raw_policy_loss = - standardized_adv * ratio
            clipped_policy_loss = - standardized_adv * clip_by_value(ratio, 1 - self.clip_range, 1 + self.clip_range)
            self.policy_loss = reduce_mean(maximum(raw_policy_loss, clipped_policy_loss))

            self.value_loss = mean_squared_error(self.value_target_ph, self.value)

            # entropy loss
            self.entropy_loss = - reduce_mean(self.policy.entropy)

            # total loss
            self.total_loss = self.policy_loss + self.value_coef * self.value_loss + self.entropy_coef * self.entropy_loss

            # optimizer
            optimizer = AdamOptimizer(learning_rate=self.learning_rate_ph)

            # training ops
            self.training_op = optimizer.minimize(self.total_loss)

            # pre training
            self.dist_param_target_ph = placeholder(dtype=float32, shape=(None, self.policy.dist_params.shape[1]), name="dist_param_label_ph")
            self.pre_training_loss = mean_squared_error(self.dist_param_target_ph, self.policy.dist_params)
            pre_training_optimizer = GradientDescentOptimizer(learning_rate=self.pre_training_learning_rate_ph)
            self.pre_training_op = pre_training_optimizer.minimize(self.pre_training_loss)

            """utility nodes"""
            # inspect model weights
            self.trainable_variables = trainable_variables()

            # saviour
            self.saver = Saver()

            # tensorboard summaries
            self.summary = merge([
                histogram("values", self.value),
                histogram("advantages", standardized_adv),
                histogram("actions", self.clipped_action),
                histogram("det_actions", replace_nan(self.policy.det_action, 0.0)),
                histogram("value_targets", self.value_target_ph),
                scalar("policy_loss", self.policy_loss),
                scalar("value_loss", self.value_loss),
                scalar("entropy_loss", self.entropy_loss)
            ])

            self.pre_summary = merge([
                histogram("pretraining_actions", self.clipped_action),
                scalar("pretraining_loss", self.pre_training_loss)
            ])


            # initialization
            init = global_variables_initializer()
            self.sess.run(init)

    def predict(self, observation):
        """
        :param observation: input environment state
        :return: action, deterministic action (mode), negative log dist value, value prediction
        """

        fetches = [self.clipped_action, self.policy.dist_params, self.policy.neg_logits, self.value]
        action, dist_params, neg_logit, value = self.sess.run(fetches, {self.observation_input: observation})

        return action, dist_params, neg_logit, value

    def train_step(self, observations, actions, old_neg_logits, value_targets, advantages, obs_as_string=False, learning_rate=None, additional_fetches=None):
        fetches = [self.training_op, self.summary] + ([] if additional_fetches is None else additional_fetches)
        obs_tensor = self.observation_string_ph if obs_as_string else self.observation_input
        feed_dict = {obs_tensor: observations, self.action_ph: actions, self.old_neg_logits: old_neg_logits, self.value_target_ph: value_targets,
                     self.advantage_ph: advantages}

        if learning_rate is not None:
            feed_dict.update({self.learning_rate_ph: learning_rate})

        return self.sess.run(fetches, feed_dict)

    def pre_train_step(self, observations, dist_param_targets, obs_as_string=False, learning_rate=None, additional_fetches=None):
        fetches = [self.pre_training_op, self.pre_summary] + ([] if additional_fetches is None else additional_fetches)
        obs_tensor = self.observation_string_ph if obs_as_string else self.observation_input
        feed_dict = {obs_tensor: observations, self.dist_param_target_ph: dist_param_targets}

        if learning_rate is not None:
            feed_dict.update({self.pre_training_learning_rate_ph: learning_rate})

        return self.sess.run(fetches, feed_dict)

    def simple_save(self, path):
        with self.graph.as_default():
            simple_save(self.sess, path, inputs={"obs": self.observation_input}, outputs={"action": self.clipped_action})

    def save(self, path):
        with self.graph.as_default():
            self.saver.save(sess=self.sess, save_path=path)

    def restore(self, path):
        with self.graph.as_default():
            self.saver.restore(sess=self.sess, save_path=path)

    def close_session(self):
        self.sess.close()

    def get_trainable_variables(self):
        return self.sess.run(self.trainable_variables)
