import tensorflow as tf


def bpr_loss(user_emb, pos_item_emb, neg_item_emb):
    score = tf.reduce_sum(tf.multiply(user_emb, pos_item_emb), 1) - tf.reduce_sum(tf.multiply(user_emb, neg_item_emb), 1)
    loss = -tf.reduce_sum(tf.log(tf.sigmoid(score) + 10e-8))
    return loss

import tensorflow as tf
import tensorflow_probability as tfp

class QuantileBPRLoss(tf.keras.Model):
    def __init__(self, n_items, loss_func):
        super().__init__()
        self.loss_func = loss_func
        self.kappa_fixed = 1.0
        self.kappa = self.add_weight(
            name="kappa",
            shape=(n_items,),
            initializer="ones",
            trainable=True
        )
        self.theta = self.add_weight(
            name="theta",
            shape=(n_items,),
            initializer="zeros",
            trainable=True
        )
        print(f'[DEBUG] self.loss_func: {self.loss_func}')

    def calculate_cdf(self, item_ids, days, scales, shapes, all_items=True):
        days = tf.cast(days, tf.float32)
        scales = tf.cast(scales, tf.float32)
        shapes = tf.cast(shapes, tf.float32)

        dist = tfp.distributions.Weibull(scale=scales, concentration=shapes)
        cdf_vals = dist.cdf(days)

        if not all_items:
            mask = shapes < 1
            return tf.where(mask, -tf.ones_like(cdf_vals), cdf_vals)
        else:
            return cdf_vals

    def calculate_inverse_cdf(self, item_ids, cdf_values, scales, shapes, all_items=True):
        cdf_values = tf.cast(cdf_values, tf.float32)
        scales = tf.cast(scales, tf.float32)
        shapes = tf.cast(shapes, tf.float32)

        cdf_values = tf.clip_by_value(cdf_values, 1e-6, 1 - 1e-6)
        dist = tfp.distributions.Weibull(scale=scales, concentration=shapes)
        icdf_vals = dist.quantile(cdf_values)

        if not all_items:
            mask = shapes < 1
            return tf.where(mask, -tf.ones_like(icdf_vals), icdf_vals)
        else:
            return icdf_vals

    def call(self, user_emb, pos_item_emb, neg_item_emb, pos_idx, neg_idx, itts, scales, shapes):
        itts = tf.cast(itts, tf.float32)

        pos_cdfs = self.calculate_cdf(pos_idx, itts, scales, shapes)
        qs = tf.sigmoid(self.theta)
        pos_qs = tf.gather(qs, pos_idx)
        quantile_thresholds = self.calculate_inverse_cdf(pos_idx, pos_qs, scales, shapes)
        pos_kappa = tf.gather(self.kappa, pos_idx)

        if self.loss_func == 0:
            pos_scores = tf.reduce_sum(user_emb * pos_item_emb, axis=1)
        elif self.loss_func == 1:
            pos_weights = tf.sigmoid(self.kappa_fixed * (itts - quantile_thresholds))
            pos_scores = tf.reduce_sum(user_emb * pos_item_emb * tf.expand_dims(pos_weights, axis=1), axis=1)
        elif self.loss_func == 2:
            pos_weights = tf.sigmoid(pos_kappa * (itts - quantile_thresholds))
            pos_scores = tf.reduce_sum(user_emb * pos_item_emb * tf.expand_dims(pos_weights, axis=1), axis=1)
        elif self.loss_func == 3:
            pos_weights = tf.sigmoid(pos_kappa * (itts / (quantile_thresholds + 1e-6)))
            pos_scores = tf.reduce_sum(user_emb * pos_item_emb * tf.expand_dims(pos_weights, axis=1), axis=1)
        elif self.loss_func == 4:
            pos_weights = tf.sigmoid(pos_kappa * (pos_cdfs - pos_qs))
            pos_scores = tf.reduce_sum(user_emb * pos_item_emb * tf.expand_dims(pos_weights, axis=1), axis=1)
        elif self.loss_func == 5:
            pos_weights = tf.sigmoid(pos_kappa * (pos_cdfs / (pos_qs + 1e-6)))
            pos_scores = tf.reduce_sum(user_emb * pos_item_emb * tf.expand_dims(pos_weights, axis=1), axis=1)
        else:
            raise ValueError("loss_func is not defined.")

        neg_scores = tf.reduce_sum(user_emb * neg_item_emb, axis=1)
        loss = -tf.math.log(1e-6 + tf.sigmoid(pos_scores - neg_scores))

        return tf.reduce_mean(loss)


def InfoNCE(view1, view2, temperature):
    pos_score = tf.reduce_sum(tf.multiply(view1, view2), axis=1)
    ttl_score = tf.matmul(view1, view2, transpose_a=False, transpose_b=True)
    pos_score = tf.exp(pos_score / temperature)
    ttl_score = tf.reduce_sum(tf.exp(ttl_score / temperature), axis=1)
    cl_loss = -tf.reduce_sum(tf.log(pos_score / ttl_score))
    return cl_loss


# Sampled Softmax
def ssm_loss(user_emb, pos_item_emb, neg_item_emb):
    user_emb = tf.nn.l2_normalize(user_emb, 1)
    pos_item_emb = tf.nn.l2_normalize(pos_item_emb, 1)
    neg_item_emb = tf.nn.l2_normalize(neg_item_emb, 1)
    pos_score = tf.reduce_sum(tf.multiply(user_emb, pos_item_emb), 1)
    ttl_score = tf.matmul(user_emb, neg_item_emb, transpose_a=False, transpose_b=True)
    ttl_score = tf.concat([tf.reshape(pos_score, (-1, 1)), ttl_score], axis=1)
    pos_score = tf.exp(pos_score / 0.2)
    ttl_score = tf.reduce_sum(tf.exp(ttl_score / 0.2), axis=1)
    return -tf.reduce_mean(tf.log(pos_score / ttl_score))
