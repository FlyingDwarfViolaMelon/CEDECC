import numpy as np
import keras.backend as K
from keras.engine.topology import Layer, InputSpec
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import SGD, RMSprop, Adam
from sklearn.cluster import KMeans
from keras.layers import BatchNormalization
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras import objectives
import scipy.io as sio
import math
import random
from sklearn.preprocessing import MinMaxScaler


def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def vae_loss(x, x_decoded_mean):
    xent_loss = objectives.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
    return xent_loss + kl_loss


class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: degrees of freedom parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
         Measure the similarity between embedded point z_i and centroid µ_j.
                 q_ij = 1/(1+dist(x_i, µ_j)^2), then normalize it.
                 q_ij can be interpreted as the probability of assigning sample i to cluster j.
                 (i.e., a soft assignment)
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))  # Make sure each sample's 10 values add up to 1.
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# computing an auxiliary target distribution
def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T

# computing cluster confidence
def cluster_confidence(q, delta):
    index = q.argmax(1)
    index_set = set(index)
    list_set = list(index_set)
    category = np.array(list_set)
    cc = 0
    for c in range(len(category)):
        cluster_c = np.where(index == category[c])
        cluster_c = cluster_c[0]
        qc = q[cluster_c, :]
        qc_number = qc.shape[0]
        Qmax = []
        Qsecond = []
        for j in range(qc_number):
            q_sort = sorted(qc[j, :])
            qmax = q_sort[-1]
            qsecond = q_sort[-2]
            Qmax = np.append(Qmax, qmax)
            Qsecond = np.append(Qsecond, qsecond)
        cc = cc + 1/(1 + math.exp(-(sum(Qmax)/sum(Qsecond)) * delta))
    return cc

# main function
filename = 'cars.mat'  # dataset name
save_tmp = 'cars_result.mat' # result save name
label_number = sio.loadmat(filename)['fea'].shape[0]
original_dim = sio.loadmat(filename)['fea'].shape[1]

# dataset contains fea and gt matrix
xy = np.column_stack((sio.loadmat(filename)['fea'], sio.loadmat(filename)['gt']))
X = xy[0:label_number, 0:original_dim]
X = MinMaxScaler().fit_transform(X)
# X = np.divide(X, 255.)
y = xy[0:label_number, original_dim]
y = y[:label_number]
print('Data loaded successfully')

dims = [original_dim, 8, 6]  # layer parameter
rounds = 10  # run time [10, 100]
tol = 0.01  # tolerance threshold to stop training [0.001, 0.1]
pretrain_epochs = 30  # [30, 150]
batch_size = 128  # [128, 256]
delta = 1  # cluster confidence adjustment coefficient [0.6, 1]

cluster_confidence_Before = []
cluster_confidence_After = []
cluster_result = np.zeros([label_number, 1])

for k in range(rounds):
    # n_clusters = math.floor(2 + (math.sqrt(label_number) - 2) * random.uniform(0, 1))  # [2, sqrt(N)]
    n_clusters = 3
    n_stacks = len(dims) - 1
    x = Input(shape=(dims[0],), name='input')
    h = x

    # internal layers in encoder
    for i in range(n_stacks-1):
        h = Dense(dims[i + 1], activation='relu', kernel_initializer='glorot_uniform', name='encoder_%d' % i)(h)
        #h = BatchNormalization()(h)
        #h = Dropout(0.5)(h)

    # hidden layer
    z_mean = Dense(dims[-1], kernel_initializer='glorot_uniform', name='encoder_%d' % (n_stacks - 1))(h)
    #z_mean = BatchNormalization()(z_mean)
    #z_mean = Dropout(0.5)(z_mean)

    z_log_sigma = Dense(dims[-1], kernel_initializer='glorot_uniform')(z_mean)
    # z_log_sigma = BatchNormalization()(z_log_sigma)
    # z_log_sigma = Dropout(0.5)(z_log_sigma)

    # note that "output_shape" isn't necessary with the TensorFlow backend
    # so you could write `Lambda(sampling)([z_mean, z_log_sigma])`
    # z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])
    z = Lambda(sampling)([z_mean, z_log_sigma])

    m = z
    # internal layers in decoder
    for i in range(n_stacks-1, 0, -1):
        m = Dense(dims[i], activation='relu', kernel_initializer='glorot_uniform', name='decoder_%d' % i)(m)

    # output
    m = Dense(dims[0], activation='sigmoid', kernel_initializer='glorot_uniform', name='decoder_0')(m)

    # end-to-end autoencoder
    vae = Model(x, m)

    # encoder, from inputs to latent space
    encoder = Model(x, z_mean)

    autoencoder, encoder = vae, encoder

    # Pretrain VAE autoencoder
    print('rounds: ' + str(k + 1))
    save_interval = (x.shape[0] // batch_size) * 5
    autoencoder.compile(optimizer='adam', loss=vae_loss)
    autoencoder.fit(X, X, batch_size=batch_size, epochs=pretrain_epochs)
    clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoder.output)
    model = Model(inputs=encoder.input, outputs=clustering_layer)
    model.compile(optimizer='adam', loss='kld')
    model.summary()
    # initialize cluster centers using k-means
    kmeans = KMeans(n_clusters=n_clusters, n_init=200)
    y_pred_kmeans = kmeans.fit_predict(encoder.predict(X))
    if min(y_pred_kmeans) == 0:
        y_pred_kmeans = y_pred_kmeans + 1
    y_pred_last = np.copy(y_pred_kmeans)
    model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

    # deep clustering
    loss = 0
    index = 0
    maxiter = 10000  # [10000, 20000]
    update_interval = 140  # [100, 200]
    index_array = np.arange(X.shape[0])
    q_before = model.predict(X, verbose=0)
    cluster_confidence_before = cluster_confidence(q_before, delta)
    # Start training
    for ite in range(int(maxiter)):
        if ite % update_interval == 0:
            q = model.predict(X, verbose=0)
            p = target_distribution(q)  # update the auxiliary target distribution p

            # evaluate the clustering performance
            y_pred = q.argmax(1)

            # check stop criterion
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
            y_pred_last = np.copy(y_pred)
            if ite > 0 and delta_label < tol:
                print('delta_label ', delta_label, '< tol ', tol)
                print('Reached tolerance threshold. Stopping training.')
                break
        idx = index_array[index * batch_size: min((index + 1) * batch_size, X.shape[0])]
        loss = model.train_on_batch(x=X[idx], y=p[idx])
        index = index + 1 if (index + 1) * batch_size <= X.shape[0] else 0

    # Final Evaluation
    q = model.predict(X, verbose=0)
    cluster_confidence_after = cluster_confidence(q, delta)
    p = target_distribution(q)  # update the auxiliary target distribution p
    # evaluate the clustering performance
    y_pred = q.argmax(1)
    y_pred = y_pred[:label_number]

    if min(y_pred) == 0:
        y_pred = y_pred + 1

    if y is not None:
        loss = np.round(loss, 5)
        y_pred_last = y_pred
        y_pred_last = np.reshape(y_pred_last, (label_number, 1))
        cluster_result = np.c_[cluster_result, y_pred_last]
        cluster_confidence_Before = np.append(cluster_confidence_Before,cluster_confidence_before)
        cluster_confidence_After = np.append(cluster_confidence_After, cluster_confidence_after)

print('Running out!!')
cluster_result = np.delete(cluster_result, [0], axis=1)
sio.savemat(save_tmp, {'cluster_confidence_Before':cluster_confidence_Before,'cluster_confidence_After':cluster_confidence_After,'cluster_result':cluster_result,'gt':y})
