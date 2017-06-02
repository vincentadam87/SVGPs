import tensorflow as tf
import numpy as np
from settings import float_type, jitter_level,std_qmu_init, np_float_type, np_int_type
from functions import eye, variational_expectations
from mean_functions import Zero
from kullback_leiblers import gauss_kl_white, gauss_kl_white_diag, gauss_kl, gauss_kl_diag
from conditionals import conditional
from quadrature import hermgauss


class ChainedGPs_DS(object):
    """
    Chained Gaussian Processes

    The key reference for this algorithm is:
    ::
      @article{saul2016chained,
        title={Chained Gaussian Processes},
        author={Saul, Alan D and Hensman, James and Vehtari, Aki and Lawrence, Neil D},
        journal={arXiv preprint arXiv:1604.05263},
        year={2016}
      }
    
    """
    def __init__(self, X, Y, kerns,likelihood,Zs,mean_functions=None, whiten=True,q_diag=False, f_indices=None,
                 n_samp=10):
        '''
        - X is a data matrix, size N x D
        - Y is a data matrix, size N x R
        - kerns, likelihood, mean_functions are appropriate (single or list of) GPflow objects
        - Zs is a list of  matrices of pseudo inputs, size M[k] x C
        - num_latent is the number of latent process to use, default to
          Y.shape[1]
        - q_diag is a boolean. If True, the covariance is approximated by a
          diagonal matrix.
        - whiten is a boolean. If True, we use the whitened representation of
          the inducing points.
        '''
        self.likelihood = likelihood
        self.kerns = kerns
        self.C = len(kerns)
        self.D = X.get_shape()[1]
        self.mean_functions = [Zero() for _ in range(self.C)] if mean_functions is None else mean_functions
        self.f_indices = f_indices # function of one variable
        self.X = X
        self.Y = Y
        self.Zs = Zs
        self.num_inducing = [z.get_shape()[0] for z in Zs]
        self.num_latent = Y.get_shape()[-1]
        self.num_data = Y.get_shape()[0]
        self.whiten=whiten
        self.q_diag = q_diag
        self.n_samp =n_samp
        self.initialize_inference()

    def initialize_inference(self):

        with tf.variable_scope("inference") as scope:
            self.q_mu,self.q_sqrt = [],[]
            for c in range(self.C):
                q_mu = np.ones((self.num_inducing[c], self.num_latent))
                #np.random.randn(self.num_inducing[c], self.num_latent)*std_qmu_init
                self.q_mu.append( tf.get_variable("q_mu%d"%c,[self.num_inducing[c], self.num_latent],\
                          initializer=tf.constant_initializer(q_mu,\
                                                              dtype=float_type)))

                if self.q_diag:
                    q_sqrt = np.ones((self.num_inducing[c], self.num_latent))
                    self.q_sqrt.append( tf.get_variable("q_sqrt%d"%c,[self.num_inducing[c],self.num_latent], \
                                          initializer=tf.constant_initializer(q_sqrt,dtype=float_type)) )

                else:
                    q_sqrt = np.array([np.eye(self.num_inducing[c]) for _ in range(self.num_latent)]).swapaxes(0, 2)
                    self.q_sqrt.append( tf.get_variable("q_sqrt%d"%c,[self.num_inducing[c],self.num_inducing[c],self.num_latent], \
                                          initializer=tf.constant_initializer(q_sqrt,dtype=float_type)) )

    def build_prior_KL(self):

        KL = tf.Variable(0,name='KL',trainable=False,dtype=float_type)
        for i in range(self.C):
            if self.whiten:
                if self.q_diag:
                    KL += gauss_kl_white_diag(self.q_mu[i], self.q_sqrt[i])
                else:
                    KL += gauss_kl_white(self.q_mu[i], self.q_sqrt[i])
            else:
                K = self.kerns[i].K(self.Zs[self.f_indices[i]]) + eye(self.num_inducing[i]) * jitter_level
                if self.q_diag:
                    KL += gauss_kl_diag(self.q_mu[i], self.q_sqrt[i], K)
                else:
                    KL += gauss_kl(self.q_mu[i], self.q_sqrt[i], K)
        return KL

    def get_covariate(self,Xnew,c):
        return tf.transpose(tf.gather(tf.transpose(Xnew),self.f_indices[c]))

    def build_predict_fs(self, Xnew):
        mus, vars = [],[]
        for c in range(self.C):
            x = self.get_covariate(Xnew,c)
            mu, var = conditional(x, self.Zs[c], self.kerns[c], self.q_mu[c],
                                     q_sqrt=self.q_sqrt[c], full_cov=False, whiten=self.whiten)
            mus.append(mu+self.mean_functions[c](x))
            vars.append(var)
        return tf.stack(mus),tf.stack(vars)

    def sample_fs(self, Xnew):
        fs_mean, fs_var = self.build_predict_fs( Xnew) # C x N x R

        samples_shape = fs_mean.get_shape().as_list() + [self.n_samp]
        return tf.random_normal(shape=samples_shape, dtype=float_type) * \
                  tf.sqrt(tf.expand_dims(fs_var, -1)) + tf.expand_dims(fs_mean, -1)

    def sample_predictor(self,Xnew):
        raise NotImplementedError

    def build_likelihood(self):
        """
        This gives a variational bound on the model likelihood.
        """
        cost = -self.build_prior_KL()
        samples_pred = self.sample_predictor(self.X)
        sto_exp = self.likelihood.logp(samples_pred,tf.expand_dims(self.Y,-1))
        cost += tf.reduce_sum(sto_exp)/self.n_samp
        return cost


class SVAGP_DS(ChainedGPs_DS):
    """
    Sparse Variational Additive Gaussian Process
    """

    def sample_predictor(self,Xnew):
        samples_fs =self.sample_fs(Xnew)
        return tf.reduce_sum(samples_fs,axis=0)


class SVMGP_DS(ChainedGPs_DS):
    """
    Sparse Variational Multiplicative Gaussian Process
    """

    def sample_predictor(self,Xnew):
        samples_fs =self.sample_fs(Xnew)
        return tf.reduce_prod(samples_fs,axis=0)


class SVGP(object):

    def __init__(self, X, Y, kern,likelihood,Z,mean_function=Zero(),num_latent=None, whiten=True,q_diag=True):

        self.likelihood = likelihood
        self.kern = kern
        self.mean_function = mean_function
        self.X = X
        self.Y = Y
        self.Z = Z
        self.D = X.get_shape()[1]
        self.num_latent = Y.get_shape()[1]
        self.num_inducing = Z.get_shape()[0]
        self.num_data = Y.get_shape()[0]
        self.whiten=whiten
        self.q_diag = q_diag

        self.q_mu =  tf.get_variable("q_mu",[self.num_inducing, self.num_latent],
                                     initializer=tf.constant_initializer(np.zeros((self.num_inducing, self.num_latent))))

        if self.q_diag:
            q_sqrt = np.ones((self.num_inducing, self.num_latent))
            self.q_sqrt= tf.get_variable("q_sqrt",[self.num_inducing,self.num_latent], \
                                  initializer=tf.constant_initializer(q_sqrt,dtype=float_type))
        else:
            q_sqrt = np.array([np.eye(self.num_inducing) for _ in range(self.num_latent)]).swapaxes(0, 2)
            self.q_sqrt= tf.get_variable("q_sqrt",[self.num_inducing,self.num_inducing,self.num_latent], \
                                  initializer=tf.constant_initializer(q_sqrt,dtype=float_type))


    def build_prior_KL(self):

        if self.whiten:
            if self.q_diag:
                KL = gauss_kl_white_diag(self.q_mu, self.q_sqrt)
            else:
                KL = gauss_kl_white(self.q_mu, self.q_sqrt)
        else:
            K = self.kern.K(self.Z) + eye(self.num_inducing) * jitter_level
            if self.q_diag:
                KL = gauss_kl_diag(self.q_mu, self.q_sqrt, K)
            else:
                KL = gauss_kl(self.q_mu, self.q_sqrt, K)
        return KL


    def build_predictor(self,Xnew, full_cov=False):
        return self.build_predict(Xnew, full_cov=full_cov)


    def build_likelihood(self):
        """
        This gives a variational bound on the model likelihood.
        """

        # Get prior KL.
        KL = self.build_prior_KL()

        # Get conditionals
        fmean, fvar = self.build_predictor(self.X, full_cov=False)

        # Get variational expectations.
        var_exp = self.likelihood.variational_expectations(fmean, fvar, self.Y)

        # re-scale for minibatch size
        scale = tf.cast(self.num_data, float_type) / \
                tf.cast(tf.shape(self.X)[0], float_type)

        return tf.reduce_sum(var_exp) * scale - KL


    def build_predict(self, Xnew, full_cov=False):
        mu, var = conditional(Xnew, self.Z, self.kern, self.q_mu,
                                           q_sqrt=self.q_sqrt, full_cov=full_cov, whiten=self.whiten)
        return mu + self.mean_function(Xnew), var


class SVGP_shared(SVGP):

    def build_predictor(self,Xnew, full_cov=False):
        X_ =tf.reshape(Xnew,[-1,1])
        fmean, fvar = [tf.reshape(u,tf.shape(Xnew)) for u in self.build_predict(X_, full_cov=full_cov)]
        return tf.reduce_sum(fmean,-1,keep_dims=True),tf.reduce_sum(fvar,-1,keep_dims=True)