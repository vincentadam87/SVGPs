import tensorflow as tf
import numpy as np
from settings import float_type, jitter_level,std_qmu_init, np_float_type, np_int_type
from functions import eye, variational_expectations, block_diagonal
from mean_functions import Zero
from kullback_leiblers import gauss_kl_white, gauss_kl_white_diag, gauss_kl, gauss_kl_diag
from conditionals import conditional, conditional_batch, conditional_stack
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
    
    Independent latent GPs : f_c ~ GP(m_c,k_c), for all c
    Mean field posterior : q(F,U)=\prod_c q_c(f_c,u_c), 
    Sparse GP posterior : q_c(f_c,u_c) = p_c(f_c|u_c)q(u_c), for all c
    Arbitrary likelihood p(y_n|x_n, F) = p(y_n|x_n, F_n), for all i
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

    def build_predict_batch(self,Xnew):
        mu, var = conditional_batch(Xnew, self.Z, self.kern, self.q_mu,
                                           q_sqrt=self.q_sqrt, whiten=self.whiten)
        return mu , var # (removed mean for now)


class SVGP_shared(SVGP):

    def build_predictor(self,Xnew, full_cov=False):
        fmean, fvar = self.build_predict_batch(Xnew)
        return tf.reduce_sum(fmean,1),tf.reduce_sum(fvar,[1,2])


class CSVGPs(object):
    """
    Coupled SVGPs
    """
    def __init__(self, X, Y, kerns,likelihood,Zs, n_param=0,
                 f_indices=None,mean_functions=None, whiten=True, W=None,
                 n_samp=50,sampling=False):
        '''
        - X is a data matrix, size N x D
        - Y is a data matrix, size N x R
        - kerns/mean_functions: list of C kernels/mean_functions
        - likelihood: Likelihood object
        - Zs: list of C matrices of pseudo inputs, size M[c] x D[c]
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
        self.f_indices = [[c] for c in range(self.C)] if f_indices is None else f_indices
        self.num_latent = Y.get_shape()[-1]
        self.num_data = Y.get_shape()[0]
        self.whiten=whiten
        self.n_samp =n_samp
        self.sampling = sampling
        self.W = W if W is not None else np.ones((self.C,1),dtype=np.float32)

        self.num_inducing = [z.shape[0] for z in Zs]

        self.Mtot = np.sum(self.num_inducing)

        with tf.variable_scope("inference") as scope:
            q_mu = np.random.randn(self.Mtot, self.num_latent)*std_qmu_init
            self.q_mu = tf.get_variable("q_mu", [self.Mtot, self.num_latent],
                                        initializer=tf.constant_initializer(q_mu))
            q_sqrt = np.array([np.eye(self.Mtot) for _ in range(self.num_latent)]).swapaxes(0, 2)
            self.q_sqrt = tf.get_variable("q_sqrt", [self.Mtot, self.Mtot, self.num_latent], \
                                          initializer=tf.constant_initializer(q_sqrt, dtype=float_type))

        self.Zs = []
        with tf.variable_scope("ind_points") as scope:
            for c,z_np in enumerate(Zs):
                with tf.variable_scope("ind_points%d" % c) as scope:
                    self.Zs.append(tf.Variable(z_np, tf.float32, name='Z'))

    def get_covariate(self,Xnew,c):
        return tf.transpose(tf.gather(tf.transpose(Xnew),self.f_indices[c]))

    def build_prior_KL(self):
        if self.whiten:
            KL = gauss_kl_white(self.q_mu, self.q_sqrt)
        else:
            diag = [self.kerns[d].K(self.Zs[d]) for d in range(self.C)]
            K = block_diagonal(diag)+ \
                    eye(self.Mtot) * jitter_level
            KL = gauss_kl(self.q_mu, self.q_sqrt, K)
        return KL

    def build_predict_joint(self,Xnew):
        """
        Batch Computing of q(f^n_1,...,f^n_C)
        - Xnew: N x D
        - Output: 
         - mean: N x C x R
         - var: N x C x C x R 
        """
        mu, var = conditional_stack(Xnew, self.Zs, self.kerns, self.q_mu,f_indices=self.f_indices,\
                                           q_sqrt=self.q_sqrt, whiten=self.whiten)
        return mu , var # (removed mean function for now)

    def build_predictor(self,Xnew):
        raise NotImplementedError

    def sample_joint(self,Xnew):
        mu, var = self.build_predict_joint(Xnew) # N * C * R, N * C * C * R
        L = tf.cholesky( tf.transpose(var,(3,0,1,2)) ) # R x N x C x C
        L_tiled = tf.tile( tf.expand_dims(L, 0), [self.n_samp, 1, 1, 1, 1])  # Nsamp x R x N x C x C
        shape = tf.shape(L_tiled)[:-1] # Nsamp x R x N x C
        x = tf.expand_dims(tf.random_normal(shape),-1) # Nsamp x R x N x C
        s = tf.transpose( tf.matmul(L_tiled, x), (0,2,3,1,4) ) # Nsamp x N x C x R x 1
        return tf.reshape(s,tf.shape(s)[:-1]) + tf.expand_dims(mu,0) # Nsamp x N x C x R


    def build_likelihood(self):
        # Get prior KL.
        KL = self.build_prior_KL()
        if self.sampling:
            samples_pred = self.sample_predictor(self.X)
            sto_exp = self.likelihood.logp(samples_pred, tf.expand_dims(self.Y, 0))
            var_exp = tf.reduce_mean(sto_exp,0)
        else:
            # Get conditionals
            fmean, fvar = self.build_predictor(self.X)
            # Get variational expectations.
            var_exp = self.likelihood.variational_expectations(fmean, fvar, self.Y)
        return tf.reduce_sum(var_exp)  - KL


class SVAGP(CSVGPs):


    def build_predictor(self,Xnew):
        fmean, fvar = self.build_predict_joint(Xnew)
        return tf.reduce_sum(fmean*self.W,1),tf.reduce_sum(fvar*tf.square(self.W),[1,2])

    def sample_predictor(self,Xnew):
        s = self.sample_joint(Xnew) # Nsamp x N x C x R
        return tf.reduce_sum(s*self.W,2) # Nsamp x N x R


class SVMGP(CSVGPs):

    def sample_predictor(self,Xnew):
        s = self.sample_joint(Xnew) # Nsamp x N x C x R
        return tf.reduce_prod(s,2) # Nsamp x N x R