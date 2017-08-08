# Copyright 2016 Valentine Svensson, James Hensman, alexggmatthews
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ------------------------------------------
# Modification notice:
# This file was modified by Vincent ADAM
# ------------------------------------------


import tensorflow as tf
from settings import  jitter_level, float_type
from functions import eye, block_diagonal


def conditional(Xnew, X, kern, f, full_cov=False, q_sqrt=None, whiten=False):
    """
    Given F, representing the GP at the points X, produce the mean and
    (co-)variance of the GP at the points Xnew.

    Additionally, there my be Gaussian uncertainty about F as represented by
    q_sqrt. In this case `f` represents the mean of the distribution and
    q_sqrt the square-root of the covariance.

    Additionally, the GP may have been centered (whitened) so that
        p(v) = N( 0, I)
        f = L v
    thus
        p(f) = N(0, LL^T) = N(0, K).
    In this case 'f' represents the values taken by v.

    The method can either return the diagonals of the covariance matrix for
    each output of the full covariance matrix (full_cov).

    We assume K independent GPs, represented by the columns of f (and the
    last dimension of q_sqrt).

     - Xnew is a data matrix, size N x D
     - X are data points, size M x D
     - kern is a GPflow kernel
     - f is a data matrix, M x R, representing the function values at X, for R functions.
     - q_sqrt (optional) is a matrix of standard-deviations or Cholesky
       matrices, size M x R or M x M x R
     - whiten (optional) is a boolean: whether to whiten the representation
       as described above.

    These functions are now considered deprecated, subsumed into this one:
        gp_predict
        gaussian_gp_predict
        gp_predict_whitened
        gaussian_gp_predict_whitened

    """

    # compute kernel stuff
    num_data = tf.shape(X)[0]
    Kmn = kern.K(X, Xnew)
    Kmm = kern.K(X) + eye(num_data) * jitter_level
    Lm = tf.cholesky(Kmm)

    # Compute the projection matrix A
    A = tf.matrix_triangular_solve(Lm, Kmn, lower=True)

    # compute the covariance due to the conditioning
    if full_cov:
        fvar = kern.K(Xnew) - tf.matmul(A, A, transpose_a=True)
        shape = tf.stack([tf.shape(f)[1], 1, 1])
    else:
        fvar = kern.Kdiag(Xnew) - tf.reduce_sum(tf.square(A), 0)
        shape = tf.stack([tf.shape(f)[1], 1])
    fvar = tf.tile(tf.expand_dims(fvar, 0), shape)  # R x N x N or R x N

    # another backsubstitution in the unwhitened case
    if not whiten:
        A = tf.matrix_triangular_solve(tf.transpose(Lm), A, lower=False)

    # construct the conditional mean
    fmean = tf.matmul(tf.transpose(A), f)

    if q_sqrt is not None:
        if q_sqrt.get_shape().ndims == 2:
            LTA = A * tf.expand_dims(tf.transpose(q_sqrt), 2)  # R x M x N
        elif q_sqrt.get_shape().ndims == 3:
            L = tf.matrix_band_part(tf.transpose(q_sqrt, (2, 0, 1)), -1, 0)  # R x M x M
            A_tiled = tf.tile(tf.expand_dims(A, 0), tf.stack([tf.shape(f)[1], 1, 1]))
            LTA = tf.matmul(L, A_tiled, transpose_a=True)  # R x M x N
        else:  # pragma: no cover
            raise ValueError("Bad dimension for q_sqrt: %s" %
                             str(q_sqrt.get_shape().ndims))
        if full_cov:
            fvar = fvar + tf.matmul(LTA, LTA, transpose_a=True)  # R x N x N
        else:
            fvar = fvar + tf.reduce_sum(tf.square(LTA), 1)  # R x N
    fvar = tf.transpose(fvar)  # N x R or N x N x R

    return fmean, fvar



def conditional_batch(Xnew, X, kern, f, q_sqrt=None, whiten=False):
    """
    Batch version of conditional
    Somewhere between full_cov=True and full_cov=False
    Computes full covariance over specified blocks
     - Xnew is a data matrix, size N x D x B
     - X are data points, size M x D
    """

    # kernel matrix for conditionning inputs
    num_data = tf.shape(X)[0]
    Kmm = kern.K(X) + eye(num_data) * jitter_level
    Lm = tf.cholesky(Kmm)

    # kernel matrices for all batch to predict on
    Knn = kern.K(Xnew) # N x N x B
    Knm = kern.K(Xnew,X) # N x M x B

    # Compute the projection matrix A
    Lm = tf.tile(tf.expand_dims(Lm,0),tf.stack([tf.shape(Xnew)[-1],1,1])) # B x M x M
    A = tf.matrix_triangular_solve(Lm, tf.transpose(Knm,(2,1,0)), lower=True) # B x M x N

    fvar = tf.transpose(Knn,(2,0,1))- tf.matmul(tf.transpose(A,(0,2,1)),A) # B x N x N
    shape = tf.stack([1, tf.shape(f)[1], 1, 1])
    fvar = tf.tile(tf.expand_dims(fvar, 1), shape)  # B x R x N x N

    # another backsubstitution in the unwhitened case
    if not whiten:
        A = tf.matrix_triangular_solve(Lm, A, lower=False) # B x M x N
    # construct the conditional mean
    fmean = tf.einsum('bmn,mr->bnr',A, f) #  B x N x R

    if q_sqrt is not None:
        if q_sqrt.get_shape().ndims == 2: # M x R
            LTA = tf.einsum('mr,bmn->brmn',q_sqrt,A) # B x R x M x N
        elif q_sqrt.get_shape().ndims == 3: # M x M x R
            L = tf.matrix_band_part(tf.transpose(q_sqrt, (2, 0, 1)), -1, 0)  # R x M x M  (lower triangular)
            LTA = tf.einsum('rmw,bwn->brmn',L,A) # B x R x M x N
        else:  # pragma: no cover
            raise ValueError("Bad dimension for q_sqrt: %s" % str(q_sqrt.get_shape().ndims))
        fvar = fvar + tf.matmul(tf.transpose(LTA,(0,1,3,2)), LTA)   # B x R x N x N

    fvar = tf.transpose(fvar,(0,2,3,1))  # B x N x N x R

    return fmean, fvar


def conditional_stack(Xnew, Xs, kerns, f, f_indices=None, q_sqrt=None, whiten=False):

    """
    Xnew : N x D
    Xs  : list Mi x 1
    kerns : list of kernels
    f : sum_i Mi x 1
    """

    # prior covariance on Xs
    C = len(kerns)
    f_indices = [[c] for c in range(C)] if f_indices is None else f_indices
    N = Xnew.get_shape()[0].value
    Ms = tf.stack([x.get_shape()[0] for x in Xs])
    Mtot = tf.reduce_sum(Ms)
    Ks = [kerns[c].K(Xs[c]) for c in range(C)]
    Kmm = block_diagonal(Ks)
    Kmm += eye(Mtot) * jitter_level
    Lm = tf.cholesky(Kmm)

    # Prior covariance across dimension for each data point (diag)
    Knn = []
    for c in range(C):
        xc = tf.transpose(tf.gather(tf.transpose(Xnew),f_indices[c]))
        Knn.append(kerns[c].Kdiag(xc) )
    Knn = tf.transpose(tf.matrix_diag((tf.stack(Knn,-1))),(1,2,0)) # C x C x N
    # Prior conditional
    uu = []
    for c in range(C):
        u = []
        for c_ in range(C):
            if c==c_:
                xc = tf.transpose(tf.gather(tf.transpose(Xnew), f_indices[c]))
                u.append(kerns[c].K( Xs[c],xc))  #  M x N
            else:
                u.append(tf.zeros([Ms[c_],N]))
        uu.append(tf.concat(u,0))
    Knm = tf.stack(uu,0) # C x M x N

    # Compute the projection matrix A
    Lm = tf.tile(tf.expand_dims(Lm,0),tf.stack([N,1,1])) # B x M x M
    A = tf.matrix_triangular_solve(Lm, tf.transpose(Knm,(2,1,0)), lower=True) # B x M x N
    fvar = tf.transpose(Knn,(2,0,1))- tf.matmul(tf.transpose(A,(0,2,1)),A) # B x N x N
    shape = tf.stack([1, tf.shape(f)[1], 1, 1])
    fvar = tf.tile(tf.expand_dims(fvar, 1), shape)  # B x R x N x N

    # another backsubstitution in the unwhitened case
    if not whiten:
        A = tf.matrix_triangular_solve(Lm, A, lower=False) # B x M x N
    # construct the conditional mean
    fmean = tf.einsum('bmn,mr->bnr',A, f) #  B x N x R

    if q_sqrt is not None:
        if q_sqrt.get_shape().ndims == 2: # M x R
            LTA = tf.einsum('mr,bmn->brmn',q_sqrt,A) # B x R x M x N
        elif q_sqrt.get_shape().ndims == 3: # M x M x R
            L = tf.matrix_band_part(tf.transpose(q_sqrt, (2, 0, 1)), -1, 0)  # R x M x M  (lower triangular)
            LTA = tf.einsum('rmw,bwn->brmn',L,A) # B x R x M x N
        else:  # pragma: no cover
            raise ValueError("Bad dimension for q_sqrt: %s" % str(q_sqrt.get_shape().ndims))
        fvar = fvar + tf.matmul(tf.transpose(LTA,(0,1,3,2)), LTA)   # B x R x N x N

    fvar = tf.transpose(fvar,(0,2,3,1))  # B x N x N x R

    return fmean, fvar
