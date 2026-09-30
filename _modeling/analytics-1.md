---
title: "Sparse KPCA Demo"
excerpt: "This project is a brief exploration of Michael Tipping's proposed method for sparse kernel principal component analysis for Microsoft Research in 2000."
excerpt_image: /images/kpca_cover.png
collection: modeling
date: 2026-09-08
---
# Introduction
This project is a brief exploration of Michael Tipping's probabilistic method for [Sparse Kernel Principal Component Analysis](https://proceedings.neurips.cc/paper_files/paper/2000/file/bf201d5407a6509fa536afc4b380577e-Paper.pdf), which is based on a sparse approximation of the covariance matrix in a high-dimensional feature space. I came across Tipping's paper when first learning about kernel PCA during my master's program, and was intrigued by the simplicity of Tipping's method as a learning data scientist. All that's required is some background in linear algebra and an understanding of partial derivatives. Some understanding of [expectation maximization through factor analysis](https://pages.hmc.edu/ruye/MachineLearning/lectures/ch8/node12.html) is helpful to understand Tipping's brief reference to the method, but isn't entirely necessary. 

Unlike standard PCA, kernel PCA is a dimensionality reduction technique that preserves non-linear patterns in data with the right kernel function. A _kernel function_ provides a pairwise dissimilarity measure for all feature vectors once projected into a higher-dimensional space. This function is selected with two things in mind. First, we want to select an appropriate (but not needlessly complex) high-dimensional space in which the tranformed data _becomes_ linearly separable, allowing us to perform standard PCA in said space. Second, each kernel function is chosen so we do not need to directly compute the transformation of each original vector.

More specifically, a kernel function is always computed via the inner product of two _original_ data points, avoiding direct computation of their feature vectors. (I'll use the term "feature vector" to refer to the high-dimensional mapping of an original data point from here on out.) The kernel function defines an \\(n \times n\\) kernel matrix (where \\(n\\) is the number of data points), which can be operated on to compute the covariance matrix in said high-dimensional space. This way, we're able to obtain the principal components of even an infinite dimensional data matrix (in theory) through a finite kernel matrix. Importantly, we avoid the impossible task of performing computation on feature vectors that have infinite dimensions, such as those obtained using the common Gaussian (or RBF) kernel function.

Plenty has been written on how to select the optimal kernel function and its parameters depending on your dataset's characteristics. For the purposes of exploring Tipping's method, I'll skip that discussion and point to David Duvenaud's [Kernel Cookbook](https://www.cs.toronto.edu/~duvenaud/cookbook/) and [this article](https://journals.sagepub.com/doi/pdf/10.1260/1748-3018.8.2.163), which describes a kernel parameter selection algorithm to minimize cosine similarity between clusters in the feature space and maximize similiarity within them. This method is a more efficient alternative to grid search. For this exploration, though, the toy dataset I used was simple enough to manually select a decent parameter. 

## Kernel Matrix Sparsification

While KPCA is efficient in its avoidance of high-dimensional transformations, the eigendecomposition of the full \\(n \times n\\) kernel matrix — a necessary step in solving for the the feature space principal components — can still be computationally expensive. Tipping's method uses a sparse representation of the covariance matrix, making its eigendecomposition a much lighter task. The covariance matrix \\(\mathbb{S_{\Phi}}\\) is approximated by the covariance model \\(\mathbb{C}\\), which is computed via the weighted sum of the outer products of each feature vector. Under the assumption \\(\Phi_i \sim \cal{N}(0, \mathbb{C})\\), an expectation maximization update can be used to find the optimal set of weight parameters for computing \\(\mathbb{C}\\). 

Tipping was first to publicly recognize that, if we assume noise is fixed and contributes equally to the observed variance across all dimensions of the feature space, the likelihood with respect to the weight parameters is maximized when many weights are equal or very close to zero. Then, the principal components of the weighted kernel matrix (weighted according to the found parameters) can be used to compute the projection of each feature vector onto the principal components of the feature space. Again, Tipping uses some clever operations to do this without ever computing the feature vectors themselves, relying only on the weighted kernel matrix. If the feature space is chosen wisely, these projections should capture the non-linear structures and separations found in the original dataset pretty well. 

## Mathematical Background
We begin with some original dataset \\(\mathbb{X}\\), where each row vector \\(x_i\\) represents a single datapoint. In standard KPCA (and other kernel function applications like support vector machines), the kernel matrix \\(\mathbb{K}\\) is defined as the inner product matrix \\(\mathbb{\Phi} \mathbb{\Phi}^T\\), where each \\(\Phi_i\\) in \\(\mathbb{\Phi}\\) is a high-dimensional mapping of the corresponding \\(x_i\\) in \\(\mathbb{X}\\). Each kernel function \\(k\\) is cleverly designed so that \\(k(\Phi_i, \Phi_j) = \Phi_i^T \Phi_j\\) can be directly computed from the inner product of original datapoints \\(x_i\\) and \\(x_j\\).

The central assumption of Tipping's method is that all feature vectors \\(\Phi_i\\) follow the Gaussian distribution \\(\cal{N}(\mathbb{0}, \mathbb{S_\Phi})\\). The eigendecomposition of the covariance matrix \\(\mathbb{S_\Phi}\\) is not easy to compute for large datasets. Instead of using \\(\mathbb{S_\Phi}\\) directly, Tipping adopts a factor analytic approach to approximate \\(\mathbb{S_\Phi}\\)  with the covariance model \\(\mathbb{C}\\), where \\(\mathbb{C} = \sigma^2 \mathbb{I} + \sum_{i=1}^N w_i \Phi_i \Phi_i^T = \sigma^2 \mathbb{I} + \mathbb{\Phi}^T \mathbb{W} \mathbb{\Phi}\\). 

In this model, \\(\sigma^2\\) refers to an isotropic noise component, or the proportion of variability we are willing to attribute to randomness or error across all dimensions. (This is another assumption, which could be handled instead by implementing variance-based signal to noise ratios or another noise estimation technique.) 

\\(\mathbb{W}\\) is a diagonal loading matrix with weight \\(w_i\\) at each position, telling us the contribution of each \\(\Phi_i \Phi_i^T\\) to the weighted sum in \\(\mathbb{C}\\). Ignoring weight-independent terms, the log likelihood function for this Gaussian model is given by 

$$\cal{L} = -\frac{1}{2}(N \log \|\mathbb{C}\| + \text{tr}(\mathbb{C}^{-1} \mathbb{\Phi}^T \mathbb{\Phi}))$$

Using Tipping's notation, differentiating with respect to \\(w_i\\) gives 

$$\frac{\partial \cal{L}}{\partial w_i} = \frac{1}{2}(\Phi_i^T \mathbb{C}^{-1} \mathbb{\Phi^T \Phi}\mathbb{C}^{-1} \Phi_i - N \Phi_i^T \mathbb{C}^{-1} \Phi_i)$$ 

$$ \frac{\partial \cal{L}}{\partial w_i} = \frac{1}{2w_i^2}(\sum_{i=1}^N \Pi_{ni}^2 + N \mathbb{\Sigma}_{ii} - Nw_i)$$

where \\(\mathbb{\Sigma} = (\mathbb{W}^{-1} + \sigma^{-2}\mathbb{K}^{-1})\\) and \\(\mathbb{\Pi}_n = \sigma^{-2} \mathbb{\Sigma} k_n\\) (with \\(k_n\\) denoting a column vector with the kernel function of a target vector in the feature space and every training feature vector).

Setting \\(\frac{\partial \cal{L}}{\partial w_i}\\) to 0 in order to locate the function's maximum, the update equation for each \\(w_i\\) under Tipping's expectation maximization model is given by 

$$w_i^\textrm{new} = \frac{\sum_{n=1}^N \Pi_{ni}^2}{N(1- \mathbb{\Sigma}_{ii}/w_i)}$$

Tipping notes these re-estimates are equivalent to expectation-maximization updates obtained by factor analysis. In particular, \\(\Pi\\) and \\(\Sigma\\) represent the conditional mean and common covariance of Gaussian latent variables, or factors, given the feature vectors and current values of all \\(w_i\\). For simplicity, the update equation can also be thought of as the algebraic consequence of the equation \\( 0 = \frac{1}{2w_i^2}(\sum_{i=1}^N \Pi_{ni}^2 + N \mathbb{\Sigma}_{ii} - Nw_i)\\).

Since \\(\mathbb{\Sigma}\\) and \\(\mathbb{\Pi}_n\\) are in terms of the computable terms \\(\mathbb{W}\\) (the weight matrix found by the \\((i-1)\textrm{th}\\) iteration) and \\(\mathbb{K}\\), the optimal set of weights is computed with no need for high-dimensional transformations. 

Once the algorithm converges and an optimal set of weights is obtained, those \\(w_i\\) optimized at 0 (or almost 0) allow us to compute our covariance model using a fraction of our feature vectors \\(\Phi_i\\). (Recall the covariance model \\(\mathbb{C} = \sigma^2 \mathbb{I} + \sum_{i=1}^N w_i \Phi_i \Phi_i^T = \sigma^2 \mathbb{I} + \mathbb{\Phi}^T \mathbb{W} \mathbb{\Phi}\\).) 

Next, we perform eigendecomposition on our sparsified covariance model using the same method as in standard kernel PCA, plus a few added tricks. If we define \\(\mathbb{\tilde{\Phi}} = \mathbb{W^{\frac{1}{2}}} \mathbb{\Phi}\\), then we can rewrite \\(\mathbb{C} = \sigma^2 \mathbb{I} + \mathbb{\tilde{\Phi}}^T \mathbb{\tilde{\Phi}}\\). Since \\(\sigma^2\\) is an isotropic constant, the principal axes of \\(\mathbb{C}\\) are exactly those of \\(\mathbb{\tilde{\Phi}}^T \mathbb{\tilde{\Phi}}\\). We can use the well-known method of computing the eigenvectors of \\(\mathbb{\tilde{\Phi}}^T \mathbb{\tilde{\Phi}}\\) from those of \\(\mathbb{\tilde{\Phi}} \mathbb{\tilde{\Phi}}^T\\) by left-multiplying an additional \\(\mathbb{\Phi}^T\\):

$$\mathbb{\tilde{\Phi}} \mathbb{\tilde{\Phi}}^T \tilde{U} = \tilde{\lambda} \tilde{U}$$

$$(\mathbb{\tilde{\Phi}}^T \mathbb{\tilde{\Phi}}) (\mathbb{\tilde{\Phi}}^T \tilde{U}) = \tilde{\lambda} (\mathbb{\tilde{\Phi}}^T \tilde{U})$$

Thus, the eigenvectors of \\(\mathbb{\tilde{\Phi}}^T \mathbb{\tilde{\Phi}}\\) are simply given by \\(\mathbb{\tilde{\Phi}}^T \tilde{U} \mathbb{\Lambda}^{\frac{-1}{2}})\\) for each eigenvector \\(\tilde{U})\\) of \\(\mathbb{\tilde{\Phi}} \mathbb{\tilde{\Phi}}^T\\). (Note that we include scaling matrix \\(\mathbb{\Lambda}^{\frac{-1}{2}}\\) to normalize our eigenvector set.) 

Since the kernel matrix is defined by \\(\mathbb{K} = \mathbb{\Phi}\mathbb{\Phi}^T \\) and \\(\mathbb{\tilde{\Phi}} \mathbb{\tilde{\Phi}}^T= \mathbb{W}^{\frac{1}{2}} \mathbb{\Phi} \mathbb{\Phi}^T \mathbb{W}^{\frac{1}{2}}\\), we are seeking the eigenvectors of \\(\mathbb{W}^{\frac{1}{2}} \mathbb{K} \mathbb{W}^{\frac{1}{2}}\\). Since many \\(w_i\\) are optimized at \\(0\\), \\(\mathbb{W}^{\frac{1}{2}} \mathbb{K} \mathbb{W}^{\frac{1}{2}}\\) gives a sparse representation of \\(\mathbb{K}\\). 

The corresponding eigenvalues, which we don't explicitly need for the purposes of projection but can be useful for estimating PVE (proportion of variance explained by each principal component), are simply the eigenvalues of \\(\mathbb{W}^{\frac{1}{2}} \mathbb{K} \mathbb{W}^{\frac{1}{2}}\\) plus noise constant \\(\sigma^2\\).

Since each eigenvector \\(U\\) of \\(\mathbb{C}\\) is given by applying \\(\mathbb{\tilde{\Phi}}^T \tilde{U} \mathbb{\Lambda}^{\frac{-1}{2}} = \mathbb{\Phi}^T \mathbb{W}^{\frac{1}{2}} \tilde{U} \mathbb{\Lambda}^{\frac{-1}{2}}  \\) to each \\(\tilde{U}\\), they can't be directly computed directly without venturing into the feature space — something Tipping's method successfully avoids up to this point. Luckily, the projection of a feature vector onto the principal component vectors in \\(\mathbb{U}\\) can be obtained without computing in our high-dimensional space. In particular, our original feature vectors in the matrix \\(\mathbb{\Phi}\\) can be projected onto the kernel space principal components by simply computing \\(\mathbb{\Phi} \mathbb{\Phi}^T \mathbb{W}^{\frac{1}{2}} \tilde{U} \mathbb{\Lambda}^{\frac{-1}{2}}\\). Conveniently, this can be rewritten as \\(\mathbb{K} \mathbb{W}^{\frac{1}{2}} \tilde{U} \mathbb{\Lambda}^{\frac{-1}{2}}\\). This way, we are able to compute the projections directly from the non-weighted kernel matrix.

In Python, Tipping's algorithm is described by the following two functions:

```python 
def _fit_weights(K, noise, weights_init = None, tol=1e-8):
    ''' 
    Function to run Tipping's re-estimation for optimal weights on a given kernel matrix K.
    
    Args:
        K (nd-array) - m x m kernel matrix 
        noise (float) - isotropic noise component across all m dimensions
        weights_init (1d-array) - starting weights for expectation maximization
        tol (float) - convergence threshold

    Returns:
        weights_1 (1d-array) - vector with re-estimated weights
        it (int) - number of iterations until convergence 
    '''
    n = K.shape[0]

    # Initializing weights 
    weights_1 = weights_init.copy() if weights_init is not None else np.ones(n) / n

    W_1 = np.diag(weights_1)
    
    weights_0 = np.zeros(n)
    W_0 = np.diag(weights_0)
    
    it = 0
        
    # Updating weights based on expectation maximization wrt W
    while np.trace(np.abs(W_1 - W_0)) > tol:
        it += 1
        W_0, weights_0 = W_1, weights_1

        Sigma = np.linalg.inv(np.linalg.inv(W_1 + np.eye(n)*1e-8) + (1/noise) * K)

        M = (1/noise) * Sigma @ K

        for i in range(n):
            w_i = weights_0[i]
            M_i = M[i, :]
            Sigma_ii = Sigma[i, i]

            # Updating i'th weight 
            weights_1[i] = (1/n) * (np.sum(M_i)**2) + Sigma_ii

    return weights_1, it

def sparseKPCA(X, X_test = None, eps = 1e-8, poly = False, rbf = False, gamma_rbf = None, gamma_poly= None,  degree = 2, noise = 0.05):
    ''' 
    Function to perform sparse KPCA algorithm from "Sparse Kernel Principal Component Analysis" (Tipping 2000) on non-linearly separable data

    Args:
        X (nd-array) - Training data set
        X_test (nd-array) - Test data set 
        eps (float) - Small constant added for numerical stability 
        poly (Boolean) - True if using polynomial kernel
        rbf (Boolean) - True if using RBF/Gaussian kernel
        gamma_rbf (float) - Gamma value for RBF transformation k(v1,v2) = exp(-(gamma_rbf) * ||v1-v2||^2)
        gamma_poly (float) - Gamma value for polynomial transformation k(v1,v2) = (x.Ty + c)^(gamma_poly)
        degree (int) - degree for polynomial kernel

    Returns:
        proj_x (nd-array) - Feature vector projections onto kernel space principal components
    '''   
    # Checking for well-formed kernel parameters 
    if poly == rbf:
        print("Kernel parameter error. Please ensure you have chosen EITHER poly = True OR rbf = True.")
        return
    
    (n, d) = X.shape

    print('Computing Kernel matrix...')
    if rbf:
        if gamma_rbf == None:
            # Computing gamma value for RBF transformation via variance of pairwise Euclidian distances of observations in standardized data matrix 
            gamma_rbf = round(1/(2*np.median(euclidean_distances(X)**2)),4)
            print(f'RBF kernel gamma-value: {gamma_rbf}')
        K = rbf_kernel(X, gamma = gamma_rbf)

    if poly:
        K = polynomial_kernel(X, degree = degree, gamma = gamma_poly)   
        print(f'Polynomial kernel params: \n Gamma = {1/d if gamma_poly == None else gamma_poly} \n degree = {degree} \n c = 1')
        
    weights_1, n_iter = _fit_weights(K, noise)
    print(f'Expectation maximization update converged in {n_iter} iterations.')

    # Getting reduced kernel matrix based on non-zero weights
    support = np.where(weights_1 > 1e-6)[0]
    print(f'Non-zero weights retained: {len(support)} / {n}')
    K_reduced = K[np.ix_(support, support)]

    # Recomputing weights on reduced kernel matrix
    print('Re-optimizing weights for reduced K...')
    weights_reduced, n_iter_reduced = _fit_weights(K_reduced, noise, weights_init= weights_1[support])
    print(f'Re-fit converged in {n_iter_reduced} iterations.')

    # Placing reduced weights back into n-vector for operation on K
    weights_final_full = np.zeros(n)
    weights_final_full[support] = weights_reduced
    W = np.diag(weights_final_full)

    # Computing principal axes of reduced covariance model (eigh to ensure real eigenvalues)
    W_reduced = np.diag(weights_reduced)                   
    eigenvals_k, eigenvecs_k = np.linalg.eigh(W_reduced**0.5 @ K_reduced @ W_reduced**0.5)

    # Sorting eigh eigenvals in descending order (Otherwise, eigh auto-returns ascending order)
    idx = np.argsort(eigenvals_k)[::-1]
    eigenvals_k = eigenvals_k[idx]
    eigenvecs_k = eigenvecs_k[:, idx]

    # Dropping eigenvalues less than or close to 0
    keep = eigenvals_k > eps
    eigenvals_k = eigenvals_k[keep]
    eigenvecs_k = eigenvecs_k[:, keep]
    inv_sqrt_eigenvals = np.diag(1.0 / np.sqrt(eigenvals_k)) 

    # Projecting RBF feature vectors onto principal axes of covariance model 
    # If projecting test set
    if X_test is not None:
        K_star = rbf_kernel(X_test, X, gamma=gamma_rbf)
        proj_x = K_star[:, support] @ (W_reduced **0.5) @ eigenvecs_k @ inv_sqrt_eigenvals 
    
    # If evaluating training set only
    else:
        proj_x = K[:, support] @ (W_reduced **0.5) @ eigenvecs_k @ inv_sqrt_eigenvals

    # Returning scaled projections of kernel transformed clusters
    return StandardScaler().fit_transform(proj_x)
```

## Toy Dataset Demonstration

To get a sense of how this algorithm compares to non-sparse KPCA, I tested it out on a toy dataset with moon-shaped clusters. Generating two 2-dimensional clusters with sci-kitlearn's make_moons() function and running Tipping's algorithm allows us to visualize the original clusters alongside the 2D projections obtained via sparse KPCA. I also ran sci-kitlearn's linear PCA() and KernelPCA() functions to compare their output to the sparse KPCA implementation. 

Again, we are looking to see how well the two classes are separated by each method. In this case, an Gaussian (or RBF) kernel function is the best choice to capture the smooth decision boundary between the two classes. After a bit of experimentation, a gamma parameter of 15 seemed to do the best job at separating the two clusters.

```python 
# Toy example - KPCA on moon-shaped clusters
X, y = make_moons(n_samples=500, noise=0.1, random_state=2)

# Applying Sci-kit Linear PCA
X_pca = PCA().fit_transform(X)

# Applying Sci-kit KPCA 
kpca = KernelPCA(kernel='rbf', gamma = 15, eigen_solver='dense')
X_kpca = kpca.fit_transform(X)

# Applying sparseKPCA function to moons
X_skpca = sparseKPCA(X, rbf = True, gamma_rbf = 15) 
```

The 2-dimensional results for each method are shown in the plot below. Sci-kitlearn's KPCA and Tipping's sparse KPCA are clearly more successful than linear PCA in separating the two clusters!

![2-d scatterplots comparing original clusters with PCA, KPCA, and sparse KPCA principal components](/images/kpca_moons2d.png)

To get a sense of the advantage of a non-linear method in this case (and the accuracy trade-off between a sparse and dense KPCA implementation), I performed classification via logistic regression on the principal components from each method. In terms of accuracy, the sparse implementation was just as successful as KPCA performed on the a dense kernel matrix. Both non-linear methods outperformed linear PCA in terms of separating the two classes. Logistic regression achieved 86% accuracy when performed on test data's projections onto the 2 linear principal axes, compared to 100% and 96% accuracy on the same test set's classical KPCA and sparse KPCA projections (respectively) onto the principal axes in the feature space. 

It's important to note that, since the data is being projected into a high-dimensional space, the number of feature space principal components is dictated by the number of unique, nontrivial eigenvalues of the kernel matrix. For a 2-dimensional dataset such as the toy clusters generated for this implementation, linear PCA only projects the data onto 2 principal dimensions, while both non-linear methods can return as many as \\(n\\) dimensions for (\\n\\) points (or an (\\n \times n\\) kernel matrix). Since the goal of KPCA is dimensionality reduction, it isn't ideal to perform classification on such a larger set of features. Therefore, it's necessary to decide how many principal axes of the feature space we are willing to keep. 

Since there are only 2 dimensions to our generated clusters and linear PCA projections, a natural choice is to consider only the first two principal components of the KPCA and sparse KPCA projections. Performing logistic regression on only the first 2 KPCA and sparse KPCA components still shows both non-linear methods outdo linear PCA, achieving 95% accurate classifications in both cases. Still an impressive difference!