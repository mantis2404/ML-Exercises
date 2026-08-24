## PCA
- fast and linear
- can be used for image compression( 100x100 pixel converted to 50 components )
- preserves gobal structure
- good for preprocessing
- Unsupervised. It does not look at the target variable (labels) at all

#### The Mathematics   
###### Step 1: Mean Centering (Standardization)   
centre the data at the origin $(0,0)$. If the features are on vastly different scales (e.g., Age vs. Salary), you also divide by the standard deviation so no single feature dominates the algorithm.Math: $Z = \frac{x - \mu}{\sigma}$  
###### Step 2: Calculate the Covariance Matrix   
 For mean-centered data matrix $X$, the covariance matrix is $\Sigma = \frac{1}{n-1} X^T X$  
###### Step 3: Eigendecomposition    
You extract the Eigenvectors and Eigenvalues from the Covariance Matrix.Eigenvectors are the new directions (the axes). They are always orthogonal (perpendicular) to each other.Eigenvalues are scalars that tell you how much variance (information) exists along each Eigenvector. $\sum v = \lambda v$ (where $v$ is the eigenvector and $\lambda$ is the eigenvalue).   
###### Step 4: Sort and Select (Dimensionality Reduction)   
You rank the eigenvectors by their eigenvalues, from highest to lowest.Principal Component 1 (PC1) has the highest eigenvalue (most variance).Principal Component 2 (PC2) has the second highest, and so on.   
###### Step 5: Project the Data  
You multiply your original data by the selected eigenvectors to transform the data into the new, smaller dimensional space.

## t-SNE
- visualizing local clusters
- non-linear, captures complex relationships
- Great for EDA but not for ml input
- preserves local structure
- tune using `perplexity` but takes a lot of time

## UMAP
- useful for visualising patterns and use it for k-means
- reduces dimensions of embeddings
- works much better with `n_neighbours` parametr tuning
- preserves both local and global structure
- used for both model input and visualisation , best among the three

### WHAT IS LOCAL AND GLOBAL STRUCTURE

Consider a dataset of a shopping app with many features and each row represent a customer

##### LOCAL STRUCTURE
If two customers have **very similar behavior** , they should stay close to each other in reduced space too.

> "Rohit and Tanya both buy books and spend ₹500/month — they should appear next to each other even after reducing dimensions."

##### GLOBAL STRUCTURE
If one group of customers is **very different from another group**, their **clusters should stay far apart**.

> "Frequent buyers and one-time visitors should form separate distant groups — even in reduced space."

| Algorithm | What It Preserves   | What You'll See                                                                                      |
| --------- | ------------------- | ---------------------------------------------------------------------------------------------------- |
| **PCA**   | Global structure    | Big spenders and small spenders are far apart — but within those groups, similarities may get lost   |
| **t-SNE** | Local structure     | Similar customers (small groups) are tightly grouped — but overall group distances may be misleading |
| **UMAP**  | Local + some global | You see both tight groups and an approximate sense of how groups relate to each other                |
