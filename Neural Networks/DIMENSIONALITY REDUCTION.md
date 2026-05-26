<img width="650" height="281" alt="Pasted image 20250710052843" src="https://github.com/user-attachments/assets/6cae4eb7-2eff-4ed4-8355-5da14926bc36" />

As dimension increases the relative distance between points increases, but all points seem to be equally far away (same frequency for distance) and thus the concept of nearest and farther neighbors become blur, hence models like KNN and clustering perform poor on large dimensional data

<img width="226" height="223" alt="Pasted image 20250710055338" src="https://github.com/user-attachments/assets/f3de9b4c-3b8f-4f79-868a-19f494f981a4" />



| EUCLIDEAN                        | GEODESIC                                                   |
| -------------------------------- | ---------------------------------------------------------- |
| straight line between two points | shortest path along the surface/manifold between two poins |
| linear space                     | non-linear space/manifold                                  |
| PCA,K-menas                      | t-SNE,Isomap                                               |

#### t-SNE
- calculates pairwise euclidean distribution to estimate neighbors
- converts them to probability distribution (gaussian distribution), formally called conditional probability
- its like giving weights, closer points are given more importance

<img width="590" height="300" alt="Pasted image 20250710063736" src="https://github.com/user-attachments/assets/b02504d9-ec1c-4e61-933e-0da6fa84fadb" />


<img width="590" height="201" alt="Pasted image 20250710063936" src="https://github.com/user-attachments/assets/75bd7c08-5758-4af4-b398-42552e8e591f" />


in t-distribution the **extreme values** can be seen given higher value than normal distribution thus reducing the force between them(further context) hence avoiding the overcrowding problem (occurs due to less space in low dimension)

in **high dimensional space** t-sne uses **normal distribution** but in **low dimensional space** it uses **t-distribution** (has longer tails than normal dist ensuring that far points are not taken too close to the query point)

probability that point i "picks" point j as its neighbor based on a **Gaussian distribution centered at i**:
$$
p_{j|i} = \frac{\exp\left(-\frac{\|x_i - x_j\|^2}{2\sigma_i^2}\right)}{\sum_{k \ne i} \exp\left(-\frac{\|x_i - x_k\|^2}{2\sigma_i^2}\right)}
$$

- $\beta = \frac {1}{2\sigma^2}$ is more preferably used by default
- this formula treats as $x_i$ as center of Gaussian kernel and then calculates distances for how close $x_j$ is to $x_i$ 
- those distances are then converted to probabilities
- similarity computed wrt to point $x_i$
- it is asymmetric as variance is different for i and j 

$\sigma_i$ is variance for point i, tuned per point to match desired perplexity
> smaller variance means smaller spread of gaussian distribution meaning using smaller variance smaller values are assigned to points at the tails of the distribution, this is under the hood controlled by **perplexity**.

#### **WHY VARIANCE IS NOT FIXED?**
Due to **varying data densities**! in sparse regions **large $\sigma$** is required to reach far and find neighbors whereas in dense regions **small $\sigma$** works.

if equal $\sigma$ is provided then in **dense regions** too many points would be considered close and in **sparse regions** no points would be close enough

#### **PERPLEXITY**:
- controls how many nearby points influence each data point
- controls the variance for Gaussian distribution
- higher perplexity higher variance
- basically perplexity decides which point count as **neighbor**

$$
Perp (P_i) = 2^{H(P_i)}
$$
**SHANNON ENTROPY**
$$
H(P_i)=-\sum_j P_{j|i}log_2{ p_{j|i}}
$$
>- $P_i​$={$p_{j∣i}​$} is the conditional probability distribution over all other points $x_j$

shannon entropy increases as variance increases as more spreaded the distribution more uncertainty we have

So basically we are fixing perplexity (basically the number of neighbors each point should have) and through that for each point variance is adjusted such that the perplexity (neighbors are fulfilled)

#### **HOW DOES FIXING PERPLEXITY RESULTS IN DIFFERENT VARIANCE FOR DIFFERENT DATA POINTS**
 let say perplexity is fixed at **a**:
 $$2^{H(P_i)} = a$$
 so the aim is that for each $x_i$ we have to find $\sigma_i$ such that
$$H(P_i)=log_2 {a}$$

this is done through **BINARY SEARCH**:
- **Initialize search range** for $\sigma_i$ or $\beta_i$​:  
    e.g. $σ_{min}$=$1e^{-20}$, $σ_{max}$== $1e^5$
- **Repeat**:
    - Set $σ_i$=$\frac{σ_{min}+σ_{max}}{2}$ or $\beta_i$=$\frac{\beta_{min}+\beta_{max}}{2}$
    - Compute the probabilities $p_{j|i}$​ using current $\sigma_i$​
    - Compute entropy $H(P_i)$
    - If $H(P_i)>log⁡_2(Perp)$:
        → Entropy too high → probabilities too flat → **decrease** $\sigma_i$ or **increase** $\beta$
        → Set $\sigma_{\text{max}}$ = $\sigma_i$ or $\beta_{min}=\beta_i$
    - Else:  
        → Entropy too low → probabilities too sharp → **increase** $\sigma_i$ or **decrease** $\beta$
        → Set $\sigma_{\text{min}} = \sigma_i$ or $\beta_{max}=\beta_i$​
- **Stop when** $|H(P_i) - \log_2(\text{Perp})| < \epsilon$

#### **TRANSFORMING HIGH-DIMENSIONAL DATA TO LOW-DIMENSIONAL DATA**

**IN HIGH DIMENSION**

The conditional probability $p_{j|i}$ is asymmetric, meaning $p_{j|i} \ne p_{i|j}$, the give formula is not symmetric in i and j
> - $p_{j|i}$ is row wise normalized
> - $p_{ii}$ is by definition 0

$$p_{j|i} = \frac{\exp(-\beta_i \|x_i - x_j\|^2)}{\sum_{k \ne i} \exp(-\beta_i \|x_i - x_k\|^2)}$$to convert it to pairwise similarity
$$p_{ij} = \frac{p_{j|i} + p_{i|j}}{2n}$$
now $p_{ij} = p_{ji}$ and reflects the mutual similarity between $x_i$ and $x_j$, P={$p_{ij}$} represents pairwise similarity matrix in high-dimension
> - large $p_{ij}$: neighbors in high-D
> - small $p_{ij}$: far apart

**IN LOW DIMENSION**

low dimension embeddings are randomly initialized {$y_1,y_2....y_n$} and similarity between them using the **student t-distribution** is defined
$$q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k \ne l} (1 + \|y_k - y_l\|^2)^{-1}}$$
> - $q_{i|j}$ is globally normalized
> - $q_{ii}$ is by definition 0 as we are not looking for self similarity
- Q={$q_{ij}$} is pairwise similarity matrix in low dimension
- it is already symmetric as formula depends on distance (which is symmetric $\|y_i - y_j\|^2 = \|y_j - y_i\|^2$) rather than center point
- no need to find different variances using binary search
- Large $q_{ij}$: they’re currently neighbors
- Small $q_{ij}$​: they’re currently far apart

**OPTIMIZATION**

The low dimension embeddings are now optimized by minimizing the KL divergence (the loss function)
$$KL(P∥Q)=\sum_{i \ne j} p_{ij​}log(\frac{p_{ij}}{​q_{ij}}​​)​$$
$p_{ij}$ are fixed and $q_{ij}$ depends on $y_i$ and $y_j$ and are learnable

$$
\frac{\partial \text{KL}}{\partial y_i} = 4 \sum_{j \ne i} (p_{ij} - q_{ij}) \cdot \frac{y_i - y_j}{1 + \|y_i - y_j\|^2}
$$

- **If $p_{ij} \gg q_{ij}$** → model is underestimating the similarity → **pull** points closer
- **If $p_{ij} \ll q_{ij}$​** → model is overestimating → **push** them apart
- $(y_i−y_j)$: A vector (force vector) pointing **away** from $y_j$
    - Positive coefficient → **pull $y_i$​ toward $y_j$​**
    - Negative coefficient → **push $y_i$​ away from $y_j$​**
- $(1 + \|y_i - y_j\|^2)^{-1}$: This **scales** the force based on distance
	- Farther points → smaller force
	- Close points → stronger force
- acts like spring

**UPDATE WEIGHTS**

$$y_i^{t+1}=y_i^{t}−η⋅\frac{∂KL}{​∂y_i}​$$
**ADD MOMENTUM**
$$
\begin{aligned}
v_i^{(t+1)} &= \mu \cdot v_i^{(t)} - \eta \cdot \frac{\partial \text{KL}}{\partial y_i^{(t)}} \\\\
y_i^{(t+1)} &= y_i^{(t)} + v_i^{(t+1)}
\end{aligned}
$$
μ: momentum coefficient (0.5 early on, 0.8 later)

THIS IS PERFORMED FOR 1000 EPOCHS, WITH EARLY EXAGERATION FOR AROUND 250 EPOCHS

## PCA
 - fast and linear
- can be used for image compression( 100x100 pixel converted to 50 components )
- preserves gobal structure
- good for preprocessing

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

**T-SNE IS BEST FOR VISUALIZING CLUSTERS AND LOCAL STRUCTURE BUT IS NOT USED TO SEND DATA FOR MODEL TRAINING AS IT IS NON REPRODUCIBLE FOR NEW DATASET**

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

### HOW TO KNOW WHICH COLUMN IS MORE IMPORTANT

Find spread of the data over each column (variance). The column having the most spread (variance) is more important

We can visualize this by plotting the scatterplot between the two columns and then see the data spread

If two columns have similar spread (variance) then we use feature extraction(PCA and all) and then from those components we can decide which component to choose based on data spread on their axis

Can do with `VarianceThreshold`

But this is not always true and may fail sometimes like **when the scales are not same e.g. Income vs Age**

> Should once check with corelation with the target variable and according it should be removed

### WHY VARIANCE?

Why not use **Mean Absolute Distance** (absolute distance b/w mean and a point)?

It is because it is absolute distance means it uses modulus and hence when optimisation is required it will not work as **modulus is not differentiable** , hence variance is preferred.

variance ∝ spread

Moreover if we consider two points and project them on a axis (1D), the axis that have more spread will fairly tell the difference between the points whereas the axis with less variance(spread) reduces the distance b/w the two points and hence leading to wrong results

### RFE (REDUCED FEATURE EXTRACTION)
Models assigns weights to each feature and RFE then selects the features with lowest importance are eliminated and do this recursively until the desired number of features (`n_features_to_select`) remains.

`RFECV` does this based on cross-validation scores.
> RFECV = RFE + Cross-Validation


