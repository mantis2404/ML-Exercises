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
