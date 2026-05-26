- Used with large complicated data sets, reduce training time 
- Just like Gradient boost , supports custom function hence can be used for different problems
- Supports many languages

### FEATURES

- Parallel processing

>`n_jobs=-1` allows parallel processing

- Optimized DS
- Out of Core computing

 > `tree_method='hist'` (e.g. loads 10gb dataset into 8gb ram through chunks)
 
- GPU support

> `tree_method='gpu_hist'` 

- Regularised Learning Objectives
- Can handle missing values automatically 
- Flexible to Tree pruning very much
- Affected by imbalance datset

> All boosting algorithms are affected by it

### USAGE
- Imbalance datsets can be dealt with hyperparameter tuning
- Reduced overfiting
- Easy acces to SHAP values / feature selection

Can set the column values to 0 and then check accuracy to see if that column is important or not
put null values to -1 or some other value not in dataset.
for hyperparamater tuning use other frameworks rather than gridsearchcv and randomsearchcv (AutoML)
should get around 0.1
momentum feature / ewma feature / lgbl /LSTM /ADIMA /Onsomble
Elastic net + lasso +ridge for housing , onsoumbling also ,SMOGN