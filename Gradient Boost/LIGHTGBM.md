### FEATURES
- Histogram based tree growth
- Native categorical feature support

> `LightGBM` does not support `object` and `string` type columns and hence if there is any it has to be converted to `category` type and then passed through hyperparameter and then the model consider it as categorical value.`CatBoost` accepts `string `and `object` type.
> 
> `CatBoost` better for categorical values


- GPU support
- no imputatuon needed
- Faster
- Deeper and Complex trees
- Requires quite of few Hyperparameters tuning

### USAGE
- Large dataset
- Mostly numeric features
- Want fast training with high accuracy

