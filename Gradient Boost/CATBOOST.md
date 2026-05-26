### FEATURES
- Handles categorical features values natively ( encoding required for categorical target variable )
- Uses ordered Target Encoding 

> Prevents Data Leakage ( Manual in XGBoost )
> In XGBoost doing target encoding may leak the data

- Supports both classification and regression just like XGBoost
- GPU Supported
- No scaling,imputation,encoding required
- Handles missing values automatically
- Need to specify the categorical values as hyperparamter else all values consider numeric

### USAGE
- Dataset has many categorical columns
- Do not want much pre-processing
- Supports pipeline
- Do not want overfitting


#### WHAT IS DATA LEAKAGE

- Using the data to train the model which was supposed to be predicted by it.
- Like using the target value of current row/next rows to encode values or to train the model.
- Using the data of the previous rows does not account for data leakage as this depicts the real life scenarios.
- This is what Ordered Encoding does (CatBoost)