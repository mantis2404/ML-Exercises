### DIFFERENCE B/W COST FUNCTION AND LOSS FUNCTION
- **loss function** is of single training example
- **cost function** is calculated for entire dataset

##### 1. MSE or L2 LOSS(`loss='mean_squared_error'`)

- best if outliers are not present and gradient descent is to be used
- activation function is output layer should be `linear`.

| ADVANTAGES                                   | DISADVANTAGES         |
| -------------------------------------------- | --------------------- |
| Interpretability                             | error unit is squared |
| differentiable, GRADIENT DESCENT can be used | sensitive to outliers |
| 1 local minima                               |                       |
##### 2. MAE Or L1 LOSS

- best to use if outliers are present

| ADVANTAGES         | DISADVANTAGES                                |
| ------------------ | -------------------------------------------- |
| Intuitive and easy | not differentiable, have to use subgradients |
| unit same as y     |                                              |
| robust to outliers |                                              |
##### 3. HUBER LOSS

- combines **MAE** and **MSE** , if data is outlier it behaves as **MAE** and if not then behaves as **MSE**.
> outlier detection identified through hyperparamter

- best in case of segregated data ( 25% outliers and 75% normal ), would find best weights and bias accordingly.

##### 4. BINARY CROSS ENTROPY or LOG LOSS

 - used in classification with two classes
 - activation function in output layer should be `sigmoid`.

| ADVANTAGES     | DISADVANTAGES         |
| -------------- | --------------------- |
| differentiable | not intuitive         |
|                | multiple local minima |
>[!NOTE]
>Despite BCE being a convex function in itself it is a non-convex function when combined with neural networks and hence has multiple minima

##### 5. CATEGORICAL CROSS ENTROPY

- used in multi-classification
- used in softmax regression
- activation function for output layer is `softmax`.
- One Hot Encoding is used on the target variable

##### 6. SPARSE CATEGORICAL CROSS ENTRPY

- used in multi-classification
- same loss function as of **categorical cross entropy**
- faster in case of many classes
- OHE not needed, label encoding preferred

>[!NOTE]
>- Loss functions themselves may be convex or non-convex but due to the non-linear activation functions used in neural networks the overall loss function for a neural network is non-convex
>- if the activation function is linear with convex loss functions (MSE,etc..) then the overall loss function for the neural network becomes convex shaped
