- Can be used for both Regression and Classification
- Makes single leaf in the starting (mean of the data)
- Then makes a tree( can be larger than a stump)
- Tree generation is quite similar to Ada Boost
- In this a new column called pseudo `residual`( difference between actual and predicted value ) is created on basis on which the new tree is created.
- The overall sum for output of different trees is the final predicted value (remember it is regression)

> Output of tree 1 + output of tree 2 + ....

- For better prediction and lower variance , learning rate is used.( to avoid overfiting )
- The residuals become smaller( predictions tend to be better ) as new trees are added.

- In case of classification `pseudo residuals` is calculated from `log(odds)`
- Using `log(odds)` probability is calculated and according to that `pseudo residuals` ( predicted probability - real probability ) are calculated 

> Yes -> 1
  No -> 0

- Finally according to the threshold the probability is then classified as `Yes` and `No`.
- Gradient Boost usually uses 8-32 leaves


- Supports custom loss function and hence used for many techniques