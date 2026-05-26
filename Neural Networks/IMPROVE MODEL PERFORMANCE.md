- multiple hidden layers with less neurons >>> few layers with multiple neurons. 

> - hierarchy of data is easily captured with Deep Neural Network
> - the early layers capture the primitive data and then the further layers capture data on basis of it
> - helps in **Transfer learning** (Reuse some layers of an already trained neural network)
> - applicable until overifitting is not there

- **BATCH SIZE**

| SMALL BATCH                                         | LARGE BATCH                                                                                                                                                                    |
| --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **size** : 8 - 32                                   | **size **: 8192                                                                                                                                                                |
| slow                                                | fast                                                                                                                                                                           |
| generalizes better (better performance on new data) | **warming learning rate**:varying learning rate over epochs makes it better as it is fast due to its batch size.<br>the rate is small at first and then is gradually increased |

---
## PROBLEMS
### VANISHING GRADIENT PROBLEM

- derivatives w.r.t parameters are very small
- multiple multiple of these (chain rule) makes the net derivative more smaller
- then multiplying with learning rate 
- eventually the new weight is similar to old weight
- this leads to no change in loss
- the neural network training all together stops
- more prevalent in DNN's
- MOSTLY in **sigmoid** and **tanh** activation functions as they shrink the output to 0-1.

##### HOW TO RESOLVE?

1. Reduce model complexity (hidden layers)
2. Use different activation functions like `ReLU` for hidden layers
> derivative is only 0 or 1 and does not squish the output

3. Proper weight initialization 
> [[WEIGHT INITIALIZATION]]

4. [[BATCH NORMALIZATION]]
5. Using residual network

### EXPLODING GRADIENT PROBLEM

- the derivaties become too large
- product of learning rate and the gradient is large
- new weight and old weight differ a lot
- does not decrease the loss and model behaves randomly
- MOSTLY in RNN
- can be resolved using **GRADIENT CLIPPING**

### DYING RELU PROBLEM

^dyingrelu

- If ReLU is chosen as activation functions for hidden layers then the output of some neurons is 0 irrespective of the input and it becomes **dead neuron**, not participating in the network forever.
> once a neuron is dead it cannot be restored

- if >50% are dead neurons then patterns in data are not clearly observed.
- when the weighted sum (z) of inputs for a neuron < 0 **(which is the input for the activation fucntion)** then their output is 0 (ReLU function) and hence their derivatives are 0 and hence the weights are not updated.

assume z=w1x1 + w2x2 + b1
> if it is negative then weights are not updated. when is it -ve?

- **poor weight initialization**
- **high learning rate (𝜼)**: after updating the weights become negative
- **high -ve bias**: maybe due to high learning rate or high bias initialization

##### HOW TO RESOLVE?
- set low learning rate
- set +ve value for bias
> **0.01** is a good value (experimentally proven)
- use variants

### NOT ENOUGH DATA
- Transfer learning
- Unsupervised pre-training

### SLOW TRAINING
- optimizers
- learning rate scheduler 
> changes learning rate over epochs

### OVERFITTING
- [[EARLY STOPPING]]
- Reduce complexity/Increase Data
- [[DROPOUT]]
- l1/l2 regularization
> - L2 is preferred (also called **weight decay**)
> - (1-𝜼ƛ) is weight decay factor
> - only weights are accounted and not biases
> - weights decreases

### SCALING
- [[STANDARDIZATION AND NORMALIZATION]]