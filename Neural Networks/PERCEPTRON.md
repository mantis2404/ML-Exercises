### PERCEPTRON

- model for supervised learning 
- similar to neuron
- has weights and biases
- weights help determine which factor affects the model more
- Σ w<sub>i</sub>x<sub>i</sub> passes to activation function which then generates the output
- binary classifier
- works on linear with soft margin
- eq. used is Ax<sub>1</sub>+Bx<sub>2</sub>+C=0
- quite similar to logistic regression except for the learning trick

### PERCEPTRON TRICK

> If the model misclassified a point, push the decision boundary so the point ends up on the correct side.

- Starts with random line
- Every time a point is selected randomly from the sample and then checked if it is misclassified or not
- This loop is run till convergence (no point is misclassified) or 1000 times (epoch)
- Ax<sub>1</sub>+Bx<sub>2</sub>+C>0 corresponds to +ve region and Ax<sub>1</sub>+Bx<sub>2</sub>+C<0 corresponds to -ve region

##### HOW DOES THE LINE CHANGES BASED ON THE MISCLASSIFIED POINT
A new coordinate with value unity is added to point, then each coordinate is subtracted/added to the coefficients of the specific dimensions and the values thus generated are coefficients of new line

new_coeff = old_coeff - η X coordinates

> η = learning rate

###### USEFUL FOR
- Linear binary classifiers
- perceptron model

###### NOT SUITABLE FOR
- Deep neural networks
- non-linear decision boundaries
- any model having smooth diff loss function

##### CONS
- no probability output
- only update on errors
- no way to quantify ( determine which line is better )
- no loss function
- no more in production
- inefficient for noisy dataset
- no soft margins
- **JUGAAD** 

This is basically SGD use case
### PERCEPTRON LOSS FUNCTION 

$$
L(y, f(x)) = max(0, - y \times f(x))
$$
> similar to hinge loss (in SVM)

$$
L=\frac {1}{n} \sum_{i=1}^{n} max(0, - y_i \times f(x))
$$

> $f(x)=w_1\times x_1+w_2\times x_2+b$

In this loss function the contribution of correctly classified point is 0 but if it is misclassified it contributes based on its distance (weights) from the guessed line

We can change how the perceptron gives its output based on the activation and loss function.
- **step-function**- 1/0
- **sigmoid-function**- probabilities
- **softmax**- multi-class classification
- **linear(no activation function)**- linear regression


### MLP(MULTI-LAYER PERCEPTRON)

A kind of linear combination of perceptrons. Weights and biases can be added to the results of individual perceptrons which then becomes the input to another perceptron, thus creating a mlp which can generate non-linear boundaries also.

>[!NOTE]
> - weights are given to the connections between two layers
> - biases are given to nodes
##### CHANGING ARCHITECTURE OF NEURAL NETWORK

- More nodes in hidden layer help in better capturing of non-linearity 
- increase nodes in input layer (this will change the dimensions)
- increase nodes in output layer 
> mostly done in case of multiclass classification, each node representing a class, the node with most probability is used for classification

- increase no. of hidden layers (depth of neural network)
> helps in capturing complex to complex data patterns
