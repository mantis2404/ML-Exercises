- makes DNN training faster and more stable
- nonlinear function
- activations (output of neurons) are also normalized(mean=0 and std=1)

### COVARIATE SHIFT
- distribution of train and test data is different
- hence model has be retrained on similar distribution

### INTERNAL COVARIATE SHIFT
- if we hypothetically split our network into two diff neural networks then the input of the last neural network are the output of activation function which is constantly changing and so its distribution.
>it is the change in distribution of network activations due to the change in network parameters during training.

Batch normalization transforms the output of every layer into Gaussian distribution and hence reducing internal covariate shift thus making the training stable

- if batch normalization not used then learning rate be low, to avoid overshooting problem

##### KEY POINTS
- applied on mini-batch gradient descent
- applies on layer by layers basis (one layer at a time)
> optional to apply for every layer, can apply for individual layers also

- z->z(normalized)->g(z(normalized))->a OR z->g(z)->a->a(normalized)
-  z(normalized) is then transformed to 𝛄 x z(n) + β
> β and 𝛄 are **learnable parameters**, **distinct for every neuron**, with initial value β=0 and 𝛄=1 and are changed during training through back propagation

> we are first normalizing and then doing its opposite, scale and shift, why??
> - sometimes our neural network does not need its activations to be normalized, through these parameters we are giving flexibility to the network to change its distribution according to its choice (need not be normal)

- keras treat it as a different layer

###### how does batch normalization work for a single test data?
- how does it calculate mean and std
- uses Exponentially Weighted Moving Average
> while training EWMA is maintained for mean and std and thier last updated values are used for single data testing

- for every neuron in batch normalization layer 4 parameters are stored
>  - **non-learnable**: 𝛔(EWMA), µ(EWMA)
>  - **learnable**: β, 𝛄

##### PROS
- more stable training
> hyperparameters can be set to a more wider range

- faster training
> higher learning rate can be set now as the loss function is more uniform

- act as regularizer
> helps in regularization, randomness is introduced while changing values of µ and 𝛔

- reduces impact of weight initialization
> cost function becomes more uniform (rather than being stretched)