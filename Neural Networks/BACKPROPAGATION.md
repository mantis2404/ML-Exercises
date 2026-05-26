basic understanding:
```
for i in range(epochs):
	for i in range(entries):
	
		- single data -> forward propagation -> predict output
		- use that output to predict loss
		- adjust the weights and bias ( using GD )
		- calculate avg loss
```

>w<sub>new</sub>=w<sub>old</sub> - 𝜼 x $\frac{𝛿L}{𝛿w}$ 

- loss function is a function of all trainable parameters
- w<sub>new</sub>=w<sub>old</sub> - 𝜼 x $\frac{𝛿L}{𝛿w}$  , it decreases/increases the parameter,depending on value of gradient, in order to decrease the loss function
- the value of parameter always move in opposite direction of the slope

> if the slope is positive then decrease the value of parameter

- learning rate helps is avoiding overshooting, bigger 𝜼 leads to missing the minima and smaller 𝜼 may take more time to converge.
- convergence -> gradient is 0.

###### IF BACKPROPAGATION IS NOT USED
- weights will never get updated
- output will be randomized

So basically the neural network learns/modifies the parameters from each data entry to minimize the loss function ,using back-propagation, and this is done over many epochs until the parameters are such that the loss function reaches the minima ,implying that now it can predict values similar/closer to actual values. 

And hence now the model/network is trained.

### MEMOIZATION
- similar to pre-computation in CP
- the derivatives are stored
- hence no need to re-calculate everytime( while doing derivatives of hidden layers, they are useful(product + chain rule))
- saves execution time
- occupies memory


**BACKDROP = CHAIN-RULE + MEMOIZATION**
