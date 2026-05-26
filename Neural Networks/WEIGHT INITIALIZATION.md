bad weight initialization results in

- vanishing gradient problem
- exploding gradient problem
- slow convergence
- 
### WHAT NOT TO DO?

#### 1. ZERO INITIALIZATION

**in ReLU and tanh**
- the weighted sum (input for the activation function) is 0
- output of activation function is 0
- derivative is 0 and hence **weights are not updated**
- **No training**

**in sigmoid**
- if weights are initialized as 0 and activation function is sigmoid then all the nodes in the layer behave as a single node only, making the model linear.

#### 2. NON-ZERO INITIALIZATION

- the weighted sums become equal and so the output of the activation function, all the weights emerging from a single neuron are updated equally and hence combine to behave as a single weight resulting in single neuron in that layer resulting in linear model
> same as in case of zero-initialization in case of sigmoid function.

#### 3. RANDOM INITIALIZATION

**SMALL VALUES (random value x 0.01)**

- if weights are initialized with small random values then for **sigmoid** and **tanh** it results in vanishing gradient problems.
- for **ReLU** the convergence is slower but it is not much affected by the vanishing gradient problem

>Small weights → small neuron outputs → for sigmoid/tanh, this keeps inputs near 0 → where slope is high → but gradients multiply across layers → still shrink layer by layer → vanishing gradients over depth.

**LARGE VALUES [0,1]**

- the outputs of neurons can become extreme and resulting in saturated values (max/min values of the activated function) in case of **sigmoid** and **tanh**
- results in slow training and vanishing gradient (as gradient ≈ 0)
- for **ReLU** the gradients tend to be bigger due to bigger output and this results in non uniform weights updation (bigger jumps) and slower convergence and unstable training.

> Large weights → large neuron outputs → for activation functions like sigmoid or tanh, this pushes the input into saturation zones (far from 0), where the derivative is very close to 0 → hence gradients become small → weights update slowly → leads to vanishing gradient-like behavior.

### WHAT TO DO?
we want weights not too big nor too small

what we were doing:
>for small values
> - np.random.randn(20,20)* 0.01

>for large values
> - np.random.randn(20,20)* 1 

in order to generate perfect weights these numbers cannot be constant rather it should be decide based on the architecture of the input.

**variance =1/n**

> n is the number of inputs to a given node for which the weights are decided

#### XAVIER/GLOROT INITIALIZATION

##### NORMAL DISTRIBUTION 

**USED FOR TANH NOT FOR RELU**

- mostly used: √1/fan_in
> **fan_in**: no. of inputs coming to that node
- sometimes used: √2/(fan_in + fan_out)
> **fan_out**: no. of outgoing weights

##### UNIFORM DISTRIBUTION

weights initialized in [-limit,limit]
> limit = √6/(fan_in + fan_out)

#### HE 
##### NORMAL DISTRIBUTION

**USED FOR RELU**

- √2/fan_in
##### UNIFORM DISTRIBUTION

weights initialized in [-limit,limit]
> limit = √6/(fan_in)

>[!NOTE]

> - SET THROUGH `kernel_initializer`='glorot_uniform'
> - add this to the layer where you want to change your weights
> - **DEFAULT IS GLOROT UNIFORM**
> - the normal/uniform version has nothing to do with the distribution of the input data
> - the normal/uniform version initializes weights such that they follow such distribution

**GLOROT NORMAL**

![img](Pasted%20image%2020250701004701.png)

**GLOROT UNIFORM**

![img](Pasted%20image%2020250701004715.png)

**HISTOGRAMS GIVE BETTER IDEA OF WEIGHT DISTRIBUTION THAN KDE PLOT**
