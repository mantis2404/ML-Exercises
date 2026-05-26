# ACTIVATION FUNCTIONS

if no function is defined (or `activation=linear`) then the model behaves as linear model

captures non-linearity in data

###### IDEAL ACTIVATION FUNCTION SHOULD BE:

- non-linear
- differentiable
- computationally inexpensive 
> it should be easier to calculate the derivative of the function

- zero centered (normalised / mean= 0)
> the function should be extended in both +ve and -ve y values resulting in faster convergence
> - eg **tanh**

- non-saturating (should not squeeze the input)
> **sigmoid** and **tanh** squeeze the input resulting in **vanishing gradient problem**

---

### SIGMOID

$\sigma(x) = \frac{1}{1 + e^{-x}}$

##### PROS
- output: [0,1] 
- used in binary classification
- non-linear
- differentiable

##### CONS
- Saturating function, results in vanishing gradient problem
> hence not used in hidden layers

- non zero centered 
> using sigmoid the derivative of weights for that layer is either +ve or -ve, meaning the change in all weights for that layer is either +ve or -ve implying that the loss function is minimized with restrictions/biasness and hence convergence is slower

- computationally expensive
> derivative calculation for exponents is tough

---

### TANH

##### PROS
- Output: [-1,1]
- non-linear
- differentiable
- zero centered
> weight gradients are both -ve and +ve

##### CONS
- Saturating function
- computationally expensive

- - -
### RELU

##### PROS
- non-linear
> it is max(0,x) and can be combined to form non linear forms also

- not saturated in the positive region
> goes to ∞ for x>0

- computationally inexpensive
- convergence is faster than **sigmoid** and **tanh**

##### CONS
- not completely differentiable
> kink at x=0, hence assumed that derivative = 0 for x<0 and derivative = 1 for x>=0

- not zero centered
>[BATCH NORMALIZATION](BATCH%20NORMALIZATION.md) is used to solve it

- dying ReLU problem
> [MODEL PERFORMANCE](IMPROVE%20MODEL%20PERFORMANCE.md#^dyingrelu)

## VARIANTS
### - LINEAR

#### 1.LEAKY RELU

f(X) = max(0.01 x z, z)

**PROS**
- non-saturated and unbounded both sides
- easily computed
- no dying relu problem
- close to zero centered

#### 2.PARAMETRIC RELU

**_f(x) = max(ax, x)_**

similar to leaky ReLU except that rather than 0.01, a trainable hyperparameter **a** is used

more flexible
![GRAPH](Pasted%20image%2020250630051517.png)

## - NON-LINEAR
#### 1.ELU

**_f(x) = { x, if x > 0; a(exp(x) — 1), if x <= 0 }_**

![graph](Pasted%20image%2020250630051503.png)

 alpha is a constant

**PROS**
- always continuous and differentiable
- sometimes better than ReLU
- generalises better
- close to zero centered
- no dying relu problem

#### CONS
- computationally expensive

#### 2.SELU

**_f(x)={ λx, if x>0;  λα(exp(x) −1), if x≤0}_**

λ and α are constants
> **λ≈1.0505** and **α≈1.6732​**

- scaled version of eLU
- relatively newer version
- **self normalizing**: activation is normalized; mean=0 and std=1
> because of this and the property that it has both negative and positive values, it converges faster 

---
### TIPS:
- using `tanh` when inputs are 0 centered and network is shallow, gives better results than `ReLU`, which works better otherwise.
