
set through dropout ratio(p)
- also known as **regularization through randomization**
- reducing nodes help in improving accuracy
- randomly turns off some nodes in each layer for an epoch
- different nodes for each epoch
- like training on different neural network each epoch
- increases accuracy ᯈ 2%

![[Pasted image 20250629235856.png]]

- it reduces number of nodes
- each node is equally treated rather giving importance to a single node (the node having more weight has more say in the output)
> the training is such that it gives equal importance because the network does not know whether or not that one node on which it is dependent will be present or not in next epoch, hence it has to take every node at equal level
>
> - this helps is avoiding complex relationships which occur due to network's focus on a single feature

##### RANDOM FOREST AND NEURAL NETWORKS ANALOGY
- ensemble of different neural networks
> similar to **Random Forest**- ensemble of **Decision Trees**

- similar to **Random Forest** it also reduces overfitting

##### HOW ARE WEIGHTS CALCULATED?
- each node need not be present in training, but during testing all nodes are present.
- weight can not be properly calculated during training as node is not always present.
- w<sub>test</sub> = w<sub>train</sub> x (1-p)

> (1-p) is the probability of node being present during the training, hence multiplied by the weight obtained after training

#### TIPS
- If overfitting then increase p, if underfitting then decrease p
- try implementing dropout only to last layer, rather than in every layer.
- greater than 0.5 (50%) not recommended.

#### DRAWBACKS
- convergence is delayed
- loss function varies as all nodes are not available all the time
- hence debugging values of weights is also difficult