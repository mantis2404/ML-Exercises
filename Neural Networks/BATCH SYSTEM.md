## TYPES OF GRADIENT DESCENT

##### BATCH GRADIENT DESCENT
Trained on entire dataset

- in one epoch the values for entire dataset is predicted
- loss is calculated on those predictions (entire dataset)
- and then parameters are updated 
- faster than Stochastic with same number of epochs
- loss function minimizes **smoothly**
- Uses **vectorization**
- **PROS**: rather than using loop it uses dot product which makes it faster  
- **CONS**: large dataset may lead to memory exhaust
- **no. of times parameter updated = epochs**
- `batch_size`=n

##### STOCHASTIC GRADIENT DESCENT
Train using one data point at a time ( e.g. SGD)

- in one epoch, a for loop runs on entire dataset ,one random point is selected ,after being shuffled, and then its loss is calculated and according the parameters are updated.
- this is done for each value and then after 1 epoch average loss is calculated.
- convergence is faster (reaches same accuracy as batch descent with lesser number of epochs)
- loss function minimizes with **spikes** (due to random point selection)
- **PROS**: helps in avoiding being stuck at a local (small) minima and eventually pass to bigger minima due to randomized selection
- **CONS**: does not converge exactly at minima
- **no of time parameters updated = epochs X no of rows**
- `batch_size`=1

##### MINI-BATCH GRADIENT DESCENT
Train on small batches ( subsets ) of data (e.g. in deep learning)

- mix of both batch and stochastic descent
- divides dataset into multiple batches 
- then on each batch loss is calculated from predicted values using vectorization (batch descent)
- midway between batch descent and stochastic descent in every aspect
- mostly used 
- `batch_size`= <batch_size>

**USE HYPERPARAMETER `batch_size` FOR DEFINING BATCH SIZE DURING MODEL FITTING**

**MOST MODELS DO NOT EXPOSE BATCH SIZE BUT IN DEEP LEARNING IT IS POSSIBLE**

| Strategy       | Update Granularity     | Convergence Speed | Stability         | Use Case                            |
| -------------- | ---------------------- | ----------------- | ----------------- | ----------------------------------- |
| **Batch**      | Entire dataset         | Slower            | Very stable       | Small datasets or convex problems   |
| **Mini-batch** | Few samples (e.g., 32) | Medium            | Moderately stable | Deep learning, balanced performance |
| **Stochastic** | Single sample          | Fast (initially)  | Unstable, noisy   | Online learning, very large data    |

**STOCHASTIC GRADIENT DESCENT LOSS**

![img](Pasted%20image%2020250629025755.png)

**BATCH GRADIENT DESCENT LOSS**

![IMG](Pasted%20image%2020250629025917.png)

**MINI-BATCH DESCENT LOSS**

![img](Pasted%20image%2020250629025945.png)

![img](Pasted%20image%2020250621061358.png)

#### `batch_size` IS ALWAYS PROVIDED IN MULTIPLE OF 2 TO EFFECTIVELY USE RAM
>OPTIMIZES CODE AND MAKES IT FASTER

#### WHAT IF BATCH_SIZE DOES NOT DIVIDE THE NUMBER OF ROWS?
> THE LAST BATCH IS MADE WITH THE REAMAINING ROWS

> FOR EXAMPLE rows=211 ,batch_size=20 then first 10 batches will be of 20 rows and the next batch will of 11 rows only


### ONLINE LEARNING

**Online learning** is a machine learning method where the model is updated incrementally with each new data point, making it ideal for real-time, streaming, or continuously changing data.

using SGD(Stochastic Gradient Descent) is online learning in action
