- we want to minimize the loss function and hence optimizers are used.
- we use **gradient descent** in DL

##### PROBLEMS WITH GRADIENT DESCENT?
- finding the correct learning rate
- to overcome this **learning rate scheduler** was invented , where we have to pre-define the schedule of learning rate according to which it decreases.
> this means that it is dataset independent, does not perform well on every dataset

- learning rate for every parameter is same, we can not set different learning rates for different parameters
> we move equal steps in every direction and not according to the graph 

- we get stuck at local minima, rather than reaching the optimal solution- **global minima**.
- saddle point- where the slope is same in every direction and derivative is 0, leading to no updates in weights and getting stuck to it (sub optimal solution)

### EXPONENTIALLY WEIGHTED MOVING AVERAGE
- technique used to find trends in a time series data
> like fluctuating temp in a city

- used in DL to make optimizers
- weightage of a new point (latest in time) > weightage of an older point
- weightage/importance of a point/data decreases with time
- EWMA(t) = (1-β) * Y(t) + (β) * EWMA(t-1)
> - Y(t): data at current time
> - EWMA(0)=Y(0)
> - β (smoothing const): 0<β<1
> - β = 1-α

- take it as if calculating avg of last 1/(1-β) days
> - assume β=0.9 (mostly used) then calculating EWMA of a point is like calculating normal average of previous 10 days
> - if β=0.5 then it is like calculatin avg of previous 2 days
> - more the value of β more is the importance given to previous data points
> - less β give less importance to old points rather depends on the latest points

- MATHEMATICAL INTUITION
> - formula is such that the older data points are multiplied by higher power of β, making their weights small