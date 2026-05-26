- used in sequential data
> test,speech,time series,etc

### WHY NOT ANN?
- in case of text data, input size varies in ann and hence weights matrix vary which cannot be the case
- if we think of apply 0 padding by knowing the length of largest text, there would be many zero inputs.
- Also the test data can be longer than longest text in the training data
- Sequence of data (semantic meaning) is lost

Data in RNN is sent in the form of **(timesteps,input_features)**
- "movie was good"
> [[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0]]
> - 3 time steps and 5 input features