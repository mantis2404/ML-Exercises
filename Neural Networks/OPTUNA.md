There exists a relation between hyperparameters and. Optuna using **Bayesian Search** (default but can change) to finds and optimizes that function (**Objective Function**)

The accuracy of previous combinations of hyperparameters helps in predicting the next set of hyperparameters

SOME KEY TERMS:
- **study**:collection of trials aimed at optimizing the objective function
- **trial**: a single iteration of the optimization process where a specific set of hyperparameters are evaluated
- **trial parameters**: value of hyperparameters
- **sample**: algo which tells which hyperparameter to evaluate next 
> default: Tree-Structured Parzen Estimator (TPE)

also helps in deciding which model is best, by using **Define By Run** and **Dynamic Search Spaces**
- algorithms/models to be used are also treated as a hyperparameter
- if a trial is being run on some specific algorithm then that algorithm specific hyperparameters only will be tuned int that trial