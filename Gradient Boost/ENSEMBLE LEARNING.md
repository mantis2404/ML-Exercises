**ENSEMBLE TECHNIQUES ARE PREFERRED BECAUSE WE WANT MODELS WITH LOW VARIANCE AND LOW BIAS**

- base models should be different
> it can be made by using model with different algorithms or same model but trained on different datasets which will perform differently
- final output is based on majority count (for classification)
- final output is mean of all outputs (for regression)
- gives low bias and low variance
> decision tree itself is a low bias,high variance model but ensemble of decision tree makes gives low bias and low variance

### VOTING ENSEMBLE
- base models having different algorithms
- majority wins or mean of all the outputs
- base models should be different
- each base model individual accuracy > 50%
> if it is less than that then the final accuracy will be lesser than the smallest 
 -  the mathematics behind this is conditional probability

**VOTING CLASSIFIER**
- **SOFT VOTING**: avg probabilities for each class from every model is calculated and the class with highest probability is the final output
- **HARD VOTING**: each model gives it prediction (class with highest probability) and the class with highest votes from each model wins
- soft voting results in softer margins, sometimes improving the accuracy
- we can assign weights to the base models also using `weight` hyperparameter
- **HELP IN HYPERPARAMETER TUNING**: we can take our base models with diff hyperparameter values , their combined result is better than individual
>[!NOTE]
>- **Few models + multi-class**: Some classes may never win unless directly predicted. 
>- **Even models + binary**: Can cause tie votes in hard voting.
>- Use **odd number** of models or prefer **soft voting** to avoid such issues.

**VOTING REGRESSOR**
same idea as voting classifier

>[!NOTE]
>- the avg of probabilities is taken if prediction (probability)of a sample is the output
>- in case of accuracies(over the whole dataset) binomial probability is used
>$$P(\text{ensemble correct}) = \sum_{k=\lceil \frac{n}{2} \rceil}^{n} \binom{n}{k} p^k (1-p)^{n-k}$$

### BAGGING (BOOTSTRAP AGGREGATION)
- base models are same but all are trained on different chunks of same data with replacement (**BOOTSTRAPING**)
- each model behaves differently
- majority wins (**AGGREGATION**)
- it gives consistent results despite changes in original data
> if change is made in the data then that changes are also **distributed among the models** and so does its effect, hence the noise/modification in the data is reduced resulting in **low variance**
- transform LBHV models into LBLV
- default `estimator` is `DecisionTreeClassifier`/`DecisionTreeRegressor`
- `bootstrap=True` means rows can be repeated within the model
- `bootstrap=False` means rows in the model will be unique but can repeat inter model

**OOB SAMPLES**
- due to bootstrapping some samples (approx 37%) are never fed to models for training
- we can check our accuracy over those samples also to get a rough estimate using `oob_score_`

**TIPS**:
- bagging generally gives better results than Pasting
- for row sampling 25-50% mark is best suited
- Random patches and subspaces should be used when dealt with high dimensional data
- the models with **high variance** benefit from bagging
> decision trees (fully grown), svm, knn (with low neighbor count)

>[!NOTE]
>**Handling Ties in Ensembles:**
>
>- In case of a tie (e.g., 50 votes for class 0 and 50 for class 1), `BaggingClassifier`(or `RandomForestClassifier`,`VotingClassifier`) picks the **lowest class label** (e.g., 0).
>- This behavior is **consistent but arbitrary**.
>- Important for **imbalanced datasets** — can introduce bias.
>- Tree models do support `.predict_proba()`, but outputs may need **calibration** for reliability.
>
>ALSO FOUND IN
>- **Hard voting in ensembles** (`VotingClassifier`, `BaggingClassifier`, `RandomForestClassifier`)
>- **`argmax` over equal probabilities** (e.g., `softmax([0.5, 0.5])` → picks index 0)
>- **KNN (KNeighborsClassifier)** if equal votes for multiple classes → picks **lowest label**

#### **BAGGING v/s RANDOM FOREST**


| BAGGING                                                                                                                                   | RANDOM FOREST                                                                                                                              |
| ----------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| different base estimators can be used                                                                                                     | base estimator is decision tree                                                                                                            |
| the features given to a tree (estimator) are decided initially only and the branch nodes of that tree will infer from those features only | the features for each tree and its branch nodes are decided recursively, meaning every tree and its branch has all features to select from |
| **tree level sampling**                                                                                                                   | **node level sampling**                                                                                                                    |
| less randomess                                                                                                                            | more randomess, hence better performance                                                                                                   |

### BOOSTING
- same base models, each model after training notes its mistakes and next model is trained so that those mistakes are corrected
- models with **high bias and low variance** preferred

| BAGGING                                                     | BOOSTING                                                                                 |
| ----------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| LBHV                                                        | HBLV                                                                                     |
| parallel (all models get train simultaneously on diff data) | sequential (a model is trained only after leaning from previous model)                   |
| each model has equal say                                    | each model has different weights depending on how well it performed on the training data |

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/303c288a-9433-43c2-bd8d-52f50d8ca838" />

### STACKING
- base models are trained
- a new model (**meta model**) whose input is predictions of base models and the true target value is trained
- it then assigns weights to models, the model which has better predictions had more weights

  <img width="1400" height="775" alt="image" src="https://github.com/user-attachments/assets/de1a4571-e8b5-45bd-a0cd-c3d2cb12ab6a" />


**PROBLEM**
- same data used for training and predictions of base models

**SOLUTION**

- HOLD OUT METHOD: if used process is know as **Blending**
- K-FOLD METHOD: if used process is still knows as **Stacking**

**HOLD OUT APPROACH (BLENDING)**

- Training dataset is also further divided
	- X_train = X_base_train + X_base_validate
	- X = X_train + X_test
- base models trained on X_base_train
- they are used for prediction on X_base_validate and form a new dataset with it
- the new dataset is used for training meta model
- for final prediction X_test is used

**K-FOLD APPROACH (STACKING)**

- Training data is divided into K-folds (usually K=5 or 10)
- We train the base model for multiple epochs, in each iteration training folds and testing folds are different
	- out of these K folds, K-1 folds are used for training the base models
	- the last fold is used for doing predictions on base model to form a new dataset
	- each base model is trained multiple times, hence we have different variants of same model
- the above step is repeated for all base models
- training for multiple epochs gives us more data than a single fold
- BUT none of these models are used for final prediction on test data, they were only to generate data for meta model
- the final base models used are trained on complete training data (X_train)
- meta model obtained earlier than base models

<img width="850" height="284" alt="image" src="https://github.com/user-attachments/assets/dddd5f4e-6b28-4f6e-b935-d2a501f89497" />


#### MULTI LAYER STACKING

<img width="850" height="481" alt="Pasted image 20250711053650" src="https://github.com/user-attachments/assets/62af2e2a-f53a-4e2a-a2ce-0e312240b076" />

