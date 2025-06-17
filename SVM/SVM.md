### MAXIMUM MARGIN CLASSIFIER
- **What it is**: The **simplest form** of an SVM.

- **Assumes**: Data is **perfectly linearly separable** (i.e., you can draw a line or hyperplane that completely separates classes with no error).

- **Goal**: Find the hyperplane that **maximizes the margin** — the distance between the decision boundary and the nearest points from both classes.

- **Limitation**: Not useful when data is noisy or overlaps.

### SUPPORT VECTOR CLASSIFIER
- **What it is**: A **generalization** of the Maximum Margin Classifier.

- **Allows for**: **Some misclassifications** using a **soft margin** approach.
   
- **Introduces**: The **regularization parameter `C`**, which allows the model to tolerate points inside the margin or on the wrong side of the boundary.
 
- **Still linear**, but **more practical** than MMC.

### SUPPORT VECTOR MACHINE
- **What it is**: An **even more general model** than SVC.

- **Includes**: Nonlinear classifiers using the **kernel trick** (RBF, polynomial, sigmoid, etc.).

- **Covers**:

   - MMC (hard margin)
       
   - SVC (soft margin)
     
   - **Nonlinear SVMs** using **kernels**

Helps in classifying the data that no linear classifier could separate 
Transforms the lower dimension data into higher dimension(like 1D to 2D) and then finds a SVC to classify the observations

The `degree` in `poly` kernel determines the dimension the data transforms to and hence finds the respective SVC
##### KERNEL TRICK
calculating high dimensional relationships without actually transforming the data to higher dimension 
