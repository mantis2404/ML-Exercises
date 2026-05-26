- for grid like structures (more used for images)
- ANN uses matrix multiplication , CNN uses convolution
- consists of:
	 - convolution layer
	 - polling layer
	 - FC layers

### WHY NOT USE ANN?
- high computational cost
> larger the size of image more the weights 

- overfitting
- loss of important features like spatial arrangement of pixels
> distance between pixels in original image might matter to distinguish the image, which is lost when the image pixels are converted to 1-D array to shrink it

### CNN
- initial layers captures **primitive features** (edges and all)
- further layers capture complex features

#### CONVOLUTION

- identifies features in the images
- edges are boundary b/w regions of changed intensity 
- to identify them, a filter/kernel is used
> basically of matrix of 3x3 (approx), which is then convolved with the image pixel grid

$$
\begin{bmatrix}
-1 & -1 &-1 \\
0 & 0 & 0 \\
1 & 1 & 1
\end{bmatrix}
$$
> this filter identifies horizontal edges

**IMAGE * FILTER = FEATURE MAP**
> * is convolution symbol

- filter is overlayed on the image grid and then corresponding elements are multiplied, their sum results in one entry of feature map
- filter and their values are similar to weights,initialized completely randomly and then corrected through back propagation.

![img](Pasted%20image%2020250704011229.png)

black and white images have only 1 channel
> image (n x n) * filter (m x m) = feature map (n-m+1 x n-m+1)

> - using a left edge filter we get this feature map
> - blue corresponds to -ve values and red corresponds to +ve values 
> - further a ReLU function will be applied ( $max(0,x)$ ) which eliminated -ve values (blue) and we will be left with +ve values only (left edges)

- for RGB images we have 3 channels
- filters are of 3x3x3 dimensions, each for 1 channel
> image (n x n x c) * filter (m x m x c) = feature map (n-m+1 x n-m+1)
- feature map is single channel
- if multiple filters are applied on single image then the feature map is multiple channeled (3d object with volume) and its depth is determined by the number of filters applied (or no of single channeled feature map generated)

![img](Pasted%20image%2020250704012723.png)

**CONS**:
- size of feature map is smaller than the image, consecutive convolution operations result in more small size and hence the data loss
> to resolve this extra rows/columns are added around the image

- the corner pixels of images have very less participation in feature map (they are part of convolution only once) compare to the center ones.

>[!NOTE]
>WHY USE ACTIVATION FUNCTION IN `Conv2D` LAYER,ESPECIALLY `ReLU` AND `tanh`?
>- to introduce non-linearity, convolution itself is a linear function without non-linearity despite stacking up layers it will only catch linear patterns
>- help recognize different feature effectively
>- like for image classification every boundary there is not linear, cat and dog can not be separated based on linear boundary
>- reshape feature maps


#### PADDING
- Extra rows and columns are added around the image 
> pixel values are 0 (most cases) in those rows, hence also called **zero-padding**

- new dimension of feature map is (n+2p-m+1)
> p is the padding (p=1 if one row/column is added)

- `padding = 'valid'`: for no padding
- `padding = 'same'`: for corresponding padding according to the input
- padding can be custom also, different for rows and columns

#### STRIDES

- when the filter moves over the image it moves 1 pixel (1 block) at a time (whether towards right or downwards)
> stride = 1
- if stride = 2 then filter will take 2 jumps
- bigger stride reduces the dimensions of feature map
- feature map dimensions:
  
$$
\frac {n+2p-m}{s} +1
$$

- if stride > 1 then it is known as strided convolution
- strides are not used for moving across channels (irrespective of number of channels)

**REQUIREMENT**
- higher stride mean capturing only high level information
- lower stride mean capturing even minute details

- `strides = (<stride_for_row> , <stride_for_column>)`
##### WHAT IF ALL THE PIXELS ARE NOT BEING COVERED BY THE FILTER DUE TO CHANGED STRIDE VALUE
- then the last operation in which pixels are not considered is ignored
- this may result in decimal values for dimension of feature map
> to avoid this **floor** operation is used on them

#### POOLING

Resolves:
- memory issue with convolution
- translation variance
> features become location dependent with the use of convolution, which they should not be

Need to specify:
- type
- pool_size: size of **receptive field**
- stride

it is like eliminating low level details and keeping only the high level details

![img](Pasted%20image%2020250704035754.png)
Using `Max Pooling` the size of the image is downsized (28 x 28 -> 14 x 14)

In case of multiple feature maps due to multiple filters, first ReLU is applied independently on them and then pooling on single feature map

**TYPES**
- **Max pooling**: get max number from the receptive field
- **Avg pooling**; get the average of numbers from the receptive field
- **Global Max pooling**: the entire matrix becomes a receptive field and hence give the max value from the entire matrix
- **Global Avg pooling**: get the avg value of whole matrix

Global pooling is used in the end when we flatten our data and transfer it to Fully connected layers, to reduce overfitting

>[!NOTE]
>In case of multiple feature maps, for **Global pooling** we get 1 value for each map
>
>e.g., 4 x 4 x 3 transforms to 1 x 3 ,each map of 4 x 4 gives one value


**PROS**:
- reduces size of feature map
- translation invariance
> regardless of the location of the feature after applying pool the feature map is same

- Enhanced features (only in case of Max Pooling)
- no need of training (no parameter just basic maths, hence no learning through back propagation)
> hence better than strides, Moreover:
> - in strides we blindly ignore data but in pooling the data selected is significant 
> - pooling gives more flexibility and robustness

**CONS**:
- not used in image segmentation tasks
> we want our feature to be present where it is and changing it should have consequences, translation variance
- data loss

### CNN ARCHITECTURE

![img](Pasted%20image%2020250704050511.png)

### LENET-5

![img](Pasted%20image%2020250704051235.png)

| Layer  | Type                     | Details                           |
| ------ | ------------------------ | --------------------------------- |
| Input  | Image                    | 32×32 grayscale image             |
| C1     | Conv2D                   | 6 filters, 5×5 kernel → 28×28×6   |
| S2     | AvgPooling               | 2×2 pool → 14×14×6                |
| C3     | Conv2D                   | 16 filters, 5×5 kernel → 10×10×16 |
| S4     | AvgPooling               | 2×2 pool → 5×5×16                 |
| C5     | Conv2D (Fully Connected) | 120 filters, 5×5 kernel → 1×1×120 |
| F6     | Fully Connected          | 84 neurons                        |
| Output | Fully Connected          | 10 classes (softmax for digits)   |

#### **Key Features:**

- Early use of **convolution + pooling**
- Uses **tanh activation**
- No ReLU or dropout (added in modern CNNs)
- Small and efficient, ~60K parameters

### ANN vs CNN
For classification of a number of size 28 $\times$ 28 
#### ANN
- having multiple fully connected layers with input dim as 784

#### CNN
- generates feature map
- adds bias to them
- pass through activation functions
- then pass to max pooling layer
- flatten it
- connects to fully connected layers

#### SIMILARITIES
- nodes of ANN and filters of CNN
>- in nodes we have $w_1x_1+w_2x_2+...+w_nx_n + b$ and then pass it to activation function
>- in filters we perform convolution on it and have the same weighted sum, where $w_1,w_2,w_3...w_n$ are values of filters (which are trainable) and $x_1,x_2,x_3...x_n$ are values from window on image which are convoluted with filter then a bias is added and passed to activation function

#### DIFFERENCES
- trainable parameters do not depend on input in case of CNN whereas it does in case of ANN
>- for ANN more the `input_dim` (image size) more the learning parameters and hence the increased computational cost and overfitting
>- for CNN trainable parameters depend on how many filters are we using rather than the image size, as the trainable parameters are only the filter values and their biases
>- this technique in CNN reduces computational cost and overfitting

#### **TIPS**:
- a CNN architecture has increasing depth across to `Conv2D` layers and decreasing neurons across the `Dense` layers
> The `Conv2D` layers captures deeper features and the `Dense` layers compress them to make the final output

### OVER FITTING CAN BE REDUCED BY DATA AUGMENTATION

### PROBLEM WITH SEQUENTIAL MODEL
- Linear topology
> one input source and one output source
- Fails when we want the classification for different features which has nothing to do with each other (age and emotion on human face dataset) on the same dataset 

For non-linear problems we use Functional API:
- offers multiple input and output sources

### OBJECT LOCALIZATION AND DETECTION 
- box around the object detected
- the output nodes are not only cat and dog but 4 boundary boxes also
> These boundary boxes can be:
> - coordinates of corner points and height and width of the box
> - coordinates of two diagonally opposite points
> - coordinates of midpoint of boundary box and height and width

- for this **sliding window** was used
> - but it required lot of computation
> - many boundaries

- **RCNN** were introduced to solve this but they were to complicated and slow
- for computer vision top left corner is the origin, x values increases rightwards and y increases downwards
- (x1,y1) is the top left point and (x2,y2) is bottom right

