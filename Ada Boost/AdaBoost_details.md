- Forest of stumps( a root node and two leaves ). They are weak learners
- Stumps are weighted and do not enjoy equal say. Each stump is dependent on previous stump and their order is important
- Sample weight to each entry provided(equal in the starting)
- First stump is calculated on the basis of gini index

> lowest among all columns is selected

- Stump weight is calculated on the basis of how much wrong predictions it made.

> Depends on the sample weights of the entry it classified wrongly.(Total error)

> less error means more weight.

- If a stump votes for opp that what is should..its negative weight would adjust that into what is should have actually said 
- Sample weight of the entry that the stump classified wrongly is increased while that of others is decreased.

> Emphasizes that the next stump should make take prediction right as it now has more weight

- If the amount of say(weight of the stump) is large (good classifier) then the new weight of that entry is bigger than prev
- If the amount of say is small( abd classifier) then the new wieght of the wrong classified entry will be little larger than the old one
- For the correctly classified entries it is opposite. The new sample weights will be less if stump is a good classifier and will be little smaller if it is a bad classifier.
- The new weights are now normalized 
- A new dataset is build based on these sample weights or the Weighted Gini Function is used with sample weights. 

> Entries with larger weight appear more often in the new dataset

- Now new sample weights are again allotted and the process is repeated
- Now when a classification has to be made the classification with stumps having higher sum of amount of say is selected
