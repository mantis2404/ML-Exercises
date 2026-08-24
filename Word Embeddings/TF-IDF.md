# TF-IDF
TF-IDF stands for **Term Frequency-Inverse Document Frequency**.

### 1. What problem does TF-IDF solve?  
Suppose we have these documents:

-   **D1:** `The cat is sitting on the mat`
-   **D2:** `The dog is sitting on the mat`
-   **D3:** `The cat is sleeping`

If we simply count words, words like **"the"**, **"is"**, and **"sitting"** may get large counts. If we act based on the frequency of occurrence of a word we might get wrong results.

But if we're trying to understand what makes a document **distinctive**, those words aren't very useful because they occur in many documents.

TF-IDF tries to give:

> **High weight → words that are important to a particular document but uncommon across the whole collection.**

> **Low weight → words that occur in many documents.**

This is the fundamental idea behind TF-IDF. It was originally developed for information retrieval and is also used for tasks such as document classification and clustering.
### 2. TF-IDF = TF × IDF

The name literally tells us what we're doing:

$$\boxed{TF-IDF(t,d)=TF(t,d)×IDF(t)​}$$

where:

-   t = a term/word
-   d = a document
-   **TF** = how frequently the word occurs in the document
-   **IDF** = how rare the word is across documents

Together:

> **TF-IDF asks: "Is this word important to this particular document?"**

### 3. Term Frequency — TF

The simplest definition is:

$$\boxed{TF(t,d)=\frac{\text{number of times term t appears in d}}{\text{total number of terms in d​​}}}$$

### Example

Document:

> `cat cat dog fish`

Total words = 4. `cat` appears twice.
Therefore:

$$\boxed{TF(cat,d)=\frac{2}{4}​=0.5}$$

### Intuition

If a word appears repeatedly in a document, it is potentially more representative of that document.

However, there's a problem:

Suppose:

> `cat cat cat cat cat cat cat cat cat cat`

Would 10 occurrences mean the word is 10× more important than a word appearing once?

Usually not.

That's why different TF scaling schemes exist. For example, scikit-learn supports **sublinear TF scaling**, where the raw frequency tf is replaced by:

$$1+log(tf)$$

when `sublinear_tf=True`.
### 4. Inverse Document Frequency — IDF

This is the more interesting part. Suppose we have **100 documents**.
- The word `the` occurs in 95 of them.  
- The word `quantum` occurs in only 2.

Which word tells us more about a particular document?

Obviously:

> `quantum`

So IDF should give:

-   `the` → low score
-   `quantum` → high score

A common form is:

$$\boxed{IDF(t)=log(\frac{N}{DF(t)}​)​}$$

where:

-   N = total number of documents
-   DF(t) = number of documents containing term t

### Example

Suppose: N=100

`the` appears in 95 documents:

$$\boxed{IDF(the)=log(\frac{100}{95}​)}$$

Small value.

`quantum` appears in 2:

$$\boxed{IDF(quantum)=log(\frac{100}{2}​)}$$

Much larger value.

Therefore:

**$$\boxed{\text{rarer word⇒higher IDF​}}$$**

This inverse relationship is the key idea behind IDF.
### 5. Why do we use a logarithm?

This is an important detail.

Suppose:

$$\frac{N}{DF}​=1000$$

We don't necessarily want an IDF of 1000.

The logarithm compresses the scale: $log(1000)$

So extremely rare words don't completely dominate the representation.
### 6. Putting TF and IDF together
Consider:

**D1:**

> `cat cat dog`

Suppose there are 10 documents.

`cat` occurs in 2 documents.

Then:

$$TF(cat,D1)=\frac{2}{3}​$$

and approximately:

$$IDF(cat)=log(\frac{10}{2})$$

Therefore:

$$TFIDF(cat,D1)=\frac{2​}{3}log(5)$$

Now suppose `dog` occurs in 8 documents.

$$TF(dog,D1)=\frac{1}{3}​$$

but:

$$IDF(dog)=log(\frac{10}{8}​)$$

So even though both words occur in the document, **cat receives a much larger TF-IDF weight** because `cat` is much more distinctive across the corpus.

### 8. What happens to common words?

Consider:

> D1: `machine learning is interesting`  
> D2: `machine learning is powerful`  
> D3: `machine learning is useful`

`machine` occurs in every document.

Therefore its IDF is low.

But:

-   `interesting` occurs only in D1
-   `powerful` occurs only in D2
-   `useful` occurs only in D3

Therefore these words receive higher IDF.

This is exactly why TF-IDF can distinguish documents based on their more distinctive vocabulary.
Hence:

$$\boxed{IDF∝\frac{1}{DF}​​}$$

### 10. TF-IDF converts text into numbers

This is extremely important for ML.

A machine-learning model cannot directly work with:

> `"The cat is sleeping"`

We need a numerical representation.

Suppose our vocabulary is:

$$[cat,dog,sleeping,mat]$$

A document can become:

$$[0.8,0,0.5,0]$$

These numbers are its TF-IDF weights.

So TF-IDF is a form of:

$$\boxed{text→\text{numerical feature vector​}}$$

Scikit-learn's `TfidfVectorizer` converts a collection of raw documents into a TF-IDF feature matrix.
### 11. What does the resulting matrix look like?

Suppose:

```
D1 = "cat eats fish"
D2 = "dog eats meat"
D3 = "cat eats meat"
```

Vocabulary might be:

```
[cat, dog, eats, fish, meat]
```

Then TF-IDF produces something conceptually like:
|    | cat | dog | eats | fish | meat |
|---|---:|---:|---:|---:|---:|
| D1 | 0.6 | 0 | 0.4 | 0.7 | 0 |
| D2 | 0 | 0.7 | 0.4 | 0 | 0.6 |
| D3 | 0.6 | 0 | 0.4 | 0 | 0.6 |

The exact values depend on the TF, IDF and normalization formula.

Notice something important: **Every document has the same number of features.**

The columns correspond to vocabulary terms.

So if there are 10,000 vocabulary terms:

$$\boxed{\text{each document→10,000-dimensional vector​}}$$

In practice these vectors are usually represented as sparse matrices because most documents contain only a small fraction of the vocabulary. Scikit-learn's text feature extraction APIs operate with sparse representations.
### 12. TF-IDF and Bag of Words

You will probably encounter these together.

### Bag of Words

Counts words:

$$\text{text→word counts}$$

Example:

```
"cat cat dog"
```

becomes:

```
cat = 2
dog = 1
```

### TF-IDF

Takes the word frequency and reweights it according to how common the word is across documents:

BoW counts→TF-IDF weights​

Scikit-learn separates these operations into `CountVectorizer` and `TfidfTransformer`; `TfidfVectorizer` combines the two into one estimator.
### 13. Normalization

There is another step you will often see after calculating TF-IDF.

Suppose a document gets:

$$[2,4,1]$$

We can normalize it so that its vector has unit length.

With L2 normalization:

$$||x||_2​=\sqrt{2^2+4^2+1^2​}=21​$$

So:

$$x_{normalized​}=[\frac{2}{\sqrt{​21}}​,\frac{4}{\sqrt{21}}​,\frac{​1}{\sqrt{21}}​]$$

Scikit-learn's `TfidfVectorizer` uses L2 normalization by default.
#### 14. Why normalize?

Imagine:

```
D1 = "cat dog"
D2 = "cat dog cat dog cat dog cat dog"
```

Both documents have the same basic vocabulary, but D2 is much longer.

Without normalization, D2 can have much larger numerical values simply because it contains more words.

Normalization helps make the representation less dependent on document length.

With L2-normalized vectors, cosine similarity can be computed simply using their dot product.
### 15. What TF-IDF does NOT understand

This is extremely important.

Suppose:

> `The car is fast.`

and:

> `The automobile is quick.`

Humans recognize that these sentences are semantically similar.

Basic TF-IDF doesn't inherently understand that:

```
car ≈ automobile
fast ≈ quick
```

They're simply different terms/features.

$$\boxed{\text{So TF-IDF captures word importance and lexical overlap, not deep semantic meaning.}}$$

That's one reason modern NLP systems often use learned embeddings instead.
### 16. TF-IDF vs Word Embeddings

A useful mental model:
| TF-IDF | Word Embeddings |
|---|---|
| Statistical | Learned |
| Based mainly on word occurrence | Learns representations from context/data |
| Sparse | Usually dense |
| Doesn't inherently understand synonyms | Can capture semantic relationships |
| Easy to interpret | Often harder to interpret |
| Computationally simple | Generally more computationally expensive |
| Excellent baseline for many text tasks | Useful for richer semantic NLP |

TF-IDF is therefore **not obsolete**. It remains a useful feature representation for tasks such as classification, clustering and information retrieval.
### 17. TF-IDF — Uses

-   **Document similarity** → Find documents with similar vocabulary.
-   **Search / Information retrieval** → Rank documents relevant to a query.
-   **Text classification** → Convert text into features for models like SVM, Logistic Regression, etc.
-   **Document clustering** → Group similar documents.
-   **Keyword extraction** → Identify words that are distinctive to a document.
-   **Duplicate detection** → Find documents with very similar wording.
