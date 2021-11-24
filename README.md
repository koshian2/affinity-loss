# Affinity-loss
Unofficial implementation of "Max-margin Class Imbalanced Learning with Gaussian Affinity" by TensorFlow, Keras.

Munawar Hayat, Salman Khan, Waqas Zamir, Jianbing Shen, Ling Shao. Max-margin Class Imbalanced Learning with Gaussian Affinity. 2019. [https://arxiv.org/abs/1901.07711](https://arxiv.org/abs/1901.07711)

![](https://github.com/koshian2/affinity-loss/blob/master/images/affinity_loss.png)

# How to use
Use "Clustering Affinity" Layer:

```python
from affinity_loss import *
x = ClusteringAffinity(10, 1, 10.0)(some_input) # n_classes, n_centroids, sigma
```

Be sure that **the output dimension is one more than the number of classes**. This is to pass the diversity regularizer to the loss function. Use "affinity_loss" loss function on compiling.

```python
model.compile("adam", affinity_loss(0.75), [acc]) # lambda
```


# Reimplementation
MNIST, lambda=0.75, sigma=10. Evaluate on macro f1-score.

| # samples per class on test data | Softmax | Affinity m=1 | Affinity m=5 |
|:--------------------------------:|:-------:|:------------:|:------------:|
|                500               |  99.28% |    99.39%    |    99.33%    |
|                200               |  99.03% |    99.20%    |    99.12%    |
|                100               |  98.79% |    98.97%    |    98.75%    |
|                50                |  98.20% |    98.54%    |    98.65%    |
|                20                |  98.56% |    98.36%    |    98.85%    |
|                10                |  97.83% |    98.27%    |    98.85%    |

![](https://github.com/koshian2/affinity-loss/blob/master/images/affinity_09.png)

# More details(Japanese)
* [https://qiita.com/koshian2/items/20af1548125c5c32dda9](https://qiita.com/koshian2/items/20af1548125c5c32dda9)
* [https://blog.shikoan.com/affinity-loss-cifar/](https://blog.shikoan.com/affinity-loss-cifar/)
