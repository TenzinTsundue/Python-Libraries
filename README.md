# Top 20 Python Libraries for Data Science 

> [Here the link to the artical](https://www.kdnuggets.com/2018/06/top-20-python-libraries-data-science-2018.html)
> 
Python continues to take leading positions in solving data science tasks and challenges.

![image](https://activewizards.com/content/blog/Top_20_Python_libraries_for_data_science_-_2018/github-table01-by-click.jpg)

## Core Libraries & Statistics
----
### 1. Numpy

It is intended for processing large multidimensional arrays and matrics, and an extensive collection of high-level mathematical functions and implementated methods makes it possibel to perform various operations with these objects. 

```
import numpy as np
```

### 2. Scipy

Another core library for scientific computing is SciPy. It is based on NumPy and therefore extends its capabilities. SciPy main data structure is again a multidimensional array, implemented by Numpy. The package contains tools that help with solving linear algebra, probability theory, integral calculus and many more tasks.

```
import scipy
```

### 3. Pandas

Pandas is a Python library that provides high-level data structures and a vast variety of tools for analysis. The great feature of this package is the ability to translate rather complex operations with data into one or two commands. Pandas contains many built-in methods for grouping, filtering, and combining data, as well as the time-series functionality. 

```
import pandas as pd
```

### 4. StatsModels

Statsmodels is a Python module that provides many opportunities for statistical data analysis, such as statistical models estimation, performing statistical tests, etc. With its help, you can implement many machine learning methods and explore different plotting possibilities.

## Visualization
---

### 5. Matplotlib

Matplotlib is a low-level library for creating two-dimensional diagrams and graphs. With its help, you can build diverse charts, from histograms and scatterplots to non-Cartesian coordinates graphs.

![image](https://activewizards.com/content/blog/Top_20_Python_libraries_for_data_science_-_2018/255eb81a-1c7b-4649-a448-8c58b300851c.jpg)

```
import matplotlib.pyplot as plt
%matplotlib inline
```

### 6. Seaborn

Seaborn is essentially a higher-level API based on the matplotlib library. It contains more suitable default settings for processing charts. Also, there is a rich gallery of visualizations including some complex types like time series, jointplots, and violin diagrams.

![image](https://activewizards.com/content/blog/Top_20_Python_libraries_for_data_science_-_2018/1b29abe8-b437-401c-a733-d535bed9af14.jpg)

```
import seaborn as sns
```

### 7. Plotly

Plotly is a popular library that allows you to build sophisticated graphics easily. The package is adapted to work in interactive web applications. Among its remarkable visualizations are contour graphics, ternary plots, and 3D charts.

```
import plotly.graph_objects as go
```

### 8. Bokeh

The Bokeh library creates interactive and scalable visualizations in a browser using JavaScript widgets. The library provides a versatile collection of graphs, styling possibilities, interaction abilities in the form of linking plots, adding widgets, and defining callbacks, and many more useful features.

```
import bokeh
```

### 9. Pydot

Pydot is a library for generating complex oriented and non-oriented graphs. It is an interface to Graphviz, written in pure Python. With its help, it is possible to show the structure of graphs, which are very often needed when building neural networks and decision trees based algorithms.

```
import pydot
```

## Machine Learning
---

### 10. Scikit-learn

This Python module based on NumPy and SciPy is one of the best libraries for working with data. It provides algorithms for many standard **machine learning** and data mining tasks such as clustering, regression, classification, dimensionality reduction, and model selection.

```
import sklearn
from sklearn import datasets
```

### 11. XGBoost/ LightGBM/ CatBoost

Gradient boosting is one of the most popular machine learning algorithms, which lies in building an ensemble of successively refined elementary models, namely decision trees. Therefore, there are special libraries designed for fast and convenient implementation of this method. Namely, we think that XGBoost, LightGBM, and CatBoost deserve special attention. They are all competitors that solve a common problem and are used in almost the same way. These libraries provide highly optimized, scalable and fast implementations of gradient boosting, which makes them extremely popular among data scientists and Kaggle competitors.

### 12. Eli5

 It is a package for visualization and debugging machine learning models and tracking the work of an algorithm step by step. It provides support for scikit-learn, XGBoost, LightGBM, lightning, and sklearn-crfsuite libraries and performs the different tasks for each of them.

 ## Deep Learning
 ---

 ### 13. TensorFlow

TensorFlow is a popular framework for deep and machine learning, developed in Google Brain. It provides abilities to work with artificial neural networks with multiple data sets. Among the most popular TensorFlow applications are object identification, speech recognition, and more. There are also different layer-helpers on top of regular TensorFlow, such as tflearn, tf-slim, skflow, etc.

 ```
 import tensorflow as tf
 ```

### 14. PyTorch

PyTorch is a large framework that allows you to perform tensor computations with GPU acceleration, create dynamic computational graphs and automatically calculate gradients. Above this, PyTorch offers a rich API for solving applications related to neural networks.

```
import pytorch
```

### 15. Keras

Keras is a high-level library for working with neural networks, running on top of TensorFlow, Theano, and now as a result of the new releases, it is also possible to use CNTK and MxNet as the backends.

## Distributed Deep Learning
---

### 16. Disk-keras/ elephas/ spark-deep-learning

Deep learning problems are becoming crucial nowadays since more and more use cases require considerable effort and time. However, processing such an amount of data is much easier with the use of distributed computing systems like Apache Spark which again expands the possibilities for deep learning. Therefore, dist-keras, elephas, and spark-deep-learning are gaining popularity and developing rapidly, and it is very difficult to single out one of the libraries since they are all designed to solve a common task. These packages allow you to train neural networks based on the Keras library directly with the help of Apache Spark. Spark-deep-learning also provides tools to create a pipeline with Python neural networks.

## Neutural language Processing
---

### 17. NLTK

NLTK is a set of libraries, a whole platform for natural language processing. With the help of NLTK, you can process and analyze text in a variety of ways, tokenize and tag it, extract information, etc. NLTK is also used for prototyping and building research systems.

### 18. SpaCy

SpaCy is a natural language processing library with excellent examples, API documentation, and demo applications. The library is written in the Cython language which is C extension of Python. It supports almost 30 languages, provides easy deep learning integration and promises robustness and high accuracy.

### 19. Gensim

Gensim is a Python library for robust semantic analysis, topic modeling and vector-space modeling, and is built upon Numpy and Scipy. It provides an implementation of popular NLP algorithms, such as word2vec. Although gensim has its own models.wrappers.fasttext implementation, the fasttext library can also be used for efficient learning of word representations.

## Data Scraping
---

### 20. Scrapy

Scrapy is a library used to create spiders bots that scan website pages and collect structured data. In addition, Scrapy can extract data from the API. The library happens to be very handy due to its extensibility and portability.

