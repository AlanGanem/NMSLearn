# NMSLearn
> NMSLib objects avalible just like sklearn's nearest_neighbors API


## Instalation

just run:

```pip install git+https://github.com/AlanGanem/NMSLearn ```

or  clone and run setup.py

Available only for python 3.8 or lower

# Available Wrappers

- [X] `NMSLibSklearnWrapper` - Base class for building classes for different metrics
- [X] `FastJaccardNN` - Class to perform Jaccard dissimilarity based nearesst neighbors search
- [X] `FastL2NN` - Class to perform L2(euclidean) distance based nearesst neighbors search
- [X] `FastKLDivNN`  - Class to perform Kullback-Leibler divergence based nearesst neighbors search

## Usage example

```python
#import L2 nearest neighbor class
from nmslearn.neighbors import FastL2NN
#numpy for data generation
import numpy as np
```

```python
#make random data to create index
X = np.random.randn(100_000, 10)
#instantiate nearest neighbor object with default parameters
l2nn = FastL2NN(verbose = True)
#fit the index
l2nn.fit(X)
```




    FastL2NN(verbose=True)



```python
#use fitted index to query new data
query_matrix = np.random.randn(3, 10)

distances, indexes = l2nn.kneighbors(query_matrix, n_neighbors = 30, n_jobs = 8, return_distance = True)

distances, indexes
```

    kNN time total=0.008003 (sec), per query=0.002668 (sec), per query adjusted for thread number=0.021342 (sec)
    




    ([array([1.3245634, 1.6304972, 2.3469138, 2.5226061, 2.571098 , 2.5826378,
             2.58591  , 2.7503452, 2.9602616, 3.051869 , 3.0761793, 3.0987031,
             3.1178   , 3.126969 , 3.1348734, 3.2131133, 3.2408729, 3.2818627,
             3.3321965, 3.339151 , 3.3681705, 3.3815045, 3.3855784, 3.4641187,
             3.5331888, 3.5802543, 3.6158307, 3.6223383, 3.666287 , 3.6673372],
            dtype=float32),
      array([0.61047035, 0.6418276 , 0.7520662 , 0.8606815 , 0.8694241 ,
             0.89354324, 0.9396638 , 1.0212313 , 1.046349  , 1.0705577 ,
             1.0905973 , 1.1210229 , 1.1351492 , 1.214601  , 1.2423488 ,
             1.2616279 , 1.2741274 , 1.4380698 , 1.4898877 , 1.6442922 ,
             1.6522729 , 1.6648743 , 1.6743813 , 1.6807532 , 1.6844735 ,
             1.7031903 , 1.7120422 , 1.7446207 , 1.7527819 , 1.7674448 ],
            dtype=float32),
      array([3.6314204, 4.0911403, 4.2446575, 4.4768524, 4.939749 , 5.056326 ,
             5.184124 , 5.2949767, 5.351741 , 5.3570704, 5.376051 , 5.4323053,
             5.6032033, 5.6485624, 5.6798406, 5.7229824, 5.727806 , 5.7553544,
             5.7996774, 5.918634 , 5.9476724, 5.958545 , 6.0005064, 6.0113473,
             6.046746 , 6.0788283, 6.115496 , 6.121717 , 6.171587 , 6.177366 ],
            dtype=float32)],
     [array([42822, 30338, 65640, 33816, 31477, 24327, 62219, 71619, 47402,
             64592, 22439, 66049, 98434, 26807, 96418, 41685, 57696, 15364,
             49249, 39804, 69842, 52465, 13791, 11394, 56749, 83391, 14460,
             19654, 79622, 22899]),
      array([16760, 74076, 36773, 98339, 74754, 23677, 24736, 29043, 22602,
             42063,  9178, 28964, 23573, 93285,  1598, 55341,  4508, 58892,
             90815, 76816, 71175, 59025, 32525, 14222, 98113, 17727, 14136,
             60727, 52566, 22507]),
      array([68364, 33045,   675, 94592, 92453, 63129,  8206, 41737, 90293,
             33514, 42227, 97250, 84322, 35358, 11519, 25277, 33946, 74534,
             60613, 50247, 16112, 66530, 42217, 43870, 35752, 72825, 16468,
             83012, 86004, 83829])])



```python
import joblib
#serialize object with joblib or any other serializer
joblib.dump(l2nn,'l2nn.sav')

#deserialze
l2nn = joblib.load('l2nn.sav')
```

```python
#appends to index with partial_fit method
l2nn.partial_fit(X)
```




    FastL2NN(verbose=True)



```python
#query again with 

distances, indexes = l2nn.kneighbors(query_matrix, n_neighbors = 30, n_jobs = 8, return_distance = True)

distances, indexes
```

    kNN time total=0.027713 (sec), per query=0.009238 (sec), per query adjusted for thread number=0.073902 (sec)
    




    ([array([1.3245634, 1.3245634, 1.6304972, 1.6304972, 2.3469138, 2.3469138,
             2.5226061, 2.5226061, 2.571098 , 2.571098 , 2.5826378, 2.5826378,
             2.58591  , 2.58591  , 2.7503452, 2.7503452, 2.9602616, 2.9602616,
             3.051869 , 3.051869 , 3.0761793, 3.0761793, 3.0987031, 3.0987031,
             3.1178   , 3.1178   , 3.126969 , 3.126969 , 3.1348734, 3.1348734],
            dtype=float32),
      array([0.61047035, 0.61047035, 0.6418276 , 0.6418276 , 0.7520662 ,
             0.7520662 , 0.8606815 , 0.8606815 , 0.8694241 , 0.8694241 ,
             0.89354324, 0.89354324, 0.9396638 , 0.9396638 , 1.0212313 ,
             1.0212313 , 1.046349  , 1.046349  , 1.0705577 , 1.0705577 ,
             1.0905973 , 1.0905973 , 1.1210229 , 1.1210229 , 1.1351492 ,
             1.1351492 , 1.214601  , 1.214601  , 1.2423488 , 1.2423488 ],
            dtype=float32),
      array([3.6314204, 3.6314204, 4.0911403, 4.0911403, 4.2446575, 4.2446575,
             4.4768524, 4.4768524, 4.939749 , 4.939749 , 5.056326 , 5.056326 ,
             5.184124 , 5.184124 , 5.2949767, 5.2949767, 5.351741 , 5.351741 ,
             5.3570704, 5.3570704, 5.376051 , 5.376051 , 5.4323053, 5.4323053,
             5.6032033, 5.6032033, 5.6485624, 5.6485624, 5.6798406, 5.6798406],
            dtype=float32)],
     [array([42822, 42822, 30338, 30338, 65640, 65640, 33816, 33816, 31477,
             31477, 24327, 24327, 62219, 62219, 71619, 71619, 47402, 47402,
             64592, 64592, 22439, 22439, 66049, 66049, 98434, 98434, 26807,
             26807, 96418, 96418]),
      array([16760, 16760, 74076, 74076, 36773, 36773, 98339, 98339, 74754,
             74754, 23677, 23677, 24736, 24736, 29043, 29043, 22602, 22602,
             42063, 42063,  9178,  9178, 28964, 28964, 23573, 23573, 93285,
             93285,  1598,  1598]),
      array([68364, 68364, 33045, 33045,   675,   675, 94592, 94592, 92453,
             92453, 63129, 63129,  8206,  8206, 41737, 41737, 90293, 90293,
             33514, 33514, 42227, 42227, 97250, 97250, 84322, 84322, 35358,
             35358, 11519, 11519])])


