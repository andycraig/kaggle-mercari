import os
import pandas as pd
import numpy as np
import scipy
#import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV


def split_category_names(df):
    return (df
             .assign(category_name_0=df.category_name.str.split('/', n=2).str[0])
             .assign(category_name_1=df.category_name.str.split('/', n=2).str[1])
             .assign(category_name_2=df.category_name.str.split('/', n=2).str[2]))

# ItemSelector from http://scikit-learn.org/stable/auto_examples/hetero_feature_union.html 
class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.  i.e.

    >> len(data[key]) == n_samples

    Please note that this is the opposite convention to scikit-learn feature
    matrixes (where the first index corresponds to sample).

    ItemSelector only requires that the collection implement getitem
    (data[key]).  Examples include: a dict of lists, 2D numpy array, Pandas
    DataFrame, numpy record array, etc.

    >> data = {'a': [1, 5, 2, 5, 2, 8],
               'b': [9, 4, 1, 4, 1, 3]}
    >> ds = ItemSelector(key='a')
    >> data['a'] == ds.transform(data)

    ItemSelector is not designed to handle data grouped by sample.  (e.g. a
    list of dicts).  If your data is structured this way, consider a
    transformer along the lines of `sklearn.feature_extraction.DictVectorizer`.

    Parameters
    ----------
    key : hashable, required
        The key corresponding to the desired value in a mappable.
    """
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


class NanReplacer(BaseEstimator, TransformerMixin):
    """Pipeline-compatible transformer to replace NAs with specified value.
    (This functionality probably already exists somewhere in sklearn.)
    """
    def __init__(self, replacement = 'nothing'):
        self.replacement = replacement
        
    def fit(self, x, y = None):
        return self
    
    def transform(self, x):
        return x.fillna(self.replacement)


# Load train data
train_0 = (pd.read_table("../input/train.tsv", engine = 'c'))
         
train = (train_0
         .assign(target=np.log1p(train_0.price))
         .drop('price', axis=1) # To avoid accidentally training on it.
         .pipe(split_category_names)) 

del train_0

# Define bag-of-words transformations for some of the columns.
feature_union = FeatureUnion([
    ('category_name', Pipeline([('selector', ItemSelector('category_name')), 
                                ('nan_replacer', NanReplacer()),
                                ('count_vec', HashingVectorizer(ngram_range = (1, 2)))
                               ])),
    ('name', Pipeline([('selector', ItemSelector('name')), 
                       ('nan_replacer', NanReplacer()), 
                       ('count_vec', HashingVectorizer(ngram_range = (1, 2)))
                      ]))
])

# Transform training data and fit Ridge.
feature_union.fit(train) # Will also be used on test data.
X_Ridge = feature_union.transform(train)

estimator = Ridge(alpha = 0.6).fit(X_Ridge, train.target)

del train

# Load test data.
test = (pd.read_table("../input/test.tsv", engine = 'c'))
X_Ridge_test = feature_union.transform(test)

del feature_union

# Make predictions and save.
y_pred_log = estimator.predict(X_Ridge_test)
# Make predictions valid.
y_pred = np.expm1(y_pred_log)
y_pred[np.isnan(y_pred)] = 0
y_pred[y_pred < 0] = 0
(pd.DataFrame({'test_id':test.test_id, 'price':y_pred})
 .to_csv("submission.csv", index=False))
