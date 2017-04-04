import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
from sklearn.decomposition import TruncatedSVD,PCA
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer
seed = 1024
np.random.seed(seed)
path = "/Users/jerome/Documents/kaggle/quora-question-pairs/data/"

train = pd.read_csv(path+"train_porter.csv")
test = pd.read_csv(path+"test_porter.csv")

def calc_set_intersection(text_a, text_b):
    a = set(text_a.split())
    b = set(text_b.split())
    return len(a.intersection(b)) *1.0 / len(a)

print('Generate intersection')
train_interaction = train.astype(str).apply(lambda x:calc_set_intersection(x['question1'],x['question2']),axis=1)
test_interaction = test.astype(str).apply(lambda x:calc_set_intersection(x['question1'],x['question2']),axis=1)
print("interaction done for raw data")
pd.to_pickle(train_interaction,path+"train_interaction.pkl")
pd.to_pickle(test_interaction,path+"test_interaction.pkl")
print("interaction pickled done for raw data")

print('Generate porter intersection')
train_porter_interaction = train.astype(str).apply(lambda x:calc_set_intersection(x['question1_porter'],x['question2_porter']),axis=1)
test_porter_interaction = test.astype(str).apply(lambda x:calc_set_intersection(x['question1_porter'],x['question2_porter']),axis=1)
print("interaction done for porter data")

pd.to_pickle(train_porter_interaction,path+"train_porter_interaction.pkl")
pd.to_pickle(test_porter_interaction,path+"test_porter_interaction.pkl")
print("interaction pickled done for porter data")
