import pandas as pd
import numpy as np
from sklearn.feature_extraction import text
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
seed = 1024
np.random.seed(seed)
path = '/Users/jerome/Documents/kaggle/quora-question-pairs/data/'

train = pd.read_csv(path+"train.csv")
test = pd.read_csv(path+"test.csv")
test['is_duplicate'] = [-1] * test.shape[0]

def stem_str(x,stemmer=SnowballStemmer('english')):
    x = text.re.sub("[^a-zA-Z0-9]"," ", x)
    x = (" ").join([stemmer.stem(z) for z in x.split(" ")])
    x = " ".join(x.split())
    return x

porter = PorterStemmer()
snowball = SnowballStemmer('english')


print('Generate porter')
train['question1_porter'] = train['question1'].astype(str).apply(lambda x:stem_str(x.lower(),porter))
test['question1_porter'] = test['question1'].astype(str).apply(lambda x:stem_str(x.lower(),porter))
print("porter question 1 done")

train['question2_porter'] = train['question2'].astype(str).apply(lambda x:stem_str(x.lower(),porter))
test['question2_porter'] = test['question2'].astype(str).apply(lambda x:stem_str(x.lower(),porter))
print("porter question 2 done")

train.to_csv(path+'train_porter.csv')
test.to_csv(path+'test_porter.csv')

