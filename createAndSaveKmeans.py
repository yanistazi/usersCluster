from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import pandas as pd
ds = pd.read_csv("/Users/yanis/Downloads/eng-prods-swg (1).csv")

ds=ds.dropna(axis=0, how='any')
ds=ds.reset_index(drop=True,inplace=False)

ds['merge'] = ds.pro_loc_libelle.str.cat(ds.pro_loc_sousTitre, sep=' ')

document=ds["merge"]

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(document)
 

model = KMeans(n_clusters=4, init='k-means++', max_iter=100, n_init=1)
model.fit(X)
import pickle
filename = 'finalized_model.sav'
with open('finalized_model.sav','wb') as f:
    pickle.dump(model, f)
#pickle.dump(model, open(filename, 'wb'))
'''
pickle_in=open(filename ,'rb')
s=pickle.load(pickle_in) 
# some time later...

# load the model from disk
#s = pickle.load(open(filename, 'rb'))


print("Top terms per cluster:")
order_centroids = s.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(3):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),
    print
 
 
print("\n")
print("Prediction")

Y = vectorizer.transform(["iPhone 8 new technology"])
prediction = s.predict(Y)
print(prediction)

Y = vectorizer.transform(["Tefal kitchen home"])
prediction = s.predict(Y)
print(prediction)

Y = vectorizer.transform(["Canon photo "])
prediction = s.predict(Y)
print(prediction)

Y = vectorizer.transform(["Satellite gps elegance"])
prediction = s.predict(Y)
print(prediction)

Y = vectorizer.transform(["Vacuum Robot"])
prediction = s.predict(Y)
print(prediction)

Y = vectorizer.transform(["Mario video game"])
prediction = s.predict(Y)
print(prediction)
'''