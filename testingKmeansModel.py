from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import pandas as pd
import pickle
pickle_in=open('finalized_model.sav' ,'rb')
s=pickle.load(pickle_in) 
print("Top terms per cluster:")
order_centroids = s.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(4):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :15]:
        print(' %s' % terms[ind]),
    print
 
 
print("\n")
print("Prediction")

Y = vectorizer.transform(["iPhone 8 new technology"])
prediction = s.predict(Y)
print(prediction)

Y = vectorizer.transform([" iPod Nano "])
prediction = s.predict(Y)
print(prediction)

Y = vectorizer.transform(["Satellite gps elegance"])
prediction = s.predict(Y)
print(prediction)

Y = vectorizer.transform(["Playstation game"])
prediction = s.predict(Y)
print(prediction)

Y = vectorizer.transform(["game video play"])
prediction = s.predict(Y)
print(prediction)

Y = vectorizer.transform(["Canon photo "])
prediction = s.predict(Y)
print(prediction)

Y = vectorizer.transform(["TV screen  "])
prediction = s.predict(Y)
print(prediction)

Y = vectorizer.transform(["Lighter for home  "])
prediction = s.predict(Y)
print(prediction)



Y = vectorizer.transform(["speed wifi"])
prediction = s.predict(Y)
print(prediction)