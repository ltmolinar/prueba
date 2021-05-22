from sklearn.neighbors import KNeighborsClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.metrics import f1_score, make_scorer

def model():
    knn = KNeighborsClassifier(n_neighbors=50, weights='distance')

    sfs1 = SFS(knn, 
           k_features=11, 
           forward=True, 
           floating=False, 
           verbose=1,
           scoring= make_scorer(f1_score, average = 'weighted'),
           cv=5)
    return sfs1, knn

