import pandas as pd
from sklearn.decomposition import PCA

def reduce_dimensionality(X: pd.DataFrame, no_of_compoments: int) -> pd.DataFrame:
    pca = PCA(n_components= no_of_compoments)
    result = pd.DataFrame(pca.fit_transform(X), index= X.index)
    result.columns = ['PCA_' + str(i) for i in range(1, no_of_compoments+1)]
    return result
