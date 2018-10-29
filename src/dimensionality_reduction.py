from sklearn.decomposition import TruncatedSVD

class DimensionalityReduction():

    def __init__(self, method_name):
        self.method_name = method_name


    def reduce_to_three_dimension(self, X, n_component):
        if self.method_name == "svd":
            X_projected = self.svd_dimentionality_reduction(X, n_component=3)
            X, Y, Z = X_projected[:, 0], X_projected[:, 1], X_projected[:, 2]
            return X, Y, Z


    def reduce_to_two_dimension(self, X):
        if self.method_name == "svd":
            X_projected = self.svd_dimentionality_reduction(X, n_component=2)
            return X_projected



    def svd_dimentionality_reduction(self, X, n_component):
        svd = TruncatedSVD(n_components=n_component, n_iter=20, random_state=42)
        svd.fit(X)
        #print(svd.singular_values_)
        print(svd.explained_variance_ratio_)
        print(svd.explained_variance_ratio_.cumsum())
        X_projected = svd.fit_transform(X)
        #print(np.shape(X_projected))
        #print(X_projected[1,:])

        return X_projected