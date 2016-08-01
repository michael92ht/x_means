"""
以下の論文で提案された改良x-means法の実装
クラスター数を自動決定するk-meansアルゴリズムの拡張について
http://www.rd.dnc.ac.jp/~tunenori/doc/xmeans_euc.pdf
以下のページのプログラム参考(ほぼ写経)
https://gist.github.com/yasaichi/254a060eff56a3b3b858
"""
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans

class Xmeans:
    def __init__(self, k_init = 2, **k_means_args):
        self.k_init = k_init
        self.k_means_args = k_means_args
        
    def fit(self,X):
        self.__clusters = []
        
        clusters = self.Cluster.build(X,KMeans(self.k_init,**self.k_means_args).fit(X))
        self.__recursively_split(clusters)
        
        self.labels_ = np.empty(X.shape[0],dtype = np.intp)
        for i,c in enumrate(self.__clusters)
            self.labels_[c.index] = i
            
    #再帰を行う
    def __recursively_split(self,clusters):
        for cluster in clusters:
            #3以下のクラスタをさらに分ける必要性がないため
            if cluster.size <= 3:
                self.__clusters.append(cluster)
                continue
            
            k_means = KMeans(2,**self.kmeans_args).fit(cluster.data)
            c1,c2 = self.Cluster.build(cluster.data,k_means,cluster.index)
            beta = np.linalg.norm(c1.center - c2.center) / np.sqrt(np.linalg.det(c1.cov) + np.linalg.det(c2.cov))
            #stats.norm.cdf 正規分布の下側確率
            alpha = 0.5 / stats.norm.cdf(beta)
            
            
            
    #各クラスタ毎の情報を扱うクラス
    class Cluster:
        @classmethod
        def build(cls,X,k_means,index = None):
            if index == None:
                index = np.array(range(0,X.shape[0]))
            labels = range(0,k_means.get_params()["n_clusters"])    

            return tuple(cls(X,index,k_means,label) for label in labels)
        
        def __init__(self,X,index,k_means,label):
            self.data = X[k_means.labels_ == label]
            self.index = index[k_means.labels_ == label]
            self.size = self.data.shape[0]
            self.df = self.data.shape[1] * (self.data.shape[1] + 3) / 2
            self.center = k_means.cluster_centers_[label]
            self.cov = np.cov(self.data.T)
        
        #尤度関数の数値計算
        def log_likelihood(self):
            return sum(stats.multivariate_normal.logpdf(x,self.center,self.cov) for x in self.data)
            
        def bic(self):
            return -2 * self.log_likelihood() + self.df * np.log(self.size)
            
        
"""
np.random.normal(mu, var, num)

np.repeat
np.repeat([1,2], 2)] -> ([1,1,2,2])
http://docs.scipy.org/doc/numpy/reference/generated/numpy.repeat.html

np.tile
np.tile([1,2], 2) -> ([1,2,1,2])
http://docs.scipy.org/doc/numpy/reference/generated/numpy.tile.html

np.flatten()
http://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.flatten.html

"""
x = np.array([np.random.normal(loc, 0.1, 20) for loc in np.repeat([1,2], 2)]).flatten()
y = np.array([np.random.normal(loc, 0.1, 20) for loc in np.tile([1,2], 2)]).flatten()

X = np.c_[x,y]
  
k = KMeans(3,random_state=1).fit(X)



print(X.shape[0])
print(X.shape[1])
        
    
    
    
    
    
    
    