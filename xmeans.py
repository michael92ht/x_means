
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

class XMeans:
    def __init__(self, k_init = 2, **k_means_args):
        self.k_init = k_init
        self.k_means_args = k_means_args
        
    def fit(self,X):
        self.__clusters = []
        #再帰処理、クラスタの割り当てを行う
        clusters = self.Cluster.build(X,KMeans(self.k_init,**self.k_means_args).fit(X))
        self.__recursively_split(clusters)
        
        #データにクラスタ番号の再割り当て
        self.labels_ = np.empty(X.shape[0],dtype = np.intp)
        for i,c in enumerate(self.__clusters):
            self.labels_[c.index] = i
        
        self.cluster_centers_ = np.array([c.center for c in self.__clusters])
        self.cluster_log_likelihoods_ = np.array([c.log_likelihood() for c in self.__clusters])
        self.cluster_sizes_ = np.array([c.size for c in self.__clusters])
        
        return self
        
    #再帰を行う
    def __recursively_split(self,clusters):
        for cluster in clusters:
            #3以下のクラスタをさらに分ける必要性がないため
            if cluster.size <= 3:
                self.__clusters.append(cluster)
                continue
            
            k_means = KMeans(2,**self.k_means_args).fit(cluster.data)
            c1,c2 = self.Cluster.build(cluster.data,k_means,cluster.index)
            beta = np.linalg.norm(c1.center - c2.center) / np.sqrt(np.linalg.det(c1.cov) + np.linalg.det(c2.cov))
            #stats.norm.cdf 正規分布の下側確率
            alpha = 0.5 / stats.norm.cdf(beta)
 
            bic = -2 * (cluster.size * np.log(alpha) + c1.log_likelihood() + c2.log_likelihood()) + 2 * cluster.df * np.log(cluster.size)
            
            if bic < cluster.bic():
                self.__recursively_split([c1,c2])
            else:
                self.__clusters.append(cluster)
            
            
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
        def log_likelihood(self,alpha=1):
            return sum(stats.multivariate_normal.logpdf(alpha * x,self.center,self.cov) for x in self.data)
            
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
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # データの準備
    x = np.array([np.random.normal(loc, 0.1, 20) for loc in np.repeat([1,2], 2)]).flatten()
    y = np.array([np.random.normal(loc, 0.1, 20) for loc in np.tile([1,2], 2)]).flatten()

    # クラスタリングの実行
    x_means = XMeans(random_state = 1).fit(np.c_[x,y]) 
    print(x_means.labels_)
    print(x_means.cluster_centers_)
    print(x_means.cluster_log_likelihoods_)
    print(x_means.cluster_sizes_)

    # 結果をプロット
    plt.rcParams["font.family"] = "Hiragino Kaku Gothic Pro"
    plt.scatter(x, y, c = x_means.labels_, s = 30)
    plt.scatter(x_means.cluster_centers_[:,0], x_means.cluster_centers_[:,1], c = "r", marker = "+", s = 100)
    plt.xlim(0, 3)
    plt.ylim(0, 3)
    plt.title("改良x-means法の実行結果  参考: 石岡(2000)")
    plt.show()
