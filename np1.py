import numpy as np

# 配列を作成
arr = np.array([1, 2, 3])
arr  # array([1, 2, 3])

# 要素の型の情報を追加
arr = np.array([1, 2, 3], dtype=np.int32)
arr  # array([1, 2, 3], dtype=int32)

# 要素の型の情報を変更
i_arr = np.array([1, 2, 3], dtype=np.int32)
f_arr = i_arr.astype(np.float32)
f_arr  # array([ 1.,  2.,  3.], dtype=float32)
i_arr  # array([1, 2, 3], dtype=int32)，元の配列は変わらない

# 多次元配列を作成
arr = np.array([[1, 2, 3], [4, 5, 6]])
arr  # array([[1, 2, 3],[4, 5, 6]])
arr.T  # array([[1, 2], [3, 4], [5, 6]])，転地，.transopse()でも可
arr.shape  # (2, 3)，2×3行列
arr = np.array([1, 2, 3])
arr.shape  # (3,)
arr.data  # メモリの位置を示す。<memory at 0x106f54888>
arr.dtype  # データ型 dtype('int64')
arr.flat[3]  # 4

# 　特別な配列の作り方
# 要素が全部0の配列
np.zeros((2, 3))  # array([[ 0.,  0.,  0.],[ 0.,  0.,  0.]])
# 要素がすべて1の配列
np.ones((2, 3))  # array([[ 1.,  1.,  1.],[ 1.,  1.,  1.]])
X = np.diag(np.ones(3))  # [[ 1.  0.  0.], [ 0.  1.  0.], [ 0.  0.  1.]]
X = np.arange(9).reshape(3, 3)  # [[0 1 2], [3 4 5], [6 7 8]]
#ビューを返すのではなくa(=np.arange(9))のshapeを直接変更するには、a.shape = 3, 4, 5などとデータ属性に直接代入する
Y = np.zeros_like(X)  # [[0 0 0], [0 0 0], [0 0 0]]
# 要素を[0-1)の範囲でランダムに初期化する
np.random.rand(2, 3)  # array([[ 0.24025569,  0.48947483,  0.61541917],[ 0.01197138,  0.6885749 ,  0.48316059]])
# 要素を正規分布にのっとって生成する
np.random.randn(2, 3)  # array([[ 0.23397941, -1.58230063, -0.46831152],[ 1.01000451, -0.21079169,  0.80247674]])

# 配列の計算
a = np.array([[1, 2, 3], [4, 5, 6]])
3 * a  # array([[ 3,  6,  9],[12, 15, 18]])
3 + a  # array([[4, 5, 6],[7, 8, 9]])
b = np.array([[2, 3, 4], [5, 6, 7]])
a + b  # array([[ 3,  5,  7],[ 9, 11, 13]])
a * b  # array([[ 2,  6, 12],[20, 30, 42]])

v = np.array([2, 1, 3])
a + v  # array([[3, 3, 6],[6, 6, 9]])
a * v  # array([[ 2,  2,  9],[ 8,  5, 18]])

M = np.array([[1, 2, 3], [2, 3, 4]])
N = np.array([[1, 2], [3, 4], [5, 6]])
# の二つの配列の行列としての積を求めるには
M.dot(N)  # array([[22, 28],[31, 40]])

# 関数呼び出し
a = np.array([[1, 2], [3, 1]])
np.log(a)  # array([[ 0.        ,  0.69314718],[ 1.09861229,  0.        ]])
# exp、sqrtなどもある

# 統計を取る
arr = np.random.rand(100)  # 100個の乱数を生成
np.mean(arr)  # 0.52133315138159586，平均
np.max(arr)  # 0.98159897843423383
np.min(arr)  # 0.031486992721019846
np.std(arr)  # 0.2918171894076691，標準偏差
np.sum(arr)  # 52.133315138159588

arr = np.array([[1, 2, 3], [2, 3, 4]])
np.sum(arr, axis=0)  # array([3, 5, 7])
np.sum(arr, axis=1)  # array([6, 9])

# 実際に使ってみる
# 三次元空間内にある100個のベクトルの原点からのユークリッド距離の平均を求めるコード
data = np.random.randn(100, 3)
squared = data ** 2
squared_sum = np.sum(squared, axis=1)
dist = np.sqrt(squared_sum)
np.mean(dist)  # 1.5423905808984208

dist = np.linalg.norm(data, ord=2)  # とも書ける(ord = 0, ord = 1もある)

x = np.zeros(10)  # array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
np.append(x, 10)  # array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 10.])
x  # array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
x = np.append(x, 10)  # とする

x4 = np.zeros((2, 3))
np.append(x4, np.array([[1, 2, 3]]), axis=0)  # array([[0., 0., 0.], [0., 0., 0.], [1., 2., 3.]])
np.append(x4, np.array([[1], [2]]), axis=1)  # array([[0., 0., 0., 1.],[0., 0., 0., 2.]])
