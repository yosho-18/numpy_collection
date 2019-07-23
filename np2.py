import numpy as np

a = np.arange(20, 0, -2, dtype=np.int32)  # まずは1次元配列を生成
# array([20, 18, 16, 14, 12, 10,  8,  6,  4,  2])
np.where(a < 10)  # 10未満のindexを取得
# (array([6, 7, 8, 9]),)
a[np.where(a < 10)]
# array([8, 6, 4, 2]) # 10未満の要素だけのindexとなっていることが確認できます。

a = np.arange(12).reshape((3, 4))  # 3×4の二次元配列にする
# array([[ 0,  1,  2,  3],[ 4,  5,  6,  7],[ 8,  9, 10, 11]])
np.where(a % 2 == 0)  # 偶数だけ取り出してみる。
# (array([0, 0, 1, 1, 2, 2]), array([0, 2, 0, 2, 0, 2]))
# #行と列のインデックスが取り出されており、対応する(0, 0)や(0, 2)は偶数になっていることが分かります。

a = np.arange(12)
np.where(a % 2 == 0, 'even', 'odd')  # 偶数ならeven,奇数ならoddと返す。
# array(['even', 'odd', 'even', 'odd', 'even', 'odd', 'even', 'odd', 'even','odd', 'even', 'odd'],dtype='<U4')
np.where(a % 2 == 0, 'even')  # Trueの時だけ値を設定するとエラーが返ってくる。
# (エラーメッセージが表示される) ValueError: either both or neither of x and y should be given
b = np.reshape(a, (3, 4))
c = b ** 2  # array([[  0,   1,   4,   9],[ 16,  25,  36,  49],[ 64,  81, 100, 121]])
np.where(b % 2 == 0, b, c)  # 奇数のところだけcの要素に取り替える。
# array([[  0,   1,   2,   9],[  4,  25,   6,  49],[  8,  81,  10, 121]])

np.where(b % 2 == 0, b, (10, 8, 6, 4))  # broadcastingが適用され、(10, 8, 6, 4)が繰り返されたものが使われている。
# array([[ 0,  8,  2,  4],[ 4,  8,  6,  4],[ 8,  8, 10,  4]])

# linspace(start,stop,num=50,endpoint=True,retstop=False,dtype=None)
# 例えば、60HzでデータをSamplingする場合の時刻tを作りたい時などに使う。
t = np.linspace(0, 1, 60, retstep=True)  # len(t) = 60，retstep=Trueにすると、データの間隔を返してくれる。
"""(array([ 0.        ,  0.01694915,  0.03389831,  0.05084746,  0.06779661,
        0.08474576,  0.10169492,  0.11864407,  0.13559322,  0.15254237,
        0.16949153,  0.18644068,  0.20338983,  0.22033898,  0.23728814,
        0.25423729,  0.27118644,  0.28813559,  0.30508475,  0.3220339 ,
        0.33898305,  0.3559322 ,  0.37288136,  0.38983051,  0.40677966,
        0.42372881,  0.44067797,  0.45762712,  0.47457627,  0.49152542,
        0.50847458,  0.52542373,  0.54237288,  0.55932203,  0.57627119,
        0.59322034,  0.61016949,  0.62711864,  0.6440678 ,  0.66101695,
        0.6779661 ,  0.69491525,  0.71186441,  0.72881356,  0.74576271,
        0.76271186,  0.77966102,  0.79661017,  0.81355932,  0.83050847,
        0.84745763,  0.86440678,  0.88135593,  0.89830508,  0.91525424,
        0.93220339,  0.94915254,  0.96610169,  0.98305085,  1.        ]), 0.01694915254237288)#t[1] = 0.0169491525424"""
# endpointをFalseにするとstopを含まないデータが生成される。下記の例だとデータの間隔は1/60=0.01666...になる。
# endpointを含めた時の間隔は1/59=0.01694になる。
t = np.linspace(0, 1, 60, endpoint=False)  # len(t) = 60
"""[ 0.          0.01666667  0.03333333  0.05        0.06666667  0.08333333
  0.1         0.11666667  0.13333333  0.15        0.16666667  0.18333333
  0.2         0.21666667  0.23333333  0.25        0.26666667  0.28333333
  0.3         0.31666667  0.33333333  0.35        0.36666667  0.38333333
  0.4         0.41666667  0.43333333  0.45        0.46666667  0.48333333
  0.5         0.51666667  0.53333333  0.55        0.56666667  0.58333333
  0.6         0.61666667  0.63333333  0.65        0.66666667  0.68333333
  0.7         0.71666667  0.73333333  0.75        0.76666667  0.78333333
  0.8         0.81666667  0.83333333  0.85        0.86666667  0.88333333
  0.9         0.91666667  0.93333333  0.95        0.96666667  0.98333333]"""

t1 = np.logspace(2, 3, 10)
"""[  100.           129.1549665    166.81005372   215.443469     278.25594022
   359.38136638   464.15888336   599.48425032   774.26368268  1000.        ]"""
n = np.linspace(2, 3, 10)
t2 = 10 ** n
"""[  100.           129.1549665    166.81005372   215.443469     278.25594022
   359.38136638   464.15888336   599.48425032   774.26368268  1000.        ]"""
t1 = np.logspace(2, 3, 10, base=np.e)
"""[  7.3890561    8.25741109   9.22781435  10.3122585   11.52414552
  12.87845237  14.3919161   16.08324067  17.97332814  20.08553692]"""

# meshgridを作る
X = np.mgrid[0:10:2]  # [0 2 4 6 8]
XY = np.mgrid[0:10:2, 1:10:2]
"""[[[0 0 0 0 0], [2 2 2 2 2], [4 4 4 4 4], [6 6 6 6 6], [8 8 8 8 8]]
,[[1 3 5 7 9], [1 3 5 7 9], [1 3 5 7 9], [1 3 5 7 9], [1 3 5 7 9]]]"""
X, Y = np.mgrid[-2:2:0.2, -2:2:0.2]
Z = X * np.exp(-X ** 2 - Y ** 2)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet, linewidth=0.1, antialiased=False)
plt.show()

np.ogrid[0:10:2, 1:10:2]  # [array([[0],[2],[4],[6],[8]]), array([[1, 3, 5, 7, 9]])]

c = np.array([1. - 2.6j, 2.1 + 3.J, 4. - 3.2j])  # 複素数を要素とするndarrayインスタンスを生成
c.real  # 実部 array([ 1. ,  2.1,  4. ])
c.imag  # 虚部 array([-2.6,  3. , -3.2])

import numpy.random as npr
t0_idx = npr.multinomial(100, [1 / 100.] * 100 , size= 1)
t0_idx = np.argmax(t0_idx) + nsample
print(t0_idx)  # 最大値のmaxを返す
npr.randn(*samp_traj.shape) * noise_std  # 標準正規分布 (平均0, 標準偏差1)