import numpy.random as npr
import numpy as np
print(npr.rand())
print(*(100,2))
#print(npr.randn(*(100,2)))#*samp_traj.shape
orig_trajs = [[[1,3],[34,2]],[[23,4],[24,546]]]
orig_trajs = np.stack(orig_trajs, axis=0)
print(orig_trajs)

a_1d = np.arange(3)
print(a_1d)
# [0 1 2]

a_2d = np.arange(12).reshape((3, 4))
print(a_2d)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

a_3d = np.arange(24).reshape((2, 3, 4))
print(a_3d)
# [[[ 0  1  2  3]
#   [ 4  5  6  7]
#   [ 8  9 10 11]]
#  [[12 13 14 15]
#   [16 17 18 19]
#   [20 21 22 23]]]

print(a_1d.size)
# 3

print(type(a_1d.size))
# <class 'int'>

print(a_2d.size)
# 12
print(a_2d.shape)
# (3, 4)
print(a_3d.size)
# 24