'''
This Algorithm is proposed in
A Parallel Method For Earth Mover's Distance  (Wuchen Li)
In order to give a parallel method to solve OT with L2 distance.
'''


import matplotlib.pyplot as plt
import numpy as np

def shrink_2(matrix, bound):
    '''
    A function to do shrink on matrix.
    :param matrix: A  list of numpy matrix.            ( 2 ) list
    :param bound: A positive number.                   ( 1 ) number
    :return:A matrix.                                  ( 2 ) list
    '''
    val                  = np.power(np.power(matrix[0],2)+np.power(matrix[1],2),0.5)
    matrix[0][val<bound] = 0
    matrix[1][val<bound] = 0
    val[val==0]          = 1
    multiplier           = 1-(bound/val)
    matrix[0]            = np.multiply(multiplier,matrix[0])
    matrix[1]            = np.multiply(multiplier,matrix[1])


    return matrix


def primal_dual_l2(source_distribution,target_distribution,opts,initial, ctn=False):
    '''
    To find the gradient flow and optimal value.
    :param source_distribution: A matrix.                                            (n,n) matrix
    :param target_distribution: A matrix.                                            (n,n) matrix
    :param opts: includes step of primal, step of dual, max iteration, delta,lv.     ( 5 ) list
    :param initial: The initial value for M and P.                                   ( 2 ) list
    :param ctn: The char to decide whether to use continum.                          ( 1 ) char
    :return: 1) The gradient flow M.                                                 ( 2 ) list
             2) The optimal value.                                                   ( 1 ) real
    '''
    step_primal = opts[0]
    step_dual   = opts[1]
    max_iter    = opts[2]
    delta       = opts[3]
    M           = initial[0]
    P           = initial[1]

    if ctn == True:
        lv     = opts[4]
    else:
        lv     = 1

    iter        = 0

    def nabla(P,delta):
        '''
        To calculate the gradient of dual variable.
        :param P: The dual variable.                          (n,n) matrix
        :param delta:The hyperparameter.                      ( 1 ) real
        :return: 1) nabla_P_x                                 (n,n) matrix
                 2) nabla_P_y                                 (n,n) matrix
        '''

        si                = np.size(P,1)
        del_P_x           = np.zeros([si,si])
        del_P_y           = np.zeros([si,si])
        # print(np.shape(P[1:si,:]))
        # print(np.shape(del_P_x))
        del_P_x[0:si-1,:] = 1/delta*(P[1:si,:]-P[0:si-1,:])
        del_P_y[:,0:si-1] = 1/delta*(P[:,1:si]-P[:,0:si-1])
        return [del_P_x,del_P_y]



    def div(M,delta):
        '''
        To calculate the divergence of M.
        :param M: Two matrix.                 ( 2 ) list
        :param delta: The hyperparameter.     ( 1 ) real
        :return: The divergence.              (n,n) matrix
        '''

        si                = np.size(M[0],1)
        del_M_x           = M[0].copy()
        # print(np.shape(del_M_x))
        del_M_x[1:si,:]   = del_M_x[1:si,:]-del_M_x[0:si-1,:]
        del_M_y           = M[1].copy()
        del_M_y[:,1:si]   = del_M_y[:,1:si]-del_M_y[:,0:si-1]

        return 1/delta*(del_M_y+del_M_x)




    while iter < max_iter:

        na     = nabla(P,delta)
        # print(np.shape(na))
        sum    = [M[0]+step_primal*na[0],M[1]+step_primal*na[1]]
        M_plus = shrink_2(sum,step_primal)
        res    = [2*M_plus[0]-M[0],2*M_plus[1]-M[1]]
        P      = P + step_dual*(div(res,delta)+target_distribution-source_distribution)

        iter   = iter + 1
        M      = [M_plus[0].copy(),M_plus[1].copy()]

        if ctn == True:
            step_dual   = step_dual * lv
            step_primal = step_primal * lv

    optimal_value = np.sum(np.power(np.power(M[0],2)+np.power(M[1],2),0.5))

    return M, optimal_value













##Test for shrink_2
# a=np.mat([[1,3],[12,3]])
# b=np.mat([[1,4],[5,4]])
# ma=[a,b]
# print(shrink_2(ma,2))



##Test for Primal_dual_method_L2
# source_distribution = np.zeros([128,128])
# target_distribution = np.zeros([128,128])
# delta   = 4/128
# X,Y     = np.meshgrid(np.arange(-2,2, delta), np.arange(-2, 2 , delta))
# center1 = (30,30)
# center2 = (100,100)
# for i in range(128):
#     for j in range(128):
#         if (i-30)**2+(j-30)**2<100:
#             source_distribution[i,j]=1
#         if (i-100)**2+(j-100)**2<100:
#             target_distribution[i,j]=1
#
# opts = [6*10**(-6),6,30000,delta]
# M    = [np.zeros([128,128]),np.zeros([128,128])]
# P    = np.zeros([128,128])
#
# M,val = primal_dual_l2(source_distribution,target_distribution,opts,[M,P])
#
# print(val)
#
#
# plt.title('Arrows scale with plot width, not view')
# Q = plt.quiver(X,Y,M[0],M[1], units='width')
# plt.show()
