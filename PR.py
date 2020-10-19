import numpy as np
import os.path
from numpy import *
trainpath0 = 'F:/code/PR/train/0'
trainpath1 = 'F:/code/PR/train/1'
testpath0 = 'F:/code/PR/test/0'
testpath1 = 'F:/code/PR/test/1'

def getchar(str):
    c_buf = []
    for word in str:
        c_buf.append(int(word))
    return c_buf

def dataread(path):  #读取文件夹内所有txt数据
    filenames = os.listdir(path)
    dic = []
    count = 0
    for filename in filenames:
        filepath = path + '/' + filename
        buff = np.loadtxt(filepath, dtype='str')
        di = []  # 单个txt，列表
        for i in range(32):
            di.append(getchar(buff[i]))
        dic.append(np.array(di).reshape(1,1024))
        count += 1
        di.clear()  # 清空列表
    return dic

def mi(Dict):  #样本均值
    L = len(Dict)
    S = np.zeros(shape=(1,1024))
    for i in range(L):
        S += Dict[i][0]
    A = S / float(L)
    return A

def mat_Si(Dict): #离散度矩阵
    L = len(Dict)
    Si = np.zeros(shape=(1024,1024))
    for i in range(L):
        D = Dict[i]-mi(Dict) #差值
        Si += np.dot(D.T,D)
    return Si


def w_star(S,M1,M2):  #最优S=S1+S2
    Sw_1 = np.linalg.pinv(S)
    M = M1-M2
    return (M)*Sw_1

def y(Dict,W): #投影
    y = {}
    L = len(Dict)
    for i in range(L):
        y[i] = W*Dict[i].T
    return y

def dict_Avg(Dict) : #求均值
    L = len(Dict)
    S = np.zeros(shape=(1,1))
    for i in range(L):
        S += Dict[i][0]
    A = S / float(L)
    return A

def Acc(Dic,y) : #Dic为投影集，y为阈值
    L = len(Dic)
    t = 0
    for i in range(L):
        if(Dic[i][0] > y):
            t += 1
    print('percent: {:.2f}%'.format(t / L * 100))

if __name__ == "__main__":
    res = (dataread(trainpath0))
    res1 = (dataread(trainpath1))
    test0 = (dataread(trainpath0))
    test1 = (dataread(trainpath1))
    Sw = mat(mat_Si(res)+mat_Si(res1))
    w = w_star(Sw,mi(res),mi(res1))
    y_0 = ((dict_Avg(y(res,w))+dict_Avg(y(res1,w)))/2)[0][0]  #阈值
    per0 = (y(test0,w)) #测试集0投影
    per1 = (y(test1,w)) #测试集1投影
    Acc(per0,y_0)
    Acc(per1,y_0)