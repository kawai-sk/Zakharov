###############################################################################
#パラメータを定めるための関数

import math
import scipy.special

#Lが式を満たすかどうかの確認用
def check(L,m,Emax,Emin):
    v = 4*math.pi*m/L
    q = (Emax**2 - Emin**2)**0.5/Emax #通常の場合
    q2 = (Emin)**2/(Emax**2*2)        #Emin<<Emaxの場合
    coef = 2*((2*(1-v**2))**0.5)
    return L,coef*scipy.special.ellipk(q)/Emax,coef*scipy.special.ellipkm1(q2)/Emax

#Eminの探索：Kが普通に計算できる場合
def findingE1(L,m,Emax,eps):
    #計算に使う定数
    v = 4*math.pi*m/L
    K = L*Emax*0.5/(2*(1-v**2))**0.5
    #print("Emax="+str(Emax),"L="+str(L),"v="+str(v),"K="+str(K)) 確認(必要なら)

    #[0,Emax]内の二分探索
    h = Emax
    l = 0
    Emin = (h+l)/2
    q = (Emax**2 - Emin**2)**0.5/Emax
    Kq = scipy.special.ellipk(q)
    #print(q,Kq)
    while abs(Kq - K) >= eps:
        if Kq < K:
            if h == Emin: #性能限界
                break
            h = Emin
        else:
            if l == Emin: #性能限界
                break
            l = Emin
        Emin = (h+l)/2
        q = (Emax**2 - Emin**2)**0.5/Emax
        Kq = scipy.special.ellipk(q)
        #print(Emax,Emin,l,h,,Kq,K)
    if abs(Kq - K) < eps: #停止条件を達成した場合
        #print(check(L,m,Emax,Emin))
        return Emin
    else:
        #print(check(L,m,Emax,Emin))
        return "Failure"

#Eminの探索：Emin<<Emaxの場合
def findingE2(L,m,Emax,eps):
    #計算に使う定数
    v = 4*math.pi*m/L
    K = L*Emax*0.5/(2*(1-v**2))**0.5
    #print("Emax="+str(Emax),"L="+str(L),"v="+str(v),"K="+str(K))

    #10乗オーダーでの線形探索
    i = 0
    Emin = 10**(-i)
    Kq = scipy.special.ellipkm1((Emin)**2/(Emax**2*2))
    #print(K,Kq)
    while Kq < K:
        i += 1
        Emin = 10**(-i)
        Kq = scipy.special.ellipkm1((Emin)**2/(Emax**2*2))
    #print(K,Kq,Emin)
    #Emin = 2**0.5*Emax*10**(-i-now/2+now*10**(-j))

    #上の10乗オーダーのもとで，小数点以下の値を2乗オーダーで線形探索
    j = 1
    while abs(K - Kq) >= eps:
        Enew = Emin + 2**(-j)*10**(-i)
        if Emin == Enew: #性能限界
            break
        Kq2 = scipy.special.ellipkm1((Enew)**2/(Emax**2*2))
        #print(j,Kq2,K,Emin,Enew)
        if Kq2 >= K:
            Emin = Enew
            Kq = Kq2
        else:
            j += 1

    if abs(Kq2 - K) < eps:
        #print(check(L,m,Emax,Emin))
        return Emin
    else:
        #print(check(L,m,Emax,Emin))
        return "Failure"

#各パラメータの出力
def parameters(L,m,Emax,eps):
    v = 4*math.pi*m/L
    Emin = findingE1(L,m,Emax,eps)
    if Emin == "Failure":
        Emin = findingE2(L,m,Emax,eps)
    if Emin == "Failure":
        Emin = 0
    q = (Emax**2 - Emin**2)**0.5/Emax
    N_0 = 2*(2/(1-v**2))**0.5*scipy.special.ellipe(q**2)/L
    u = v/2 + 2*N_0/v - (Emax**2 + Emin**2)/(v*(1-v**2))
    return v,Emin,q,N_0,u

def parameters_ex(L,m,Emax,Emin):
    v = 4*math.pi*m/L
    q = (Emax**2 - Emin**2)**0.5/Emax
    N_0 = 2*(2/(1-v**2))**0.5*scipy.special.ellipe(q**2)/L
    u = v/2 + 2*N_0/v - (Emax**2 + Emin**2)/(v*(1-v**2))
    return v,Emin,q,N_0,u


###############################################################################
#Eminの妥当性についての検証


# L=20,Emax=1

#print(findingE1(20,1,1,10**(-10))) #数値計算1
# -> 0.0006384803266428207
#print(findingE2(20,1,1,10**(-15))) #数値計算2
# -> 0.0006384803592818985

#print(check(20,1,1,0.0006384803266428207)) #数値計算1の確認
#print(check(20,1,1,0.0006384803592818985)) #数値計算2の確認
#print(check(20,1,1,4.5147*10**(-6))) #先行研究の確認
#print(parameters(20,1,1,10**(-10)))
#print(parameters_ex(20,1,1,4.5147*10**(-6))) #実際のパラメータの差はほとんどなし


# L=20,Emax=10

#print(findingE1(20,1,10,10**(-5))) #数値計算1
# -> Failure
#print(findingE2(20,1,10,10**(-30))) #数値計算2
# -> 1.8980110180891075e-38
#print(check(20,1,10,1.8980110180891075e-38)) #数値計算2の確認
#print(check(20,1,10,1.3421*10**(-38))) #先行研究の確認
#print(parameters(20,1,10,10**(-10)))
#print(parameters_ex(20,1,10,1.3421*10**(-38))) #実際のパラメータの差はなし

# L=160,Emax=1

#print(findingE1(160,1,1,10**(-5))) #数値計算1
# -> Failure
#print(findingE2(160,1,1,10**(-9))) #数値計算2
# -> 1.2854190059006215e-24
#print(check(160,1,1,1.2854190059006215e-24)) #数値計算2の確認
#print(check(160,1,1,1.0535*10**(-31))) #先行研究の確認
#print(parameters(160,1,1,10**(-10)))
#print(parameters_ex(160,1,1,1.0535*10**(-31))) #実際のパラメータの差はなし
