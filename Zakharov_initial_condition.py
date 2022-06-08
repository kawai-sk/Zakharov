###############################################################################
#パラメータを定めるための関数

import math
import scipy.special
import numpy as np

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

###############################################################################
#初期条件

L = 20; Emax = 10; m = 1; eps = 10**(-6)
v, Emin, q, N_0, u = parameters(L,1,Emax,eps)
T = L/v; phi = v/2

#中心差分
def CD(v,K,dx):
    return [(v[(k+1)%K] - v[(k-1)%K])/(2*dx) for k in range(K)]
def SCD(v,K,dx):
    return [(v[(k+1)%K] -2*v[k] + v[(k-1)%K])/dx**2 for k in range(K)]

#L2ノルムによる距離
def dist(a,b,dx):
    dis = 0
    for i in range(len(a)):
        dis += (a[i]-b[i])**2*dx
    return dis**0.5

#Taylorで R,I,N の m=1 を求める
def initial_condition_common(K,M):
    dx = L/K; dt = T/M

    vv = (1 - v*v)**0.5
    WW = Emax*dx/(2**0.5*vv)
    W = [WW*k for k in range(K)]

    qq = q**2
    ellipjs = [scipy.special.ellipj(W[k],qq) for k in range(K)]
    sn = [ellipjs[k][0] for k in range(K)]
    cn = [ellipjs[k][1] for k in range(K)]
    dn = [ellipjs[k][2] for k in range(K)]

    F = [Emax*dn[k] for k in range(K)]

    R0 = [F[k]*math.cos(phi*k*dx) for k in range(K)]
    I0 = [F[k]*math.sin(phi*k*dx) for k in range(K)]

    vv2 = 1 - v*v
    N0 = [-F[k]**2/vv2 + N_0 for k in range(K)]

    coef = -2**0.5*Emax**2*qq*v/vv*3
    Nt0 = [coef*sn[k]*cn[k]*dn[k] for k in range(K)]

    d2N0 = SCD(N0,K,dx)
    dR0 = CD(R0,K,dx)
    d2R0 = SCD(R0,K,dx)
    dI0 = CD(I0,K,dx)
    d2I0 = SCD(I0,K,dx)
    N1 = [N0[k] + dt*Nt0[k] + dt**2*(0.5*d2N0[k] + dR0[k]**2 + dI0[k]**2 + R0[k]*d2R0[k] + I0[k]*d2I0[k]) for k in range(K)]

    return R0,I0,N0,N1,Nt0

def initial_condition1(K,M):
    R0,I0,N0,N1,Nt0 = initial_condition_common(K,M)
    dx = L/K; dt = T/M

    dN0 = CD(N0,K,dx)
    d2N0 = SCD(N0,K,dx)
    dR0 = CD(R0,K,dx)
    d2R0 = SCD(R0,K,dx)
    d4R0 = SCD(d2R0,K,dx)
    dI0 = CD(I0,K,dx)
    d2I0 = SCD(I0,K,dx)
    d4I0 = SCD(d2I0,K,dx)

    R1 = [R0[k] + dt*(-d2I0[k] + N0[k]*I0[k]) + 0.5*dt**2*(-d4R0[k] + d2N0[k]*R0[k] + 2*dN0[k]*dR0[k] + 2*N0[k]*d2R0[k] + Nt0[k]*I0[k] - N0[k]*N0[k]*R0[k]) for k in range(K)]
    I1 = [I0[k] + dt*(d2R0[k] - N0[k]*R0[k]) + 0.5*dt**2*(-d4I0[k] + d2N0[k]*I0[k] + 2*dN0[k]*dI0[k] + 2*N0[k]*d2I0[k] - Nt0[k]*R0[k] - N0[k]*N0[k]*I0[k]) for k in range(K)]
    return R0,I0,N0,R1,I1,N1

def initial_check1(K,M):
    dx = L/K; dt = T/M
    print(dt,dx)
    R1,I1,N1 = initial_condition1(K,M)[3:]

    vv = (1 - v*v)**0.5
    WW = Emax/(2**0.5*vv)
    W = [WW*(k*dx - v*dt) for k in range(K)]

    qq = q**2
    dn = [scipy.special.ellipj(W[k],qq)[2] for k in range(K)]

    F = [Emax*dn[k] for k in range(K)]

    tR1 = [F[k]*math.cos(phi*(k*dx-u*dt)) for k in range(K)]
    tI1 = [F[k]*math.sin(phi*(k*dx-u*dt)) for k in range(K)]

    vv2 = 1 - v*v
    tN1 = [-F[k]**2/vv2 + N_0 for k in range(K)]

    return dx**2 + dt**2,dist(R1,tR1,dx),dist(I1,tI1,dx),dist(N1,tN1,dx)

N = 10**3
K = math.floor(L*N)
M = math.floor(T*N**2.5)
#print(initial_check1(K,M))
# N1 は dt = dx**2 くらいだと良い感じ?
# R1,I1 の精度はかなり悪い


#TaylorでNのm=1,スキームでR,Iのm=1
def initial_condition2(K,M):
    R0,I0,N0,N1 = initial_condition_common(K,M)[:4]
    dx = L/K; dt = T/M
    Ik = np.identity(K)
    Dx = (1/dx**2)*(-2*Ik + np.eye(K,k=1) + np.eye(K,k=K-1) + np.eye(K,k=-1) + np.eye(K,k=-K+1))
    Dn = np.diag([(N0[k]+N1[k])*dt/4 for k in range(K)])
    D = 0.5*dt*Dx - Dn
    S = Ik + np.dot(D,D)
    print("S")
    Si = np.linalg.inv(S)
    print("Si")
    R1 = -np.array(R0) + 2*(np.dot(Si,np.array(R0)) - np.dot(np.dot(D,Si),np.array(I0)))
    I1 = -np.array(I0) + 2*(np.dot(np.dot(Si,D),np.array(R0)) + np.dot(Si,np.array(I0)))
    return R0,I0,N0,R1,I1,N1

def initial_check2(K,M):
    dx = L/K; dt = T/M
    print(dt,dx)
    R1,I1,N1 = initial_condition2(K,M)[3:]

    vv = (1 - v*v)**0.5
    WW = Emax/(2**0.5*vv)
    W = [WW*(k*dx - v*dt) for k in range(K)]

    qq = q**2
    dn = [scipy.special.ellipj(W[k],qq)[2] for k in range(K)]

    F = [Emax*dn[k] for k in range(K)]

    tR1 = [F[k]*math.cos(phi*(k*dx-u*dt)) for k in range(K)]
    tI1 = [F[k]*math.sin(phi*(k*dx-u*dt)) for k in range(K)]

    vv2 = 1 - v*v
    tN1 = [-F[k]**2/vv2 + N_0 for k in range(K)]

    return dx**2 + dt**2,dist(R1,tR1,dx),dist(I1,tI1,dx),dist(N1,tN1,dx)

N = 10**2*3
K = math.floor(L*N)
M = math.floor(T*N**3)
#print(initial_check1(K,M))
#print(initial_check2(K,M))
#print(SCD_mat(K,1/N))

#initial_check1に比べると少し精度が良いが,計算時間はかなり重い
#ただしスキームの中身でも同様の計算コストが各mでかかると考えると誤差程度
