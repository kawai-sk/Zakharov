###############################################################################
#パラメータを定めるための関数

import math
import scipy.special
import matplotlib.pyplot as plt

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

L = 160; Emax = 1; m = 1; eps = 10**(-9)
v, Emin, q, N_0, u = parameters(L,1,Emax,eps)
T = L/v; phi = v/2

#中心差分
def CD(v,dx):
    return [(v[(k+1)%K] - v[(k-1)%K])/(2*dx) for k in range(len(v))]
def SCD(v,dx):
    return [(v[(k+1)%K] -2*v[k] + v[(k-1)%K])/dx**2 for k in range(len(v))]

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

    d2N0 = SCD(N0,dx)
    dR0 = CD(R0,dx)
    d2R0 = SCD(R0,dx)
    dI0 = CD(I0,dx)
    d2I0 = SCD(I0,dx)
    N1 = [N0[k] + dt*Nt0[k] + dt**2*(0.5*d2N0[k] + dR0[k]**2 + dI0[k]**2 + R0[k]*d2R0[k] + I0[k]*d2I0[k]) for k in range(K)]

    return R0,I0,N0,N1


###############################################################################
#スキーム本体

import numpy as np

#L2ノルム
def norm(v,dx):
    Ltwo = 0
    for i in range(len(v)):
        Ltwo += v[i]**2*dx
    return Ltwo

#エネルギー
def FD(v,dx):
    return [(v[(k+1)%K] - v[k])/dx for k in range(len(v))]
def energy(R,I,N1,N2,DDI,dt,dx):
    K = len(R)
    dN = [(N2[k] - N1[k])/dt for k in range(1,K)]
    V = dx**2 * np.dot(DDI,dN)
    V = [0]+[V[i] for i in range(K-1)]
    dR = FD(R,dx)
    dI = FD(I,dx)
    dV = FD(V,dx)
    Energy = norm(dR,dx) + norm(dI,dx) + 0.25*norm(N1,dx) + 0.25*norm(N2,dx) + 0.5*norm(dV,dx)
    for i in range(K):
        Energy += 0.5*(N1[i]+N2[i])*(R[i]**2 + I[i]**2)*dx
    return Energy

#Glasseyスキーム本体
def Glassey(K,M):
    dx = L/K; dt = T/M
    print(dt,dx)

    # 数値解の記録
    Rs = []; Is = []; Ns = []
    R_now,I_now,N_now,N_next = initial_condition_common(K,M)
    Rs.append(R_now); Is.append(I_now); Ns.append(N_now); Ns.append(N_next)

    # ここまでに数値解を計算した時刻
    ri_t = 0
    n_t = 1

    # 各mで共通して使える行列
    Ik = np.identity(K)
    Dx = (1/dx**2)*(-2*Ik + np.eye(K,k=1) + np.eye(K,k=K-1) + np.eye(K,k=-1) + np.eye(K,k=-K+1))
    ID = np.linalg.inv(Ik-0.5*dt**2*Dx)

    while ri_t < M or n_t < M:
        #print(ri_t,n_t,M)
        if ri_t < n_t: # Nm,N(m+1),Em から E(m+1)を求める
            Dn = np.diag([N_now[k]+N_next[k] for k in range(K)])
            D = dt*(0.5*Dx - 0.25*Dn)
            A = np.block([[Ik,D],[-D,Ik]])
            b = np.linalg.solve(A,2*np.array([R_now[i] if i < K else I_now[i-K] for i in range(2*K)]))
            R_next = -np.array(R_now) + b[:K]
            I_next = -np.array(I_now) + b[K:]
            Rs.append(R_next); Is.append(I_next)
            R_now = R_next; I_now = I_next
            ri_t += 1
            if ri_t%10 == 0:
                print(ri_t,n_t,M) #実行の進捗の目安として
        else:# N(m-1),Nm,Em から N(m+1)を求める
            E = np.array([R_now[k]**2 + I_now[k]**2 for k in range(K)])
            NN = np.array(N_next) + E
            N_now,N_next = N_next, 2*np.dot(ID,NN) - np.array(N_now) - 2*E
            Ns.append(N_next)
            n_t += 1

    WantToKnow = True #ノルム・エネルギーを知りたいとき
    WantToPlot = False #ノルム・エネルギーを描画したいとき
    if WantToKnow:
        DD = -2*np.eye(K-1,k=0) + np.eye(K-1,k=1) + np.eye(K-1,k=-1)
        DDI = np.linalg.inv(DD)
        Norm = [norm(Rs[i],dx) + norm(Is[i],dx) for i in range(len(Rs))]
        dNorm = [abs(Norm[i] - Norm[0]) for i in range(1,len(Rs))]
        print("初期値に対するノルムの最大誤差:",max(dNorm))
        Energy = [energy(Rs[i+1],Is[i+1],Ns[i],Ns[i+1],DDI,dt,dx) for i in range(len(Rs)-1)]
        dEnergy = [abs(Energy[i] - Energy[0]) for i in range(len(Rs)-1)]
        print("初期値に対するエネルギーの最大誤差:",max(dEnergy))
        if WantToPlot:
            Time = [i for i in range(len(Rs)-1)]
            Energy = [Energy[0]] + Energy
            plt.plot(Time,dNorm,label="Norm")
            plt.plot(Time,dEnergy,label="Energy")
            plt.xlabel("time")
            plt.ylabel("errors of Norm and Energy")
            plt.legend()
            plt.show()
    return Rs,Is,Ns

# 真値との誤差
def checking(K,M):
    dx = L/K; dt = T/M
    #print(dt,dx)
    Rs,Is,Ns = Glassey(K,M)
    dists = []

    vv = (1 - v*v)**0.5
    vv2 = 1 - v*v
    WW = Emax/(2**0.5*vv)
    qq = q**2

    #RANGE = [i for i in range(M+1)]
    RANGE = [M] # 最終時刻での誤差だけ知りたいとき
    for i in RANGE:
        W = [WW*(k*dx - v*i*dt) for k in range(K)]

        dn = [scipy.special.ellipj(W[k],qq)[2] for k in range(K)]

        F = [Emax*dn[k] for k in range(K)]

        tR = [F[k]*math.cos(phi*(k*dx-u*i*dt)) for k in range(K)]
        tI = [F[k]*math.sin(phi*(k*dx-u*i*dt)) for k in range(K)]
        tN = [-F[k]**2/vv2 + N_0 for k in range(K)]
        dists.append([dist(Rs[i],tR,dx),dist(Is[i],tI,dx),dist(Ns[i],tN,dx)])
    print("終点での各要素の誤差:",dists[-1])
    return (dx**2 + dt**2)**0.5,dists

N = 2
K = math.floor(L*N)
M = math.floor(T*N**2)

#print(Glassey(K,M))
#print(checking(K,M))
checking(K,M)
