import math
import scipy.special
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import random
import time
from mpmath import *

###############################################################################
#パラメータを定めるための関数

#Eminの探索：Kが普通に計算できる場合
def findingE1(L,m,Emax,eps):
    #計算に使う定数
    v = 4*math.pi*m/L; K = L*Emax*0.5/(2*(1-v**2))**0.5

    #[0,Emax]内の二分探索
    h = Emax; l = 0; Emin = (h+l)/2
    q = (Emax**2 - Emin**2)**0.5/Emax; Kq = scipy.special.ellipk(q)
    while abs(Kq - K) >= eps:
        if Kq < K:
            if h == Emin: break  #性能限界
            h = Emin
        else:
            if l == Emin: break #性能限界
            l = Emin
        Emin = (h+l)/2
        q = (Emax**2 - Emin**2)**0.5/Emax; Kq = scipy.special.ellipk(q)
    if abs(Kq - K) < eps: #停止条件を達成した場合
        return Emin
    else:
        return "Failure"

#Eminの探索：Emin<<Emaxの場合
def findingE2(L,m,Emax,eps):
    #計算に使う定数
    v = 4*math.pi*m/L; K = L*Emax*0.5/(2*(1-v**2))**0.5

    #10乗オーダーでの線形探索
    i = 0; Emin = 10**(-i); Kq = scipy.special.ellipkm1((Emin)**2/(Emax**2*2))
    while Kq < K:
        i += 1; Emin = 10**(-i); Kq = scipy.special.ellipkm1((Emin)**2/(Emax**2*2))

    #上の10乗オーダーのもとで，小数点以下の値を2乗オーダーで線形探索
    j = 1
    while abs(K - Kq) >= eps:
        Enew = Emin + 2**(-j)*10**(-i)
        if Emin == Enew: break #性能限界
        Kq2 = scipy.special.ellipkm1((Enew)**2/(Emax**2*2))
        if Kq2 >= K:
            Emin = Enew; Kq = Kq2
        else:
            j += 1

    if abs(Kq2 - K) < eps:
        return Emin
    else:
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

# Emax < 0.17281841278823945 を目安に Emin > Emax の事故が起こる
# Emax > 2.173403970708827 を目安に scipy.special.ellipk が機能しなくなる
# Emax > 4/3 を目安に scipy.special.ellipj が厳密ではなくなる
L = 20; Emax = 1; m = 1; eps = 10**(-9)
v, Emin, q, N_0, u = parameters(L,1,Emax,eps)
T = L/v; phi = v/2
Param = [L,Emax,v,Emin,q,N_0,u,T,phi]

###############################################################################

def analytical_solutions(Param,t,K):
    L,Emax,v,Emin,q,N_0,u,T,phi = Param
    dx = L/K
    vv = (1 - v*v)**0.5; vv2 = 1 - v*v; WW = Emax/(2**0.5*vv); qq = q**2
    #print(Emin,q,N_0,WW,qq)
    W = [WW*(k*dx-v*t) for k in range(K)]
    dn = [scipy.special.ellipj(W[k],qq)[2] for k in range(len(W))]
    F = [Emax*dn[k] for k in range(len(W))]

    R = [F[k]*math.cos(phi*(k*dx-u*t)) for k in range(len(W))]
    I = [F[k]*math.sin(phi*(k*dx-u*t)) for k in range(len(W))]
    N = [-F[k]**2/vv2 + N_0 for k in range(len(W))]

    return R,I,N

def checking_analycal(n,m):
    Emaxs = [0.18 + (1.3-0.18)*i/m for i in range(m+1)]
    Errors = []
    L = 20; m = 1; eps = 10**(-9)
    for Emax in Emaxs:
        v, Emin, q, N_0, u = parameters(L,1,Emax,eps)
        T = L/v; phi = v/2
        Param = [L,Emax,v,Emin,q,N_0,u,T,phi]
        K = math.floor(L*n); M = math.floor(T*n)
        dt = T/M; dx = L/K
        analytical_solutions(Param,T/2,K)

#checking_analycal(1,10)

###############################################################################
#初期条件

#差分
def FD(v,dx):
    K = len(v)
    return [(v[(k+1)%K] - v[k])/dx for k in range(K)]
def CD(v,dx):
    K = len(v)
    return [(v[(k+1)%K] - v[(k-1)%K])/(2*dx) for k in range(K)]
def SCD(v,dx):
    K = len(v)
    return [(v[(k+1)%K] -2*v[k] + v[(k-1)%K])/dx**2 for k in range(K)]

#L2ノルム
def norm(v,dx):
    Ltwo = 0
    for i in range(len(v)):
        Ltwo += v[i]**2*dx
    return Ltwo

#エネルギー
def energy_DVDM(R,I,N,V,dx):
    dR = FD(R,dx); dI = FD(I,dx); dV = FD(V,dx)
    Energy = norm(dR,dx) + norm(dI,dx) + 0.5*norm(N,dx) + 0.5*norm(dV,dx)
    for i in range(len(R)):
        Energy += N[i]*(R[i]**2 + I[i]**2)*dx
    return Energy

def energy_Glassey(R,I,N1,N2,DDI,dt,dx):
    K = len(R)
    dN = [(N2[k] - N1[k])/dt for k in range(1,K)]
    V = dx**2 * np.dot(DDI,dN)
    V = [0]+[V[i] for i in range(K-1)]
    dR = FD(R,dx); dI = FD(I,dx);dV = FD(V,dx)
    Energy = norm(dR,dx) + norm(dI,dx) + 0.25*norm(N1,dx) + 0.25*norm(N2,dx) + 0.5*norm(dV,dx)
    for i in range(K):
        Energy += 0.5*(N1[i]+N2[i])*(R[i]**2 + I[i]**2)*dx
    return Energy

#L2ノルムによる距離
def dist(a,b,dx):
    dis = 0
    for i in range(len(a)):
        dis += (a[i]-b[i])**2*dx
    return dis**0.5

#Taylorで R,I,N の m=1 を求める
def initial_condition_common(Param,K,M):
    L,Emax,v,Emin,q,N_0,u,T,phi = Param
    dx = L/K; dt = T/M

    R0,I0,N0 = analytical_solutions(Param,0,K)

    vv = (1 - v*v)**0.5; WW = Emax*dx/(2**0.5*vv); qq = q**2; coef = -2**0.5*Emax**2*qq*v/vv*3

    ellipjs = [scipy.special.ellipj(WW*k,qq) for k in range(K)]
    Nt0 = [coef*ellipjs[k][0]*ellipjs[k][1]*ellipjs[k][2] for k in range(K)]

    d2N0 = SCD(N0,dx)
    dR0 = CD(R0,dx); d2R0 = SCD(R0,dx)
    dI0 = CD(I0,dx); d2I0 = SCD(I0,dx)
    N1 = [N0[k] + dt*Nt0[k] + dt**2*(0.5*d2N0[k] + dR0[k]**2 + dI0[k]**2 + R0[k]*d2R0[k] + I0[k]*d2I0[k]) for k in range(K)]

    return R0,I0,N0,N1,Nt0

def initial_condition_DVDM(Param,K,M):
    L,Emax,v,Emin,q,N_0,u,T,phi = Param
    dx = L/K; dt = T/M
    R0,I0,N0,N1,Nt0 = initial_condition_common(Param,K,M)

    DD = -2*np.eye(K-1,k=0) + np.eye(K-1,k=1) + np.eye(K-1,k=-1)
    DDI = np.linalg.inv(DD)

    dN = [Nt0[k] for k in range(1,K)]
    V0 = dx**2 * np.dot(DDI,dN)
    V0 = [0]+[V0[i] for i in range(K-1)]

    return R0,I0,N0,V0,N1

###############################################################################
# 真の解の描画

def ploting_initial_solutions(Param,K):
    L = Param[0]
    R0,I0,N0 = analytical_solutions(Param,0,K)
    E0 = [(R0[k]**2 + I0[k]**2)**0.5 for k in range(len(R0))]

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 2, 1); ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3); ax4 = fig.add_subplot(2, 2, 4)

    x = np.linspace(0, L, K)
    l1,l2,l3,l4 = "Real Part of E","Imaginary Part of E","|E|","N"

    ax1.plot(x, R0, label=l1); ax2.plot(x, I0, label=l2); ax3.plot(x, E0, label=l3); ax4.plot(x, N0, label=l4)
    ax1.legend(); ax2.legend(); ax3.legend(); ax4.legend()
    plt.show()

def ploting_solutions(Param,n):
    L,Emax,v,Emin,q,N_0,u,T,phi = Param
    K = math.floor(L*n); M = math.floor(T*n)
    dt = T/M; dx = L/K

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 2, 1); ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3); ax4 = fig.add_subplot(2, 2, 4)

    x = np.linspace(0, L, K)

    for j in range(4):
        i = (j+1)*M//4
        R,I,N = analytical_solutions(Param,i*dt,K)
        E = [(R[k]**2 + I[k]**2)**0.5 for k in range(len(R))]

        l1,l2,l3,l4 = "Real Part of E","Imaginary Part of E","|E|","N"

        ax1.plot(x, R, color=cm.hsv(i/M))#, label=l1)
        ax2.plot(x, I, color=cm.hsv(i/M))#, label=l2)
        ax3.plot(x, E, color=cm.hsv(i/M))#, label=l3)
        ax4.plot(x, N, color=cm.hsv(i/M))#, label=l4)
        #ax1.legend(); ax2.legend(); ax3.legend(); ax4.legend()
    plt.show()

def checking_invariants(n):
    Emaxs = [0.18]
    #Emaxs = [0.18 + (1.3-0.18)*i/m for i in range(m+1)]
    Errors = []
    L = 20; m = 1; eps = 10**(-9)
    for Emax in Emaxs:
        v, Emin, q, N_0, u = parameters(L,1,Emax,eps)
        T = L/v; phi = v/2
        Param = [L,Emax,v,Emin,q,N_0,u,T,phi]
        K = math.floor(L*n); M = math.floor(T*n)
        dt = T/M; dx = L/K

        time = []
        Norm = []

        for i in range(2*M+1):
            time.append(i*dt)
            R,I,N = analytical_solutions(Param,i*dt,K)
            Norm.append(norm(R,dx) + norm(I,dx))
        plt.plot(time,Norm,label="Emax="+str(Emax))
    plt.xlabel("Emax")
    plt.ylabel("Norm")
    plt.show()

def checking_norms(n):
    Emaxs = [0.18]
    Errors = []
    L = 20; m = 1; eps = 10**(-9)
    for Emax in Emaxs:
        v, Emin, q, N_0, u = parameters(L,1,Emax,eps)
        T = L/v; phi = v/2
        K = math.floor(L*n); M = math.floor(T*n)
        dt = T/M; dx = L/K

        time = []
        Norm = []

        for i in range(0,M+1):
            print(i)
            time.append(i/M)
            dx = L/K
            vv = (1 - v*v)**0.5; vv2 = 1 - v*v; WW = Emax/(2**0.5*vv); qq = q**2
            W = [WW*(k*dx-v*i*dt) for k in range(K)]
            dn = [Emax**2*scipy.special.ellipj(W[k],qq)[2] for k in range(len(W))]
            Norm.append(norm(dn,dx))
        plt.plot(time,Norm,label="Emax="+str(Emax))
    plt.xlabel("time")
    plt.ylabel("Norm")
    plt.show()

def checking_norms2(Emax,n):
    ns = [10*i for i in range(1,10)]
    L = 20; eps = 10**(-9)
    v, Emin, q, N_0, u = parameters(L,1,Emax,eps)
    vv = (1 - v*v)**0.5; vv2 = 1 - v*v; WW = Emax/(2**0.5*vv); qq = q**2
    T = L/v; phi = v/2
    L = 20; m = 1; eps = 10**(-9)
    M = math.floor(T*n); dt = T/M
    time = [i*dt for i in range(0,M+1)]
    for n in ns:
        K = math.floor(L*n); dx = L/K

        Norm = []
        for i in range(0,M+1):
            print(i)
            W = [WW*(k*dx-v*i*dt) for k in range(K)]
            dn = [Emax**2*scipy.special.ellipj(W[k],qq)[2] for k in range(len(W))]
            Norm.append(norm(dn,dx))
        plt.plot(time,Norm,label="K="+str(K))
    plt.xlabel("time")
    plt.ylabel("Norm")
    plt.show()

def checking_norms3(Emax):
    ns = [100*i for i in range(1,50)]
    L = 20; eps = 10**(-9)
    v, Emin, q, N_0, u = parameters(L,1,Emax,eps)
    vv = (1 - v*v)**0.5; vv2 = 1 - v*v; WW = Emax/(2**0.5*vv); qq = q**2
    T = L/v; phi = v/2
    L = 20; m = 1; eps = 10**(-9)
    print(L,v*T/10,1-qq)
    for n in ns:
        K = math.floor(L*n); dx = L/K
        W = [WW*(k*dx) for k in range(K)]
        dn = [scipy.special.ellipj(W[k],qq)[2] for k in range(len(W))]
        cn = [scipy.special.ellipj(W[k],qq)[1] for k in range(len(W))]
        print(max([abs(dn[k]-cn[k])for k in range(len(W))]))

#checking_norms(5)
#checking_norms2(1.3,20)
checking_norms3(1.3)
