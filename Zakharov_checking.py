import math
import scipy.special
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import random
import time
from mpmath import *

#解析解の検証用のファイル

###############################################################################
#パラメータを定めるための関数

#qの探索
def findingE1(L,m,Emax,eps):
    #計算に使う定数
    v = 4*math.pi*m/L; K = L*Emax*0.5/(2*(1-v**2))**0.5

    #[0,1]内の二分探索
    h = 1; l = 0; q = (h+l)/2
    Kq = ellipk(q)
    while abs(Kq - K) >= eps:
        if Kq < K:
            if l == q: #性能限界
                break
            l = q
        else:
            if h == q: #性能限界
                break
            h = q
        q = (h+l)/2; Kq = ellipk(q)
    if abs(Kq - K) < eps: #停止条件を達成した場合
        return q
    else:
        return "Failure"

#qの探索：
def findingE2(L,m,Emax,eps):
    #計算に使う定数
    v = 4*math.pi*m/L; K = L*Emax*0.5/(2*(1-v**2))**0.5

    #10乗オーダーでの線形探索
    i = 0; q2 = 10**(-i); Kq = scipy.special.ellipkm1(q2)
    while Kq < K:
        i += 1; q2 = 10**(-i); Kq = scipy.special.ellipkm1(q2)

    #上の10乗オーダーのもとで，小数点以下の値を2乗オーダーで線形探索
    j = 1
    while abs(K - Kq) >= eps:
        qnew = q2 + 2**(-j)*10**(-i)
        if qnew == q2: #性能限界
            break
        Kq2 = scipy.special.ellipkm1(qnew)
        if Kq2 >= K:
            q2 = qnew; Kq = Kq2
        else:
            j += 1

    if abs(Kq - K) < eps:
        print(1-q2)
        return 1-q2
    else:
        return "Failure"

#各パラメータの出力
def parameters(L,m,Emax,eps):
    v = 4*math.pi*m/L
    q = findingE1(L,m,Emax,eps)
    if q == "Failure":
        q = findingE2(L,m,Emax,eps)
    if q == "Failure":
        q = 1
    N_0 = 2*(2/(1-v**2))**0.5*float(ellipe(q**2))/L
    u = v/2 + 2*N_0/v - (2-q*2)*Emax**2/(v*(1-v**2))
    T = L/v; phi = v/2
    return [L,Emax,v,q,N_0,u,T,phi]

# Emax < 0.17281841279256 を目安に scipy.special.ellipk(q) が q=0 となり機能しなくなる
# Emax > 2.173403970708827 を目安に scipy.special.ellipkm1(1-q) が q=1 となり機能しなくなる
# Emax > 4/3 を目安に scipy.special.ellipk(q) が十分精度を確保できなくなる
L = 20; Emax = 2.1; m = 1; eps = 10**(-7)
Param = parameters(L,1,Emax,eps)
L,Emax,v,q,N_0,u,T,phi = Param

###############################################################################

def analytical_solutions(Param,t,K):
    L,Emax,v,q,N_0,u,T,phi = Param
    dx = L/K
    vv = (1 - v*v)**0.5; vv2 = 1 - v*v; WW = Emax/(2**0.5*vv)
    coef1 = -2**0.5*Emax**2*q**2*v/vv*3; coef2 = 2**0.5*v*Emax/vv; coef3 = v*Emax**2/vv2
    W = [WW*(k*dx-v*t) for k in range(K)]
    sn = [float(ellipfun('sn',W[k],q*q)) for k in range(len(W))]
    dn = [float(ellipfun('dn',W[k],q*q)) for k in range(len(W))]

    F = [Emax*dn[k] for k in range(len(W))]

    R = [F[k]*math.cos(phi*(k*dx-u*t)) for k in range(len(W))]
    I = [F[k]*math.sin(phi*(k*dx-u*t)) for k in range(len(W))]
    N = [-F[k]**2/vv2 + N_0 for k in range(len(W))]
    Nt0 = [float(coef1*sn[k]*dn[k]*ellipfun('cn',W[k],q*q)) for k in range(K)]
    #V = [coef2*float(ellipe(sn[k],q*q)) for k in range(len(W))]
    dV = [coef3*dn[k]**2 for k in range(len(W))]

    return R,I,N,Nt0,dV

def checking_analycal(n,m):
    Emaxs = [0.18 + (1.3-0.18)*i/m for i in range(m+1)]
    Errors = []
    L = 20; m = 1; eps = 10**(-9)
    for Emax in Emaxs:
        v, q, N_0, u = parameters(L,1,Emax,eps)
        T = L/v; phi = v/2
        Param = [L,Emax,v,q,N_0,u,T,phi]
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
def energy(R,I,N,dV,dx):
    dR = FD(R,dx); dI = FD(I,dx)
    Energy = norm(dR,dx) + norm(dI,dx) + 0.5*norm(N,dx) + 0.5*norm(dV,dx)
    for i in range(len(R)):
        Energy += N[i]*(R[i]**2 + I[i]**2)*dx
    return Energy

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
def initial_condition(Param,K,M):
    L,Emax,v,q,N_0,u,T,phi = Param
    dx = L/K; dt = T/M

    R0,I0,N0,Nt0,dV0 = analytical_solutions(Param,0,K)

    d2N0 = SCD(N0,dx)
    dR0 = CD(R0,dx); d2R0 = SCD(R0,dx)
    dI0 = CD(I0,dx); d2I0 = SCD(I0,dx)
    N1 = [N0[k] + dt*Nt0[k] + dt**2*(0.5*d2N0[k] + dR0[k]**2 + dI0[k]**2 + R0[k]*d2R0[k] + I0[k]*d2I0[k]) for k in range(K)]

    DD = -2*np.eye(K-1,k=0) + np.eye(K-1,k=1) + np.eye(K-1,k=-1)
    DDI = np.linalg.inv(DD)

    dN = [(N1[k]-N0[k])/dt for k in range(1,K)]
    V0 = dx**2 * np.dot(DDI,dN)
    V0 = [0]+[V0[i] for i in range(K-1)]

    return R0,I0,N0,N1,V0,dV0

###############################################################################
# 真の解の描画

def ploting_initial_solutions(Param,K):
    L = Param[0]
    R0,I0,N0 = analytical_solutions(Param,0,K)[:3]
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
    L,Emax,v,q,N_0,u,T,phi = Param
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
    Emaxs = [1.3]
    #Emaxs = [0.18 + (1.3-0.18)*i/m for i in range(m+1)]
    Errors = []
    L = 20; m = 1; eps = 10**(-9)
    for Emax in Emaxs:
        Param = parameters(L,1,Emax,eps)
        T = Param[-2]
        K = math.floor(L*n); M = math.floor(T*n)
        dt = T/M; dx = L/K

        DD = -2*np.eye(K-1,k=0) + np.eye(K-1,k=1) + np.eye(K-1,k=-1)
        DDI = np.linalg.inv(DD)

        time = []
        Norms = []
        Energys1 = []
        Energys2 = []

        for i in range(M+1):
            print(i,M)
            time.append(i*dt)
            R,I,N,Nt,dV = analytical_solutions(Param,i*dt,K)

            V = dx**2 * np.dot(DDI,Nt[1:])
            V = [0]+[V[i] for i in range(K-1)]

            Norms.append(norm(R,dx) + norm(I,dx))
            Energys1.append(energy(R,I,N,dV,dx))
            Energys2.append(energy_DVDM(R,I,N,V,dx))
        plt.plot(time,Norms,label="Norm,Emax="+str(Emax))
        plt.plot(time,Energys1,label="Energy(dV),Emax="+str(Emax))
        plt.plot(time,Energys2,label="Energy(V),Emax="+str(Emax))
        plt.legend()
    plt.xlabel("time")
    plt.ylabel("Invariants")
    plt.show()

def checking_norms(n):
    Emaxs = [0.18,1,2.1]
    Errors = []
    L = 20; m = 1; eps = 10**(-9)
    for Emax in Emaxs:
        Param = parameters(L,1,Emax,eps)
        T = Param[-2]
        K = math.floor(L*n); M = math.floor(T*n)
        dt = T/M; dx = L/K

        time = []
        Norm = []

        for i in range(0,M+1):
            print(i)
            time.append(i/M)
            dx = L/K
            vv = (1 - v*v)**0.5; vv2 = 1 - v*v; WW = Emax/(2**0.5*vv)
            print(Emax,WW,1-q,1-q*q)
            W = [WW*(k*dx-v*i*dt) for k in range(K)]
            dn = [Emax**2*scipy.special.ellipj(W[k],q*q)[2] for k in range(len(W))]
            Norm.append(norm(dn,dx))
        plt.plot(time,Norm,label="Emax="+str(Emax))
        plt.legend()
    plt.xlabel("time")
    plt.ylabel("Norm")
    plt.show()

def checking_norms2(Emax,n):
    ns = [10*i for i in range(1,10)]
    L = 20; eps = 10**(-9)

    L,Emax,v,q,N_0,u,T,phi = parameters(L,1,Emax,eps)
    vv = (1 - v*v)**0.5; vv2 = 1 - v*v; WW = Emax/(2**0.5*vv)
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
            dn = [Emax**2*scipy.special.ellipj(W[k],q*q)[2] for k in range(len(W))]
            Norm.append(norm(dn,dx))
        plt.plot(time,Norm,label="K="+str(K))
    plt.xlabel("time")
    plt.ylabel("Norm")
    plt.show()

def checking_energys(n):
    Emaxs = [0.18]
    Errors = []
    L = 20; m = 1; eps = 10**(-9)

    for Emax in Emaxs:
        L,Emax,v,q,N_0,u,T,phi = parameters(L,1,Emax,eps)
        T = L/v; phi = v/2
        Param = [L,Emax,v,q,N_0,u,T,phi]
        K = math.floor(L*n); M = math.floor(T*n)
        dt = T/M; dx = L/K

        DD = -2*np.eye(K-1,k=0) + np.eye(K-1,k=1) + np.eye(K-1,k=-1)
        DDI = np.linalg.inv(DD)

        time = []
        Energys = []
        print(Param)
        for i in range(M+1):
            print(i,M)
            time.append(i*dt)
            R,I,N,V = analytical_solutions_V(Param,i*dt,K,DDI)
            Energys.append(energy_DVDM(R,I,N,V,dx))
        plt.plot(time,Energys,label="Emax="+str(Emax))
        plt.legend()
    plt.xlabel("Emax")
    plt.ylabel("Energy")
    plt.show()
#ploting_initial_solutions(Param,100)
#checking_norms(1)
checking_norms2(1.3,10)
#checking_energys(10)
#checking_invariants(5)
