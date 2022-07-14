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
def finding(L,m,Emax,eps):
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
    return q

#各パラメータの出力
def parameters(L,m,Emax,eps):
    v = 4*math.pi*m/L
    q = finding(L,m,Emax,eps)
    N_0 = 2*(2/(1-v**2))**0.5*v**2*Emax*float(ellipe(q))/L
    u = v/2 + 2*N_0/v - (2-q**2)*Emax**2/(v*(1-v**2))
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
    vv = (1 - v*v)**0.5; vv2 = 1 - v*v; WW = Emax/(2**0.5*vv); Kq = ellipk(q)
    coef1 = -2**0.5*Emax**3*q**2*v/vv*3; coef2 = 2**0.5*v*Emax/vv; coef3 = v*Emax**2/vv2
    W = [WW*(k*dx-v*t) for k in range(K+1)]
    dn = [float(ellipfun('dn',W[k],q)) for k in range(K)]
    F = [Emax*dn[k] for k in range(K)]

    R = [F[k]*math.cos(phi*(k*dx-u*t)) for k in range(K)]
    I = [F[k]*math.sin(phi*(k*dx-u*t)) for k in range(K)]
    N = [-F[k]**2/vv2 + N_0 for k in range(K)]
    Nt = [float(coef1*dn[k]*float(ellipfun('sn',W[k],q))*ellipfun('cn',W[k],q)) for k in range(K)]

    snV = [float(asin(ellipfun('sn',W[k],q))) if -Kq < W[k] <= Kq
     else math.pi - float(asin(ellipfun('sn',2*Kq - W[k],q))) if W[k] > Kq
      else float(asin(ellipfun('sn',W[k] - 2*Kq,q))) for k in range(K)]

    V = [coef2*float(ellipe(snV[k],q)) - N_0*(k*dx-v*t)/v for k in range(K)]
    dV = [coef3*dn[k]**2 - N_0/v for k in range(K)]

    return R,I,N,Nt,dV,V

def FD(v,dx):
    K = len(v)
    return [(v[(k+1)%K] - v[k])/dx for k in range(K)]

def checking_analycal(n):
    Emaxs = [0.18]
    L = 20
    K = math.floor(L*n); dx = L/K
    L = 20; m = 1; eps = 10**(-9)
    x = np.linspace(0, L, K)
    for Emax in Emaxs:
        Param = parameters(20,1,Emax,eps)
        dV,V = analytical_solutions(Param,0,K)[-2:]
        dV2 = FD(V,dx)
        plt.plot(x,V,label="V,Emax="+str(Emax))
        plt.plot(x,dV,label="dV1,Emax="+str(Emax))
        plt.plot(x,dV2,label="dV2,Emax="+str(Emax))
        plt.legend()
    plt.xlabel("x")
    plt.ylabel("dV")
    plt.show()
checking_analycal(100)

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
    Emaxs = [1]
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
            R,I,N,Nt,dV,V = analytical_solutions(Param,i*dt,K)


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
        L,Emax,v,q,N_0,u,T,phi = parameters(L,1,Emax,eps)
        K = math.floor(L*n); M = math.floor(T*n)
        dt = T/M; dx = L/K

        time = []
        Norm = []

        for i in range(0,M+1):
            print(i)
            time.append(i/M)
            dx = L/K
            vv = (1 - v*v)**0.5; vv2 = 1 - v*v; WW = Emax/(2**0.5*vv)
            W = [WW*(k*dx-v*i*dt) for k in range(K)]
            dn = [Emax**2*scipy.special.ellipj(W[k],q)[2] for k in range(len(W))]
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
            dn = [Emax**2*scipy.special.ellipj(W[k],q)[2] for k in range(len(W))]
            Norm.append(norm(dn,dx))
        plt.plot(time,Norm,label="K="+str(K))
    plt.xlabel("time")
    plt.ylabel("Norm")
    plt.show()

def checking_norms3(Emax,n):
    L = 20; eps = 10**(-9)

    L,Emax,v,q,N_0,u,T,phi = parameters(L,1,Emax,eps)
    vv = (1 - v*v)**0.5; vv2 = 1 - v*v; WW = Emax/(2**0.5*vv)
    T = L/v; phi = v/2
    L = 20; m = 1; eps = 10**(-9)
    M = math.floor(T*n); dt = T/M
    time = [i*dt for i in range(0,M+1)]
    K = math.floor(L*n); dx = L/K
    Norm1 = []; Norm2 = []; Norm3 = []
    for i in range(0,M+1):
        print(i,M)
        W = [WW*(k*dx-v*i*dt) for k in range(K)]
        dn = [Emax**2*float(ellipfun('dn',W[k],q)) for k in range(len(W))]
        n1 = 0; n2 = 0
        for k in range(K):
            if W[k] >= 0:
                n1 += dn[k]**2*dx
            else:
                n2 += dn[k]**2*dx
        Norm1.append(n1); Norm2.append(n2); Norm3.append(n1+n2)
    plt.plot(time,Norm1,label="kΔx-mvΔt:positive")
    plt.plot(time,Norm2,label="kΔx-mvΔt:negative")
    plt.plot(time,Norm3,label="full")
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("Norm")
    plt.show()

def checking_norms4(Emax,n,times):
    L = 20; eps = 10**(-9)

    L,Emax,v,q,N_0,u,T,phi = parameters(L,1,Emax,eps)
    vv = (1 - v*v)**0.5; vv2 = 1 - v*v; WW = Emax/(2**0.5*vv)
    T = L/v; phi = v/2
    L = 20; m = 1; eps = 10**(-9)
    M = math.floor(T*n); dt = T/M
    K = math.floor(L*n); dx = L/K
    time = [i*dt*M/times for i in range(times+1)]
    x = np.linspace(0, L, K)

    for i in range(times+1):
        t = time[i]
        vv = (1 - v*v)**0.5; vv2 = 1 - v*v; WW = Emax/(2**0.5*vv); coef2 = 2**0.5*v*Emax/vv
        W = [WW*(k*dx-v*t) for k in range(K)]
        dn = [float(ellipfun('dn',W[k],q)) for k in range(K)]
        snV = [float(asin(ellipfun('sn',W[k],q))) if -Kq < W[k] <= Kq
         else math.pi - float(asin(ellipfun('sn',2*Kq - W[k],q))) if W[k] > Kq
          else float(asin(ellipfun('sn',W[k] - 2*Kq,q))) for k in range(K)]

        V = [coef2*float(ellipe(snV[k],q)) - N_0*(k*dx-v*t)/v for k in range(K)]

        #dV = FD(V,dx)
        #plt.plot(x,dn,label="dn,t="+str(t))
        #plt.plot(x,sn,label="sn,t="+str(t))
        plt.plot(x,snV,label="snV,t="+str(t))
        #plt.plot(x,V,label="V,t="+str(t))
        plt.legend()
    plt.xlabel("time")
    plt.ylabel("dn")
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
#checking_norms3(1,20)
#checking_norms4(2.1,20,2)
#checking_energys(10)
#checking_invariants(5)
