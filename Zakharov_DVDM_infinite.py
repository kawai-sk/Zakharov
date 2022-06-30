import math
import cmath
import scipy.special
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import random
import time
import os
import pandas as pd
import csv
from mpmath import *

###############################################################################
#パラメータを定める

Ltent = 20; Emax = 2.1; m = 1; eps = 10**(-9)
v = 4*math.pi*m/Ltent
u = v/2 - Emax**2/(v*(1-v**2))
T = Ltent/v; phi = v/2
Param2 = [Ltent,T,Emax,v,u,phi]

###############################################################################

def analytical_solutions_infinite(Param2,t,K):
    Ltent,T,Emax,v,u,phi = Param2
    dx = Ltent/K
    vv = (1 - v*v)**0.5; vv2 = 1 - v*v; WW = Emax/(2**0.5*vv)
    W = np.array([WW*((k-K)*dx-v*t) for k in range(3*K)])
    F = Emax/np.cosh(W)

    R = [F[k]*math.cos(phi*((k-K)*dx-u*t)) for k in range(len(W))]
    I = [F[k]*math.sin(phi*((k-K)*dx-u*t)) for k in range(len(W))]
    N = [-F[k]**2/vv2 for k in range(len(W))]

    return R,I,N

def analytical_solutions_infiniteV(Param,t,K,DDI):
    Ltent,T,Emax,v,u,phi = Param2
    dx = Ltent/K
    vv = (1 - v*v)**0.5; vv2 = 1 - v*v; WW = Emax/(2**0.5*vv); coef = -2**0.5*Emax**3*v/vv*3
    W = np.array([WW*((k-K)*dx-v*t) for k in range(3*K)])
    F = Emax/np.cosh(W)

    R = [F[k]*math.cos(phi*((k-K)*dx-u*t)) for k in range(len(W))]
    I = [F[k]*math.sin(phi*((k-K)*dx-u*t)) for k in range(len(W))]
    N = [-F[k]**2/vv2 for k in range(len(W))]
    Nt = [coef * np.sinh(W[k])/np.cosh(W[k])**3 for k in range(1,len(W))]

    V = dx**2 * np.dot(DDI,Nt)
    V = [0]+[V[i] for i in range(K-1)]

    return R,I,N,V
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
        Ltwo += abs(v[i])**2*dx
    return Ltwo

#L2ノルムによる距離
def dist(a,b,dx):
    dis = 0
    for i in range(len(a)):
        dis += abs(a[i]-b[i])**2*dx
    return dis**0.5

def energy_DVDM(R,I,N,V,dx):
    dR = FD(R,dx); dI = FD(I,dx); dV = FD(V,dx)
    Energy = norm(dR,dx) + norm(dI,dx) + 0.5*norm(N,dx) + 0.5*norm(dV,dx)
    for i in range(len(R)):
        Energy += N[i]*(R[i]**2 + I[i]**2)*dx
    return Energy

#Taylorで R,I,N の m=1 を求める
def initial_condition_infinite_common(Param2,K,M):
    Ltent,T,Emax,v = Param2[:4]
    dx = Ltent/K; dt = T/M

    R0,I0,N0 = analytical_solutions_infinite(Param2,0,K)

    vv = (1 - v*v)**0.5; WW = Emax*dx/(2**0.5*vv); coef = -2**0.5*Emax**3*v/vv*3

    W = [WW*(k-K) for k in range(3*K)]
    Nt0 = [coef * np.sinh(W[k])/np.cosh(W[k])**3 for k in range(len(R0))]

    d2N0 = SCD(N0,dx)
    dR0 = CD(R0,dx); d2R0 = SCD(R0,dx)
    dI0 = CD(I0,dx); d2I0 = SCD(I0,dx)
    N1 = [N0[k] + dt*Nt0[k] + dt**2*(0.5*d2N0[k] + dR0[k]**2 + dI0[k]**2 + R0[k]*d2R0[k] + I0[k]*d2I0[k]) for k in range(len(R0))]
    return R0,I0,N0,N1,Nt0

def initial_condition_infinite_DVDM(Param2,K,M):
    Ltent,T = Param2[:2]
    dx = Ltent/K; dt = T/M
    R0,I0,N0,N1,Nt0 = initial_condition_infinite_common(Param2,K,M)

    DD = -2*np.eye(len(N0)-1,k=0) + np.eye(len(N0)-1,k=1) + np.eye(len(N0)-1,k=-1)
    DDI = np.linalg.inv(DD)

    dN = [Nt0[k] for k in range(1,len(N0))]
    V0 = dx**2 * np.dot(DDI,dN)
    V0 = [0]+[V0[i] for i in range(len(N0)-1)]

    return R0,I0,N0,V0,N1

###############################################################################
#解析解の描画

def ploting_initial_solutions_infinite(Param2,K):
    Ltent = Param2[0]
    R0,I0,N0 = analytical_solutions_infinite(Param2,0,K)
    E0 = [(R0[k]**2 + I0[k]**2)**0.5 for k in range(len(R0))]

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 2, 1); ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3); ax4 = fig.add_subplot(2, 2, 4)

    x = np.linspace(-Ltent, 2*Ltent, 3*K)
    l1,l2,l3,l4 = "Real Part of E","Imaginary Part of E","|E|","N"

    ax1.plot(x, R0, label=l1); ax2.plot(x, I0, label=l2); ax3.plot(x, E0, label=l3); ax4.plot(x, N0, label=l4)
    ax1.legend(); ax2.legend(); ax3.legend(); ax4.legend()
    plt.show()

def checking_invariants(n):
    Emaxs = [0.18,1,2.1]
    #Emaxs = [0.18 + (1.3-0.18)*i/m for i in range(m+1)]
    Errors = []
    Ltent = 20; m = 1; eps = 10**(-9)
    v = 4*math.pi*m/Ltent
    T = Ltent/v; phi = v/2
    K = math.floor(Ltent*n); M = math.floor(T*n)
    dt = T/M; dx = Ltent/K

    for Emax in Emaxs:
        u = v/2 - Emax**2/(v*(1-v**2))
        Param2 = [Ltent,T,Emax,v,u,phi]

        time = []
        Norms = []
        Energys = []
        DD = -2*np.eye(3*K-1,k=0) + np.eye(3*K-1,k=1) + np.eye(3*K-1,k=-1)
        DDI = np.linalg.inv(DD)

        for i in range(M+1):
            time.append(i*dt)
            R,I,N,V = analytical_solutions_infiniteV(Param2,i*dt,K,DDI)
            Norms.append(norm(R,dx) + norm(I,dx))
            Energys.append(energy_DVDM(R,I,N,V,dx))

        plt.plot(time,Norms,label="Emax="+str(Emax)+",norm")
        plt.legend()
        #plt.plot(time,Energys,label="Emax="+str(Emax)+",Energy")
        #plt.legend()
    plt.xlabel("time")
    plt.ylabel("Invariants")
    plt.show()

checking_invariants(10)
#ploting_initial_solutions_infinite(Param2,math.floor(Ltent*10))
###############################################################################
#スキーム本体

# Glassey スキーム
def Glassey_infinite(Param2,K,M):
    Ltent,T = Param2[:2]
    dx = Ltent/K; dt = T/M #print(dt,dx)
    start = time.perf_counter()

    # 数値解の記録
    Rs = []; Is = []; Ns = []
    R_now,I_now,N_now,N_next,_ = initial_condition_infinite_common(Param2,K,M)
    Rs.append(R_now); Is.append(I_now); Ns.append(N_now); Ns.append(N_next)
    K0 = len(R_now)

    # ここまでに数値解を計算した時刻
    ri_t = 0
    n_t = 1

    # 各mで共通して使える行列
    Ik = np.identity(K0)
    Dx = (1/dx**2)*(-2*Ik + np.eye(K0,k=1) + np.eye(K0,k=K0-1) + np.eye(K0,k=-1) + np.eye(K0,k=-K0+1))
    ID = np.linalg.inv(Ik-0.5*dt**2*Dx)

    while ri_t < M or n_t < M:
        #print(ri_t,n_t,M)
        if ri_t < n_t: # Nm,N(m+1),Em から E(m+1)を求める
            Dn = np.diag([N_now[k]+N_next[k] for k in range(K0)])
            D = dt*(0.5*Dx - 0.25*Dn)
            A = np.block([[Ik,D],[-D,Ik]])
            b = np.linalg.solve(A,2*np.array([R_now[i] if i < K0 else I_now[i-K0] for i in range(2*K0)]))
            R_next = -np.array(R_now) + b[:K0]
            I_next = -np.array(I_now) + b[K0:]
            Rs.append(R_next); Is.append(I_next)
            R_now = R_next; I_now = I_next
            ri_t += 1
            if ri_t%10 == 0:
                print(ri_t,n_t,M) #実行の進捗の目安として
        else: # N(m-1),Nm,Em から N(m+1)を求める
            N_before = N_now; N_now = N_next
            E = np.array([R_now[k]**2 + I_now[k]**2 for k in range(K0)])
            NN = np.array(N_now) + E
            N_next = np.dot(ID,2*NN) - np.array(N_before) - 2*E
            Ns.append(N_next)
            n_t += 1
    end = time.perf_counter()
    print("Glassey実行時間:",end-start)

    return [[str(end-start)]+[0 for i in range(len(Rs[0])-1)]],Rs,Is,Ns

def checking_Glassey_infinite(Param2,K,M):
    Ltent,T = Param2[:2]
    dx = Ltent/K; dt = T/M #print(dt,dx)
    Rs,Is,Ns = Glassey_infinite(Param,K,M)[:3]
    eEs = [];eNs = []
    rEs = [];rNs = []

    RANGE = [i for i in range(len(Rs))]
    #RANGE = [len(Rs)-1] # 最終時刻での誤差だけ知りたいとき
    for i in RANGE:
        tR,tI,tN = analytical_solutions_infinite(Param,i*dt,K)

        tnorm = (norm(tR,dx) + norm(tI,dx))**0.5

        eEs.append((dist(Rs[i],tR,dx)**2+dist(Is[i],tI,dx)**2)**0.5); eNs.append(dist(Ns[i],tN,dx))
        rEs.append(eEs[i]/tnorm); rNs.append(eNs[i]/norm(tN,dx)**2)
    print("各要素の最大誤差:",max(eEs),max(eNs))
    print("各要素の最大誤差比:",max(rEs),max(rNs))
    for i in range(4):
        j = math.floor(T*(i+1)/(4*dt))
        print("t:",32*(i+1)/4,",",eEs[j],eNs[j],",",rEs[j],rNs[j])
    return (dx**2 + dt**2)**0.5,eEs,eNs

# DVDMスキーム本体
# Newton法の初期値をGlasseyで求める
def DVDM_Glassey_infinite(Param2,K,M,eps):
    Ltent,T = Param2[:2]
    start = time.perf_counter()
    dx = Ltent/K; dt = T/M; #print(dt,dx)

    # 数値解の記録
    Rs = []; Is = []; Ns = []; Vs = []
    R_now,I_now,N_now,V_now,N_next = initial_condition_infinite_DVDM(Param2,K,M)
    Rs.append(R_now); Is.append(I_now); Ns.append(N_now); Vs.append(V_now)
    K0 = len(R_now)

    tR,tI,tN = analytical_solutions_infinite(Param2,0,K)

    m = 0

    Ik = np.identity(K0); Zk = np.zeros((K0,K0))
    Dx = (-2*Ik + np.eye(K0,k=1) + np.eye(K0,k=K0-1) + np.eye(K0,k=-1) + np.eye(K0,k=-K0+1))/dx**2
    ID = np.linalg.inv(Ik-0.5*dt**2*Dx)

    DR_now = np.dot(Dx,np.array(R_now)); DI_now = np.dot(Dx,np.array(I_now)); DV_now = np.dot(Dx,np.array(V_now))

    while m*dt < T:
        t = 0
        dN = 0.5*dt*Dx - 0.25*dt*np.diag(N_now)
        dR_now = 0.25*dt*np.diag(R_now); dI_now = 0.25*dt*np.diag(I_now)

        F0 = np.array([- R_now[i%K0] + 0.5*dt*DI_now[i%K0] if i//K0 == 0
        else -I_now[i%K0] - 0.5*dt*DR_now[i%K0] if i//K0 == 1
        else -N_now[i%K0] - 0.5*dt*DV_now[i%K0] if i//K0 == 2
        else - V_now[i%K0] - 0.5*dt*(N_now[i%K0] + R_now[i%K0]**2 + I_now[i%K0]**2) for i in range(4*K0)])

        if m > 0:
            E = np.array([R_now[k]**2 + I_now[k]**2 for k in range(K0)])
            N_next = 2*np.dot(ID,np.array(N_now) + E) - np.array(N_before) - 2*E
        D = dt*(0.5*Dx - 0.25*np.diag([N_now[k]+N_next[k] for k in range(K0)]))
        A = np.block([[Ik,D],[-D,Ik]])
        b = np.linalg.solve(A,2*np.array([R_now[i] if i < K0 else I_now[i-K0] for i in range(2*K0)]))
        R_next = -np.array(R_now) + b[:K0]; I_next = -np.array(I_now) + b[K0:]

        V_next = [-V_now[k]+ 0.5*dt*(N_next[k] + N_now[k] + R_now[k]**2 + R_next[k]**2 + I_now[k]**2 + I_next[k]**2) for k in range(K0)]
        DR_next = np.dot(Dx,np.array(R_next)); DI_next = np.dot(Dx,np.array(I_next)); DV_next = np.dot(Dx,np.array(V_next))

        F = F0 + np.array([R_next[i%K0] + 0.5*dt*DI_next[i%K0] - 0.25*dt*(I_next[i%K0] + I_now[i%K0])*(N_next[i%K0] + N_now[i%K0]) if i//K0 == 0
            else I_next[i%K0] - 0.5*dt*DR_next[i%K0] + 0.25*dt*(R_next[i%K0] + R_now[i%K0])*(N_next[i%K0] + N_now[i%K0]) if i//K0 == 1
            else N_next[i%K0] - 0.5*dt*DV_next[i%K0] if i//K0 == 2
            else V_next[i%K0] - 0.5*dt*(N_next[i%K0] + R_next[i%K0]**2 + I_next[i%K0]**2) for i in range(4*K0)])
        #print(m,"Start:",norm(F,dx)**0.5)

        while norm(F,dx)**0.5 >= eps:
            dNN = dN - 0.25*dt*np.diag(N_next)
            dR = dt*np.diag(R_next); dI = dt*np.diag(I_next)
            dRR = 0.25*dR + dR_now; dII = 0.25*dI + dI_now
            DF = np.block([[Ik,dNN,-dII,Zk],[-dNN,Ik,dRR,Zk],[Zk,Zk,Ik,-0.5*dt*Dx],[-dR,-dI,-0.5*dt*Ik,Ik]])
            r = np.linalg.solve(DF,F)

            R_next -= r[:K0]; I_next -= r[K0:2*K0]; N_next -= r[2*K0:3*K0]; V_next -= r[3*K0:]
            DR_next = np.dot(Dx,np.array(R_next)); DI_next = np.dot(Dx,np.array(I_next)); DV_next = np.dot(Dx,np.array(V_next))

            F = F0 + np.array([R_next[i%K0] + 0.5*dt*DI_next[i%K0] - 0.25*dt*(I_next[i%K0] + I_now[i%K0])*(N_next[i%K0] + N_now[i%K0]) if i//K0 == 0
                else I_next[i%K0] - 0.5*dt*DR_next[i%K0] + 0.25*dt*(R_next[i%K0] + R_now[i%K0])*(N_next[i%K0] + N_now[i%K0]) if i//K0 == 1
                else N_next[i%K0] - 0.5*dt*DV_next[i%K0] if i//K0 == 2
                else V_next[i%K0] - 0.5*dt*(N_next[i%K0] + R_next[i%K0]**2 + I_next[i%K0]**2) for i in range(4*K0)])

            t += 1
            if t > 1000:
                return "Failure"
        Rs.append(R_next); Is.append(I_next); Ns.append(N_next); Vs.append(V_next)
        R_now = R_next; I_now = I_next; N_before = N_now; N_now = N_next; V_now = V_next
        DR_now = DR_next; DI_now = DI_next; DV_now = DV_next;
        m += 1
        if m%10 == 0:
            print("時刻:",m,"終点:",M)
    end = time.perf_counter()
    print("DVDM実行時間:",end-start)

    return [[str(end-start)]+[0 for i in range(len(Rs[0])-1)]],Rs,Is,Ns,Vs

def checking_DVDM(Param2,K,M,eps):
    Ltent,T = Param2[:2]
    dx = Ltent/K; dt = T/M #print(dt,dx)
    Rs,Is,Ns = DVDM_Glassey_infinite(Param2,K,M,eps)[:3]
    eEs = []; eNs = []
    rEs = []; rNs = []

    RANGE = [i for i in range(len(Rs))]
    #RANGE = [len(Rs)-1] # 最終時刻での誤差だけ知りたいとき
    for i in RANGE:
        tR,tI,tN = analytical_solutions_infinite(Param,i*dt,K)

        tnorm = (norm(tR,dx) + norm(tI,dx))**0.5

        eEs.append((dist(Rs[i],tR,dx)**2+dist(Is[i],tI,dx)**2)**0.5); eNs.append(dist(Ns[i],tN,dx))
        rEs.append(eEs[i]/tnorm); rNs.append(eNs[i]/norm(tN,dx)**2)
    print("各要素の最大誤差:",max(eEs),max(eNs))
    print("各要素の最大誤差比:",max(rEs),max(rNs))
    for i in range(4):
        j = math.floor(T*(i+1)/(4*dt))
        print("t:",32*(i+1)/4,",",eEs[j],eNs[j],",",rEs[j],rNs[j])
    return (dx**2 + dt**2)**0.5,eEs,eNs

# Glassey,DVDM,解析解を T/3 ごとに比較
def comparing(Ltent,Emax,N,eps):
    m = 1; v = 4*math.pi*m/Ltent; u = v/2 - Emax**2/(v*(1-v**2))
    T = Ltent/v; phi = v/2
    Param2 = [Ltent,T,Emax,v,u,phi]
    K = math.floor(Ltent*N); M = math.floor(T*N)
    dx = Ltent/K; dt = T/M

    RG,IG,NG = [],[],[]
    RD,ID,ND = [],[],[]

    fname = "L="+str(Ltent)+"Emax="+str(Emax)+"N="+str(N)+"Glasseyinf.csv"
    if not os.path.isfile(fname):
        time,Rs,Ns,Is = Glassey_infinite(Param2,K,M)
        pd.DataFrame(time+Rs+Ns+Is).to_csv(fname)
    with open(fname) as f:
        for row in csv.reader(f, quoting=csv.QUOTE_NONNUMERIC):
            if row[0] in [M//3+1,2*M//3+1,3*M//3+1]:
                RG.append(np.array(row[1:]))
            if row[0] in [M+2+M//3,M+2+2*M//3,M+2+3*M//3]:
                IG.append(np.array(row[1:]))
            if row[0] in [2*M+3+M//3,2*M+3+2*M//3,2*M+3+3*M//3]:
                NG.append(np.array(row[1:]))

    fname = "L="+str(Ltent)+"Emax="+str(Emax)+"N="+str(N)+"DVDMinf.csv"
    if not os.path.isfile(fname):
        time,Rs,Ns,Is = DVDM_Glassey_infinite(Param2,K,M,eps)[:4]
        pd.DataFrame(time+Rs+Ns+Is).to_csv(fname)
    with open(fname) as f:
        for row in csv.reader(f, quoting=csv.QUOTE_NONNUMERIC):
            if row[0] in [M//3+1,2*M//3+1,3*M//3+1]:
                RD.append(np.array(row[1:]))
            if row[0] in [M+2+M//3,M+2+2*M//3,M+2+3*M//3]:
                ID.append(np.array(row[1:]))
            if row[0] in [2*M+3+M//3,2*M+2+2*M//3,2*M+3+3*M//3]:
                ND.append(np.array(row[1:]))
    x = np.linspace(-Ltent, 2*Ltent, 3*K)

    fig = plt.figure()
    axs = []
    for i in range(3):
        axs.append(fig.add_subplot(3, 3, 3*i+1))
        axs.append(fig.add_subplot(3, 3, 3*i+2))
        axs.append(fig.add_subplot(3, 3, 3*i+3))

    for i in range(3):
        tR,tI,tN = analytical_solutions_infinite(Param2,((i+1)*M//3)*dt,K)

        ax = axs[3*i:3*i+3]

        l1,l2,l3 = "Glassey","DVDM","analytical"
        ax[0].plot(x, RG[i], label=l1)
        ax[0].plot(x, RD[i], label=l2)
        ax[0].plot(x, tR, label=l3)
        ax[1].plot(x, IG[i], label=l1)
        ax[1].plot(x, ID[i], label=l2)
        ax[1].plot(x, tI, label=l3)
        ax[2].plot(x, NG[i], label=l1)
        ax[2].plot(x, ND[i], label=l2)
        ax[2].plot(x, tN, label=l3)
        ax[0].legend(); ax[1].legend(); ax[2].legend()
    plt.show()


N = 20
K = math.floor(Ltent*N)
M = math.floor(T*N)

#print("L=",L,"Emax=",Emax)
#print("N=",N,"dt=",T/M,"dx=",L/K)
#print(checking_Glassey(Param,K,M))
#Glassey(Param,K,M)
#DVDM_Glassey(Param,K,M,10**(-5))
#print(checking_DVDM(Param,K,M,10**(-5)))
#print(checking_DVDM(Param,K,M,10**(-8)))
#comparing(20,1,10,10**(-8))
