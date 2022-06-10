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
    q = (Emax**2 - Emin**2)**0.5/Emax
    Kq = scipy.special.ellipk(q)
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
        if Emin == Enew: #性能限界
            break
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
L = 20; Emax = 1.3; m = 1; eps = 10**(-9)
v, Emin, q, N_0, u = parameters(L,1,Emax,eps)
T = L/v; phi = v/2
Param = [L,Emax,v,Emin,q,N_0,u,T,phi]

###############################################################################

def analytical_solutions(Param,t,K):
    L,Emax,v,Emin,q,N_0,u,T,phi = Param
    dx = L/K
    vv = (1 - v*v)**0.5; vv2 = 1 - v*v; WW = Emax/(2**0.5*vv)
    W = [WW*(k*dx-v*t) for k in range(K)]
    dn = [scipy.special.ellipj(W[k],q)[2] for k in range(len(W))]
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

    vv = (1 - v*v)**0.5; WW = Emax*dx/(2**0.5*vv); coef = -2**0.5*Emax**2*q**2*v/vv*3

    ellipjs = [scipy.special.ellipj(WW*k,q) for k in range(K)]
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
#スキーム本体

# Glassey スキーム
def Glassey(Param,K,M):
    start = time.perf_counter()
    dx = L/K; dt = T/M #print(dt,dx)

    # 数値解の記録
    Rs = []; Is = []; Ns = []
    R_now,I_now,N_now,N_next = initial_condition_common(Param,K,M)[:4]
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
            R_next = -np.array(R_now) + b[:K]; I_next = -np.array(I_now) + b[K:]
            Rs.append(R_next); Is.append(I_next)
            R_now = R_next; I_now = I_next
            ri_t += 1
            if ri_t%10 == 0:
                print(ri_t,n_t,M) #実行の進捗の目安として
        else: # N(m-1),Nm,Em から N(m+1)を求める
            E = np.array([R_now[k]**2 + I_now[k]**2 for k in range(K)])
            NN = np.array(N_next) + E
            N_now, N_next = N_next, 2*np.dot(ID,NN) - np.array(N_now) - 2*E
            Ns.append(N_next)
            n_t += 1
    end = time.perf_counter()
    print("Glassey実行時間:",end-start)

    WantToKnow = True #ノルム・エネルギーを知りたいとき
    WantToPlot = False #ノルム・エネルギーを描画したいとき
    if WantToKnow:
        Norm = [norm(Rs[i],dx) + norm(Is[i],dx) for i in range(len(Rs))]
        dNorm = [abs(Norm[i] - Norm[0]) for i in range(1,len(Rs))]
        rNorm = [dNorm[i]/Norm[0] for i in range(len(Rs)-1)]

        DD = -2*np.eye(K-1,k=0) + np.eye(K-1,k=1) + np.eye(K-1,k=-1)
        DDI = np.linalg.inv(DD)
        Energy = [energy_Glassey(Rs[i+1],Is[i+1],Ns[i],Ns[i+1],DDI,dt,dx) for i in range(len(Rs)-1)]
        dEnergy = [abs(Energy[i] - Energy[0]) for i in range(len(Rs)-1)]
        rEnergy = [dEnergy[i]/abs(Energy[0]) for i in range(len(Rs)-1)]

        print("初期値に対するノルムの最大誤差:",max(dNorm))
        print("初期値に対するノルムの最大誤差比:",max(rNorm))

        print("初期値に対するエネルギーの最大誤差:",max(dEnergy))
        print("初期値に対するエネルギーの最大誤差比:",max(rEnergy))
        if WantToPlot:
            Time = [i for i in range(1,len(Rs))]
            plt.plot(Time,dNorm,label="Norm")
            plt.plot(Time,dEnergy,label="Energy")
            plt.xlabel("time")
            plt.ylabel("errors of Norm and Energy")
            plt.legend()
            plt.show()
    return Rs,Is,Ns

def checking_Glassey(Param,K,M):
    dx = L/K; dt = T/M #print(dt,dx)
    Rs,Is,Ns = Glassey(Param,K,M)[:3]
    eEs = [];eNs = []
    rEs = [];rNs = []

    RANGE = [i for i in range(len(Rs))]
    #RANGE = [len(Rs)-1] # 最終時刻での誤差だけ知りたいとき
    for i in RANGE:
        tR,tI,tN = analytical_solutions(Param,i*dt,K)

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
def DVDM_Glassey(Param,K,M,eps):
    start = time.perf_counter()
    dx = L/K; dt = T/M; #print(dt,dx)

    # 数値解の記録
    Rs = []; Is = []; Ns = []; Vs = []
    R_now,I_now,N_now,V_now,N_next = initial_condition_DVDM(Param,K,M)
    Rs.append(R_now); Is.append(I_now); Ns.append(N_now); Vs.append(V_now)

    m = 0

    Ik = np.identity(K); Zk = np.zeros((K,K))
    Dx = (-2*Ik + np.eye(K,k=1) + np.eye(K,k=K-1) + np.eye(K,k=-1) + np.eye(K,k=-K+1))/dx**2
    ID = np.linalg.inv(Ik-0.5*dt**2*Dx)

    DR_now = np.dot(Dx,np.array(R_now)); DI_now = np.dot(Dx,np.array(I_now)); DV_now = np.dot(Dx,np.array(V_now))

    while m*dt < T:
        t = 0
        dN = 0.5*dt*Dx - 0.25*dt*np.diag(N_now)
        dR_now = 0.25*np.diag(R_now); dI_now = 0.25*np.diag(I_now)

        F0 = np.array([- R_now[i%K] + 0.5*dt*DI_now[i%K] if i//K == 0
        else -I_now[i%K] - 0.5*dt*DR_now[i%K] if i//K == 1
        else -N_now[i%K] - 0.5*dt*DV_now[i%K] if i//K == 2
        else - V_now[i%K] - 0.5*dt*(N_now[i%K] + R_now[i%K]**2 + I_now[i%K]**2) for i in range(4*K)])

        if m > 0:
            E = np.array([R_now[k]**2 + I_now[k]**2 for k in range(K)])
            N_next = 2*np.dot(ID,np.array(N_now) + E) - np.array(N_before) - 2*E
        D = dt*(0.5*Dx - 0.25*np.diag([N_now[k]+N_next[k] for k in range(K)]))
        A = np.block([[Ik,D],[-D,Ik]])
        b = np.linalg.solve(A,2*np.array([R_now[i] if i < K else I_now[i-K] for i in range(2*K)]))
        R_next = -np.array(R_now) + b[:K]; I_next = -np.array(I_now) + b[K:]

        V_next = [-V_now[k]+ 0.5*dt*(N_next[k] + N_now[k] + R_now[k]**2 + R_next[k]**2 + I_now[k]**2 + I_next[k]**2) for k in range(K)]
        DR_next = np.dot(Dx,np.array(R_next)); DI_next = np.dot(Dx,np.array(I_next)); DV_next = np.dot(Dx,np.array(V_next))

        F = F0 + np.array([R_next[i%K] + 0.5*dt*DI_next[i%K] - 0.25*dt*(I_next[i%K] + I_now[i%K])*(N_next[i%K] + N_now[i%K]) if i//K == 0
            else I_next[i%K] - 0.5*dt*DR_next[i%K] + 0.25*dt*(R_next[i%K] + R_now[i%K])*(N_next[i%K] + N_now[i%K]) if i//K == 1
            else N_next[i%K] - 0.5*dt*DV_next[i%K] if i//K == 2
            else V_next[i%K] - 0.5*dt*(N_next[i%K] + R_next[i%K]**2 + I_next[i%K]**2) for i in range(4*K)])
        #print(m,"Start:",norm(F,dx)**0.5)

        while norm(F,dx)**0.5 >= eps:
            dNN = dN - 0.25*dt*np.diag(N_next)
            dR = dt*np.diag(R_next); dI = dt*np.diag(I_next)
            dRR = 0.25*dR + dR_now; dII = 0.25*dI + dI_now
            DF = np.block([[Ik,dNN,-dII,Zk],[-dNN,Ik,dRR,Zk],[Zk,Zk,Ik,-0.5*dt*Dx],[-dR,-dI,-0.5*dt*Ik,Ik]])
            r = np.linalg.solve(DF,F)

            R_next -= r[:K]; I_next -= r[K:2*K]; N_next -= r[2*K:3*K]; V_next -= r[3*K:]
            DR_next = np.dot(Dx,np.array(R_next)); DI_next = np.dot(Dx,np.array(I_next)); DV_next = np.dot(Dx,np.array(V_next))

            F = F0 + np.array([R_next[i%K] + 0.5*dt*DI_next[i%K] - 0.25*dt*(I_next[i%K] + I_now[i%K])*(N_next[i%K] + N_now[i%K]) if i//K == 0
                else I_next[i%K] - 0.5*dt*DR_next[i%K] + 0.25*dt*(R_next[i%K] + R_now[i%K])*(N_next[i%K] + N_now[i%K]) if i//K == 1
                else N_next[i%K] - 0.5*dt*DV_next[i%K] if i//K == 2
                else V_next[i%K] - 0.5*dt*(N_next[i%K] + R_next[i%K]**2 + I_next[i%K]**2) for i in range(4*K)])

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

    WantToKnow = False #ノルム・エネルギーを知りたいとき
    WantToPlot = False #ノルム・エネルギーを描画したいとき
    if WantToKnow:
        Norm = [norm(Rs[i],dx) + norm(Is[i],dx) for i in range(len(Rs))]
        dNorm = [abs(Norm[i] - Norm[0]) for i in range(len(Rs))]
        rNorm = [dNorm[i]/Norm[0] for i in range(len(Rs))]

        Energy = [energy_DVDM(Rs[i],Is[i],Ns[i],Vs[i],dx) for i in range(len(Rs))]
        dEnergy = [abs(Energy[i] - Energy[0]) for i in range(len(Rs))]
        rEnergy = [dEnergy[i]/abs(Energy[0]) for i in range(len(Rs))]

        print("初期値に対するノルムの最大誤差:",max(dNorm))
        print("初期値に対するノルムの最大誤差比:",max(rNorm))

        print("初期値に対するエネルギーの最大誤差:",max(dEnergy))
        print("初期値に対するエネルギーの最大誤差比:",max(rEnergy))

        if WantToPlot:
            Time = [i for i in range(len(Rs))]
            plt.plot(Time,dNorm,label="Norm")
            plt.plot(Time,dEnergy,label="Energy")
            plt.xlabel("time")
            plt.ylabel("errors of Norm and Energy")
            plt.legend()
            plt.show()
    return Rs,Is,Ns,Vs

def checking_DVDM(Param,K,M,eps):
    dx = L/K; dt = T/M #print(dt,dx)
    Rs,Is,Ns = DVDM_Glassey(Param,K,M,eps)[:3]
    eEs = []; eNs = []
    rEs = []; rNs = []

    RANGE = [i for i in range(len(Rs))]
    #RANGE = [len(Rs)-1] # 最終時刻での誤差だけ知りたいとき
    for i in RANGE:
        tR,tI,tN = analytical_solutions(Param,i*dt,K)

        tnorm = (norm(tR,dx) + norm(tI,dx))**0.5

        eEs.append((dist(Rs[i],tR,dx)**2+dist(Is[i],tI,dx)**2)**0.5); eNs.append(dist(Ns[i],tN,dx))
        rEs.append(eEs[i]/tnorm); rNs.append(eNs[i]/norm(tN,dx)**2)
    print("各要素の最大誤差:",max(eEs),max(eNs))
    print("各要素の最大誤差比:",max(rEs),max(rNs))
    for i in range(4):
        j = math.floor(T*(i+1)/(4*dt))
        print("t:",32*(i+1)/4,",",eEs[j],eNs[j],",",rEs[j],rNs[j])
    return (dx**2 + dt**2)**0.5,eEs,eNs

N = 1
K = math.floor(L*N)
M = math.floor(T*N)

#print("L=",L,"Emax=",Emax)
#print("N=",N,"dt=",T/M,"dx=",L/K)
#print(checking_Glassey(Param,K,M))
#print(checking_DVDM(Param,K,M,10**(-5)))
#print(checking_DVDM(Param,K,M,10**(-8)))
