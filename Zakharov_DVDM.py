import math
import scipy.special
import matplotlib.pyplot as plt
import numpy as np
from sympy import *
import random

###############################################################################
#パラメータを定めるための関数

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

    return R0,I0,N0,N1,Nt0

def initial_condition_DVDM(K,M):
    dx = L/K; dt = T/M
    R0,I0,N0,N1,Nt0 = initial_condition_common(K,M)

    DD = -2*np.eye(K-1,k=0) + np.eye(K-1,k=1) + np.eye(K-1,k=-1)
    DDI = np.linalg.inv(DD)

    dN = [Nt0[k] for k in range(1,K)]
    V0 = dx**2 * np.dot(DDI,dN)
    V0 = [0]+[V0[i] for i in range(K-1)]

    return R0,I0,N0,V0,N1

###############################################################################
#スキーム本体

#L2ノルム
def norm(v,dx):
    Ltwo = 0
    for i in range(len(v)):
        Ltwo += v[i]**2*dx
    return Ltwo

#エネルギー
def FD(v,dx):
    return [(v[(k+1)%K] - v[k])/dx for k in range(len(v))]

def energy(R,I,N,V,dt,dx):
    dR = FD(R,dx)
    dI = FD(I,dx)
    dV = FD(V,dx)
    Energy = norm(dR,dx) + norm(dI,dx) + 0.5*norm(N,dx) + 0.5*norm(dV,dx)
    for i in range(K):
        Energy += N[i]*(R[i]**2 + I[i]**2)*dx
    return Energy

#Glasseyスキーム本体

# R,I,N,V について解く
def DVDM1(K,M,eps):
    dx = L/K; dt = T/M
    print(dt,dx)

    # 数値解の記録
    Rs = []; Is = []; Ns = []; Vs = []
    R0,I0,N0,V0 = initial_condition_DVDM(K,M)[:4]
    Rs.append(R0); Is.append(I0); Ns.append(N0); Vs.append(V0)
    R_now = R0; I_now = I0; N_now = N0; V_now = V0;

    m = 0

    Ik = np.identity(K)
    Zk = np.zeros((K,K))
    Dx = (-2*Ik + np.eye(K,k=1) + np.eye(K,k=K-1) + np.eye(K,k=-1) + np.eye(K,k=-K+1))/dx**2
    S = Ik - 0.25*dt**2*Dx
    DR_now = np.dot(Dx,np.array(R_now))
    DI_now = np.dot(Dx,np.array(I_now))
    DV_now = np.dot(Dx,np.array(V_now))

    while m < M:
        F0 = np.array([- R_now[i%K] + 0.5*dt*DI_now[i%K] if i//K == 0
        else -I_now[i%K] - 0.5*dt*DR_now[i%K] if i//K == 1
        else -N_now[i%K] - 0.5*dt*DV_now[i%K] if i//K == 2
        else - V_now[i%K] - 0.5*dt*(N_now[i%K] + R_now[i%K]**2 + I_now[i%K]**2) for i in range(4*K)])

        R_next = [R_now[k]*random.random() for k in range(K)]
        I_next = [I_now[k]*random.random() for k in range(K)]
        N_next = [N_now[k]*random.random() for k in range(K)]
        V_next = [V_now[k]*random.random() for k in range(K)]
        DR_next = np.dot(Dx,np.array(R_next))
        DI_next = np.dot(Dx,np.array(I_next))
        DV_next = np.dot(Dx,np.array(V_next))
        F = F0 + np.array([R_next[i%K] + 0.5*dt*DI_next[i%K] - 0.25*dt*(I_next[i%K] + I_now[i%K])*(N_next[i%K] + N_now[i%K]) if i//K == 0
            else I_next[i%K] - 0.5*dt*DR_next[i%K] + 0.25*dt*(R_next[i%K] + R_now[i%K])*(N_next[i%K] + N_now[i%K]) if i//K == 1
            else N_next[i%K] - 0.5*dt*DV_next[i%K] if i//K == 2
            else V_next[i%K] - 0.5*dt*(N_next[i%K] + R_next[i%K]**2 + I_next[i%K]**2) for i in range(4*K)])


        dN = 0.5*dt*Dx - 0.25*dt*np.diag(N_now)
        dR_now = 0.25*np.diag(R_now)
        dI_now = 0.25*np.diag(I_now)
        t = 0
        while norm(F,dx)**0.5 >= eps:
            dNN = dN - 0.25*dt*np.diag(N_next)
            dR = dt*np.diag(R_next)
            dI = dt*np.diag(I_next)
            dRR = 0.25*dR + dR_now
            dII = 0.25*dI + dI_now

            DF = np.block([[Ik,dNN,-dII,Zk],[-dNN,Ik,dRR,Zk],[Zk,Zk,Ik,-0.5*dt*Dx],[-dR,-dI,-0.5*dt*Ik,Ik]])
            r = np.linalg.solve(DF,F)

            R_next -= r[:K]
            I_next -= r[K:2*K]
            N_next -= r[2*K:3*K]
            V_next -= r[3*K:]

            DR_next = np.dot(Dx,np.array(R_next))
            DI_next = np.dot(Dx,np.array(I_next))
            DV_next = np.dot(Dx,np.array(V_next))

            F = F0 + np.array([R_next[i%K] + 0.5*dt*DI_next[i%K] - 0.25*dt*(I_next[i%K] + I_now[i%K])*(N_next[i%K] + N_now[i%K]) if i//K == 0
                else I_next[i%K] - 0.5*dt*DR_next[i%K] + 0.25*dt*(R_next[i%K] + R_now[i%K])*(N_next[i%K] + N_now[i%K]) if i//K == 1
                else N_next[i%K] - 0.5*dt*DV_next[i%K] if i//K == 2
                else V_next[i%K] - 0.5*dt*(N_next[i%K] + R_next[i%K]**2 + I_next[i%K]**2) for i in range(4*K)])
            #print(F)
            t += 1
            if t%100 == 0:
                print("時刻:",m,"反復",t,norm(F,dx))

        Rs.append(R_next); Is.append(I_next); Ns.append(N_next); Vs.append(V_next)
        R_now = R_next; I_now = I_next; N_now = N_next; V_now = V_next
        DR_now = DR_next; DI_now = DI_next; DV_now = DV_next;
        m += 1
        print("時刻:",m,"終点:",M)

    WantToKnow = True #ノルム・エネルギーを知りたいとき
    WantToPlot = True #ノルム・エネルギーを描画したいとき
    if WantToKnow:
        Norm = [norm(Rs[i],dx) + norm(Is[i],dx) for i in range(len(Rs))]
        dNorm = [abs(Norm[i] - Norm[0]) for i in range(len(Rs))]
        print("初期値に対するノルムの最大誤差:",max(dNorm))
        Energy = [energy(Rs[i],Is[i],Ns[i],Vs[i],dt,dx) for i in range(len(Rs))]
        dEnergy = [abs(Energy[i] - Energy[0]) for i in range(len(Rs))]
        #print(Energy)
        print("初期値に対するエネルギーの最大誤差:",max(dEnergy))
        if WantToPlot:
            Time = [i for i in range(len(Rs))]
            plt.plot(Time,dNorm,label="Norm")
            plt.plot(Time,dEnergy,label="Energy")
            plt.xlabel("time")
            plt.ylabel("errors of Norm and Energy")
            plt.legend()
            plt.show()
    return Rs,Is,Ns,Vs

def checking1(K,M,eps):
    dx = L/K; dt = T/M
    #print(dt,dx)
    Rs,Is,Ns = DVDM1(K,M,eps)[:3]
    dists = []

    vv = (1 - v*v)**0.5
    vv2 = 1 - v*v
    WW = Emax/(2**0.5*vv)
    qq = q**2

    RANGE = [i for i in range(M+1)]
    #RANGE = [M] # 最終時刻での誤差だけ知りたいとき
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

# R,I,Nについてのみ解く
def DVDM2(K,M,eps):
    dx = L/K; dt = T/M
    print(dt,dx)

    # 数値解の記録
    Rs = []; Is = []; Ns = []
    R0,I0,N0,_,R1,I1,N1,__ = initial_condition_DVDM(K,M)
    Rs.append(R0); Rs.append(R1); Is.append(I0); Is.append(I1); Ns.append(N0); Ns.append(N1)
    R_now = R1; I_now = I1; N_now = N1;
    R_before = R0; I_before = I0; N_before = N0

    m = 1

    Ik = np.identity(K)
    Zk = np.zeros((K,K))
    Dx = (-2*Ik + np.eye(K,k=1) + np.eye(K,k=K-1) + np.eye(K,k=-1) + np.eye(K,k=-K+1))/dx**2
    S = Ik - 0.25*dt**2*Dx

    DR_now = np.dot(Dx,np.array(R_now))
    DI_now = np.dot(Dx,np.array(I_now))
    DR_nb = np.dot(Dx,np.array([2*R_now[k]**2 + R_before[k]**2 for k in range(K)]))
    DI_nb = np.dot(Dx,np.array([2*I_now[k]**2 + I_before[k]**2 for k in range(K)]))
    DN_nb = np.dot(Dx,2*np.array(N_now) + np.array(N_before))

    import random
    while m < M:
        F0 = np.array([- R_now[i%K] + 0.5*dt*DI_now[i%K] if i//K == 0
        else -I_now[i%K] - 0.5*dt*DR_now[i%K] if i//K == 1
        else -2*N_now[i%K] + N_before[i%K] - 0.25*dt**2*DN_nb[i%K] -0.25*dt**2*(DR_nb[i%K]+DI_nb[i%K]) for i in range(3*K)])

        R_next = R_now; I_next = I_now; N_next = N_now;
        DR_next = np.dot(Dx,np.array(R_next))
        DI_next = np.dot(Dx,np.array(I_next))
        DRN = np.dot(Dx,np.array([R_next[k]**2 for k in range(K)]))
        DIN = np.dot(Dx,np.array([I_next[k]**2 for k in range(K)]))
        DN_next = np.dot(Dx,np.array(N_next))

        F = F0 + np.array([R_next[i%K] + 0.5*dt*DI_next[i%K] - 0.25*dt*(I_next[i%K] + I_now[i%K])*(N_next[i%K] + N_now[i%K]) if i//K == 0
        else I_next[i%K] - 0.5*dt*DR_next[i%K] + 0.25*dt*(R_next[i%K] + R_now[i%K])*(N_next[i%K] + N_now[i%K]) if i//K == 1
        else N_next[i%K] - 0.25*dt**2*DN_next[i%K] -0.25*dt**2*(DRN[i%K]+DIN[i%K]) for i in range(3*K)])

        dN = 0.5*dt*Dx - 0.25*dt*np.diag(N_now)
        dR_now = 0.25*np.diag(R_now)
        dI_now = 0.25*np.diag(I_now)

        t = 0
        while norm(F,dx) >= eps:
            dNN = dN - 0.25*dt*np.diag(N_next)
            dR = 0.5*dt**2*np.dot(Dx,np.diag(R_next))
            dI = 0.5*dt**2*np.dot(Dx,np.diag(I_next))
            dRR = 0.25*dR + dR_now
            dII = 0.25*dI + dI_now

            DF = np.block([[Ik,dNN,-dII],[-dNN,Ik,dRR],[-dR,-dI,S]])
            r = np.linalg.solve(DF,F)

            R_next -= r[:K]
            I_next -= r[K:2*K]
            N_next -= r[2*K:]

            DR_next = np.dot(Dx,np.array(R_next))
            DI_next = np.dot(Dx,np.array(I_next))
            DRN = np.dot(Dx,np.array([R_next[k]**2 for k in range(K)]))
            DIN = np.dot(Dx,np.array([I_next[k]**2 for k in range(K)]))
            DN_next = np.dot(Dx,np.array(N_next))

            F = F0 + np.array([R_next[i%K] + 0.5*dt*DI_next[i%K] - 0.25*dt*(I_next[i%K] + I_now[i%K])*(N_next[i%K] + N_now[i%K]) if i//K == 0
            else I_next[i%K] - 0.5*dt*DR_next[i%K] + 0.25*dt*(R_next[i%K] + R_now[i%K])*(N_next[i%K] + N_now[i%K]) if i//K == 1
            else N_next[i%K] - 0.25*dt**2*DN_next[i%K] -0.25*dt**2*(DRN[i%K]+DIN[i%K]) for i in range(3*K)])
            #print(F)
            t += 1
            if t%100 == 0:
                print("時刻:",m,"反復",t)

        Rs.append(R_next); Is.append(I_next); Ns.append(N_next)
        R_now,R_before = R_next,R_now; I_now,I_before = I_next,I_now; N_now,N_before = N_next,N_now
        DR_now = np.dot(Dx,np.array(R_now))
        DI_now = np.dot(Dx,np.array(I_now))
        DR_nb = np.dot(Dx,np.array([2*R_now[k]**2 + R_before[k]**2 for k in range(K)]))
        DI_nb = np.dot(Dx,np.array([2*I_now[k]**2 + I_before[k]**2 for k in range(K)]))
        DN_nb = np.dot(Dx,2*np.array(N_now) + np.array(N_before))
        m += 1
        print("時刻:",m,"終点:",M)
    return Rs,Is,Ns,Vs

def checking2(K,M,eps):
    dx = L/K; dt = T/M
    #print(dt,dx)
    Rs,Is,Ns = DVDM2(K,M,eps)[:3]
    dists = []

    vv = (1 - v*v)**0.5
    vv2 = 1 - v*v
    WW = Emax/(2**0.5*vv)
    qq = q**2

    RANGE = [i for i in range(M+1)]
    #RANGE = [M] # 最終時刻での誤差だけ知りたいとき
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

# Newton法の初期値をGlasseyで求める
def DVDM_Glassey(K,M,eps):
    dx = L/K; dt = T/M
    print(dt,dx)

    # 数値解の記録
    Rs = []; Is = []; Ns = []; Vs = []
    R0,I0,N0,V0,N1 = initial_condition_DVDM(K,M)
    Rs.append(R0); Is.append(I0); Ns.append(N0); Ns.append(N1); Vs.append(V0)
    R_now = R0; I_now = I0; N_before = N0; N_now = N0; V_now = V0;

    m = 0

    Ik = np.identity(K)
    Zk = np.zeros((K,K))
    Dx = (-2*Ik + np.eye(K,k=1) + np.eye(K,k=K-1) + np.eye(K,k=-1) + np.eye(K,k=-K+1))/dx**2
    ID = np.linalg.inv(Ik-0.5*dt**2*Dx)
    S = Ik - 0.25*dt**2*Dx

    DR_now = np.dot(Dx,np.array(R_now))
    DI_now = np.dot(Dx,np.array(I_now))
    DV_now = np.dot(Dx,np.array(V_now))

    while m < M:
        F0 = np.array([- R_now[i%K] + 0.5*dt*DI_now[i%K] if i//K == 0
        else -I_now[i%K] - 0.5*dt*DR_now[i%K] if i//K == 1
        else -N_now[i%K] - 0.5*dt*DV_now[i%K] if i//K == 2
        else - V_now[i%K] - 0.5*dt*(N_now[i%K] + R_now[i%K]**2 + I_now[i%K]**2) for i in range(4*K)])

        if m == 0:
            N_next = N1
        else:
            E = np.array([R_now[k]**2 + I_now[k]**2 for k in range(K)])
            NN = np.array(N_now) + E
            N_next = 2*np.dot(ID,NN) - np.array(N_before) - 2*E
        Dn = np.diag([N_now[k]+N_next[k] for k in range(K)])
        D = dt*(0.5*Dx - 0.25*Dn)
        A = np.block([[Ik,D],[-D,Ik]])
        b = np.linalg.solve(A,2*np.array([R_now[i] if i < K else I_now[i-K] for i in range(2*K)]))
        R_next = -np.array(R_now) + b[:K]
        I_next = -np.array(I_now) + b[K:]

        V_next = [-V_now[k]+ 0.5*dt*(N_next[k] + N_now[k] + R_now[k]**2 + R_next[k]**2 + I_now[k]**2 + I_next[k]**2) for k in range(K)]

        DR_next = np.dot(Dx,np.array(R_next))
        DI_next = np.dot(Dx,np.array(I_next))
        DV_next = np.dot(Dx,np.array(V_next))
        F = F0 + np.array([R_next[i%K] + 0.5*dt*DI_next[i%K] - 0.25*dt*(I_next[i%K] + I_now[i%K])*(N_next[i%K] + N_now[i%K]) if i//K == 0
            else I_next[i%K] - 0.5*dt*DR_next[i%K] + 0.25*dt*(R_next[i%K] + R_now[i%K])*(N_next[i%K] + N_now[i%K]) if i//K == 1
            else N_next[i%K] - 0.5*dt*DV_next[i%K] if i//K == 2
            else V_next[i%K] - 0.5*dt*(N_next[i%K] + R_next[i%K]**2 + I_next[i%K]**2) for i in range(4*K)])
        print(m,"Start:",norm(F,dx)**0.5)

        R_tent,I_tent,N_tent,V_tent = R_next,I_next,N_next,V_next

        dN = 0.5*dt*Dx - 0.25*dt*np.diag(N_now)
        dR_now = 0.25*np.diag(R_now)
        dI_now = 0.25*np.diag(I_now)
        t = 0
        while norm(F,dx)**0.5 >= eps:
            before = norm(F,dx)**0.5
            dNN = dN - 0.25*dt*np.diag(N_next)
            dR = dt*np.diag(R_next)
            dI = dt*np.diag(I_next)
            dRR = 0.25*dR + dR_now
            dII = 0.25*dI + dI_now

            DF = np.block([[Ik,dNN,-dII,Zk],[-dNN,Ik,dRR,Zk],[Zk,Zk,Ik,-0.5*dt*Dx],[-dR,-dI,-0.5*dt*Ik,Ik]])
            r = np.linalg.solve(DF,F)

            R_next -= r[:K]
            I_next -= r[K:2*K]
            N_next -= r[2*K:3*K]
            V_next -= r[3*K:]

            DR_next = np.dot(Dx,np.array(R_next))
            DI_next = np.dot(Dx,np.array(I_next))
            DV_next = np.dot(Dx,np.array(V_next))

            F = F0 + np.array([R_next[i%K] + 0.5*dt*DI_next[i%K] - 0.25*dt*(I_next[i%K] + I_now[i%K])*(N_next[i%K] + N_now[i%K]) if i//K == 0
                else I_next[i%K] - 0.5*dt*DR_next[i%K] + 0.25*dt*(R_next[i%K] + R_now[i%K])*(N_next[i%K] + N_now[i%K]) if i//K == 1
                else N_next[i%K] - 0.5*dt*DV_next[i%K] if i//K == 2
                else V_next[i%K] - 0.5*dt*(N_next[i%K] + R_next[i%K]**2 + I_next[i%K]**2) for i in range(4*K)])

            t += 1
            if t%100 == 0:
                print("時刻:",m,"反復",t,norm(F,dx)**0.5)

        Rs.append(R_next); Is.append(I_next); Ns.append(N_next); Vs.append(V_next)
        R_now = R_next; I_now = I_next; N_before = N_now; N_now = N_next; V_now = V_next
        DR_now = DR_next; DI_now = DI_next; DV_now = DV_next;
        m += 1
        print("時刻:",m,"終点:",M)

    WantToKnow = True #ノルム・エネルギーを知りたいとき
    WantToPlot = False #ノルム・エネルギーを描画したいとき
    if WantToKnow:
        Norm = [norm(Rs[i],dx) + norm(Is[i],dx) for i in range(len(Rs))]
        dNorm = [abs(Norm[i] - Norm[0]) for i in range(len(Rs))]
        print("初期値に対するノルムの最大誤差:",max(dNorm))
        Energy = [energy(Rs[i],Is[i],Ns[i],Vs[i],dt,dx) for i in range(len(Rs))]
        dEnergy = [abs(Energy[i] - Energy[0]) for i in range(len(Rs))]
        #print(Energy)
        print("初期値に対するエネルギーの最大誤差:",max(dEnergy))
        if WantToPlot:
            Time = [i for i in range(len(Rs))]
            plt.plot(Time,dNorm,label="Norm")
            plt.plot(Time,dEnergy,label="Energy")
            plt.xlabel("time")
            plt.ylabel("errors of Norm and Energy")
            plt.legend()
            plt.show()
    return Rs,Is,Ns,Vs

def checking3(K,M,eps):
    dx = L/K; dt = T/M
    #print(dt,dx)
    Rs,Is,Ns = DVDM_Glassey(K,M,eps)[:3]
    dists = []

    vv = (1 - v*v)**0.5
    vv2 = 1 - v*v
    WW = Emax/(2**0.5*vv)
    qq = q**2

    RANGE = [i for i in range(M+1)]
    #RANGE = [M] # 最終時刻での誤差だけ知りたいとき
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

#DVDM1(K,M,10**(-3))
#print(checking2(K,M,10**(-6)))
#DVDM_Glassey(K,M,10**(-5))
print(checking3(K,M,10**(-8)))

# V を考慮せずに解く実装2は論外レベルで不安定
