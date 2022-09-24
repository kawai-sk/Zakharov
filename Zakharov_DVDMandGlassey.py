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
#パラメータを定めるための関数

from mpmath import *
mp.dps = 100
#print(mp)

#qの探索：普通に計算できる場合
def finding(L,m,Emax,eps):
    #計算に使う定数
    v = 4*math.pi*m/L; K = L*Emax*0.5/(2*(1-v**2))**0.5

    #[0,1]内の二分探索
    h = mpf(str(1)); l = mpf(str(0)); q = (h+l)/mpf(str(2))
    Kq = float(ellipk(q))
    while abs(Kq - K) >= eps:
        #print(Emax,q,Kq,K)
        if Kq < K:
            if l == q:
                print("Failure") #性能限界
                break
            l = q
        else:
            if h == q:
                print("Failure") #性能限界
                break
            h = q
        q = (h+l)/mpf(str(2)); Kq = float(ellipk(q))
    return q

#各パラメータの出力
def parameters(L,m,Emax,eps):
    v = 4*math.pi*m/L
    q = finding(L,m,Emax,eps)
    N_0 = 2**1.5*v**2*Emax*float(ellipe(q))/(L*(1-v**2)**0.5)
    u = v/2 + 2*N_0/v - (2-float(q))*Emax**2/(v*(1-v**2))
    T = L/v; phi = v/2
    return [L,Emax,v,q,N_0,u,T,phi]

# Emax < 0.17281841279256 を目安に ellipk(q) が q<0 となり機能しなくなる
#print(parameters(20,1,0.2,10**(-8)))
###############################################################################

def analytical_solutions(Param,t,K):
    L,Emax,v,q,N_0,u,T,phi = Param
    dx = L/K
    vv = (1 - v*v)**0.5; vv2 = 1 - v*v; WW = Emax/(2**0.5*vv); Kq = float(ellipk(q))
    coef1 = -2**0.5*Emax**3*float(q)*v/vv**3; coef2 = 2**0.5*v*Emax/vv; coef3 = v*Emax**2/vv2
    W = [WW*(k*dx-v*t) for k in range(K)]
    dn = [float(ellipfun('dn',W[k],q)) for k in range(K)]
    F = [Emax*dn[k] for k in range(K)]

    R = [F[k]*math.cos(phi*(k*dx-u*t)) for k in range(K)]
    I = [F[k]*math.sin(phi*(k*dx-u*t)) for k in range(K)]
    N = [-F[k]**2/vv2 + N_0 for k in range(K)]
    Nt = [coef1*dn[k]*float(ellipfun('sn',W[k],q)*ellipfun('cn',W[k],q)) for k in range(K)]

    snV = [float(asin(ellipfun('sn',W[k],q))) if -Kq < W[k] <= Kq
     else math.pi - float(asin(ellipfun('sn',2*Kq - W[k],q))) if W[k] > Kq
      else float(asin(ellipfun('sn',W[k] + 2*Kq,q))) - math.pi for k in range(K)]

    V = [coef2*float(ellipe(snV[k],q)) - N_0*(k*dx-v*t)/v for k in range(K)]

    dV = [coef3*dn[k]**2 - N_0/v for k in range(K)]

    return R,I,N,Nt,V,dV

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
        dis += abs(a[i]-b[i])**2*dx
    return dis**0.5

def true_invariants(Emax,eps):
    L = 20
    Param = parameters(L,1,Emax,eps)
    Norm_before = -2; Norm = -1; Energy_before = -2; Energy = -1
    n = 0
    while abs(Norm_before-Norm) >= eps or abs(Energy_before-Energy) >= eps:
        Norm_before = Norm; Energy_before = Energy; n += 10
        K = math.floor(20*n); dx = L/K
        R0,I0,N0,Nt0,V0,dV0 = analytical_solutions(Param,0,K)
        Norm = norm(R0,dx)+norm(I0,dx)
        Energy = energy_DVDM(R0,I0,N0,V0,dx)
        print(Norm,Energy)
    return Norm,Energy

#Taylorで R,I,N の m=1 を求める
def initial_condition(Param,K,M,eps,Ntype):

    L,Emax,v,q,N_0,u,T,phi = Param
    dx = L/K; dt = T/M

    R0,I0,N0,Nt0,V0,dV0 = analytical_solutions(Param,0,K)

    print("|Nt0 - (V0)xx|:",dist(Nt0,SCD(V0,dx),dx))
    if False:
        x = np.linspace(0,L,K)
        plt.plot(x,Nt0,label="Nt,t=0")
        plt.plot(x,SCD(V0,dx),label="Vxx,t=0")
        plt.legend()
        plt.show()

    # 以下，Glasseyで使うN1の計算
    # Taylor展開
    if Ntype == 1 or Ntype == 3:
        d2N0 = SCD(N0,dx)
        dR0 = CD(R0,dx); d2R0 = SCD(R0,dx)
        dI0 = CD(I0,dx); d2I0 = SCD(I0,dx)
        N1 = [N0[k] + dt*Nt0[k] + dt**2*(0.5*d2N0[k] + dR0[k]**2 + dI0[k]**2 + R0[k]*d2R0[k] + I0[k]*d2I0[k]) for k in range(K)]

    # 解析解が既知の場合
    if Ntype == 2:
        N1 = analytical_solutions(Param,dt,K)[2]

    #DVDM
    if Ntype == 3:
        R_now,I_now,N_now,N_next,V_now = R0,I0,N0,N1,V0

        Ik = np.identity(K); Zk = np.zeros((K,K))
        Dx = (-2*Ik + np.eye(K,k=1) + np.eye(K,k=K-1) + np.eye(K,k=-1) + np.eye(K,k=-K+1))/dx**2
        ID = np.linalg.inv(Ik-0.5*dt**2*Dx)

        DR_now = np.dot(Dx,np.array(R_now)); DI_now = np.dot(Dx,np.array(I_now)); DV_now = np.dot(Dx,np.array(V_now))

        dN = 0.5*dt*Dx - 0.25*dt*np.diag(N_now)
        dR_now = 0.25*dt*np.diag(R_now); dI_now = 0.25*dt*np.diag(I_now)

        F0 = np.array([- R_now[i%K] + 0.5*dt*DI_now[i%K] if i//K == 0
            else -I_now[i%K] - 0.5*dt*DR_now[i%K] if i//K == 1
            else -N_now[i%K] - 0.5*dt*DV_now[i%K] if i//K == 2
            else - V_now[i%K] - 0.5*dt*(N_now[i%K] + R_now[i%K]**2 + I_now[i%K]**2) for i in range(4*K)])

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
        N1 = N_next

    WantToPlot = False
    if WantToPlot:
        fig = plt.figure()
        axR = fig.add_subplot(1, 4, 1)
        axI = fig.add_subplot(1, 4, 2)
        axN = fig.add_subplot(1, 4, 3)
        axV = fig.add_subplot(1, 4, 4)
        x = np.linspace(0,20,K)
        axR.plot(x,R0,label="ReE,t=0")
        axI.plot(x,I0,label="ImE,t=0")
        axN.plot(x,N0,label="N,t=0")
        axV.plot(x,V0,label="N,t=0")
        axR.legend(); axI.legend(); axN.legend(); axV.legend()
        plt.show()

    WantToCheck = False
    if WantToCheck:
        Ik = np.identity(K)
        Dx = (1/dx**2)*(-2*Ik + np.eye(K,k=1) + np.eye(K,k=K-1) + np.eye(K,k=-1) + np.eye(K,k=-K+1))
        ID = np.linalg.inv(Ik-0.5*dt**2*Dx)
        Dn = np.diag([N0[k]+N1[k] for k in range(K)])
        D = dt*(0.5*Dx - 0.25*Dn)
        A = np.block([[Ik,D],[-D,Ik]])
        b = np.linalg.solve(A,np.array([2*R0[i] if i < K else 2*I0[i-K] for i in range(2*K)]))
        R1 = -np.array(R0) + b[:K]
        I1 = -np.array(I0) + b[K:]

        if Ntype == 1:
            l = "G:"
        if Ntype == 2:
            l = "GT:"
        if Ntype == 3:
            l = "GD:"

        DD = -2*np.eye(K-1,k=0) + np.eye(K-1,k=1) + np.eye(K-1,k=-1)
        DDI = np.linalg.inv(DD)
        dN = [(N1[k] - N0[k])/dt for k in range(1,K)]
        VG = dx**2 * np.dot(DDI,dN)
        VG = [0]+[VG[i] for i in range(K-1)]
        V0 = [V0[i]-V0[0] for i in range(K)]
        x = np.linspace(0,L,K)
        #plt.plot(x,Nt0,label="Nt,t=0")
        plt.plot(x,V0,label="V_0,t=0")
        plt.plot(x,VG,label="V_G,t=0")
        plt.legend()
        plt.show()

        print("||VG-V0||:",dist(VG,V0,dx))
        print("dN,VGxx:",(N1[0]-N0[0])/dt,(VG[1]+VG[-1])/(dx**2))
        dR1 = FD(R1,dx); dI1 = FD(I1,dx);dVG = FD(VG,dx)
        innerG = 0
        for i in range(K):
            innerG += 0.5*(N0[i]+N1[i])*(R1[i]**2 + I1[i]**2)*dx
        print(l+"Glassey")
        print("|dE|^2:"+str(norm(dR1,dx) + norm(dI1,dx)))
        print("|N|^2:"+str(0.25*norm(N0,dx) + 0.25*norm(N1,dx)))
        print("|dV|^2:"+str(0.5*norm(dVG,dx)))
        print("(N,|E|^2):"+str(innerG))
        #EnergyG = energy_Glassey(R1,I1,N0,N1,DDI,dt,dx)
        dR0 = FD(R0,dx); dI0 = FD(I0,dx); dV0 = FD(V0,dx)
        innerD = 0
        for i in range(len(R0)):
            innerD += N0[i]*(R0[i]**2 + I0[i]**2)*dx
        print("DVDM")
        print("|dE|^2:"+str(norm(dR0,dx) + norm(dI0,dx)))
        print("|N|^2:"+str(0.5*norm(N0,dx)))
        print("|dV|^2:"+str(0.5*norm(dV0,dx)))
        print("(N,|E|^2):"+str(innerD))
        #EnergyD = energy_DVDM(R0,I0,N0,V0,dx)
        #print(l+str(EnergyG)+",D:"+str(EnergyD))

    return R0,I0,N0,N1,V0,dV0

def initial_condition_solitons(Emax,K,M,eps,Ntype,Vtype):
    L,Emax,v,q,N_0,u,T,phi = parameters(20,1,Emax,10**(-8))
    K0 = 8*K
    Khalf = math.floor(0.5*K)
    K1 = math.floor(3.5*K)
    K2 = K1 + K
    dt = T/M; dx = L/K

    vv = (1 - v*v)**0.5; vv2 = 1 - v*v; WW = Emax/(2**0.5*vv); coef = -2**0.5*Emax**3*q*v/vv**3
    Kq = ellipk(q); coef2 = 2**0.5*v*Emax/vv
    W = [WW*(k*dx-10) for k in range(K)]
    dn = [float(ellipfun('dn',W[k],q)) for k in range(K)]
    F = [Emax*dn[k] for k in range(K)]

    Rbase1 = [F[k]*math.cos(phi*(k*dx-10)) for k in range(K)]
    Ibase1 = [F[k]*math.sin(phi*(k*dx-10)) for k in range(K)]
    Rbase2 = [Rbase1[k] for k in range(K)]
    Ibase2 = [-Ibase1[k] for k in range(K)]

    Nbase = [-F[k]**2/vv2 + N_0 for k in range(K)]

    R0 = [Rbase1[0] for k in range(3*K)]+Rbase1+Rbase2+[Rbase1[0] for k in range(3*K)]
    I0 = [Ibase1[0] for k in range(3*K)]+Ibase1+Ibase2+[Ibase1[0] for k in range(3*K)]
    N0 = [Nbase[0] for k in range(3*K)]+Nbase+Nbase+[Nbase[0] for k in range(3*K)]

    if Vtype == 1 or Vtype == 3:

        snV = [float(asin(ellipfun('sn',W[k],q))) if -Kq < W[k] <= Kq
         else math.pi - float(asin(ellipfun('sn',2*Kq - W[k],q))) if W[k] > Kq
          else float(asin(ellipfun('sn',W[k] + 2*Kq,q))) - math.pi for k in range(K)]

        if Vtype == 1:
            Vbase1 = [coef2*float(ellipe(snV[k],q)) - N_0*k*dx/v for k in range(K)]
            Vbase1 = [Vbase1[k] - Vbase1[0] for k in range(K)]
            Vbase2 = [-Vbase1[k] for k in range(K)]

            V0 = [Vbase1[0] for k in range(3*K)]+Vbase1+Vbase2+[Vbase1[0] for k in range(3*K)]

        if Vtype == 3:
            Vbase1 = [coef2*float(ellipe(snV[k],q)) + N_0*L/(2*v) for k in range(K)]
            Vbase2 = [-coef2*float(ellipe(snV[k],q)) + N_0*L/(2*v)  for k in range(K)]

            V0 = [Vbase1[0] for k in range(3*K)]+Vbase1+Vbase2+[Vbase1[0] for k in range(3*K)]

        #Taylor
        if Ntype == 1 or Ntype == 3:
            Ntbase1 = [float(coef*dn[k]*ellipfun('sn',W[k],q)*ellipfun('cn',W[k],q)) for k in range(K)]
            Ntbase2 = [-Ntbase1[k] for k in range(K)]
            Ntbase = [Ntbase1[0] for k in range(3*K)]+Ntbase1+Ntbase2+[Ntbase1[0] for k in range(3*K)]

            d2N = SCD(Nbase,dx)
            dR1 = CD(Rbase1,dx); d2R1 = SCD(Rbase1,dx)
            dI1 = CD(Ibase1,dx); d2I1 = SCD(Ibase1,dx)
            N1b1 = [Nbase[k] + dt*Ntbase1[k] + dt**2*(0.5*d2N[k] + dR1[k]**2 + dI1[k]**2 + Rbase1[k]*d2R1[k] + Ibase1[k]*d2I1[k]) for k in range(K)]

            dR2 = CD(Rbase2,dx); d2R2 = SCD(Rbase2,dx)
            dI2 = CD(Ibase2,dx); d2I2 = SCD(Ibase2,dx)
            N1b2 = [Nbase[k] + dt*Ntbase2[k] + dt**2*(0.5*d2N[k] + dR2[k]**2 + dI2[k]**2 + Rbase2[k]*d2R2[k] + Ibase2[k]*d2I2[k]) for k in range(K)]

            N1 = [N1b1[0] for k in range(3*K)]+N1b1+N1b2+[N1b1[0] for k in range(3*K)]
            #print(dist(Ntbase,SCD(V0,dx),dx))

        #1ソリトンでの解析解が既知で，2ソリトンでも(t=Δtでは)正しいと仮定
        if Ntype == 2:
            Wdt1 = [WW*(k*dx-10-v*dt) for k in range(K)]
            Wdt2 = [WW*(k*dx-10+v*dt) for k in range(K)]
            dndt1 = [float(ellipfun('dn',Wdt1[k],q)) for k in range(K)]
            dndt2 = [float(ellipfun('dn',Wdt2[k],q)) for k in range(K)]
            Fdt1 = [Emax*dndt1[k] for k in range(K)]
            Fdt2 = [Emax*dndt2[k] for k in range(K)]

            Ndtbase1 = [-Fdt1[k]**2/vv2 + N_0 for k in range(len(W))]
            Ndtbase2 = [-Fdt2[k]**2/vv2 + N_0 for k in range(len(W))]

            N1 = [Ndtbase1[0] for k in range(3*K)]+Ndtbase1+Ndtbase2+[Ndtbase2[-1] for k in range(3*K)]

        #Vに基づくNt
        if Ntype == 3:
            R_now,I_now,N_now,N_next,V_now = R0,I0,N0,N1,V0

            K *= 8

            Ik = np.identity(K); Zk = np.zeros((K,K))
            Dx = (-2*Ik + np.eye(K,k=1) + np.eye(K,k=K-1) + np.eye(K,k=-1) + np.eye(K,k=-K+1))/dx**2
            ID = np.linalg.inv(Ik-0.5*dt**2*Dx)

            DR_now = np.dot(Dx,np.array(R_now)); DI_now = np.dot(Dx,np.array(I_now)); DV_now = np.dot(Dx,np.array(V_now))

            dN = 0.5*dt*Dx - 0.25*dt*np.diag(N_now)
            dR_now = 0.25*dt*np.diag(R_now); dI_now = 0.25*dt*np.diag(I_now)

            F0 = np.array([- R_now[i%K] + 0.5*dt*DI_now[i%K] if i//K == 0
                else -I_now[i%K] - 0.5*dt*DR_now[i%K] if i//K == 1
                else -N_now[i%K] - 0.5*dt*DV_now[i%K] if i//K == 2
                else - V_now[i%K] - 0.5*dt*(N_now[i%K] + R_now[i%K]**2 + I_now[i%K]**2) for i in range(4*K)])

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
            print("N:",dist(N1,N_next,dx))
            N1 = N_next
            K = K//8

    if Vtype == 2:
        Ntbase1 = [float(coef*dn[k]*ellipfun('sn',W[k],q)*ellipfun('cn',W[k],q)) for k in range(K)]
        Ntbase2 = [-Ntbase1[k] for k in range(K)]
        Ntbase = [Ntbase1[0] for k in range(3*K)]+Ntbase1+Ntbase2+[Ntbase1[0] for k in range(3*K)]

        d2N = SCD(Nbase,dx)
        dR1 = CD(Rbase1,dx); d2R1 = SCD(Rbase1,dx)
        dI1 = CD(Ibase1,dx); d2I1 = SCD(Ibase1,dx)
        N1b1 = [Nbase[k] + dt*Ntbase1[k] + dt**2*(0.5*d2N[k] + dR1[k]**2 + dI1[k]**2 + Rbase1[k]*d2R1[k] + Ibase1[k]*d2I1[k]) for k in range(K)]

        dR2 = CD(Rbase2,dx); d2R2 = SCD(Rbase2,dx)
        dI2 = CD(Ibase2,dx); d2I2 = SCD(Ibase2,dx)
        N1b2 = [Nbase[k] + dt*Ntbase2[k] + dt**2*(0.5*d2N[k] + dR2[k]**2 + dI2[k]**2 + Rbase2[k]*d2R2[k] + Ibase2[k]*d2I2[k]) for k in range(K)]

        N1 = [N1b1[0] for k in range(3*K)]+N1b1+N1b2+[N1b1[0] for k in range(3*K)]

        DD = -2*np.eye(8*K-1,k=0) + np.eye(8*K-1,k=1) + np.eye(8*K-1,k=-1)
        DDI = np.linalg.inv(DD)
        dN = [(N1[k] - N0[k])/dt for k in range(1,8*K)]
        V0 = dx**2 * np.dot(DDI,dN)
        V0 = [0]+[V0[i] for i in range(8*K-1)]

    WantToPlot = True
    if WantToPlot:
        x = np.linspace(0,160,8*K)
        plt.plot(x,N0,label="N,t=0")
        plt.plot(x,V0,label="V,t=0")
        plt.legend()
        plt.show()

    WantToCheck = True
    if WantToCheck:
        Ik = np.identity(K0)
        Dx = (1/dx**2)*(-2*Ik + np.eye(K0,k=1) + np.eye(K0,k=K0-1) + np.eye(K0,k=-1) + np.eye(K0,k=-K0+1))
        ID = np.linalg.inv(Ik-0.5*dt**2*Dx)
        Dn = np.diag([N0[k]+N1[k] for k in range(K0)])
        D = dt*(0.5*Dx - 0.25*Dn)
        A = np.block([[Ik,D],[-D,Ik]])
        b = np.linalg.solve(A,np.array([2*R0[i] if i < K0 else 2*I0[i-K0] for i in range(2*K0)]))
        R1 = -np.array(R0) + b[:K0]
        I1 = -np.array(I0) + b[K0:]

        if Ntype == 1:
            l = "G:"
        if Ntype == 2:
            l = "GT:"
        if Ntype == 3:
            l = "GD:"
        if Ntype == 4:
            l = "Gupdate:"
        K = len(R1)
        DD = -2*np.eye(K-1,k=0) + np.eye(K-1,k=1) + np.eye(K-1,k=-1)
        DDI = np.linalg.inv(DD)
        dN = [(N1[k] - N0[k])/dt for k in range(K)]
        VG = dx**2 * np.dot(DDI,dN[1:])
        VG = [0]+[VG[i] for i in range(K-1)]
        if True:
            x = np.linspace(0,160,K)
            fig = plt.figure()
            fig1 = fig.add_subplot(2, 1, 1)
            fig2 = fig.add_subplot(2, 1, 2)
            fig1.plot(x,Ntbase,label="Nt,t=0")
            fig1.plot(x,SCD(V0,dx),label="V0_xx,t=0")
            fig1.plot(x,SCD(VG,dx),label="VG_xx,t=0")
            fig1.legend()
            fig2.plot(x,Ntbase,label="Nt,t=0")
            fig2.plot(x,dN,label="dN,t=0")
            fig2.plot(x,V0,label="V_0,t=0")
            fig2.plot(x,VG,label="V_G,t=0")
            fig2.legend()
            plt.show()
        print(dN)
        print(SCD(VG,dx))
        V0 = [V0[i]-V0[0] for i in range(K)]
        print("V:",dist(VG,V0,dx))
        print("dN,ddV:",(N1[0]-N0[0])/dt,(VG[1]+VG[-1])/(dx**2))
        dR1 = FD(R1,dx); dI1 = FD(I1,dx);dVG = FD(VG,dx)
        innerG = 0
        for i in range(K):
            innerG += 0.5*(N0[i]+N1[i])*(R1[i]**2 + I1[i]**2)*dx
        print(l+"Glassey")
        print("|dE|^2:"+str(norm(dR1,dx) + norm(dI1,dx)))
        print("|N|^2:"+str(0.25*norm(N0,dx) + 0.25*norm(N1,dx)))
        print("|dV|^2:"+str(0.5*norm(dVG,dx)))
        print("(N,|E|^2):"+str(innerG))
        #EnergyG = energy_Glassey(R1,I1,N0,N1,DDI,dt,dx)
        dR0 = FD(R0,dx); dI0 = FD(I0,dx); dV0 = FD(V0,dx)
        innerD = 0
        for i in range(len(R0)):
            innerD += N0[i]*(R0[i]**2 + I0[i]**2)*dx
        print("DVDM")
        print("|dE|^2:"+str(norm(dR0,dx) + norm(dI0,dx)))
        print("|N|^2:"+str(0.5*norm(N0,dx)))
        print("|dV|^2:"+str(0.5*norm(dV0,dx)))
        print("(N,|E|^2):"+str(innerD))
        #EnergyD = energy_DVDM(R0,I0,N0,V0,dx)
        #print(l+str(EnergyG)+",D:"+str(EnergyD))

    return R0,I0,N0,N1,V0

#Emax = 0.1; n = 10; Param = parameters(20,1,Emax,10**(-8)); T = Param[-2]; K = math.floor(20*n); M = math.floor(T*n)
#initial_condition(Param,K,M,10**(-8),1)
#initial_condition_solitons(Emax,K,M,10**(-8),1,3)
###############################################################################
#スキーム本体

# Glassey スキーム
def Glassey(Param,K,M,Ntype,Stype,Vtype):
    L = Param[0]; T = Param[-2]
    start = time.perf_counter()
    dx = L/K; dt = T/M #print(dt,dx)

    # 数値解の記録
    Rs = []; Is = []; Ns = []
    if Stype == 1:
        R_now,I_now,N_now,N_next = initial_condition(Param,K,M,10**(-8),Ntype)[:4]
    if Stype == 2:
        R_now,I_now,N_now,N_next = initial_condition_solitons(Param[1],K,M,10**(-8),Ntype,Vtype)[:4]
        K *= 8
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
            b = np.linalg.solve(A,np.array([2*R_now[i] if i < K else 2*I_now[i-K] for i in range(2*K)]))
            R_next = -np.array(R_now) + b[:K]
            I_next = -np.array(I_now) + b[K:]
            Rs.append(R_next); Is.append(I_next)
            R_now = R_next; I_now = I_next
            ri_t += 1
            if ri_t%10 == 0:
                print(ri_t,n_t,M) #実行の進捗の目安として
        else: # N(m-1),Nm,Em から N(m+1)を求める
            N_before = N_now; N_now = N_next
            E = np.array([R_now[k]**2 + I_now[k]**2 for k in range(K)])
            NN = np.array(N_now) + E
            N_next = np.dot(ID,2*NN) - np.array(N_before) - 2*E
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
        print("保存量初期値:",Norm[0],Energy[0])

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

    return [[str(end-start)]+[0 for i in range(len(Rs[0])-1)]],Rs,Is,Ns

def checking_Glassey(L,Emax,n,Ntype):
    Param = parameters(L,1,Emax,eps)
    T = Param[-2]
    K = math.floor(L*n); M = math.floor(T*n)
    dx = L/K; dt = T/M

    Rs,Is,Ns = [],[],[]

    if Ntype == 1:
        fname = "L="+str(L)+"Emax="+str(Emax)+"N="+str(n)+"Glassey.csv"
    if Ntype == 2:
        fname = "L="+str(L)+"Emax="+str(Emax)+"N="+str(n)+"GlasseyA.csv"
    if Ntype == 3:
        fname = "L="+str(L)+"Emax="+str(Emax)+"N="+str(n)+"GlasseyD.csv"

    if not os.path.isfile(fname):
        time,Rs,Is,Ns = Glassey(Param,K,M,Ntype,1,False)
        pd.DataFrame(time+Rs+Is+Ns).to_csv(fname)
    with open(fname) as f:
        for row in csv.reader(f, quoting=csv.QUOTE_NONNUMERIC):
            if row[0] in [i+1 for i in range(M+1)]:
                Rs.append(np.array(row[1:]))
            if row[0] in [M+2+i for i in range(M+1)]:
                Is.append(np.array(row[1:]))
            if row[0] in [2*M+3+i for i in range(M+1)]:
                Ns.append(np.array(row[1:]))

    tRs,tIs,tNs,tVs = [],[],[],[]
    fname = "L="+str(L)+"Emax="+str(Emax)+"N="+str(n)+"Analytic.csv"

    if os.path.isfile(fname):
        with open(fname) as f:
            for row in csv.reader(f, quoting=csv.QUOTE_NONNUMERIC):
                if row[0] in [i for i in range(M+1)]:
                    tRs.append(np.array(row[1:]))
                if row[0] in [M+1+i for i in range(M+1)]:
                    tIs.append(np.array(row[1:]))
                if row[0] in [2*M+2+i for i in range(M+1)]:
                    tNs.append(np.array(row[1:]))
    else:
        for i in range(M+1):
            if i%10 == 0:
                print(i,M)
            tR,tI,tN,_,tV,_ = analytical_solutions(Param,i*dt,K)
            tRs.append(tR); tIs.append(tI); tNs.append(tN); tVs.append(tV)
        pd.DataFrame(tRs+tIs+tNs+tVs).to_csv(fname)

    eEs = [];eNs = []
    rEs = [];rNs = []

    RANGE = [i for i in range(len(Rs))]
    #RANGE = [len(Rs)-1] # 最終時刻での誤差だけ知りたいとき
    for i in RANGE:
        tnorm = (norm(tRs[i],dx) + norm(tIs[i],dx))**0.5

        eEs.append((dist(Rs[i],tRs[i],dx)**2+dist(Is[i],tIs[i],dx)**2)**0.5); eNs.append(dist(Ns[i],tNs[i],dx))
        #print(T*i/M,eEs[-1],eNs[-1])
        rEs.append(eEs[i]/tnorm); rNs.append(eNs[i]/(norm(tNs[i],dx)**0.5))
    print("各要素の最大誤差:",max(eEs),max(eNs))
    print("各要素の最大誤差比:",max(rEs),max(rNs))
    for i in range(4):
        j = math.floor(T*(i+1)/(4*dt))
        print("t:",32*(i+1)/4,",",eEs[j],eNs[j],",",rEs[j],rNs[j])
    return (dx**2 + dt**2)**0.5,eEs,eNs

# DVDMスキーム本体
# Newton法の初期値をGlasseyで求める
def DVDM_Glassey(Param,K,M,eps,Stype,Vtype):
    L = Param[0]; T = Param[-2]
    start = time.perf_counter()
    dx = L/K; dt = T/M #print(dt,dx)

    # 数値解の記録
    Rs = []; Is = []; Ns = []; Vs = []
    if Stype == 1:
        R_now,I_now,N_now,N_next,V_now = initial_condition(Param,K,M,eps,1)[:-1]
    if Stype == 2:
        R_now,I_now,N_now,N_next,V_now = initial_condition_solitons(Param[1],K,M,eps,1,Vtype)
        K *= 8
    Rs.append(R_now); Is.append(I_now); Ns.append(N_now); Vs.append(V_now)

    m = 0
    Ik = np.identity(K); Zk = np.zeros((K,K))
    Dx = (-2*Ik + np.eye(K,k=1) + np.eye(K,k=K-1) + np.eye(K,k=-1) + np.eye(K,k=-K+1))/dx**2
    ID = np.linalg.inv(Ik-0.5*dt**2*Dx)

    DR_now = np.dot(Dx,np.array(R_now)); DI_now = np.dot(Dx,np.array(I_now)); DV_now = np.dot(Dx,np.array(V_now))

    while m*dt < T:
        t = 0
        dN = 0.5*dt*Dx - 0.25*dt*np.diag(N_now)
        dR_now = 0.25*dt*np.diag(R_now); dI_now = 0.25*dt*np.diag(I_now)

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

    WantToKnow = True #ノルム・エネルギーを知りたいとき
    WantToPlot = False #ノルム・エネルギーを描画したいとき
    if WantToKnow:
        Norm = [norm(Rs[i],dx) + norm(Is[i],dx) for i in range(len(Rs))]
        dNorm = [abs(Norm[i] - Norm[0]) for i in range(len(Rs))]
        rNorm = [dNorm[i]/Norm[0] for i in range(len(Rs))]

        Energy = [energy_DVDM(Rs[i],Is[i],Ns[i],Vs[i],dx) for i in range(len(Rs))]
        dEnergy = [abs(Energy[i] - Energy[0]) for i in range(len(Rs))]
        rEnergy = [dEnergy[i]/abs(Energy[0]) for i in range(len(Rs))]

        print("保存量初期値:",Norm[0],Energy[0])

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
    return [[str(end-start)]+[0 for i in range(len(Rs[0])-1)]],Rs,Is,Ns,Vs

def checking_DVDM(L,Emax,n,eps):
    Param = parameters(L,1,Emax,eps)
    T = Param[-2]
    K = math.floor(L*n); M = math.floor(T*n)
    dx = L/K; dt = T/M

    Rs,Is,Ns,Vs = [],[],[],[]
    fname = "L="+str(L)+"Emax="+str(Emax)+"N="+str(n)+"DVDM.csv"
    if not os.path.isfile(fname):
        time,Rs,Ns,Is,Vs = DVDM_Glassey(Param,K,M,eps,1)
        pd.DataFrame(time+Rs+Is+Ns+Vs).to_csv(fname)
    with open(fname) as f:
        for row in csv.reader(f, quoting=csv.QUOTE_NONNUMERIC):
            if row[0] in [i+1 for i in range(M+1)]:
                Rs.append(np.array(row[1:]))
            if row[0] in [M+2+i for i in range(M+1)]:
                Is.append(np.array(row[1:]))
            if row[0] in [2*M+3+i for i in range(M+1)]:
                Ns.append(np.array(row[1:]))
            if row[0] in [3*M+4+i for i in range(M+1)]:
                Vs.append(np.array(row[1:]))

    tRs,tIs,tNs,tVs = [],[],[],[]
    fname = "L="+str(L)+"Emax="+str(Emax)+"N="+str(n)+"Analytic.csv"

    if not os.path.isfile(fname):
        for i in range(M+1):
            if i%10 == 0:
                print(i,M)
            tR,tI,tN,_,tV,_ = analytical_solutions(Param,i*dt,K)
            tRs.append(tR); tIs.append(tI); tNs.append(tN); tVs.append(tV)
        pd.DataFrame(tRs+tIs+tNs+tVs).to_csv(fname)
    else:
        with open(fname) as f:
            for row in csv.reader(f, quoting=csv.QUOTE_NONNUMERIC):
                if row[0] in [i for i in range(M+1)]:
                    tRs.append(np.array(row[1:]))
                if row[0] in [M+1+i for i in range(M+1)]:
                    tIs.append(np.array(row[1:]))
                if row[0] in [2*M+2+i for i in range(M+1)]:
                    tNs.append(np.array(row[1:]))
                if row[0] in [3*M+3+i for i in range(M+1)]:
                    tVs.append(np.array(row[1:]))

    eEs = []; eNs = []; eVs = []
    rEs = []; rNs = []; rVs = []

    RANGE = [i for i in range(len(Rs))]
    #RANGE = [len(Rs)-1] # 最終時刻での誤差だけ知りたいとき
    for i in RANGE:
        tnorm = (norm(tRs[i],dx) + norm(tIs[i],dx))**0.5

        eEs.append((dist(Rs[i],tRs[i],dx)**2+dist(Is[i],tIs[i],dx)**2)**0.5); eNs.append(dist(Ns[i],tNs[i],dx)); eVs.append(dist(Vs[i],tVs[i],dx))
        rEs.append(eEs[i]/tnorm); rNs.append(eNs[i]/(norm(tNs[i],dx)**0.5)); rVs.append(eVs[i]/(norm(tVs[i],dx)**0.5))
        if i%10 == 0:
            print(i,M)
    print("各要素の最大誤差:",max(eEs),max(eNs),max(eVs))
    print("各要素の最大誤差比:",max(rEs),max(rNs),max(rVs))
    for i in range(4):
        j = math.floor(T*(i+1)/(4*dt))
        print("t:",32*(i+1)/4,",",eEs[j],eNs[j],eVs[j],",",rEs[j],rNs[j],rVs[j])
    return (dx**2 + dt**2)**0.5,eEs,eNs,eVs

# Glassey,DVDM,解析解を T/3 ごとに比較
def comparing(L,Emax,n,eps,times):
    Param = parameters(L,1,Emax,eps)
    T = Param[-2]
    K = math.floor(L*n); M = math.floor(T*n)
    dx = L/K; dt = T/M

    if n == 10:
        tNorm,tEnergy = true_invariants(Emax,10**(-3))
        print("保存量真値:",tNorm,tEnergy)

    RG,IG,NG = [],[],[]
    RGD,IGD,NGD = [],[],[]
    RD,ID,ND,VD = [],[],[],[]
    Rindex = [i*M//times+1 for i in range(times+1)]
    Iindex = [M+2+i*M//times for i in range(times+1)]
    Nindex = [2*M+3+i*M//times for i in range(times+1)]
    Vindex = [3*M+4+i*M//times for i in range(times+1)]

    fname = "L="+str(L)+"Emax="+str(Emax)+"N="+str(n)+"Glassey.csv"

    if not os.path.isfile(fname):
        time,Rs,Is,Ns = Glassey(Param,K,M,1,1,1)
        pd.DataFrame(time+Rs+Is+Ns).to_csv(fname)
    with open(fname) as f:
        for row in csv.reader(f, quoting=csv.QUOTE_NONNUMERIC):
            if row[0] in Rindex:
                RG.append(np.array(row[1:]))
            if row[0] in Iindex:
                IG.append(np.array(row[1:]))
            if row[0] in Nindex:
                NG.append(np.array(row[1:]))

    fname = "L="+str(L)+"Emax="+str(Emax)+"N="+str(n)+"GlasseyD.csv"
    if not os.path.isfile(fname):
        time,Rs,Is,Ns = Glassey(Param,K,M,3,1,1)
        pd.DataFrame(time+Rs+Is+Ns).to_csv(fname)
    with open(fname) as f:
        for row in csv.reader(f, quoting=csv.QUOTE_NONNUMERIC):
            if row[0] in Rindex:
                RGD.append(np.array(row[1:]))
            if row[0] in Iindex:
                IGD.append(np.array(row[1:]))
            if row[0] in Nindex:
                NGD.append(np.array(row[1:]))

    fname = "L="+str(L)+"Emax="+str(Emax)+"N="+str(n)+"DVDM.csv"
    if not os.path.isfile(fname):
        time,Rs,Ns,Is,Vs = DVDM_Glassey(Param,K,M,eps,1,1)
        pd.DataFrame(time+Rs+Ns+Is+Vs).to_csv(fname)
    with open(fname) as f:
        for row in csv.reader(f, quoting=csv.QUOTE_NONNUMERIC):
            if row[0] in Rindex:
                RD.append(np.array(row[1:]))
            if row[0] in Iindex:
                ID.append(np.array(row[1:]))
            if row[0] in Nindex:
                ND.append(np.array(row[1:]))
            if row[0] in Vindex:
                VD.append(np.array(row[1:]))

    x = np.linspace(0, L, K)

    fig = plt.figure()
    axs = []
    for i in range(times):
        axs.append(fig.add_subplot(times, 4, 4*i+1))
        axs.append(fig.add_subplot(times, 4, 4*i+2))
        axs.append(fig.add_subplot(times, 4, 4*i+3))
        axs.append(fig.add_subplot(times, 4, 4*i+4))

    for i in range(times):
        t = (i+1)*M//times
        tR,tI,tN,_,tV = analytical_solutions(Param,t*dt,K)[:5]

        ax = axs[4*i:4*i+4]
        l11,l12,l2,l3 = "G","G+D","D","A"

        ax[0].plot(x, RG[i+1], label=l11)
        ax[0].plot(x, RGD[i+1], label=l12)
        ax[0].plot(x, RD[i+1], label=l2)
        ax[0].plot(x, tR, label=l3)
        ax[0].set_ylabel("ReE")
        ax[1].plot(x, IG[i+1], label=l11)
        ax[1].plot(x, IGD[i+1], label=l12)
        ax[1].plot(x, ID[i+1], label=l2)
        ax[1].plot(x, tI, label=l3)
        ax[1].set_ylabel("ImE")
        ax[2].plot(x, NG[i+1], label=l11)
        ax[2].plot(x, NGD[i+1], label=l12)
        ax[2].plot(x, ND[i+1], label=l2)
        ax[2].plot(x, tN, label=l3)
        ax[2].set_ylabel("N")
        ax[3].plot(x, VD[i+1], label=l2)
        ax[3].plot(x, tV, label=l3)
        ax[3].set_ylabel("V")
        ax[0].legend(); ax[1].legend(); ax[2].legend(); ax[3].legend()
    plt.show()

#print(checking_Glassey(20,5,80,1))
#print(checking_Glassey(20,5,80,3))
#print(checking_DVDM(20,5,80,10**(-8)))
#comparing(20,5,80,10**(-8),4)


###############################################################################
#収束先の検証

def comp_nsGlassey(L,Emax,N1,N2,Ntype):
    m = 1; eps = 10**(-9)
    Param = parameters(L,m,Emax,eps)
    T = Param[-2]
    Rf,If,Nf = [],[],[]
    numbers = [N1,N2]
    if Ntype == 1:
        Gname = "Glassey.csv"
    if Ntype == 2:
        Gname == "GlasseyA.csv"
    if Ntype == 3:
        Gname == "GlasseyD.csv"
    fnames = ["L="+str(L)+"Emax="+str(Emax)+"N="+str(numbers[i])+Gname for i in range(2)]
    for i in range(2):
        n = numbers[i]
        fname = fnames[i]
        K = math.floor(L*n); M = math.floor(T*n)
        if not os.path.isfile(fname):
            time,Rs,Ns,Is = Glassey(Param,K,M,Ntype,1,1)
            pd.DataFrame(time+Rs+Ns+Is).to_csv(fname)
        with open(fname) as f:
            for row in csv.reader(f, quoting=csv.QUOTE_NONNUMERIC):
                if row[0] == M+1:
                    Rf.append(np.array(row[1:]))
                if row[0] == 2*M+2:
                    If.append(np.array(row[1:]))
                if row[0] == 3*M+3:
                    Nf.append(np.array(row[1:]))
    gx = math.gcd(len(Rf[0]),len(Rf[1]))
    lx = [int(len(Rf[0])/gx),int(len(Rf[1])/gx)]

    error = 0
    dx = L/gx
    for k in range(gx):
        error += (Rf[0][k*lx[0]] - Rf[1][k*lx[1]])**2*dx
        error += (If[0][k*lx[0]] - If[1][k*lx[1]])**2*dx
        error += (Nf[0][k*lx[0]] - Nf[1][k*lx[1]])**2*dx
    return error**0.5

def conv_nsGlassey(L,Emax,n,Ntype):
    m = 1; eps = 10**(-9)
    Param = parameters(L,m,Emax,eps)
    T = Param[-2]
    if Ntype == 1:
        fname = "L="+str(L)+"Emax="+str(Emax)+"N="+str(n)+"Glassey.csv"
    if Ntype == 2:
        fname = "L="+str(L)+"Emax="+str(Emax)+"N="+str(n)+"GlasseyA.csv"
    if Ntype == 3:
        fname = "L="+str(L)+"Emax="+str(Emax)+"N="+str(n)+"GlasseyD.csv"

    K = math.floor(L*n); M = math.floor(T*n)
    if not os.path.isfile(fname):
        time,Rs,Ns,Is = Glassey(Param,K,M,Ntype,1,1)
        pd.DataFrame(time+Rs+Ns+Is).to_csv(fname)
    with open(fname) as f:
        for row in csv.reader(f, quoting=csv.QUOTE_NONNUMERIC):
            if row[0] == M+1:
                Rf = np.array(row[1:])
            if row[0] == 2*M+2:
                If = np.array(row[1:])
            if row[0] == 3*M+3:
                Nf = np.array(row[1:])

    error = 0
    dx = L/K
    tR,tI,tN = analytical_solutions(Param,T,K)[:3]
    return (dist(Rf,tR,dx)**2 + dist(If,tI,dx)**2 + dist(Nf,tN,dx)**2)**0.5

def comp_nsDVDM(L,Emax,N1,N2):
    m = 1; eps = 10**(-9)
    Param = parameters(L,m,Emax,eps)
    T = Param[-2]
    Rf,If,Nf = [],[],[]
    numbers = [N1,N2]
    fnames = ["L="+str(L)+"Emax="+str(Emax)+"N="+str(numbers[i])+"DVDM.csv" for i in range(2)]
    for i in range(2):
        n = numbers[i]
        fname = fnames[i]
        K = math.floor(L*n); M = math.floor(T*n)
        if not os.path.isfile(fname):
            time,Rs,Ns,Is,Vs = DVDM_Glassey(Param,K,M,eps,1,1)
            pd.DataFrame(time+Rs+Ns+Is+Vs).to_csv(fname)
        with open(fname) as f:
            for row in csv.reader(f, quoting=csv.QUOTE_NONNUMERIC):
                if row[0] == M+1:
                    Rf.append(np.array(row[1:]))
                if row[0] == 2*M+2:
                    If.append(np.array(row[1:]))
                if row[0] == 3*M+3:
                    Nf.append(np.array(row[1:]))
    gx = math.gcd(len(Rf[0]),len(Rf[1]))
    lx = [int(len(Rf[0])/gx),int(len(Rf[1])/gx)]

    error = 0
    dx = L/gx
    for k in range(gx):
        error += (Rf[0][k*lx[0]] - Rf[1][k*lx[1]])**2*dx
        error += (If[0][k*lx[0]] - If[1][k*lx[1]])**2*dx
        error += (Nf[0][k*lx[0]] - Nf[1][k*lx[1]])**2*dx
    return error**0.5

def conv_nsDVDM(L,Emax,n):
    m = 1; eps = 10**(-9)
    Param = parameters(L,m,Emax,eps)
    T = Param[-2]
    fname = "L="+str(L)+"Emax="+str(Emax)+"N="+str(n)+"DVDM.csv"
    K = math.floor(L*n); M = math.floor(T*n)
    if not os.path.isfile(fname):
        time,Rs,Ns,Is,Vs = DVDM_Glassey(Param,K,M,eps,1,1)
        pd.DataFrame(time+Rs+Ns+Is+Vs).to_csv(fname)
    with open(fname) as f:
        for row in csv.reader(f, quoting=csv.QUOTE_NONNUMERIC):
            if row[0] == M+1:
                Rf = np.array(row[1:])
            if row[0] == 2*M+2:
                If = np.array(row[1:])
            if row[0] == 3*M+3:
                Nf = np.array(row[1:])

    error = 0
    dx = L/K
    tR,tI,tN = analytical_solutions(Param,T,K)
    return (dist(Rf,tR,dx)**2 + dist(If,tI,dx)**2 + dist(Nf,tN,dx)**2)**0.5

#print(comp_nsGlassey(20,2.1,50,60))
#print(conv_nsGlassey(20,1,30))
#print(comp_nsDVDM(20,2.1,30,40))
#print(conv_nsDVDM(20,1,30))

###############################################################################
#2ソリトン衝突

#NtとVをそれぞれ1ソリトン解の場合の継ぎはぎにした場合
def comparing_solitons(Emax,n,times):
    Param = parameters(20,1,Emax,10**(-8))
    T = Param[-2]
    K = math.floor(20*n); M = math.floor(T*n)
    dx = 20/K; dt = T/M

    RG,IG,NG = [],[],[]
    RGD,IGD,NGD = [],[],[]
    RD,ID,ND,VD = [],[],[],[]

    fname = "Emax="+str(Emax)+"N="+str(n)+"GlasseyS.csv"

    if not os.path.isfile(fname):
        time,Rs,Is,Ns = Glassey(Param,K,M,1,2,1)
        pd.DataFrame(time+Rs+Is+Ns).to_csv(fname)
    with open(fname) as f:
        for row in csv.reader(f, quoting=csv.QUOTE_NONNUMERIC):
            if row[0] in [i*M//times+1 for i in range(times+1)]:
                RG.append(np.array(row[1:]))
            if row[0] in [M+2+i*M//times for i in range(times+1)]:
                IG.append(np.array(row[1:]))
            if row[0] in [2*M+3+i*M//times for i in range(times+1)]:
                NG.append(np.array(row[1:]))

    fname = "Emax="+str(Emax)+"N="+str(n)+"GlasseySD.csv"
    if not os.path.isfile(fname):
        time,Rs,Is,Ns = Glassey(Param,K,M,3,2,1)
        pd.DataFrame(time+Rs+Is+Ns).to_csv(fname)
    with open(fname) as f:
        for row in csv.reader(f, quoting=csv.QUOTE_NONNUMERIC):
            if row[0] in [i*M//times+1 for i in range(times+1)]:
                RGD.append(np.array(row[1:]))
            if row[0] in [M+2+i*M//times for i in range(times+1)]:
                IGD.append(np.array(row[1:]))
            if row[0] in [2*M+3+i*M//times for i in range(times+1)]:
                NGD.append(np.array(row[1:]))

    fname = "Emax="+str(Emax)+"N="+str(n)+"DVDMS.csv"
    if not os.path.isfile(fname):
        time,Rs,Ns,Is,Vs = DVDM_Glassey(Param,K,M,10**(-8),2,1)
        pd.DataFrame(time+Rs+Ns+Is+Vs).to_csv(fname)
    with open(fname) as f:
        for row in csv.reader(f, quoting=csv.QUOTE_NONNUMERIC):
            if row[0] in [i*M//times+1 for i in range(times+1)]:
                RD.append(np.array(row[1:]))
            if row[0] in [M+2+i*M//times for i in range(times+1)]:
                ID.append(np.array(row[1:]))
            if row[0] in [2*M+3+i*M//times for i in range(times+1)]:
                ND.append(np.array(row[1:]))
            if row[0] in [3*M+4+i*M//times for i in range(times+1)]:
                VD.append(np.array(row[1:]))

    x = np.linspace(0, 160, 8*K)

    fig = plt.figure()
    axs = []
    for i in range(times+1):
        axs.append(fig.add_subplot(times+1, 5, 5*i+1))
        axs.append(fig.add_subplot(times+1, 5, 5*i+2))
        axs.append(fig.add_subplot(times+1, 5, 5*i+3))
        axs.append(fig.add_subplot(times+1, 5, 5*i+4))
        axs.append(fig.add_subplot(times+1, 5, 5*i+5))

    l1 = "G"; l2 = "G+D"; l3 = "D"

    for i in range(times+1):
        ax = axs[5*i:5*i+5]

        EG = [(RG[i][k]**2+IG[i][k]**2)**0.5 for k in range(len(RG[i]))]
        EGD = [(RGD[i][k]**2+IGD[i][k]**2)**0.5 for k in range(len(RG[i]))]
        ED = [(RD[i][k]**2+ID[i][k]**2)**0.5 for k in range(len(RG[i]))]

        ax[0].plot(x, RG[i], label="ReE,"+l1)
        ax[0].plot(x, RGD[i], label="ReE,"+l2)
        ax[0].plot(x, RD[i], label="ReE,"+l3)
        ax[1].plot(x, IG[i], label="ImE,"+l1)
        ax[1].plot(x, IGD[i], label="ImE,"+l2)
        ax[1].plot(x, ID[i], label="ImE,"+l3)
        ax[2].plot(x, EG, label="|E|,"+l1)
        ax[2].plot(x, EGD, label="|E|,"+l2)
        ax[2].plot(x, ED, label="|E|,"+l3)
        ax[3].plot(x, NG[i], label="N,"+l1)
        ax[3].plot(x, NGD[i], label="N,"+l2)
        ax[3].plot(x, ND[i], label="N,"+l3)
        ax[4].plot(x, VD[i], label="V,"+l3)
        ax[0].legend(); ax[1].legend(); ax[2].legend(); ax[3].legend(); ax[4].legend()
    plt.show()

#Ntの情報に基づくVを数値的に計算した場合
def comparing_solitons_Ntbase(Emax,n,times):
    Param = parameters(20,1,Emax,10**(-8))
    T = Param[-2]
    K = math.floor(20*n); M = math.floor(T*n)
    dx = 20/K; dt = T/M

    RG,IG,NG = [],[],[]

    RD,ID,ND,VD = [],[],[],[]

    fname = "Emax="+str(Emax)+"N="+str(n)+"Ntbase"+"GlasseyS.csv"

    if not os.path.isfile(fname):
        time,Rs,Is,Ns = Glassey(Param,K,M,1,2,2)
        pd.DataFrame(time+Rs+Is+Ns).to_csv(fname)
    with open(fname) as f:
        for row in csv.reader(f, quoting=csv.QUOTE_NONNUMERIC):
            if row[0] in [i*M//times+1 for i in range(times+1)]:
                RG.append(np.array(row[1:]))
            if row[0] in [M+2+i*M//times for i in range(times+1)]:
                IG.append(np.array(row[1:]))
            if row[0] in [2*M+3+i*M//times for i in range(times+1)]:
                NG.append(np.array(row[1:]))

    fname = "Emax="+str(Emax)+"N="+str(n)+"Ntbase"+"DVDMS.csv"
    if not os.path.isfile(fname):
        time,Rs,Ns,Is,Vs = DVDM_Glassey(Param,K,M,10**(-8),2,2)
        pd.DataFrame(time+Rs+Ns+Is+Vs).to_csv(fname)
    with open(fname) as f:
        for row in csv.reader(f, quoting=csv.QUOTE_NONNUMERIC):
            if row[0] in [i*M//times+1 for i in range(times+1)]:
                RD.append(np.array(row[1:]))
            if row[0] in [M+2+i*M//times for i in range(times+1)]:
                ID.append(np.array(row[1:]))
            if row[0] in [2*M+3+i*M//times for i in range(times+1)]:
                ND.append(np.array(row[1:]))
            if row[0] in [3*M+4+i*M//times for i in range(times+1)]:
                VD.append(np.array(row[1:]))

    x = np.linspace(0, 160, 8*K)

    fig = plt.figure()
    axs = []
    for i in range(times+1):
        axs.append(fig.add_subplot(times+1, 5, 5*i+1))
        axs.append(fig.add_subplot(times+1, 5, 5*i+2))
        axs.append(fig.add_subplot(times+1, 5, 5*i+3))
        axs.append(fig.add_subplot(times+1, 5, 5*i+4))
        axs.append(fig.add_subplot(times+1, 5, 5*i+5))

    l1 = "G"; l2 = "D"

    for i in range(times+1):
        ax = axs[5*i:5*i+5]

        EG = [(RG[i][k]**2+IG[i][k]**2)**0.5 for k in range(len(RG[i]))]
        ED = [(RD[i][k]**2+ID[i][k]**2)**0.5 for k in range(len(RG[i]))]

        ax[0].plot(x, RG[i], label="ReE,"+l1)
        ax[0].plot(x, RD[i], label="ReE,"+l2)
        ax[1].plot(x, IG[i], label="ImE,"+l1)
        ax[1].plot(x, ID[i], label="ImE,"+l2)
        ax[2].plot(x, EG, label="|E|,"+l1)
        ax[2].plot(x, ED, label="|E|,"+l2)
        ax[3].plot(x, NG[i], label="N,"+l1)
        ax[3].plot(x, ND[i], label="N,"+l2)
        ax[4].plot(x, VD[i], label="V,"+l2)
        ax[0].legend(); ax[1].legend(); ax[2].legend(); ax[3].legend(); ax[4].legend()
    plt.show()

#Ntの情報に基づくVを理論的に計算した場合
def comparing_solitons_Vupdate(Emax,n,times):
    Param = parameters(20,1,Emax,10**(-8))
    T = Param[-2]
    K = math.floor(20*n); M = math.floor(T*n)
    dx = 20/K; dt = T/M

    RG,IG,NG = [],[],[]
    RGD,IGD,NGD = [],[],[]
    RD,ID,ND,VD = [],[],[],[]

    fname = "Emax="+str(Emax)+"N="+str(n)+"Vupdate"+"GlasseyS.csv"

    if not os.path.isfile(fname):
        time,Rs,Is,Ns = Glassey(Param,K,M,1,2,3)
        pd.DataFrame(time+Rs+Is+Ns).to_csv(fname)
    with open(fname) as f:
        for row in csv.reader(f, quoting=csv.QUOTE_NONNUMERIC):
            if row[0] in [i*M//times+1 for i in range(times+1)]:
                RG.append(np.array(row[1:]))
            if row[0] in [M+2+i*M//times for i in range(times+1)]:
                IG.append(np.array(row[1:]))
            if row[0] in [2*M+3+i*M//times for i in range(times+1)]:
                NG.append(np.array(row[1:]))

    fname = "Emax="+str(Emax)+"N="+str(n)+"Vupdate"+"GlasseySD.csv"
    if not os.path.isfile(fname):
        time,Rs,Is,Ns = Glassey(Param,K,M,3,2,3)
        pd.DataFrame(time+Rs+Is+Ns).to_csv(fname)
    with open(fname) as f:
        for row in csv.reader(f, quoting=csv.QUOTE_NONNUMERIC):
            if row[0] in [i*M//times+1 for i in range(times+1)]:
                RGD.append(np.array(row[1:]))
            if row[0] in [M+2+i*M//times for i in range(times+1)]:
                IGD.append(np.array(row[1:]))
            if row[0] in [2*M+3+i*M//times for i in range(times+1)]:
                NGD.append(np.array(row[1:]))

    fname = "Emax="+str(Emax)+"N="+str(n)+"Vupdate"+"DVDMS.csv"
    if not os.path.isfile(fname):
        time,Rs,Ns,Is,Vs = DVDM_Glassey(Param,K,M,10**(-8),2,3)
        pd.DataFrame(time+Rs+Ns+Is+Vs).to_csv(fname)
    with open(fname) as f:
        for row in csv.reader(f, quoting=csv.QUOTE_NONNUMERIC):
            if row[0] in [i*M//times+1 for i in range(times+1)]:
                RD.append(np.array(row[1:]))
            if row[0] in [M+2+i*M//times for i in range(times+1)]:
                ID.append(np.array(row[1:]))
            if row[0] in [2*M+3+i*M//times for i in range(times+1)]:
                ND.append(np.array(row[1:]))
            if row[0] in [3*M+4+i*M//times for i in range(times+1)]:
                VD.append(np.array(row[1:]))

    x = np.linspace(0, 160, 8*K)

    fig = plt.figure()
    axs = []
    for i in range(times+1):
        axs.append(fig.add_subplot(times+1, 5, 5*i+1))
        axs.append(fig.add_subplot(times+1, 5, 5*i+2))
        axs.append(fig.add_subplot(times+1, 5, 5*i+3))
        axs.append(fig.add_subplot(times+1, 5, 5*i+4))
        axs.append(fig.add_subplot(times+1, 5, 5*i+5))

    l1 = "G"; l2 = "G+D"; l3 = "D"

    for i in range(times+1):
        ax = axs[5*i:5*i+5]

        EG = [(RG[i][k]**2+IG[i][k]**2)**0.5 for k in range(len(RG[i]))]
        EGD = [(RGD[i][k]**2+IGD[i][k]**2)**0.5 for k in range(len(RG[i]))]
        ED = [(RD[i][k]**2+ID[i][k]**2)**0.5 for k in range(len(RG[i]))]

        ax[0].plot(x, RG[i], label="ReE,"+l1)
        ax[0].plot(x, RGD[i], label="ReE,"+l2)
        ax[0].plot(x, RD[i], label="ReE,"+l3)
        ax[1].plot(x, IG[i], label="ImE,"+l1)
        ax[1].plot(x, IGD[i], label="ImE,"+l2)
        ax[1].plot(x, ID[i], label="ImE,"+l3)
        ax[2].plot(x, EG, label="|E|,"+l1)
        ax[2].plot(x, EGD, label="|E|,"+l2)
        ax[2].plot(x, ED, label="|E|,"+l3)
        ax[3].plot(x, NG[i], label="N,"+l1)
        ax[3].plot(x, NGD[i], label="N,"+l2)
        ax[3].plot(x, ND[i], label="N,"+l3)
        ax[4].plot(x, VD[i], label="V,"+l3)
        ax[0].legend(); ax[1].legend(); ax[2].legend(); ax[3].legend(); ax[4].legend()
    plt.show()

def comparing_solitons_Glassey(Emax,n,times):
    Param = parameters(20,1,Emax,10**(-8))
    T = Param[-2]
    K = math.floor(20*n); M = math.floor(T*n)
    dx = 20/K; dt = T/M

    RG,IG,NG = [],[],[]
    RGD,IGD,NGD = [],[],[]

    fname = "Emax="+str(Emax)+"N="+str(n)+"GlasseyS.csv"

    if not os.path.isfile(fname):
        time,Rs,Is,Ns = Glassey(Param,K,M,1,2,1)
        pd.DataFrame(time+Rs+Is+Ns).to_csv(fname)
    with open(fname) as f:
        for row in csv.reader(f, quoting=csv.QUOTE_NONNUMERIC):
            if row[0] in [i*M//times+1 for i in range(times+1)]:
                RG.append(np.array(row[1:]))
            if row[0] in [M+2+i*M//times for i in range(times+1)]:
                IG.append(np.array(row[1:]))
            if row[0] in [2*M+3+i*M//times for i in range(times+1)]:
                NG.append(np.array(row[1:]))

    fname = "Emax="+str(Emax)+"N="+str(n)+"GlasseySD.csv"
    if not os.path.isfile(fname):
        time,Rs,Is,Ns = Glassey(Param,K,M,3,2,1)
        pd.DataFrame(time+Rs+Is+Ns).to_csv(fname)
    with open(fname) as f:
        for row in csv.reader(f, quoting=csv.QUOTE_NONNUMERIC):
            if row[0] in [i*M//times+1 for i in range(times+1)]:
                RGD.append(np.array(row[1:]))
            if row[0] in [M+2+i*M//times for i in range(times+1)]:
                IGD.append(np.array(row[1:]))
            if row[0] in [2*M+3+i*M//times for i in range(times+1)]:
                NGD.append(np.array(row[1:]))

    x = np.linspace(0, 160, 8*K)

    fig = plt.figure()
    axs = []
    for i in range(times+1):
        axs.append(fig.add_subplot(times+1, 4, 4*i+1))
        axs.append(fig.add_subplot(times+1, 4, 4*i+2))
        axs.append(fig.add_subplot(times+1, 4, 4*i+3))
        axs.append(fig.add_subplot(times+1, 4, 4*i+4))

    l1 = "G"; l2 = "G+D"

    for i in range(times+1):
        ax = axs[4*i:4*i+4]

        EG = [(RG[i][k]**2+IG[i][k]**2)**0.5 for k in range(len(RG[i]))]
        EGD = [(RGD[i][k]**2+IGD[i][k]**2)**0.5 for k in range(len(RG[i]))]

        ax[0].plot(x, RG[i], label=l1)
        ax[0].plot(x, RGD[i], label=l2)
        ax[0].set_ylabel("ReE")
        ax[1].plot(x, IG[i], label=l1)
        ax[1].plot(x, IGD[i], label=l2)
        ax[1].set_ylabel("ImE")
        ax[2].plot(x, EG, label=l1)
        ax[2].plot(x, EGD, label=l2)
        ax[2].set_ylabel("|E|")
        ax[3].plot(x, NG[i], label=l1)
        ax[3].plot(x, NGD[i], label=l2)
        ax[3].set_ylabel("N")
        ax[0].legend(); ax[1].legend(); ax[2].legend(); ax[3].legend()
    plt.show()

def comparing_solitons_DVDM(Emax,n,times):
    Param = parameters(20,1,Emax,10**(-8))
    T = Param[-2]
    K = math.floor(20*n); M = math.floor(T*n)
    dx = 20/K; dt = T/M

    RD1,ID1,ND1,VD1 = [],[],[],[]
    RD2,ID2,ND2,VD2 = [],[],[],[]
    RD3,ID3,ND3,VD3 = [],[],[],[]

    fname = "Emax="+str(Emax)+"N="+str(n)+"DVDMS.csv"
    if not os.path.isfile(fname):
        time,Rs,Ns,Is,Vs = DVDM_Glassey(Param,K,M,10**(-8),2,1)
        pd.DataFrame(time+Rs+Ns+Is+Vs).to_csv(fname)
    with open(fname) as f:
        for row in csv.reader(f, quoting=csv.QUOTE_NONNUMERIC):
            if row[0] in [i*M//times+1 for i in range(times+1)]:
                RD1.append(np.array(row[1:]))
            if row[0] in [M+2+i*M//times for i in range(times+1)]:
                ID1.append(np.array(row[1:]))
            if row[0] in [2*M+3+i*M//times for i in range(times+1)]:
                ND1.append(np.array(row[1:]))
            if row[0] in [3*M+4+i*M//times for i in range(times+1)]:
                VD1.append(np.array(row[1:]))

    fname = "Emax="+str(Emax)+"N="+str(n)+"Ntbase"+"DVDMS.csv"
    if not os.path.isfile(fname):
        time,Rs,Ns,Is,Vs = DVDM_Glassey(Param,K,M,10**(-8),2,2)
        pd.DataFrame(time+Rs+Ns+Is+Vs).to_csv(fname)
    with open(fname) as f:
        for row in csv.reader(f, quoting=csv.QUOTE_NONNUMERIC):
            if row[0] in [i*M//times+1 for i in range(times+1)]:
                RD2.append(np.array(row[1:]))
            if row[0] in [M+2+i*M//times for i in range(times+1)]:
                ID2.append(np.array(row[1:]))
            if row[0] in [2*M+3+i*M//times for i in range(times+1)]:
                ND2.append(np.array(row[1:]))
            if row[0] in [3*M+4+i*M//times for i in range(times+1)]:
                VD2.append(np.array(row[1:]))

    fname = "Emax="+str(Emax)+"N="+str(n)+"Vupdate"+"DVDMS.csv"
    if not os.path.isfile(fname):
        time,Rs,Ns,Is,Vs = DVDM_Glassey(Param,K,M,10**(-8),2,3)
        pd.DataFrame(time+Rs+Ns+Is+Vs).to_csv(fname)
    with open(fname) as f:
        for row in csv.reader(f, quoting=csv.QUOTE_NONNUMERIC):
            if row[0] in [i*M//times+1 for i in range(times+1)]:
                RD3.append(np.array(row[1:]))
            if row[0] in [M+2+i*M//times for i in range(times+1)]:
                ID3.append(np.array(row[1:]))
            if row[0] in [2*M+3+i*M//times for i in range(times+1)]:
                ND3.append(np.array(row[1:]))
            if row[0] in [3*M+4+i*M//times for i in range(times+1)]:
                VD3.append(np.array(row[1:]))

    x = np.linspace(0, 160, 8*K)

    fig = plt.figure()
    axs = []
    for i in range(times+1):
        axs.append(fig.add_subplot(times+1, 5, 5*i+1))
        axs.append(fig.add_subplot(times+1, 5, 5*i+2))
        axs.append(fig.add_subplot(times+1, 5, 5*i+3))
        axs.append(fig.add_subplot(times+1, 5, 5*i+4))
        axs.append(fig.add_subplot(times+1, 5, 5*i+5))

    l1 = "1"; l2 = "2"; l3 = "3"

    for i in range(times+1):
        ax = axs[5*i:5*i+5]

        ED1 = [(RD1[i][k]**2+ID1[i][k]**2)**0.5 for k in range(len(RD1[i]))]
        ED2 = [(RD2[i][k]**2+ID2[i][k]**2)**0.5 for k in range(len(RD1[i]))]
        ED3 = [(RD3[i][k]**2+ID3[i][k]**2)**0.5 for k in range(len(RD1[i]))]

        ax[0].plot(x, RD1[i], label="ReE,"+l1)
        ax[0].plot(x, RD2[i], label="ReE,"+l2)
        ax[0].plot(x, RD3[i], label="ReE,"+l3)
        ax[1].plot(x, ID1[i], label="ImE,"+l1)
        ax[1].plot(x, ID2[i], label="ImE,"+l2)
        ax[1].plot(x, ID3[i], label="ImE,"+l3)
        ax[2].plot(x, ED1, label="|E|,"+l1)
        ax[2].plot(x, ED2, label="|E|,"+l2)
        ax[2].plot(x, ED3, label="|E|,"+l3)
        ax[3].plot(x, ND1[i], label="N,"+l1)
        ax[3].plot(x, ND2[i], label="N,"+l2)
        ax[3].plot(x, ND3[i], label="N,"+l3)
        ax[4].plot(x, VD1[i], label="V,"+l1)
        ax[4].plot(x, VD2[i], label="V,"+l2)
        ax[4].plot(x, VD3[i], label="V,"+l3)
        ax[0].legend(); ax[1].legend(); ax[2].legend(); ax[3].legend(); ax[4].legend()
    plt.show()

#comparing_solitons(1,20,1)
#comparing_solitons_Ntbase(3,20,3)
#comparing_solitons_Vupdate(3,20,3)
#comparing_solitons(1.3,20,1)
#comparing_solitons_DVDM(1,20,1)
