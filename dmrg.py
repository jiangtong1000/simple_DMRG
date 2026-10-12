#!/usr/bin/env python3
# Python 3 compatibility copy prepared in 2026; see original/dmrg_2017.py.
# The algorithm is unchanged.
#dmrg programe in the language of MPS
#1 dimenssional Heisenberg model spin -1/2
import math
import numpy as np
import matplotlib.pyplot as plt

tol = 10**(-4) #iteration tolerance
N = 8 # number of spins
D = 100
d = 2
##############################################################
#MPO representations of Hamiltonian
##############################################################
I = np.identity(2)
S_plus = np.zeros((2,2))
S_plus[0][1] = 1.
S_minus = np.zeros((2,2))
S_minus[1][0] = 1.
S_z = 1./2*np.identity(2)
S_z[1][1] = -1/2.
h = 0.#magenetic field
Jz = 1.#S^z coulping constant
J = -1. #S^+S^- coulping constant

fifth_row = np.zeros((1,5,2,2))
fifth_row[0][0] = -h*S_z[:,:]
fifth_row[0][1] = J/2*S_minus[:,:]
fifth_row[0][2] = J/2*S_plus[:,:]
fifth_row[0][3] = Jz*S_z[:,:]
fifth_row[0][4] = I

print (fifth_row)

first_column = np.zeros((5,1,2,2))
first_column[0][0] = I
first_column[1][0] = S_plus[:,:]
first_column[2][0] = S_minus[:,:]
first_column[3][0] = S_z
first_column[4][0] = -h*S_z[:,:]

print (first_column)

operator_Wi = np.zeros((5,5,2,2))
operator_Wi[:,0,:,:] = first_column[:,0]
operator_Wi[4,:,:,:] = fifth_row[0,:]

operator_W1 = np.zeros((1,5,2,2))
operator_W1[0,:,:,:] = fifth_row[0,:]
operator_WL = np.zeros((5,1,2,2))
operator_WL[:,0,:,:] = first_column[:,0]
W=[]
W.append(operator_W1)
for l in range(1,N-1):
    W.append(operator_Wi)
W.append(operator_WL)

##############################################################
#generate initial guess of psi,which is right-normalized MPS
#ie. consist B-type matrices only
##############################################################
M = []
for l in range(1,N//2+1):
    M.append(np.random.uniform(size=(min(D,d**(l-1)),d,min(D,d**l))))
for l in range(N//2+1,N+1):
    M.append(np.random.uniform(size=(min(D,d**(N-l+1)),d,min(D,d**(N-l)))))
for l in range(N-1,0,-1):
    (a_lm1,sigma_l,a_l) = M[l].shape
    M[l] = np.reshape(M[l],(a_lm1,(sigma_l*a_l)))
    U,S,V = np.linalg.svd(M[l],full_matrices=False)
    S = np.diag(S)
    s_lm1 = S.shape[1]
    M[l] = np.reshape(V,(s_lm1,sigma_l,a_l))
    US = np.tensordot(U,S,axes=(1,0))
    M[l-1] = np.tensordot(M[l-1],US,axes=(2,0))
    # 记得判断归一化这么做是否正确
    if l==1:
        sum = 0
        for i in range(d):
            M0_i = np.matrix(M[0][:,i,:])
            sum+=M0_i*M0_i.H
        M[0] /=np.sqrt(sum)
        #for i in range(d):
        #   M[0][:,i,:] = M[0][:,i,:]/np.sqrt(sum)


##############################################################
#Initialize the R-expressions from site L-1 through 1
##############################################################
initial_R = [np.ones(shape=(1,1,1))]#used in the first step of each right sweep
for l in range(N-1):
    initial_R.append(None)
initial_R.append(np.ones(shape=(1,1,1)))#used in the first step of each left sweep
for l in range(N-1,0,-1):
    BW =  np.tensordot(M[l],W[l],axes=(1,2))
    BWB = np.tensordot(BW,M[l],axes=(4,1))
    initial_R[l] = np.tensordot(BWB,initial_R[l+1],axes=((1,3,5),(0,1,2)))


##############################################################
#define the update function for L AND R expressions
#combine the L AND R into one expression, LR
#for instance,While right_sweep optimizing the l site,
#the left of l is L part ,the right the R part
##############################################################
LR = []
for l in range(N+1):
    LR.append(initial_R[l])
def optimize_sigma_l(M,W,LR,l,type):
    L = LR[l-1]
    R = LR[l]
    #if l==1:
        #print('Lshape',L.shape)
        #print('Rshape',R.shape)
        #print('W',W[l-1].shape)
    alm_1,sigma_l,a_l = L.shape[0],W[l-1].shape[2],R.shape[0]
    if type == "right_sweep":
        LW = np.tensordot(L,W[l-1],axes=(1,0))
        #print('LW',LW.shape)
        LWR = np.tensordot(LW,R,axes=(2,1))
        print(LWR.shape)
        #print('LWR',LWR.shape)
        H = np.moveaxis(LWR, [2,4],[1,2])
        H = H.reshape(alm_1*sigma_l*a_l,alm_1*sigma_l*a_l)

        #print(H.shape)
        [w,v] = np.linalg.eigh(H)
        #print('v',v.shape)
        M[l-1] = v[:,0].reshape(alm_1,sigma_l,a_l)
        M[l-1] = M[l-1].reshape(alm_1*sigma_l,a_l)
        U,S,V = np.linalg.svd(M[l-1],full_matrices=False)
        S = np.diag(S)
        s_lm1 = S.shape[0]
        M[l-1] = U.reshape(alm_1,sigma_l,s_lm1)
        print(alm_1,sigma_l,s_lm1)
        #S = np.diag(S)
        SV = np.tensordot(S,V,axes = (1,0))
        M[l] = np.tensordot(SV,M[l],axes=(1,0))
    elif type == "left_sweep":
        WR = np.tensordot(W[l-1],R,axes=(1,1))
        LWR = np.tensordot(L,WR,axes=(1,0))
        print(LWR.shape)
        H = np.moveaxis(LWR, [2,4],[1,2])
        H = H.reshape(alm_1*sigma_l*a_l,alm_1*sigma_l*a_l)
        [w,v] = np.linalg.eigh(H)
        M[l-1] = v[:,0].reshape(alm_1,a_l*sigma_l)
        U,S,V = np.linalg.svd(M[l-1],full_matrices=False)
        print (np.sum(S**2))
        S = np.diag(S)
        s_lm1 = S.shape[0]
        M[l-1] = V.reshape(s_lm1,sigma_l,a_l)
        print(s_lm1,sigma_l,a_l)
        US = np.tensordot(U,S,axes = (1,0))
        #S = np.diag(S)
        M[l-2] = np.tensordot(M[l-2],US,axes=(2,0))
    return w[0],M
# after optimize the matrice of site l (here,M[l-1]),
# then upadate the LR[l](for right sweep),or the LR[l-1](for left sweep)
def LR_update(M,W,LR,l,type):
    if type == "right_sweep":
        BW = np.tensordot(M[l-1],W[l-1],axes=(1,2))
        BWB = np.tensordot(BW,M[l-1],axes=(4,1))
        LR[l] = np.tensordot(LR[l-1],BWB,axes=((0,1,2),(0,2,4)))
    elif type == "left_sweep":
        BW =  np.tensordot(M[l-1],W[l-1],axes=(1,2))
        BWB = np.tensordot(BW,M[l-1],axes=(4,1))
        LR[l-1] = np.tensordot(BWB,LR[l],axes=((1,3,5),(0,1,2)))
    return LR
#def main(num):
new_energy = 0
old_energy = 0
num = 1
e=[]
E=[]
while num < 4:
    print(num)
    if num%2 == 1:
        type = "right_sweep"
        print('the No.%i sweep,left to right' % num)
        for l in range(1,N):
            old_energy = new_energy
            new_energy, M = optimize_sigma_l(M,W,LR,l,type)
            E.append(new_energy)
            LR = LR_update(M,W,LR,l,type)
            print(new_energy)
            print ("m", [m.shape[2] for m in M])
            print ("m", [w.shape[0] for w in W])
    elif num%2 == 0:
        type = "left_sweep"
        print('the No.%i sweep,test,right to left' % num)
        for l in range(N,1,-1):
            old_energy = new_energy
            new_energy,M = optimize_sigma_l(M,W,LR,l,type)
            E.append(new_energy)
            print(new_energy)
            LR = LR_update(M,W,LR,l,type)
    if abs(new_energy-old_energy)<tol:
        break
    a = new_energy-old_energy
    num=num+1
    #print('the ground state energy')
    #print(new_energy)
    e.append(new_energy)
plt.plot(e,'bo',ms=100)
plt.savefig("e.pdf")
plt.plot(E,'bo')
plt.savefig("ee.pdf")
    #print(old_energy)
#return new_energy
