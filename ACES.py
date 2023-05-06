import os
import sys
import numpy as np
import random

n1 = 10
r1 = 4
j1 = 0
# m = len(GateVariables[n]) 
m1 = 480
rank = 0
jmax = 100

GateNames = ["CX", "XC", "XYZ", "ZYX", "YXZ", "XZY", "ZXY", "YZX", "Measure"]

Gates = [np.array([[1, 0, 0, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 0, 0, 1]]),
    np.array([[1, 0, 1, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 1, 0, 1]]),
    np.array([[1, 0], [0, 1]]),
    np.array([[0, 1], [1, 0]]),
    np.array([[1, 0], [1, 1]]),
    np.array([[1, 1], [0, 1]]),
    np.array([[0, 1], [1, 1]]),
    np.array([[1, 1], [1, 0]])]

InverseGates = []

for gate in Gates:
    inverse_gate = np.linalg.inv(gate)
    InverseGates.append(inverse_gate % 2)

AllOneQubitGates = []
AllTwoQubitGates = []

for i in range(len(Gates)):
    if Gates[i].size == 4:
        AllOneQubitGates.append(i + 1)
    elif Gates[i].size == 16:
        AllTwoQubitGates.append(i + 1)

NumberOfOneQubitGates = len(AllOneQubitGates)
NumberOfTwoQubitGates = len(AllTwoQubitGates)

def GateInversion(k):
    # for i in range(len(Gates)):
    #     if np.array_equal(InverseGates[i], Gates[k - 1]):
    #         return i + 1
    # return -1
    if k >= 0 and k <= 6:
        return k
    elif k == 7:
        return 8
    elif k == 8:
        return 7
    else:
        return -1

def RandomOneQubitLayer(n):
    output = []
    for _ in range(n):
        output.append(random.choice(AllOneQubitGates))
    return output

def RandomTwoQubitLayer(n, offset):
    o = offset % 2
    t = []
    length = ((n + o) // 2) - o
    for _ in range(length):
        t.append(random.choice(AllTwoQubitGates))
    for i in range(len(t)):
        t.insert(2 * i + 1, 0)

    if ((n + o) % 2) == 1:
        t.append(random.choice(AllOneQubitGates))
    if o == 1:
        t.insert(0, random.choice(AllOneQubitGates))

    return t

def DirectSum(c):
    dim = 0
    for i in c:
        if i > 0:
            if i == 1 or i == 2:
                dim += 4
            else:
                dim += 2

    d_sum = np.zeros([dim, dim], dtype=int)

    curr = 0

    for j in c:
        if j > 0:
            mat = Gates[j - 1]
            rows, cols = mat.shape

            for k in range(rows):
                for l in range(cols):
                    d_sum[curr + k][curr + l] += mat[k][l]

            if j == 1 or j == 2:
                curr += 4
            else:
                curr += 2
    
    return d_sum

def RandomCircuit(n, d):
    circ = []
    for k in range(1, d + 1):
        if k % 2 == 1:
            circ.append(RandomOneQubitLayer(n))
        else:
            temp = ((k // 2) - 1) % 2
            circ.append(RandomTwoQubitLayer(n, temp))

    return np.array(circ)

def RandomCircuit2(n, d):
    circ = []
    for _ in range(d):
        b = random.randint(0, 1)
        if b == 0:
            circ.append(RandomOneQubitLayer(n))
        else:
            r = random.randint(0, 1)
            circ.append(RandomTwoQubitLayer(n, r))
    
    return np.array(circ)

def InverseCircuit(c):
    inv_c = []
    for i in range(len(c)):
        layer = c[len(c) - 1 - i]
        new_layer = []
        for g in layer:
            new_layer.append(GateInversion(g))
        inv_c.append(new_layer)

    return np.array(inv_c)

def RandomMirrorCircuit(n, d):
    t = RandomCircuit(n, d // 2)
    t = np.concatenate((t, InverseCircuit(t)))
    if d % 2 == 0:
        return t
    else:
        t = np.vstack((t, RandomOneQubitLayer(n)))
        return t
    
def RandomMirrorCircuit2(n, d):
    t = RandomCircuit2(n, d // 2)
    t = np.concatenate((t, InverseCircuit(t)))
    if d % 2 == 0:
        return t
    else:
        t = np.vstack((t, RandomOneQubitLayer(n)))
        return t
    
def CircuitHistory(c):
    cc = []
    for layer in c:
        cc.append(DirectSum(layer))
    
    ccc = [cc[0]]
    for i in range(1, len(cc)):
        new_layer = np.matmul(cc[i], ccc[i - 1]) % 2
        ccc.append(new_layer)
    
    ccc = np.array(ccc)
    
    return ccc

def PauliHistory(c):
    temp = np.identity(2 * n1, dtype=int)
    id = np.array([temp])
    joint = np.concatenate((id, CircuitHistory(c)))

    return np.transpose(joint, (2, 0, 1))

def IntegerDigits(m, k):
    temp = bin(m)
    digits = [int(d) for d in str(temp)[2:]]

    length = len(digits)

    if length > k:
        digits = digits[length - k: length]
    elif length < k:
        for _ in range(k - length):
            digits.insert(0, 0)
    
    return digits

def PadRight(l, n):
    length = len(l)
    if length < n:
        l += [0] * (n - length)
    
    return l

def RotateRight(l, n):
    n %= len(l)
    return l[-n:] + l[:-n]

def Table(n, k):
    paulis = []

    for j in range(n - k + 1):
        for m in range(1, 4 ** k):
            paulis.append(RotateRight(PadRight(IntegerDigits(m, 2 * k), 2 * n), 2 * j))
    
    return np.array(paulis)

def KNNPaulis(n, k):
    paulis = Table(n, k).tolist()

    new_paulis = []
    [new_paulis.append(x) for x in paulis if x not in new_paulis]

    return np.array(new_paulis)

def CircuitLayerType(c):
    types = []
    
    for k in range(len(c)):
        if c[k][0] <= NumberOfTwoQubitGates:
            types.append(2)
        elif c[k][1] <= NumberOfTwoQubitGates:
            types.append(3)
        else:
            types.append(1)
    
    return types

def CircuitPauliPartition(c, p, n):
    # tt = np.array([0, 0, 0])
    t = [[], [], []]
    types = CircuitLayerType(c)
    for i in range(len(types)):
        if types[i] == 1:
            t[0].append(i)
        elif types[i] == 2:
            t[1].append(i)
        elif types[i] == 3:
            t[2].append(i)
        else:
            raise(Exception)
    tL = [len(t[0]), len(t[1]), len(t[2])]

    temp = np.zeros((2 * n, n), dtype=int)
    for i in range(n):
        temp[2 * i][i] += 2
        temp[2 * i + 1][i] += 1
    tt = [temp]

    if n % 2 == 1:
        raise(Exception)
    else:
        temp1 = np.zeros((2 * n, n), dtype=int)
        for i in range(0, n, 2):
            temp1[2 * i][i] += 8
            temp1[2 * i + 1][i] += 4
            temp1[2 * i + 2][i] += 2
            temp1[2 * i + 3][i] += 1
        
        temp2 = np.zeros((2 * n, n), dtype=int)
        temp2[0][0] += 2
        temp2[1][0] += 1
        temp2[2 * n - 2][n - 1] += 2
        temp2[2 * n - 1][n - 1] += 1
        for j in range(0, n - 2, 2):
            temp2[2 * j + 2][j + 1] += 8
            temp2[2 * j + 3][j + 1] += 4
            temp2[2 * j + 4][j + 1] += 2
            temp2[2 * j + 5][j + 1] += 1
        
        tt.append(temp1)
        tt.append(temp2)
    
    pp = np.zeros((2 * n, len(c), n), dtype=int)
    
    for j in range(3):
        if tL[j] > 0:
            for k in t[j]:
                pp[:,k,:] = np.matmul(p[:,k,:], tt[j])

    return pp

def Arow(c, cpp, ip):
    temp = cpp[ip]
    xor = temp[0]
    for i in range(1, len(temp)):
        xor = xor ^ temp[i]
    
    keys = []
    values = []

    row_num = len(xor)
    col_num = len(xor[0])
    for i in range(row_num):
        for j in range(col_num):
            if xor[i][j] != 0:
                keys.append([i, j])
                values.append(xor[i][j])
    
    extracted = []
    for key in keys:
        extracted.append(c[key[0]][key[1]])
    
    arow = []
    for i in range(len(extracted)):
        new = [extracted[i], keys[i][1] + 1, values[i]]
        arow.append(new)
        
    return arow

def zeromaker(n):
    zeros = []
    for _ in range(n):
        zeros.append(0)
    return zeros

def Amat(c, cpp, ip):
    pp = []
    for i in ip:
        temp = []
        for j in range(len(i)):
            if i[j] != 0:
                temp.append(j)
        pp.append(temp)

    gate_var = GateVariables1(n1)
    num_gates = len(gate_var)

    amat = []
    for k in range(len(pp)):
        arow = Arow(c, cpp, pp[k])
        row = zeromaker(num_gates)
        for gate in arow:
            index = gate_var.index(gate)
            row[index] += 1
        
        amat.append(row)

    return np.ndarray(amat)

def tuples(l):
    newlist = []
    for i in l[0]:
        for j in l[1]:
            for k in l[2]:
                newlist.append([i, j, k])
    return newlist

def GateVariables1(n: int, model=""):
    m = []
    for c in model:
        m.append(c)
    
    a = [[1],[1],[1]]
    b = [[NumberOfTwoQubitGates + 1], [1], [1]]
    c = [[NumberOfOneQubitGates + NumberOfTwoQubitGates + 1], [1], [1]]

    if "G" not in m:
        a[0] = AllTwoQubitGates
        b[0] = AllOneQubitGates
    
    if "Q" not in m:
        a[1] = [i for i in range(1, n)]
        b[1] = [i for i in range(1, n + 1)]
        c[1] = [i for i in range(1, n + 1)]
    
    if "P" not in m:
        a[2] = [i for i in range(1, 16)]
        b[2] = [i for i in range(1, 4)]

    if "M" not in m:
        c[2] = [i for i in range(1, 4)]

    newlist = []
    for i in tuples(a):
        newlist.append(i)
    for j in tuples(b):
        newlist.append(j)
    for k in tuples(c):
        newlist.append(k)
    
    return newlist

def GateVariables2(n:int, x, model=""):
    pass

def Pmarg(l):
    i = l[0]
    j = l[1]
    k = l[2]
    return [i, j, 1]

def Qmarg(l):
    i = l[0]
    j = l[1]
    k = l[2]
    return [i, 1, k]

def Mmarg(l):
    i = l[0]
    j = l[1]
    k = l[2]
    return [i, j, 1 if (i > NumberOfOneQubitGates + NumberOfTwoQubitGates) else k]

def Gmarg(l):
    i = l[0]
    j = l[1]
    k = l[2]
    
    if i <= NumberOfTwoQubitGates:
        f = 1
    elif i <= NumberOfOneQubitGates + NumberOfTwoQubitGates:
        f = 2
    else:
        f = 3
    return [f, j, k]

def Averaged_Circuits():
    n = 10
    r = 4
    j = 0
    gate_vars = GateVariables1(n)
    m = len(gate_vars)
    rank = 0
    jmax = 100
    while rank != m and j < jmax:
        j += 1
        Aaa = []
        Aaa = np.array(Aaa)
        Mmm = []
        Mmm = np.array(Mmm)
        dd = [2, 2, 2, 2, 2, 3, 5, 8, 13, 21]
        kk = [2, 2, 2, 2, 1, 1, 1, 1, 1, 1]

        for d in range(len(dd)):
            c = RandomMirrorCircuit(n, dd[d])
            print(type(c))
            c2 = RandomCircuit2(n, r)
            print(type(c2))
            c = np.concatenate((c, c2))
            
            p = PauliHistory(c)

            cm = c
            const_array = []
            for i in range(n):
                const_array.append(NumberOfOneQubitGates + NumberOfTwoQubitGates + 1)
            const_array = np.array([const_array])
            cm = np.concatenate((cm, const_array))
            
            cpp = CircuitPauliPartition(cm, p, n)
            knn_p = KNNPaulis(n, kk[d])

            aa = Amat(cm, cpp, knn_p)
            if Aaa.ndim < 2:
                Aaa = aa
            else:
                Aaa = np.concatenate((Aaa, aa))
            if Mmm.ndim < 2:
                Mmm = cm
            else:
                Mmm = np.concatenate((Mmm, cm))

        # Aaa = np.array(Aaa)
        # rank = np.linalg.matrix_rank(Aaa)
        # print("On try j = ", j, ", the rank ratio is ", rank, "/", m, ".")
        # Aaa.tolist()

    if rank == m:
        print("Total number of estimated fidelities = ",len(Aaa))
        return Aaa

    else:
        return None























def Differences(l):
    curr = l[0]
    newlist = []
    for i in range(1, len(l)):
        temp = l[i]
        newlist.append(temp - curr)
        curr = temp
    return newlist

def RandomSimplex(d):
    l = [0, 1]
    for _ in range(d - 1):
        l.append(random.uniform(0,1))
    
    l.sort()
    return Differences(l)

def Accumulate(l):
    newlist = []
    curr = 0
    for i in range(len(l)):
        curr += l[i]
        newlist.append(curr)
    
    return newlist

def ProjectSimplex(p):
    n = len(p)
    m = p.sort(reverse=True)
    
    m_acc = Accumulate(m)
    
    max = 0
    for i in range(len(m)):
        diff = m[i] - m_acc[i] + 1
        if diff > 0:
            max = i + 1
    
    
    

Averaged_Circuits()



# c = np.array([[4,7,4,7,5,5,3,8,8,8], [4,8,4,8,5,5,3,7,7,7], [4,7,6,4,8,6,6,4,4,5], [5,2,0,1,0,2,0,1,0,6], [7,2,0,1,0,1,0,1,0,3], [3,7,5,8,4,4,6,4,7,3]])
# cm = np.array([[4,7,4,7,5,5,3,8,8,8], [4,8,4,8,5,5,3,7,7,7], [4,7,6,4,8,6,6,4,4,5], [5,2,0,1,0,2,0,1,0,6], [7,2,0,1,0,1,0,1,0,3], [3,7,5,8,4,4,6,4,7,3], [9,9,9,9,9,9,9,9,9,9]])
# p = PauliHistory(c)

# cpp = CircuitPauliPartition(cm, p, n1)



# knn_p = KNNPaulis(n1, 2)

# ippp = []
# for i in range(len(knn_p)):
#     ret = []
#     curr = knn_p[i]
#     for j in range(len(curr)):
#         if curr[j] != 0:
#             ret.append(j)
    
#     ippp.append(ret)

# val = Arow(cm, cpp, ippp[79])

# print(val)

# amat = Amat(cm, cpp, knn_p)
# row = amat[69]
# indices = []
# for i in range(len(row)):
#     if row[i] != 0:
#         indices.append(i)

# print(indices)


# # Amat(cm, cpp, knn_p)