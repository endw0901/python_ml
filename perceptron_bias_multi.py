# perceptron with bias
import numpy as np
def AndGate(x1, x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

print("AND Gate")
print(AndGate(0,0))
print(AndGate(0,1))
print(AndGate(1,0))
print(AndGate(1,1))

def NandGate(x1, x2):
    x = np.array([x1,x2])
    w = np.array([-0.5,-0.5])
    b = 0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

print("NAND Gate")
print(NandGate(0,0))
print(NandGate(0,1))
print(NandGate(1,0))
print(NandGate(1,1))


def OrGate(x1, x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.2
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

print("OR Gate")
print(OrGate(0,0))
print(OrGate(0,1))
print(OrGate(1,0))
print(OrGate(1,1))

def XorGate(x1,x2):
    s1 = NandGate(x1, x2)
    s2 = OrGate(x1,x2)
    y = AndGate(s1,s2)
    return y

print("XOR Gate")
print(XorGate(0,0))
print(XorGate(1,0))
print(XorGate(0,1))
print(XorGate(1,1))