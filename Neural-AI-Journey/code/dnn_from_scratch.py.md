# NOTE: Place 'mnist_train.csv' in the same directory before running
import numpy as np
path=r'mnist_train.csv'
data=np.loadtxt(path,delimiter=',',skiprows=1,max_rows=60000)
ans=data[:,0].astype(int)#All elements in the 0th row will be come to there numbers 0 to 9
X=data[:,1:].T/255
m=X.shape[1]#It gives us the shape of the X = 60000
YOH=np.zeros((10,m))# It is a space of Y
YOH[ans,np.arange(m)]=1.0
accuracy_history = []
def softmax(Z):
    exp_Z=np.exp(Z-np.max(Z,axis=0,keepdims=True))
    return exp_Z/(np.sum(exp_Z, axis=0, keepdims=True) + 1e-8)
def ReLU(Z):
    A=np.maximum(0,Z)
    return A
import pickle
with open('my_mnist_model.pkl', 'rb') as f:
    model_data = pickle.load(f)
W1 = model_data["W1"]
b1 = model_data["b1"]
W2 = model_data["W2"]
b2 = model_data["b2"]
W3 = model_data["W3"]
b3 = model_data["b3"]
beta=0.9
lr=0.1
#INITIALIZE VELOCITIES(The Ball at Rest)
vdw1,vdb1=np.zeros_like(W1),np.zeros_like(b1)
vdw2,vdb2=np.zeros_like(W2),np.zeros_like(b2)
vdw3,vdb3=np.zeros_like(W3),np.zeros_like(b3)
for epoch in range(20):
    P=np.random.permutation(m)
    X_shift=X[:,P]
    Y_shift=YOH[:,P]
    batchsize=128
    #Mini-batch loop
    for j in range(0,m,batchsize):
        x_batch=X_shift[:,j:j+batchsize]
        y_batch=Y_shift[:,j:j+batchsize]
        current_m=x_batch.shape[1]

        # Forward Pass
        Z1=np.dot(W1,x_batch)+b1
        A1=ReLU(Z1)
        Z2=np.dot(W2,A1)+b2
        A2=ReLU(Z2)
        Z3=np.dot(W3,A2)+b3
        A3=softmax(Z3) 
        # BACKWARD PASS
        dz3 = A3 - y_batch
        dw3 = np.dot(dz3, A2.T) * (1/current_m)
        db3 = np.sum(dz3, axis=1, keepdims=True) * (1/current_m)

        dz2 = np.dot(W3.T, dz3) * (Z2 > 0)
        dw2 = np.dot(dz2, A1.T) * (1/current_m)
        db2 = np.sum(dz2, axis=1, keepdims=True) * (1/current_m)

        dz1 = np.dot(W2.T, dz2) * (Z1 > 0)
        dw1 = np.dot(dz1, x_batch.T) * (1/current_m)
        db1 = np.sum(dz1, axis=1, keepdims=True) * (1/current_m)
        vdw1=beta*vdw1 +(1-beta)*dw1
        vdw2=beta*vdw2+(1-beta)*dw2
        vdw3=beta*vdw3+(1-beta)*dw3
        vdb1=beta*vdb1+(1-beta)*db1
        vdb2=beta*vdb2+(1-beta)*db2
        vdb3=beta*vdb3+(1-beta)*db3 
        # Update  
        W1 -= lr * vdw1
        W2 -= lr * vdw2
        W3 -= lr * vdw3
        b1 -= lr * vdb1
        b2 -= lr * vdb2
        b3 -= lr * vdb3
    if epoch % 1 == 0:
        Z1_all = np.dot(W1, X) + b1
        A1_all = ReLU(Z1_all)
        Z2_all = np.dot(W2, A1_all) + b2
        A2_all = ReLU(Z2_all)
        Z3_all = np.dot(W3, A2_all) + b3
        A3_all = softmax(Z3_all)
        
        predictions = np.argmax(A3_all, axis=0)
        accuracy = np.mean(predictions == ans)
        accuracy_history.append(accuracy)
        print(f"Epoch: {epoch}, Accuracy: {accuracy * 100:.2f}%")
        if epoch % 10 == 0:
    # 1. 'Speed' (Magnitude): How hard is the gravity pulling?
            grad_magnitude = np.linalg.norm(dw3) 
    # 2. 'Direction' (Mean Sign): Are most weights being pushed Up (+) or Down (-)?
    # If this flips between + and - every iteration, you are OVERSHOOTING.
            direction = np.mean(np.sign(dw3)) 
            ratio = np.mean(np.abs(vdw3)) / (np.mean(np.abs(dw3)) + 1e-8)
            print(f"Gradient Strength: {grad_magnitude:.6f}")
            print(f"Mean Direction: {direction:.4f} ")
            print(f"Momentum Ratio: {ratio:.2f}")
import pickle
model_data = {
    "W1": W1, "b1": b1,
    "W2": W2, "b2": b2,
    "W3": W3, "b3": b3
}
with open('my_mnist_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)
import matplotlib.pyplot as plt
plt.plot(accuracy_history)
plt.title("The Climb to 99%")
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.show()