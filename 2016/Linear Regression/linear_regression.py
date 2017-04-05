import numpy as np
import matplotlib.pyplot as plt

def genData():
    x = np.random.rand(100,2)
    y = np.zeros(shape=100)
    #basically a straight line
    for i in range(0, 100):
        x[i][0] = 1
        # our target variable
        e = np.random.uniform(-0.1,0.1,size=1)
        y[i] = np.sin(2*np.pi*x[i][1]) + e[0]
    return x,y


def gradientDescent(x,y,theta,alpha):


    xTrans = x.transpose()

    for i in range(0,9000):
        hypothesis = np.dot(x,theta)
        loss = hypothesis - y
        gradient = np.dot(xTrans,loss)
        theta = theta - alpha * gradient

    return theta


def gradientDescent2(x,y,theta,alpha):
    xTrans = x.transpose();
    gradientThetaZero = 0
    gradientThetaOne = 0
    gradientThetaTwo = 0
    gradientThetaThree = 0
    gradientThetaFour = 0
    gradientThetaFive = 0
    for i in range(0,99):
        hypothesis = np.dot(x[i],theta)

        loss = hypothesis - y[i]


        gradientThetaZero = gradientThetaZero + loss * x[i][0]
        gradientThetaOne = gradientThetaOne + loss * x[i][1]
        gradientThetaTwo = gradientThetaTwo + loss * x[i][2]
        gradientThetaThree = gradientThetaThree + loss * x[i][3]
        gradientThetaFour = gradientThetaFour + loss * x[i][4]
        gradientThetaFive = gradientThetaFive + loss * x[i][5]

        theta[0] = theta[0] - alpha * gradientThetaZero
        theta[1] = theta[1] - alpha * gradientThetaOne
        theta[2] = theta[2] - alpha * gradientThetaTwo
        theta[3] = theta[3] - alpha * gradientThetaThree
        theta[4] = theta[4] - alpha * gradientThetaFour
        theta[5] = theta[5] - alpha * gradientThetaFive
    return theta


def gradientDescent3(x, y, theta, alpha, m, numIterations):
    xTrans = x.transpose()
    for i in range(0, numIterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        # avg cost per example (the 2 in 2*m doesn't really matter here.
        # But to be consistent with the gradient, I include it)
        cost = np.sum(loss ** 2) / (2 * m)
        #print("Iteration %d | Cost: %f" % (i, cost))
        # avg gradient per example
        gradient = np.dot(xTrans, loss) / m
        # update
        theta = theta - alpha * gradient
    return theta




def buildXMatrix(x):
    return x





x,y = genData()
p = np.polyfit(x[:,1],y,3)

xvalues = x[:,1]
x2values = xvalues**2
x2values = np.reshape(x2values,(100,1))
x3values = xvalues**3
x3values = np.reshape(x3values,(100,1))
x4values = xvalues**4
x4values = np.reshape(x4values,(100,1))
x5values = xvalues**5
x5values = np.reshape(x5values,(100,1))
x6values = xvalues**6
x6values = np.reshape(x6values,(100,1))
x7values = xvalues**7
x7values = np.reshape(x7values,(100,1))


xMatrix = np.concatenate((x,x2values),axis=1)
xMatrix = np.concatenate((xMatrix,x3values),axis=1)
xMatrix = np.concatenate((xMatrix,x5values),axis=1)
xMatrix = np.concatenate((xMatrix,x6values),axis=1)
xMatrix = np.concatenate((xMatrix,x7values),axis=1)

theta = np.ones(7)

theta = gradientDescent(xMatrix,y,theta,0.010)
print(theta)


yarray = []
for i in range(0,100):
    yarray = theta[0]+theta[1]*x[:,1]+theta[
    2]*x[:,1]**2+theta[3]*x[:,1]**3

#print(len(yarray))
xfunction = np.arange(0.,1,0.000001)


plt.plot(x[:,1],y,"ro",xfunction,theta[0]+theta[1]*xfunction+theta[2]*xfunction**2+theta[3]*xfunction**3+theta[4]*xfunction**5+theta[5]*xfunction**6+theta[6]*xfunction**7,xfunction,np.sin(2*np.pi*xfunction))
plt.axis([0,1,-1,1]);
plt.show();
