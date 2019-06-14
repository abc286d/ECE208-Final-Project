import numpy as np
import matplotlib.pyplot as plt
import random
from itertools import combinations

def extractGroup(person):
    # person should look like "COMxxx-yyy"
    idx = person.find('-')
    return int(person[3:idx])

def myCurve(a, b, x):
    # a, b are parameters for the curve, x is the time to predict
    res = 1 - a * np.e**(-b * x**1)
    if res >= 0.996:
        res = 1
    return res

def train(x, y):
    l = len(x)
    x2 = x.reshape((-1, 1)) 
    temp = np.ones((l, 1))

    x2 = np.power(x2, 1)

    A = np.hstack((x2, -temp))
    y2 = y
    y2[y2 >= 1] = 0.996      
    y2 = -np.log(1 - y2)
    y2 = y2.reshape((-1, 1))

    # if y2[-1, 0] > 0.1:
    #     A = A[20:, :]
    #     y2 = y2[20:, :]

    A = A[10:, :]
    y2 = y2[10:, :]

    params = np.linalg.inv(A.T@A)@A.T@y2
    b = params[0, 0]
    a = np.e**(params[1, 0])
    return (a, b)

def calMAE(x, y):
    comSet = set(x)
    MAE = 0
    for com in comSet:
        MAE += abs(x.index(com)-y.index(com))
    MAE /= len(comSet)
    return MAE

def calKendallTau(x, y):
    comSet = set(x)
    l =  len(comSet)
    KendallTau = 0
    for ele1, ele2 in combinations(comSet, 2):
        a = x.index(ele1) - x.index(ele2)
        b = y.index(ele1) - y.index(ele2)
        if a * b < 0:
            KendallTau += 1
    KendallTau = KendallTau / (l * (l-1)/2)
    return KendallTau

def findNearestTimeStamp(comData, comNum, time):
    for idx, val in enumerate(comData[comNum]['x']):
        if val >= time:
            break
    return idx


if __name__ == "__main__": 
    #-------------- read files ---------------#
    f =  open("./transmission_network.txt")
    lines = f.readlines()
    records = [line.split() for line in lines]
    infected = set()
    comData = {}
    timeStamp = [0.0]

    for record in records:
        time = float(record[2])
        if time != 0:
            timeStamp.append(time)

        for i in (0, 1):
            if record[i] == "None" or record[i] in infected: continue
            infected.add(record[i])
            groupNum = extractGroup(record[i])
            if groupNum not in comData:
                comData[groupNum] = {'x': [0.0], 'y': [0]}
            if time == 0.0:
                comData[groupNum]['y'][0] += 1/500
            else:
                comData[groupNum]['x'].append(time)
                lastNum = comData[groupNum]['y'][-1]
                comData[groupNum]['y'].append(lastNum + 1/500)
    f.close()

    #------------ train and predict new ---------------#
    # time should not be too late, like 100 years, that's not so useful
    # typically, we know the first 5 years' data, and predict 0.5, 1, 1.5, 2... years later, what the increment will be
    # The old train and predict is of no use.

    comTestNo = list(range(400))
    timeNum = len(timeStamp)

    timeInterval = 0.5 # predict every #(0.5) years
    trainTime = [3, 5, 10]  # use the first #(5) years to train
    predictionSlot = 10 # predict for the next # years
    parameters = {}
    

    for aTrainTime in trainTime:
        print("Now use {} years to train".format(aTrainTime))
        predCurve = []
        randomCurve = []

        predCurve_MAE = []
        randomCurve_MAE = []

        predictionTime = [aTrainTime + (i+1) * timeInterval for i in range(int(predictionSlot / timeInterval))]
        
        # train model
        
        for currCom in comTestNo:
            idx = findNearestTimeStamp(comData, currCom, aTrainTime)
            # print("Now training community No. {}".format(currCom))
            a, b = train(np.array(comData[currCom]['x'][:idx]), np.array(comData[currCom]['y'][:idx]))
            parameters[currCom] = (a, b, idx)
            
        # prediction step
        for aPredictionTime in predictionTime:
            predRes = []
            realRes = []

            print("predict at time {}".format(aPredictionTime))
            for currCom in comTestNo:
                a, b, train_idx = parameters[currCom]    
                pred = myCurve(a, b, aPredictionTime)
                pred_inc = (pred - comData[currCom]['y'][train_idx]) / comData[currCom]['y'][train_idx]
                predRes.append((currCom, pred_inc))
                
                pred_idx = findNearestTimeStamp(comData, currCom, aPredictionTime)
                real_inc = (comData[currCom]['y'][pred_idx] - comData[currCom]['y'][train_idx])/ \
                    comData[currCom]['y'][train_idx]
                realRes.append((currCom, real_inc))
            
            # calculate MAE and KendallTau
            predRes = sorted(predRes, key = lambda x : (-x[1], x[0]))
            realRes = sorted(realRes, key = lambda x : (-x[1], x[0]))
                        
            predOrd, _ = zip(*predRes)
            realOrd, _ = zip(*realRes)
            MAE = calMAE(predOrd, realOrd)

            randomOrd = list(predOrd)
            random.shuffle(randomOrd)
            MAE_random = calMAE(randomOrd, realOrd)
            print("MAE of the prediction is {}".format(MAE))
            print("MAE of a random guess is {}".format(MAE_random))

            predCurve_MAE.append(MAE)
            randomCurve_MAE.append(MAE_random)


            KendallTau = calKendallTau(predOrd, realOrd)
            KendallTau_random = calKendallTau(randomOrd, realOrd)
            print("Kendall Tau distance of the prediction is {0:.3f}".format(KendallTau))
            print("Kendall Tau distance of a random guss is {0:.3f}".format(KendallTau_random))
            print()

            predCurve.append(KendallTau)
            randomCurve.append(KendallTau_random)
        
        plt.figure()
        plt.title("Use {} years infection data to train".format(aTrainTime))
        plt.plot(predictionTime, predCurve, 'o', label = "prediction")
        plt.plot(predictionTime, randomCurve, 'o', label = "random")
        plt.xlabel("time/year")
        plt.ylabel("Kendall Tau distance of the rank")
        plt.legend()
        plt.show()
        # plt.savefig("problem1_KDT_{}years".format(aTrainTime))

        plt.figure()
        plt.title("Use {} years infection data to train".format(aTrainTime))
        plt.plot(predictionTime, predCurve_MAE, 'o', label = "prediction")
        plt.plot(predictionTime, randomCurve_MAE, 'o', label = "random")
        plt.xlabel("time/year")
        plt.ylabel("MAE of the rank")
        plt.legend()
        plt.show()
        # plt.savefig("problem1_MAE_{}years".format(aTrainTime))


    # #------------ train and predict ---------------#
    # comTestNo = list(range(400))
    # timeNum = len(timeStamp)

    # trainPercent = [0.3, 0.4] # how much data are used to train 
    # testNum = 3 # how many tests we will do

    # for aTrainPercent in trainPercent:
    #     print("### {} of the dataset is used to train ###".format(aTrainPercent))
    #     trainTime = timeStamp[int(timeNum * aTrainPercent)]
    #     testPercent = [aTrainPercent + (i+1) * (1-aTrainPercent)/(testNum+1) for i in range(testNum)]

    #     for aTestPercent in testPercent:
    #         print("Test point at {0:.3f}".format(aTestPercent))
    #         predRes = []
    #         realRes = []

    #         testTime = timeStamp[int(timeNum * aTestPercent)]
    #         for currCom in comTestNo:
    #             idx = findNearestTimeStamp(comData, currCom, trainTime)
    #             # print("Now training community No. {}".format(currCom))
    #             a, b = train(np.array(comData[currCom]['x'][:idx]), np.array(comData[currCom]['y'][:idx]))
    #             pred = myCurve(a, b, testTime)
    #             predRes.append((currCom, pred))
                
    #             idx = findNearestTimeStamp(comData, currCom, testTime)
    #             realRes.append((currCom, comData[currCom]['y'][idx]))

    #         predRes = sorted(predRes, key = lambda x : (-x[1], x[0]))
    #         realRes = sorted(realRes, key = lambda x : (-x[1], x[0]))
    #         # print("Prediction result is ")
    #         # print(predRes)
    #         # print("Real result is ")
    #         # print(realRes)

    #         predOrd, _ = zip(*predRes)
    #         realOrd, _ = zip(*realRes)
    #         MAE = calMAE(predOrd, realOrd)

    #         randomOrd = list(predOrd)
    #         random.shuffle(randomOrd)
    #         MAE_random = calMAE(randomOrd, realOrd)
    #         print("MAE of the prediction is {}".format(MAE))
    #         print("MAE of a random guess is {}".format(MAE_random))

    #         KendallTau = calKendallTau(predOrd, realOrd)
    #         KendallTau_random = calKendallTau(randomOrd, realOrd)
    #         print("Kendall Tau distance of the prediction is {0:.3f}".format(KendallTau))
    #         print("Kendall Tau distance of a random guss is {0:.3f}".format(KendallTau_random))
    #         print()
            

    # # y = 1 - a * e^(-b * x^c)
    # plt.figure()
    # x = np.arange(0, 100, 1)
    # y = 1 - 10 * np.exp(-0.1 * np.power(x, 2))
    # plt.plot(x, y)
    # plt.show()


    # fit y = 1 - a * e^(-bx), use least square solution
    # satisfying result!
    # comNo = [105]
    # for i in comNo:
    #     powPara = 1.25

    #     x = np.array(comData[i]['x'])
    #     y = np.array(comData[i]['y'])

    #     l = len(x)
    #     x2 = x.reshape((-1, 1)) 

    #     x2 = np.power(x2, powPara)

    #     temp = np.ones((l, 1))
    #     A = np.hstack((x2, -temp))
    #     y2 = y
    #     y2[y2 >= 1] = 0.99      
    #     y2 = -np.log(1 - y2)
    #     y2 = y2.reshape((-1, 1))

    #     A = A[20:100, :]
    #     y2 = y2[20:100, :]
    #     params = np.linalg.inv(A.T@A)@A.T@y2
    #     b = params[0]
    #     a = np.exp(params[1])

    #     p = lambda a, b, x : 1 - a * np.exp(-b * np.power(x, powPara))
    #     y_fit = p(a, b, x)
    #     plt.plot(x, y, label = "original #{}".format(i))
    #     plt.plot(x, p(a, b, x), label = "fit #{}".format(i))
    #     plt.vlines(x[99], 0, 1, colors = "c", linestyles = "dashed")
    # plt.legend()
    # plt.show()

    # using polynomial fit, performance not so good 
    # plt.figure()
    # comNo = [50, 150, 350, 399]
    # for i in comNo:
    #     vertex_time = 0
    #     for idx, val in enumerate(comData[i]['y']):
    #         if np.allclose(val, 1):
    #             vertex_time = idx
    #     x = np.array(comData[i]['x'][:vertex_time+1])
    #     y = np.array(comData[i]['y'][:vertex_time+1])
    #     paras = np.polyfit(x, y, 2)
    #     p = np.poly1d(paras)
    #     plt.plot(x, y, label = "original #{}".format(i))
    #     plt.plot(x, p(x), label = "fit #{}".format(i))
    # plt.legend()
    # plt.show()

    # comNo = list(range(0, 400, 50))
    # # print several figure in this community
    # plt.figure()
    # for i in comNo:
    #     plt.plot(comData[i]['x'], comData[i]['y'], label = "community #{}".format(i))
    # plt.legend()
    # plt.show()

    # # print the first order difference of these communities
    # # use interval to smooth the figure (y[k] - y[k-interval])/(x[k] - x[k-interval])
    # plt.figure()
    # interval = 20
    # for i in comNo:
    #     x = comData[i]['x'][interval: ]
    #     delta_y = [(comData[i]['y'][k] - comData[i]['y'][k-interval])/(comData[i]['x'][k] - comData[i]['x'][k-interval]) \
    #         for k in range(interval, len(comData[i]['y']))]
    #     plt.plot(x, delta_y, label = "community #{}".format(i))
    # plt.legend()
    # plt.show()



    