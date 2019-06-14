from Problem1_new import extractGroup, myCurve, train, calMAE, calKendallTau, findNearestTimeStamp
import numpy as np
import matplotlib.pyplot as plt
import random

def readInfectionData(path):
    f =  open(path)
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
    return comData, timeStamp

def readDiagnosisData(path):
    f2 = open(path)
    lines2 = f2.readlines()
    records2 = [line.split() for line in lines2]
    diagData = {}
    timeStamp2 = []

    for record in records2:
        time = float(record[2])
        timeStamp2.append(time)
        groupNum = extractGroup(record[1])
        if groupNum not in diagData:
            diagData[groupNum] = {'x': [time], 'y': [1/500]}
        else:
            diagData[groupNum]['x'].append(time)
            lastNum = diagData[groupNum]['y'][-1]
            diagData[groupNum]['y'].append(lastNum + 1/500)
    f2.close()
    return diagData, timeStamp2


if __name__ == "__main__":
    # # read data from two datasets
    # comData, timeStamp = readInfectionData("./62_transmission_network.txt")
    # diagData, timeStamp2 = readDiagnosisData("./62_transmission_network.diagnosis.txt") 

    # # first step is to train a' = m1 * a + n1 and b' = m2 * b + n2
    # infection_a = []
    # infection_b = []
    # diagnose_a = []
    # diagnose_b = []

    # comNo = list(range(380))
    # trainTime = 20 # use the first 30 years to train
    # for currCom in comNo:
    #     idx = findNearestTimeStamp(comData, currCom, trainTime)
    #     a, b = train(np.array(comData[currCom]['x'][:idx]), np.array(comData[currCom]['y'][:idx]))
    #     infection_a.append(a)
    #     infection_b.append(b)
        
    #     idx2 = findNearestTimeStamp(diagData, currCom, trainTime)
    #     a2, b2 = train(np.array(diagData[currCom]['x'][:idx2]), np.array(diagData[currCom]['y'][:idx2]))
    #     diagnose_a.append(a2)
    #     diagnose_b.append(b2)
    # # print(comData[10]['x'])

    # plt.figure()
    # plt.title("diagnosis_a vs infection_a")
    # plt.plot(diagnose_a, infection_a, 'o')
    # plt.xlabel("diagnosis_a")
    # plt.ylabel("infection_a")
    # plt.savefig("./Figures/diagnosis_a vs infection_a")
    # plt.show()
    
    # plt.figure()
    # plt.title("diagnosis_b vs infection_b")
    # plt.plot(diagnose_b, infection_b, 'o')
    # plt.xlabel("diagnosis_b")
    # plt.ylabel("infection_b")
    # plt.savefig("./Figures/diagnosis_b vs infection_b")
    # plt.show()

    # res1 = np.polyfit(np.array(diagnose_a[:280]), np.array(infection_a[:280]), 1) # 1 order polynomial fit
    # res2 = np.polyfit(np.array(diagnose_b[:280]), np.array(infection_b[:280]), 1)

    # m1 = res1[0]
    # n1 = res1[1]
    # m2 = res2[0]
    # n2 = res2[1]
    # print(m1, n1, m2, n2)

    # after the former step, we have already got the parameters
    # thus, we get the transformation equation between diagnosis curve and infection curve
    m1 = 0.8159286373110696
    n1 = 0.046177469227587684
    m2 = 1.5605843099216667 
    n2 = -0.020509224915104696

    # now do the prediction 
    comData, timeStamp = readInfectionData("./transmission_network.txt")
    diagData, timeStamp2 = readDiagnosisData("./transmission_network.diagnosis.txt") 

    comTestNo = list(range(400))
    timeNum = len(timeStamp)

    timeInterval = 0.5 # predict every #(0.5) years
    trainTime = [6, 8, 10]  # use the first # years to train
    predictionSlot = 10 # predict for the next # years
    parameters = {}
    

    for aTrainTime in trainTime:
        print("Now use {} years diagnosis data to train".format(aTrainTime))
        predCurve = []
        randomCurve = []

        predCurve_MAE = []
        randomCurve_MAE = []

        predictionTime = [aTrainTime + (i+1) * timeInterval for i in range(int(predictionSlot / timeInterval))]
        
        # train model
        for currCom in comTestNo:
            idx = findNearestTimeStamp(diagData, currCom, aTrainTime)
            a, b = train(np.array(diagData[currCom]['x'][:idx]), np.array(diagData[currCom]['y'][:idx]))

            print(idx)
            print(diagData[currCom]['y'][idx])
            # transform the diagnosis curve to infection curve
            aa = m1 * a + n1
            bb = m2 * b + n2
            parameters[currCom] = aa, bb
            
        # prediction step
        for aPredictionTime in predictionTime:
            predRes = []
            realRes = []

            print("predict at time {}".format(aPredictionTime))
            for currCom in comTestNo:
                a, b = parameters[currCom]  
                pred = myCurve(a, b, aPredictionTime)
                pred_init = myCurve(a, b, aTrainTime)

                pred_inc = (pred - pred_init) / pred_init
                predRes.append((currCom, pred_inc))
                
                train_idx = findNearestTimeStamp(comData, currCom, aTrainTime)
                pred_idx = findNearestTimeStamp(comData, currCom, aPredictionTime)
                real_inc = (comData[currCom]['y'][pred_idx] - comData[currCom]['y'][train_idx])/ \
                    comData[currCom]['y'][train_idx]
                realRes.append((currCom, real_inc))
            
            # calculate MAE and KendallTau
            predRes = sorted(predRes, key = lambda x : (-x[1], x[0]))
            realRes = sorted(realRes, key = lambda x : (-x[1], x[0]))
            # print(predRes)
            # print(realRes)
                        
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
        plt.title("Use {} years diagnosis data to train".format(aTrainTime))
        plt.plot(predictionTime, predCurve, 'o', label = "prediction")
        plt.plot(predictionTime, randomCurve, 'o', label = "random")
        plt.xlabel("time/year")
        plt.ylabel("Kendall Tau distance of the rank")
        plt.legend()
        # plt.show()
        plt.savefig("problem2_KDT_{}years".format(aTrainTime))

        plt.figure()
        plt.title("Use {} years diagnosis data to train".format(aTrainTime))
        plt.plot(predictionTime, predCurve_MAE, 'o', label = "prediction")
        plt.plot(predictionTime, randomCurve_MAE, 'o', label = "random")
        plt.xlabel("time/year")
        plt.ylabel("MAE of the rank")
        plt.legend()
        # plt.show()
        plt.savefig("problem2_MAE_{}years".format(aTrainTime))







        











    # comTestNo = list(range(400))
    # timeNum = len(timeStamp2)

    # trainPercent = [0.2, 0.3, 0.4, 0.5, 0.6]
    # testNum = 5 # how many tests we will do

    # for aTrainPercent in trainPercent:
    #     print("### {} of the dataset is used to train ###".format(aTrainPercent))
    #     trainTime = timeStamp2[int(timeNum * aTrainPercent)]
    #     testPercent = [aTrainPercent + (i+1) * (1-aTrainPercent)/(testNum+1) for i in range(testNum)]

    #     for aTestPercent in testPercent:
    #         print("Test point at {0:.3f}".format(aTestPercent))
    #         predRes = []
    #         realRes = []

    #         testTime = timeStamp2[int(timeNum * aTestPercent)]
    #         for currCom in comTestNo:
    #             idx = findNearestTimeStamp(comData, currCom, trainTime)
    #             a, b = train(np.array(diagData[currCom]['x'][:idx]), np.array(diagData[currCom]['y'][:idx]))
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



    # comNo = [20, 29, 388, 299]
    # # print several figure in this community
    # plt.figure()
    # for i in comNo:
    #     plt.plot(comData[i]['x'], comData[i]['y'], label = "community #{} infected time".format(i))
    #     plt.plot(diagData[i]['x'], diagData[i]['y'], label = "community #{} diagnosis time".format(i))
    # plt.legend()
    # plt.show()


    