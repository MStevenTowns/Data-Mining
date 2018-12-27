import csv
import time
import numpy as np

intergratedData=dict()
posData=[]
posdict={"NotSpecified":0,"Standing":1,"Sitting":2,"Walking":3,"Running":4,"Climbing(up)":5,"Climbing(down)":6}
posIndexdict={0:"NotSpecified",1:"Standing",2:"Sitting",3:"Walking",4:"Running",5:"Climbing(up)",6:"Climbing(down)"}
csvtitle=["Date&Time","attr_x","attr_y","attr_z","attr_azimuth","attr_pitch","attr_roll","Posture_Label"]

def readAccData():
    global intergratedData
    with open('../subject1/data/acc_csv/SensorAccelerometerData_labeled_day2.csv', newline='') as csvfile:
    #with open('../subject1/data/acc_csv/SensorExample.csv', newline='') as csvfile:
        csvreader=csv.DictReader(csvfile,delimiter=',',quotechar='|')
        for row in csvreader:
            intergratedData[row['attr_time']]=row
            
def readOriData():
    global intergratedData
    with open('../subject1/data/ori_csv/SensorOrientationData_labeled_day2.csv', newline='') as csvfile:
        csvreader=csv.DictReader(csvfile,delimiter=',',quotechar='|')
        for row in csvreader:
            if(row['attr_time'] in intergratedData.keys()):
                intergratedData[row['attr_time']]['attr_azimuth']=row['attr_azimuth']
                intergratedData[row['attr_time']]['attr_pitch']=row['attr_pitch']
                intergratedData[row['attr_time']]['attr_roll']=row['attr_roll']
            #intergratedData[row['attr_time']]['attr_azimuth']
            
def readEnvData():
    global intergratedData
    with open('../subject1/data/acc_csv/SensorExample.csv', newline='') as csvfile:
        csvreader=csv.DictReader(csvfile,delimiter=',',quotechar='|')
        for row in csvreader:
            intergratedData[row['attr_time']]=row
            
def readPosData():
    global posData
    with open('../subject1/data/posture.csv', newline='') as csvfile:
        csvreader=csv.reader(csvfile,delimiter=',',quotechar='|')
        for row in csvreader:
            posData.append(row)

def toint(tempstr):
    tempstr=tempstr.split(":")
    tempstr[2]=tempstr[2].split(".")[0]
    return int(tempstr[0])*3600+int(tempstr[1])*60+int(tempstr[2])

def tostr(time):
    hour=time//3600
    minute=(time-hour*3600)//60
    second=time%60
    return ("%02d:%02d:%02d"%(hour,minute,second))

def findPos(date,time):
    for i in range(1,len(posData)):
        [sdate,stime]=posData[i][3].split(" ")
        [edate,etime]=posData[i][4].split(" ")
        if sdate==date and edate==date and toint(stime)<=time and toint(etime)>=time:
            return posdict[posData[i][5]]
    return 0

def transferData():
    dealedData=[]
    labelData=[]
    dataDict=dict()
    currentdata=None
    olddata=None
    for datadate in intergratedData.keys():
        if(len(intergratedData[datadate])>10):
            [date,time]=datadate.split(" ")
            time=toint(time)
            if(time not in dataDict.keys()):
                currentdata=np.array([float(intergratedData[datadate]['attr_x']),
                             float(intergratedData[datadate]['attr_y']),
                             float(intergratedData[datadate]['attr_z']),
                             float(intergratedData[datadate]['attr_azimuth']),
                             float(intergratedData[datadate]['attr_pitch']),
                             float(intergratedData[datadate]['attr_roll']),
                             ])
                if olddata is not None:
                    newdata=currentdata-olddata
                    dataDict[time]=newdata 
                    dealedData.append(list([date+" "+tostr(time)])+list(newdata)+list([posIndexdict[findPos(date,time)]]))
                    #dealedData.append(list(newdata))
                    labelData.append(findPos(date,time))
                olddata=currentdata
    return dealedData,labelData

def writeCSV():
    dealedData,labelData=transferData()
    with open('../subject1/data/IntergratedDataDay2.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(csvtitle)
        for row in dealedData:
            csvwriter.writerow(row)
    print("Write CSV File Successfully.")

def run():
    readAccData()
    readOriData()
    readPosData()
    writeCSV()
    #data=transferData()

def test():
    readPosData()
    a="11.03.15 07:58:34.941"
    [date,time]=a.split(" ")
    time="07:58:02"
    print(tostr(toint(time)))

    
def readData():
    data=[]
    lab=[]
    with open('../subject1/data/IntergratedData2.csv', newline='') as csvfile:
        csvreader=csv.reader(csvfile,delimiter=',',quotechar='|')
        for row in csvreader:
            data.append(row[1:7])
            lab.append(row[7])
            break
        print(data)
        print(lab)
run()
#data,lab=transferData()
#print(len(lab))
#test()
#readData()
