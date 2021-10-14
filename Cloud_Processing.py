def PTYPE(column,indexs,Test,PtypeData):
    for station in column:
        sublist = PtypeData[PtypeData[2] == station].drop_duplicates().reset_index()
        count = 0
        for time in sublist[3]:
            #print(time)
            Test.loc[time,station] = sublist.loc[count,4]
            #print()
            count +=1
    return(Test)
	
def PTYPESECOND(column,indexs,Test,PtypeData):
    PtypeData = PtypeData[PtypeData.duplicated([3])]
    PtypeData = PtypeData.drop_duplicates()
    PtypeData = PtypeData[PtypeData[4] != "CLR"]
    print(PtypeData.shape)
    for station in column:
        sublist = PtypeData[PtypeData[2] == station].reset_index()
        count = 0
        for time in sublist[3]:
            Test.loc[time,station] = sublist.loc[count,4]
            count +=1
    return(Test)

def PTYPETHIRD(column,indexs,Test,PtypeData):
    PtypeData = PtypeData[PtypeData.duplicated([3],keep="last")]
    PtypeData = PtypeData[PtypeData[4]!= "CLR"]
    print(PtypeData.shape)
    for station in column:
        sublist = PtypeData[PtypeData[2] == station].reset_index()
        #sublist = sublist.drop_duplicates().reset_index()
        count = 0
        for time in sublist[3]:
            #print(time)
            Test.loc[time,station] = sublist.loc[count,4]
            #print()
            count +=1
    return(Test)

import os
import pandas as pd
os.chdir("/scratch/brown/brittoa/ERA5_data/") #location where data was first processed
PtypeFiles = []
for d in os.listdir(): #grad only CLOUD files
    if d.startswith('CLOUD'):
        PtypeFiles.append(d)  
for file in PtypeFiles[:]: 
    print(file)
    PtypeData = pd.read_csv(file, header = None, error_bad_lines=False) #import data
    PtypeData = PtypeData[PtypeData[4]!="VV"]
    TimeStamps = PtypeData[3].drop_duplicates().reset_index().drop(columns=["index"]) 
    Stations = PtypeData[2].drop_duplicates().sort_values().reset_index().drop(columns=["index"])
    column = [x for x in Stations[2]]
    indexs = [x for x in TimeStamps[3]]
    PtypeDataSlim = PtypeData.drop_duplicates()
    PtypeDataSlim = PtypeDataSlim[PtypeDataSlim[4]!="CLR"]
    Test = pd.DataFrame(columns = column,index=indexs) #construct square dataframe of data
    Test = PTYPE(column, indexs, Test, PtypeDataSlim) #first duplicate only
    Test = Test.fillna(1) #default cloud is clear skies
    #Test = Test.replace("CLR", 1)
    Test = Test.replace("FEW", 2) #replace strings with INT to help with model later
    Test = Test.replace("SCT", 3)
    Test = Test.replace("BKN", 4)
    Test = Test.replace("OVC", 5)
    Test = Test.replace("VV" , 6)
    os.chdir("..")
    os.chdir("ProcessedDataCLOUD/")
    Test.to_csv("CLOUDFIRSTTYPE"+file[5:]) #save first dataframe
    os.chdir("..")
    os.chdir("ERA5_data/")
    print("FIRST DONE")
    Test = PTYPESECOND(column, indexs, Test, PtypeData) #grab second cloud type
    #Test = Test.replace("CLR", 1)
    Test = Test.replace("FEW", 2)
    Test = Test.replace("SCT", 3)
    Test = Test.replace("BKN", 4)
    Test = Test.replace("OVC", 5)
    Test = Test.replace("VV" , 6)
    os.chdir("..")
    os.chdir("ProcessedDataCLOUD/")
    Test.to_csv("CLOUDSECONDTYPE"+file[5:])
    os.chdir("..")
    os.chdir("ERA5_data/")
    print("SECOND DONE")
    Test = PTYPETHIRD(column, indexs, Test, PtypeData) #grab third cloud type
    #Test = Test.replace("CLR", 1)
    Test = Test.replace("FEW", 2)
    Test = Test.replace("SCT", 3)
    Test = Test.replace("BKN", 4)
    Test = Test.replace("OVC", 5)
    Test = Test.replace("VV" , 6)
    os.chdir("..")
    os.chdir("ProcessedDataCLOUD/")
    Test.to_csv("CLOUDTHIRDTYPE"+file[5:])
    os.chdir("..")
    os.chdir("ERA5_data/")
    print("THIRD DONE")
