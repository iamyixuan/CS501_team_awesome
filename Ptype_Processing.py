import os
import pandas as pd

def PTYPE(column,indexs,Test,PtypeData):
    for station in column:
        sublist = PtypeData[PtypeData["STID"] == station].reset_index()
        count = 0
        for time in sublist["DDHHMM_UTC"]:
            #print(time)
            Test.loc[time,station] = sublist.loc[count,"PTYPE"]
            #print()
            count +=1
    return(Test)
  
  
PtypeFiles = []
for d in os.listdir():
    if d.startswith('test2'):
        PtypeFiles.append(d)  
for file in PtypeFiles[5:]:
    print(file)
    PtypeData = pd.read_csv(file)
    PtypeData = PtypeData.drop(columns=["MMDDYY_LST","HHMMSS_LST"])
    TimeStamps = PtypeData['DDHHMM_UTC'].drop_duplicates().sort_values().reset_index().drop(columns=["index"])
    Stations = PtypeData["STID"].drop_duplicates().sort_values().reset_index().drop(columns=["index"])
    column = [x for x in Stations["STID"]]
    indexs = [x for x in TimeStamps["DDHHMM_UTC"]]
    Test = pd.DataFrame(columns = column,index=indexs)
    Test = PTYPE(column, indexs, Test, PtypeData)
    Test = Test.fillna(0)
    Test = Test.replace("RA", 1)
    Test = Test.replace("SN", 2)
    Test = Test.replace("FZ", 3)
    Test = Test.replace("PL", 4)
    Test = Test.replace("UP", 5)
    os.chdir("..")
    os.chdir("ProcessedData/")
    Test.to_csv("PTYPE"+file[5:])
    os.chdir("..")
    os.chdir("ERA5_data/")
