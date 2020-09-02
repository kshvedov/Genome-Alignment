#---#---#---#--->
# Programmer: Konstantin Shvedov
# Class: CptS 471
# Programming Assignment: Project 1
# Date: 14/02/2020
# Description:
#	This program holds two algorithms
# 	Global Alignment and Local Alignment
# 	Both use affine gap penalty function
# 	there are no structs since this is python
# 	I did use numpy which uses c like arrays for
# 	efficiency
#---#---#---#---<
import re
import os
import sys
import time
import datetime
import numpy as np

# Tries openning file, if something wrong, prints error and exits
def GetFastaData(fname):
    try:
        fIn = open(fname)
        data = {}
        new = 0

        name = ""
        seq = ""

        for line in fIn:
            # If starts with arrow, record as name and expect sequence after 
            if line[0] == ">":
                new = 1
                name = line
                name = name.replace(">", "")
                name = name.split()
                name = name[0]

            # If empty line than sequence has ended
            elif (line[0] == "\n" or line[0] == " ") and new == 1:
                new = 0
                data[name] = seq
                #print(name+": ", len(seq))
                name = ""
                seq = ""

            elif new == 1:
                temp = line.replace("\n", "")
                seq += temp

        if name != "" and seq != "":
            data[name] = seq
            #print(name+": ", len(seq))

        fIn.close()

        if len(data) < 2:
            print("Not Enough Data!")
            sys.exit(0)

        return data

    except Exception as e:
        print(e)
        sys.exit(0)

# Reads Config File
def GetConfig(fname):
    try:
        fIn = open(fname)
        data = {}

        for line in fIn:
            if line[0] != " " and line[0] != "\n":
                # Some of this seems redundant but the example provided
                # Has 4 spaces for a tab. Replaces 2-6 spaces with single
                temp = line.replace("\n", "")
                temp = temp.replace("      ", " ")
                temp = temp.replace("     ", " ")
                temp = temp.replace("    ", " ")
                temp = temp.replace("   ", " ")
                temp = temp.replace("  ", " ")
                temp = re.split(" |\t", temp)
                data[temp[0]] = int(temp[1])

        fIn.close()
        return data

    except Exception as e:
        print(e)
        sys.exit(0)

# If doesnt exist set to defaults otherwise keep the same
# Match = 1
# Mismatch = -2
# H = -5
# G = -1
def SetConfig(config):
    config["match"] = config.get("match", 1)
    config["mismatch"] = config.get("mismatch", -2)
    config["h"] = config.get("h", -5)
    config["g"] = config.get("g", -2)

    return config

# Turn first two inputs of dict as str1 and str2
# Returns the array and name of string
def DataToStrArr(data):
    str1 = []
    str2 = []
    count = 0
    for k in data.keys():
        if count == 0:
            nStr1 = k
            str1 = " " + data[k]
            str1 = str1.upper()
            str1 = list(" " + data[k])
            str1 = np.asarray(str1)
        if count == 1:
            nStr2 = k
            str2 = " " + data[k]
            str2 = str2.upper()
            str2 = list(" " + data[k])
            str2 = np.asarray(str2)
        count += 1

    return nStr1, nStr2, str1, str2



# Function to initialise the table for Global Alignment
# Table structure:
# Size of i x j
# Index 0 is the Substitution case max value from previous cell
# Index 1 is the Insertion    case max value from previous cell
# Index 2 is the Deletion     case max value from previous cell
# Index 3 is the index of the max value within this cell to
# Index 4 is used to identify where previous max came from
#   for an easier retrace
# avoid max calculation on the retrace
def InitTableGlobal(dptable, lStr1, lStr2, h, g):
    dptable[0][0][0] = 0

    for i in range(lStr1):
        dptable[i][0][1] = h + i * g

    for j in range(lStr2):
        dptable[0][j][2] = h + j * g

    return dptable



# Function to initialise the table for Global Alignment
# Table structure:
# Size of i x j
# Index 0 is the Substitution case max value from previous cell
# Index 1 is the Insertion    case max value from previous cell
# Index 2 is the Deletion     case max value from previous cell
# Index 3 is the index of the max value within this cell to
# Index 4 is used to identify where previous max came from
#   for an easier retrace
# avoid max calculation on the retrace
def InitTableLocal(dptable, lStr1, lStr2, h, g):
    dptable[0][0][0] = 0
    dptable[0][0][3] = 0

    for i in range(lStr1):
        dptable[i][0][1] = 0
        dptable[i][0][3] = 1

    for j in range(lStr2):
        dptable[0][j][2] = 0
        dptable[0][j][3] = 2

    return dptable



# This function populates the table using the:
# Needleman-Wunsch algorithm of Optimal Global Alignmnet
# Using the affine gap penalty function
# (Some parts of this function can be split up to be reusable, but
#  that means it will be slower, therefore I am keeping it all local)
def CalculateGlobalTable(dptable, str1, str2, match, mismatch, h, g):
    start = time.time()

    for i in range(1, len(str1)):
        if (i % 100) == 0:
            end = time.time()
            print("Time Taken for ", i ,": ", str(end-start))

        for j in range(1, len(str2)):

            # S --> Substitution
            s = np.copy(dptable[i-1][j-1][0:3])
            s += Match(str1[i], str2[j], match, mismatch)
            smax = np.amax(s)
            dptable[i][j][0] = smax

            # I --> Insertion
            ins = np.copy(dptable[i-1][j][0:3])
            ins[0:3] += h + g
            ins[1] -= h
            insmax = np.amax(ins)
            dptable[i][j][1] = insmax

            # D --> Deletion
            d = np.copy(dptable[i][j-1][0:3])
            d[0:2] += h + g
            d[2] += g
            dmax = np.amax(d)
            dptable[i][j][2] = dmax

            # Saving max index for ease of traversal back
            dptable[i][j][3] = np.argmax(dptable[i][j][0:3])
            
            
            # If was an Insertion check if previous was also
            if dptable[i][j][3] == 1.0:
                temp = np.argmax(ins)
                dptable[i][j][4] = temp

            # If was an Deletion check if previous was also
            elif dptable[i][j][3] == 2.0:
                temp = np.argmax(d)
                dptable[i][j][4] = temp

    end = time.time()
    print("Time Taken: ", str(end-start))
    return dptable



# This function populates the table using the:
# Needleman-Wunsch algorithm of Optimal Global Alignmnet
# Using the affine gap penalty function
# (Some parts of this function can be split up to be reusable, but
#  that means it will be slower, therefore I am keeping it all local)
def CalculateLocalTable(dptable, str1, str2, match, mismatch, h, g):
    start = time.time()

    for i in range(1, len(str1)):
        if (i % 100) == 0:
            end = time.time()
            print("Time Taken for ", i ,": ", str(end-start))

        for j in range(1, len(str2)):
            # S --> Substitution
            s = np.copy(dptable[i-1][j-1][0:4])
            s[0:3] += Match(str1[i], str2[j], match, mismatch)
            s[3] = 0
            smax = np.amax(s)
            dptable[i][j][0] = smax

            # I --> Insertion
            ins = np.copy(dptable[i-1][j][0:4])
            ins[0:3] += h + g
            ins[1] -= h
            ins[3] = 0
            insmax = np.amax(ins)
            dptable[i][j][1] = insmax

            # D --> Deletion
            d = np.copy(dptable[i][j-1][0:4])
            d[0:3] += h + g
            d[2] += g
            d[3] = 0
            dmax = np.amax(d)
            dptable[i][j][2] = dmax

            # Saving max index for ease of traversal back
            dptable[i][j][3] = np.argmax(dptable[i][j][0:4])

            # If was an Insertion check if previous was also
            if dptable[i][j][3] == 1.0:
                temp = np.argmax(ins)
                dptable[i][j][4] = temp

            # If was an Deletion check if previous was also
            elif dptable[i][j][3] == 2.0:
                temp = np.argmax(d)
                dptable[i][j][4] = temp

                
    end = time.time()
    print("Time Taken: ", str(end-start))

    return dptable

# This function is the one used in notes, will return an int
# If letters are the same, will return match score
# If letters are different, will return mismatch score
def Match(ai, bj, match, mismatch):
    if ai == bj:
        return match
    else:
        return mismatch

# This function is used for the Global Retrace
def RetraceGlobalAlignment(dptable, str1, str2, h, g):
    i = len(str1) - 1
    j = len(str2) - 1

    aString = []
    countDict = {"matches": 0, "mismatches": 0, "gaps": 0, "opening gaps": 0}

    while i >= 1 and j >= 1:
        i, j, aString, countDict = RetraceAlgorithm(dptable, str1, str2, i, j, aString, countDict, h, g)

    return  aString, countDict

# This function is used for the Local Retrace
def RetraceLocalAlignment(dptable, str1, str2, h, g):
    # Return Location of Max Value from the 3d array
    i, j , temp = np.unravel_index(np.argmax(dptable, axis=None), dptable.shape)

    aString = []
    countDict = {"matches": 0, "mismatches": 0, "gaps": 0, "opening gaps": 0}

    # While the max value in cell is not 0
    while dptable[i][j][int(dptable[i][j][3])] > 0.0:
        i, j, aString, countDict = RetraceAlgorithm(dptable, str1, str2, i, j, aString, countDict, h, g)

    return  aString, countDict

# The Retrace algorithm was so similar that both local and Global could reuse this portion
def RetraceAlgorithm(dptable, str1, str2, i, j, aString, countDict, h, g):
    loop = False
    if dptable[i][j][3] == 0:   # Substitution Case
        if str1[i] == str2[j]:
            countDict["matches"] += 1
            aString.append([str1[i], "|", str2[j], "SMA", int(dptable[i][j][0]), i, j])
        else:
            countDict["mismatches"] += 1
            aString.append([str1[i], " ", str2[j], "SMI", int(dptable[i][j][0]), i, j])

        i -= 1
        j -= 1

    elif dptable[i][j][3] == 1: # Insertion Case
        countDict["gaps"] += 1
        aString.append([str1[i], " ", "-", "I", int(dptable[i][j][1]), i, j])
        i -= 1
        while dptable[i+1][j][4] == 1.0: # Checks if previous was also Insertion
            countDict["gaps"] += 1
            aString.append([str1[i], " ", "-", "I", int(dptable[i][j][1]), i, j])
            i -= 1
            loop = True
        if loop:
            countDict["gaps"] += 1
            aString.append([str1[i], " ", "-", "I", int(dptable[i][j][1]), i, j])
            i -= 1

    elif dptable[i][j][3] == 2: # Deletion Case
        countDict["gaps"] += 1
        aString.append(["-", " ", str2[j], "D", int(dptable[i][j][2]), i, j])
        j -= 1
        while dptable[i][j+1][4] == 2.0: # Checks if previous was also Deletion
            countDict["gaps"] += 1
            aString.append(["-", " ", str2[j], "D", int(dptable[i][j][2]), i, j])
            j -= 1
            loop = True
        if loop:
            countDict["gaps"] += 1
            aString.append(["-", " ", str2[j], "D", int(dptable[i][j][2]), i, j])
            j -= 1

    return i, j, aString, countDict

# Prints results in the wanted output format
def PrintResults(nStr1, nStr2, outList, cd, fOut):
    out = open(fOut, 'a')
    i = len(outList)

    ls1 = len(nStr1)
    ls2 = len(nStr2)
    middle = ""

    #Evens out Names so output has a very strict format
    if ls1 >= ls2:
        nStr1 = nStr1.ljust(ls1 + 4, " ")
        nStr2 = nStr2.ljust(ls1 + 4, " ")
        middle = middle.ljust(ls1 + 4, " ")
    if ls1 < ls2:
        nStr1 = nStr1.ljust(ls2 + 4, " ")
        nStr2 = nStr2.ljust(ls2 + 4, " ")
        middle = middle.ljust(ls1 + 4, " ")


    count = 1
    gCount = 0

    line1 = ""
    line2 = ""
    line3 = ""

    prev = ""
    cur = ""

    # Creates lines of three where chars are added one by one to
    # Look like the desired format (no more than 60 chars of
    # sequence per a line)
    for c in range(i - 1, -1, -1):
        cur = outList[c][3]
        if (cur == "D" or cur == "I") and (cur != prev):
            cd["opening gaps"] += 1
        prev = cur
        
        if count == 1:
            line1 = nStr1 + str(outList[c][5]).ljust(6, " ")
            line2 = middle+ "".ljust(6, " ")
            line3 = nStr2 + str(outList[c][6]).ljust(6, " ")
        line1 += outList[c][0]
        line2 += outList[c][1]
        line3 += outList[c][2]

        if count < 60:
            count += 1
        else:
            count = 1
            print(line1, " ", outList[c][5])
            print(line2)
            print(line3, " ", outList[c][6],"\n")
            print(line1, " ", outList[c][5], file = out)
            print(line2, file = out)
            print(line3, " ", outList[c][6],"\n", file = out)
            line1 = ""
            line2 = ""
            line3 = ""
        
        gCount += 1

    print(line1, " ", outList[0][5])
    print(line2)
    print(line3, " ", outList[0][6], "\n\n")
    print(line1, " ", outList[0][5], file = out)
    print(line2, file = out)
    print(line3, " ", outList[0][6], "\n\n", file = out)


    # This prints out all the desired information about the matches in the format required
    print("Report:\n")
    print("Report:\n", file = out)

    print("Global optimal score = ", outList[0][4], "\n")
    print("Global optimal score = ", outList[0][4], "\n", file = out)

    print("Number of: matches = ", cd["matches"], ", mismatches = ", cd["mismatches"],
          ", gaps = ", cd["gaps"], ", opening gaps = ", cd["opening gaps"], "\n")
    print("Number of: matches = ", cd["matches"], ", mismatches = ", cd["mismatches"],
          ", gaps = ", cd["gaps"], ", opening gaps = ", cd["opening gaps"], "\n", file = out)

    print("Identities = ", str(cd["matches"]) + "/" + str(len(outList)),
          " (" + str(int((cd["matches"]/len(outList))*100)) + "%),  Gaps = ", end = "")
    print(str(cd["gaps"]) + "/" + str(len(outList)),
          " (" + str(int((cd["gaps"]/len(outList))*100)) + "%)")

    print("Identities = ", str(cd["matches"]) + "/" + str(len(outList)),
          " (" + str(int((cd["matches"]/len(outList))*100)) + "%),  Gaps = ", end = "", file = out)
    print(str(cd["gaps"]) + "/" + str(len(outList)),
          " (" + str(int((cd["gaps"]/len(outList))*100)) + "%)", file = out)

    out.close()

    return

# SetUp for Both Alignments
def ExecuteAlignment(data, type, config, fOut):
    match = config["match"]
    mismatch = config["mismatch"]
    h = config["h"]
    g = config["g"]

    #---#---#---#---#--->
    # Print configs to screen and outfile
    out = open(fOut, 'a')
    print("Scores:\t\tmatch = ", match, ", mismatch = ", mismatch,
          ", h = ", h, ", g = ", g, "\n")
    print("Scores:\t\tmatch = ", match, ", mismatch = ", mismatch,
          ", h = ", h, ", g = ", g, "\n", file = out)
    out.close()
    #---#---#---#---#---<

    # Turn first two inputs of dict as str1 and str2
    nStr1, nStr2, str1, str2 = DataToStrArr(data)
    lStr1 = len(str1)
    lStr2 = len(str2)

    #---#---#---#---#--->
    # Print information about strings to screen and outfile
    # -1 for length due to adding an empty space in DataToStrArr
    out = open(fOut, 'a')
    print("Sequence 1 = \"" + nStr1 + "\", length = ", len(str1) - 1, " characters")
    print("Sequence 1 = \"" + nStr1 + "\", length = ", len(str1) - 1, " characters", file = out)
    print("Sequence 2 = \"" + nStr2 + "\", length = ", len(str2) - 1, " characters\n")
    print("Sequence 2 = \"" + nStr2 + "\", length = ", len(str2) - 1, " characters\n", file = out)
    out.close()
    #---#---#---#---#---<

    #---#---#---#---#--->
    # Sequence 1 Down   (i)
    # Sequence 2 Across (j)
    # Table structure:
    # Size of i x j of 4 int arrays
    # Index 0 is the Substitution case max value from previous cell
    # Index 1 is the Insertion    case max value from previous cell
    # Index 2 is the Deletion     case max value from previous cell
    # Index 3 is the index of the max value within this cell to
    # avoid max calculation on the retrace
    # All values initialised to -infinity
    dptable = np.full((lStr1, lStr2, 5), np.NINF)
    #---#---#---#---#---<

    # Checks what functions to run, Global or Local
    # If type is not 0 or 1 will quit
    outList = []
    if type ==  0:
        dptable = InitTableGlobal(dptable, lStr1, lStr2, h, g)
        dptable = CalculateGlobalTable(dptable, str1, str2, match, mismatch, h, g)
        outList, countDict = RetraceGlobalAlignment(dptable, str1, str2, h, g)
    elif type == 1:
        dptable = InitTableLocal(dptable, lStr1, lStr2, h, g)
        dptable = CalculateLocalTable(dptable, str1, str2, match, mismatch, h, g)
        outList, countDict = RetraceLocalAlignment(dptable, str1, str2, h, g)
    else:
        print("Only Global (0) and Local (0) allighnment available")
        sys.exit(0)

    PrintResults(nStr1, nStr2, outList, countDict, fOut)

    return

# fname contains s1 and s2
# type is 0 for global and 1 for local
# params is a file name or empty
def ExecuteAlgorithm(fname, type, params):
    # Create an out txt file to print to
    fOut = CreateOutputName()

    # Retrieves Data From File
    data = GetFastaData(fname)
    config = {}

    # If file with config is set uses the values from there
    if params != "":
        config = GetConfig(params)

    # Sets the remaining values to default ones
    config = SetConfig(config)

    # Main algorithm executed
    # If type is 0 --> Global
    # If type is 1 --> Local
    ExecuteAlignment(data, type, config, fOut)


# Creates a name for an output file using the time it was created
def CreateOutputName():
    fName = "rs" # rs is results
    fNameNum = 0
    txt = ".txt" # can technically be any kind, txt is easily readable though
    fName = fName + (datetime.datetime.now()).strftime("_%d-%m-%Y_%I-%M%p") + txt

    return fName



# A small function to test if the string represents an int
def IsInt(i):
    try: 
        int(i)
        return True
    except ValueError:
        return False


# Main for python program, it checks everything to make sure the
# Correct arguments are passed
if __name__ == '__main__':
    argNum = len(sys.argv)

    # Check to see that all arguments passed are of the correct format
    if argNum > 4 or argNum < 3:
        print("Wrong Number of Arguments!")
        sys.exit(0)
    elif IsInt(sys.argv[2]) == False:
        print("Second Argument is 0 for Global Alignment or 1 for Local Alignment")
        sys.exit(0)
    elif not os.path.isfile(sys.argv[1]):
        print("File in first argument doesnt exist!")
        sys.exit(0)

    # Depending on the number of arguments launches the execution algorithem with
    # different inputs
    if argNum == 3:
        ExecuteAlgorithm(sys.argv[1], int(sys.argv[2]), "")
    elif argNum == 4:
        if os.path.isfile(sys.argv[1]):
            ExecuteAlgorithm(sys.argv[1], int(sys.argv[2]), sys.argv[3])
        else:
            print("File in argument 3 doesnt exist!")
            sys.exit(0)
    else:
        print("Wrong Number of Arguments!")
        sys.exit(0)

    #---#---#---#--->
    #Below commands were used for testing the algorithem from an IDE
    #Above code is used when launching from command propt with arguments
    #---#---#---#---<

    #ExecuteAlgorithm("test2.fasta", 0, "parameters.config")
    #ExecuteAlgorithm("test2.fasta", 1, "parameters.config")
    #ExecuteAlgorithm("Opsin1_colorblindness_gene.fasta", 0, "parameters.config")
    #ExecuteAlgorithm("Human-Mouse-BRCA2-cds.fasta", 0, "parameters.config")
    #ExecuteAlgorithm("Opsin1_colorblindness_gene.fasta", 1, "parameters.config")
    #ExecuteAlgorithm("Human-Mouse-BRCA2-cds.fasta", 1, "parameters.config")