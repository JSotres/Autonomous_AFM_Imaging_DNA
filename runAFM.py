import win32com.client
import os
import plasmidAnalysis
import json
import argparse

if __name__ == "__main__":

    # A json file, containing scan parameters, needs to be provided
    # as an input when running the file
    ap = argparse.ArgumentParser()
    ap.add_argument("-i",
                    "--input",
                    required=True,
                    help="input json file with scan parameters"
                    )
    args = vars(ap.parse_args())    

    # We read the json file where all parameters set by the user
    # are declared
    with open(args["input"]) as f:
        inputData = json.load(f)

    # Different addresses are available to access the Nanoscope COM server:
    # 1) IEaseOfUse, 2) IPlugInApi, 3) IzApi
    # each one allows to access different commands.
    # Below are the addresses in our case, note that these will be different
    # in your case if you intend to use this code. You will need to purchase
    # the Nanoscript feature from Bruker to get them. 
    # In our case we used the IEaseOfUse commands set.
    # We provide the corresponding address in the input json file, and
    # use it below to create an automation object with methods and properties
    # for controlling the AFM 
    xlApp = win32com.client.Dispatch(inputData["NanoScriptCode"])

    # We set in the AFM control software the scanning parameters,
    # which were read from the json file where this parameters were
    # provided and are at this stage in the dictionary inputData
    #
    # Number of lines
    setattr(xlApp,"ScanLines",inputData["scanLines"])
    # Number of columns
    setattr(xlApp,"SamplesPerLine",inputData["samplesPerLine"])
    # Lateral size (in microns)
    setattr(xlApp,"ScanSize",inputData["scanSize"])
    
    # Spatial coordinates for thpoint where engaging will be performed 
    setattr(xlApp,"XOffset",inputData["xOffset"])
    setattr(xlApp,"YOffset",inputData["yOffset"])
    # Folder where AFM images will be saved
    print(inputData["captureDir"])
    setattr(xlApp,"CaptureDir",inputData["captureDir"])
    # Name for the first image to be saved
    setattr(xlApp,"CaptureFileName",inputData["captureFile"])

    # We now create folders used to save and access images
    # used for the autonomous imaging algorithm. The name for these
    # folders are also provided in the json file containing input parameters
    try:
        os.mkdir(os.path.join(inputData["captureDir"],"All_Images"))
        os.mkdir(os.path.join(inputData["captureDir"],"Zoomed_Molecules"))
        os.mkdir(os.path.join(inputData["captureDir"],"Initial_Zoomed_Molecules"))
        os.mkdir(os.path.join(inputData["captureDir"],"All_Yolo_Images"))
        os.mkdir(os.path.join(inputData["captureDir"],"Molecule_Being_Followed"))
    except OSError:
        print ("Creation of folders failed")
    else:
        print ("Folders succesfully created")



    # We initialize two additional variables:
    #    area: identifies the location of the sample being scanned
    #    flag:
    area = 0
    flag = 0

    # Command for engaging the AFM
    xlApp.Engage()

    # Set point (deflection, amplitude, etc, depending on the operation mode)
    setattr(xlApp,"SetPoint_NV",inputData["setPoint"])

    # Scan Rate (in Hz)
    setattr(xlApp,"ScanRate",inputData["scanRate"])
    
    while area < 25:
        xlApp.ScanSkipToLine(40,1)
        capturefile1 = getattr(xlApp,"CaptureFileName")
        image = os.path.join(inputData["captureDir"],capturefile1)
        xlApp.Capture()
        sto = getattr(xlApp,"IsCapturedone")
        if sto == 1:
            result = plasmidAnalysis.plasmidAnalysis(image, flag, area)
            xoffset = result[0]
            yoffset = result[1]
            scansize = result[2]
            flag = result[3]
            area = result[4]
            setattr(xlApp,"ScanSize",scansize)
            setattr(xlApp,"XOffset",xoffset)
            setattr(xlApp,"YOffset",yoffset)



       
    xlApp.Withdraw()
    print("Experiment finished")
