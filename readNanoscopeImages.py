###############################################################################
# Import of neccesary packages
###############################################################################
import re
import numpy as np
import argparse
import io
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


class NanoscopeImage():
    '''
    Class that open and read Nanoscope Image Files.
    
    Attributes:
        file_name: Name of the Nanoscope Force Volume File
        database_name: Name of the database.
        header_end: For checking whether the end of the Force Volume
                    file header has been reached.
        eof: For checking whether the end of the Force Volume file
             has been reached.
        headerParameters: Dictionary for parameters of relevance within
                          the Force Volume file header.
        *** Attributes specific for the Force Volume Measurement ***
        numberOfMapRows: Number of rows (fast axis) in the Force
                         Volume measurement.
        numberOfMapColumns: Number of columns (slow axis) in the Force
                            Volume measurement.
        mapLength: Lateral dimension (in nm) of the scanned area.
        rampLength: Ramped distance (in nm).
        numberOfRampPoints: number of points in each ramp.
        scanIncrement: distance between ramp points.
        pixelLengthColumn: Lateral dimension of a pixel in the slow axis.
        pixelLengthRow: Lateral dimension of a pixel in the fast axis.
        *** Numpy arrays where the data read from the FV file are stored ***
        FVDataArray: For storing FV data.
        topographyArray: For storing topography(height) data.
        *** Connector anc cursor for sqlite ***
        connector
        cursor
    Methods:
        __init__()
        readHeader()
        searchForParameters()
        searchForHeaderEnd()
        headerToParameters()
        readTopography()
        readFV()
        connectToDataBase()
        checkTableExists()
        closeDataBaseConnection()
        createTables()
        populateTables()
    '''

    def __init__(self, file_name):
        '''
        Initializes an object of the class NanoscopeForceVolumeFiles,
        uses it for reading the data in the parsed Force Volume file
        and saves it in a sqlite database.
        Input parameters:
        file_name: name of the Force Volume file.
        database_name: name of the sqlite database
        '''

        # We initialize the attribute headerParameters, a dictionary
        # with keys corresponding to strings that identify the lines
        # in the Force Volume file header with relevant information
        self.headerParameters = {'Data offset':[], 'Data length':[],
                                 'Samps/line':[], 'Number of lines':[],
                                 'Scan Size':[], 'Line Direction':[],
                                 'Valid data len X':[], 'Valid data len Y':[],
                                 '2:Image Data':[], 'Z magnify':[],
                                 '@2:Z scale':[], '@Sens. Zsens':[],
                                 '@2:AFMSetDeflection':[], 'Bytes/pixel':[],
                                 'X Offset':[],  'Y Offset':[]
                                 }

        # At the beginning we are not at the end of the header
        # or at the endof the file
        self.header_end = 0
        self.eof = 0

        # Name of the Force Volume file
        self.file_name = file_name

        self.Image = []
        
    def readHeader(self):
        '''
        Reads the header of the Force Volume File file_name
        '''
        file = open(self.file_name, 'r', encoding='cp1252')

        # Keep reading the file line by line until the end of
        # the header (or the end of the file) is reached.
        # For each line, check whether it contains the keys
        # of headParameters, and if so populate their values, by
        # calling to searchForParameters(). Then, check if the end
        # of the header has been reached by calling searchForHeaderEnd()
        while (not self.header_end) and (not self.eof):
            for line in file:
                self.searchForParameters(line)
                self.searchForHeaderEnd(line, r'\*File list end')
                if self.header_end == 1:
                    break
            else:
                self.eof = 1
        file.close()

    def searchForParameters(self, _line):
        '''
        Identifies whether the input string, _line, contains one of the
        keys of headParameters. If so, pupulates its values with numbers
        contained in _line as well.
        '''
        for key in self.headerParameters:
            if re.search(re.escape(key), _line):
                # print(_line)
                if key == 'Line Direction':
                    searchString = re.findall(r'\w+$', _line)
                    searchString = searchString[0]
                    self.headerParameters[key].append(searchString)
                elif key == '2:Image Data':
                    searchString = re.split(r'"', _line)
                    searchString = searchString[-2]
                    self.headerParameters[key].append(searchString)
                elif key == 'Bytes/pixel':
                    numbers = re.findall(r'\d+$', _line)
                    self.headerParameters[key].append(int(numbers[0]))
                else:
                    numbers = re.findall(r'-?\d+\.?\d+', _line)
                    if numbers == []:
                        numbers = [0]
                    # If _line contains the strings 'LSB' or '@', only populate
                    # the key value with the last number from _line. If not,
                    # populate it with all numbers.
                    if re.search(r'LSB', _line) or re.search(r'@', _line):
                        self.headerParameters[key].append(float(numbers[-1]))
                    else:
                        for number in numbers:
                            self.headerParameters[key].append(float(number))

    def searchForHeaderEnd(self, _line, _string):
        '''
        Checks if the end of the header has been reached
        '''
        if re.search(r'\*File list end', _line):
            self.header_end = 1
        else:
            self.header_end = 0

    def readImages(self):
        file = open(self.file_name, 'rb')
        for i in range(len(self.headerParameters['Data offset'])):
            self.Image.append({
                'Channel': self.headerParameters['2:Image Data'][i],
                'Line Direction': self.headerParameters['Line Direction'][i],
                'Image Data': np.empty([
                    int(self.headerParameters['Samps/line'][i]),
                    int(self.headerParameters['Number of lines'][i])
                ]),
                'Processed Image Data': np.empty([
                    int(self.headerParameters['Samps/line'][i]),
                    int(self.headerParameters['Number of lines'][i])
                ]),
                'Rows': int(self.headerParameters['Samps/line'][i]),
                'Columns': int(self.headerParameters['Number of lines'][i]),
                'Set Point': self.headerParameters['@2:AFMSetDeflection'][0]
            })
            #print(self.Image[i]['Set Point'])
            file.seek(int(self.headerParameters['Data offset'][i]))
            s = file.read(int(self.headerParameters['Data length'][i+1]))
            s = np.frombuffer(
                s,
                dtype='int32',#'<i{}'.format(2*self.headerParameters['Bytes/pixel'][i]),
                count=int(
                    self.headerParameters['Number of lines'][i])*int(self.headerParameters['Samps/line'][i])).reshape((int(self.headerParameters['Number of lines'][i]), int(self.headerParameters['Samps/line'][i])
                    )
                )
            if self.Image[i]['Channel'] == 'Height':
                s=s*self.headerParameters['@Sens. Zsens'][0]*self.headerParameters['@2:Z scale'][i]/pow(2, 8 * self.headerParameters['Bytes/pixel'][i])
            else:
                s=s*self.headerParameters['@2:Z scale'][i]/pow(2, 8 * self.headerParameters['Bytes/pixel'][i])
            self.Image[i]['Image Data'] = s
            self.Image[i]['Processed Image Data'] = s
        file.close()

    def getChannel(self, channel, direction):
        image = next(
            item for item in self.Image if item["Channel"] == channel and item["Line Direction"] == direction
        )
        return image

    def getChannelIndex(self, channel, direction):
        index = next(
            idx for idx in range(len(self.Image)) if self.Image[idx]["Channel"] == channel and self.Image[idx]["Line Direction"] == direction
        )
        return index

    def flattenImage(self, channel, direction, jdx):
        idx = self.getChannelIndex(channel,direction)
        s = self.Image[idx]['Processed Image Data'].copy()
        if jdx == 6 :
            def func(x, a, b , c, d, e, f, g):
                return a * x**6 + b * x**5 + c * x**4 + d * x**3 + e * x**2 + f * x + g
            initialParameters = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        elif jdx == 3:
            def func(x, a, b , c, d):
                return a * x**3 + b * x**2 + c * x + d
            initialParameters = np.array([1.0, 1.0, 1.0, 1.0])

        for j in range(self.Image[idx]['Rows']):
            xData = np.arange(self.Image[idx]['Columns'])
            yData = self.Image[idx]['Processed Image Data'][j,:]
            fittedParameters, pcov = curve_fit(func, xData, yData, initialParameters)
            modelPredictions = func(xData, *fittedParameters)
            s[j,:] -= modelPredictions

        self.Image[idx]['Processed Image Data'] = s

    def equalizeTopImage(self, channel, direction, percentile):
        idx = self.getChannelIndex(channel,direction)
        l = np.percentile(self.Image[0]['Processed Image Data'], percentile)
        s = self.Image[idx]['Processed Image Data'].copy()
        s[s>l]=l
        self.Image[idx]['Processed Image Data'] = s

    def equalizeImage(self, channel, direction, width):
        idx = self.getChannelIndex(channel,direction)
        l = np.std(self.Image[idx]['Processed Image Data'])
        mp = np.mean(self.Image[idx]['Processed Image Data'])
        
        s = self.Image[idx]['Processed Image Data'].copy()

        s[s>mp+width*l]=mp+width*l
        s[s<mp-width*l]=mp-width*l

        self.Image[idx]['Processed Image Data'] = s

    
