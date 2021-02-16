from readNanoscopeImages import *
from identificationNet import *
import cv2
import math
import numpy as np
import os
import logging
import time


class YoloAnalysis():
    def __init__(self, filename):
        # Initialize attributes
        # AFM image name
        self.filename = filename
        # sets the name that will be used for saving
        # images in different folders
        self.savingName =  self.filename.split('.')[1] + '.jpeg'
        # Saving directory
        self.savingDir = os.path.dirname(self.filename)
        # x coordinate for the bounding boxes
        self.xYolo = None
        # y coordinate for the bounding boxes
        self.yYolo = None
        # width of the bounding boxes
        self.wYolo = None
        # height of the bounding boxes
        self.hYolo = None
        # distance to the image center
        self.distance = None
        # new scan size
        self.newScanSize = None
        # atribute to provide info on analysis
        self.flag = None
        # areas to move
        self.areas = [[0.0,0.0],
                      [10.0,0.0],
                      [10.0,10.0],
                      [0.0,10.0],
                      [-10.0,10.0],
                      [-10.0,0.0],
                      [-10.0,-10.0],
                      [0.0,-10.0],
                      [10.0,-10.0],
                      [20.0,-10.0],
                      [20.0,0.0],
                      [20.0,10.0],
                      [20.0,20.0],
                      [10.0,20.0],
                      [0.0,20.0],
                      [-10.0,20.0],
                      [-20.0,20.0],
                      [-20.0,10.0],
                      [-20.0,0.0],
                      [-20.0,-10.0],
        ]
        # setup Yolov3 network
        # assumes that weight and configuration files are in the
        # same folder
        self.YOLO3Net = cv2.dnn.readNet("yolov3.weights", "yolov3_testing.cfg")
        layer_names = self.YOLO3Net.getLayerNames()
        self.output_layers = [layer_names[i[0] - 1] for i in self.YOLO3Net.getUnconnectedOutLayers()]
        #setup embedding network for molecule identification
        self.siamese_net_embedding = get_embeddingNetwork()
        self.siamese_net_embedding.load_weights('Test_Model_6_triple_loss_weights.h5')
        # Shuts down logs from tensorflow
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
        logging.getLogger('tensorflow').setLevel(logging.FATAL)
        
        
    def readSPMImage(self):
        self.imageObject = NanoscopeImage(self.filename)
        self.imageObject.readHeader()
        self.ScanSizeMicrons = self.imageObject.headerParameters['Scan Size'][0]/1000.
        self.imageObject.readImages()
        self.imageObject.flattenImage('Height','Retrace', 3)
        self.imageObject.equalizeImage('Height','Retrace',2)
        imgx = self.imageObject.Image[0]['Processed Image Data']
        img = (255*(imgx - np.min(imgx))/np.ptp(imgx)).astype(int)
        img = np.flip(img, axis=0)

        fullPath = self.savingDir+'\All_Images\\'+'{}'

        cv2.imwrite(fullPath.format(self.savingName), img)
        self.image = cv2.imread(fullPath.format(self.savingName))

        log_string = 'Image {} read and saved in folder All_Images'.format(self.filename)
        self.toLogFile(log_string)
        
        
    def applyYolo(self):
        # Applies YOLO3 to the processed topography image.
        # What follows below is a standard code for applying YOLO3 with cv2
        # Only variations of the standard code are commented at this stage
        height, width, channels = self.image.shape
        blob = cv2.dnn.blobFromImage(self.image, 0.00392, (416,416), (0, 0, 0), True, crop=False)
        self.YOLO3Net.setInput(blob)
        outs = self.YOLO3Net.forward(self.output_layers)

        class_ids = []
        confidences = []
        boxes = []
        boxes_relative_positions = []
        distances = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.38:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    # We append to the list boxes_relative_positions the (x, y) coordinates
                    # and new scan size in case the current molecule is selected for zooming.
                    # The new scan lateral size will the maximum between 1.5 time the largest side
                    # of the Yolo bounding box or half the value of the lateral size of the
                    # current image
                    newScanSize = max(detection[2]*1.5,detection[3]*1.5)*self.ScanSizeMicrons
                    newScanSize = max(newScanSize, self.ScanSizeMicrons/2.)
                    boxes_relative_positions.append([-(detection[0]-0.5)*self.ScanSizeMicrons+self.imageObject.headerParameters['X Offset'][0]/1000.,
                                                     -(0.5-detection[1])*self.ScanSizeMicrons+self.imageObject.headerParameters['Y Offset'][0]/1000.,
                                                     newScanSize
                                                     ])
                    # We also calculate and append to the list distances the distance from
                    # the image center of the current molecule
                    distance_center = math.sqrt((((x+w/2) - width/2)/width)**2 + ((  (y+h/2) - height/2  )/height)**2)
                    distances.append(distance_center)
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.38, 0.5)

        distances = np.array(distances)[indexes]
        distances = np.squeeze(distances)
        boxes = np.array(boxes)[indexes]
        boxes = np.squeeze(boxes)
        boxes_relative_positions = np.array(boxes_relative_positions)[indexes]
        boxes_relative_positions = np.squeeze(boxes_relative_positions)

        # Orders bounding boxes with respect to their
        # distance to the center of the image
        indexesIncreasingDistance = np.argsort(distances)

        log_string = 'Yolo Applied. {} molecules found.'.format(indexesIncreasingDistance.size)
        self.toLogFile(log_string)

        if indexesIncreasingDistance.size == 1:
            # Case where only one molecule was found
            # It is needed to specifically deal with
            # this case, as the shape of the numpy array matters
            self.distances = distances
            self.boxes = boxes
            self.boxes_relative_positions = np.array([boxes_relative_positions])
        else:
            # This is valid for cases where multiple molecules were found
            # as well as for cases where no molecule was found
            self.distances = distances[indexesIncreasingDistance]
            self.boxes = boxes[indexesIncreasingDistance]
            self.boxes_relative_positions = boxes_relative_positions[indexesIncreasingDistance]
        
    def identifyMoleculeToFollow(self):
        # Identifies a suitable molecule in the image to focus on,
        # checking that it does not match any of those in folder X
        # If one is found, it returns the coordinates, new scan size
        # and a flag with value of 1
        # If it does not find a suitable moelcule, it return a flag
        # with value of 0 (values of coordinates do not matter, maybe
        # an empty list)
        if self.boxes_relative_positions.size == 0:
            # No molecules were found
            self.indexNextMolecule = None
        else:
            # Molecules were found
            #
            # First, we select on which molecule we zoom next
            # For this, we initialize self.indexNextMolecule to None,
            # if after trying to identify the next molecule, the value of
            # this attribute remains None, it means that no suitable molecule
            # found.
            # What needs to be done in this case depends on what the previous
            # was about i.e.:
            # if it was an inital 5um scan (self.flag = 0), we just move to a
            # complete different area
            # if we were zooming on a molecule, we zoom out
            self.indexNextMolecule = None
            if self.flag == 0:
                # Starting from a maximum scan area image, check that we
                # zoom on a molecule not imaged before            
                self.indexNextMolecule = self.checkNegativeSimilarity()
            elif self.flag == 1:
                # Check that we keep zooming on the same molecule
                self.indexNextMolecule = self.checkPositiveSimilarity()
            if self.indexNextMolecule is not None:
                # We state that we keep zooming
                self.flag = 1
                # and select the new coordinates that will be provided to Nanoscope
                self.xYolo, self.yYolo, self.newScanSize = self.boxes_relative_positions[self.indexNextMolecule]

                log_string = 'A molecule to zoom in was identified. Flag set to 1.'
                self.toLogFile(log_string)
              
    def saveYoloImageColor(self):
            
        # Creates copies of the topography image, required to use cv2.addWeighted
        imageYolo = self.image.copy()
        overlay = self.image.copy()
        borderimage = self.image.copy()

        # We deal differently with the cases where the (array) boxes_relative_positions
        # contains just one box and that where it contains different boxes
        if self.distances.size == 1:
            # Only one box found
            xc, yc, wc, hc = self.boxes
            boxes_relative_positions = np.squeeze(self.boxes_relative_positions)
            color = (255,0,0) # blue
            cv2.rectangle(overlay, (xc, yc), (xc + wc, yc + hc), color, -1)
            cv2.rectangle(borderimage, (xc, yc), (xc + wc, yc + hc), (248,248,248), 1)
        elif self.distances.size > 1:
            # Several boxes found. We iterate over all of them, and surround the bounding box for the molecule
            # where the next image will be centered with a  white rectangle and fill it with blue transparency.
            # Other bounding boxes will be delimited by a black rectangle and filled with red transparency.
            for i in range(self.boxes.shape[0]):
                x, y, w, h = self.boxes[i]                
                if i==self.indexNextMolecule:
                    color = (255,0,0) # blue
                    color2 = (248,248,248) # almost white
                else:
                    color = (0,0,255) # red
                    color2 = (0,0,0) # black
                cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
                cv2.rectangle(borderimage, (x, y), (x + w, y + h), color2, 1)

        alpha = 0.3
        cv2.addWeighted(overlay, alpha, borderimage, 1 - alpha, 0, imageYolo)
        fullPath = self.savingDir+'\All_Yolo_Images\\'+'{}'
        cv2.imwrite(fullPath.format(self.savingName), imageYolo)

        # If one or more molecules were found, cut the molecules
        # that will be followed, and save it in the folder Molecule_Being_Followed
        if self.distances.size > 0 and self.indexNextMolecule is not None:
            self.cuttedImage = self.cutMolecule(self.indexNextMolecule)
            fullPath = self.savingDir+'\Molecule_Being_Followed\\'+'{}'
            cv2.imwrite(fullPath.format(self.savingName), self.cuttedImage)

    def checkEnoughZoom(self):
        if self.newScanSize > self.ScanSizeMicrons/2.:
            d = self.savingDir+'\Molecule_Being_Followed'
            dirs = os.listdir(d)
            for file in dirs:
                comparisonFile = os.path.join(d,file)
                break
            image = cv2.imread(comparisonFile)
            fullPath = self.savingDir+'\Initial_Zoomed_Molecules\\'+'{}'
            cv2.imwrite(fullPath.format(self.savingName), image)
            fullPath = self.savingDir+'\Zoomed_Molecules\\'+'{}'
            cv2.imwrite(fullPath.format(self.savingName), self.cuttedImage)
            self.flag = 2

    def checkPositiveSimilarity(self):
        d = self.savingDir+'\Molecule_Being_Followed'
        dirs = os.listdir(d)
        indexNextMolecule = None

        # Chooses for comparison the last imaged saved in the
        # Molecule_Being_Followed folder
        for file in dirs:
            comparisonFile = os.path.join(d,file)

        i1 = prepareSavedImageForSiameseEmbedding(comparisonFile,96)
        pred_i1 = self.siamese_net_embedding.predict(i1)
         
        # Checks if the Siamese Net identifies the molecule within
        # the latest image and, in that, case, returns the corresponding index
        for i in range(self.boxes_relative_positions.shape[0]):
            i2 = prepareNumpyImageForSiameseEmbedding(self.cutMolecule(i),96)
            pred_i2 = self.siamese_net_embedding.predict(i2)
            prediction = compute_dist(pred_i1,pred_i2)
            if prediction < 0.25:
                indexNextMolecule = i
                log_string = 'Same molecule as in previous scan was found'
                self.toLogFile(log_string)            
                return indexNextMolecule
        # Otherwise, it jus returns None as an index value
        return indexNextMolecule

    def checkNegativeSimilarity(self):
        d = self.savingDir+'\Initial_Zoomed_Molecules'
        dirs = os.listdir(d)
        indexNextMolecule = None

        for i in range(self.boxes_relative_positions.shape[0]):
            prediction = 1
            i2 = prepareNumpyImageForSiameseEmbedding(self.cutMolecule(i),96)
            pred_i2 = self.siamese_net_embedding.predict(i2)
            for file in dirs:
                i1 = prepareSavedImageForSiameseEmbedding(os.path.join(d,file),96)
                pred_i1 = self.siamese_net_embedding.predict(i1)
                prediction_new = compute_dist(pred_i1,pred_i2)
                prediction = min(prediction, prediction_new)
            if prediction > 0.25:
                indexNextMolecule = i
                log_string = 'A new molecule to zoom in has been identified'
                self.toLogFile(log_string)
                return indexNextMolecule
        return indexNextMolecule

    def moveToDifferentArea(self, area=0, maxScanSize=5):
        if int(area) < 20:
            self.xYolo, self.yYolo, self.newScanSize = np.array([
                    self.areas[int(area)+1][0],
                    self.areas[int(area)+1][1],
                    maxScanSize
            ])
            log_string = 'Moving to a Different Area ({}).'.format(int(area)+1)
            self.toLogFile(log_string)
        else:
            log_string = 'Sample completely scanned'
            self.toLogFile(log_string)

    def ZoomOut(self, maxScanSize=5):
        log_string = 'The molecule we were zooming in was not found. Zooming out.'
        self.toLogFile(log_string)
        
        # Reads the latest imaged saved in the folder "Molecule_Being_Followed"
        d = self.savingDir+'\Molecule_Being_Followed'
        dirs = os.listdir(d)
        for file in dirs:
            finalMoleculeImage = os.path.join(d,file)
        image = cv2.imread(finalMoleculeImage)

        # Writes this image in the folders "Zoomed_Molecules" and "Initial_Zoomed_Molecules"
        fullPath = self.savingDir+'\Zoomed_Molecules\\'+'{}'
        cv2.imwrite(fullPath.format(file), image)
        fullPath = self.savingDir+'\Initial_Zoomed_Molecules\\'+'{}'
        cv2.imwrite(fullPath.format(file), image)
        
        
        # Clean the folder Molecule_Being_Followed
        for file in dirs:
            os.remove(os.path.join(d,file))

        # Zooms Out
        self.xYolo, self.yYolo, self.newScanSize = np.array([
                self.imageObject.headerParameters['X Offset'][0]/1000.,
                self.imageObject.headerParameters['Y Offset'][0]/1000.,
                maxScanSize
        ])

        # Sets flag to zero i.e., a new scan with max scan size should start
        self.flag = 0
            

    def cutMolecule(self, indexNextMolecule):
        if self.distances.size == 1:
            xc, yc, wc, hc = self.boxes
        else:
            xc, yc, wc, hc = self.boxes[indexNextMolecule]
        if self.distances.size > 0:
            l=max(wc,hc)
            height, width, channels = self.image.shape
            xmin = max(0, math.floor(xc - l*0.2))
            xmax = min(width, math.floor(xc + l* 1.2))
            ymin = max(0, math.floor(yc - l*0.2))
            ymax = min(height, math.floor(yc + l*1.2))
            img_array = self.image[ymin:ymax,xmin:xmax,:]
            
            # Resizing to 96, the input for which the siamese network was trained
            cuttedImage = cv2.resize(img_array, (96,96))
            return cuttedImage

    def toLogFile(self, str):
        fullPath = self.savingDir+'\\'+'{}'
        file1 = open(fullPath.format('logfile.txt'),"a")
        file1.write(str+"\n")
        file1.close()

    


def plasmidAnalysis(image, flag=0, area=0, maxScanSize=5, experimentFinished=0):
    # Finds lists of coordinates (x,y), widths and heights and distances
    # from the image center for the plasmid molecules contained in the
    # image
    yoloOutput = YoloAnalysis(image)
    log_string = '*********************************************************************************************'
    yoloOutput.toLogFile(log_string)
    log_string = 'Analyzing image {}. Received flag = {} and area = {}'.format(image, flag, area)
    yoloOutput.toLogFile(log_string)
    log_string = '*********************************************************************************************'
    yoloOutput.toLogFile(log_string)
    yoloOutput.toLogFile('')
    yoloOutput.readSPMImage()
    yoloOutput.applyYolo()

    yoloOutput.flag = flag
    yoloOutput.identifyMoleculeToFollow()
    yoloOutput.saveYoloImageColor()

    if yoloOutput.flag == 0:
        # A scan of maximum area value has just been performed and no suitable
        # molecule to be followed was found.
        # In this case we move to a different area
        log_string = f'A {maxScanSize}um scan was performed and no suitable molecules were found.'
        yoloOutput.toLogFile(log_string)
        yoloOutput.moveToDifferentArea(area, maxScanSize)
        area+= 1
        

    elif yoloOutput.flag ==1:
        # This can refer to 2 different situations:
        # 1) A suitable molecule to zoom in was found, and new
        #    coordinates identified. In this case indexNextMolecule is not None
        # 2) We were zooming on a molecule, but we do not see it any longer.
        #    In this case we zoom out
        # We check for the second situation. If this is the case, we zoom out.
        # If it is not the case, it means that we found the molecule, but we need
        # to check them if we zoomed enough
        if yoloOutput.indexNextMolecule is None:
            yoloOutput.ZoomOut(maxScanSize)
        else:
            yoloOutput.checkEnoughZoom()

        # If it results that we zoomed enough (flag value changed to 3),
        # we first clean the Molecule_Being_Followed folder, and set
        # the new scan size to the maximum value
        if yoloOutput.flag == 2:
            d = yoloOutput.savingDir+'\Molecule_Being_Followed'
            dirs = os.listdir(d)
            for f in dirs:
                os.remove(os.path.join(d,f))
            yoloOutput.newScanSize = maxScanSize
            # and finally set the flag to 0
            yoloOutput.flag = 0
        

    if math.fabs(yoloOutput.xYolo) > 25. or math.fabs(yoloOutput.yYolo) > 25. or area == 20:
        experimentFinished = 1
    returnArray = np.array([yoloOutput.xYolo,
                            yoloOutput.yYolo,
                            yoloOutput.newScanSize,
                            yoloOutput.flag,
                            area,
                            experimentFinished])
    log_string = 'Leaving plasmidAnalysis function. xYolo: {}, yYolo: {}, newScanSize: {}, flag: {}, area: {}'.format(
        yoloOutput.xYolo,
        yoloOutput.yYolo,
        yoloOutput.newScanSize,
        yoloOutput.flag,
        area)
    yoloOutput.toLogFile(log_string)
    yoloOutput.toLogFile('')

    return returnArray
    


