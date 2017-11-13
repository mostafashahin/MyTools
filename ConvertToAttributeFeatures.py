from __future__ import print_function
import numpy as np
import sys
from os.path import join, basename
from sklearn.externals import joblib
sys.path.append('/panfs/vol/m/moshahi34/TamuTools/code')
from GetDataFromListV2 import GetX
from DNN_DecodeV2 import Decode
if len(sys.argv) != 6:
    print('Usage: python ConvertToAttributeFeatures.py sFileList NumFrames AttributeParam sFeatType sOutFolder')
    sys.exit(1)
sFileList = sys.argv[1]
iNumFrams = int(sys.argv[2])
sAttributeParam = sys.argv[3]
sFeatType = sys.argv[4]
sOutFolder = sys.argv[5]
with open(sAttributeParam) as fAttribParams:
    lAttribParams = [[int(i) for i in sLine.split()[:-1]]+[sLine.split()[-1]] for sLine in fAttribParams.read().splitlines()]
    print(lAttribParams[0])

def GetPhoneFeat(arFeatData,lStartIndxs,nFrams,lAttribParams,sFeatType):
    iBNFeatSize = lAttribParams[0][4]
    if sFeatType == 'A': #Attribute Features
        print('helllllllo')
        arAttributeFeatures = np.empty((arFeatData.shape[0],len(lAttribParams)),dtype='float')
        for iParam in range(len(lAttribParams)):
            print('********************',iParam)
            iAttributeIndx,iNFrams,iNhl,iNhu,iBNFeatSize,sBestParamFile = lAttribParams[iParam]
            Py, yP = Decode(iNhu, iNhl, 2, arFeatData, sBestParamFile)
            arAttributeFeatures[:,iParam] = Py[:,0]
        arFeat = arAttributeFeatures
    elif sFeatType == 'F':
        arFeat = arFeatData[:,312:390]
    elif sFeatType == 'AF':
        arAttributeFeatures = np.empty((arFeatData.shape[0],len(lAttribParams)),dtype='float')
        for iParam in range(len(lAttribParams)):
            iAttributeIndx,iNFrams,iNhl,iNhu,iBNFeatSize,sBestParamFile = lAttribParams[iParam]
            Py, yP = Decode(iNhu, iNhl, 2, arFeatData, sBestParamFile)
            arAttributeFeatures[:,iParam] = Py[:,0]
        arFeat = np.c_[arAttributeFeatures,arFeatData[:,312:390]]
    else:
        print('hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhheeeeeeeeeeeeeeeeeeeeeeeeeeeeaaaaaaaaaaaaaaaaaaaaaarrrrrrrrrrrrrr')
        arAttributeFeatures = np.empty((arFeatData.shape[0],len(lAttribParams)*iBNFeatSize),dtype='float')
        for iParam in range(len(lAttribParams)):
            iAttributeIndx,iNFrams,iNhl,iNhu,iBNFeatSize,sBestParamFile = lAttribParams[iParam]
            Py, yP = Decode(iNhu, iNhl, 2, arFeatData, sBestParamFile,iBNhl=3,iBNhu=iBNFeatSize)
            arAttributeFeatures[:,iParam*iBNFeatSize:iParam*iBNFeatSize+iBNFeatSize] = Py
        arFeat = arAttributeFeatures
    iNumPad = (nFrams-1)/2
    arPad = np.zeros((iNumPad,arFeat.shape[1]))
    vOffst = np.arange(len(lStartIndxs))*iNumPad
    vIndxs = lStartIndxs + vOffst
    iMask = np.ones((arFeat.shape[0]+iNumPad*len(lStartIndxs)),dtype='bool')
    for indx in vIndxs:
        arFeat = np.insert(arFeat,indx,arPad,axis=0)
        iMask[indx:indx+iNumPad] = False
        temp = np.where(iMask)
    vSelectedIndx = np.where(iMask)[0]
    arFeat = np.r_[arFeat,arPad]
    lCotextData = [arFeat[vSelectedIndx+i] for i in range(-iNumPad,iNumPad+1)]
    arContextFeatData = np.c_[tuple(lCotextData)]
    return arContextFeatData

with open(sFileList) as fList:
    lChunkFiles = fList.read().splitlines()

arFeatData,lStartIndxs = GetX(lChunkFiles,iNumFrams,78)
arAttrFeatures = GetPhoneFeat(arFeatData,lStartIndxs,1,lAttribParams,sFeatType)
for i in xrange(len(lChunkFiles)-1):
    sFile = lChunkFiles[i]
    iStart = lStartIndxs[i]
    iEnd = lStartIndxs[i+1]
    arData = arAttrFeatures[iStart:iEnd]
    joblib.dump(arData,join(sOutFolder,basename(sFile)))
    print(sFile,arData.shape)
sFile = lChunkFiles[-1]
iStart = lStartIndxs[-1]
arData = arAttrFeatures[iStart:]
joblib.dump(arData,join(sOutFolder,basename(sFile)))
print(sFile,arData.shape)
print(arFeatData.shape,lStartIndxs,arAttrFeatures.shape)
