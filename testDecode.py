import sys,os
sys.path.append('/panfs/vol/m/moshahi34/TamuTools_Working/code')
from GetDataFromListV2 import GetX
from DNN_DecodeV2 import Decode
from sklearn.externals import joblib
sBestParamFile = 'Bestparam_13_9_6_2048_2.pkl.gz'
with open(sys.argv[1]) as fList:
    lChunkFiles = [os.path.join(sys.argv[2],os.path.splitext(featFile)[0]+'.feat') for featFile in fList.read().splitlines()]
arFeatData,lStartIndxs = GetX(lChunkFiles[:10],9,78)
BN,y = Decode(2048, 6, 2, arFeatData, sBestParamFile,3,3)
print(BN.shape,BN[0])
