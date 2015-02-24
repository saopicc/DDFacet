from ClassData import ClassMultiPointingData,ClassSinglePointingData
import ClassParam
from ClassME import MeasurementEquation
from ClassJonesMachine import ClassJonesMachine
from ClassSkyMachine import ClassSkyMachine

X=ClassParam.ClassParam(MDC,GD,NameParsetME=MME.NameParsetME)
X.InitMachine()

for ID in MDC.ListID:
    #DC=ClassSinglePointingData(MDC,PointingID=ID)
    J=ClassJonesMachine(X,ID)
    S=ClassSkyMachine(X,ID)
    MME.setJM(J,PointingID=ID)
    MME.setSM(S,PointingID=ID)

#print MME
