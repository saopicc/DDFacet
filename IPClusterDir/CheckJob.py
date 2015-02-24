import ModColor
import MyLogger
import MyLogger
log=MyLogger.getLogger("CheckJob")
from progressbar import ProgressBar
import time

def Check(r,Name,V,Progress=False,TitlePBAR="Reading MS"):
    
    if Progress:

        nMS=len(V.targets)
        pBAR = ProgressBar('white', block='=', empty=' ',Title=TitlePBAR)

        while True:
            isDone=r.progress
            comment='src %i/%i' % (isDone,nMS)
            pBAR.render(int(100* float(isDone) / nMS), comment)
            time.sleep(0.1)
            if isDone==nMS:
                break

    r.wait()
    if r.successful():
        print>>log, "Job "+ModColor.Str(Name,col="green")+" : "+ ModColor.Str("sucessfull",col="blue")
    else:
        print>>log, "Job "+ModColor.Str(Name,col="green")+" : "+ ModColor.Str("failure",col="red")
        print>>log, r.get()

def LaunchAndCheck(V,StrExec,irc=None,Progress=False,TitlePBAR="Reading MS"):
    
    r=V.execute(StrExec)
    Check(r,StrExec,V,Progress=Progress,TitlePBAR=TitlePBAR)


######################"


def SendAndCheck(V,Name,data,Progress=False,TitlePBAR=None):

    Dico={Name:data}
    r=V.push(Dico)

    if Progress:
        if TitlePBAR==None:
            TitlePBAR="Sending %s"%Name
        nMS=len(V.targets)
        pBAR = ProgressBar('white', block='=', empty=' ',Title=TitlePBAR)

        while True:
            isDone=r.progress
            comment='src %i/%i' % (isDone,nMS)
            pBAR.render(int(100* float(isDone) / nMS), comment)

            if isDone==nMS:
                break




        
