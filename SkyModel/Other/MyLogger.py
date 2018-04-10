import logging
import sys

class LoggerWriter:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message != '\n':
            self.logger.log(self.level, message)

import ModColor


class MyLogger():
    def __init__(self):
#fmt="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"
        fmt=" - %(asctime)s - %(name)-25.25s |   %(message)s"
        datefmt='%H:%M:%S'#'%H:%M:%S.%f'
        logging.basicConfig(level=logging.DEBUG,format=fmt,datefmt=datefmt)
        self.Dico={}
        self.Silent=False


    def getLogger(self,name,disable=False):

        if not(name in self.Dico.keys()):
            logger = logging.getLogger(name)
            fp = LoggerWriter(logger, logging.INFO)
            self.Dico[name]=fp
            
        #self.Dico[name].logger.log(logging.DEBUG, "Get Logger for: %s"%name)
        log=self.Dico[name]

            


        return log




    #logger2 = logging.getLogger("demo.X")
    #debug_fp = LoggerWriter(logger2, logging.DEBUG)
    #print>>fp, ModColor.Str("An INFO message")
    #print >> debug_fp, "A DEBUG message"
    #print >> debug_fp, 1

M=MyLogger()

getLogger=M.getLogger

itsLog=getLogger("MyLogger")
import ModColor
def setSilent(Lname):
    print>>itsLog, ModColor.Str("Set silent: %s"%Lname,col="red")
    if type(Lname)==str:
        log=getLogger(Lname)
        log.logger.setLevel(logging.CRITICAL)
    elif type(Lname)==list:
        for name in Lname:
            log=getLogger(name)
            log.logger.setLevel(logging.CRITICAL)


def setLoud(Lname):
    print>>itsLog, ModColor.Str("Set loud: %s"%Lname,col="green")
    if type(Lname)==str:
        log=getLogger(Lname)
        log.logger.setLevel(logging.DEBUG)
    elif type(Lname)==list:
        for name in Lname:
            log=getLogger(name)
            log.logger.setLevel(logging.DEBUG)


if __name__=="__main__":
    log=getLogger("a.x")
    print>>log, "a.x"
