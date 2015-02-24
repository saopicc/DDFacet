import ToolsDir.GetClassSize

try:
    del(DicoSizes)
except:
    pass

<<<<<<< HEAD
#listObjects=%who_ls
=======
#ll=%who_ls

from IPython import get_ipython
ipython = get_ipython()
listObjects=ipython.magic("who_ls")
>>>>>>> f407ac8010d8535da3e38af2106d40357b9e0f1b

DicoSizes={}

for l in listObjects:
    DicoSizes[l]=ToolsDir.GetClassSize.asizeof(l)

print DicoSizes
