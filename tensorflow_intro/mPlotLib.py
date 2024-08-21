import numpy as np
from math import sqrt
import matplotlib.pyplot as mplt
 
age = np.linspace(0,10,30)
weight = age ** 3

# mplt.plot(weight,age, "r")
# mplt.xlabel("we")
# mplt.ylabel("yo")
# mplt.title("we-yo corr")

# arr = np.arange(0,40)
# arr1 = arr ** 0.1

# mplt.subplot(4,2,2)
# mplt.plot(arr,arr1, "g|-")

# mplt.subplot(4,2,1)
# mplt.plot(arr1,arr,"b--")


# mplt.subplot(4,2,4)
# mplt.plot(arr1,arr,"r-.")

# arr = np.arange(0,40)
# arr1 = arr ** 0.1

# myFig = mplt.figure()

# figureAxes = myFig.add_axes([.8,.8,.4,.9])
# figureAxes.plot(arr,arr1,"r")
# figureAxes.set_xlabel("we")
# figureAxes.set_xlabel("yo")
# figureAxes.set_title("we-yo corr")

# myFig2= mplt.figure()

# ax1 = myFig2.add_axes([.1,.1,.9,.9])
# ax2 = myFig2.add_axes([.2,.6,.3,.3])

# ax2.plot(weight,age)
# ax1.plot(age,weight)

# ax1.set_title("mainGraph")
# ax2.set_title("secGraph")

# ax1.set_ylabel("YAS")
# ax1.set_xlabel("BOY")

# (myFig, myAxes)  = mplt.subplots(nrows=1,ncols=2)
# for ax in myAxes:
    # ax.plot(age,weight, "b")
# mplt.tight_layout()

# myFig = mplt.figure()

# myAxes1 = myFig.add_axes([.5,.5,.9,.9])

# myAxes1.plot(age ,(age * 0.1 ) **2  , "g", label="we-yo graph")
# myAxes1.plot(age, (age * 0.1 ) **3, "r", label="the same in other way")
# myAxes1.legend(loc=0)

# myFig.savefig("myAxes1.png", dpi=200)

(myfig, myaxes) = mplt.subplots()
myaxes.plot(age,weight, "b", label="normal")
myaxes.plot(age,weight ** 0.9, "r", label="anormal")
myfig.legend(loc=0)

mplt.show()















