import pandas as pd

index1 = ["simpson","simpson","simpson","south","south","south",]
index2 = ["Homer","Bart","Marge","Cartman","Kenny","Kyle"]
indexTog = list(zip(ilkIndex,icIndex))

mI = pd.MultiIndex.from_tuples(indexTog)
arr = [[40,"A"],[10,"B"],[25,"C"],[82,"D"],[16,"E"],[25,"F"]]

df1 = pd.DataFrame(arr, index=mI, columns = ["yo","occ"])
