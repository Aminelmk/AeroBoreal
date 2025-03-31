import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

# filename = "temp/output_0_nx_5_ny_100_Cl.csv"
# data = pd.read_csv(filename)
# ny = int(re.search(r"ny_(\d+)", filename).group(1))
# plt.figure()
# plt.plot(data["y"][:ny], data["Cl"][:ny])
# plt.xlabel("y")
# plt.ylabel("Cl")
# plt.title("Cl distribution for elliptical wing")
# plt.grid()

W_VLM = np.array([19526.6286653, 19153.9315174, 19167.7613361, 19167.1984362, 19167.2220284, 19167.2210367, 19167.2210786, 19167.2210768, 19167.2210769, 19167.2210769])
W_struc = np.array([19091.0105155, 19119.8756939, 19118.6994764, 19118.7487754, 19118.7467029, 19118.7467903, 19118.7467867, 19118.7467868, 19118.7467868, 19118.7467868])


it = np.arange(10)
plt.figure()
plt.plot(it[1:], W_VLM[1:], "-o", label="VLM")
plt.plot(it[1:], W_struc[1:], "-o", label="Structure")
plt.legend()
plt.xlabel("Iteration")
plt.ylabel("W")
plt.xticks(it)
plt.title("Travail virtuel")
plt.grid()


database = pd.read_csv("database/x.6/mach_0.40.csv", header=None)
CL = [-0.623700005209, -0.508828740566, -0.386942970378, -0.260309332138, -0.130807767556, 0, 0.130807767556, 0.260309332138, 0.386942970378, 0.508828740566, 0.623700005209, 0.726900665213, 0.812797973104, 0.874573370146, 0.907114795489, 0.907901268275, 0.877076844958, 0.834382111108]
ALPHA = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
plt.figure()
plt.plot(ALPHA, CL, "-o", label="VLM, AR = 1e12")
plt.plot(database[0], database[1], "-", label="database")
plt.xlabel("Alpha")
plt.ylabel("CL")
plt.title("CL vs Alpha")
plt.legend()
plt.grid()
# CL = -0.623700005209 CD = 0.00959179500147
# CL = -0.508828740566 CD = 0.00627738010902
# CL = -0.386942970378 CD = 0.00401151704521
# CL = -0.260309332138 CD = 0.00264519877507
# CL = -0.130807767556 CD = 0.00195420355252
# CL = 0 CD = 0.001201532906
# CL = 0.130807767556 CD = 0.00195420355252
# CL = 0.260309332138 CD = 0.00264519877507
# CL = 0.386942970378 CD = 0.00401151704521
# CL = 0.508828740566 CD = 0.00627738010902
# CL = 0.623700005209 CD = 0.00959179500147
# CL = 0.726900665213 CD = 0.0142966767623
# CL = 0.812797973104 CD = 0.0209579561842
# CL = 0.874573370146 CD = 0.0300972808309
# CL = 0.907114795489 CD = 0.0419545428606
# CL = 0.907901268275 CD = 0.0561365648573
# CL = 0.877076844958 CD = 0.0720058313055
# CL = 0.834382111108 CD = 0.0898668315666