import sys
import subprocess
sys.path.append("./")

from SOLVEUR_COUPLE import solveur_couple
solveur_couple.solve("SOLVEUR_COUPLE/input_main.txt")
subprocess.run(["python", "SOLVEUR_COUPLE/write_vtu.py"], check=True)