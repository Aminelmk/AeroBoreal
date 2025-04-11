import solveur_couple
import vtk
input_file = "input_main.txt" 

print(f"Lancement du solver avec : {input_file}")
solveur_couple.solve(input_file)
print("Exécution terminée.")
