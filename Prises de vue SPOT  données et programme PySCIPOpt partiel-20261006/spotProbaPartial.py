# Resolution d'un problème de planification de prise de vue  sans incertitude
# Helene Fargier, oct 2025
#
#  Version bootstrap, avec exemple de declaration de modele, ajout de variables de decision,
# d'une fonction objectif  ne prenant pas en compte les incertitudes
# une seule contrainte implementée



# on charge le solveur lineaire
from pyscipopt import Model, quicksum
from itertools import product

# on charge les données
from spotProba5 import (
    AN,
    DD,
    DU,
    PM,
    TY,
    VI,
    Failure,
    PMmax,
    PA,
    ProbaInf,
    ProbaSup,
    nbImages,
    nbInstruments,
)


# creation du modele lineaire
#############################

#model
mymodel = Model()

#======================
# Variables de décision
#======================


# pour chaque image i ,  le solveur doit affecter la variable booleen selection[i) à 1 ssi  l'image i est selectionnée
selection = {}
for i in range(nbImages):
    selection[i] = mymodel.addVar(vtype='B', name='select' + str(i))

#### pour chaque image i ,  le solveur doit affecter la variable booleen selection[i) à 1 ssi  l'image i est selectionnée
assignedTo = {}
for i in range(nbImages):
    ass_i = {}
    for j in range(nbInstruments):
        ass_i[j] = mymodel.addVar(vtype='B', name='assignto' + str(i) + '_' + str(j))
    assignedTo[i] = ass_i

# la fonction objectif
######################

# en l'absence d'incertitude, on maximise la somme des payoff

# mymodel.setObjective(quicksum(PA[i] * selection[i] for i in range(nbImages)), sense='maximize')
proba_no_cloud = [1.0 - ProbaSup[i] for i in range(nbImages)]

obj_terms = []
for i in range(nbImages):
    p_clear = proba_no_cloud[i]
    if TY[i] == 1:
        # Mono : dépend de l'instrument choisi
        for j in range(nbInstruments):
            coef = PA[i] * p_clear * (1.0 - Failure[j])
            obj_terms.append(coef * assignedTo[i][j])
    else:
        # Stéréo : requiert instrument 0 ET instrument 2 sans panne
        coef = PA[i] * p_clear * (1.0 - Failure[0]) * (1.0 - Failure[2])
        obj_terms.append(coef * selection[i])

mymodel.setObjective(quicksum(obj_terms), sense="maximize")


#======================
# Contraintes
#======================

# la contrainte de non chevauchement
# considérons un instrument
# si, sur cet insrument, le temps de transition entre 2 images ima1 et ima2
# ne tient pas entre la fin de ima1 et le debut de ima2
# alors une seule de ces deux images au plus peut etre assignée à l'instrument

# for ima1,ima2 in product(range(nbImages), range(nbImages)):
#     if ima1 < ima2:
#         for ins in range(nbInstruments):
#             if  abs(DD[ima1][ins] - DD[ima2][ins]) * VI < DU * VI + abs(AN[ima1][ins] - AN[ima2][ins]):
#                 mymodel.addCons(assignedTo[ima1][ins] + assignedTo[ima2][ins] <= 1)


# A. Liaison entre sélection et instruments (TY)
for i in range(nbImages):
    if TY[i] == 1:
        # Mono : exactement un instrument si sélectionnée, 0 sinon
        mymodel.addCons(
            quicksum(assignedTo[i][j] for j in range(nbInstruments))
            == selection[i]
        )
    elif TY[i] == 2:
        # Stéréo : requiert instruments 0 et 2, instrument 1 interdit
        mymodel.addCons(assignedTo[i][0] == selection[i])
        mymodel.addCons(assignedTo[i][1] == 0)
        mymodel.addCons(assignedTo[i][2] == selection[i])

# B. Capacité mémoire
mymodel.addCons(
    quicksum(PM[i] * selection[i] for i in range(nbImages)) <= PMmax
)

# C. Non-chevauchement temporel et cinématique du miroir
for ima1, ima2 in product(range(nbImages), range(nbImages)):
    if ima1 < ima2:
        for ins in range(nbInstruments):
            # Pour les images stéréo, l'instrument 1 n'est pas utilisé (date fictive = 0)
            if (TY[ima1] == 2 and ins == 1) or (TY[ima2] == 2 and ins == 1):
                continue

            delta_t = abs(DD[ima1][ins] - DD[ima2][ins])
            delta_angle = abs(AN[ima1][ins] - AN[ima2][ins])

            if delta_t * VI < DU * VI + delta_angle:
                mymodel.addCons(
                    assignedTo[ima1][ins] + assignedTo[ima2][ins] <= 1
                )

# resolution et affichage des resulats
#########################################

#visualiser le problem lineaire cree
mymodel.writeProblem("pb.cip")

# lancer l'optimisation
print("Resolution")
mymodel.hideOutput(False)
mymodel.optimize()

#afficiher  les resultats mode "scip"
print('statut ' + mymodel.getStatus())
print("solution", end='\t')
print(mymodel.getBestSol())

# afficher les resultats prorement
if mymodel.getStatus() == 'optimal':
    print("\n\nProblème resolu, valeur de l'objectif " + str(mymodel.getObjVal()))
    sol=mymodel.getBestSol()
    for ima in range(nbImages):
        for ins in range(nbInstruments):
            if (mymodel.getVal(assignedTo[ima][ins]) > 0):
                print("Image" + str(ima) + " selectionnée et  assignée à  " + str(ins) + "  (debut à " + str( DD[ima][ins]) + ")")

