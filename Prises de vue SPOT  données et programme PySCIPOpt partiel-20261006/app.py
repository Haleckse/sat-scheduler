import glob
import importlib.util
from itertools import product
import os
import pandas as pd
import plotly.express as px
from pyscipopt import Model, quicksum
import streamlit as st

st.set_page_config(page_title="Planificateur SPOT", layout="wide")
st.title("🛰️ Planification d'acquisition satellite (SPOT-5)")

# -------------------------------------------------------------
# 1. Détection automatique des instances dans le répertoire
# -------------------------------------------------------------
fichiers_instances = sorted(
    [
        f
        for f in glob.glob("spotProba*.py")
        if not f.endswith("app.py")
    ]
)

if not fichiers_instances:
    st.error("Aucun fichier d'instance spotProba*.py trouvé à la racine.")
    st.stop()

# Barre latérale : Paramètres
st.sidebar.header("Configuration")
fichier_choisi = st.sidebar.selectbox(
    "Choisir l'instance :", fichiers_instances
)

# Chargement dynamique du module sélectionné
spec = importlib.util.spec_from_file_location("instance_mod", fichier_choisi)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Paramètres de simulation éditables
st.sidebar.subheader("Surpasser les paramètres")
pm_max_val = st.sidebar.number_input(
    "Capacité mémoire max (PMmax)",
    min_value=10,
    max_value=2000,
    value=int(mod.PMmax),
    step=10,
)
mode_meteo = st.sidebar.radio(
    "Prise en compte météo :",
    ["Pessimiste (ProbaSup)", "Moyen (ProbaInf + ProbaSup)/2", "Optimiste (ProbaInf)"],
)


# -------------------------------------------------------------
# 2. Résolution du modèle
# -------------------------------------------------------------
def resoudre(data, pm_max_override, critere_meteo):
    mymodel = Model("Planification_SPOT")
    mymodel.hideOutput(True)

    nbImages = data.nbImages
    nbInstruments = data.nbInstruments

    # Variables de décision
    selection = {
        i: mymodel.addVar(vtype="B", name=f"select_{i}")
        for i in range(nbImages)
    }
    assignedTo = {
        i: {
            j: mymodel.addVar(vtype="B", name=f"ass_{i}_{j}")
            for j in range(nbInstruments)
        }
        for i in range(nbImages)
    }

    # Calcul des probabilités de couverture nuageuse
    if critere_meteo == "Pessimiste (ProbaSup)":
        p_clear = [1.0 - data.ProbaSup[i] for i in range(nbImages)]
    elif critere_meteo == "Optimiste (ProbaInf)":
        p_clear = [1.0 - data.ProbaInf[i] for i in range(nbImages)]
    else:
        p_clear = [
            1.0 - (data.ProbaInf[i] + data.ProbaSup[i]) / 2.0
            for i in range(nbImages)
        ]

    # Fonction objectif : espérance de gain
    obj_terms = []
    for i in range(nbImages):
        if data.TY[i] == 1:
            for j in range(nbInstruments):
                coef = data.PA[i] * p_clear[i] * (1.0 - data.Failure[j])
                obj_terms.append(coef * assignedTo[i][j])
        else:  # Stéréo : instruments 0 et 2 obligatoires
            coef = (
                data.PA[i]
                * p_clear[i]
                * (1.0 - data.Failure[0])
                * (1.0 - data.Failure[2])
            )
            obj_terms.append(coef * selection[i])

    mymodel.setObjective(quicksum(obj_terms), sense="maximize")

    # Contrainte A : Cohérence sélection/instruments selon TY
    for i in range(nbImages):
        if data.TY[i] == 1:
            mymodel.addCons(
                quicksum(assignedTo[i][j] for j in range(nbInstruments))
                == selection[i]
            )
        elif data.TY[i] == 2:
            mymodel.addCons(assignedTo[i][0] == selection[i])
            mymodel.addCons(assignedTo[i][1] == 0)
            mymodel.addCons(assignedTo[i][2] == selection[i])

    # Contrainte B : Capacité mémoire
    mymodel.addCons(
        quicksum(data.PM[i] * selection[i] for i in range(nbImages))
        <= pm_max_override
    )

    # Contrainte C : Non-chevauchement et dépointage miroir
    for ima1, ima2 in product(range(nbImages), range(nbImages)):
        if ima1 < ima2:
            for ins in range(nbInstruments):
                # Ignore l'instrument central pour les images stéréo
                if (data.TY[ima1] == 2 and ins == 1) or (
                    data.TY[ima2] == 2 and ins == 1
                ):
                    continue

                dt = abs(data.DD[ima1][ins] - data.DD[ima2][ins])
                da = abs(data.AN[ima1][ins] - data.AN[ima2][ins])
                if dt * data.VI < data.DU * data.VI + da:
                    mymodel.addCons(
                        assignedTo[ima1][ins] + assignedTo[ima2][ins] <= 1
                    )

    mymodel.optimize()
    return mymodel, selection, assignedTo


# -------------------------------------------------------------
# 3. Affichage des résultats
# -------------------------------------------------------------
if st.button("Lancer l'optimisation", type="primary"):
    with st.spinner("Résolution en cours via SCIP..."):
        model, selection, assignedTo = resoudre(mod, pm_max_val, mode_meteo)

    if model.getStatus() == "optimal":
        gain_val = model.getObjVal()
        mem_used = sum(
            mod.PM[i]
            for i in range(mod.nbImages)
            if model.getVal(selection[i]) > 0.5
        )
        nb_sel = sum(
            1
            for i in range(mod.nbImages)
            if model.getVal(selection[i]) > 0.5
        )

        # Cartouches indicateurs
        col1, col2, col3 = st.columns(3)
        col1.metric("Espérance de gain", f"{gain_val:.2f}")
        col2.metric("Mémoire utilisée", f"{mem_used} / {pm_max_val}")
        col3.metric("Images sélectionnées", f"{nb_sel} / {mod.nbImages}")

        # Données de planification
        records = []
        for ima in range(mod.nbImages):
            for ins in range(mod.nbInstruments):
                if model.getVal(assignedTo[ima][ins]) > 0.5:
                    debut = mod.DD[ima][ins]
                    fin = debut + mod.DU
                    records.append(
                        {
                            "Tâche": f"Img {ima} ({'Mono' if mod.TY[ima]==1 else 'Stéréo'})",
                            "Instrument": f"Instrument {ins+1}",
                            "Début (s)": debut,
                            "Fin (s)": fin,
                            "Durée (s)": mod.DU,
                            "Angle (°)": mod.AN[ima][ins],
                            "Gain facial": mod.PA[ima],
                        }
                    )

        df = pd.DataFrame(records)

       # Remplacer cette partie qui plante :
        # fig = px.timeline(...)
        # fig.layout.xaxis.type = "linear"
        # for d in fig.data:
        #     d.x = d.x_end - d.x_start
        #     d.base = ...

        # Par celle-ci :
        st.subheader("Planning d'acquisition (Gantt)")
        fig = px.bar(
            df,
            base="Début (s)",
            x="Durée (s)",
            y="Instrument",
            color="Tâche",
            orientation="h",
            hover_data=["Début (s)", "Fin (s)", "Gain facial", "Angle (°)"],
        )

        fig.update_layout(
            xaxis_title="Temps (secondes)",
            yaxis_title="Instrument",
            barmode="stack",
        )
        fig.update_yaxes(
            categoryorder="array",
            categoryarray=["Instrument 1", "Instrument 2", "Instrument 3"],
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Tableau des prises de vue programmées")
        st.dataframe(df, use_container_width=True)
    else:
        st.error(f"Statut solveur : {model.getStatus()}")
