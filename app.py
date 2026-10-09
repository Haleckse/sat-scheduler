"""
Application Web interactive Streamlit pour la planification des prises de vue satellite SPOT-5.
Utilise PySCIPOpt (solveur SCIP) et Plotly pour la modélisation et la visualisation interactive.
"""

import glob
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from spot_solver import (
    SpotInstance,
    OptimizationResult,
    solve_spot,
    compare_weather_scenarios,
    WEATHER_MODES,
)

# -------------------------------------------------------------
# Configuration de la page Streamlit
# -------------------------------------------------------------
st.set_page_config(
    page_title="Planificateur SPOT-5",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Style CSS personnalisé
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        border-left: 4px solid #1E88E5;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🛰️ Planification d'Acquisition Satellite (SPOT-5)")
st.caption(
    "Optimisation sous contraintes cinématiques, de mémoire et d'incertitudes (météo & fiabilité instruments) via PLNE / SCIP."
)


# -------------------------------------------------------------
# Fonctions de mise en cache
# -------------------------------------------------------------
@st.cache_data
def get_cached_instance(filepath: str) -> SpotInstance:
    return SpotInstance.from_file(filepath)


@st.cache_data
def run_cached_solve(
    filepath: str,
    pm_max: int,
    weather_mode: str,
    vi: float,
    failures: tuple,
) -> OptimizationResult:
    instance = SpotInstance.from_file(filepath)
    return solve_spot(
        instance=instance,
        pm_max_override=pm_max,
        weather_mode=weather_mode,
        vi_override=vi,
        failure_override=list(failures),
    )


@st.cache_data
def run_cached_comparison(
    filepath: str,
    pm_max: int,
    vi: float,
    failures: tuple,
) -> dict:
    instance = SpotInstance.from_file(filepath)
    return compare_weather_scenarios(
        instance=instance,
        pm_max_override=pm_max,
        vi_override=vi,
        failure_override=list(failures),
    )


# -------------------------------------------------------------
# 1. Barre latérale : Détection des instances et configuration
# -------------------------------------------------------------
fichiers_instances = sorted(
    [
        f
        for f in glob.glob("spotProba*.py")
        if not f.endswith("app.py") and not f.endswith("spot_solver.py")
    ]
)

if not fichiers_instances:
    st.error("❌ Aucun fichier d'instance `spotProba*.py` trouvé à la racine.")
    st.stop()

st.sidebar.header("⚙️ Configuration de l'Instance")
fichier_choisi = st.sidebar.selectbox(
    "Fichier d'instance :",
    fichiers_instances,
    index=0,
    help="Sélectionnez un jeu de données de test SPOT.",
)

try:
    instance = get_cached_instance(fichier_choisi)
except Exception as e:
    st.sidebar.error(f"Erreur de chargement : {e}")
    st.stop()

st.sidebar.info(
    f"📌 **{instance.name}**\n\n"
    f"• Images candidates : **{instance.nbImages}**\n"
    f"• Instruments : **{instance.nbInstruments}**\n"
    f"• Capacité mémoire nominale : **{instance.PMmax}**"
)

st.sidebar.subheader("🎛️ Paramètres & Surcharges")

pm_max_val = st.sidebar.number_input(
    "Capacité mémoire max (PMmax)",
    min_value=1,
    max_value=5000,
    value=int(instance.PMmax),
    step=10,
    help="Limite maximale de mémoire disponible à bord pour la session.",
)

vi_val = st.sidebar.number_input(
    "Vitesse rotation miroir (VI en °/s)",
    min_value=0.1,
    max_value=10.0,
    value=float(instance.VI),
    step=0.1,
    help="Vitesse de dépointage du miroir entre deux prises de vue consécutives.",
)

mode_meteo = st.sidebar.selectbox(
    "Prise en compte météo :",
    list(WEATHER_MODES.keys()),
    index=0,
    help="Hypothèse sur la probabilité de couverture nuageuse.",
)

with st.sidebar.expander("🛠️ Taux de panne instruments"):
    st.caption("Ajustez la probabilité de défaillance de chaque instrument :")
    fail_overrides = []
    for ins_idx in range(instance.nbInstruments):
        default_fail = instance.Failure[ins_idx] if ins_idx < len(instance.Failure) else 0.0
        val = st.slider(
            f"Taux panne Instrument {ins_idx+1}",
            min_value=0.0,
            max_value=1.0,
            value=float(default_fail),
            step=0.01,
            key=f"fail_ins_{ins_idx}",
        )
        fail_overrides.append(val)

# -------------------------------------------------------------
# 2. Résolution et onglets
# -------------------------------------------------------------
failures_tuple = tuple(fail_overrides)

with st.spinner("⚡ Résolution en cours via SCIP..."):
    res = run_cached_solve(
        filepath=fichier_choisi,
        pm_max=pm_max_val,
        weather_mode=mode_meteo,
        vi=vi_val,
        failures=failures_tuple,
    )

tab_gantt, tab_scenarios, tab_conflicts, tab_data = st.tabs(
    [
        "📊 Planning & Gantt",
        "⚖️ Comparaison Multi-Scénarios",
        "⚡ Conflits Cinématiques",
        "📋 Données de l'Instance",
    ]
)

# -------------------------------------------------------------
# ONGLET 1 : Planning & Gantt
# -------------------------------------------------------------
with tab_gantt:
    if not res.is_optimal:
        st.warning(f"⚠️ Le solveur a terminé avec le statut : **{res.status}**")
        if res.status in ["infeasible", "infeasible_or_unbounded"]:
            st.error(
                "Le problème est infaisable avec ces contraintes. "
                "Essayez d'augmenter la capacité mémoire $PM_{max}$ ou de vérifier les paramètres."
            )
    else:
        # Métriques clés
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric(
                "Espérance de gain",
                f"{res.objective_value:.2f}",
                help="Valeur optimale de la fonction objectif maximisée par SCIP.",
            )
        with col2:
            mem_pct = (res.memory_used / res.memory_max * 100) if res.memory_max > 0 else 0
            st.metric(
                "Mémoire utilisée",
                f"{res.memory_used} / {res.memory_max}",
                delta=f"{mem_pct:.1f}% occupé",
                delta_color="off",
            )
        with col3:
            sel_pct = (res.selected_count / res.total_images * 100) if res.total_images > 0 else 0
            st.metric(
                "Images retenues",
                f"{res.selected_count} / {res.total_images}",
                delta=f"{sel_pct:.1f}% acceptées",
            )
        with col4:
            st.metric(
                "Répartition types",
                f"{res.mono_count} Mono / {res.stereo_count} Stéréo",
            )
        with col5:
            st.metric(
                "Temps résolution",
                f"{res.solve_duration_sec * 1000:.1f} ms",
            )

        st.markdown("---")

        # Visualisation Gantt
        st.subheader("📅 Diagramme de Gantt des Acquisitions")
        if res.schedule_df.empty:
            st.info("Aucune image n'a été sélectionnée pour cette configuration.")
        else:
            fig_gantt = px.bar(
                res.schedule_df,
                base="Début (s)",
                x="Durée (s)",
                y="Instrument",
                color="Tâche",
                orientation="h",
                hover_data=[
                    "ID",
                    "Type",
                    "Début (s)",
                    "Fin (s)",
                    "Gain facial",
                    "Mémoire (PM)",
                    "Angle (°)",
                    "P_Clear",
                ],
                color_discrete_sequence=px.colors.qualitative.Bold,
            )

            fig_gantt.update_layout(
                xaxis_title="Temps (secondes)",
                yaxis_title="Instrument",
                barmode="stack",
                height=380,
                margin=dict(l=10, r=10, t=30, b=30),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            fig_gantt.update_yaxes(
                categoryorder="array",
                categoryarray=[f"Instrument {i+1}" for i in range(instance.nbInstruments)],
            )
            st.plotly_chart(fig_gantt, use_container_width=True)

            # Tableau détaillé et options d'exportation
            st.subheader("📋 Détail des prises de vue programmées")
            st.dataframe(
                res.schedule_df.drop(columns=["Instrument_ID"], errors="ignore"),
                use_container_width=True,
                hide_index=True,
            )

            # Exportation des résultats
            col_exp1, col_exp2 = st.columns([1, 1])
            with col_exp1:
                csv_data = res.schedule_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Télécharger le planning (CSV)",
                    data=csv_data,
                    file_name=f"planning_{instance.name.replace('.py', '')}_{mode_meteo}.csv",
                    mime="text/csv",
                )
            with col_exp2:
                json_data = res.schedule_df.to_json(orient="records", indent=2)
                st.download_button(
                    label="📥 Télécharger le planning (JSON)",
                    data=json_data,
                    file_name=f"planning_{instance.name.replace('.py', '')}_{mode_meteo}.json",
                    mime="application/json",
                )

# -------------------------------------------------------------
# ONGLET 2 : Comparaison Multi-Scénarios
# -------------------------------------------------------------
with tab_scenarios:
    st.subheader("⚖️ Comparaison des Politiques Météo")
    st.write(
        "Analyse de la sensibilité du plan d'acquisition selon l'hypothèse de couverture nuageuse retenue."
    )

    comp_results = run_cached_comparison(
        filepath=fichier_choisi,
        pm_max=pm_max_val,
        vi=vi_val,
        failures=failures_tuple,
    )

    summary_rows = []
    all_selected_sets = {}

    for mode_name, r in comp_results.items():
        all_selected_sets[mode_name] = set(r.selected_indices)
        summary_rows.append({
            "Scénario": mode_name,
            "Statut": r.status,
            "Espérance de Gain": round(r.objective_value, 2),
            "Images Sélectionnées": f"{r.selected_count} / {r.total_images}",
            "Mono": r.mono_count,
            "Stéréo": r.stereo_count,
            "Mémoire Consommée": f"{r.memory_used} / {r.memory_max}",
            "Temps CPU (ms)": round(r.solve_duration_sec * 1000, 1),
        })

    df_summary = pd.DataFrame(summary_rows)
    st.dataframe(df_summary, use_container_width=True, hide_index=True)

    # Graphique comparatif
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        fig_comp_gain = px.bar(
            df_summary,
            x="Scénario",
            y="Espérance de Gain",
            color="Scénario",
            title="Gain Espéré selon le Scénario",
            text_auto=".2f",
        )
        fig_comp_gain.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_comp_gain, use_container_width=True)

    with col_chart2:
        # Robustesse des sélections
        if all_selected_sets:
            common_images = set.intersection(*all_selected_sets.values())
            union_images = set.union(*all_selected_sets.values())
            weather_dependent = union_images - common_images

            st.write("#### 🛡️ Analyse de Robustesse")
            st.metric("Images retenues dans TOUS les scénarios (Robustes)", len(common_images))
            st.metric("Images sensibles aux variations météo", len(weather_dependent))

            if common_images:
                st.success(f"**Images robustes recommandées :** {sorted(list(common_images))}")
            if weather_dependent:
                st.info(f"**Images conditionnelles :** {sorted(list(weather_dependent))}")

# -------------------------------------------------------------
# ONGLET 3 : Conflits Cinématiques
# -------------------------------------------------------------
with tab_conflicts:
    st.subheader("⚡ Analyse des Incompatibilités et Conflits de Dépointage")
    st.write(
        r"Deux prises de vue $i_1$ et $i_2$ sur le même instrument sont en conflit si le temps de transition "
        r"est insuffisant pour réorienter le miroir : $\Delta t \cdot V_I < DU \cdot V_I + \Delta \alpha$."
    )

    conflicts = instance.compute_conflicts(vi_override=vi_val)
    st.metric("Nombre total de conflits cinématiques détectés", len(conflicts))

    if conflicts:
        df_conflicts = pd.DataFrame(conflicts)
        
        col_c1, col_c2 = st.columns([1, 1])
        with col_c1:
            st.write("#### 🔍 Paires de prises de vue incompatibles")
            st.dataframe(df_conflicts, use_container_width=True, hide_index=True)

        with col_c2:
            st.write("#### 📊 Répartition des conflits par instrument")
            confl_by_ins = df_conflicts["Instrument"].value_counts().reset_index()
            confl_by_ins.columns = ["Instrument", "Nombre de conflits"]
            fig_conf = px.pie(
                confl_by_ins,
                names="Instrument",
                values="Nombre de conflits",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig_conf.update_layout(height=350)
            st.plotly_chart(fig_conf, use_container_width=True)
    else:
        st.success("✅ Aucun conflit cinématique détecté sur cette instance avec cette vitesse miroir.")

# -------------------------------------------------------------
# ONGLET 4 : Données de l'Instance
# -------------------------------------------------------------
with tab_data:
    st.subheader("📋 Liste Exhaustive des Prises de Vue Candidates")
    df_raw = instance.get_raw_dataframe()

    col_filter1, col_filter2 = st.columns(2)
    with col_filter1:
        filtre_type = st.multiselect("Filtrer par type :", ["Mono", "Stéréo"], default=["Mono", "Stéréo"])
    with col_filter2:
        recherche_id = st.text_input("Rechercher un ID d'image :", placeholder="ex: 0, 1...")

    df_filtered = df_raw[df_raw["Type"].isin(filtre_type)]
    if recherche_id:
        try:
            ids_searched = [int(x.strip()) for x in recherche_id.split(",") if x.strip().isdigit()]
            if ids_searched:
                df_filtered = df_filtered[df_filtered["Image ID"].isin(ids_searched)]
        except Exception:
            pass

    st.dataframe(df_filtered, use_container_width=True, hide_index=True)

    # Distributions
    st.write("#### 📈 Distribution des Paramètres de l'Instance")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        fig_hist_pa = px.histogram(
            df_raw,
            x="Gain facial (PA)",
            color="Type",
            barmode="overlay",
            title="Distribution des Gains Faciaux (PA)",
        )
        fig_hist_pa.update_layout(height=300)
        st.plotly_chart(fig_hist_pa, use_container_width=True)
    with col_d2:
        fig_hist_pm = px.histogram(
            df_raw,
            x="Mémoire (PM)",
            color="Type",
            barmode="overlay",
            title="Distribution des Coûts Mémoire (PM)",
        )
        fig_hist_pm.update_layout(height=300)
        st.plotly_chart(fig_hist_pm, use_container_width=True)
