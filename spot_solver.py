"""
Module de modélisation et résolution du problème de planification SPOT-5 via PySCIPOpt.
"""

from dataclasses import dataclass, field
import importlib.util
from itertools import product
import time
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from pyscipopt import Model, quicksum


WEATHER_MODES = {
    "Pessimiste (ProbaSup)": "pessimistic",
    "Moyen (ProbaInf + ProbaSup)/2": "average",
    "Optimiste (ProbaInf)": "optimistic",
    "Déterministe (Sans nuages)": "clear",
}


@dataclass
class SpotInstance:
    """Représentation d'une instance de planification satellite SPOT."""
    name: str
    nbImages: int
    TY: List[int]
    PM: List[int]
    PA: List[float]
    nbInstruments: int
    DD: List[List[int]]
    AN: List[List[float]]
    DU: int
    VI: float
    PMmax: int
    ProbaInf: List[float]
    ProbaSup: List[float]
    Failure: List[float]

    @classmethod
    def from_file(cls, filepath: str) -> "SpotInstance":
        """Charge une instance depuis un fichier Python spotProba*.py."""
        name = filepath.split("/")[-1]
        spec = importlib.util.spec_from_file_location("instance_mod", filepath)
        if spec is None or spec.loader is None:
            raise ValueError(f"Impossible de charger le fichier d'instance : {filepath}")
        
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Extraction sécurisée avec valeurs par défaut
        nbImages = getattr(mod, "nbImages", len(getattr(mod, "TY", [])))
        TY = list(getattr(mod, "TY", [1] * nbImages))
        PM = list(getattr(mod, "PM", [10] * nbImages))
        PA = [float(x) for x in getattr(mod, "PA", [10.0] * nbImages)]
        nbInstruments = getattr(mod, "nbInstruments", 3)
        DD = [list(row) for row in getattr(mod, "DD", [])]
        AN = [list(row) for row in getattr(mod, "AN", [])]
        DU = int(getattr(mod, "DU", 20))
        VI = float(getattr(mod, "VI", 1.0))
        PMmax = int(getattr(mod, "PMmax", 100))
        ProbaInf = [float(x) for x in getattr(mod, "ProbaInf", [0.0] * nbImages)]
        ProbaSup = [float(x) for x in getattr(mod, "ProbaSup", [0.0] * nbImages)]
        Failure = [float(x) for x in getattr(mod, "Failure", [0.0] * nbInstruments)]

        return cls(
            name=name,
            nbImages=nbImages,
            TY=TY,
            PM=PM,
            PA=PA,
            nbInstruments=nbInstruments,
            DD=DD,
            AN=AN,
            DU=DU,
            VI=VI,
            PMmax=PMmax,
            ProbaInf=ProbaInf,
            ProbaSup=ProbaSup,
            Failure=Failure,
        )

    def get_raw_dataframe(self) -> pd.DataFrame:
        """Retourne un DataFrame descriptif des images candidates."""
        rows = []
        for i in range(self.nbImages):
            row = {
                "Image ID": i,
                "Type": "Mono" if self.TY[i] == 1 else "Stéréo",
                "Gain facial (PA)": self.PA[i],
                "Mémoire (PM)": self.PM[i],
                "Proba Nuages Min": self.ProbaInf[i] if i < len(self.ProbaInf) else 0.0,
                "Proba Nuages Max": self.ProbaSup[i] if i < len(self.ProbaSup) else 0.0,
            }
            for ins in range(self.nbInstruments):
                dd_val = self.DD[i][ins] if i < len(self.DD) and ins < len(self.DD[i]) else 0
                an_val = self.AN[i][ins] if i < len(self.AN) and ins < len(self.AN[i]) else 0
                row[f"Début Inst {ins+1} (s)"] = dd_val
                row[f"Angle Inst {ins+1} (°)"] = an_val
            rows.append(row)
        return pd.DataFrame(rows)

    def compute_conflicts(self, vi_override: Optional[float] = None) -> List[Dict[str, Any]]:
        """Calcule l'ensemble des paires de prises de vue incompatibles cinématiquement."""
        vi = vi_override if vi_override is not None else self.VI
        conflicts = []
        for ima1, ima2 in product(range(self.nbImages), range(self.nbImages)):
            if ima1 < ima2:
                for ins in range(self.nbInstruments):
                    if (self.TY[ima1] == 2 and ins == 1) or (self.TY[ima2] == 2 and ins == 1):
                        continue

                    dt = abs(self.DD[ima1][ins] - self.DD[ima2][ins])
                    da = abs(self.AN[ima1][ins] - self.AN[ima2][ins])
                    min_time_required = self.DU + (da / vi if vi > 0 else float("inf"))

                    if dt * vi < self.DU * vi + da:
                        conflicts.append({
                            "Image 1": ima1,
                            "Image 2": ima2,
                            "Instrument": f"Instrument {ins+1}",
                            "Instrument_ID": ins + 1,
                            "Delta T (s)": dt,
                            "Delta Angle (°)": da,
                            "Temps min requis (s)": round(min_time_required, 2),
                        })
        return conflicts


@dataclass
class OptimizationResult:
    """Résultats d'une optimisation de planification."""
    instance_name: str
    weather_mode: str
    status: str
    is_optimal: bool
    objective_value: float
    memory_used: int
    memory_max: int
    selected_count: int
    total_images: int
    mono_count: int
    stereo_count: int
    schedule_df: pd.DataFrame
    solve_duration_sec: float
    selected_indices: List[int] = field(default_factory=list)
    raw_assignments: Dict[int, List[int]] = field(default_factory=dict)


def compute_p_clear(instance: SpotInstance, weather_mode: str) -> List[float]:
    """Calcule la probabilité de temps clair p_clear pour chaque image selon le critère choisi."""
    mode_key = WEATHER_MODES.get(weather_mode, weather_mode)
    p_clear = []
    for i in range(instance.nbImages):
        p_inf = instance.ProbaInf[i] if i < len(instance.ProbaInf) else 0.0
        p_sup = instance.ProbaSup[i] if i < len(instance.ProbaSup) else 0.0

        if mode_key == "pessimistic" or "Pessimiste" in weather_mode:
            p_clear.append(max(0.0, 1.0 - p_sup))
        elif mode_key == "optimistic" or "Optimiste" in weather_mode:
            p_clear.append(max(0.0, 1.0 - p_inf))
        elif mode_key == "clear" or "Déterministe" in weather_mode:
            p_clear.append(1.0)
        else:  # average
            p_clear.append(max(0.0, 1.0 - (p_inf + p_sup) / 2.0))
    return p_clear


def solve_spot(
    instance: SpotInstance,
    pm_max_override: Optional[int] = None,
    weather_mode: str = "Pessimiste (ProbaSup)",
    vi_override: Optional[float] = None,
    failure_override: Optional[List[float]] = None,
) -> OptimizationResult:
    """Construit et résout le problème PLNE SPOT-5 avec PySCIPOpt."""
    start_time = time.time()

    pm_max = pm_max_override if pm_max_override is not None else instance.PMmax
    vi = vi_override if vi_override is not None else instance.VI
    failure = failure_override if failure_override is not None else instance.Failure

    model = Model(f"Planification_SPOT_{instance.name}")
    model.hideOutput(True)

    nb_images = instance.nbImages
    nb_instruments = instance.nbInstruments

    # Variables de décision
    selection = {
        i: model.addVar(vtype="B", name=f"select_{i}")
        for i in range(nb_images)
    }
    assigned_to = {
        i: {
            j: model.addVar(vtype="B", name=f"ass_{i}_{j}")
            for j in range(nb_instruments)
        }
        for i in range(nb_images)
    }

    # Calcul des probabilités de ciel clair
    p_clear = compute_p_clear(instance, weather_mode)

    # Fonction objectif : Espérance de gain
    obj_terms = []
    for i in range(nb_images):
        if instance.TY[i] == 1:
            # Mono : dépend de l'instrument affecté
            for j in range(nb_instruments):
                fail_rate = failure[j] if j < len(failure) else 0.0
                coef = instance.PA[i] * p_clear[i] * (1.0 - fail_rate)
                obj_terms.append(coef * assigned_to[i][j])
        else:
            # Stéréo : instruments 0 et 2 obligatoires
            f0 = failure[0] if len(failure) > 0 else 0.0
            f2 = failure[2] if len(failure) > 2 else 0.0
            coef = instance.PA[i] * p_clear[i] * (1.0 - f0) * (1.0 - f2)
            obj_terms.append(coef * selection[i])

    model.setObjective(quicksum(obj_terms), sense="maximize")

    # Contrainte A : Cohérence sélection / instruments
    for i in range(nb_images):
        if instance.TY[i] == 1:
            model.addCons(
                quicksum(assigned_to[i][j] for j in range(nb_instruments)) == selection[i]
            )
        elif instance.TY[i] == 2:
            model.addCons(assigned_to[i][0] == selection[i])
            model.addCons(assigned_to[i][1] == 0)
            if nb_instruments > 2:
                model.addCons(assigned_to[i][2] == selection[i])

    # Contrainte B : Capacité mémoire
    model.addCons(
        quicksum(instance.PM[i] * selection[i] for i in range(nb_images)) <= pm_max
    )

    # Contrainte C : Non-chevauchement temporel et dépointage miroir
    for ima1, ima2 in product(range(nb_images), range(nb_images)):
        if ima1 < ima2:
            for ins in range(nb_instruments):
                if (instance.TY[ima1] == 2 and ins == 1) or (instance.TY[ima2] == 2 and ins == 1):
                    continue

                dt = abs(instance.DD[ima1][ins] - instance.DD[ima2][ins])
                da = abs(instance.AN[ima1][ins] - instance.AN[ima2][ins])
                if dt * vi < instance.DU * vi + da:
                    model.addCons(assigned_to[ima1][ins] + assigned_to[ima2][ins] <= 1)

    # Résolution
    model.optimize()
    solve_duration = time.time() - start_time
    status = model.getStatus()
    is_optimal = status == "optimal"

    records = []
    selected_indices = []
    raw_assignments: Dict[int, List[int]] = {}
    obj_val = 0.0
    mem_used = 0
    mono_count = 0
    stereo_count = 0

    if is_optimal:
        obj_val = float(model.getObjVal())
        for i in range(nb_images):
            if model.getVal(selection[i]) > 0.5:
                selected_indices.append(i)
                mem_used += instance.PM[i]
                if instance.TY[i] == 1:
                    mono_count += 1
                else:
                    stereo_count += 1

                raw_assignments[i] = []
                for ins in range(nb_instruments):
                    if model.getVal(assigned_to[i][ins]) > 0.5:
                        raw_assignments[i].append(ins)
                        debut = instance.DD[i][ins]
                        fin = debut + instance.DU
                        records.append({
                            "ID": i,
                            "Tâche": f"Img {i} ({'Mono' if instance.TY[i] == 1 else 'Stéréo'})",
                            "Type": "Mono" if instance.TY[i] == 1 else "Stéréo",
                            "Instrument": f"Instrument {ins+1}",
                            "Instrument_ID": ins + 1,
                            "Début (s)": debut,
                            "Fin (s)": fin,
                            "Durée (s)": instance.DU,
                            "Angle (°)": instance.AN[i][ins],
                            "Gain facial": instance.PA[i],
                            "Mémoire (PM)": instance.PM[i],
                            "P_Clear": round(p_clear[i], 3),
                        })

    schedule_df = pd.DataFrame(records)
    if not schedule_df.empty:
        schedule_df.sort_values(by=["Début (s)", "Instrument_ID"], inplace=True)

    return OptimizationResult(
        instance_name=instance.name,
        weather_mode=weather_mode,
        status=status,
        is_optimal=is_optimal,
        objective_value=obj_val,
        memory_used=mem_used,
        memory_max=pm_max,
        selected_count=len(selected_indices),
        total_images=nb_images,
        mono_count=mono_count,
        stereo_count=stereo_count,
        schedule_df=schedule_df,
        solve_duration_sec=solve_duration,
        selected_indices=selected_indices,
        raw_assignments=raw_assignments,
    )


def compare_weather_scenarios(
    instance: SpotInstance,
    pm_max_override: Optional[int] = None,
    vi_override: Optional[float] = None,
    failure_override: Optional[List[float]] = None,
) -> Dict[str, OptimizationResult]:
    """Exécute l'optimisation sur l'ensemble des scénarios météo pour comparaison."""
    scenarios = [
        "Pessimiste (ProbaSup)",
        "Moyen (ProbaInf + ProbaSup)/2",
        "Optimiste (ProbaInf)",
        "Déterministe (Sans nuages)",
    ]
    results = {}
    for sc in scenarios:
        results[sc] = solve_spot(
            instance=instance,
            pm_max_override=pm_max_override,
            weather_mode=sc,
            vi_override=vi_override,
            failure_override=failure_override,
        )
    return results
