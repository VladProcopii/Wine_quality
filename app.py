from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, silhouette_score
)

# ---------------------------------------------------------
# Configurare
# ---------------------------------------------------------
st.set_page_config(
    page_title="Wine Quality | CRISP-DM",
    page_icon="🍷",
    layout="wide"
)
sns.set_theme(style="whitegrid")

st.title("🍷 Wine Quality Analysis")
st.caption("Proiect de analiză a datelor conform metodologiei CRISP-DM")

DATA_PATH = Path(__file__).resolve().parent / "winequality-white.csv"


@st.cache_data
def load_data(path):
    return pd.read_csv(path, sep=";")


@st.cache_resource
def run_analysis(path):
    """Reproducește metodologia din notebook pentru rezultate comparabile."""
    data = pd.read_csv(path, sep=";")
    X = data.drop(columns=["quality"])
    y_reg = data["quality"]

    def classify_quality(q):
        if q <= 5:
            return "Low"
        elif q == 6:
            return "Medium"
        return "High"

    y_clf = data["quality"].apply(classify_quality)

    # Păstrează combinațiile identice ale predictorilor în același grup.
    groups = pd.factorize(pd.MultiIndex.from_frame(X))[0]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y_reg, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_reg_train, y_reg_test = y_reg.iloc[train_idx], y_reg.iloc[test_idx]
    y_clf_train, y_clf_test = y_clf.iloc[train_idx], y_clf.iloc[test_idx]

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X.columns, index=X_test.index
    )

    # Regresie
    linear = LinearRegression().fit(X_train_scaled, y_reg_train)
    pred_linear = linear.predict(X_test_scaled)
    rf_reg = RandomForestRegressor(
        n_estimators=200, random_state=42, n_jobs=-1
    ).fit(X_train, y_reg_train)
    pred_rf_reg = rf_reg.predict(X_test)

    regression_results = pd.DataFrame([
        {
            "Model": "Linear Regression",
            "MAE": mean_absolute_error(y_reg_test, pred_linear),
            "RMSE": np.sqrt(mean_squared_error(y_reg_test, pred_linear)),
            "R²": r2_score(y_reg_test, pred_linear),
        },
        {
            "Model": "Random Forest",
            "MAE": mean_absolute_error(y_reg_test, pred_rf_reg),
            "RMSE": np.sqrt(mean_squared_error(y_reg_test, pred_rf_reg)),
            "R²": r2_score(y_reg_test, pred_rf_reg),
        },
    ])

    # Clasificare
    logistic = LogisticRegression(max_iter=2000, random_state=42).fit(
        X_train_scaled, y_clf_train
    )
    pred_logistic = logistic.predict(X_test_scaled)
    rf_clf = RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1
    ).fit(X_train, y_clf_train)
    pred_rf_clf = rf_clf.predict(X_test)

    classification_results = pd.DataFrame([
        {
            "Model": "Logistic Regression",
            "Accuracy": accuracy_score(y_clf_test, pred_logistic),
            "Precision": precision_score(y_clf_test, pred_logistic, average="weighted", zero_division=0),
            "Recall": recall_score(y_clf_test, pred_logistic, average="weighted", zero_division=0),
            "F1-score": f1_score(y_clf_test, pred_logistic, average="weighted", zero_division=0),
        },
        {
            "Model": "Random Forest",
            "Accuracy": accuracy_score(y_clf_test, pred_rf_clf),
            "Precision": precision_score(y_clf_test, pred_rf_clf, average="weighted", zero_division=0),
            "Recall": recall_score(y_clf_test, pred_rf_clf, average="weighted", zero_division=0),
            "F1-score": f1_score(y_clf_test, pred_rf_clf, average="weighted", zero_division=0),
        },
    ])
    class_report = pd.DataFrame(
        classification_report(
            y_clf_test, pred_rf_clf, labels=["Low", "Medium", "High"],
            output_dict=True, zero_division=0
        )
    ).T
    conf_matrix = confusion_matrix(
        y_clf_test, pred_rf_clf, labels=["Low", "Medium", "High"]
    )

    # Clustering: compararea k=2..8 pe datele standardizate
    ks = list(range(2, 9))
    inertia_values, silhouette_values = [], []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_train_scaled)
        inertia_values.append(km.inertia_)
        silhouette_values.append(silhouette_score(X_train_scaled, labels))

    kmeans_final = KMeans(n_clusters=2, random_state=42, n_init=10)
    cluster_labels = kmeans_final.fit_predict(X_train_scaled)
    clustered = X_train.copy()
    clustered["cluster"] = cluster_labels
    clustered["quality"] = y_reg_train

    # PCA
    pca = PCA()
    X_train_pca = pca.fit_transform(X_train_scaled)
    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    loadings = pd.DataFrame(
        pca.components_[:2].T, index=X.columns, columns=["PC1", "PC2"]
    )

    # Pattern analysis
    selected_features = ["alcohol", "volatile acidity", "density", "residual sugar"]
    pattern_means = data.groupby("quality")[selected_features].mean()

    # Anomaly detection
    isolation_forest = IsolationForest(contamination=0.02, random_state=42)
    anomaly_predictions = isolation_forest.fit_predict(X_train_scaled)
    anomaly_data = X_train.copy()
    anomaly_data["quality"] = y_reg_train
    anomaly_data["anomaly_status"] = pd.Series(anomaly_predictions, index=X_train.index).map(
        {1: "Normal", -1: "Anomalie"}
    )

    return {
        "data": data, "X": X, "X_train": X_train, "X_test": X_test,
        "y_reg_train": y_reg_train, "y_reg_test": y_reg_test,
        "y_clf_train": y_clf_train, "y_clf_test": y_clf_test,
        "X_train_scaled": X_train_scaled, "X_test_scaled": X_test_scaled,
        "pred_linear": pred_linear, "pred_rf_reg": pred_rf_reg,
        "regression_results": regression_results,
        "classification_results": classification_results,
        "pred_rf_clf": pred_rf_clf, "class_report": class_report,
        "conf_matrix": conf_matrix, "ks": ks, "inertia": inertia_values,
        "silhouette": silhouette_values, "clustered": clustered,
        "explained": explained, "cumulative": cumulative,
        "X_train_pca": X_train_pca, "loadings": loadings,
        "pattern_means": pattern_means, "anomaly_data": anomaly_data,
    }


if not DATA_PATH.exists():
    st.error(f"Fișierul CSV nu a fost găsit: {DATA_PATH}")
    st.info("Pune fișierul winequality-white.csv în același folder cu app.py.")
    st.stop()

df = load_data(str(DATA_PATH))
with st.spinner("Se calculează modelele și indicatorii analizei..."):
    result = run_analysis(str(DATA_PATH))

# Navigare
pages = [
    "Overview",
    "Data Understanding",
    "Regression",
    "Classification",
    "Clustering",
    "PCA",
    "Pattern Analysis",
    "Anomaly Detection",
    "CRISP-DM Conclusions",
]
page = st.sidebar.radio("Navigare", pages)
st.sidebar.divider()
st.sidebar.markdown("**Dataset:** White Wine Quality")
st.sidebar.write(f"Observații: {len(df):,}")
st.sidebar.write(f"Variabile explicative: {df.shape[1] - 1}")
st.sidebar.caption("Modelele sunt calculate local pe calculatorul tău.")

# ---------------------------------------------------------
# Overview
# ---------------------------------------------------------
if page == "Overview":
    st.header("Prezentarea proiectului")
    st.markdown("""
    **Obiectiv:** studierea caracteristicilor fizico-chimice ale vinului alb,
    identificarea tiparelor și construirea unor modele pentru estimarea calității.

    Proiectul urmează metodologia **CRISP-DM**: Business Understanding,
    Data Understanding, Data Preparation, Modeling, Evaluation și Conclusions.
    """)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observații", f"{df.shape[0]:,}")
    c2.metric("Variabile explicative", df.shape[1] - 1)
    c3.metric("Valori lipsă", f"{df.isna().sum().sum():,}")
    c4.metric("Rânduri duplicate", f"{df.duplicated().sum():,}")

    st.subheader("Distribuția scorurilor de calitate")
    counts = df["quality"].value_counts().sort_index()
    st.bar_chart(counts)
    st.caption("Majoritatea observațiilor se concentrează în jurul scorurilor 5 și 6.")

    left, right = st.columns(2)
    with left:
        st.subheader("Etapele CRISP-DM")
        st.markdown("""
        1. **Business Understanding** — formularea obiectivelor.
        2. **Data Understanding** — explorarea și verificarea datelor.
        3. **Data Preparation** — definirea variabilelor, împărțirea și standardizarea.
        4. **Modeling** — regresie, clasificare, clustering, PCA și anomalii.
        5. **Evaluation** — compararea rezultatelor și interpretarea limitărilor.
        6. **Deployment** — prezentarea rezultatelor într-un dashboard local.
        """)
    with right:
        st.subheader("Întrebări de cercetare")
        st.markdown("""
        - Ce caracteristici sunt asociate cu scorul de calitate?
        - Cât de precis poate fi estimată calitatea?
        - Pot fi identificate categorii de calitate?
        - Există grupuri cu profiluri chimice similare?
        - Câte componente PCA sunt necesare pentru a păstra 80% din varianță?
        - Ce observații au profiluri neobișnuite?
        """)

# ---------------------------------------------------------
# Data Understanding
# ---------------------------------------------------------
elif page == "Data Understanding":
    st.header("Data Understanding & Data Preparation")
    st.subheader("Calitatea datelor")
    quality_table = pd.DataFrame({
        "Indicator": ["Număr observații", "Număr coloane", "Valori lipsă", "Rânduri duplicate"],
        "Valoare": [
            len(df), df.shape[1], int(df.isna().sum().sum()), int(df.duplicated().sum())
        ]
    })
    st.dataframe(quality_table, width="stretch", hide_index=True)

    with st.expander("Valori lipsă pe variabile"):
        st.dataframe(df.isna().sum().rename("Valori lipsă").to_frame(), width="stretch")
    with st.expander("Tipuri de date"):
        st.dataframe(df.dtypes.astype(str).rename("Tip de date").to_frame(), width="stretch")
    with st.expander("Distribuția exactă a scorurilor"):
        st.dataframe(
            df["quality"].value_counts().sort_index().rename_axis("Quality").rename("Count").to_frame(),
            width="stretch"
        )

    st.subheader("Distribuțiile caracteristicilor")
    feature = st.selectbox("Selectează o caracteristică", df.columns.drop("quality").tolist())
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.histplot(df[feature], bins=30, kde=True, ax=ax)
    ax.set_title(f"Distribuția variabilei: {feature}")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Matricea de corelație")
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Corelații între variabile")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Alcool în funcție de calitate")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=df, x="quality", y="alcohol", ax=ax)
    ax.set_title("Conținutul de alcool în funcție de scorul de calitate")
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("""
    **Pregătirea datelor:** `quality` este variabila țintă. Pentru clasificare,
    scorurile au fost grupate în Low (≤5), Medium (=6) și High (≥7).
    Setul a fost împărțit în train/test cu `GroupShuffleSplit`, astfel încât
    combinațiile identice ale predictorilor să nu fie împărțite între seturi.
    `StandardScaler` a fost ajustat numai pe setul de antrenare.
    """)

# ---------------------------------------------------------
# Regression
# ---------------------------------------------------------
elif page == "Regression":
    st.header("Regresie: estimarea scorului de calitate")
    st.markdown("Variabila țintă este scorul numeric `quality`. Comparam un model liniar cu Random Forest.")
    metrics = result["regression_results"].copy()
    c1, c2 = st.columns(2)
    best = metrics.loc[metrics["RMSE"].idxmin()]
    c1.metric("Model cu RMSE minim", best["Model"])
    c2.metric("R² al Random Forest", f'{metrics.loc[metrics["Model"] == "Random Forest", "R²"].iloc[0]:.3f}')

    st.subheader("Compararea modelelor")
    st.dataframe(metrics.round(3), width="stretch", hide_index=True)
    st.bar_chart(metrics.set_index("Model")[["MAE", "RMSE"]])

    st.subheader("Valori reale vs. predicții")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(result["y_reg_test"], result["pred_linear"], alpha=0.35, label="Linear Regression")
    ax.scatter(result["y_reg_test"], result["pred_rf_reg"], alpha=0.35, label="Random Forest")
    ax.plot([3, 9], [3, 9], "k--", label="Predicție perfectă")
    ax.set_xlabel("Scor real")
    ax.set_ylabel("Scor prezis")
    ax.legend()
    ax.set_title("Valori reale versus valori prezise")
    st.pyplot(fig)
    plt.close(fig)

    st.info(
        "Interpretare: Random Forest are rezultate mai bune în acest experiment, "
        "dar R² de aproximativ 0,31 arată că o parte importantă a variației calității "
        "rămâne neexplicată."
    )

# ---------------------------------------------------------
# Classification
# ---------------------------------------------------------
elif page == "Classification":
    st.header("Clasificare: Low, Medium și High")
    st.markdown("Low = quality ≤ 5; Medium = quality = 6; High = quality ≥ 7.")
    metrics = result["classification_results"]
    st.subheader("Compararea modelelor")
    st.dataframe(metrics.round(3), width="stretch", hide_index=True)
    st.bar_chart(metrics.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-score"]])

    st.subheader("Matricea de confuzie — Random Forest")
    cm = pd.DataFrame(
        result["conf_matrix"],
        index=["Real Low", "Real Medium", "Real High"],
        columns=["Predicție Low", "Predicție Medium", "Predicție High"]
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Clasa prezisă")
    ax.set_ylabel("Clasa reală")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Raport pe clase — Random Forest")
    report = result["class_report"].round(3)
    report.index.name = "Clasă / metrică"
    st.dataframe(report, width="stretch")

    st.warning(
        "Categoria High este mai dificil de identificat. În experimentul din notebook, "
        "recall-ul pentru High a fost 0,39; de aceea acuratețea globală nu descrie singură "
        "performanța modelului."
    )

# ---------------------------------------------------------
# Clustering
# ---------------------------------------------------------
elif page == "Clustering":
    st.header("Clustering cu K-Means")
    st.markdown("Numărul de clustere a fost comparat pentru k între 2 și 8.")
    cluster_metrics = pd.DataFrame({
        "k": result["ks"],
        "Inertia": result["inertia"],
        "Silhouette Score": result["silhouette"]
    })
    st.dataframe(cluster_metrics.round(4), width="stretch", hide_index=True)

    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(result["ks"], result["inertia"], marker="o")
        ax.set_xlabel("Numărul de clustere (k)")
        ax.set_ylabel("Inertia")
        ax.set_title("Metoda Elbow")
        st.pyplot(fig)
        plt.close(fig)
    with right:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(result["ks"], result["silhouette"], marker="o")
        ax.set_xlabel("Numărul de clustere (k)")
        ax.set_ylabel("Silhouette Score")
        ax.set_title("Compararea scorurilor Silhouette")
        st.pyplot(fig)
        plt.close(fig)

    clustered = result["clustered"]
    st.subheader("Distribuția clusterelor (k=2)")
    counts = clustered["cluster"].value_counts().sort_index().rename("Observații")
    st.bar_chart(counts)
    st.subheader("Profilul mediu al clusterelor")
    means = clustered.groupby("cluster")[result["X_train"].columns].mean().round(3)
    st.dataframe(means, width="stretch")
    st.subheader("Calitatea medie pe cluster")
    st.dataframe(
        clustered.groupby("cluster")["quality"].agg(["count", "mean", "median"]).round(3),
        width="stretch"
    )
    st.info(
        "Silhouette Score pentru k=2 este aproximativ 0,213. Separarea este relativ slabă; "
        "clusterele reflectă diferențe de profil chimic și nu categorii perfecte de calitate."
    )

# ---------------------------------------------------------
# PCA
# ---------------------------------------------------------
elif page == "PCA":
    st.header("Reducerea dimensionalității — PCA")
    explained = result["explained"]
    cumulative = result["cumulative"]
    n80 = int(np.argmax(cumulative >= 0.80) + 1)

    c1, c2, c3 = st.columns(3)
    c1.metric("Predictori inițiali", result["X_train"].shape[1])
    c2.metric("Componente pentru ≥80%", n80)
    c3.metric("Varianță explicată de 6 PC", f"{cumulative[5] * 100:.2f}%")

    variance_df = pd.DataFrame({
        "Componentă": [f"PC{i}" for i in range(1, len(explained) + 1)],
        "Varianță individuală": explained,
        "Varianță cumulativă": cumulative
    })
    st.subheader("Varianța explicată")
    st.dataframe(variance_df.round(4), width="stretch", hide_index=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(1, len(explained) + 1), explained, alpha=0.7, label="Individuală")
    ax.plot(range(1, len(cumulative) + 1), cumulative, marker="o", label="Cumulativă")
    ax.axhline(0.80, linestyle="--", label="Prag 80%")
    ax.set_xlabel("Componentă principală")
    ax.set_ylabel("Proporția varianței")
    ax.set_title("Varianța explicată de PCA")
    ax.set_xticks(range(1, len(explained) + 1))
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Proiecția pe primele două componente")
    pca_df = pd.DataFrame(
        result["X_train_pca"][:, :2],
        columns=["PC1", "PC2"],
        index=result["X_train"].index
    )
    pca_df["quality"] = result["y_reg_train"].values
    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(pca_df["PC1"], pca_df["PC2"], c=pca_df["quality"], cmap="viridis", alpha=0.6, s=18)
    ax.set_xlabel(f'PC1 ({explained[0] * 100:.2f}%)')
    ax.set_ylabel(f'PC2 ({explained[1] * 100:.2f}%)')
    ax.set_title("Proiecția vinurilor pe PC1 și PC2")
    fig.colorbar(scatter, ax=ax, label="Calitate")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Contribuția caracteristicilor la PC1 și PC2")
    loadings = result["loadings"]
    st.dataframe(loadings, width="stretch")
    st.bar_chart(loadings.abs().sort_values("PC1", ascending=False)[["PC1", "PC2"]])

    st.caption(
        "Primele două componente explică aproximativ 43,36% din varianță; "
        "proiecția 2D este utilă vizual, dar nu păstrează întreaga informație."
    )

# ---------------------------------------------------------
# Pattern analysis
# ---------------------------------------------------------
elif page == "Pattern Analysis":
    st.header("Analiza tiparelor și a relațiilor")
    selected_features = ["alcohol", "volatile acidity", "density", "residual sugar"]
    feature = st.selectbox("Selectează caracteristica", selected_features)

    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.boxplot(data=df, x="quality", y=feature, ax=ax)
        ax.set_title(f"{feature} în funcție de calitate")
        st.pyplot(fig)
        plt.close(fig)
    with right:
        means = df.groupby("quality")[feature].mean()
        st.subheader("Media pe scor de calitate")
        st.bar_chart(means)

    st.subheader("Tabelul mediilor pentru caracteristicile selectate")
    st.dataframe(result["pattern_means"].round(3), width="stretch")

    st.markdown("""
    **Interpretare:** alcoolul prezintă o tendință generală de creștere odată cu scorul
    de calitate, iar densitatea tinde să scadă. Aciditatea volatilă are o relație
    neliniară, iar zahărul rezidual nu prezintă o tendință monotonă clară.
    Categoriile rare trebuie interpretate cu prudență. Aceste relații sunt asocieri,
    nu dovezi de cauzalitate.
    """)

# ---------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------
elif page == "Anomaly Detection":
    st.header("Detectarea anomaliilor — Isolation Forest")
    anomalies = result["anomaly_data"]
    counts = anomalies["anomaly_status"].value_counts()
    n_anomalies = int((anomalies["anomaly_status"] == "Anomalie").sum())
    n_normal = int((anomalies["anomaly_status"] == "Normal").sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Observații analizate", f"{len(anomalies):,}")
    c2.metric("Anomalii identificate", f"{n_anomalies}")
    c3.metric("Proporție anomalii", f"{n_anomalies / len(anomalies) * 100:.2f}%")

    st.caption("Modelul a fost configurat cu contamination=0.02; proporția este stabilită de parametrul ales.")
    st.subheader("Distribuția observațiilor")
    st.bar_chart(counts)

    st.subheader("Compararea valorilor medii")
    means = anomalies.groupby("anomaly_status")[result["X_train"].columns].mean().T
    st.dataframe(means.round(3), width="stretch")
    st.subheader("Calitatea medie")
    st.dataframe(
        anomalies.groupby("anomaly_status")["quality"].agg(["count", "mean", "median"]).round(3),
        width="stretch"
    )

    st.subheader("Observații identificate drept anomalii")
    anomaly_rows = anomalies[anomalies["anomaly_status"] == "Anomalie"].sort_values("quality")
    st.dataframe(anomaly_rows, width="stretch")

    st.download_button(
        "Descarcă observațiile anormale",
        data=anomaly_rows.to_csv(index=True).encode("utf-8"),
        file_name="wine_quality_anomalies.csv",
        mime="text/csv"
    )
    st.warning(
        "O anomalie este o observație cu un profil neobișnuit, nu neapărat o eroare "
        "sau un vin de calitate inferioară."
    )

# ---------------------------------------------------------
# Conclusions
# ---------------------------------------------------------
elif page == "CRISP-DM Conclusions":
    st.header("Concluzii generale")
    st.markdown("""
    Proiectul a analizat setul de date Wine Quality pentru vinurile albe, urmând
    principalele etape ale metodologiei CRISP-DM și aplicând șase tipuri de analiză.

    **Regresie.** Random Forest a depășit regresia liniară, obținând MAE aproximativ
    0,576 și R² aproximativ 0,311. Modelul surprinde o parte din variația scorului,
    dar puterea predictivă rămâne limitată.

    **Clasificare.** Random Forest a obținut o acuratețe de aproximativ 58,8%,
    față de 56,2% pentru Logistic Regression. Categoria Low a fost identificată
    mai bine decât categoria High, pentru care recall-ul a fost de aproximativ 0,39.

    **Clustering.** K-Means a identificat două grupuri cu profiluri chimice diferite.
    Clusterul cu alcool mediu mai ridicat și densitate mai redusă a avut o calitate
    medie mai mare. Silhouette Score de aproximativ 0,213 indică o separare modestă.

    **PCA.** Primele șase componente explică aproximativ 81,26% din varianța totală,
    reducând dimensionalitatea de la 11 caracteristici la 6 componente.

    **Analiza tiparelor.** Alcoolul este asociat pozitiv cu scorul de calitate, iar
    densitatea are o asociere negativă generală. Corelațiile nu demonstrează cauzalitate.

    **Detectarea anomaliilor.** Isolation Forest a identificat 78 de observații
    potențial neobișnuite în setul de antrenare. Acestea trebuie investigate, nu
    eliminate automat.

    **Concluzie finală.** Caracteristicile fizico-chimice oferă informații utile despre
    calitatea vinului, dar nu o explică integral. Pentru rezultate mai bune ar putea
    fi testate alte modele, ajustarea hiperparametrilor și validări suplimentare.
    """)

    st.subheader("Rezultatele principale")
    st.dataframe(result["regression_results"].round(3), width="stretch", hide_index=True)
    st.dataframe(result["classification_results"].round(3), width="stretch", hide_index=True)

st.divider()
st.caption("Wine Quality Dataset | CRISP-DM | Aplicație Streamlit locală")
