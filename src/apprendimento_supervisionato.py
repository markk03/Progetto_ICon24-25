import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
import warnings

warnings.filterwarnings("ignore")


# Funzione che carica il dataset arricchito dal ragionatore semantico e prepara le feature.
def load_datasets(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['smogon_tier'])

    y = df['smogon_tier']

    # Dataset 1: Baseline (solo statistiche originali)
    features_base = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']
    X_base = df[features_base]

    # Dataset 2: OntoBK (statistiche + conoscenza dedotta)
    df_encoded = pd.get_dummies(df, columns=['inferred_role'])
    role_columns = [col for col in df_encoded.columns if 'inferred_role_' in col]
    features_enriched = features_base + ['real_weaknesses'] + role_columns
    X_enriched = df_encoded[features_enriched]

    return X_base, X_enriched, y, df_encoded, features_enriched


# Funzione che addestra i modelli usando la Nested Cross-Validation.
def run_ml_evaluation(X, y, phase_title, results_dict, phase_key):
    print(f"\n{phase_title}")

    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='macro', zero_division=0),
        'recall': make_scorer(recall_score, average='macro', zero_division=0),
        'f1': make_scorer(f1_score, average='macro', zero_division=0)
    }

    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models_and_grids = {
        "Random Forest": (RandomForestClassifier(random_state=42), {'classifier__n_estimators': [50, 100]}),
        "SVM (Kernel)": (SVC(random_state=42), {'classifier__C': [0.1, 1, 10], 'classifier__kernel': ['rbf']}),
        "MLP (Rete Neurale)": (MLPClassifier(max_iter=500, random_state=42),
                               {'classifier__hidden_layer_sizes': [(50,), (100,)]})
    }

    for name, (model, param_grid) in models_and_grids.items():
        print(f" -> Valutazione di {name}...")
        pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', model)])

        clf = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=inner_cv, scoring='accuracy', n_jobs=-1)
        results = cross_validate(clf, X, y, cv=outer_cv, scoring=scoring, n_jobs=-1)

        # Estrazione degli array dei punteggi per calcolare le statistiche
        acc_scores = results['test_accuracy'] * 100
        prec_scores = results['test_precision'] * 100
        rec_scores = results['test_recall'] * 100
        f1_scores = results['test_f1'] * 100

        # Stampa dettagliata con Media, Deviazione Standard (std) e Varianza (var)
        print(
            f"    Accuracy:  Media = {acc_scores.mean():.2f}% | Std = {acc_scores.std():.2f}% | Var = {acc_scores.var():.4f}")
        print(
            f"    Precision: Media = {prec_scores.mean():.2f}% | Std = {prec_scores.std():.2f}% | Var = {prec_scores.var():.4f}")
        print(
            f"    Recall:    Media = {rec_scores.mean():.2f}% | Std = {rec_scores.std():.2f}% | Var = {rec_scores.var():.4f}")
        print(
            f"    F1-Score:  Media = {f1_scores.mean():.2f}% | Std = {f1_scores.std():.2f}% | Var = {f1_scores.var():.4f}\n")

        results_dict[phase_key][name] = f1_scores.mean()


# Funzione che genera i grafici e li salva nella cartella dedicata.
def generate_plots(results_dict, X_enriched, y, features_enriched, graphics_dir):
    print("\nGenerazione grafici in corso...")

    modelli_nomi = list(results_dict['Baseline'].keys())
    f1_base = [results_dict['Baseline'][m] for m in modelli_nomi]
    f1_onto = [results_dict['OntoBK'][m] for m in modelli_nomi]

    x = np.arange(len(modelli_nomi))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, f1_base, width, label='Baseline', color='indianred')
    ax.bar(x + width / 2, f1_onto, width, label='OntoBK', color='mediumseagreen')

    ax.set_ylabel('F1-Score Medio (%)')
    ax.set_title('Impatto dell\'Ontologia sulle Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(modelli_nomi)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 100)
    plt.savefig(os.path.join(graphics_dir, 'confronto_performance.png'))

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_enriched, y)

    feat_df = pd.DataFrame({'Feature': features_enriched, 'Importanza': rf.feature_importances_})
    feat_df = feat_df.sort_values(by='Importanza', ascending=True).tail(10)

    plt.figure(figsize=(10, 6))
    plt.barh(feat_df['Feature'], feat_df['Importanza'], color='cornflowerblue')
    plt.title('Top 10 Feature più importanti (Random Forest)')
    plt.tight_layout()
    plt.savefig(os.path.join(graphics_dir, 'feature_importance.png'))

    print("Grafici salvati nella cartella 'graphics'.")