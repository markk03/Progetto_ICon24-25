import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import learning_curve
import warnings

warnings.filterwarnings("ignore")


# CARICAMENTO DATASET
def load_datasets(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['smogon_tier'])
    y = df['smogon_tier']

    features_base = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']
    X_base = df[features_base]

    # Encoding dei ruoli inferiti dall'ontologia
    df_encoded = pd.get_dummies(df, columns=['inferred_role'])
    role_columns = [col for col in df_encoded.columns if 'inferred_role_' in col]
    features_enriched = features_base + ['real_weaknesses'] + role_columns
    X_enriched = df_encoded[features_enriched]

    return X_base, X_enriched, y, df_encoded, features_enriched


# 2. VALUTAZIONE MODELLI (Nested Cross-Validation)
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

    # Griglia di iperparametri per i due modelli scelti
    models_and_grids = {
        "k-NN (Instance-based)": (KNeighborsClassifier(), {
            'classifier__n_neighbors': [3, 5, 11],
            'classifier__weights': ['uniform', 'distance']
        }),
        "SVM (Kernel RBF)": (SVC(random_state=42), {
            'classifier__C': [0.1, 1, 10],
            'classifier__kernel': ['rbf']
        })
    }

    for name, (model, param_grid) in models_and_grids.items():
        print(f" Valutazione di {name}...")
        pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', model)])

        clf = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=inner_cv, scoring='f1_macro', n_jobs=-1)
        results = cross_validate(clf, X, y, cv=outer_cv, scoring=scoring, n_jobs=-1)

        acc_scores = results['test_accuracy'] * 100
        prec_scores = results['test_precision'] * 100
        rec_scores = results['test_recall'] * 100
        f1_scores = results['test_f1'] * 100

        # Output statistico completo (Media, Std, Varianza)
        print(
            f"    Accuracy:  Media = {acc_scores.mean():.2f}% | Std = {acc_scores.std():.2f}% | Var = {acc_scores.var():.4f}")
        print(
            f"    Precision: Media = {prec_scores.mean():.2f}% | Std = {prec_scores.std():.2f}% | Var = {prec_scores.var():.4f}")
        print(
            f"    Recall:    Media = {rec_scores.mean():.2f}% | Std = {rec_scores.std():.2f}% | Var = {rec_scores.var():.4f}")
        print(
            f"    F1-Score:  Media = {f1_scores.mean():.2f}% | Std = {f1_scores.std():.2f}% | Var = {f1_scores.var():.4f}\n")

        results_dict[phase_key][name] = f1_scores.mean()


# GENERAZIONE GRAFICI DIAGNOSTICI
def generate_plots(results_dict, X_enriched, y, features_enriched, graphics_dir):
    print("\nGenerazione grafici di validazione in corso...")

    # A. GRAFICO COMPARATIVO (F1-Score)
    modelli_nomi = list(results_dict['Baseline'].keys())
    f1_base = [results_dict['Baseline'][m] for m in modelli_nomi]
    f1_onto = [results_dict['OntoBK'][m] for m in modelli_nomi]

    x = np.arange(len(modelli_nomi))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x - width / 2, f1_base, width, label='Baseline', color='indianred', alpha=0.8)
    ax.bar(x + width / 2, f1_onto, width, label='OntoBK', color='mediumseagreen', alpha=0.8)
    ax.set_ylabel('F1-Score Medio (%)')
    ax.set_title('Efficacia Ontologica: Baseline vs OntoBK')
    ax.set_xticks(x)
    ax.set_xticklabels(modelli_nomi)
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(graphics_dir, 'confronto_f1_knn_svm.png'))
    plt.close()

    # LEARNING CURVES
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_enriched)

    models_to_analyze = {
        "k-NN": KNeighborsClassifier(n_neighbors=5),
        "SVM (Kernel)": SVC(C=1, kernel='rbf', random_state=42)
    }

    for name, model in models_to_analyze.items():
        print(f"  Analisi diagnostica: {name}...")
        train_sizes, train_scores, test_scores = learning_curve(
            model, X_scaled, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10), scoring='f1_macro'
        )

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        plt.figure(figsize=(9, 6))
        plt.title(f"Diagnosi Modello: Learning Curve ({name})")
        plt.xlabel("Campioni di Addestramento")
        plt.ylabel("F1-Score (Macro)")
        plt.grid(True, alpha=0.3, linestyle='--')

        # Area di incertezza (Deviazione Standard)
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="firebrick")
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15, color="forestgreen")


        plt.plot(train_sizes, train_mean, 'o-', color="firebrick", linewidth=2, label="Training Score")
        plt.plot(train_sizes, test_mean, 's-', color="forestgreen", linewidth=2, label="Cross-Validation Score")

        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(graphics_dir, f'learning_curve_{name.lower().replace(" ", "_")}.png'))
        plt.close()