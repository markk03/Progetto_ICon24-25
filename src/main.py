import os
import costruzione_ontologia
import ragionatore_semantico
import apprendimento_supervisionato
import teambuilder_csp
from owlready2 import get_ontology

# Configurazione dinamica dei percorsi
DIR_SRC = os.path.dirname(os.path.abspath(__file__))
DIR_ROOT = os.path.dirname(DIR_SRC)

# Definisce le cartelle del progetto
DIR_DATASET = os.path.join(DIR_ROOT, 'dataset')
DIR_ONTOLOGY = os.path.join(DIR_ROOT, 'ontology')
DIR_GRAPHICS = os.path.join(DIR_ROOT, 'graphics')

# Definisce i percorsi esatti dei file
CSV_BASE = os.path.join(DIR_DATASET, 'pokemon.csv')
CSV_ENRICHED = os.path.join(DIR_DATASET, 'pokemon_enriched.csv')
OWL_BASE = os.path.join(DIR_ONTOLOGY, 'pokemon_base.owl')
OWL_INFERRED = os.path.join(DIR_ONTOLOGY, 'pokemon_inferred.owl')


# La funzione main è la funzione orchestratrice del sistema esperto, che coordina il flusso di lavoro attraverso
# quattro fasi sequenziali. Partendo dalla costruzione dell'ontologia e dall'integrazione della background knowledge,
# il programma attiva il ragionatore semantico HermiT per dedurre nuovi ruoli tattici e generare un dataset arricchito.
# Questi dati semantici alimentano sia la fase di apprendimento supervisionato sia il modulo CSP finale, che utilizza le
# inferenze logiche ottenute per pianificare e comporre una squadra di Pokémon bilanciata e competitiva.
def main():
    print("\n" + "=" * 60)
    print("SISTEMA ESPERTO POKÉMON")
    print("=" * 60)

    print("\nFASE 1: Costruzione dell'ontologia")
    ontologia = get_ontology("http://ic.esame.it/pokemon.owl")
    costruzione_ontologia.build_tbox(ontologia)
    costruzione_ontologia.init_background_knowledge(ontologia)
    costruzione_ontologia.populate_abox(ontologia, CSV_BASE)
    ontologia.save(file=OWL_BASE, format="rdfxml")
    print(f"File salvato in: {OWL_BASE}")

    print("\nFASE 2: Ragionamento Semantico (HermiT)")
    ontologia_inferita = ragionatore_semantico.run_reasoning(OWL_BASE, OWL_INFERRED)
    if ontologia_inferita:
        ragionatore_semantico.export_enriched_dataset(ontologia_inferita, CSV_BASE, CSV_ENRICHED)

    print("\nFASE 3: Apprendimento Supervisionato")
    X_base, X_enriched, y, df_encoded, features_enriched = apprendimento_supervisionato.load_datasets(CSV_ENRICHED)
    risultati = {'Baseline': {}, 'OntoBK': {}}

    apprendimento_supervisionato.run_ml_evaluation(X_base, y, "Addestramento Baseline (Senza Ontologia)...", risultati,
                                                   'Baseline')
    apprendimento_supervisionato.run_ml_evaluation(X_enriched, y, "Addestramento OntoBK (Con Ontologia)...", risultati,
                                                   'OntoBK')
    apprendimento_supervisionato.generate_plots(risultati, X_enriched, y, features_enriched, DIR_GRAPHICS)

    print("\nFASE 4: Risoluzione Vincoli (Teambuilder CSP)")
    dati_campione = teambuilder_csp.load_dynamic_pool(CSV_ENRICHED)
    team_solution, info_dict = teambuilder_csp.solve_teambuilder(dati_campione)
    teambuilder_csp.print_team(team_solution, info_dict)


if __name__ == "__main__":
    main()