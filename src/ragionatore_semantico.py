import pandas as pd
from owlready2 import *


# Funzione che carica l'ontologia di base e avvia il ragionatore semantico HermiT.
# Inferisce automaticamente la nuova conoscenza e salva i risultati.
def run_reasoning(input_path, output_path):
    try:
        onto = get_ontology(input_path).load()
    except FileNotFoundError:
        print(f"[ERRORE] File '{input_path}' non trovato.")
        return None

    print("Avvio di HermiT per il ragionamento logico...")
    with onto:
        sync_reasoner(infer_property_values=True)

    onto.save(file=output_path, format="rdfxml")
    print(f"Ontologia inferita salvata in '{output_path}'.")
    return onto


# Funzione che applica la logica euristica del Sistema Esperto per classificare ogni Pokémon nei tier
# competitivi incrociando statistiche base, vulnerabilità e ruoli.
def calculate_smogon_tier(bst, num_debolezze, ruolo):
    if bst >= 600 or (bst >= 550 and num_debolezze <= 1):
        return "Uber"
    elif bst >= 500 and ruolo != "None" and num_debolezze <= 3:
        return "OverUsed"
    elif bst >= 450 or (bst >= 400 and ruolo != "None"):
        return "UnderUsed"
    elif bst >= 350:
        return "RarelyUsed"
    else:
        return "NeverUsed"


# Funzione che estrae la conoscenza inferta dall'ontologia e la integra nel dataset originale
# per generare la versione arricchita (OntoBK).
def export_enriched_dataset(onto, csv_source, csv_output):
    df = pd.read_csv(csv_source)
    inferred_roles = []
    real_weaknesses_count = []
    smogon_tiers = []

    role_map = {}
    roles_list = ["PhysicalSweeper", "SpecialSweeper", "PhysicalWall", "SpecialWall", "PhysicalGlassCannon",
                  "SpecialGlassCannon", "BulkyOffense"]
    for r_name in roles_list:
        r_inst = onto.search_one(iri=f"*{r_name}")
        if r_inst:
            for p in onto.search(has_role=r_inst):
                role_map[p.name] = r_name

    def clean_text(t):
        if pd.isna(t) or str(t).strip() == "": return None
        return str(t).strip().replace("'", "").replace("’", "").replace(".", "").title().replace(" ", "_").replace("-",
                                                                                                                   "_")

    for index, row in df.iterrows():
        poke_name = clean_text(row['name'])
        p_instance = onto.search_one(iri=f"*{poke_name}")
        bst = row['hp'] + row['attack'] + row['defense'] + row['sp_attack'] + row['sp_defense'] + row['speed']

        ruolo = role_map.get(poke_name, "None")
        inferred_roles.append(ruolo)

        num_debolezze = 0
        if p_instance:
            raw_w = set([w.name for w in p_instance.has_vulnerability])
            imm = set([i.name for i in p_instance.has_immunity])
            actual_w = raw_w - imm
            num_debolezze = len(actual_w)

        real_weaknesses_count.append(num_debolezze)
        tier = calculate_smogon_tier(bst, num_debolezze, ruolo)
        smogon_tiers.append(tier)

    df['inferred_role'] = inferred_roles
    df['real_weaknesses'] = real_weaknesses_count
    df['smogon_tier'] = smogon_tiers

    df.to_csv(csv_output, index=False)
    print(f"Dataset arricchito salvato in '{csv_output}'.")