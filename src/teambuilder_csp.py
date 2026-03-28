import pandas as pd
from constraint import *


# Funzione che carica il dataset e preleva un campione casuale di 80 Pokémon.
# Pesca da tutti i tier, affidandosi ai vincoli del CSP per bilanciare la potenza della squadra,
# ma pretende che ogni Pokémon abbia un ruolo tattico validato dall'ontologia.
def load_dynamic_pool(csv_path):
    df = pd.read_csv(csv_path)
    df_validi = df.dropna(subset=['smogon_tier', 'inferred_role', 'type1'])
    df_validi = df_validi[df_validi['inferred_role'] != "None"]

    # Peschiamo 80 Pokémon a caso tra tutti i validi.
    # Estraiamo pool finché non ne troviamo uno "promettente" per prevenire l'esplosione combinatoria.
    while True:
        pool_dinamico = df_validi.sample(n=80)

        ruoli_presenti = pool_dinamico['inferred_role'].tolist()
        tipi_presenti = pool_dinamico['type1'].unique()

        ha_attaccante = 'PhysicalSweeper' in ruoli_presenti or 'SpecialSweeper' in ruoli_presenti
        ha_difensore = 'PhysicalWall' in ruoli_presenti or 'SpecialWall' in ruoli_presenti
        ha_varieta_tipi = len(tipi_presenti) >= 6

        if ha_attaccante and ha_difensore and ha_varieta_tipi:
            return pool_dinamico


# Funzione che imposta i vincoli del CSP (unicità, tipi diversi, regole Smogon, sinergia tattica).
# Avvia la ricerca nello spazio degli stati e restituisce la prima squadra valida trovata.
def solve_teambuilder(pool_df):
    problem = Problem()

    pokemon_dict = {}
    for index, row in pool_df.iterrows():
        pokemon_dict[row['name']] = {
            'type1': row['type1'],
            'type2': row['type2'],
            'tier': row['smogon_tier'],
            'role': row['inferred_role']
        }

    slots = ['Slot_1', 'Slot_2', 'Slot_3', 'Slot_4', 'Slot_5', 'Slot_6']
    problem.addVariables(slots, list(pokemon_dict.keys()))

    # Vincolo 1: impone che tutti e 6 gli slot contengano Pokémon diversi.
    problem.addConstraint(AllDifferentConstraint())

    def diff_types(*squadra):
        tipi_primari = [pokemon_dict[p]['type1'] for p in squadra]
        return len(set(tipi_primari)) == 6

    # Vincolo 2: impone che i tipi primari (type1) dei 6 Pokémon scelti siano tutti differenti tra loro.
    problem.addConstraint(diff_types, slots)

    def smogon_rules(*squadra):
        tiers = [pokemon_dict[p]['tier'] for p in squadra]
        uber_count = tiers.count('Uber')
        ou_count = tiers.count('OverUsed')
        return uber_count <= 1 and ou_count <= 2

    # Vincolo 3: evita la creazione di team sbilanciati composti solo da Pokémon leggendari.
    # Il vincolo limita la squadra a un massimo di 1 Pokémon 'Uber' e un massimo di 2 'OverUsed'.
    problem.addConstraint(smogon_rules, slots)

    def role_synergy(*squadra):
        ruoli = [pokemon_dict[p]['role'] for p in squadra]
        ha_attaccante = 'PhysicalSweeper' in ruoli or 'SpecialSweeper' in ruoli
        ha_difensore = 'PhysicalWall' in ruoli or 'SpecialWall' in ruoli
        return ha_attaccante and ha_difensore

    # Vincolo 4: sfrutta i ruoli inferiti dall'ontologia, obbligando la squadra ad avere almeno
    # un attaccante (Sweeper) e almeno un difensore (Wall), garantendo un team bilanciato.
    problem.addConstraint(role_synergy, slots)

    print("Ricerca della prima squadra ottimale nell'albero delle combinazioni...")
    return problem.getSolution(), pokemon_dict


# Funzione che formatta e stampa a video il risultato restituito dal risolutore CSP.
def print_team(soluzione, dict_dati):
    if soluzione:
        print("\n --- SQUADRA PERFETTA TROVATA! ---")
        for slot in sorted(soluzione.keys()):
            p_name = soluzione[slot]
            dati = dict_dati[p_name]
            print(
                f" > {slot}: {p_name:<20} | Tipo: {dati['type1']:<8} | Tier: {dati['tier']:<10} | Ruolo: {dati['role']}")
    else:
        print("\nNessuna squadra trovata per questo pool. Riprova.")