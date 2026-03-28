import pandas as pd
from owlready2 import *


# Funzione che pulisce e normalizza una stringa di testo rimuovendo caratteri speciali.
# Restituisce il testo in formato Title_Case, oppure None se il campo è vuoto.
def clean_text(testo):
    if pd.isna(testo) or str(testo).strip() == "":
        return None
    testo = str(testo).strip().replace("'", "").replace("’", "").replace(".", "")
    return testo.title().replace(" ", "_").replace("-", "_")


# Funzione che costruisce lo schema dell'ontologia (TBox).
# Definisce le classi principali, le proprietà degli oggetti e dei dati, e le regole SWRL.
def build_tbox(onto):
    with onto:
        class Pokemon(Thing):
            pass

        class Type(Thing):
            pass

        class Role(Thing):
            pass

        class has_type(ObjectProperty):
            domain = [Pokemon]
            range = [Type]

        class has_role(ObjectProperty):
            domain = [Pokemon]
            range = [Role]

        class is_weak_to(ObjectProperty):
            domain = [Type]
            range = [Type]

        class is_immune_to(ObjectProperty):
            domain = [Type]
            range = [Type]

        class has_vulnerability(ObjectProperty):
            domain = [Pokemon]
            range = [Type]
            # Property Chain: Tipo + Debolezza = Pokémon Vulnerabile
            property_chain = [PropertyChain([has_type, is_weak_to])]

        class has_immunity(ObjectProperty):
            domain = [Pokemon]
            range = [Type]
            # Property Chain: Tipo + Immunità = Pokémon Immune
            property_chain = [PropertyChain([has_type, is_immune_to])]

        class has_high_hp(DataProperty, FunctionalProperty):
            domain = [Pokemon]
            range = [bool]

        class has_low_hp(DataProperty, FunctionalProperty):
            domain = [Pokemon]
            range = [bool]

        class has_high_attack(DataProperty, FunctionalProperty):
            domain = [Pokemon]
            range = [bool]

        class has_high_sp_attack(DataProperty, FunctionalProperty):
            domain = [Pokemon]
            range = [bool]

        class has_high_defense(DataProperty, FunctionalProperty):
            domain = [Pokemon]
            range = [bool]

        class has_high_sp_defense(DataProperty, FunctionalProperty):
            domain = [Pokemon]
            range = [bool]

        class has_low_defense(DataProperty, FunctionalProperty):
            domain = [Pokemon]
            range = [bool]

        class has_high_speed(DataProperty, FunctionalProperty):
            domain = [Pokemon]
            range = [bool]

        class has_low_speed(DataProperty, FunctionalProperty):
            domain = [Pokemon]
            range = [bool]

        # Creazione diretta degli individui (Ruoli) richiamandoli dall'ontologia
        onto.Role("PhysicalSweeper")
        onto.Role("SpecialSweeper")
        onto.Role("PhysicalWall")
        onto.Role("SpecialWall")
        onto.Role("PhysicalGlassCannon")
        onto.Role("SpecialGlassCannon")
        onto.Role("BulkyOffense")

        # Regole SWRL
        rules = [
            "Pokemon(?p), has_high_attack(?p, true), has_high_speed(?p, true) -> has_role(?p, PhysicalSweeper)",
            "Pokemon(?p), has_high_sp_attack(?p, true), has_high_speed(?p, true) -> has_role(?p, SpecialSweeper)",
            "Pokemon(?p), has_high_hp(?p, true), has_high_defense(?p, true) -> has_role(?p, PhysicalWall)",
            "Pokemon(?p), has_high_hp(?p, true), has_high_sp_defense(?p, true) -> has_role(?p, SpecialWall)",
            "Pokemon(?p), has_high_attack(?p, true), has_low_defense(?p, true), has_low_hp(?p, true) -> has_role(?p, PhysicalGlassCannon)",
            "Pokemon(?p), has_high_sp_attack(?p, true), has_low_defense(?p, true), has_low_hp(?p, true) -> has_role(?p, SpecialGlassCannon)",
            "Pokemon(?p), has_high_attack(?p, true), has_high_hp(?p, true), has_low_speed(?p, true) -> has_role(?p, BulkyOffense)"
        ]
        for r in rules: Imp().set_as_rule(r)


# Funzione che popola l'ontologia con le relazioni elementali di base.
# Inserisce per ogni Tipo le rispettive debolezze e immunità.
def init_background_knowledge(onto):
    type_weaknesses = {
        "Normal": ["Fighting"], "Fire": ["Water", "Rock", "Ground"], "Water": ["Grass", "Electric"],
        "Electric": ["Ground"], "Grass": ["Fire", "Ice", "Bug", "Flying", "Poison"],
        "Ice": ["Fire", "Fighting", "Rock", "Steel"], "Fighting": ["Flying", "Psychic", "Fairy"],
        "Ground": ["Water", "Grass", "Ice"], "Flying": ["Electric", "Ice", "Rock"],
        "Psychic": ["Bug", "Ghost", "Dark"], "Bug": ["Fire", "Flying", "Rock"],
        "Rock": ["Water", "Grass", "Fighting", "Ground", "Steel"], "Ghost": ["Ghost", "Dark"],
        "Dragon": ["Ice", "Dragon", "Fairy"], "Dark": ["Fighting", "Bug", "Fairy"],
        "Steel": ["Fire", "Fighting", "Ground"], "Fairy": ["Poison", "Steel"], "Poison": ["Ground", "Psychic"]
    }
    type_immunities = {
        "Fire": [], "Water": [], "Electric": [], "Grass": [], "Ice": [], "Fighting": [],
        "Poison": [], "Psychic": [], "Bug": [], "Dragon": [], "Rock": [],
        "Flying": ["Ground"], "Steel": ["Poison"], "Normal": ["Ghost"],
        "Ghost": ["Normal", "Fighting"], "Ground": ["Electric"], "Fairy": ["Dragon"], "Dark": ["Psychic"]
    }

    with onto:
        for t_name, weaknesses in type_weaknesses.items():
            t_instance = onto.Type(t_name)
            for w_name in weaknesses: t_instance.is_weak_to.append(onto.Type(w_name))
        for t_name, immunities in type_immunities.items():
            t_instance = onto.Type(t_name)
            for i_name in immunities: t_instance.is_immune_to.append(onto.Type(i_name))


# Funzione che legge il dataset CSV originale e popola la ABox dell'ontologia.
# Crea gli individui Pokémon e assegna le loro proprietà booleane in base alle statistiche.
def populate_abox(onto, csv_path):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"[ERRORE] File '{csv_path}' non trovato.")
        return

    for index, row in df.iterrows():
        poke_name = clean_text(row['name'])
        p = onto.Pokemon(poke_name)

        p.has_high_hp = bool(row['hp'] >= 85)
        p.has_low_hp = bool(row['hp'] <= 65)
        p.has_high_attack = bool(row['attack'] >= 90)
        p.has_high_sp_attack = bool(row['sp_attack'] >= 90)
        p.has_high_defense = bool(row['defense'] >= 90)
        p.has_high_sp_defense = bool(row['sp_defense'] >= 90)
        p.has_low_defense = bool(row['defense'] <= 65)
        p.has_high_speed = bool(row['speed'] >= 85)
        p.has_low_speed = bool(row['speed'] <= 65)

        t1_clean = clean_text(row['type1'])
        if t1_clean: p.has_type.append(onto.Type(t1_clean))
        t2_clean = clean_text(row['type2'])
        if t2_clean: p.has_type.append(onto.Type(t2_clean))

    print(f"Inseriti {len(list(onto.Pokemon.instances()))} Pokémon.")
