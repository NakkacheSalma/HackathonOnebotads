
import streamlit as st
import json
import time
from agent_ia_lib import *

st.set_page_config(page_title="OneBotAds - Agent IA", layout="centered")
st.title("🤖 Démonstration Agent IA OneBotAds")

st.markdown("""
Ce démonstrateur montre un agent IA autonome qui :
- 🔍 extrait les données de campagne à partir d’une simple description utilisateur,
- 🧪 effectue des split-tests créatifs,
- 🎯 génère 10 AdSets optimisés,
- 📊 simule les performances sur 4 jours,
- 🧠 recommande la meilleure stratégie future,
- 📝 produit un résumé de campagne prêt à utiliser.

**Entrez votre brief ci-dessous 👇**
""")

# Initialiser une variable de session pour garder l'état
if "infos_partielles" not in st.session_state:
    st.session_state["infos_partielles"] = None
if "infos_completes_validees" not in st.session_state:
    st.session_state["infos_completes_validees"] = False
if "infos" not in st.session_state:
    st.session_state["infos"] = {}

# Entrée utilisateur
description = st.text_area(
    "📝 Décris ta campagne publicitaire",
    placeholder="Ex : Je veux promouvoir un complément alimentaire pour femmes de 25-34 ans sur Instagram...",
    height=100
)

# Bouton pour lancer l'extraction
if st.button("🚀 Lancer le Workflow IA"):
    st.session_state["infos_completes_validees"] = False
    st.session_state["infos_partielles"] = None
    st.session_state["infos"] = {}

    st.markdown("---")
    st.subheader("🧠 1. Extraction des informations")
    with st.spinner("Analyse IA avec Gemma..."):
        infos_partielles = extraire_infos_partielles(description)
        time.sleep(1)

    if not infos_partielles:
        st.error("❌ Extraction échouée.")
        st.stop()

    st.success("✅ Extraction terminée")
    st.json(infos_partielles)

    st.session_state["infos_partielles"] = infos_partielles
    st.session_state["infos"] = infos_partielles.copy()

# Si extraction réussie, affichage du formulaire de complétion
if st.session_state["infos_partielles"]:
    st.subheader("📋 2. Complétion manuelle des infos manquantes")
    st.markdown("Complète les champs manquants ci-dessous 👇")

    champs_requis = {
        "nom_entreprise": "Nom de l'entreprise",
        "plateforme_publicité": "Plateforme de publicité",
        "objet_publicité": "Produit ou service promu",
        "localisation_audience": "Localisation de l'audience",
        "tranche_age": "Tranche d'âge",
        "genre": "Genre ciblé",
        "budget": "Budget (en €)",
        "duree": "Durée (en jours)"
    }

    infos = st.session_state["infos"]

    for champ, label in champs_requis.items():
        # Afficher un input uniquement si la valeur est manquante ou nulle
        if infos.get(champ) in [None, "null", "", "inconnu"]:
            infos[champ] = st.text_input(label, key=champ)
        else:
            st.text(f"{label} : {infos[champ]}")

    # Bouton pour valider les infos complètes
    if st.button("Valider les informations complètes"):
        # Vérifier que tous les champs sont remplis
        if not all(infos.get(k) for k in champs_requis.keys()):
            st.warning("🕐 Merci de compléter tous les champs avant de continuer.")
        else:
            # Mettre à jour les infos partielles avec les compléments saisis
            for champ in champs_requis.keys():
                if st.session_state["infos_partielles"].get(champ) in [None, "null", "", "inconnu"]:
                    st.session_state["infos_partielles"][champ] = infos[champ]
            st.success("✅ Infos complètes prêtes")
            st.json(infos)
            st.session_state["infos_completes_validees"] = True
            st.session_state["infos"] = infos

# Si les infos complètes ont été validées, poursuivre le workflow
if st.session_state["infos_completes_validees"]:
    infos = st.session_state["infos"]
    st.subheader("🧪 3. Split Tests IA")
    produit = infos["objet_publicité"]
    alpha_weights = [0.5, 0.3, 0.2]

    text_options = [prompt_adcopy(produit) for _ in range(3)]
    image_options = [prompt_image_generation(produit) for _ in range(3)]
    age_options = ["18-24", "25-34", "35-44"]

    best_text = split_test("text", text_options, infos, jour=1, alpha_weights=alpha_weights)
    best_image = split_test("image_prompt", image_options, infos, jour=2, alpha_weights=alpha_weights)
    best_age = split_test("tranche_age", age_options, infos, jour=3, alpha_weights=alpha_weights)

    st.json({
        "Texte choisi": best_text,
        "Image choisie": best_image,
        "Tranche d'âge choisie": best_age
    })

    st.subheader("🧠 4. Génération des 10 AdSets optimisés")
    infos["text"] = best_text["option"]
    infos["image_prompt"] = best_image["option"]
    infos["tranche_age"] = best_age["option"]

    adsets = generer_adsets_depuis_objectif(infos, nb_adsets=10)

    with open("adsets.json", "w", encoding="utf-8") as f:
        json.dump(adsets, f, indent=4, ensure_ascii=False)

    st.success("✅ AdSets générés")
    for i, ad in enumerate(adsets, start=1):
        with st.expander(f"📦 AdSet {i}"):
            st.json(ad)

    with open("adsets.json", "r", encoding="utf-8") as f:
        adsets_content = f.read()
    st.download_button("📥 Télécharger les AdSets", adsets_content, file_name="adsets.json", mime="application/json")

    st.subheader("📊 5. Simulation sur 4 jours")
    all_performances = []
    for jour in range(4, 8):
        st.markdown(f"### 📆 Jour {jour}")
        perf = rapport_journalier(adsets, jour=jour)
        all_performances.extend(perf)
        st.json(perf)

        with open(f"rapport_jour_{jour}.json", "r", encoding="utf-8") as f:
            content = f.read()
        st.download_button(f"📥 Télécharger rapport jour {jour}", content, file_name=f"rapport_jour_{jour}.json", mime="application/json")

    st.subheader("🧠 6. Recommandation IA finale")
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder

    encoder = LabelEncoder()
    X, y = [], []
    for perf in all_performances:
        age = infos.get("tranche_age", "18-24")
        fmt = "VIDEO" if "video_script" in perf else "IMAGE"
        a_enc = encoder.fit_transform([age])[0]
        f_enc = encoder.fit_transform([fmt])[0]
        X.append([a_enc, f_enc])
        y.append(perf["roas"])

    model = RandomForestRegressor()
    model.fit(X, y)

    best_pred = 0
    best_combo = (None, None)
    for age in ["18-24", "25-34", "35-44"]:
        for fmt in ["VIDEO", "IMAGE"]:
            a_enc = encoder.fit_transform([age])[0]
            f_enc = encoder.fit_transform([fmt])[0]
            pred = model.predict([[a_enc, f_enc]])[0]
            if pred > best_pred:
                best_pred = pred
                best_combo = (age, fmt)

    st.success(f"🎯 Meilleure stratégie prédite : {best_combo[0]} + {best_combo[1]} (ROAS estimé = {round(best_pred, 2)})")

    st.subheader("📌 7. Résumé complet de la campagne")
    import glob

    split_tests = {}
    for file in glob.glob("split_test_*.json"):
        var = file.split("_")[2]
        jour = file.split("_")[-1].split(".")[0]
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
            file_content = json.dumps(data, indent=4)
        if var not in split_tests:
            split_tests[var] = []
        split_tests[var].append({"jour": jour, "options": data})
        st.download_button(f"📥 Télécharger split test {var} - jour {jour}", file_content, file_name=file, mime="application/json")

    generer_resume_campagne(infos, split_tests, all_performances, best_combo)
    with open("resume_campagne.json", "r", encoding="utf-8") as f:
        resume = json.load(f)
    st.json(resume)

    resume_content = json.dumps(resume, indent=4, ensure_ascii=False)
    st.download_button("📥 Télécharger le résumé de campagne", resume_content, file_name="resume_campagne.json", mime="application/json")

    st.balloons()


