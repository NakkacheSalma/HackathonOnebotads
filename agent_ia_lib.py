
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, random, json, uuid
from datetime import datetime, timedelta

model_id = "google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

def generate_with_gemma(prompt, max_new_tokens=200):
    input_tokens = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**input_tokens, max_new_tokens=max_new_tokens, return_dict_in_generate=True)
    generated_tokens = outputs.sequences[0][input_tokens["input_ids"].shape[1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

def extraire_infos_partielles(texte_utilisateur):
    prompt = f"""<start_of_turn>user
Voici un texte décrivant une campagne publicitaire. Extrait les informations suivantes dans un dictionnaire JSON :
nom_entreprise, plateforme_publicité, objet_publicité, localisation_audience, tranche_age, genre, budget, duree.
Si une information n'est pas trouvée, mets sa valeur à null.
Texte : '''{texte_utilisateur}'''
Réponds uniquement avec le JSON.
<end_of_turn>
<start_of_turn>model
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=256)
    reponse = tokenizer.decode(outputs[0], skip_special_tokens=True)
    try:
        json_debut = reponse.find("{")
        json_fin = reponse.rfind("}") + 1
        return json.loads(reponse[json_debut:json_fin])
    except: return None

def prompt_adcopy(objet):
    tons = ["dynamique", "inspirant", "fun"]
    angles = ["transformation", "bien-être", "urgence"]
    prompt = f"""<start_of_turn>user
Tu es un copywriter. Rédige un texte court (max 20 mots) :
Produit : {objet}
Ton : {random.choice(tons)}
Angle : {random.choice(angles)}
Public : jeunes adultes 18-35 ans
<end_of_turn>
<start_of_turn>model
"""
    return generate_with_gemma(prompt)

def prompt_image_generation(objet):
    formats = ["illustration", "photo", "mise en scène"]
    couleurs = ["vives", "pastels", "contrastées"]
    prompt = f"""<start_of_turn>user
Décris une image publicitaire pour : {objet}
Format : {random.choice(formats)} / Couleurs : {random.choice(couleurs)}
<end_of_turn>
<start_of_turn>model
"""
    return generate_with_gemma(prompt)

def prompt_video_script(objet):
    prompt = f"""<start_of_turn>user
Script vidéo publicitaire (3 scènes) pour : {objet}
<end_of_turn>
<start_of_turn>model
"""
    return generate_with_gemma(prompt, 300)

def generer_adsets_depuis_objectif(infos, nb_adsets=10):
    genre_map = {"homme": "GENDER_MALE", "femme": "GENDER_FEMALE", "tous": "GENDER_UNDEFINED"}
    cible_genre = genre_map.get(infos.get("genre", "tous").lower(), "GENDER_UNDEFINED")
    age_map = {"18-24": ["AGE_18_24"], "25-34": ["AGE_25_34"], "35-44": ["AGE_35_44"]}
    targeting_age = age_map.get(infos.get("tranche_age", "18-24"), ["AGE_18_24"])
    loc = infos.get("localisation_audience", "FR")
    adsets, start = [], datetime.now() + timedelta(days=1)
    end = start + timedelta(days=int(infos.get("duree", 10)))
    budget_journalier = int(infos.get("budget", 500)) // nb_adsets
    for i in range(nb_adsets):
        format = random.choice(["SINGLE_IMAGE", "VIDEO"])
        copy = prompt_adcopy(infos["objet_publicité"]).strip()
        content = {"image_prompt": prompt_image_generation(infos["objet_publicité"]).strip()} if format=="SINGLE_IMAGE" else {"video_script": prompt_video_script(infos["objet_publicité"]).strip()}
        adsets.append({
            "adgroup": {
                "advertiser_id": "7443888987275542529",
                "campaign_id": str(uuid.uuid4()),
                "adgroup_name": f"AdSet {i+1}",
                "placement_type": "PLACEMENT_TYPE_AUTO",
                "external_action": "LINK_CLICK",
                "optimization_goal": "CLICK",
                "billing_event": "CLICK",
                "budget_mode": "BUDGET_MODE_DAY",
                "budget": budget_journalier,
                "schedule_type": "SCHEDULE_START_END",
                "start_time": start.strftime("%Y-%m-%d %H:%M:%S"),
                "end_time": end.strftime("%Y-%m-%d %H:%M:%S"),
                "bid": 100,
                "targeting": {
                    "age": targeting_age,
                    "gender": cible_genre,
                    "location": {"country": [loc]},
                    "language": ["fr"],
                    "interest_category_v2": [random.choice(["600001", "600002", "600003"])]
                }
            },
            "creative": {
                "ad_name": f"Creative {i+1}",
                "creative_material_mode": "CUSTOM_CREATIVE",
                "creatives": [{
                    "ad_format": format,
                    "image_mode": "SQUARE",
                    "image_ids": ["REPLACE_WITH_IMAGE_ID"],
                    "title": copy,
                    "call_to_action": "LEARN_MORE",
                    **content
                }]
            }
        })
    return adsets

def simuler_performances(ad):
    return {
        "ad_id": ad["adgroup"]["adgroup_name"],
        "spend": round(random.uniform(5, 25), 2),
        "clicks": random.randint(0, 50),
        "conversions": random.randint(0, 10)
    }

def calculer_roas(perf, revenu_par_conversion=20):
    return round((perf["conversions"] * revenu_par_conversion) / (perf["spend"] or 1), 2)

def prendre_decision(roas, seuil=1.2):
    return "DISABLE" if roas < seuil else "SCALE"

def split_test(variable_name, options, base_infos, jour, alpha_weights):
    results = []
    for opt in options:
        infos = base_infos.copy()
        infos[variable_name] = opt
        ad = generer_adsets_depuis_objectif(infos, 1)[0]
        perf = simuler_performances(ad)
        roas = calculer_roas(perf)
        ctr = perf["clicks"] / (perf["spend"] or 1)
        conv = perf["conversions"] / (perf["clicks"] or 1)
        cpl = perf["spend"] / (perf["conversions"] or 1)
        score = alpha_weights[0]*ctr + alpha_weights[1]*conv - alpha_weights[2]*cpl
        results.append({"option": opt, "score": round(score, 3), "roas": roas, "ctr": round(ctr,2), "conv_rate": round(conv,2), "cpl": round(cpl,2), "decision": prendre_decision(roas)})
    with open(f"split_test_{variable_name}_jour_{jour}.json", "w", encoding="utf-8") as f: json.dump(results, f, indent=4)
    return max(results, key=lambda x: x["score"])

def rapport_journalier(adsets, jour=1):
    journal = []
    for ad in adsets:
        perf = simuler_performances(ad)
        roas = calculer_roas(perf)
        decision = prendre_decision(roas)
        perf.update({"roas": roas, "decision": decision})
        journal.append(perf)
    with open(f"rapport_jour_{jour}.json", "w") as f:
        json.dump(journal, f, indent=4)
    return journal

def generer_resume_campagne(infos_completes, split_tests, performances, recommandation):
    top3 = sorted(performances, key=lambda x: x["roas"], reverse=True)[:3]
    resume = {
        "objectif": infos_completes,
        "split_tests": split_tests,
        "meilleurs_adsets": top3,
        "roi_moyen_top3": round(sum(p["roas"] for p in top3)/len(top3), 2),
        "recommandation_modele": {
            "meilleure_tranche_age": recommandation[0],
            "meilleur_format": recommandation[1]
        }
    }
    with open("resume_campagne.json", "w") as f:
        json.dump(resume, f, indent=4)
