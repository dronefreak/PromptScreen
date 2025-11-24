import json
from omegaconf import DictConfig

from defence.abstract_defence import AbstractDefence
from defence.classifier_cluster import ClassifierCluster
from defence.heuristic_channel import HeuristicVectorAnalyzer
from defence.linear_svm import JailbreakInferenceAPI
from defence.shieldgemma import ShieldGemma2BClassifier
from defence.vectordb import VectorDBScanner, VectorDB
from defence.scanner import Scanner


def initialize_guards(cfg: DictConfig) -> dict[str, AbstractDefence]:
    db_client = VectorDB(
        model=cfg.vectordb.model,
        collection=cfg.vectordb.collection,
        db_dir=cfg.vectordb.db_dir,
        n_results=cfg.vectordb.n_results,
    )

    try:
        with open(cfg.threat_file, "r") as f:
            threat_data = json.load(f)
            known_threats = [entry['prompt'] for entry in threat_data]
            metadatas = [{'category': entry.get('category', 'unknown')} for entry in threat_data]

            if known_threats:
                db_client.add_texts(texts=known_threats, metadatas=metadatas)
                print(f"- VectorDB populated with {len(known_threats)} threats.")
            else:
                print("- Warning: Threat file was found but is empty. VectorDB has no threats.")

        print(f"- VectorDB populated with threats.")
    except Exception as e:
        print(f"Warning: Could not populate VectorDB. Error: {e}")

    guard_list = {
        "heuristic": lambda: HeuristicVectorAnalyzer(3, 3),
        "svm": lambda: JailbreakInferenceAPI(cfg.model_dir),
        "cluster": lambda: ClassifierCluster(),
        "shieldgemma": lambda: ShieldGemma2BClassifier(cfg.huggingface_token),
        "vectordb": lambda: VectorDBScanner(db_client, cfg.vectordb.threshold),
        "scanner": lambda: Scanner("rules/yara")
    }

    initialized_guards = {}
    for name in cfg.active_defences:
        if name in guard_list:
            initialized_guards[name.capitalize()] = guard_list[name]()
            print(f"- Guard '{name.capitalize()}' initialized.")
        else:
            print(f"Warning: Guard '{name}' not recognized.")

    return initialized_guards


def initialize_all_guards(cfg: DictConfig) -> dict[str, AbstractDefence]:
    db_client = VectorDB(
        model=cfg.vectordb.model,
        collection=cfg.vectordb.collection,
        db_dir=cfg.vectordb.db_dir,
        n_results=cfg.vectordb.n_results,
    )

    try:
        with open(cfg.threat_file, "r") as f:
            threat_data = json.load(f)
            known_threats = [entry['prompt'] for entry in threat_data]
            metadatas = [{'category': entry.get('category', 'unknown')} for entry in threat_data]

            if known_threats:
                db_client.add_texts(texts=known_threats, metadatas=metadatas)
                print(f"- VectorDB populated with {len(known_threats)} threats.")
            else:
                print("- Warning: Threat file was found but is empty. VectorDB has no threats.")

        print(f"- VectorDB populated with threats.")
    except Exception as e:
        print(f"Warning: Could not populate VectorDB. Error: {e}")

    guard_list = {
        "heuristic": lambda: HeuristicVectorAnalyzer(3, 3),
        "svm": lambda: JailbreakInferenceAPI(cfg.model_dir),
        "cluster": lambda: ClassifierCluster(),
        "shieldgemma": lambda: ShieldGemma2BClassifier(cfg.huggingface_token),
        "vectordb": lambda: VectorDBScanner(db_client, cfg.vectordb.threshold),
        "scanner": lambda: Scanner("rules/yara")
    }

    initialized_guards = {name: constructor() for name, constructor in guard_list.items()}

    print(f"- All guards initialized: {list(initialized_guards.keys())}")
    return initialized_guards
