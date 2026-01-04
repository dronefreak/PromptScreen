import json
from pathlib import Path
from omegaconf import DictConfig

from defence.abstract_defence import AbstractDefence
from defence.classifier_cluster import ClassifierCluster
from defence.heuristic_channel import HeuristicVectorAnalyzer
from defence.linear_svm import JailbreakInferenceAPI
from defence.shieldgemma import ShieldGemma2BClassifier
from defence.vectordb import VectorDBScanner, VectorDB
from defence.train.train_classifier import JailbreakClassifier
from defence.scanner import Scanner


def initialize_guards(cfg: DictConfig) -> dict[str, AbstractDefence]:
    db_client = None

    if "vectordb" in cfg.active_defences:
        try:
            db_client = VectorDB(
                model=cfg.vectordb.model,
                collection=cfg.vectordb.collection,
                db_dir=cfg.vectordb.db_dir,
                n_results=cfg.vectordb.n_results,
            )

            with open(cfg.threat_file, "r") as f:
                threat_data = json.load(f)
                known_threats = [entry['prompt'] for entry in threat_data]
                metadatas = [{'category': entry.get('classification', 'unknown')} for entry in threat_data]

                if known_threats:
                    db_client.add_texts(texts=known_threats, metadatas=metadatas)
                    print(f"- VectorDB populated with {len(known_threats)} threats.")
                else:
                    print("- Warning: Threat file was found but is empty. VectorDB has no threats.")

        except Exception as e:
            print(f"Warning: Could not populate VectorDB. Error: {e}")

    if "svm" in cfg.active_defences:
        variant = cfg.get('variant', 'word_ngram_1_2')
        model_dir_path = Path(cfg.model_dir)
        variant_model_exists = (model_dir_path / f"linear_svm_model_{variant}.joblib").exists()
        variant_features_exist = (model_dir_path / f"feature_union_{variant}.joblib").exists()
        model_files_exist = variant_model_exists and variant_features_exist
        should_train = cfg.train or not model_files_exist

        if should_train:
            if not model_files_exist:
                print(f"Model artifacts for variant '{variant}' not found. Forcing training...")
            else:
                print(f"Starting training for variant '{variant}' as requested by 'train=true' configuration.")            
            try:
                trainer = JailbreakClassifier(
                    json_file_path=cfg.train_file, 
                    model_output_dir=cfg.model_dir,
                    variant=variant  # Pass variant parameter
                )
                trainer.train()
                print("\nTraining finished successfully.")
            except FileNotFoundError:
                print(f"Error: Training file '{cfg.train_file}' not found.")
            except Exception as e:
                print(f"An error occurred during training: {e}")
        else:
             print(f"Skipping training: Model artifacts for variant '{variant}' already exist in '{cfg.model_dir}' and 'train=false'.")

        print("-" * 20)


    guard_list = {
        "heuristic": lambda: HeuristicVectorAnalyzer(3, 3),
        "svm": lambda: JailbreakInferenceAPI(cfg.model_dir, variant=variant),
        "cluster": lambda: ClassifierCluster(),
        "shieldgemma": lambda: ShieldGemma2BClassifier(cfg.huggingface_token),
        "vectordb": lambda: VectorDBScanner(db_client, cfg.vectordb.threshold),
        "scanner": lambda: Scanner("rules/yara")
    }

    initialized_guards = {}
    for name in cfg.active_defences:
        if name in guard_list:
            if name == "vectordb" and db_client is None:
                print(f"Warning: Skipping 'Vectordb' guard because initialization failed or was skipped.")
                continue

            initialized_guards[name] = guard_list[name]()
            if name == "svm":
                print(f"- Guard '{name}' initialized with variant '{variant}'.")
            else:
                print(f"- Guard '{name}' initialized.")
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
            metadatas = [{'category': entry.get('classification', 'unknown')} for entry in threat_data]

            if known_threats:
                db_client.add_texts(texts=known_threats, metadatas=metadatas)
                print(f"- VectorDB populated with {len(known_threats)} threats.")
            else:
                print("- Warning: Threat file was found but is empty. VectorDB has no threats.")

    except Exception as e:
        print(f"Warning: Could not populate VectorDB. Error: {e}")

    variant = cfg.get('variant', 'word_ngram_1_2')

    model_dir_path = Path(cfg.model_dir)

    variant_model_exists = (model_dir_path / f"linear_svm_model_{variant}.joblib").exists()
    variant_features_exist = (model_dir_path / f"feature_union_{variant}.joblib").exists()
    model_files_exist = variant_model_exists and variant_features_exist
    should_train = cfg.train or not model_files_exist

    if should_train:
        if not model_files_exist:
            print(f"Model artifacts for variant '{variant}' not found. Forcing training...")
        else:
            print(f"Starting training for variant '{variant}' as requested by 'train=true' configuration.")
        
        try: 
            trainer = JailbreakClassifier(
                json_file_path=cfg.train_file, 
                model_output_dir=cfg.model_dir,
                variant=variant
            )
            trainer.train()
            print(f"\nTraining finished successfully for variant '{variant}'.")
        except FileNotFoundError:
            print(f"Error: Training file '{cfg.train_file}' not found.")
        except Exception as e:
            print(f"An error occurred during training: {e}")
    else:
        print(f"Skipping training: Model artifacts for variant '{variant}' already exist in '{cfg.model_dir}' and 'train=false'.")
    print("-" * 20)

    guard_list = {
        "heuristic": lambda: HeuristicVectorAnalyzer(3, 3),
        "svm": lambda: JailbreakInferenceAPI(cfg.model_dir, variant=variant),
        "cluster": lambda: ClassifierCluster(),
        "shieldgemma": lambda: ShieldGemma2BClassifier(cfg.huggingface_token),
        "vectordb": lambda: VectorDBScanner(db_client, cfg.vectordb.threshold),
        "scanner": lambda: Scanner("rules/yara")
    }

    initialized_guards = {name: constructor() for name, constructor in guard_list.items()}

    print(f"- All guards initialized: {list(initialized_guards.keys())}")
    print(f"- SVM using variant: {variant}")
    return initialized_guards
