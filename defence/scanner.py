import os
import yara
from typing import override
from defence.abstract_defence import AbstractDefence
from defence.ds.analysis_result import AnalysisResult


class Scanner(AbstractDefence):
    def __init__(self, rules_dir: str):
        if not os.path.exists(rules_dir) or not os.path.isdir(rules_dir):
            raise FileNotFoundError(f"YARA rules directory '{rules_dir}' not found or invalid.")

        self.rules_dir = rules_dir
        self.compiled_rules = None
        self._load_rules()

    def _load_rules(self):
        yara_paths = {}
        for file in os.listdir(self.rules_dir):
            if file.lower().endswith(('.yar', '.yara')):
                yara_paths[file] = os.path.join(self.rules_dir, file)

        if not yara_paths:
            raise FileNotFoundError(f"No YARA rule files found in '{self.rules_dir}'.")

        try:
            self.compiled_rules = yara.compile(filepaths=yara_paths)
        except Exception as e:
            raise RuntimeError(f"Failed to compile YARA rules: {e}")

    @override
    def analyse(self, query: str) -> AnalysisResult:
        if not query.strip():
            return AnalysisResult("YARA scanner", True)

        try:
            matches = self.compiled_rules.match(data=query)
        except Exception as e:
            return AnalysisResult("YARA scanner found no matches", True)

        if matches:
            matched_rules = ', '.join([match.rule for match in matches])
            return AnalysisResult(f"YARA scanner found matched rules: {matched_rules}", False)

        return AnalysisResult("YARA scanner found no matches", True)
