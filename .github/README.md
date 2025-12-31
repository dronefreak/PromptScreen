# PromptScreen (now on PyPI!)

**A prompt injection and jailbreak detection system for LLMs**

[![📦 PyPI](https://img.shields.io/pypi/v/promptscreen?logo=pypi&logoColor=white&color=3776AB)](https://pypi.org/project/promptscreen/)
[![🐍 Python Versions](https://img.shields.io/badge/Python-3.9%2B+-3776AB?logo=python&logoColor=white&style=for-the-badge)](https://pypi.org/project/promptscreen/)
[![🧪 Tests](https://img.shields.io/badge/GitHub-Tests-2088FF?logo=github&logoColor=white&style=for-the-badge)](https://github.com/cross-codes/Prompt-Injection-And-Jailbreaking/actions/workflows/tests.yml)
[![📊 Codecov](https://img.shields.io/badge/Codecov-Coverage-FF4D00?logo=codecov&logoColor=white&style=for-the-badge)](https://codecov.io/gh/cross-codes/Prompt-Injection-And-Jailbreaking)
[![⚖️ License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green?logo=open-source-initiative&logoColor=white&style=for-the-badge)](https://opensource.org/licenses/Apache-2.0)

PromptScreen is an open-source library that provides multiple defense layers against prompt injection attacks and jailbreak attempts in LLM applications. Designed for production use, it offers plug-and-play guards that can be integrated into any LLM pipeline.

---

## Quick Start

We're excited to announce that PromptScreen is now available via pip:

```bash
pip install promptscreen
```

```python
from promptscreen import HeuristicVectorAnalyzer

guard = HeuristicVectorAnalyzer(threshold=2, pm_shot_lim=3)
result = guard.analyse("Your prompt here")

if result.get_verdict():
    print("✓ Safe prompt")
else:
    print(f"✗ Blocked: {result.get_type()}")
```

## Installation Options

```bash
# Core package (fast guards only)
pip install promptscreen

# With ML guards (ShieldGemma, ClassifierCluster)
pip install promptscreen[ml]

# With vector database guard
pip install promptscreen[vectordb]

# Everything
pip install promptscreen[all]
```

## Available Guards

- **HeuristicVectorAnalyzer** - Fast pattern-based detection
- **Scanner (YARA)** - Bundled YARA rules
- **InjectionScanner** - Command injection detection
- **JailbreakInferenceAPI (SVM)** - ML classifier
- **VectorDBScanner** - Similarity search (optional)
- **ClassifierCluster** - Dual ML models (optional)
- **ShieldGemma** - Google's safety model (optional)

## Documentation

- [README](https://github.com/dronefreak/PromptScreen#readme)
- [Examples](https://github.com/dronefreak/PromptScreen/tree/main/examples)
- [Security Policy](https://github.com/dronefreak/PromptScreen/blob/main/SECURITY.md)
- [Contributing Guide](https://github.com/dronefreak/PromptScreen/blob/main/CONTRIBUTING.md)

## Links

- **PyPI:** https://pypi.org/project/promptscreen/
- **GitHub:** https://github.com/dronefreak/PromptScreen
- **Issues:** https://github.com/dronefreak/PromptScreen/issues

---

**As always Hare Krishna and happy coding!**
