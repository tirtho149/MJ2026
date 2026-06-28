"""Bangla email NLP — Nova-native refactor.

Two Colab notebooks were merged into one importable package:

  * ``Bangla_Email_Synthetic_Generation.ipynb``  -> :mod:`bangla_email.generate`
  * ``Bangla_Email_NLP_Qwen2.5_3B.ipynb``        -> :mod:`bangla_email.preprocess`
                                                    + :mod:`bangla_email.classify`

All Colab-only bits (``!pip install``, ``google.colab.files`` up/downloads,
interactive restarts) are gone.  The GPU work runs through a single vLLM engine
on one Nova GPU (TP=1); the OOM ``compare_models`` step that produced the
``EngineDeadError`` in the screenshots is fixed in :mod:`bangla_email.generate`.
"""

__all__ = ["config", "preprocess", "generate", "classify", "seed_data"]
