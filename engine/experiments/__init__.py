"""Edge Tribunal: preregistered falsification and research-promotion governance.

An isolated experiment court for frozen quantitative hypotheses. It decides
whether a hypothesis survived a fair, preregistered attempt to kill it — it
never decides that anything "makes money" and never authorizes trading.
Under the repository's BID-only data contract the highest historical verdict
is FORWARD_TEST_ELIGIBLE.

Entry point: ``python -m engine.experiments.edge_tribunal``.
"""

from engine.experiments.errors import TribunalError

__all__ = ["TribunalError"]
