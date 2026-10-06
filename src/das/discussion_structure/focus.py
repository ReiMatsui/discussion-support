"""Bayesian belief filter with tree-distance transition priors and hysteresis."""

from .config import Config
from .models import Distribution, Issue, Tree, Turn


class Focus:
    def __init__(self, config: Config):
        self.config = config
        self.belief = {"root": 1.0}
        self.candidate = None
        self.since = 0
        self.count = 0

    def update(self, tree: Tree, observation: Distribution, turn: Turn, shift: bool) -> str | None:
        ids = [n.id for n in tree.nodes.values() if isinstance(n, Issue)]
        obs = dict.fromkeys(ids, 0.0)
        for target, p in observation.probabilities.items():
            if target in tree.nodes:
                obs[tree.issue_id(target)] += p
        if not sum(obs.values()):
            self.candidate, self.count = None, 0
            return None
        stay = self.config.shift_stay_prior if shift else self.config.stay_prior
        prior = dict.fromkeys(ids, 0.0)
        for source, belief in self.belief.items():
            weights = {}
            source_path = [n.id for n in tree.path(source)]
            for dest in ids:
                dest_path = [n.id for n in tree.path(dest)]
                common = len(set(source_path) & set(dest_path))
                distance = len(source_path) + len(dest_path) - 2 * common
                if dest != source:
                    weights[dest] = 1 / (1 + distance) ** 2
            total = sum(weights.values())
            prior[source] += belief * (stay if total else 1)
            for dest, weight in weights.items():
                prior[dest] += belief * (1 - stay) * weight / total
        # Longer substantive utterances carry more evidence (bounded at 2).
        weight = min(2, max(0.5, len(turn.text) / self.config.observation_chars))
        post = {i: max(prior[i], 1e-12) * max(obs[i], 1e-6) ** weight for i in ids}
        total = sum(post.values())
        self.belief = {i: p / total for i, p in post.items()}
        candidate = max(self.belief, key=self.belief.get)
        if candidate == tree.focus_id or self.belief[candidate] <= self.config.focus_threshold:
            self.candidate, self.count = None, 0
            return None
        if self.candidate != candidate:
            self.candidate, self.since, self.count = candidate, turn.end_ms, 0
        self.count += 1
        if (
            self.count >= self.config.focus_turns
            and turn.end_ms - self.since >= self.config.focus_ms
        ):
            old = tree.focus_id
            tree.focus_id = candidate
            self.candidate, self.count = None, 0
            return old
        return None
