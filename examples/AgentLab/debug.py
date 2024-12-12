import numpy as np


def uct(q, w, num_node_visits, num_parent_visits) -> float:
    return q + w * np.sqrt(np.log(num_parent_visits) / max(1, num_node_visits))


uct_score = uct(q=0.8, w=1.0, num_node_visits=5, num_parent_visits=6)
uct_score2 = uct(q=0.2, w=1.0, num_node_visits=0, num_parent_visits=6)
print(f"uct_score: {uct_score:.2f}, uct_score2: {uct_score2:.2f}")
