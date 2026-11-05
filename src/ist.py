import torch
import random
import numpy as np

from spflow.meta import Scope
from spflow.modules.leaves.normal import Normal
from spflow.modules.products.elementwise_product import ElementwiseProduct
from spflow.modules.sums.repetition_mixing_layer import RepetitionMixingLayer
from spflow.modules.rat.factorize import Factorize
from spflow.modules.ops.split import SplitMode
from spflow.modules.sums.sum import Sum

from scipy.stats import rankdata



class InformedSplitTensorizedSPN:

    def __init__(self, data: torch.Tensor, root_sum_node: int, n_region_nodes: int, num_repetitions: int, depth: int, dependency_test: str = "gtest", ordering: str = "heuristic"):
        self.data = data
        self.root_sum_node = root_sum_node
        self.n_root_nodes = root_sum_node
        self.depth = depth
        self.n_region_nodes = n_region_nodes
        self.num_repetitions = num_repetitions
        self.dependency_test = dependency_test
        self.ordering = ordering
        self.scope = list(range(data.shape[1]))

        # Create one leaf module per input feature.
        self.leaf_modules = [
            Normal(
                scope=Scope([i]),
                out_channels=4 
                num_repetitions=num_repetitions
            )
            for i in self.scope
        ]

        # Determnine what features should be grouped in which order: 
        tree = self.determine_tree_structure()
        self.tree = tree
        # Build a RAT SPN from this: 
        self.root_node = self.create_spn(tree)

    '''
    Deterine the structure of the tree
    '''

    def determine_tree_structure(self):
        '''Retuns the tree structure as a nested dict
        e.g. Scope [1,2,3,4] => [[1,3], [3,4]]
        '''
        return self.determine_split(self.scope, self.depth)

    def determine_split(self, scope, depth):
        '''Recurisvely splits scope into nested format'''
        if depth == 0 or len(scope) <= 1:
            return [scope]

        left_branch, right_branch = self.determine_factorization(
            self.data, scope, depth
        )

        if not left_branch or not right_branch:
            return [scope]

        return [
            self.determine_split(left_branch, depth - 1),
            self.determine_split(right_branch, depth - 1)
        ]

    def determine_factorization(self, data, scope, depth):
        '''Sorts scope into branches based on score'''
        scores = self.determine_score_matrix(scope)

        if self.ordering == "heuristic":
            left_branch, right_branch = (
                self.sort_scopes_equally_based_on_score(scope, scores)
            )
        elif self.ordering == "optimal":
            left_branch, right_branch = (
                self.sort_scopes_equally_based_on_score_opt(scope, scores)
            )
        elif self.ordering == "random":
            left_branch, right_branch = (
                self.sort_scopes_equally_based_on_score_rd(scope, scores)
            )

        return left_branch, right_branch

    '''
    Dependence Score Calculation 
    '''
    def determine_score_matrix(self, scope, data=None):
        """Pairwise dependence scores on `data` (defaults to all training data)."""
        data = self.data if data is None else data

        if self.dependency_test == "rdc":
            M = self._compute_rdc_matrix(data, scope)
            pair_score = lambda i, j: M[i, j]   # positions within scope
        elif self.dependency_test == "gtest":
            pair_score = lambda i, j: self._pairwise_gtest(data, scope[i], scope[j])
        else:
            raise ValueError(f"Unknown dependency_test: {self.dependency_test}")

        scores = {}
        for i in range(len(scope)):
            for j in range(i + 1, len(scope)):
                scores[(scope[i], scope[j])] = pair_score(i, j)
        return scores

    def _compute_rdc_matrix(self, data, scope, k=10, s=1/6, max_rows=5000, seed=0):
        X = data[:, scope].detach().cpu().numpy()
        rng = np.random.default_rng(seed)

        if X.shape[0] > max_rows:
            X = X[rng.choice(X.shape[0], max_rows, replace=False)]

        n, d = X.shape
        Qs = []
        for i in range(d):
            copula = np.column_stack([rankdata(X[:, i]) / n, np.ones(n)])
            feats = np.sin(copula @ (rng.normal(size=(2, k)) * s / 2))
            feats -= feats.mean(axis=0)
            q, _ = np.linalg.qr(feats)
            Qs.append(q)

        M = np.zeros((d, d))
        for i in range(d):
            for j in range(i + 1, d):
                M[i, j] = M[j, i] = np.linalg.svd(
                    Qs[i].T @ Qs[j], compute_uv=False
                )[0]
        return M  # indexed by position in `scope`

    def _pairwise_gtest(self, data, index_i, index_j):
        """G-test after discretizing each feature at its median."""

        slice_i = data[:, index_i].detach().cpu().numpy()
        slice_j = data[:, index_j].detach().cpu().numpy()

        median_i = np.median(slice_i)
        median_j = np.median(slice_j)

        slice_i = np.where(slice_i > median_i, 1, 0)
        slice_j = np.where(slice_j > median_j, 1, 0)

        unique_i = np.unique(slice_i)
        unique_j = np.unique(slice_j)

        score = 0.0
        n = slice_j.size

        for i in unique_i:
            for j in unique_j:
                i_in_slice_i = (slice_i == i).astype(int)
                j_in_slice_j = (slice_j == j).astype(int)

                marg_freq_i = np.sum(i_in_slice_i)
                marg_freq_j = np.sum(j_in_slice_j)

                expected_count = (marg_freq_i * marg_freq_j) / n
                observed = np.sum(i_in_slice_i * j_in_slice_j)

                if expected_count > 0 and observed > 0:
                    score += float(
                        observed * np.log(observed / expected_count)
                    )

        return score

    '''
    Sorting based on score 
    '''
    def sort_scopes_equally_based_on_score_rd(self, scope, scores=None):
      # Convert scope to a list so we can shuffle it randomly
      elements = list(scope)
      random.shuffle(elements)

      # Split down the middle to guarantee max size difference of 1
      mid = len(elements) // 2
      left_branch = elements[:mid]
      right_branch = elements[mid:]

      return left_branch, right_branch

    def sort_scopes_equally_based_on_score(self, scope, scores):
        left_branch = []
        right_branch = []

        even_split = len(scope) // 2
        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)

        for (key_a, key_b), value in sorted_scores:

            if key_a in left_branch:
                target = left_branch
                other = right_branch
                candidate = key_b

            elif key_a in right_branch:
                target = right_branch
                other = left_branch
                candidate = key_b

            elif key_b in left_branch:
                target = left_branch
                other = right_branch
                candidate = key_a

            elif key_b in right_branch:
                target = right_branch
                other = left_branch
                candidate = key_a

            else:
                # Start a new pair in the currently smaller branch.
                target = (
                    left_branch
                    if len(left_branch) <= len(right_branch)
                    else right_branch
                )
                other = right_branch if target is left_branch else left_branch

                for candidate in (key_a, key_b):
                    if candidate not in left_branch and candidate not in right_branch:
                        target.append(candidate)

                continue

            if (
                candidate not in left_branch
                and candidate not in right_branch
                and len(target) < even_split
            ):
                target.append(candidate)

        # Assign any features not yet assigned.
        for feature in scope:
            if feature not in left_branch and feature not in right_branch:
                target = (
                    left_branch
                    if len(left_branch) <= len(right_branch)
                    else right_branch
                )
                target.append(feature)

        # Ensure the two branches are balanced.
        while len(left_branch) > len(right_branch) + 1:
            right_branch.append(left_branch.pop())

        while len(right_branch) > len(left_branch) + 1:
            left_branch.append(right_branch.pop())

        return left_branch, right_branch

    def sort_scopes_equally_based_on_score_opt(self, scope, scores):
        n = len(scope)
        if n < 2:
            return list(scope), []

        # Symmetric weight matrix over positions in `scope`.
        pos = {f: i for i, f in enumerate(scope)}
        W = np.zeros((n, n))
        for (a, b), w in scores.items():
            W[pos[a], pos[b]] = W[pos[b], pos[a]] = w

        n_left = (n + 1) // 2          # sizes differ by at most 1

        # 1) Spectral initialization: median split of the Fiedler vector.
        laplacian = np.diag(W.sum(axis=1)) - W
        _, vecs = np.linalg.eigh(laplacian)
        order = np.argsort(vecs[:, 1], kind="stable")
        in_left = np.zeros(n, dtype=bool)
        in_left[order[:n_left]] = True

        # 2) Greedy swap refinement: each swap strictly lowers the cut weight.
        for _ in range(n * n):
            to_left = W @ in_left.astype(float)
            to_right = W.sum(axis=1) - to_left

            # D[v] = cost of v's cross edges minus its internal edges.
            D = np.where(in_left, to_right - to_left, to_left - to_right)

            L = np.flatnonzero(in_left)
            R = np.flatnonzero(~in_left)

            gain = D[L][:, None] + D[R][None, :] - 2 * W[np.ix_(L, R)]
            k = np.unravel_index(np.argmax(gain), gain.shape)

            if gain[k] <= 1e-12:
                break

            in_left[L[k[0]]] = False
            in_left[R[k[1]]] = True

        left_branch = [scope[i] for i in range(n) if in_left[i]]
        right_branch = [scope[i] for i in range(n) if not in_left[i]]
        return left_branch, right_branch


    '''
    Build SPN based on structure 
    '''

    def make_fixed_factorize(self, groups_per_rep):

        class FixedFactorize(Factorize):

            def _factorize(self, depth, num_repetitions):
                query = list(self.inputs[0].scope.query)
                position = {
                    feature: i for i, feature in enumerate(query)
                }

                num_groups = len(groups_per_rep[0])

                if len(groups_per_rep) != num_repetitions:
                    raise ValueError(
                        f"Expected {num_repetitions} repetitions, "
                        f"got {len(groups_per_rep)}."
                    )

                indices = torch.zeros(
                    len(query),
                    num_groups,
                    num_repetitions
                )

                for r, groups in enumerate(groups_per_rep):

                    if len(groups) != num_groups:
                        raise ValueError(
                            "All repetitions must have the same number of groups."
                        )

                    assigned = []

                    for group_idx, group in enumerate(groups):
                        for feature in group:

                            if feature not in position:
                                raise ValueError(
                                    f"Unknown feature: {feature}"
                                )

                            indices[
                                position[feature],
                                group_idx,
                                r
                            ] = 1

                            assigned.append(feature)

                    if len(assigned) != len(set(assigned)):
                        raise ValueError(
                            "A feature appears in more than one group."
                        )

                    if set(assigned) != set(query):
                        raise ValueError(
                            "Groups must cover every input feature exactly once."
                        )

                return indices

        return FixedFactorize

    def get_leaf_groups(self, tree):
        """Extract bottom-level feature groups in tree order."""

        if isinstance(tree, int):
            return [[tree]]

        if not isinstance(tree, (list, tuple)):
            raise TypeError(f"Unexpected tree node: {type(tree)}")

        # A list of feature IDs is one group, not several groups.
        if all(isinstance(x, int) for x in tree):
            return [list(tree)]

        groups = []

        for child in tree:
            groups.extend(self.get_leaf_groups(child))

        return groups

    def create_spn(self, tree):

        leaf_groups = self.get_leaf_groups(tree)

        groups_per_rep = [
            leaf_groups for _ in range(self.num_repetitions)
        ]

        FixedFactorize = self.make_fixed_factorize(groups_per_rep)

        fac_layer = FixedFactorize(
            inputs=self.leaf_modules,
            depth=self.depth,
            num_repetitions=self.num_repetitions
        )

        split_mode = SplitMode("interleaved")
        root = None

        for i in range(self.depth):

            current_input = fac_layer if i == 0 else root

            out_prod = ElementwiseProduct(
                inputs=split_mode.create(current_input)
            )

            if i == self.depth - 1:
                out_channels = self.n_root_nodes
            else:
                out_channels = self.n_region_nodes

            root = Sum(
                inputs=out_prod,
                out_channels=out_channels,
                num_repetitions=self.num_repetitions
            )

        root = RepetitionMixingLayer(
            inputs=root,
            out_channels=self.n_root_nodes,
            num_repetitions=self.num_repetitions
        )

        if self.n_root_nodes > 1:
            root = Sum(
                inputs=root,
                out_channels=1,
                num_repetitions=1
            )

        return root

    def get_tree(self): 
        return self.tree
    