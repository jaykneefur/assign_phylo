"""Helper functions for HW3"""
import numpy as np
from copy import deepcopy
from matplotlib.axes import Axes


class Node:
    def __init__(
        self,
        name: str,
        left: "Node",
        left_distance: float,
        right: "Node",
        right_distance: float,
        confidence: float = None,
    ):
        """A node in a binary tree produced by neighbor joining algorithm.

        Parameters
        ----------
        name: str
            Name of the node.
        left: Node
            Left child.
        left_distance: float
            The distance to the left child.
        right: Node
            Right child.
        right_distance: float
            The distance to the right child.
        confidence: float
            The confidence level of the split determined by the bootstrap method.
            Only used if you implement Bonus Problem 1.

        Notes
        -----
        The current public API needs to remain as it is, i.e., don't change the
        names of the properties in the template, as the tests expect this kind
        of structure. However, feel free to add any methods/properties/attributes
        that you might need in your tree construction.

        """
        self.name = name
        self.left = left
        self.left_distance = left_distance
        self.right = right
        self.right_distance = right_distance
        self.confidence = confidence


def neighbor_joining(distances: np.ndarray, labels: list) -> Node:
    """The Neighbor-Joining algorithm.

    For the same results as in the later test dendrograms;
    add new nodes to the end of the list/matrix and
    in case of ties, use np.argmin to choose the joining pair.

    Parameters
    ----------
    distances: np.ndarray
        A 2d square, symmetric distance matrix containing distances between
        data points. The diagonal entries should always be zero; d(x, x) = 0.
    labels: list
        A list of labels corresponding to entries in the distances matrix.
        Use them to set names of nodes.

    Returns
    -------
    Node
        A root node of the neighbor joining tree.

    """
    
    #self.tree = Node()

    D = np.array(distances, dtype=float)
    
    # Every label starts as a leaf: a Node with no children
    nodes = [Node(label, None, 0, None, 0) for label in labels]
    counter = 0
    
    while len(nodes) > 2:
        n = len(nodes)
    
        # Step 1: Q matrix. r[i] = total distance from i to all others
        r = D.sum(axis=1)
        Q = (n - 2) * D - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)  # never join a node with itself
    
        # Step 2: pair with the smallest Q (np.argmin picks the first on ties, as the docstring asks)
        i, j = np.unravel_index(np.argmin(Q), Q.shape)
        if i > j:
            i, j = j, i
    
        # Step 3: branch lengths from i and j to their new parent
        dist_i = D[i, j] / 2 + (r[i] - r[j]) / (2 * (n - 2))
        dist_j = D[i, j] - dist_i
    
        counter += 1
        new_node = Node(f"node_{counter}", nodes[i], dist_i, nodes[j], dist_j)
    
        # Step 4: distances from the new node to every remaining node
        new_row = (D[i, :] + D[j, :] - D[i, j]) / 2
        keep = [k for k in range(n) if k not in (i, j)]
        new_row = new_row[keep]
    
        # Remove i and j, then add the new node at the END of the matrix and list
        D = D[np.ix_(keep, keep)]
        D = np.vstack([np.column_stack([D, new_row]), np.append(new_row, 0.0)])
        nodes = [nodes[k] for k in keep] + [new_node]
    
    # Step 5: join the last two under the root, half the distance each
    half = D[0, 1] / 2
    return Node("ROOT", nodes[0], half, nodes[1], half)


def plot_nj_tree(tree: Node, ax: Axes = None, **kwargs) -> None:
    """A function for plotting neighbor joining phylogeny dendrogram.

    Parameters
    ----------
    tree: Node
        The root of the phylogenetic tree produced by `neighbor_joining(...)`.
    ax: Axes
        A matplotlib Axes object which should be used for plotting.
    kwargs
        Feel free to replace/use these with any additional arguments you need.
        But make sure your function can work without them, for testing purposes.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>>
    >>> tree = neighbor_joining(distances)
    >>> fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    >>> plot_nj_tree(tree=tree, ax=ax)
    >>> fig.savefig("example.png")

    """

    import matplotlib.pyplot as plt
    if ax is None:
        ax = plt.gca()

    # Optional extras (the function still works without them; used for Problem 3 colours)
    leaf_colors = kwargs.get("leaf_colors", {})  # e.g. {"A": "tab:red"}
    fontsize = kwargs.get("fontsize", 12)
    line_color = kwargs.get("line_color", "k")

    next_leaf_y = [0]  # leaves get y = 0, 1, 2, ... from the bottom up

    def draw(node, x):
        """Draw the subtree under `node`, whose x position is `x`. Returns its y."""
        # Leaf: place it at the next free y and label it
        if node.left is None and node.right is None:
            y = next_leaf_y[0]
            next_leaf_y[0] += 1
            ax.text(x, y, f"  {node.name}", fontsize=fontsize, verticalalignment="center",
                    color=leaf_colors.get(node.name, "k"))
            return y

        # Internal node: draw both children first, further right by their branch lengths
        x_left = x + node.left_distance
        x_right = x + node.right_distance
        y_left = draw(node.left, x_left)
        y_right = draw(node.right, x_right)

        # Horizontal branches: from this node out to each child, at the child's height
        ax.plot([x, x_left], [y_left, y_left], color=line_color, lw=1.5)
        ax.plot([x, x_right], [y_right, y_right], color=line_color, lw=1.5)
        # Vertical line joining the two children at this node's position
        ax.plot([x, x], [y_left, y_right], color=line_color, lw=1.5)

        # Bootstrap confidence (only shown if the bonus is done)
        if node.confidence is not None:
            ax.text(x, (y_left + y_right) / 2, f"{node.confidence:.2f} ", fontsize=fontsize - 3,
                    ha="right", va="bottom", color="tab:blue")
        return (y_left + y_right) / 2  # this node sits halfway between its children

    root_y = draw(tree, 0)
    ax.plot([-0.2, 0], [root_y, root_y], color=line_color, lw=1.5)  # short stub for the root

    for loc in ["top", "right"]:
        ax.spines[loc].set_visible(False)
    return ax


def _find_a_parent_to_node(tree: Node, node: Node) -> tuple:
    """Utility function for reroot_tree"""
    stack = [tree]

    while len(stack) > 0:

        current_node = stack.pop()
        if node.name == current_node.left.name:
            return current_node, "left"
        elif node.name == current_node.right.name:
            return current_node, "right"

        stack += [
            n for n in [current_node.left, current_node.right] if n.left is not None
        ]

    return None



def _remove_child_from_parent(parent_node: Node, child_location: str) -> None:
    """Utility function for reroot_tree"""
    setattr(parent_node, child_location, None)
    setattr(parent_node, f"{child_location}_distance", 0.0)


def reroot_tree(original_tree: Node, outgroup_node: Node) -> Node:
    """A function to create a new root and invert a tree accordingly.

    This function reroots tree with nodes in original format. If you
    added any other relational parameters to your nodes, these parameters
    will not be inverted! You can modify this implementation or create
    additional functions to fix them.

    Parameters
    ----------
    original_tree: Node
        A root node of the original tree.
    outgroup_node: Node
        A Node to set as an outgroup (already included in a tree).
        Find it by it's name and then use it as parameter.

    Returns
    -------
    Node
        Inverted tree with a new root node.
    """
    tree = deepcopy(original_tree)

    parent, child_loc = _find_a_parent_to_node(tree, outgroup_node)
    distance = getattr(parent, f"{child_loc}_distance")
    _remove_child_from_parent(parent, child_loc)

    new_root = Node("new_root", parent, distance / 2, outgroup_node, distance / 2)
    child = parent

    while tree != child:
        parent, child_loc = _find_a_parent_to_node(tree, child)

        distance = getattr(parent, f"{child_loc}_distance")
        _remove_child_from_parent(parent, child_loc)

        empty_side = "left" if child.left is None else "right"
        setattr(child, f"{empty_side}_distance", distance)
        setattr(child, empty_side, parent)

        if tree.name == parent.name:
            break
        child = parent

    other_child_loc = "right" if child_loc == "left" else "left"
    other_child_distance = getattr(parent, f"{other_child_loc}_distance")

    setattr(child, f"{empty_side}_distance", other_child_distance + distance)
    setattr(child, empty_side, getattr(parent, other_child_loc))

    return new_root


def sort_children_by_leaves(tree: Node) -> None:
    """Sort the children of a tree by their corresponding number of leaves.

    The tree can be changed inplace.

    Parameters
    ----------
    tree: Node
        The root node of the tree.

    """
    def count_leaves(node):
        if node.left is None and node.right is None:
            return 1
        n_left = count_leaves(node.left)
        n_right = count_leaves(node.right)
        if n_left > n_right:  # swap so the child with fewer leaves is on the left
            node.left, node.right = node.right, node.left
            node.left_distance, node.right_distance = node.right_distance, node.left_distance
        return n_left + n_right

    count_leaves(tree)


def plot_nj_tree_radial(tree: Node, ax: Axes = None, **kwargs) -> None:
    """A function for plotting neighbor joining phylogeny dendrogram
    with a radial layout.

    Parameters
    ----------
    tree: Node
        The root of the phylogenetic tree produced by `neighbor_joining(...)`.
    ax: Axes
        A matplotlib Axes object which should be used for plotting.
    kwargs
        Feel free to replace/use these with any additional arguments you need.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>>
    >>> tree = neighbor_joining(distances)
    >>> fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    >>> plot_nj_tree_radial(tree=tree, ax=ax)
    >>> fig.savefig("example_radial.png")

    """
    raise NotImplementedError()

def global_alignment(seq1, seq2, scoring_function):
    n, m = len(seq1), len(seq2)

    # Score table: (n+1) rows, (m+1) columns, all zeros to start
    D = [[0.0] * (m + 1) for _ in range(n + 1)]

    # First column: seq1 letters aligned to gaps
    for i in range(1, n + 1):
        D[i][0] = D[i-1][0] + scoring_function(seq1[i-1], "-")

    # First row: seq2 letters aligned to gaps
    for j in range(1, m + 1):
        D[0][j] = D[0][j-1] + scoring_function("-", seq2[j-1])

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = max(
                D[i-1][j-1] + scoring_function(seq1[i-1], seq2[j-1]),
                D[i-1][j]   + scoring_function(seq1[i-1], "-"),
                D[i][j-1]   + scoring_function("-", seq2[j-1]),
            )

    # Traceback from bottom-right
    i, j = n, m
    aligned1 = ""
    aligned2 = ""

    while i > 0 or j > 0:
        # Diagonal: letter matched/mismatched with letter
        if i > 0 and j > 0 and D[i][j] == D[i-1][j-1] + scoring_function(seq1[i-1], seq2[j-1]):
            aligned1 = seq1[i-1] + aligned1
            aligned2 = seq2[j-1] + aligned2
            i -= 1
            j -= 1

        # Up: seq1 letter aligned to a gap
        elif i > 0 and D[i][j] == D[i-1][j] + scoring_function(seq1[i-1], "-"):
            aligned1 = seq1[i-1] + aligned1
            aligned2 = "-" + aligned2
            i -= 1

        # Left: gap aligned to seq2 letter
        else:
            aligned1 = "-" + aligned1
            aligned2 = seq2[j-1] + aligned2
            j -= 1

    return aligned1, aligned2, float(D[n][m])

from Bio.Align import substitution_matrices

BLOSUM62 = substitution_matrices.load("BLOSUM62")
GAP_PENALTY = -4  # at least as bad as BLOSUM62's worst mismatch

# Convert to a plain dict once, for speed
_blosum_dict = {(a, b): BLOSUM62[a][b] for a in BLOSUM62.alphabet for b in BLOSUM62.alphabet}

def scoring_function_blosum62(aa_i, aa_j):
    if aa_i == "-" or aa_j == "-":
        return GAP_PENALTY
    return _blosum_dict[(aa_i, aa_j)]