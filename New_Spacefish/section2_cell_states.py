# =============================================================================
# SECTION 2 - Which gene states define distinct cell states?
#
# Copy PART 1 into one notebook cell (defines every function, runs nothing) and
# PART 2 into the next cell (runs all four approaches SPLIT BY CELL TYPE and prints
# a readable per-cluster summary for each).
#
# Expects these to already exist from earlier in the notebook:
#   df         - recurrent cells (columns: cell_type, binary_state, ...)
#   GENES      - list of the real gene names, in column order
#   output_dir - Path where the summary CSV is written
# =============================================================================


# =============================================================================
# PART 1 - FUNCTIONS (define once, nothing runs here)
# =============================================================================

# ---- Shared helpers + readable cluster summariser ---------------------------

def build_state_matrix(cells, genes):
    """Collapse cells down to their unique gene states.

    Returns:
        state_matrix    - one row per distinct on/off pattern, one column per gene
                          (0/1), ordered from most to least common.
        cells_per_state - how many cells carry each state (same row order).
    """
    cells_per_state = cells.groupby("binary_state").size().sort_values(ascending=False)
    state_matrix = pd.DataFrame(
        [[int(bit) for bit in state] for state in cells_per_state.index],
        index=cells_per_state.index, columns=genes,
    )
    return state_matrix, cells_per_state


def binary_state_to_code(binary_state):
    """'100100...' -> '1_4' (1-indexed positions of the ON genes; '0' if none on)."""
    genes_on = [str(position + 1) for position, bit in enumerate(binary_state) if bit == "1"]
    return "_".join(genes_on) if genes_on else "0"


def state_code_to_gene_names(state_code, genes):
    """'1_4_7' -> 'CDH5, KDR, VWF' using the 1-indexed gene order ('0' -> 'none')."""
    if state_code == "0":
        return "none"
    return ", ".join(genes[int(position) - 1] for position in state_code.split("_"))


def describe_clusters(cells, cell_labels, genes, method_name, cell_type, top_states=3):
    """Print a plain-English summary of one method's clusters for one cell type.

    For every cluster it lists the gene states that dominate it, e.g.
    "cluster 1 (120 cells, 34%): likely states include 60% 1_4 (CDH5, KDR); ...".
    Cells with label -1 (unassigned) are ignored. Returns the same information as a
    tidy DataFrame so it can be collected and saved.
    """
    labels = pd.Series(cell_labels, index=cells.index)
    assigned = labels[labels >= 0]
    cluster_sizes = assigned.value_counts()
    n_cells = len(assigned)
    print(f"{len(cluster_sizes)} clusters identified for {cell_type} cells "
          f"({method_name}, {n_cells:,} cells):")

    summary_rows = []
    for cluster_id in cluster_sizes.index:
        member_index = assigned[assigned == cluster_id].index
        n = len(member_index)
        state_counts = cells.loc[member_index, "binary_state"].value_counts()
        described = [
            f"{100 * count / n:.0f}% {binary_state_to_code(state)} "
            f"({state_code_to_gene_names(binary_state_to_code(state), genes)})"
            for state, count in state_counts.head(top_states).items()
        ]
        main_states = "; ".join(described)
        print(f"  cluster {cluster_id} ({n:,} cells, {100 * n / n_cells:.0f}%): "
              f"likely states include {main_states}")
        summary_rows.append({
            "cell_type": cell_type,
            "method": method_name,
            "cluster": cluster_id,
            "n_cells": n,
            "pct_of_cell_type": 100 * n / n_cells,
            "main_states": main_states,
        })
    print()
    return pd.DataFrame(summary_rows)


# ---- Approach 1: hierarchical clustering of gene states ----------------------

def hierarchical_cell_labels(cells, genes, metric="jaccard", max_clusters=8):
    """Cluster the unique gene states, then hand each cell its state's cluster.

    Biological question: do the on/off gene patterns fall into a few recurring
    programmes ("cell states") within this lineage?

    The `metric` decides what "similar" means:
        "jaccard" - similar when states share the same ON genes (OFF genes ignored).
        "hamming" - similar when states agree on ON and OFF genes (silent genes count).
    States are joined with average linkage and the tree is cut so it yields at most
    `max_clusters` groups (keeps the summary readable).

    Returns:
        cell_labels - one cluster id per cell (all -1 if too few states to cluster).
        plot_bundle - (state_matrix, linkage_tree, cluster_ids) for the clustermap,
                      or None when clustering was skipped.
    """
    state_matrix, _ = build_state_matrix(cells, genes)
    if len(state_matrix) < 3:
        return np.full(len(cells), -1), None
    linkage_tree = linkage(pdist(state_matrix.values, metric=metric), method="average")
    n_clusters = min(max_clusters, len(state_matrix))
    cluster_ids = fcluster(linkage_tree, t=n_clusters, criterion="maxclust")
    state_to_cluster = pd.Series(cluster_ids, index=state_matrix.index)
    cell_labels = cells["binary_state"].map(state_to_cluster).to_numpy()
    return cell_labels, (state_matrix, linkage_tree, cluster_ids)


def plot_state_clustermap(state_matrix, linkage_tree, cluster_ids, cell_type, n_cells):
    """Clustermap of the gene states for one cell type, rows coloured by cluster."""
    cluster_colors = [sns.color_palette("tab20", cluster_ids.max())[c - 1] for c in cluster_ids]
    grid = sns.clustermap(
        state_matrix, row_linkage=linkage_tree, col_cluster=False, cmap="Greys",
        row_colors=cluster_colors, yticklabels=False, cbar_pos=None,
        figsize=(9, min(14, 0.18 * len(state_matrix) + 3)))
    grid.ax_heatmap.set_xlabel("Gene")
    grid.figure.suptitle(
        f"{cell_type}: {len(state_matrix)} gene states ({n_cells:,} cells) "
        f"-> {cluster_ids.max()} clusters", y=1.02)
    plt.show()


# ---- Approach 2: Bernoulli mixture (probabilistic) --------------------------

def fit_bernoulli_mixture_once(gene_matrix, n_states, max_iter=300, tol=1e-6, rng=None):
    """One EM run of a Bernoulli mixture.

    Returns (weights, on_probabilities, log_likelihood, responsibilities).
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    n_cells, n_genes = gene_matrix.shape
    on_probabilities = rng.uniform(0.1, 0.9, size=(n_states, n_genes))
    weights = np.full(n_states, 1.0 / n_states)
    log_likelihood_prev = -np.inf
    responsibilities = np.full((n_cells, n_states), 1.0 / n_states)
    for _ in range(max_iter):
        log_prob = (gene_matrix @ np.log(on_probabilities.T)
                    + (1 - gene_matrix) @ np.log(1 - on_probabilities.T)) + np.log(weights)
        row_max = log_prob.max(axis=1, keepdims=True)
        log_norm = row_max.ravel() + np.log(np.exp(log_prob - row_max).sum(axis=1))
        log_likelihood = log_norm.sum()
        responsibilities = np.exp(log_prob - log_norm[:, None])
        cells_per_state = responsibilities.sum(axis=0) + 1e-9
        weights = cells_per_state / n_cells
        on_probabilities = np.clip((responsibilities.T @ gene_matrix) / cells_per_state[:, None],
                                   1e-3, 1 - 1e-3)
        if abs(log_likelihood - log_likelihood_prev) < tol * (abs(log_likelihood_prev) + 1e-9):
            break
        log_likelihood_prev = log_likelihood
    return weights, on_probabilities, log_likelihood, responsibilities


def fit_bernoulli_mixture(gene_matrix, n_states, n_restarts=8, seed=0):
    """Sort cells into `n_states` hidden states using a Bernoulli mixture.

    The idea: imagine a few hidden cell states. Each state has, for every gene, a
    chance that the gene is switched on, and each cell is a slightly noisy copy of one
    state. Fitting the model tells us those states and which one each cell most likely
    belongs to. Because it is a probability model of how the on/off patterns arise, it
    is a good sanity check on the distance-based clustering. It assumes genes turn on
    independently within a state and that a 0 really means "off" (not just missed).

    Runs the fit several times from random starts and keeps the best. Returns a dict
    with the state weights, per-gene on-probabilities, one hard label per cell, and the
    BIC score (lower = better fit once the number of states is penalised).
    """
    gene_matrix = np.asarray(gene_matrix, dtype=float)
    rng = np.random.default_rng(seed)
    best_fit = None
    for _ in range(n_restarts):
        candidate = fit_bernoulli_mixture_once(gene_matrix, n_states, rng=rng)
        if best_fit is None or candidate[2] > best_fit[2]:
            best_fit = candidate
    weights, on_probabilities, log_likelihood, responsibilities = best_fit
    labels = responsibilities.argmax(axis=1)
    n_cells, n_genes = gene_matrix.shape
    n_params = n_states * n_genes + (n_states - 1)
    bic = -2 * log_likelihood + n_params * np.log(n_cells)
    return {
        "n_states": n_states,
        "weights": weights,
        "on_probabilities": on_probabilities,
        "labels": labels,
        "bic": bic,
    }


def select_bernoulli_mixture(gene_matrix, candidate_states=range(2, 11), seed=0):
    """Fit the mixture for several numbers of states and keep the lowest-BIC one.

    Returns (best_fit, fits_by_number_of_states).
    """
    fits_by_number_of_states = {
        n_states: fit_bernoulli_mixture(gene_matrix, n_states, seed=seed)
        for n_states in candidate_states
    }
    best_n_states = min(fits_by_number_of_states, key=lambda k: fits_by_number_of_states[k]["bic"])
    return fits_by_number_of_states[best_n_states], fits_by_number_of_states


# ---- Approach 3: KMeans (at matched K) + UMAP on the gene vectors ------------
# (Leiden is kept below as an optional graph view but is no longer the third method
# in PART 2 - KMeans lines up far better with the other two.)

def merge_small_clusters(labels, max_clusters):
    """Keep the largest clusters, fold the rest into one 'other' group.

    Guarantees at most `max_clusters` labels and renumbers them 0..n-1 by size
    (the merged 'other' group, if any, is the last id). Needed because rare gene
    states form disconnected islands in the graph that resolution tuning can't merge.
    """
    sizes = pd.Series(labels).value_counts()
    if len(sizes) <= max_clusters:
        remap = {old: new for new, old in enumerate(sizes.index)}
        return np.array([remap[label] for label in labels])
    kept = list(sizes.index[:max_clusters - 1])
    remap = {old: new for new, old in enumerate(kept)}
    other_id = max_clusters - 1
    return np.array([remap.get(label, other_id) for label in labels])


def kmeans_cluster_gene_space(gene_matrix, n_clusters, seed=0):
    """Cluster cells directly on their 0/1 gene vectors with KMeans at exactly K.

    Euclidean KMeans on binary vectors groups cells by their overall on/off profile
    (close to Hamming), giving a third view forced to the same K so it is directly
    comparable with the Bernoulli mixture and hierarchical clustering. Unlike the
    graph/Leiden view it never leaves rare states stranded in an 'other' bucket.
    """
    n_clusters = min(n_clusters, len(np.unique(gene_matrix, axis=0)))
    return KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(gene_matrix)


def leiden_cluster_gene_space(gene_matrix, k_neighbors=15, seed=0, max_clusters=8, n_clusters=None):
    """Cluster cells by shared ON genes.

    Builds a k-nearest-neighbour graph with the Jaccard metric (cells are neighbours
    when they switch on the same genes; shared OFF genes are ignored) and finds
    communities in it with Leiden.

    If `n_clusters` is given the resolution is *raised* until at least that many
    communities appear, then they are merged down to exactly `n_clusters` (used to
    match the K chosen by the Bernoulli mixture so the methods are comparable).
    Otherwise the resolution is *lowered* until at most `max_clusters` communities
    remain. Either way leftover rare-state islands are folded into a single 'other'
    cluster. Returns one label per cell.
    """
    neighbors = NearestNeighbors(n_neighbors=k_neighbors, metric="jaccard").fit(gene_matrix)
    _, neighbor_index = neighbors.kneighbors(gene_matrix)
    edges = {(min(cell, int(other)), max(cell, int(other)))
             for cell in range(neighbor_index.shape[0]) for other in neighbor_index[cell, 1:]}
    graph = ig.Graph(n=len(gene_matrix), edges=list(edges))

    def partition_at(resolution):
        return np.array(leidenalg.find_partition(
            graph, leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=resolution, seed=seed).membership)

    if n_clusters is not None:
        labels = None
        for resolution in [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]:
            labels = partition_at(resolution)
            if len(set(labels)) >= n_clusters:
                break
        return merge_small_clusters(labels, n_clusters)

    labels = None
    for resolution in [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01]:
        labels = partition_at(resolution)
        if len(set(labels)) <= max_clusters:
            break
    return merge_small_clusters(labels, max_clusters)


def umap_embed_gene_space(gene_matrix, k_neighbors=15, seed=0):
    """2-D UMAP of the same Jaccard neighbour structure - for visualisation only."""
    return umap.UMAP(metric="jaccard", n_neighbors=k_neighbors, min_dist=0.1,
                     random_state=seed).fit_transform(gene_matrix)


# =============================================================================
# PART 2 - RUN EVERYTHING, SPLIT BY CELL TYPE (put this in the next cell)
#
# How to read the printout:
#   "5 clusters identified for endothelial cells (KMeans, 1,234 cells):
#      cluster 3 (410 cells, 33%): likely states include 60% 1_4 (CDH5, KDR); ..."
#   Each approach answers the same question a different way; the ARI lines at the end
#   of each cell type say how much they agree (higher = more robust / more likely real).
# =============================================================================

CELL_TYPES = ["endothelial", "fibroblast"]
MAX_CLUSTERS = 8

for column in ["hier", "bmm", "kmeans"]:
    df[column] = -1
df["umap1"] = np.nan
df["umap2"] = np.nan

all_summaries = []

for cell_type in CELL_TYPES:
    cells = df[df.cell_type == cell_type]
    if len(cells) < 10:
        print(f"{cell_type}: only {len(cells)} cells - skipping\n")
        continue
    cell_gene_matrix = cells[GENES].astype(int).to_numpy()
    print("=" * 78)
    print(f"CELL TYPE: {cell_type}  ({len(cells):,} cells)")
    print("=" * 78)

    # --- Approach 1: Bernoulli mixture first - BIC picks K, forced on the rest ---
    best_mixture, mixtures_by_k = select_bernoulli_mixture(
        cell_gene_matrix, candidate_states=range(2, MAX_CLUSTERS + 1))
    bmm_labels = best_mixture["labels"]
    K = best_mixture["n_states"]
    print(f"Bernoulli BIC picked K = {K}; forcing K on KMeans and hierarchical.\n")
    all_summaries.append(describe_clusters(cells, bmm_labels, GENES, "Bernoulli mixture", cell_type))

    candidate_ks = list(mixtures_by_k)
    bic_values = [mixtures_by_k[k]["bic"] for k in candidate_ks]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), gridspec_kw={"width_ratios": [1, 2]})
    axes[0].plot(candidate_ks, bic_values, "o-")
    axes[0].axvline(K, color="red", ls="--", label=f"best K = {K}")
    axes[0].set(xlabel="K (number of states)", ylabel="BIC",
                title=f"{cell_type}: mixture model selection")
    axes[0].legend()
    component_probabilities = pd.DataFrame(
        best_mixture["on_probabilities"], columns=GENES,
        index=[f"state {k} ({weight:.0%})" for k, weight in enumerate(best_mixture["weights"])])
    sns.heatmap(component_probabilities, cmap="viridis", vmin=0, vmax=1, ax=axes[1],
                cbar_kws={"label": "P(gene on | state)"})
    axes[1].set_title(f"{cell_type}: Bernoulli mixture states (K={K})")
    plt.tight_layout(); plt.show()

    # --- Approach 2: KMeans on the gene vectors, forced to the same K ---
    k_neighbors = min(15, len(cells) - 1)
    kmeans_labels = kmeans_cluster_gene_space(cell_gene_matrix, n_clusters=K)
    all_summaries.append(describe_clusters(cells, kmeans_labels, GENES, "KMeans", cell_type))

    # --- Approach 3: hierarchical at K - try both metrics, keep the more agreeing one ---
    hier_options = {}
    for metric in ["jaccard", "hamming"]:
        labels_metric, plot_metric = hierarchical_cell_labels(cells, GENES, metric=metric, max_clusters=K)
        mean_ari = np.mean([
            adjusted_rand_score(labels_metric, bmm_labels),
            adjusted_rand_score(labels_metric, kmeans_labels),
        ])
        hier_options[metric] = (labels_metric, plot_metric, mean_ari)
    hier_metric = max(hier_options, key=lambda m: hier_options[m][2])
    hier_labels, hier_plot, _ = hier_options[hier_metric]
    print(f"Hierarchical metric agreement (mean ARI vs Bernoulli & KMeans): "
          f"jaccard = {hier_options['jaccard'][2]:.3f}, hamming = {hier_options['hamming'][2]:.3f} "
          f"-> using {hier_metric}\n")
    all_summaries.append(describe_clusters(cells, hier_labels, GENES, f"Hierarchical ({hier_metric})", cell_type))
    if hier_plot is not None:
        plot_state_clustermap(*hier_plot, cell_type, len(cells))

    # UMAP: collapse identical gene states to one marker, sized by how many cells share it.
    embedding = umap_embed_gene_space(cell_gene_matrix, k_neighbors=k_neighbors)
    plot_points = pd.DataFrame({
        "umap1": embedding[:, 0], "umap2": embedding[:, 1],
        "state": cells["binary_state"].to_numpy(),
        "hier": hier_labels, "bmm": bmm_labels, "kmeans": kmeans_labels,
    })
    state_points = (plot_points
        .groupby("state", as_index=False)
        .agg(umap1=("umap1", "first"), umap2=("umap2", "first"),
             hier=("hier", "first"), bmm=("bmm", "first"),
             kmeans=("kmeans", "first"), n_cells=("state", "size"))
        .sort_values("n_cells", ascending=False))
    marker_sizes = 15 + 240 * state_points["n_cells"] / state_points["n_cells"].max()

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for ax, column_name, name in zip(axes, ["hier", "bmm", "kmeans"],
                                     ["Hierarchical", "Bernoulli mixture", "KMeans"]):
        ax.scatter(state_points["umap1"], state_points["umap2"], s=marker_sizes,
                   c=state_points[column_name], cmap="tab20", alpha=0.5,
                   edgecolor="white", linewidth=0.3)
        ax.set_title(f"{cell_type} UMAP - {name} (marker area = cell count)")
        ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    plt.tight_layout(); plt.show()

    # --- Approach 4: at matched K, do the methods agree within this cell type? ---
    print(f"Agreement for {cell_type} at K={K} (Adjusted Rand Index, 1 = identical, ~0 = chance):")
    method_labels = {"hierarchical": hier_labels, "bernoulli": bmm_labels, "kmeans": kmeans_labels}
    for method_a, method_b in [("kmeans", "bernoulli"), ("kmeans", "hierarchical"), ("bernoulli", "hierarchical")]:
        labels_a, labels_b = method_labels[method_a], method_labels[method_b]
        both_assigned = (labels_a >= 0) & (labels_b >= 0)
        ari = adjusted_rand_score(labels_a[both_assigned], labels_b[both_assigned])
        print(f"  {method_a:>12} vs {method_b:<12}: ARI = {ari:.3f}")
    print()

    # Store labels back on df (offset so cell types keep distinct ids for section 3).
    hier_offset = int(df["hier"].max()) + 1
    bmm_offset = int(df["bmm"].max()) + 1
    kmeans_offset = int(df["kmeans"].max()) + 1
    df.loc[cells.index, "hier"] = np.where(hier_labels >= 0, hier_labels + hier_offset, -1)
    df.loc[cells.index, "bmm"] = bmm_labels + bmm_offset
    df.loc[cells.index, "kmeans"] = kmeans_labels + kmeans_offset
    df.loc[cells.index, "umap1"] = embedding[:, 0]
    df.loc[cells.index, "umap2"] = embedding[:, 1]

cluster_overview = pd.concat(all_summaries, ignore_index=True)
cluster_overview.to_csv(output_dir / "section2_cluster_overview.csv", index=False)
print("Saved -> section2_cluster_overview.csv")
