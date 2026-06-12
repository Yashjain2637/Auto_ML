import pandas as pd
import numpy as np
import pickle, warnings, os

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

warnings.filterwarnings("ignore")


class UnsupervisedPipeline:

    def __init__(self, filepath, max_k=10, dbscan_eps=0.5,
                 dbscan_min_samples=5, min_support=0.2, min_confidence=0.5):

        self.filepath           = filepath
        self.max_k              = max_k
        self.dbscan_eps         = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.min_support        = min_support
        self.min_confidence     = min_confidence

        self.df             = None
        self.df_clean       = None
        self.X_scaled       = None
        self.label_encoders = {}
        self.scaler         = StandardScaler()

        self.best_k             = None
        self.kmeans_model       = None
        self.hierarchical_model = None
        self.dbscan_model       = None
        self.gmm_model          = None
        self.pca_model          = None

        self.kmeans_labels      = None
        self.hierarchical_labels= None
        self.dbscan_labels      = None
        self.gmm_labels         = None
        self.X_pca              = None

        self.linkage_matrix     = None
        self.association_df     = None

        # scores for best model comparison
        self.model_scores       = {}


    # step 1 - load data
    def load_data(self):
        print("\n--- Step 1 : Loading Data ---")
        ext = os.path.splitext(self.filepath)[1].lower()
        if ext in [".xlsx", ".xls"]:
            self.df = pd.read_excel(self.filepath)
        elif ext == ".csv":
            self.df = pd.read_csv(self.filepath)
        else:
            raise ValueError("only .csv and .xlsx files are supported")
        print(f"data loaded : {self.df.shape[0]} rows and {self.df.shape[1]} columns")


    # step 2 - drop irrelevant columns
    def drop_irrelevant_columns(self):
        print("\n--- Step 2 : Dropping Irrelevant Columns ---")

        cols_to_drop = []
        num_cols     = self.df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols     = self.df.select_dtypes(include=[object, "category"]).columns.tolist()

        # fill nan temporarily for calculations
        temp_df = self.df[num_cols].copy()
        for col in num_cols:
            temp_df[col] = temp_df[col].fillna(temp_df[col].median())

        # 1. zero variance columns - no information at all
        zero_var = [c for c in num_cols if temp_df[c].var() == 0]
        cols_to_drop.extend(zero_var)

        # 2. low std columns - almost no variation
        low_var = [c for c in num_cols if c not in cols_to_drop and temp_df[c].std() < 0.01]
        cols_to_drop.extend(low_var)

        # 3. constant categorical columns
        const_cat = [c for c in cat_cols if self.df[c].nunique() <= 1]
        cols_to_drop.extend(const_cat)

        # 4. id-like columns - almost all unique values
        id_like = [c for c in cat_cols if self.df[c].nunique() / len(self.df) > 0.95]
        cols_to_drop.extend(id_like)

        # 5. drop all at once
        cols_to_drop = list(set(cols_to_drop))
        self.df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

        rem_num = self.df.select_dtypes(include=[np.number]).columns.tolist()
        rem_cat = self.df.select_dtypes(include=[object, "category"]).columns.tolist()

        print(f"  columns dropped      : {cols_to_drop}")
        print(f"  numerical features   : {rem_num}")
        print(f"  categorical features : {rem_cat}")
        print(f"  total remaining      : {self.df.shape[1]} columns")


    # step 3 - preprocess
    def preprocess(self):
        print("\n--- Step 3 : Preprocessing ---")

        self.df_clean = self.df.copy()
        num_cols = self.df_clean.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = self.df_clean.select_dtypes(include=[object, "category"]).columns.tolist()

        # numerical - fill missing with median then fix skewed columns
        if num_cols:
            self.df_clean[num_cols] = SimpleImputer(strategy="median").fit_transform(self.df_clean[num_cols])
            skewed_cols = [c for c in num_cols if abs(self.df_clean[c].skew()) > 0.75 and self.df_clean[c].min() >= 0]
            for col in skewed_cols:
                self.df_clean[col] = np.log1p(self.df_clean[col])

        # categorical - fill missing with mode then label encode
        if cat_cols:
            self.df_clean[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(self.df_clean[cat_cols])
            for col in cat_cols:
                le = LabelEncoder()
                self.df_clean[col] = le.fit_transform(self.df_clean[col].astype(str))
                self.label_encoders[col] = le

        self.X_scaled = self.scaler.fit_transform(self.df_clean)

        print(f"  Preprocessing Summary :")
        print(f"  numerical features    : {num_cols}")
        print(f"  missing values filled : median for numerical , mode for categorical")
        print(f"  categorical encoded   : {cat_cols}")
        print(f"  final shape           : {self.X_scaled.shape}")


    # step 4 - find best k using elbow and silhouette
    def find_best_k(self):
        print("\n--- Step 4 : Finding Best K ---")

        inertias          = []
        silhouette_scores = []
        k_range           = range(2, self.max_k + 1)

        for k in k_range:
            km     = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(self.X_scaled)
            inertias.append(km.inertia_)
            silhouette_scores.append(silhouette_score(self.X_scaled, labels))
            print(f"  k = {k}  |  inertia = {km.inertia_:.2f}  |  silhouette = {silhouette_scores[-1]:.4f}")

        self.best_k           = list(k_range)[np.argmax(silhouette_scores)]
        self.inertias         = inertias
        self.silhouette_scores= silhouette_scores
        self.k_range          = list(k_range)
        print(f"\n  best k : {self.best_k}  (silhouette = {max(silhouette_scores):.4f})")


    # step 5 - kmeans clustering
    def run_kmeans(self):
        print("\n--- Step 5 : KMeans Clustering ---")

        self.kmeans_model  = KMeans(n_clusters=self.best_k, random_state=42, n_init=10)
        self.kmeans_labels = self.kmeans_model.fit_predict(self.X_scaled)
        score              = silhouette_score(self.X_scaled, self.kmeans_labels)
        self.model_scores["KMeans"] = score

        unique, counts = np.unique(self.kmeans_labels, return_counts=True)
        print(f"  clusters : {self.best_k}  |  silhouette score : {score:.4f}")
        for c, n in zip(unique, counts):
            print(f"  cluster {c}  ->  {n} samples")


    # step 6 - hierarchical clustering
    def run_hierarchical(self):
        print("\n--- Step 6 : Hierarchical Clustering ---")

        self.hierarchical_model  = AgglomerativeClustering(n_clusters=self.best_k)
        self.hierarchical_labels = self.hierarchical_model.fit_predict(self.X_scaled)
        score                    = silhouette_score(self.X_scaled, self.hierarchical_labels)
        self.model_scores["Hierarchical"] = score

        # linkage matrix for dendrogram
        self.linkage_matrix = linkage(self.X_scaled[:min(200, len(self.X_scaled))], method="ward")

        unique, counts = np.unique(self.hierarchical_labels, return_counts=True)
        print(f"  clusters : {self.best_k}  |  silhouette score : {score:.4f}")
        for c, n in zip(unique, counts):
            print(f"  cluster {c}  ->  {n} samples")


    # step 7 - dbscan
    def run_dbscan(self):
        print("\n--- Step 7 : DBSCAN Clustering ---")

        self.dbscan_model  = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
        self.dbscan_labels = self.dbscan_model.fit_predict(self.X_scaled)

        unique     = np.unique(self.dbscan_labels)
        n_clusters = len(unique[unique != -1])
        n_noise    = np.sum(self.dbscan_labels == -1)

        if n_clusters > 1:
            mask  = self.dbscan_labels != -1
            score = silhouette_score(self.X_scaled[mask], self.dbscan_labels[mask])
            self.model_scores["DBSCAN"] = score
            print(f"  clusters : {n_clusters}  |  noise : {n_noise}  |  silhouette score : {score:.4f}")
        else:
            self.model_scores["DBSCAN"] = -1
            print(f"  clusters : {n_clusters}  |  noise : {n_noise}  |  silhouette : not applicable")


    # step 8 - gaussian mixture model
    def run_gmm(self):
        print("\n--- Step 8 : Gaussian Mixture Model ---")

        self.gmm_model  = GaussianMixture(n_components=self.best_k, random_state=42)
        self.gmm_labels = self.gmm_model.fit_predict(self.X_scaled)
        score           = silhouette_score(self.X_scaled, self.gmm_labels)
        self.model_scores["GMM"] = score

        unique, counts = np.unique(self.gmm_labels, return_counts=True)
        print(f"  components : {self.best_k}  |  silhouette score : {score:.4f}")
        for c, n in zip(unique, counts):
            print(f"  component {c}  ->  {n} samples")


    # step 9 - pca
    def run_pca(self):
        print("\n--- Step 9 : PCA (Dimensionality Reduction) ---")

        n_components   = min(self.X_scaled.shape[1], self.X_scaled.shape[0], 10)
        self.pca_model = PCA(n_components=n_components, random_state=42)
        X_pca_full     = self.pca_model.fit_transform(self.X_scaled)
        self.X_pca     = X_pca_full[:, :2]

        explained  = self.pca_model.explained_variance_ratio_
        cumulative = np.cumsum(explained)

        print(f"  total components : {n_components}")
        for i, (e, c) in enumerate(zip(explained, cumulative)):
            print(f"  PC{i+1}  ->  explained = {e*100:.2f}%  |  cumulative = {c*100:.2f}%")


    # step 10 - apriori association rules
    def run_apriori(self):
        print("\n--- Step 10 : Apriori - Association Rules ---")

        # apriori works on categorical / binary data
        # we will use original categorical columns if available
        cat_cols = self.df.select_dtypes(include=[object, "category"]).columns.tolist()

        if len(cat_cols) == 0:
            # convert numerical to binary using median split
            print("  no categorical columns found - converting numerical to binary using median")
            df_binary = pd.DataFrame()
            for col in self.df_clean.columns:
                median = self.df_clean[col].median()
                df_binary[f"{col}_high"] = (self.df_clean[col] >= median).astype(bool)
        else:
            df_binary = pd.get_dummies(self.df[cat_cols].fillna("missing")).astype(bool)

        # run apriori
        frequent_items = apriori(df_binary, min_support=self.min_support, use_colnames=True)

        if len(frequent_items) == 0:
            print(f"  no frequent itemsets found at support = {self.min_support}")
            print(f"  try lowering min_support parameter")
            self.association_df = pd.DataFrame()
            return

        rules = association_rules(frequent_items, metric="confidence", min_threshold=self.min_confidence)
        rules = rules.sort_values("lift", ascending=False)
        self.association_df = rules

        print(f"  frequent itemsets found : {len(frequent_items)}")
        print(f"  association rules found : {len(rules)}")
        if len(rules) > 0:
            print(f"\n  Top 5 Rules by Lift :")
            print(f"  {'Antecedent':<30} {'Consequent':<25} {'Support':>8} {'Confidence':>10} {'Lift':>8}")
            print(f"  {'-'*30} {'-'*25} {'-'*8} {'-'*10} {'-'*8}")
            for _, row in rules.head(5).iterrows():
                ant = str(list(row["antecedents"]))[:28]
                con = str(list(row["consequents"]))[:23]
                print(f"  {ant:<30} {con:<25} {row['support']:>8.4f} {row['confidence']:>10.4f} {row['lift']:>8.4f}")


    # step 11 - find best clustering model
    def find_best_model(self):
        print("\n--- Step 11 : Best Clustering Model ---")
        print(f"\n  {'Model':<20} {'Silhouette Score':>18}  Decision")
        print(f"  {'-'*20} {'-'*18}  {'-'*15}")

        best_name  = None
        best_score = -1

        for model, score in self.model_scores.items():
            decision = ""
            if score == max(self.model_scores.values()):
                decision = "<-- best"
                best_name  = model
                best_score = score
            print(f"  {model:<20} {score:>18.4f}  {decision}")

        print(f"\n  best model : {best_name}  (silhouette score = {best_score:.4f})")
        print(f"\n  Note : Silhouette Score closer to 1.0 means better defined clusters")
        self.best_clustering_model = best_name
        return best_name


    # step 12 - save all plots as png and pdf
    def save_visualizations(self):
        print("\n--- Step 12 : Saving Visualizations ---")

        os.makedirs("plots", exist_ok=True)
        pdf_filename = "unsupervised_results.pdf"

        def save_fig(fig, name):
            fig.savefig(f"plots/{name}.png", dpi=150, bbox_inches="tight")

        with PdfPages(pdf_filename) as pdf:

            # plot 1 - elbow curve
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(self.k_range, self.inertias, marker="o", color="steelblue", linewidth=2)
            ax.set_title("Elbow Curve - Inertia vs K", fontsize=14)
            ax.set_xlabel("Number of Clusters (K)")
            ax.set_ylabel("Inertia")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            save_fig(fig, "01_elbow_curve")
            pdf.savefig(fig)
            plt.close()

            # plot 2 - silhouette score
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(self.k_range, self.silhouette_scores, marker="s", color="tomato", linewidth=2)
            ax.axvline(x=self.best_k, color="green", linestyle="--", label=f"best k = {self.best_k}")
            ax.set_title("Silhouette Score vs K", fontsize=14)
            ax.set_xlabel("K")
            ax.set_ylabel("Silhouette Score")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            save_fig(fig, "02_silhouette_score")
            pdf.savefig(fig)
            plt.close()

            # plot 3 - kmeans on pca 2d
            fig, ax = plt.subplots(figsize=(8, 6))
            sc = ax.scatter(self.X_pca[:, 0], self.X_pca[:, 1],
                            c=self.kmeans_labels, cmap="tab10", alpha=0.6, s=30)
            plt.colorbar(sc, ax=ax, label="Cluster")
            ax.set_title(f"KMeans Clusters on PCA 2D  (k = {self.best_k})", fontsize=14)
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            save_fig(fig, "03_kmeans_clusters")
            pdf.savefig(fig)
            plt.close()

            # plot 4 - hierarchical on pca 2d
            fig, ax = plt.subplots(figsize=(8, 6))
            sc = ax.scatter(self.X_pca[:, 0], self.X_pca[:, 1],
                            c=self.hierarchical_labels, cmap="tab10", alpha=0.6, s=30)
            plt.colorbar(sc, ax=ax, label="Cluster")
            ax.set_title(f"Hierarchical Clusters on PCA 2D  (k = {self.best_k})", fontsize=14)
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            save_fig(fig, "04_hierarchical_clusters")
            pdf.savefig(fig)
            plt.close()

            # plot 5 - dendrogram
            fig, ax = plt.subplots(figsize=(10, 5))
            dendrogram(self.linkage_matrix, ax=ax, truncate_mode="level", p=5,
                       leaf_rotation=90, leaf_font_size=8)
            ax.set_title("Hierarchical Clustering Dendrogram", fontsize=14)
            ax.set_xlabel("Sample Index")
            ax.set_ylabel("Distance")
            plt.tight_layout()
            save_fig(fig, "05_dendrogram")
            pdf.savefig(fig)
            plt.close()

            # plot 6 - dbscan on pca 2d
            fig, ax = plt.subplots(figsize=(8, 6))
            sc = ax.scatter(self.X_pca[:, 0], self.X_pca[:, 1],
                            c=self.dbscan_labels, cmap="tab10", alpha=0.6, s=30)
            plt.colorbar(sc, ax=ax, label="Cluster (-1 = noise)")
            ax.set_title("DBSCAN Clusters on PCA 2D", fontsize=14)
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            save_fig(fig, "06_dbscan_clusters")
            pdf.savefig(fig)
            plt.close()

            # plot 7 - gmm on pca 2d
            fig, ax = plt.subplots(figsize=(8, 6))
            sc = ax.scatter(self.X_pca[:, 0], self.X_pca[:, 1],
                            c=self.gmm_labels, cmap="tab10", alpha=0.6, s=30)
            plt.colorbar(sc, ax=ax, label="Component")
            ax.set_title(f"GMM Clusters on PCA 2D  (components = {self.best_k})", fontsize=14)
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            save_fig(fig, "07_gmm_clusters")
            pdf.savefig(fig)
            plt.close()

            # plot 8 - pca explained variance
            explained = self.pca_model.explained_variance_ratio_
            fig, ax   = plt.subplots(figsize=(8, 5))
            ax.bar(range(1, len(explained) + 1), explained * 100, color="mediumseagreen", alpha=0.8)
            ax.plot(range(1, len(explained) + 1), np.cumsum(explained) * 100,
                    marker="o", color="navy", label="cumulative")
            ax.set_title("PCA - Explained Variance per Component", fontsize=14)
            ax.set_xlabel("Principal Component")
            ax.set_ylabel("Explained Variance (%)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            save_fig(fig, "08_pca_variance")
            pdf.savefig(fig)
            plt.close()

            # plot 9 - cluster size comparison all models
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            all_labels = [
                (self.kmeans_labels,       "KMeans",       axes[0, 0], "steelblue"),
                (self.hierarchical_labels, "Hierarchical", axes[0, 1], "mediumseagreen"),
                (self.dbscan_labels,       "DBSCAN",       axes[1, 0], "tomato"),
                (self.gmm_labels,          "GMM",          axes[1, 1], "mediumpurple"),
            ]
            for labels, title, ax, color in all_labels:
                unique, counts = np.unique(labels, return_counts=True)
                ax.bar([str(c) for c in unique], counts, color=color, alpha=0.8)
                ax.set_title(f"{title} Cluster Sizes")
                ax.set_xlabel("Cluster")
                ax.set_ylabel("Samples")
                ax.grid(True, alpha=0.3)
            plt.suptitle("Cluster Size Comparison - All Models", fontsize=15, y=1.01)
            plt.tight_layout()
            save_fig(fig, "09_cluster_sizes")
            pdf.savefig(fig)
            plt.close()

            # plot 10 - model comparison bar chart
            fig, ax = plt.subplots(figsize=(8, 5))
            models  = list(self.model_scores.keys())
            scores  = list(self.model_scores.values())
            colors  = ["steelblue" if m != self.best_clustering_model else "gold" for m in models]
            bars    = ax.bar(models, scores, color=colors, alpha=0.85, edgecolor="black")
            for bar, score in zip(bars, scores):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{score:.4f}", ha="center", va="bottom", fontsize=10)
            ax.set_title("Model Comparison - Silhouette Score\n(higher is better | gold = best model)", fontsize=13)
            ax.set_ylabel("Silhouette Score")
            ax.set_ylim(min(0, min(scores)) - 0.05, max(scores) + 0.1)
            ax.grid(True, alpha=0.3, axis="y")
            plt.tight_layout()
            save_fig(fig, "10_model_comparison")
            pdf.savefig(fig)
            plt.close()

            # plot 11 - apriori top rules
            if self.association_df is not None and len(self.association_df) > 0:
                top_rules = self.association_df.head(10)
                labels    = [f"{list(r['antecedents'])} -> {list(r['consequents'])}"
                             for _, r in top_rules.iterrows()]
                labels    = [l[:50] for l in labels]
                fig, ax   = plt.subplots(figsize=(10, 6))
                ax.barh(range(len(labels)), top_rules["lift"].values, color="coral", alpha=0.8)
                ax.set_yticks(range(len(labels)))
                ax.set_yticklabels(labels, fontsize=8)
                ax.set_title("Apriori - Top 10 Association Rules by Lift", fontsize=13)
                ax.set_xlabel("Lift")
                ax.grid(True, alpha=0.3, axis="x")
                plt.tight_layout()
                save_fig(fig, "11_apriori_rules")
                pdf.savefig(fig)
                plt.close()

        print(f"  pdf saved  : {pdf_filename}")
        print(f"  png files  : plots/ folder  ({11} plots)")
        return pdf_filename


    # step 13 - save all models as pkl
    def save_models(self):
        print("\n--- Step 13 : Saving Models ---")

        bundle = {
            "kmeans_model"       : self.kmeans_model,
            "hierarchical_model" : self.hierarchical_model,
            "dbscan_model"       : self.dbscan_model,
            "gmm_model"          : self.gmm_model,
            "pca_model"          : self.pca_model,
            "scaler"             : self.scaler,
            "label_encoders"     : self.label_encoders,
            "best_k"             : self.best_k,
            "best_model"         : self.best_clustering_model,
            "kmeans_labels"      : self.kmeans_labels,
            "hierarchical_labels": self.hierarchical_labels,
            "dbscan_labels"      : self.dbscan_labels,
            "gmm_labels"         : self.gmm_labels,
        }

        filename = "unsupervised_models.pkl"
        pickle.dump(bundle, open(filename, "wb"))
        print(f"  models saved : {filename}")
        return filename


    # run all steps in order
    def run(self):
        print("\n========== Unsupervised Pipeline Started ==========")
        self.load_data()
        self.drop_irrelevant_columns()
        self.preprocess()
        self.find_best_k()
        self.run_kmeans()
        self.run_hierarchical()
        self.run_dbscan()
        self.run_gmm()
        self.run_pca()
        self.run_apriori()
        best = self.find_best_model()
        pdf  = self.save_visualizations()
        pkl  = self.save_models()
        print(f"\n========== Done ==========")
        print(f"  best clustering model  : {best}")
        print(f"  visualizations pdf     : {pdf}")
        print(f"  png plots folder       : plots/")
        print(f"  models pkl             : {pkl}\n")


if __name__ == "__main__":

    # enter your file path here
    filepath = "your_data.csv"

    pipeline = UnsupervisedPipeline("Attrition - Attrition.csv")
    pipeline.run()