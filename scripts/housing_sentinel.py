import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr, pearsonr

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (silhouette_score, davies_bouldin_score, r2_score, 
                             mean_squared_error, mean_absolute_error, accuracy_score, 
                             f1_score, precision_score, recall_score, classification_report)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import KFold

RANDOM_STATE = 42
CV_FOLDS = 5
sns.set_theme(style="whitegrid")


def target_encode_oof(df, col, target_col, n_splits=5):
    oof = pd.Series(index=df.index, dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    for train_idx, val_idx in kf.split(df):
        train, val = df.iloc[train_idx], df.iloc[val_idx]
        means = train.groupby(col)[target_col].mean()
        oof.iloc[val_idx] = val[col].map(means).fillna(df[target_col].mean())
    oof.fillna(df[target_col].mean(), inplace=True)
    return oof


class AmesHousingAnalyzer:
    
    def __init__(self, filepath, target_column='SalePrice', random_state=RANDOM_STATE):
        self.filepath = filepath
        self.target_column = target_column
        self.random_state = random_state
        self.df = None
        self.df_preprocessed = None
        self.df_clean = None
        self.pca_model = None
        self.df_pca = None
        self.selected_features = []
        self.results = {}
        self.preprocessing_applied = False
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(base_dir, "images")
        os.makedirs(self.output_dir, exist_ok=True)

    def _save_fig(self, filename, dpi=300):
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close()

    @staticmethod
    def _freedman_diaconis_bins(x):
        x = x[~np.isnan(x)]
        if len(x) < 2:
            return 10
        iqr = np.subtract(*np.percentile(x, [75, 25]))
        if iqr == 0:
            return int(np.sqrt(len(x)))
        h = 2 * iqr * (len(x) ** (-1 / 3))
        if h <= 0:
            return int(np.sqrt(len(x)))
        return max(10, int(np.ceil((x.max() - x.min()) / h)))

    def load_data(self):
        self.df = pd.read_csv(self.filepath)
        return self

    def analyze_target(self, target="SalePrice"):
        s = self.df[target].dropna()
        bins = self._freedman_diaconis_bins(s.values)
        log_s = np.log1p(s.clip(lower=0))

        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        axs = axs.flatten()

        axs[0].hist(s, bins=bins, edgecolor="black")
        axs[0].set_title("SalePrice distribution")

        axs[1].boxplot(s, vert=False)
        axs[1].set_title("Boxplot")

        axs[2].hist(log_s, bins=self._freedman_diaconis_bins(log_s.values),
                    edgecolor="black")
        axs[2].set_title("log1p(SalePrice)")

        stats.probplot(log_s, dist="norm", plot=axs[3])
        axs[3].set_title("Q-Q plot (log)")

        sns.kdeplot(s, ax=axs[4], fill=True)
        axs[4].set_title("KDE")

        axs[5].scatter(range(len(s)), s, s=6)
        axs[5].set_title("Index vs value")

        plt.tight_layout()
        self._save_fig("01_target_distribution.png")

    def missing_value_report(self, top_n=30):
        missing = self.df.isnull().sum()
        missing_pct = 100 * missing / len(self.df)
        missing_df = (
            pd.DataFrame({"missing_pct": missing_pct})
            .query("missing_pct > 0")
            .sort_values("missing_pct", ascending=False)
            .head(top_n)
        )

        if missing_df.empty:
            return

        missing_df.plot(kind="barh", figsize=(12, 8))
        plt.xlabel("Percent missing (%)")
        plt.title("Missing values by feature")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        self._save_fig("02_missing_values.png")

    def show_correlations(self, target="SalePrice", top_k=15):
        numeric = self.df.select_dtypes(include=[np.number])
        corr = numeric.corrwith(numeric[target]).abs().drop(target)
        top = corr.sort_values(ascending=False).head(top_k)

        fig, ax = plt.subplots(1, 2, figsize=(14, 5))

        sns.barplot(x=top.values, y=top.index, ax=ax[0])
        ax[0].set_title("Top correlations with target")

        sub_corr = numeric[top.index.tolist() + [target]].corr()
        mask = np.triu(np.ones_like(sub_corr, dtype=bool))
        sns.heatmap(sub_corr, mask=mask, cmap="coolwarm", center=0, ax=ax[1])
        ax[1].set_title("Correlation matrix")

        plt.tight_layout()
        self._save_fig("03_correlations.png")

        sns.pairplot(self.df, x_vars=top.index[:6], y_vars=[target], kind="reg")
        self._save_fig("04_scatter_top_features.png")

    def prepare_numeric(self):
        numeric = self.df.select_dtypes(include=[np.number])
        numeric = numeric.drop(columns=["SalePrice", "Order", "PID"], errors="ignore")
        numeric = numeric.fillna(numeric.median())

        scaler = StandardScaler()
        self.df_clean = pd.DataFrame(
            scaler.fit_transform(numeric),
            columns=numeric.columns
        )

    def run_pca(self):
        self.prepare_numeric()
        self.pca_model = PCA(n_components=0.85, random_state=self.random_state)
        self.df_pca = self.pca_model.fit_transform(self.df_clean)

        evr = self.pca_model.explained_variance_ratio_
        plt.figure(figsize=(10, 4))
        plt.bar(range(1, len(evr) + 1), evr)
        plt.plot(range(1, len(evr) + 1), np.cumsum(evr), marker="o")
        plt.axhline(0.85, linestyle="--")
        plt.xlabel("Component")
        plt.ylabel("Explained variance")
        plt.title("PCA scree plot")
        plt.tight_layout()
        self._save_fig("05_pca_scree.png")

    def clustering(self):
        inertias, silhouettes = [], []
        ks = range(2, 11)

        for k in ks:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=50)
            labels = km.fit_predict(self.df_pca)
            inertias.append(km.inertia_)
            silhouettes.append(silhouette_score(self.df_pca, labels))

        fig, ax = plt.subplots(1, 2, figsize=(14, 4))
        ax[0].plot(ks, inertias, marker="o")
        ax[0].set_title("Elbow")

        ax[1].plot(ks, silhouettes, marker="o")
        ax[1].set_title("Silhouette")

        plt.tight_layout()
        self._save_fig("06_kmeans_selection.png")

        best_k = ks[np.argmax(silhouettes)]
        km = KMeans(n_clusters=best_k, random_state=self.random_state, n_init=50)
        labels = km.fit_predict(self.df_pca)

        plt.figure(figsize=(8, 5))
        plt.scatter(self.df_pca[:, 0], self.df_pca[:, 1], c=labels, s=15)
        plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
                    c="red", marker="X", s=150)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title(f"KMeans clustering (k={best_k})")
        plt.tight_layout()
        self._save_fig("07_kmeans_clusters.png")

    def feature_importance(self):
        numeric = self.df.select_dtypes(include=[np.number]).dropna()
        X = numeric.drop(columns=["SalePrice", "Order", "PID"], errors="ignore")
        y = np.log1p(numeric["SalePrice"])

        rf = RandomForestRegressor(n_estimators=200, random_state=self.random_state)
        rf.fit(X, y)

        perm = permutation_importance(rf, X, y, n_repeats=10,
                                      random_state=self.random_state)
        idx = np.argsort(perm.importances_mean)[::-1][:15]

        imp_df = pd.DataFrame({
            "feature": X.columns[idx],
            "importance": perm.importances_mean[idx]
        })

        plt.figure(figsize=(8, 6))
        sns.barplot(x="importance", y="feature", data=imp_df)
        plt.title("Permutation importance")
        plt.tight_layout()
        self._save_fig("08_feature_importance.png")

    def preprocess_data(self, handle_outliers=True, cardinality_threshold=10):
        df = self.df.copy()
        
        id_cols = ['Order', 'PID']
        existing_ids = [c for c in id_cols if c in df.columns]
        if existing_ids:
            df.drop(columns=existing_ids, inplace=True)
        
        rename_map = {'1st Flr SF': 'FirstFlrSF', '2nd Flr SF': 'SecondFlrSF', 'Total Bsmt SF': 'TotalBsmtSF'}
        df.rename(columns=rename_map, inplace=True)
        
        area_parts = [c for c in ['FirstFlrSF', 'SecondFlrSF', 'TotalBsmtSF'] if c in df.columns]
        if area_parts:
            df['TotalSF'] = df[area_parts].sum(axis=1)
        
        if 'Year Built' in df.columns and 'Yr Sold' in df.columns:
            df['HouseAge'] = df['Yr Sold'] - df['Year Built']
            
        if 'Year Remod/Add' in df.columns and 'Yr Sold' in df.columns:
            df['SinceRemod'] = df['Yr Sold'] - df['Year Remod/Add']
        
        if 'Pool Area' in df.columns:
            df['HasPool'] = (df['Pool Area'] > 0).astype(int)
        if 'Fireplaces' in df.columns:
            df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)
        if 'Garage Type' in df.columns:
            df['HasGarage'] = (~df['Garage Type'].isnull()).astype(int)
        
        if self.target_column in df.columns:
            target_skew = df[self.target_column].skew()
            if abs(target_skew) > 0.75:
                df['SalePrice_log'] = np.log1p(df[self.target_column])
        
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for t in [self.target_column, 'SalePrice_log']:
            if t in num_cols:
                num_cols.remove(t)
        
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        low_card = [c for c in cat_cols if df[c].nunique() <= cardinality_threshold]
        high_card = [c for c in cat_cols if df[c].nunique() > cardinality_threshold]
        
        for c in num_cols:
            if df[c].isnull().sum() > 0:
                df[c] = df[c].fillna(df[c].median())
        for c in cat_cols:
            df[c] = df[c].fillna('Missing')
        
        if handle_outliers:
            for col in num_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        for col in low_card:
            df[col] = df[col].astype('category')
        
        scaler = MinMaxScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
        
        for col in high_card:
            if self.target_column in df.columns:
                encoded = target_encode_oof(df, col, self.target_column, n_splits=5)
                df[f'{col}_encoded'] = encoded
                df.drop(columns=[col], inplace=True)
        
        self.df_preprocessed = df
        self.preprocessing_applied = True
        return self
    
    def select_features_mi(self, k=40):
        if not self.preprocessing_applied:
            self.preprocess_data()
        
        df = self.df_preprocessed.copy()
        
        target_col = 'SalePrice_log' if 'SalePrice_log' in df.columns else self.target_column
        if target_col not in df.columns:
            return self
        
        y = df[target_col]
        
        exclude_cols = [self.target_column, 'SalePrice_log']
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        X_num = df[feature_cols].select_dtypes(include=[np.number])
        
        X_num_filled = X_num.fillna(X_num.median())
        
        mi_scores = mutual_info_regression(X_num_filled, y, random_state=self.random_state)
        mi_series = pd.Series(mi_scores, index=X_num_filled.columns).sort_values(ascending=False)
        
        self.selected_features = mi_series.head(k).index.tolist()
        
        return self
    
    def evaluate_regression(self, n_splits=5):
        if not self.preprocessing_applied:
            self.preprocess_data()
        
        if not self.selected_features:
            self.select_features_mi(k=40)
        
        df = self.df_preprocessed.copy()
        target_col = 'SalePrice_log' if 'SalePrice_log' in df.columns else self.target_column
        
        X = df[self.selected_features]
        y = df[target_col]
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        r2_scores, rmse_scores, mae_scores = [], [], []
        
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            rf = RandomForestRegressor(n_estimators=200, random_state=self.random_state, n_jobs=-1)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            
            r2_scores.append(r2_score(y_test, y_pred))
            rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
            mae_scores.append(mean_absolute_error(y_test, y_pred))
        
        mean_r2 = np.mean(r2_scores)
        std_r2 = np.std(r2_scores)
        mean_rmse = np.mean(rmse_scores)
        std_rmse = np.std(rmse_scores)
        mean_mae = np.mean(mae_scores)
        
        rf_final = RandomForestRegressor(n_estimators=200, random_state=self.random_state, n_jobs=-1)
        rf_final.fit(X, y)
        feature_importances = pd.Series(rf_final.feature_importances_, index=X.columns).sort_values(ascending=False)
        
        verdict = "Excellent" if mean_r2 > 0.85 else "Good" if mean_r2 > 0.70 else "Moderate"
        
        self.results['regression'] = {
            'mean_r2': mean_r2,
            'std_r2': std_r2,
            'mean_rmse': mean_rmse,
            'std_rmse': std_rmse,
            'mean_mae': mean_mae,
            'feature_importances': feature_importances,
            'verdict': verdict
        }
        
        return self
    
    def evaluate_clustering(self, method='kmeans'):
        if not self.preprocessing_applied:
            self.preprocess_data()
        
        if not self.selected_features:
            self.select_features_mi(k=40)
        
        df = self.df_preprocessed.copy()
        X = df[self.selected_features]
        
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        pca = PCA(n_components=0.90, random_state=self.random_state)
        X_pca = pca.fit_transform(X_scaled)
        
        if method == 'kmeans':
            best_k = 2
            best_silhouette = -1
            
            for k in range(2, 11):
                km = KMeans(n_clusters=k, random_state=self.random_state, n_init=50)
                labels = km.fit_predict(X_pca)
                silhouette = silhouette_score(X_pca, labels)
                
                if silhouette > best_silhouette:
                    best_silhouette = silhouette
                    best_k = k
            
            verdict = "Excellent" if best_silhouette > 0.50 else "Good" if best_silhouette > 0.35 else "Moderate"
            
            self.results['clustering'] = {
                'method': 'kmeans',
                'best_k': best_k,
                'best_silhouette': best_silhouette,
                'verdict': verdict
            }
        
        elif method == 'dbscan':
            best_params = None
            best_silhouette = -1
            
            eps_values = np.linspace(0.3, 2.0, 8)
            min_samples_values = [3, 5, 10]
            
            for eps in eps_values:
                for min_samples in min_samples_values:
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                    labels = dbscan.fit_predict(X_pca)
                    
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    if n_clusters < 2:
                        continue
                    
                    silhouette = silhouette_score(X_pca, labels)
                    if silhouette > best_silhouette:
                        best_silhouette = silhouette
                        best_params = {'eps': eps, 'min_samples': min_samples}
            
            verdict = "Excellent" if best_silhouette > 0.50 else "Good" if best_silhouette > 0.35 else "Moderate"
            
            self.results['clustering'] = {
                'method': 'dbscan',
                'best_params': best_params,
                'best_silhouette': best_silhouette,
                'verdict': verdict
            }
        
        return self
    
    def evaluate_classification(self, n_bins=3, n_splits=5):
        if not self.preprocessing_applied:
            self.preprocess_data()
        
        if not self.selected_features:
            self.select_features_mi(k=40)
        
        df = self.df_preprocessed.copy()
        target_col = 'SalePrice_log' if 'SalePrice_log' in df.columns else self.target_column
        
        y_binned = pd.qcut(df[target_col], q=n_bins, labels=False, duplicates='drop')
        
        class_counts = y_binned.value_counts()
        balance_ratio = class_counts.min() / class_counts.max()
        
        X = df[self.selected_features]
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        accuracy_scores, f1_scores = [], []
        
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y_binned.iloc[train_idx], y_binned.iloc[test_idx]
            
            rf = RandomForestClassifier(n_estimators=200, random_state=self.random_state, n_jobs=-1)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            
            accuracy_scores.append(accuracy_score(y_test, y_pred))
            f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
        
        mean_accuracy = np.mean(accuracy_scores)
        mean_f1 = np.mean(f1_scores)
        baseline_accuracy = class_counts.max() / len(y_binned)
        
        verdict = "Excellent" if mean_f1 > 0.85 else "Good" if mean_f1 > 0.70 else "Moderate"
        
        self.results['classification'] = {
            'mean_accuracy': mean_accuracy,
            'mean_f1': mean_f1,
            'balance_ratio': balance_ratio,
            'baseline_accuracy': baseline_accuracy,
            'verdict': verdict
        }
        
        return self

    def run_visualization_report(self):
        self.analyze_target()
        self.missing_value_report()
        self.show_correlations()
        self.run_pca()
        self.clustering()
        self.feature_importance()

    def run_full_analysis(self, n_features=40, clustering_method='kmeans', 
                         classification_bins=3, cv_folds=5):
        self.load_data()
        self.preprocess_data(handle_outliers=True, cardinality_threshold=10)
        self.select_features_mi(k=n_features)
        self.evaluate_regression(n_splits=cv_folds)
        self.evaluate_clustering(method=clustering_method)
        self.evaluate_classification(n_bins=classification_bins, n_splits=cv_folds)
        self.run_visualization_report()
        return self


if __name__ == "__main__":
    analyzer = AmesHousingAnalyzer(
        filepath='AmesHousing.csv',
        target_column='SalePrice'
    )
    
    analyzer.run_full_analysis(
        n_features=40,
        clustering_method='kmeans',
        classification_bins=3,
        cv_folds=5
    )