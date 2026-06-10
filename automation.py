import pandas as pd
import numpy as np
import pickle, warnings, os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, f1_score

warnings.filterwarnings("ignore")


class AutoMLPipeline:

    # regression models and their hyperparameters
    regression_models = {
        "LinearRegression": {
            "model": LinearRegression(),
            "params": {
                "fit_intercept": [True, False]
            }
        },
        "DecisionTreeRegressor": {
            "model": DecisionTreeRegressor(random_state=42),
            "params": {
                "max_depth": [None, 3, 5, 10],
                "min_samples_split": [2, 5, 10]
            }
        },
        "RandomForestRegressor": {
            "model": RandomForestRegressor(random_state=42),
            "params": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 5, 10],
                "min_samples_split": [2, 5]
            }
        },
        "KNeighborsRegressor": {
            "model": KNeighborsRegressor(),
            "params": {
                "n_neighbors": [3, 5, 7, 10],
                "weights": ["uniform", "distance"],
                "p": [1, 2]
            }
        }
    }

    # classification models and their hyperparameters
    classification_models = {
        "LogisticRegression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "params": {
                "C": [0.01, 0.1, 1, 10],
                "solver": ["lbfgs", "liblinear"]
            }
        },
        "DecisionTreeClassifier": {
            "model": DecisionTreeClassifier(random_state=42),
            "params": {
                "max_depth": [None, 3, 5, 10],
                "criterion": ["gini", "entropy"]
            }
        },
        "RandomForestClassifier": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 5, 10],
                "criterion": ["gini", "entropy"]
            }
        },
        "NaiveBayes": {
            "model": GaussianNB(),
            "params": {
                "var_smoothing": [1e-9, 1e-8, 1e-7]
            }
        },
        "SVM": {
            "model": SVC(random_state=42),
            "params": {
                "C": [0.1, 1, 10],
                "kernel": ["rbf", "linear"],
                "gamma": ["scale", "auto"]
            }
        },
        "KNeighborsClassifier": {
            "model": KNeighborsClassifier(),
            "params": {
                "n_neighbors": [3, 5, 7, 10],
                "weights": ["uniform", "distance"],
                "p": [1, 2]
            }
        }
    }

    def __init__(self, filepath, target_column, variance_threshold=0.0,
                 corr_threshold=0.05, multicollinearity_threshold=0.95,
                 test_size=0.2, cv=5, random_state=42):

        self.filepath                    = filepath
        self.target_column               = target_column
        self.variance_threshold          = variance_threshold
        self.corr_threshold              = corr_threshold
        self.multicollinearity_threshold = multicollinearity_threshold
        self.test_size                   = test_size
        self.cv                          = cv
        self.random_state                = random_state

        self.df              = None
        self.task_type       = None
        self.X_train         = None
        self.X_test          = None
        self.y_train         = None
        self.y_test          = None
        self.results_df      = None
        self.best_model_name = None
        self.best_model      = None
        self.label_encoders  = {}
        self.scaler          = StandardScaler()


    # step 1 - load the data from given file path
    def load_data(self):
        print("\n--- Step 1 : Loading Data ---")
        ext = os.path.splitext(self.filepath)[1].lower()
        if ext in [".xlsx", ".xls"]:
            self.df = pd.read_excel(self.filepath)
        elif ext == ".csv":
            self.df = pd.read_csv(self.filepath)
        else:
            raise ValueError("only .csv and .xlsx files are supported")
        print(f"data loaded successfully : {self.df.shape[0]} rows and {self.df.shape[1]} columns")
        print(f"target column : {self.target_column}")


    # step 2 - drop irrelevant columns using variance, correlation and multicollinearity
    def drop_irrelevant_columns(self):
        print("\n--- Step 2 : Dropping Irrelevant Columns ---")

        dropped  = {}
        features = [c for c in self.df.columns if c != self.target_column]
        num_cols = self.df[features].select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = self.df[features].select_dtypes(include=[object, "category"]).columns.tolist()

        # convert target to numeric for correlation calculation
        target_series = self.df[self.target_column].copy()
        try:
            target_numeric = target_series.astype(float)
        except (ValueError, TypeError):
            target_numeric = pd.Series(
                LabelEncoder().fit_transform(target_series.fillna("missing").astype(str)),
                index=self.df.index, dtype=float
            )

        # fill nan temporarily just for variance and correlation calculation
        temp_df = self.df[num_cols].copy()
        for col in num_cols:
            temp_df[col] = temp_df[col].fillna(temp_df[col].median())

        # method 1 - variance filter
        # columns with very low or zero variance have no useful information
        print("\n  Method 1 : Variance Filter")
        variances = temp_df.var()
        for col in num_cols:
            if variances[col] <= self.variance_threshold:
                dropped[col] = f"variance = {variances[col]:.6f} which is too low"
                print(f"  drop  {col}  ->  {dropped[col]}")

        # method 2 - correlation with target
        # if a feature has very low correlation with target it is not useful
        print(f"\n  Method 2 : Correlation with Target  (threshold = {self.corr_threshold})")
        print(f"  {'Column':<25} {'Covariance':>14}  {'|Correlation|':>14}  Decision")
        print(f"  {'-'*25} {'-'*14}  {'-'*14}  {'-'*20}")

        remaining    = [c for c in num_cols if c not in dropped]
        corr_target  = {}

        for col in remaining:
            cov  = np.cov(temp_df[col], target_numeric)[0, 1]
            corr = np.corrcoef(temp_df[col], target_numeric)[0, 1]
            corr_target[col] = corr
            if abs(corr) < self.corr_threshold:
                dropped[col] = f"|r| = {abs(corr):.4f} too low correlation with target"
                decision = f"drop  (|r| = {abs(corr):.4f})"
            else:
                decision = f"keep  (|r| = {abs(corr):.4f})"
            print(f"  {col:<25} {cov:>14.4f}  {abs(corr):>14.4f}  {decision}")

        # method 3 - multicollinearity between features
        # if two features are highly correlated with each other
        # we keep the one that has higher correlation with target
        print(f"\n  Method 3 : Multicollinearity  (threshold = {self.multicollinearity_threshold})")
        active = [c for c in remaining if c not in dropped]

        if len(active) > 1:
            corr_matrix = temp_df[active].corr().abs()
            seen = set()
            for i in range(len(active)):
                for j in range(i + 1, len(active)):
                    c1, c2 = active[i], active[j]
                    if c1 in seen or c2 in seen:
                        continue
                    if corr_matrix.loc[c1, c2] >= self.multicollinearity_threshold:
                        r1 = abs(corr_target.get(c1, 0))
                        r2 = abs(corr_target.get(c2, 0))
                        drop_col = c2 if r1 >= r2 else c1
                        keep_col = c1 if drop_col == c2 else c2
                        seen.add(drop_col)
                        if drop_col not in dropped:
                            dropped[drop_col] = f"multicollinear with {keep_col}  |r| = {corr_matrix.loc[c1,c2]:.4f}"
                        print(f"  drop  {drop_col}  ->  correlated with {keep_col}  |r| = {corr_matrix.loc[c1,c2]:.4f}")
        else:
            print("  no multicollinearity issue found")

        # method 4 - constant categorical columns
        print("\n  Method 4 : Constant Categorical Columns")
        for col in cat_cols:
            if self.df[col].nunique() <= 1:
                dropped[col] = "constant column  only one unique value"
                print(f"  drop  {col}  ->  {dropped[col]}")

        # now actually drop them from dataframe
        self.df.drop(columns=list(dropped.keys()), inplace=True, errors="ignore")

        # summary
        remaining_features = [c for c in self.df.columns if c != self.target_column]
        remaining_num = self.df[remaining_features].select_dtypes(include=[np.number]).columns.tolist()
        remaining_cat = self.df[remaining_features].select_dtypes(include=[object, "category"]).columns.tolist()

        print(f"\n  Summary :")
        print(f"  columns dropped        : {list(dropped.keys())}")
        print(f"  numerical features     : {remaining_num}")
        print(f"  categorical features   : {remaining_cat}")
        print(f"  total remaining        : {len(remaining_features)} features + target column")


    # step 3 - detect if it is regression or classification problem
    def detect_task_type(self):
        print("\n--- Step 3 : Detecting Task Type ---")
        target = self.df[self.target_column]
        if target.dtype == object or target.nunique() <= 20:
            self.task_type = "classification"
        else:
            self.task_type = "regression"
        print(f"target dtype    : {target.dtype}")
        print(f"unique values   : {target.nunique()}")
        print(f"task type       : {self.task_type}")


    # step 4 - preprocess numerical and categorical features separately
    def preprocess(self):
        print("\n--- Step 4 : Preprocessing ---")

        features = self.df.drop(columns=[self.target_column])
        target   = self.df[self.target_column].copy()

        num_cols = features.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = features.select_dtypes(include=[object, "category"]).columns.tolist()

        print(f"numerical columns   : {num_cols}")
        print(f"categorical columns : {cat_cols}")

        # numerical - fill missing with median then apply log transform for skewed columns
        if num_cols:
            features[num_cols] = SimpleImputer(strategy="median").fit_transform(features[num_cols])
            print("\n  numerical missing values filled with median")

            skewed = features[num_cols].apply(lambda c: c.skew()).abs()
            skewed_cols = skewed[skewed > 0.75].index.tolist()
            for col in skewed_cols:
                if features[col].min() >= 0:
                    features[col] = np.log1p(features[col])
            if skewed_cols:
                print(f"  log1p applied to skewed columns : {skewed_cols}")

        # categorical - fill missing with mode then label encode
        if cat_cols:
            features[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(features[cat_cols])
            print("\n  categorical missing values filled with mode")

            for col in cat_cols:
                le = LabelEncoder()
                features[col] = le.fit_transform(features[col].astype(str))
                self.label_encoders[col] = le
            print(f"  label encoding done for : {cat_cols}")

        # encode target column if classification and it is string type
        if self.task_type == "classification" and target.dtype == object:
            le_target = LabelEncoder()
            target = le_target.fit_transform(target.astype(str))
            self.label_encoders["__target__"] = le_target
            print("\n  target column label encoded")

        # scale all features
        X = self.scaler.fit_transform(features)
        y = np.array(target)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        print(f"\n  Preprocessing Summary :")
        print(f"  numerical features     : {num_cols}")
        print(f"  missing values filled  : median for numerical , mode for categorical")
        print(f"  skewed columns fixed   : {skewed_cols if num_cols else []}")
        print(f"  categorical encoded    : {cat_cols}")
        print(f"  train size : {self.X_train.shape}  |  test size : {self.X_test.shape}")


    # step 5 - train all models with gridsearch hyperparameter tuning
    def train_models(self):
        print("\n--- Step 5 : Training Models with Hyperparameter Tuning ---")

        if self.task_type == "regression":
            model_configs = self.regression_models
        else:
            model_configs = self.classification_models

        scoring = "r2" if self.task_type == "regression" else "accuracy"
        results = []

        for name, config in model_configs.items():
            print(f"\n  training {name} ...")
            gs = GridSearchCV(
                estimator=config["model"],
                param_grid=config["params"],
                cv=self.cv,
                scoring=scoring,
                n_jobs=-1
            )
            gs.fit(self.X_train, self.y_train)
            best_model = gs.best_estimator_
            y_pred     = best_model.predict(self.X_test)

            if self.task_type == "regression":
                r2   = r2_score(self.y_test, y_pred)
                mae  = mean_absolute_error(self.y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
                print(f"  best params : {gs.best_params_}")
                print(f"  R2 = {r2:.4f}   MAE = {mae:.4f}   RMSE = {rmse:.4f}   CV_R2 = {gs.best_score_:.4f}")
                results.append({
                    "Model"      : name,
                    "R2_Score"   : round(r2, 4),
                    "MAE"        : round(mae, 4),
                    "RMSE"       : round(rmse, 4),
                    "CV_R2"      : round(gs.best_score_, 4),
                    "Best_Params": str(gs.best_params_),
                    "_estimator" : best_model
                })
            else:
                acc = accuracy_score(self.y_test, y_pred)
                f1  = f1_score(self.y_test, y_pred, average="weighted")
                print(f"  best params : {gs.best_params_}")
                print(f"  Accuracy = {acc:.4f}   F1 = {f1:.4f}   CV_Accuracy = {gs.best_score_:.4f}")
                results.append({
                    "Model"      : name,
                    "Accuracy"   : round(acc, 4),
                    "F1_Score"   : round(f1, 4),
                    "CV_Accuracy": round(gs.best_score_, 4),
                    "Best_Params": str(gs.best_params_),
                    "_estimator" : best_model
                })

        self.results_df = pd.DataFrame(results)


    # step 6 - show results table and pick the best model
    def show_results(self):
        print("\n--- Step 6 : Model Comparison Table ---")

        show_cols = [c for c in self.results_df.columns if c not in ["_estimator", "Best_Params"]]
        print(self.results_df[show_cols].to_string(index=False))

        metric   = "R2_Score" if self.task_type == "regression" else "Accuracy"
        best_idx = self.results_df[metric].idxmax()

        self.best_model_name = self.results_df.loc[best_idx, "Model"]
        self.best_model      = self.results_df.loc[best_idx, "_estimator"]
        best_score           = self.results_df.loc[best_idx, metric]

        print(f"\n  best model : {self.best_model_name}  ({metric} = {best_score})")


    # step 7 - retrain best model on full data and save as pkl file
    def export_best_model(self):
        print("\n--- Step 7 : Saving Best Model ---")

        X_full = np.vstack([self.X_train, self.X_test])
        y_full = np.concatenate([self.y_train, self.y_test])
        self.best_model.fit(X_full, y_full)
        print(f"  {self.best_model_name} retrained on full data  ({len(y_full)} samples)")

        bundle = {
            "model"         : self.best_model,
            "model_name"    : self.best_model_name,
            "task_type"     : self.task_type,
            "scaler"        : self.scaler,
            "label_encoders": self.label_encoders,
            "target_column" : self.target_column
        }

        filename = f"best_model_{self.best_model_name}.pkl"
        pickle.dump(bundle, open(filename, "wb"))
        print(f"  model saved as : {filename}")
        return filename


    # run all steps in order
    def run(self):
        print("\n========== AutoML Pipeline Started ==========")
        self.load_data()
        self.drop_irrelevant_columns()
        self.detect_task_type()
        self.preprocess()
        self.train_models()
        self.show_results()
        pkl = self.export_best_model()
        print(f"\n========== Done  |  best model saved : {pkl} ==========\n")
        return self.results_df, self.best_model_name, self.best_model


def run_pipeline(filepath, target_column, **kwargs):
    pipeline = AutoMLPipeline(filepath=filepath, target_column=target_column, **kwargs)
    return pipeline.run()


if __name__ == "__main__":

    # enter your file path and target column name here
    filepath      = "Attrition - Attrition.csv"
    target_column = "Attrition"

    results_df, best_name, best_model = run_pipeline(filepath, target_column)
