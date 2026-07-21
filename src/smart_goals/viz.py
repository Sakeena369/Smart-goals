from scipy import stats
import matplotlib.pyplot as plt
import mlflow
from sklearn.inspection import PartialDependenceDisplay
import lightgbm as lgb

def plot_importance(clf, importance_type="gain", max_num_features=20):
    try: 
        plt.figure(figsize=(20, 30))
        lgb.plot_importance(clf, importance_type=importance_type, max_num_features=max_num_features)
        plt.tight_layout(w_pad=1000)
        mlflow.log_figure(plt.gcf(), "feature_importance.png")
    except Exception as e:
        print(f"Error generating feature importance plot: {e}")

def plot_partial_dependence(clf, X_form, feature: str) -> None:
    try:
        PartialDependenceDisplay.from_estimator(clf, X_form, [feature])
        artifact_name = f"partial_dependence_plot_{feature.replace('num__', '').lower()}.png"
        mlflow.log_figure(plt.gcf(), artifact_name)
        plt.close(plt.gcf())
    except Exception as e:
        print(f"Error generating PDP for {feature}: {e}")

# Get predictions and residuals
def plot_diagnostics(model, y_true, y_pred):

    # Get predictions and residuals
    residuals = y_true - y_pred

    # Create the 4 diagnostic plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Residuals vs Fitted
    axes[0, 0].scatter(y_pred, residuals, alpha=0.5)
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_xlabel('Fitted values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Fitted')

    # 2. Q-Q Plot
    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Normal Q-Q')

    # 3. Scale-Location (Spread-Location)
    standardized_residuals = residuals / residuals.std()
    axes[1, 0].scatter(y_pred, abs(standardized_residuals)**0.5, alpha=0.5)
    axes[1, 0].set_xlabel('Fitted values')
    axes[1, 0].set_ylabel('√|Standardized residuals|')
    axes[1, 0].set_title('Scale-Location')

    # 4. Residuals vs Leverage (using actual vs predicted instead)
    axes[1, 1].scatter(y_true, y_pred, alpha=0.5)
    axes[1, 1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[1, 1].set_xlabel('Actual values')
    axes[1, 1].set_ylabel('Predicted values')
    axes[1, 1].set_title('Actual vs Predicted')

    plt.tight_layout()
    mlflow.log_figure(fig, "regression_diagnostics.png")
    plt.show()