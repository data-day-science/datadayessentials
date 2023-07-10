from typing import Union
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    """A class to evaluate the outputs of a binary classifier."""

    def __init__(self, model=None):
        self.model = model

    def set_model(self, model):
        self.model = model

    def make_predictions(
        self, y_proba: Union[pd.Series, list, np.array], boundary: float = 0.5
    ) -> np.array:
        """return the class predictions based off the supplied boundary

        Args:
             y_proba (pd.DataFrame): model input data.  Should be a list like object that contains confidence
        predictions from a binary model.
            boundary (float, optional): boundary for class separation. Defaults to 0.5.

        Returns:
            np.array: array of predictions for the given input data
        """

        return np.where(np.array(y_proba) <= boundary, 0, 1)

    def get_probas(self, X: pd.DataFrame) -> np.array:
        """returns the probas for the given input data

        Args:
            X (pd.DataFrame): model input data.  Should be processed and in the correct format for the model to accept

        Returns:
            np.array: array of probas for the given input data
        """
        return self.model.predict_proba(X)[:, 1]

    def calculate_metrics(
        self,
        y_pred: Union[list, np.array],
        y_proba: Union[list, np.array],
        y: Union[list, np.array],
    ) -> dict:
        """calculates the model metrics for the given predictions and truth labels.

        The function calculates:
        1. precision
        2. recall
        3. f1
        4.roc_auc
        5. gini
        6. confusion matrix
        7. false positive rates
        8. true positive rates

        Args:
            y_pred (Union[list,np.array]): the class predictions
            y_proba (Union[list,np.array]): class probabilities
            y (Union[list,np.array]): truth labels

        Returns:
            dict: dictionary of calculated metrics
        """
        false_positve_rates, true_positives_rates, _ = metrics.roc_curve(y, y_proba)
        model_metrics = {
            "precision": metrics.precision_score(y, y_pred),
            "recall": metrics.recall_score(y, y_pred),
            "f1": metrics.f1_score(y, y_pred),
            "roc_auc": metrics.auc(false_positve_rates, true_positives_rates),
            "gini": (2 * (metrics.auc(false_positve_rates, true_positives_rates))) - 1,
            "confusion matrix": metrics.confusion_matrix(y, y_pred),
            "false positive rates": false_positve_rates,
            "true positive rates": true_positives_rates,
        }
        return model_metrics

    def plot_figures(
        self,
        y_pred: Union[list, np.array],
        y_proba: Union[list, np.array],
        y: Union[list, np.array],
        metrics: dict,
        labels: list = ["positive", "negative"],
        boundary: float = 0.5,
    ) -> plt.figure:
        """creates an output plot showing the confusion matrix, class distributions and roc_auc

        Args:
            y_pred (Union[list,np.array]): the class predictions
            y_proba (Union[list,np.array]): class probabilities
            y (Union[list,np.array]): truth labels
            metrics (dict): dictionary of calculated metrics
            labels (list, optional): class labels. Defaults to ["positive", "negative"].
            boundary (float, optional): boundary for class separation. Defaults to 0.5.
        Returns:
            plt.figure: evaluation figure
        """

        tn, fp, fn, tp = [i for i in metrics["confusion matrix"].ravel()]

        # Plot outputs
        fig = plt.figure(figsize=(15, 4))
        # confusion matrix
        plt.subplot(131)
        ax = sns.heatmap(
            metrics["confusion matrix"],
            annot=True,
            cmap="Blues",
            cbar=False,
            annot_kws={"size": 14},
            fmt="g",
        )
        cmlabels = [
            "True Negatives",
            "False Positives",
            "False Negatives",
            "True Positives",
        ]
        for i, t in enumerate(ax.texts):
            t.set_text(t.get_text() + "\n" + cmlabels[i])
        plt.title("Confusion Matrix", size=15)
        plt.xlabel("Predicted Values", size=13)
        plt.ylabel("True Values", size=13)

        # 2 -- Distributions of Predicted Probabilities of both classes
        plt.subplot(132)
        df_probas = pd.DataFrame({"probPos": y_proba.squeeze(), "target": y})
        plt.hist(
            df_probas[df_probas.target == 1].probPos,
            density=True,
            bins=25,
            alpha=0.5,
            color="green",
            label=labels[0],
        )
        plt.hist(
            df_probas[df_probas.target == 0].probPos,
            density=True,
            bins=25,
            alpha=0.5,
            color="red",
            label=labels[1],
        )
        plt.axvline(boundary, color="blue", linestyle="--", label="Boundary")
        plt.xlim([0, 1])
        plt.title("Distributions of Predictions", size=15)
        plt.xlabel("Positive Probability (predicted)", size=13)
        plt.ylabel("Samples (normalized scale)", size=13)
        plt.legend(loc="upper right")

        # 3 -- ROC curve with annotated decision point

        plt.subplot(133)
        plt.plot(
            metrics["false positive rates"],
            metrics["true positive rates"],
            color="green",
            lw=1,
            label="ROC curve (area = %0.2f)" % metrics["roc_auc"],
        )
        plt.plot([0, 1], [0, 1], lw=1, linestyle="--", color="grey")
        # plot current decision point:

        plt.plot(
            fp / (fp + tn),
            tp / (tp + fn),
            "bo",
            markersize=8,
            label="Decision Point",
        )
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", size=13)
        plt.ylabel("True Positive Rate", size=13)
        plt.title("ROC Curve", size=15)
        plt.legend(loc="lower right")
        plt.subplots_adjust(wspace=0.3)
        plt.show()

        return fig

    def print_summary(self, model_metrics: dict):
        """displays some summary information

        Args:
            model_metrics (dict): dictionary of calculated metrics
        """
        print(
            "Precision: {} | Recall: {} F1 Score: {} | Gini Score: {}".format(
                round(model_metrics["precision"], 2),
                round(model_metrics["recall"], 2),
                round(model_metrics["f1"], 2),
                round(model_metrics["gini"], 4),
            )
        )

    def run(
        self, X: pd.DataFrame, y: Union[list, pd.Series, np.array], verbose=True
    ) -> dict:
        """runs the evaluator and returns a figure and summary output

        Args:
            X (pd.DataFrame): model input data.  Should be processed and in the correct format for the model to accept
            y (Union[list,pd.Series,np.array]): class truth labels
            verbose(bool): print outputs

        Returns:
            dict: returns a dictionary of the figure object and the dictionary of metrics
        """
        y_proba = self.get_probas(X)
        y_pred = self.make_predictions(y_proba)
        metrics = self.calculate_metrics(y_pred, y_proba, y)
        if verbose:
            print(y_pred.shape, y_proba.shape)
            fig = self.plot_figures(y_pred, y_proba, y, metrics)
            self.print_summary(metrics)
        else:
            fig = None
        return {"model_performance_figure": fig, "model_metrics": metrics}