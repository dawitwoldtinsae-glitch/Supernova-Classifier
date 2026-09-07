
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

def logistic_regression(**kwargs):
    """this function will accept some parameters and will return model of logistic regression
    with proper pipeline.
    parameter:
             **kwargs: some additional arguments that will be used.
    return:
          unfitted model of logistic regression"""
    pipeline_logistic_regression=Pipeline([('scaler',StandardScaler()),
                                         ('logistic_regression',LogisticRegression(random_state=0,**kwargs))])
    return pipeline_logistic_regression
def decision_tree(**kwargs):
    """
    this function will accept some parameters and will return the pipelined
    DecisionTreeClassifier.
    :param kwargs: some additional parameters used in the model
    :return: unfitted pipeline
    """
    pipeline_decision_tree=Pipeline([('decision_tree',DecisionTreeClassifier(random_state=0,**kwargs))])
    return pipeline_decision_tree
def knn(**kwargs):
    """
    this function will accept some parameters and will return the pipelined
    KNeighborsClassifier.
    :param kwargs: some additional parameters used in the model
    :return: unfitted pipeline
    """
    pipeline_knn=Pipeline([
        ('scaler',StandardScaler()),('knn',KNeighborsClassifier(**kwargs))
    ])
    return pipeline_knn
def random_forest(**kwargs):
    """
        this function will accept some parameters and will return the pipelined
        RandomForestClassifier.
        :param kwargs: some additional parameters used in the model
        :return: unfitted pipeline
        """
    pipeline_random_forest=Pipeline([
        ('random_forest',RandomForestClassifier(random_state=0,**kwargs))
    ])
    return pipeline_random_forest