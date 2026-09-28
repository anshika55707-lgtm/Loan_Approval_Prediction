from flask import Flask, render_template, request
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

app = Flask(__name__)


# =========================================================
# LOAD AND PREPARE DATA
# =========================================================

df = pd.read_csv("data/loan_approval_dataset.csv")

df = df.drop_duplicates()

df = df.drop(columns=["Loan ID", "Customer ID"])


# =========================================================
# ENCODE CATEGORICAL COLUMNS
# =========================================================

encoders = {}

categorical_columns = df.select_dtypes(
    include=["object"]
).columns

for column in categorical_columns:

    encoder = LabelEncoder()

    df[column] = encoder.fit_transform(
        df[column].astype(str)
    )

    encoders[column] = encoder


# =========================================================
# FEATURES AND TARGET
# =========================================================

X = df.drop("Loan_Approved", axis=1)

y = df["Loan_Approved"]


# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =========================================================
# RANDOM FOREST MODEL FOR PREDICTION
# =========================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


# =========================================================
# CLASSIFICATION MODELS
# =========================================================

classification_models = {

    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "Decision Tree":
        DecisionTreeClassifier(
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),

    "KNN":
        KNeighborsClassifier()
}


classification_results = []


for name, classifier in classification_models.items():

    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    classification_results.append({

        "Algorithm": name,

        "Accuracy":
            round(
                accuracy_score(y_test, y_pred) * 100,
                2
            ),

        "Precision":
            round(
                precision_score(
                    y_test,
                    y_pred,
                    zero_division=0
                ) * 100,
                2
            ),

        "Recall":
            round(
                recall_score(
                    y_test,
                    y_pred,
                    zero_division=0
                ) * 100,
                2
            ),

        "F1-Score":
            round(
                f1_score(
                    y_test,
                    y_pred,
                    zero_division=0
                ) * 100,
                2
            )
    })


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():

    return render_template("home.html")


# =========================================================
# PREDICTION PAGE
# =========================================================

@app.route("/prediction")
def prediction():

    return render_template("index.html")


# =========================================================
# VISUALIZATION PAGE
# =========================================================

@app.route("/visualization")
def visualization():

    return render_template(
        "visualization.html"
    )


# =========================================================
# CLASSIFICATION PAGE
# =========================================================

@app.route("/classification")
def classification():

    return render_template(
        "classification.html",
        results=classification_results
    )


# =========================================================
# PREPROCESSING PAGE
# =========================================================

@app.route("/preprocessing")
def preprocessing():

    return render_template(
        "preprocessing.html"
    )


# =========================================================
# LOAN PREDICTION
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        input_data = {}


        # -------------------------------------------------
        # GET INPUT VALUES
        # -------------------------------------------------

        for column in X.columns:

            value = request.form[column]


            # Categorical column
            if column in encoders:

                value = encoders[column].transform(
                    [value]
                )[0]


            # Numerical column
            else:

                value = float(value)


            input_data[column] = value


        # -------------------------------------------------
        # CREATE INPUT DATAFRAME
        # -------------------------------------------------

        input_df = pd.DataFrame(
            [input_data]
        )


        # -------------------------------------------------
        # PREDICTION
        # -------------------------------------------------

        prediction = model.predict(
            input_df
        )[0]


        # -------------------------------------------------
        # PREDICTION PROBABILITY
        # -------------------------------------------------

        probability = (
            model.predict_proba(input_df)[0][prediction]
            * 100
        )

        probability = round(
            probability,
            2
        )


        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        if prediction == 1:

            result = "Loan Approved"

        else:

            result = "Loan Not Approved"


        # -------------------------------------------------
        # SEND RESULT TO HTML
        # -------------------------------------------------

        return render_template(

            "result.html",

            prediction=result,

            probability=probability

        )


    except Exception as e:

        return render_template(

            "result.html",

            prediction="Error: " + str(e),

            probability=None

        )


# =========================================================
# RUN FLASK APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(debug=True)