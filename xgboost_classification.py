from utils import read_config, write_to_logs, load_db_Config, create_db, getTable, accuracy, true_negative
from utils import true_positive, false_positive, false_negative, macro_precision, micro_precision
import xgboost
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder

# parser contain all the configuration details
# log_number is the current log dump file number
parser, log_number = read_config("config.txt", False)

# for debugging
DEBUG = False
if parser['config']['debug'] == "True":
    DEBUG = True

if DEBUG:
    write_to_logs("\n# XGBoost Version: " + str(xgboost.__version__) + "\n", log_number)

# ############## main #############

# dbName : contain the name of the database
# tableList : list of all the tables in the database(dbName)
# columnHeaderDict : dictionary of tables and its column names. {table1 : [col1, col2,..], table2 : [...], ...}
dbName, tableList, columnHeaderDict = load_db_Config(parser, log_number)

# Generating sql query
qStmt = "Select * from "

for tab in tableList:
    qStmt += tab + ' natural join '

# Removing the postfix ( "natural join" ) from qStmt
qStmt = qStmt[:qStmt.rfind('natural join ')]
# Adding a condition to the qStmt
qStmt += "where rank != 'None'; "

if DEBUG:
    write_to_logs("# Printing the sql query:\n", log_number)
    write_to_logs("qStmt: ", log_number)
    write_to_logs(qStmt, log_number)
    write_to_logs("\n", log_number)

# creating the database using imdb.sql dump.
create_db(parser['files']['sql_file'], parser['files']['db_file'], log_number, DEBUG)

# results: List of rows(in dictionary format) [{col1:val,col2:val,..},{col1:val,col2:val,..},...]
# resultHeaderList: List of all the column names
df, result, resultHeaderList = getTable(qStmt, parser['files']['db_file'])

if DEBUG:
    write_to_logs("\n", log_number)
    write_to_logs("# SQL table header:", log_number)
    write_to_logs(resultHeaderList, log_number)
    write_to_logs("\n", log_number)
    write_to_logs("# SQL table (first 5 rows):\n", log_number)
    for i in range(5):
        write_to_logs(result[i], log_number)
        write_to_logs("\n", log_number)
    write_to_logs("\n", log_number)

    write_to_logs("df:\n", log_number)
    write_to_logs(str(df), log_number)
    write_to_logs("\n", log_number)

# Finding the missing cell percentage in each column
# no need to find the missing value as there is no missing value in this dataset.
# missing_props = df.isna().mean(axis=0)
# print(missing_props)

# Shuffling the dataframe
shuffled = df.sample(frac=1).reset_index()

if DEBUG:
    write_to_logs("\n# Shuffled dataframe:", log_number)
    write_to_logs("\n" + str(shuffled), log_number)

# separating the feature and the target
X = shuffled.drop("movies_genre", axis=1)
y = shuffled.movies_genre

categorical_pipeline = Pipeline(
    steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("oh-encode", OneHotEncoder(handle_unknown="ignore", sparse=False)),
    ]
)

numeric_pipeline = Pipeline(
    steps=[("impute", SimpleImputer(strategy="mean")),
           ("scale", StandardScaler())]
)

cat_cols = X.select_dtypes(exclude="number").columns
num_cols = X.select_dtypes(include="number").columns

if DEBUG:
    write_to_logs("\n# cat_cols: " + str(cat_cols) + "\n", log_number)
    write_to_logs("\n# num_cols: " + str(num_cols) + "\n", log_number)

full_processor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, num_cols),
        ("categorical", categorical_pipeline, cat_cols),
    ]
)

# Apply preprocessing
X_processed = full_processor.fit_transform(X)

y_processed = SimpleImputer(strategy="most_frequent").fit_transform(
    y.values.reshape(-1, 1)
)

# print(y_processed.reshape(1,-1))
le = LabelEncoder()
y_processed_ = le.fit_transform(y_processed.reshape(1, -1)[0])

# print(y_processed_.reshape(-1,1))
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y_processed_.reshape(-1, 1), stratify=y_processed_, random_state=1121218
)

# y_train = y_train.reshape(-1,1)
# y_test = y_test.reshape(-1,1)
# print(y_train.shape)


if DEBUG:
    write_to_logs("\n# Test - Train Sample:", log_number)

    write_to_logs("\nTrain X:\n" + str(X_train), log_number)
    write_to_logs("\nTrain y:\n" + str(y_train), log_number)

    write_to_logs("\nTest X:\n" + str(X_test), log_number)
    write_to_logs("\nTest y:\n" + str(y_test), log_number)

# Init classifier
xgb_cl = XGBClassifier()

# print(y_train)
# Fit
xgb_cl.fit(X_train, y_train)

# Predict
preds = xgb_cl.predict(X_test)

# Score
score = accuracy_score(y_test, preds)
print("Accuracy @1 using sklearn library: " + str(score))

acc = accuracy(y_test, preds)
print("Accuracy: " + str(acc))

tp = true_positive(y_test, preds)
print("True Positive: " + str(tp))

tn = true_negative(y_test, preds)
print("True Negative: " + str(tn))

fp = false_positive(y_test, preds)
print("False Positive: " + str(fp))

fn = false_negative(y_test, preds)
print("False Negative: " + str(fn))

macro_prec = precision_score(y_test, preds, average='macro')
print("Macro-Averaged Precision using sklearn library : " + str(macro_prec))

macro_prec_implement = macro_precision(y_test, preds)
print("Macro-Averaged Precision: " + str(macro_prec_implement))

micro_prec = precision_score(y_test, preds, average='micro')
print("Micro-Averaged Precision using sklearn library : " + str(micro_prec))

micro_prec_implement = micro_precision(y_test, preds)
print("Micro-Averaged Precision: " + str(micro_prec_implement))

if DEBUG:
    write_to_logs("\nAccuracy @1 using sklearn library: " + str(score), log_number)
    write_to_logs("\n# Accuracy @1: " + str(score), log_number)
    write_to_logs("\n# True Positive: " + str(tp), log_number)
    write_to_logs("\n# True Negative: " + str(tn), log_number)
    write_to_logs("\n# False Positive: " + str(fp), log_number)
    write_to_logs("\n# False Negative: " + str(fn), log_number)
    write_to_logs("\n# Macro-Averaged Precision using sklearn library : " + str(macro_prec), log_number)
    write_to_logs("\n# Macro-Averaged Precision: " + str(macro_prec_implement), log_number)
    write_to_logs("\n# Micro-Averaged Precision using sklearn library : " + str(micro_prec), log_number)
    write_to_logs("\n# Micro-Averaged Precision: " + str(micro_prec_implement), log_number)

precision, recall, F1 = precision_recall_fscore_support(y_test, preds, beta=1.0, average='weighted')
print("Weighted")
print("Precision: " + str(precision))
print("Recall: " + str(recall))
print("F1 Score: " + str(F1))

if DEBUG:
    write_to_logs("\n# Weighted:", log_number)
    write_to_logs("\nPrecision: " + str(precision), log_number)
    write_to_logs("\nRecall: " + str(recall), log_number)
    write_to_logs("\nF1 Score: " + str(F1), log_number)

precision, recall, F1 = precision_recall_fscore_support(y_test, preds, beta=1.0, average='macro')
print("macro")
print("Precision: " + str(precision))
print("Recall: " + str(recall))
print("F1 Score: " + str(F1))

if DEBUG:
    write_to_logs("\n# macro:", log_number)
    write_to_logs("\nPrecision: " + str(precision), log_number)
    write_to_logs("\nRecall: " + str(recall), log_number)
    write_to_logs("\nF1 Score: " + str(F1), log_number)

precision, recall, F1 = precision_recall_fscore_support(y_test, preds, beta=1.0, average='micro')
print("micro")
print("Precision: " + str(precision))
print("Recall: " + str(recall))
print("F1 Score: " + str(F1))

if DEBUG:
    write_to_logs("\n# micro:", log_number)
    write_to_logs("\nPrecision: " + str(precision), log_number)
    write_to_logs("\nRecall: " + str(recall), log_number)
    write_to_logs("\nF1 Score: " + str(F1), log_number)

# ############# end main ############

next_log_number = read_config("config.txt", True)
if DEBUG:
    write_to_logs("\n# Current log number is: " + str(log_number), log_number)
    write_to_logs("\n# Next log number is: " + str(next_log_number), log_number)
