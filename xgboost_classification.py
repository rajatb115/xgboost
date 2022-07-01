from utils import read_config, write_to_logs, load_db_Config, create_db, getTable
import xgboost
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
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

le = LabelEncoder()
y_processed_ = le.fit_transform(y_processed)

X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y_processed_, stratify=y_processed_, random_state=1121218
)

if DEBUG:
    write_to_logs("\n# Test - Train Sample:", log_number)

    write_to_logs("\nTrain X:\n" + str(X_train), log_number)
    write_to_logs("\nTrain y:\n" + str(y_train), log_number)

    write_to_logs("\nTest X:\n" + str(X_test), log_number)
    write_to_logs("\nTest y:\n" + str(y_test), log_number)

# Init classifier
xgb_cl = XGBClassifier()

# Fit
xgb_cl.fit(X_train, y_train)

# Predict
preds = xgb_cl.predict(X_test)

# Score
score = accuracy_score(y_test, preds)
print(score)

if DEBUG:
    write_to_logs("\n# SCORE: " + str(score), log_number)

# ############# end main ############

next_log_number = read_config("config.txt", True)
if DEBUG:
    write_to_logs("\n# Current log number is: " + str(log_number), log_number)
    write_to_logs("\n# Next log number is: " + str(next_log_number), log_number)

