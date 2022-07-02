import configparser
import sqlite3
from os.path import exists
import pandas as pd
import numpy as np


# to write the log in the file
def write_to_logs(log, log_number):
    # creating log
    log_info = open("logs/log_" + str(log_number) + ".txt", 'a')
    log_info.write(str(log))
    log_info.close()


# This function read the current log_number from the file and write the next log_number into the file
def current_log_number(log_cnt_file_path, update=True):
    # read log_cnt file to check the log sequence
    open_temp_log_cnt = open(log_cnt_file_path, 'r')
    read_temp_log_cnt = open_temp_log_cnt.readline()
    log_number = int(read_temp_log_cnt)
    open_temp_log_cnt.close()

    if update:
        # writing the next log_sequence number in the file
        open_temp = open(log_cnt_file_path, 'w')
        next_log_number = log_number + 1
        open_temp.write(str(next_log_number))
        open_temp.close()

        return next_log_number

    return log_number


# This function read the configuration from the config file
def read_config(config_path, update):
    parser = configparser.ConfigParser()
    parser.read(config_path)
    log_number = 0

    # Debug
    if str(parser['config']['debug']) == "True":

        # reading current log number
        log_number = current_log_number(parser['files']['log_cnt'], update)

        if update:
            return log_number

        write_to_logs("# Printing config file data:\n", log_number)
        write_to_logs("debug: " + str(parser['config']['debug']) + "\n", log_number)
        write_to_logs("log_files: " + str(parser['files']['log_cnt']) + "\n", log_number)

    return parser, log_number


# This function reads the database related names from the config file
def load_db_Config(parser, log_number):
    dbName = parser['db']['db_name'].strip()
    tableList = []
    columnDict = {}

    for table_name in (parser['db']['tables_name']).strip().split(","):
        tableList.append(table_name.strip())

    for col in (parser['db']['columns_header']).strip().split(","):
        colSplit = col.strip().split(":")
        tabName = colSplit[0]
        colName = colSplit[1]

        if tabName not in columnDict:
            columnDict[tabName] = []

        columnDict[tabName].append(colName)

    # Debug
    if str(parser['config']['debug']) == "True":
        # creating log
        log_info = open("logs/log_" + str(log_number) + ".txt", 'a')

        log_info.write("# Printing config file data for database:\n")
        log_info.write("db_name: " + str(parser['db']['db_name']) + "\n")
        log_info.write("tables_name: " + str(parser['db']['tables_name']) + "\n")
        log_info.write("columns_header: " + str(parser['db']['columns_header']) + "\n")
        log_info.write("\n")

        log_info.close()

    return dbName, tableList, columnDict


def create_db(sql_file, db_file, log_number, DEBUG=False):
    if DEBUG:
        write_to_logs("\n", log_number)
        write_to_logs("# Creating the database\n", log_number)
        write_to_logs("# Printing sql details\ndb_file_path: ", log_number)
        write_to_logs(db_file, log_number)
        write_to_logs("\nsql_file_path: ", log_number)
        write_to_logs(sql_file, log_number)
        write_to_logs("\n", log_number)

    file_exists = exists(db_file)
    if file_exists:
        if DEBUG:
            write_to_logs("# " + str(db_file) + " file already exist\n", log_number)
        return
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    sql_file_open = open(sql_file, 'r')
    sql_as_string = sql_file_open.read()

    cursor.executescript(sql_as_string)
    conn.close()

    if DEBUG:
        write_to_logs("# Completed database extraction\n", log_number)


# To get the joint results
def getTable(tmp_qStmt, db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    queryStatement = tmp_qStmt
    cursor.execute(queryStatement)

    df = pd.read_sql(queryStatement, conn)

    # Get the column names
    columnHeaderList = [description[0] for description in cursor.description]
    joinResult = []

    # get the rows of the table
    resultSetSize = 0
    for result_1 in cursor:
        resultDict = {}
        resultSetSize += 1

        for col_i in range(len(columnHeaderList)):
            resultDict[columnHeaderList[col_i]] = result_1[col_i]

        joinResult.append(resultDict)

    return df, joinResult, columnHeaderList


# ### self help functions
def accuracy(y_true, y_pred):
    """
    Function to calculate accuracy
    -> param y_true: list of true values
    -> param y_pred: list of predicted values
    -> return: accuracy score
    """

    # Initializing variable to store count of correctly predicted classes
    correct_predictions = 0

    for yt, yp in zip(y_true, y_pred):

        if yt == yp:
            correct_predictions += 1

    # returns accuracy
    return correct_predictions / len(y_true)


# Functions to compute True Positives, True Negatives, False Positives and False Negatives
def true_positive(y_true, y_pred):
    tp = 0

    for yt, yp in zip(y_true, y_pred):

        if yt == 1 and yp == 1:
            tp += 1

    return tp


def true_negative(y_true, y_pred):
    tn = 0

    for yt, yp in zip(y_true, y_pred):

        if yt == 0 and yp == 0:
            tn += 1

    return tn


def false_positive(y_true, y_pred):
    fp = 0

    for yt, yp in zip(y_true, y_pred):

        if yt == 0 and yp == 1:
            fp += 1

    return fp


def false_negative(y_true, y_pred):
    fn = 0

    for yt, yp in zip(y_true, y_pred):

        if yt == 1 and yp == 0:
            fn += 1

    return fn


# Computation of macro-averaged precision

def macro_precision(y_true, y_pred):
    # find the number of classes
    num_classes = len(np.unique(y_true))

    # initialize precision to 0
    precision = 0

    # loop over all classes
    for class_ in list(np.unique(y_true)):
        # all classes except current are considered negative
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]

        # compute true positive for current class
        tp = true_positive(temp_true, temp_pred)

        # compute false positive for current class
        fp = false_positive(temp_true, temp_pred)

        # compute precision for current class
        temp_precision = tp / (tp + fp + 1e-6)
        # keep adding precision for all classes
        precision += temp_precision

    # calculate and return average precision over all classes
    precision /= num_classes

    return precision


def micro_precision(y_true, y_pred):
    # find the number of classes
    num_classes = len(np.unique(y_true))

    # initialize tp and fp to 0
    tp = 0
    fp = 0

    # loop over all classes
    for class_ in np.unique(y_true):
        # all classes except current are considered negative
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]

        # calculate true positive for current class
        # and update overall tp
        tp += true_positive(temp_true, temp_pred)

        # calculate false positive for current class
        # and update overall tp
        fp += false_positive(temp_true, temp_pred)

    # calculate and return overall precision
    precision = tp / (tp + fp)
    return precision
