from sqlalchemy import create_engine
from config import *
import sys


def load_data(messages_filepath, categories_filepath):
    """
    Merge two dataset from sources into one dataset

    :param messages_filepath: path to the messages data source
    :param categories_filepath: path to the categories data source
    :return: a merged dataset (pandas.DataFrame)
    """
    df_msgs = pd.read_csv(messages_filepath)
    df_categories = pd.read_csv(categories_filepath)
    df = df_msgs.merge(df_categories, on="id")  # merge two df based on 'id'
    return df


def clean_data(df):
    """
    Clean the data (remove duplicates, fill NaN, change the format ...)

    :param df: raw data (pandas.DataFrame)
    :return: a cleaned dataset (pandas.DataFrame)
    """
    categories = df["categories"].str.split(";", expand=True)
    columns = categories.iloc[0, :].values  # it doesn't matter which row

    # remove the last two chars (f.e. "-1", "-0")
    new_cols = [col[:-2] for col in columns]
    categories.columns = new_cols

    # change the categories in number
    # loop through all columns
    for col in categories:
        categories[col] = categories[col].str[-1]  # get the last char
        categories[col] = pd.to_numeric(categories[col])  # change the char in number
    df.drop("categories", axis=1, inplace=True)  # drop the raw column
    df[categories.columns] = categories  # add the categories columns into the df
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    """
    Save the dataset into database

    :param df: dataset to save (pandas.DataFrame)
    :param database_filename: path to save
    :return: None
    """
    engine = create_engine("sqlite:///{}".format(database_filename))
    db_file_name = database_filename.split("/")[-1]
    table_name = db_file_name.split(".")[0]
    df.to_sql(table_name, engine, index=False, if_exists="replace")


def process_save_data():
    """
    main processing function
    """
    logger.info('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'.format(messages_filepath, categories_filepath))
    df = load_data(messages_filepath, categories_filepath)

    logger.info('Cleaning data...')
    df = clean_data(df)

    logger.info('Saving data...\n    DATABASE: {}'.format(database_filepath))
    save_data(df, database_filepath)

    logger.info('Cleaned data saved to database!')


if __name__ == "__main__":
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        process_save_data()
    else:
        logger.error(
            'Please provide the filepaths of the messages and categories '
            'datasets as the first and second argument respectively, as '
            'well as the filepath of the database to save the cleaned data '
            'to as the third argument. \n\nExample: python process_data.py '
            'disaster_messages.csv disaster_categories.csv '
            'DisasterResponse.db'
        )
