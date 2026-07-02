import os

SEED = 42


# Main Paths
# Percorso assoluto del file constants.py
current_file_path = os.path.abspath(__file__)

# Percorso della cartella 'src'
src_folder_path = os.path.dirname(os.path.dirname(current_file_path))

# Percorso della cartella 'data'
data_folder_path = os.path.join(src_folder_path, '..', 'data')

# Percorsi per le sottocartelle 'raw' e 'processed' dentro 'data'
DATA_RAW_FOLDER_PATH = os.path.join(data_folder_path, 'raw')
DATA_PROCESSED_FOLDER_PATH = os.path.join(data_folder_path, 'processed')

# shapefile
SHAPEFILE_FOLDER_PATH = os.path.join(data_folder_path, 'shapefile')
SHAPEFILE_REGIONS_FILE_PATH = os.path.join(SHAPEFILE_FOLDER_PATH, 'georef-italy-regione-millesime.shp')


# Raw Data
TBL_ADDRESSES_PATH = os.path.join(DATA_RAW_FOLDER_PATH, 'tbl_addresses.csv')
TBL_CUSTOMER_ACCOUNTS_PATH = os.path.join(DATA_RAW_FOLDER_PATH, 'tbl_customer_accounts.csv')
TBL_CUSTOMER_REVIEWS_PATH = os.path.join(DATA_RAW_FOLDER_PATH, 'tbl_customer_reviews.csv')
TBL_CUSTOMERS_PATH = os.path.join(DATA_RAW_FOLDER_PATH, 'tbl_customers.csv')
TBL_LABELLED_REVIEWS_PATH = os.path.join(DATA_RAW_FOLDER_PATH, 'tbl_labelled_reviews.csv')
TBL_ORDERS_PATH = os.path.join(DATA_RAW_FOLDER_PATH, 'tbl_orders.csv')
TBL_PRODUCTS_PATH = os.path.join(DATA_RAW_FOLDER_PATH, 'tbl_products.csv')

# Processed Data
TBL_CUSTOMER_REVIEWS_PROCESSED_PATH = os.path.join(DATA_PROCESSED_FOLDER_PATH, 'tbl_customer_review_preprocessed.csv')
TBL_LABELLED_REVIEWS_PROCESSED_PATH = os.path.join(DATA_PROCESSED_FOLDER_PATH, 'tbl_labelled_review_preprocessed.csv')

# Models
models_folder_path = os.path.join(src_folder_path, '..', 'models')
BEST_MODEL_SENTIMENT_PATH = os.path.join(models_folder_path, 'pipeline_logit_final.pkl')


