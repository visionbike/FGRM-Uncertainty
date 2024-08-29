DATA_NAME="cholecseg8k"
DATA_PATH="./datasets/${DATA_NAME}"
DATA_PATH_RAW="$DATA_PATH/raw"

# extract cholecseg8k dataset
if [ ! -d "${DATA_PATH_RAW}" ]; then
  mkdir -p "${DATA_PATH_RAW}"
  unzip "${DATA_NAME}.zip" -d "${DATA_PATH_RAW}"
fi

# preprocess cholecseg8k dataset
python preprocess_cholecseg8k.py --data_root "${DATA_PATH}" --anno_mode "tissue"
