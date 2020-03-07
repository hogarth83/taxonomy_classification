import wget


def data_loader():
    """
    Load data from Google Drive
    :return:
    Share data into /data folder: predict_paylaod.json, taxonomy_mappings.json, train_data.json
    """

    # load data from google drive
    wget.download('https://docs.google.com/uc?export=download&id=1a0dhmA76EvimwbHCWtcj1J1yqnLRu2Ec', 'data/predict_paylaod.json')
    wget.download('https://docs.google.com/uc?export=download&id=1zSbSvUt-qeiv0JXD3zbmZQiw7xOb22E1', 'data/taxonomy_mappings.json')
    wget.download('https://docs.google.com/uc?export=download&id=1LP7k3aBR34L0UjsvDqUSa7owyYmG9EM7', 'data/train_data.json')


