import argparse

def get_training_args():
    parser = argparse.ArgumentParser(
        description = 'Training Image Classifier',
    )

    parser.add_argument('data_dir', action='store')
    parser.add_argument('--save_dir', action='store', dest='save_dir', default='.')
    parser.add_argument('--arch', action='store', dest='arch')
    parser.add_argument('--learning_rate', action='store', dest='lr', type=float, default=0.001)
    parser.add_argument('--hidden_units', action='store', dest='hidden_uniits', type=int, default=4096)
    parser.add_argument('--epochs', action='store', dest='epochs', type=int, default=1)
    parser.add_argument('--gpu', action="store_true", dest='gpu', default=False)

    return parser

def get_prediction_args():
    parser = argparse.ArgumentParser(
        description = 'Predict Image',
    )

    parser.add_argument('img_dir', action='store')
    parser.add_argument('checkpoint', action='store')
    parser.add_argument('--top_k', action="store", dest='top_k', type=int, default=1)
    parser.add_argument('--category_names', action='store', dest='category_file', default=None)
    parser.add_argument('--gpu', action="store_true", dest='gpu', default=False)

    return parser
