import sys
import json

from src.test_helper import load_dataset

def main(targets):
    if 'test' in targets:
        with open('config/test_param.json') as fh:
            test_params = json.load(fh)
        load_dataset(**test_params)
        print('Test passed!')

if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)