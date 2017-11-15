import sys
sys.path.insert(0, '../config.py')

import config
from pymongo import MongoClient
import pymongo


class BackTestingResults:
    def fetchActualResults(self):
        print('fetching actual results')

    def diffActualVsPredicted(self, actual, predicted):
        print('calculated % diff between actual and predicted results')
