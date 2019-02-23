"""
This module will download the SQLite version of the ChEMBL database if it
doesn't exist in ../data, and use it to create the data sets for analysis.
"""

import os
import sqlite3
import tarfile
import urllib.request


class ChEMBL_SQLite(object):
    """ChEMBL data http://www.ebi.ac.uk/chembl version chembl_24_1.

    Args:
        path (string, optional): Where the data will be downloaded. Defaults to
            the `data` directory of code.
    """
    url = "ftp://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_24_1/"
    filename = "chembl_24_1_sqlite.tar.gz"
    dbpath = "chembl_24/chembl_24_sqlite/chembl_24.db"

    def __init__(self, path="../data/"):
        self.path = path

    def db_connect(self):
        """Returns a connection to the ChEMBL database,
        will download if it does not exist.

        Returns:
            conn (sqlite3.Connection): The connection to the database
        """
        if not os.path.isfile(self.path + self.dbpath):
            self._download()

        conn = sqlite3.connect(self.path + self.dbpath)
        return conn

    def _download(self):
        """Downloads the ChEMBL database if it doesn't exist"""
        if not os.path.isfile(self.path + self.dbpath):
            delete_tar = False

            if not os.path.isfile(self.path + self.filename):
                delete_tar = True
                print("Downloading ChEMBL database")
                file, _ = urllib.request.urlretrieve(self.url + self.filename,
                                                     self.path + self.filename)

            print("Extracting tarfile")
            tarfile.open(self.path + self.filename).extractall(path=self.path)
            if delete_tar:
                os.remove(self.path + self.filename)
