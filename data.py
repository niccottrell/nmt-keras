import abc
from pickle import dump, load

from numpy.core.multiarray import array

from helpers import lang2
from pymongo import MongoClient


class DataSource(object):

    @abc.abstractmethod
    def load_clean_sentences(self, subset=None):
        return

    @abc.abstractmethod
    def save_clean_data(self, sentences, subset=None):
        return


class Files(DataSource):

    def get_filename(self, subset=None):
        filename = 'eng-' + lang2
        if subset is not None:
            filename += '-' + subset
        filename += '.pkl'
        return filename

    def load_clean_sentences(self, subset=None):
        """
        load a clean dataset
        :param subset: str
        :return: ndarray
        """
        filename = self.get_filename(subset)
        return load(open(filename, 'rb'))

    # save a list of clean sentences to file
    def save_clean_data(self, sentences, subset=None):
        filename = self.get_filename(subset)
        dump(sentences, open(filename, 'wb'))
        print('Saved {0} to {1}'.format(sentences.shape, filename))


class Mongo(DataSource):

    client = MongoClient('mongodb://localhost:27017/')
    db = client.nmt
    collection = db['eng_' + lang2]

    @staticmethod
    def _chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def save_clean_data(self, sentences):
        # process sentences in chunks
        for chunk in self._chunks(sentences, 1000):
            new_docs = []
            for pair in chunk:
                new_docs.append({'eng': pair[0], lang2: pair[1]})
            # insert this chunk into MongoDB (in order)
                self.collection.insert_many(new_docs)

    def reset_test_flag(self, count):
        # unset test flag
        self.collection.update({}, {'$unset': {'test': 1}}, multi=True)
        # pick random docs and update
        pipeline = [{"$sample": {"size": count}}]
        docs = list(self.collection.aggregate(pipeline))
        for doc in docs:
            # print("Updating doc: %s" % doc['_id'])
            self.collection.update_one({'_id': doc['_id']}, {'$set': {'test': True}})

    def load_clean_sentences(self, subset=None):
        """
        load a clean dataset
        :param subset: str
        :return: ndarray
        """
        if subset is None or subset == 'both':
            cursor = self.collection.find()
        elif subset == 'train':
            cursor = self.collection.find({'test': {'$exists': False}})
        elif subset == 'test':
            cursor = self.collection.find({'test': True})
        else:
            raise ValueError("Unexpected subset value")
        pairs = list()
        for doc in cursor:
            pairs.append([doc['eng'], doc[lang2]])
        return array(pairs)
