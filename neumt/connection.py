from pymongo import MongoClient

client = MongoClient('mongodb://router.mongo.sprawk.com:27017/')

def connect_mongo() -> MongoClient:
    return client
