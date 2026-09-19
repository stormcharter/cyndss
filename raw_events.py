import pymongo
client = pymongo.MongoClient("mongodb://127.0.0.1:27017/?directConnection=true")
client["threat_intel"].drop_collection("raw_events")
print("База ідеально чиста!")
exit()