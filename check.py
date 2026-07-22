from src.smart_goals.utils.data_loaders import load_store_goals

#load the store goals data
store_goals = load_store_goals()

#print the store goals data
print(store_goals.head())
print(store_goals.tail())
