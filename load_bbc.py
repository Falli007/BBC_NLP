import pandas as pd

# create function to load data
def load_data(article_amount: int, category: str):
    rows = [] # create empty list
    for i in range(1, article_amount + 1): # use a for loop to iterate over the file names
        if i < 10:
            filename = f'data/{category}/00{i}.txt'
        elif i < 100:
            filename = f'data/{category}/0{i}.txt'
        else:
            filename = f'data/{category}/{i}.txt'
        print(f"Processing {filename}") # print the file name
        with open(filename, encoding='utf-8') as f: # open the file
            text = f.read()
            rows.append({'Category': category, 'Text': text, 'Filename': filename, 'Subcategory': None}) # add the file to the list 
    return rows # return the list

business = []
entertainment = []
politics = []
sport = []
tech = []
# load data
business.extend(load_data(510, "business"))
entertainment.extend(load_data(386, "entertainment"))
politics.extend(load_data(417, "politics"))
sport.extend(load_data(511, "sport"))
tech.extend(load_data(401, "tech"))


def save_to_csv(rows: list, category: str):
    categoryframe = pd.DataFrame(rows, columns=['Category','Text', 'Filename', 'Subcategory'])
    categoryframe.to_csv(f'data/{category}.csv', index=False)

save_to_csv(business, "business")
save_to_csv(entertainment, "entertainment")
save_to_csv(politics, "politics")
save_to_csv(sport, "sport")
save_to_csv(tech, "tech")