import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import json
from supar import Parser
import sys
from tqdm import tqdm
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

def get_index_positions(list_of_elems, element):
    ''' Returns the indexes of all occurrences of give element in
    the list- listOfElements '''
    list_verbs = []
    index_pos = 0
    while True:
        try:
            # Search for item in list from indexPos to the end of list
            index_pos = list_of_elems.index(element, index_pos)
            if list_of_elems[(index_pos+1)] == '(_':
                # Add the index position in list
                list_verbs.append(list_of_elems[(index_pos+2)][:-1])
                index_pos += 1
            else:
                index_pos += 1
            if index_pos == len(list_of_elems):
                break
        except ValueError as e:
            break
    return list_verbs

with open("positive_pair_good_classification.json", "r") as f:
    positive_pair_good_classification = json.loads(f.read())
with open("positive_pair_bad_classification.json", "r") as f:
    positive_pair_bad_classification = json.loads(f.read())
with open("negative_pair_good_classification.json", "r") as f:
    negative_pair_good_classification = json.loads(f.read())
with open("negative_pair_bad_classification.json", "r") as f:
    negative_pair_bad_classification = json.loads(f.read())
f = open('datasets/SVO-Probes/SVO_captions.json')
captions = json.load(f)

text = " ".join(captions[str(number)] for number in positive_pair_good_classification['data'])
word_cloud = WordCloud(collocations = False, background_color = 'white').generate(text)

avg_length = 0
count = 0
for number in positive_pair_good_classification['data']:
    avg_length += len(captions[str(number)].split())
    count += 1
print('positive pair good classification')
print(avg_length/count)

plt.imshow(word_cloud, interpolation='bilinear')
plt.axis("off")
plt.show()
plt.savefig('positive_pair_good_classification.png')

text = " ".join(captions[str(number)] for number in positive_pair_bad_classification['data'])
word_cloud = WordCloud(collocations = False, background_color = 'white').generate(text)

avg_length = 0
count = 0
for number in positive_pair_bad_classification['data']:
    avg_length += len(captions[str(number)].split())
    count += 1
print('positive pair bad classification')
print(avg_length/count)

plt.imshow(word_cloud, interpolation='bilinear')
plt.axis("off")
plt.show()
plt.savefig('positive_pair_bad_classification.png')

text = " ".join(captions[str(number)] for number in negative_pair_good_classification['data'])
word_cloud = WordCloud(collocations = False, background_color = 'white').generate(text)

avg_length = 0
count = 0
for number in negative_pair_good_classification['data']:
    avg_length += len(captions[str(number)].split())
    count += 1
print('negative pair good classification')
print(avg_length/count)

plt.imshow(word_cloud, interpolation='bilinear')
plt.axis("off")
plt.show()
plt.savefig('negative_pair_good_classification.png')

text = " ".join(captions[str(number)] for number in negative_pair_bad_classification['data'])
word_cloud = WordCloud(collocations = False, background_color = 'white').generate(text)

avg_length = 0
count = 0
for number in negative_pair_bad_classification['data']:
    avg_length += len(captions[str(number)].split())
    count += 1
print('negative pair bad classification')
print(avg_length/count)

plt.imshow(word_cloud, interpolation='bilinear')
plt.axis("off")
plt.show()
plt.savefig('negative_pair_bad_classification.png')

wnl = WordNetLemmatizer()
print('Parsing tree')
verbs_positive_pair_good_classification= []
parser = Parser.load('crf-con-en')
for i in tqdm(positive_pair_good_classification['data']):
    caption = captions[str(i)].split()
    dataset = parser.predict(caption, verbose=False)
    list_verbs = get_index_positions(str(dataset[0]).split(), '(VP')
    verbs_positive_pair_good_classification += list_verbs

text = " ".join(wnl.lemmatize(verb, 'v') for verb in verbs_positive_pair_good_classification)
word_cloud = WordCloud(collocations = False, background_color = 'white').generate(text)

plt.imshow(word_cloud, interpolation='bilinear')
plt.axis("off")
plt.show()
plt.savefig('positive_pair_good_classification_verbs.png')

print('Parsing tree')
verbs_positive_pair_bad_classification= []
for i in tqdm(positive_pair_bad_classification['data']):
    caption = captions[str(i)].split()
    dataset = parser.predict(caption, verbose=False)
    list_verbs = get_index_positions(str(dataset[0]).split(), '(VP')
    verbs_positive_pair_bad_classification += list_verbs

text = " ".join(wnl.lemmatize(verb, 'v') for verb in verbs_positive_pair_bad_classification)
word_cloud = WordCloud(collocations = False, background_color = 'white').generate(text)

plt.imshow(word_cloud, interpolation='bilinear')
plt.axis("off")
plt.show()
plt.savefig('positive_pair_bad_classification_verbs.png')

print('Parsing tree')
verbs_negative_pair_good_classification= []
for i in tqdm(negative_pair_good_classification['data']):
    caption = captions[str(i)].split()
    dataset = parser.predict(caption, verbose=False)
    list_verbs = get_index_positions(str(dataset[0]).split(), '(VP')
    verbs_negative_pair_good_classification += list_verbs

text = " ".join(wnl.lemmatize(verb, 'v') for verb in verbs_negative_pair_good_classification)
word_cloud = WordCloud(collocations = False, background_color = 'white').generate(text)

plt.imshow(word_cloud, interpolation='bilinear')
plt.axis("off")
plt.show()
plt.savefig('negative_pair_good_classification_verbs.png')

print('Parsing tree')
verbs_negative_pair_bad_classification= []
for i in tqdm(negative_pair_bad_classification['data']):
    caption = captions[str(i)].split()
    dataset = parser.predict(caption, verbose=False)
    list_verbs = get_index_positions(str(dataset[0]).split(), '(VP')
    verbs_negative_pair_bad_classification += list_verbs

text = " ".join(wnl.lemmatize(verb, 'v') for verb in verbs_negative_pair_bad_classification)
word_cloud = WordCloud(collocations = False, background_color = 'white').generate(text)

plt.imshow(word_cloud, interpolation='bilinear')
plt.axis("off")
plt.show()
plt.savefig('negative_pair_bad_classification_verbs.png')