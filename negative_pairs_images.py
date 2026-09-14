import json

with open("negative_pair_good_classification_img.json", "r") as f:
    negative_pair_good_classification_img = json.loads(f.read())
with open("negative_pair_bad_classification_img.json", "r") as f:
    negative_pair_bad_classification_img = json.loads(f.read())

frequency_good = {}

# iterating over the list
for item in negative_pair_good_classification_img['data']:
   # checking the element in dictionary
   if item in frequency_good:
      # incrementing the counr
      frequency_good[item] += 1
   else:
      # initializing the count
      frequency_good[item] = 1

# printing the frequency
good_images = {}
for (key,item) in frequency_good.items():
    if item > 6:
        good_images[key] = item

frequency_bad = {}

# iterating over the list
for item in negative_pair_bad_classification_img['data']:
   # checking the element in dictionary
   if item in frequency_bad:
      # incrementing the counr
      frequency_bad[item] += 1
   else:
      # initializing the count
      frequency_bad[item] = 1

# printing the frequency
bad_images = {}
for (key,item) in frequency_bad.items():
    if item > 6:
        bad_images[key] = item

with open("negative_pair_good_classification.json", "r") as f:
    negative_pair_good_classification = json.loads(f.read())
with open("negative_pair_bad_classification.json", "r") as f:
    negative_pair_bad_classification = json.loads(f.read())

print('good')
good_images_caption = {}
is_it_bad = {}
comparison_good_bad = {}
for key in good_images.keys():
    indexes = [i for i, e in enumerate(negative_pair_good_classification_img['data']) if e == key]
    values = [negative_pair_good_classification['data'][i] for i in indexes]
    good_images_caption[key] = values
    if key in negative_pair_bad_classification_img['data']:
        is_it_bad[key] = True
    else:
        is_it_bad[key] = False
    if is_it_bad[key] == True:
        indexes = [i for i, e in enumerate(negative_pair_bad_classification_img['data']) if e == key]
        values = [negative_pair_bad_classification['data'][i] for i in indexes]
        comparison_good_bad[key] = values
print(good_images)
print(good_images_caption)
print(is_it_bad)
print(comparison_good_bad)


print('bad')
bad_images_caption = {}
is_it_good = {}
comparison_bad_good = {}
for key in bad_images.keys():
    indexes = [i for i, e in enumerate(negative_pair_bad_classification_img['data']) if e == key]
    values = [negative_pair_bad_classification['data'][i] for i in indexes]
    bad_images_caption[key] = values
    if key in negative_pair_good_classification_img['data']:
        is_it_good[key] = True
    else:
        is_it_good[key] = False
    if is_it_good[key] == True:
        indexes = [i for i, e in enumerate(negative_pair_good_classification_img['data']) if e == key]
        values = [negative_pair_good_classification['data'][i] for i in indexes]
        comparison_bad_good[key] = values
print(bad_images)
print(bad_images_caption)
print(is_it_good)
print(comparison_bad_good)