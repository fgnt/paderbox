__author__ = 'raphaelk'
import json
import os

#  function to convert a string containing digits
#  into a string containing the corresponding words
def into_words(ident):
    split = ident.split('_')
    word = split[2]
    words = ""
    for n in range(0,len(word)-1):
        if word[n] == "1":
            words = words + "one "
        elif word[n] == "2":
            words = words + "two "
        elif word[n] == "3":
            words = words + "three "
        elif word[n] == "4":
            words = words + "four "
        elif word[n] == "5":
            words = words + "five "
        elif word[n] == "6":
            words = words + "six "
        elif word[n] == "7":
            words = words + "seven "
        elif word[n] == "8":
            words = words + "eight "
        elif word[n] == "9":
            words = words + "nine "
        elif word[n] == "o":
            words = words + "oh "
        elif word[n] == "z":
            words = words + "zero "
    return words[0:len(words)-1]

data_1_digit = {}
data_2_digit = {}
data_3_digit = {}
data_4_digit = {}
data_5_digit = {}
data_7_digit = {}
data_all_digit = {}
data_words = {}
count = 0
# searching for all files in /net/speechdb/tidigits/tidigits/train
# sort all files(.wav) according to number of spoken digits
# every file gets an ID containing gender,speaker-ID, spoken digits, production number
# those informatin get saved into data_n_digits
# data_words saving all the written digits of a IDs spoken digits part
# finally adding all Objects into one (data_digits)
for root, dirs, files in os.walk("/net/speechdb/tidigits/tidigits/train"):
    for name in files:
        path_with_wav = (root+"/"+name)
        path = path_with_wav.split('.')
        pathsplit = path[0].split('/')
        ident = pathsplit[6]+'_'+ pathsplit[7]+'_'+pathsplit[8]
        #print(ident)
        count = count +1
        data_all_digit[ident] = {"observed":{"ch1":path_with_wav}}
        data_words[ident] = into_words(ident)
        digit_len = (len(pathsplit[8])-1)
        if digit_len == 1:
            data_1_digit[ident] = {"observed":{"ch1":path_with_wav}}
        elif digit_len == 2:
            data_2_digit[ident] = {"observed":{"ch1":path_with_wav}}
        elif digit_len == 3:
            data_3_digit[ident] = {"observed":{"ch1":path_with_wav}}
        elif digit_len == 4:
            data_4_digit[ident] = {"observed":{"ch1":path_with_wav}}
        elif digit_len == 5:
            data_5_digit[ident] = {"observed":{"ch1":path_with_wav}}
        elif digit_len == 7:
            data_7_digit[ident] = {"observed":{"ch1":path_with_wav}}
print (count)
data_digits_train = {"wav": {"1-digit": data_1_digit, "2-digit": data_2_digit, "3-digit": data_3_digit, "4-digit": data_4_digit, "5-digit": data_5_digit, "7-digit": data_7_digit, "All digits": data_all_digit}, }
data_1_digit = {}
data_2_digit = {}
data_3_digit = {}
data_4_digit = {}
data_5_digit = {}
data_7_digit = {}
data_all_digit = {}
data_words = {}
count = 0
# searching for all files in /net/speechdb/tidigits/tidigits/train
for root, dirs, files in os.walk("/net/speechdb/tidigits/tidigits/test"):
    for name in files:
        path_with_wav = (root+"/"+name)
        path = path_with_wav.split('.')
        pathsplit = path[0].split('/')
        ident = pathsplit[6]+'_'+ pathsplit[7]+'_'+pathsplit[8]
        #print(ident)
        count = count +1
        data_all_digit[ident] = {"observed":{"ch1":path_with_wav}}
        data_words[ident] = into_words(ident)
        digit_len = (len(pathsplit[8])-1)
        if digit_len == 1:
            data_1_digit[ident] = {"observed":{"ch1":path_with_wav}}
        elif digit_len == 2:
            data_2_digit[ident] = {"observed":{"ch1":path_with_wav}}
        elif digit_len == 3:
            data_3_digit[ident] = {"observed":{"ch1":path_with_wav}}
        elif digit_len == 4:
            data_4_digit[ident] = {"observed":{"ch1":path_with_wav}}
        elif digit_len == 5:
            data_5_digit[ident] = {"observed":{"ch1":path_with_wav}}
        elif digit_len == 7:
            data_7_digit[ident] = {"observed":{"ch1":path_with_wav}}
print (count)
data_digits_test = {"wav": {"1-digit": data_1_digit, "2-digit": data_2_digit, "3-digit": data_3_digit, "4-digit": data_4_digit, "5-digit": data_5_digit, "7-digit": data_7_digit, "All digits": data_all_digit}, }

# adding all created Objects together
# and creating the .JSON file
data = {"train": {"flists": data_digits_train, },"test": {"flists": data_digits_test, },"orth": {"word": data_words},}
with open('tidigits.json','w') as file:
    json.dump(data, file,indent=4, separators=(', ', ': '),ensure_ascii=False)
