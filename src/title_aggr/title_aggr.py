import re
import yaml
import os
import pandas as pd
import pickle as pkl
from collections import Counter

path = '/data_process/title_aggr/'

class TitleAggr(object):
    def __init__(self, grain='fun'):
        with open(path + 'replace_dict_1.yaml', 'r') as f:
            self.replace_dict_1 = yaml.load(f, Loader=yaml.FullLoader)
        with open(path + 'replace_dict_2.yaml', 'r') as f:
            self.replace_dict_2 = yaml.load(f, Loader=yaml.FullLoader)

        with open(path + 'IT_RES', "r") as f:
            it_res = f.readlines()
        with open(path + 'FI_RES', "r") as f:
            fi_res = f.readlines()
        self.res = set([x.strip() for x in it_res+fi_res])

        with open(path + 'IT_FUN', "r") as f:
            it_fun = f.readlines()
        with open(path + 'FI_FUN', "r") as f:
            fi_fun = f.readlines()
        self.fun = set([x.strip() for x in it_fun+fi_fun])

        with open(path + 'LEV', "r") as f:
            lev = f.readlines()
        self.lev = set([x.strip() for x in lev])

        self.grain = grain

    @staticmethod
    def split_title(title):
        """ delete punctuation marks 

        Args:
            title (_type_): _description_

        Returns:
            _type_: _description_
        """
        pattern = r',|\.|/|;|\'|`|\[|\]|<|>|\?|:|"|\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|-|=|\_|\+|，|。|、|；|‘|’|【|】|·|！| |…|（|）'
        title_split = re.split(pattern, title)
        while '' in title_split:
            title_split.remove('')
        return title_split

    @staticmethod
    def get_alpha_str(s):
        """ Extract the English Alphabets

        Args:
            s (_type_): _description_

        Returns:
            _type_: _description_
        """
        result = re.split(r'[^A-Za-z]', s)
        while '' in result:
            result.remove('')

        result = ' '.join(result)
        return result


    def is_title(self, title):
        """ Identify legal title names

        Args:
            title (_type_): _description_

        Returns:
            _type_: _description_
        """
        p = 0
        re_li = list()

        for item in title.split(' '):
            if item in self.res | self.lev | self.fun and self.grain == 'fun':              # use FUN or not
                # if item in res | lev:
                re_li.append(item)
                if item in self.res:
                    p = 1
            if item in self.res | self.lev and self.grain == 'res':
                re_li.append(item)
                if item in self.res:
                    p = 1

        if p == 0:
            return ''
        else:
            return ' '.join(sorted(set(re_li)))


    def filter_title(self, title):
        if '(' and ')' in title:
            title = re.sub(u" \\(.*?\\)|\\[.*?]|\\{.*?}", "", title)

        for word in self.replace_dict_1.keys():
            if word in title:
                title = title.replace(word, self.replace_dict_1[word])

        title = self.get_alpha_str(title)

        title = title.lower()
        title = ' '.join([self.replace_dict_2[word] if word in self.replace_dict_2.keys() else word for word in self.split_title(title)])
        title = self.is_title(title)

        return title


if __name__ == "__main__":
    cleaner = TitleAggr('fun')
    print(cleaner.filter_title('Test Manager'))
