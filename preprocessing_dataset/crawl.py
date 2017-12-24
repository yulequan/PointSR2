from bs4 import BeautifulSoup
import urllib2
import re
import os
from tqdm import tqdm
from subprocess import Popen,PIPE,STDOUT


def get_single_category(category=7, name=None, page=None, save_root_path='.'):
    category_links = []
    if page!=None:
        pages = [page]
    else:
        pages = range(1,50,24)
    for pg in pages:
        url = "https://archive3d.net/?category=%d&page=%d"%(category,pg)
        print url
        f = urllib2.urlopen(url)
        soup = BeautifulSoup(f.read(),"lxml")
        tags = soup.find_all('a',href=re.compile('a=download'))
        if len(tags)==0:
            break
        links = []
        for item in tags:
            id = item['href'].split('&')[-1][3:]
            link = "https://archive3d.net/?a=download&do=get&id="+id
            links.append(link)
        category_links.extend(links)
    print "Total %d links in this category"%len(category_links)
    
    if name is not None:
        category = str(category)+'_'+name
    else:
        category = str(category)

    save_path = save_root_path+'/'+ category
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    id = -1
    for link in tqdm(category_links):
        id = id +1
        zipfile = '%s/%s_%d.zip' % (save_path, category, id)
        cmd1 = """ wget '%s' -O %s""" % (link, zipfile)
        cmd2 = """ unzip -o %s -d %s""" % (zipfile, zipfile[:-4])
        cmd3 = """ mkdir %s;unrar e -o+ %s %s""" % (zipfile[:-4],zipfile,zipfile[:-4])
        # print cmd
        sts1 = Popen(cmd1, shell=True).wait()
        sts2 = Popen(cmd2, shell=True).wait()
        if sts2:
            print "Use unrar"
            sts3 = Popen(cmd3, shell=True).wait()
        else:
            print "Use unzip"

def get_multi_category():
    save_root_path = '/home/lqyu/data'
    if not os.path.exists(save_root_path):
        os.makedirs(save_root_path)

    category_list = {'tool': 8,'KitchenEquipment':436}
    category_list = {'HomeApplicances':222,'Educational':2095,'OfficeEquipment': 223}
    for (name, id) in category_list.items():
        print "handle catogery: %s" % name
        get_single_category(id, name, save_root_path=save_root_path)



if __name__ == '__main__':
    get_multi_category()
