from bs4 import BeautifulSoup
import time
from urllib import request
from bs4 import BeautifulSoup
import re


# fetch text from ncode.syosetu.com

def match_class(target):                                                        
    def do_match(tag):                                                          
        classes = tag.get('class', [])                                          
        return all(c in classes for c in target)                                
    return do_match  

pattern = '<p class="novel_title">(.*?)</p>'
url_main = "https://ncode.syosetu.com/n2267be/"


res = request.urlopen(url_main)
soup = BeautifulSoup(res, "html.parser")
find_title = str(soup.find_all(match_class(["novel_title"]))[0])
print(find_title)
match = re.search(pattern, find_title)

if match:
    print("will read : ", match.group(1)) # prints Ｒｅ：ゼロから始める異世界生活
    title = match.group(1)

else:
    print('No match found')
    raise ValueError("No match found")

num_parts = 100
with open(title + ".txt", "w", encoding="utf_8_sig") as f:
    part = 1
    while True:
    # for part in range(1, num_parts+1):
        # 作品本文ページのURL
        url = url_main + "{:d}/".format(part)
        try : 
            res = request.urlopen(url)
        except: #handle 404 error
            print("part {:d} not found".format(part))
            break
        soup = BeautifulSoup(res, "html.parser")

        # CSSセレクタで本文を指定
        select = honbun = soup.select_one("#novel_honbun")
        if (select == None):
            print("part {:d} not found".format(part))
            break
        else:
            honbun = select.text
            honbun += "\n"  # 次の部分との間は念のため改行しておく
        
        # 保存
        f.write(honbun)
        
        if part % 10 == 0:
            print("part {:d}downloaded".format(part))  # 進捗を表示

        time.sleep(0.1)  # 次の部分取得までは1秒間の時間を空ける

        part += 1
        # if (part == 2):
        #     break
