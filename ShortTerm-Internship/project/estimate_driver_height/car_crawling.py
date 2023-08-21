import ssl

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import urllib.request
import subprocess

import os, time, uuid, io
import csv
from PIL import Image

class ImageCrawling:  
    def __init__(self):
        subprocess.Popen(r'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe --remote-debugging-port=9222 --user-data-dir="C:\\chromeCookie"')

        option = Options()
        option.add_experimental_option("debuggerAddress", "127.0.0.1:9222")
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=option)
        self.driver.maximize_window()
        
        
    def page_down(self, pauseTime):
        # 스크롤 높이 가져오기
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        while True:
            # page down
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            
            time.sleep(pauseTime)
            
            # 로딩 후 새로운 스크롤 페이지의 높이 가져오기 
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                try:
                    self.driver.find_element(By.CSS_SELECTOR, ".mye4qd").click()
                except:
                    break
            last_height = new_height

    def load_keyword(self, keywordDir):
        keyword_list = []
        
        # CSV 파일 불러와서 변환
        with open(keywordDir, 'r', newline='', encoding='utf-8') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                if len(row) == 2:  # 두 개의 열로 구성된 경우
                    combined_value = f"{row[0]}+{row[1]}"
                    keyword_list.append(combined_value)
                    
        return keyword_list
        
    def run(self, keywordDir, saveDir, overWidth=640):
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)
                    
        keyword_list = self.load_keyword(keywordDir)
        for keyword in keyword_list:
            manufacturer = keyword.split('+')[0]
            modelName = keyword.split('+')[1]
            print(f"\n---{manufacturer} {modelName}---")

            path = os.path.join(saveDir, f"{manufacturer}+{modelName}")
            if not os.path.exists(path):
                os.makedirs(path)
                
            self.driver.get("https://www.google.co.kr/imghp?hl=ko&tab=wi&authuser=0&ogbl")
            elem = self.driver.find_element("name", "q")
            elem.send_keys(keyword)
            elem.send_keys(Keys.RETURN)

            self.page_down(pauseTime=1)    # page 끝까지 내리기
            
            # 로딩될 때까지 기다리기
            images = WebDriverWait(self.driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".rg_i.Q4LuWd"))
            )
            
            count = 0
            for image in images:
                try:
                    if count >= 10:
                        continue
                    
                    image.click()
                    time.sleep(1)
                    imgUrl = self.driver.find_element(By.CSS_SELECTOR, 'img.r48jcc.pT0Scc.iPVvYb').get_attribute("src")
                    
                    # 이미지 크기 정보 가져오기
                    image_response = urllib.request.urlopen(imgUrl, timeout=3)
                    image = Image.open(io.BytesIO(image_response.read()))
                    image_width, _ = image.size
                    # 너비와 높이가 모두 overWidth 이상인 경우에만 저장
                    if image_width >= overWidth:
                        count += 1
                        print(f"download({count}/10): {manufacturer}+{modelName}, size:({image_width}x__)")
                        urllib.request.urlretrieve(
                            imgUrl,
                            os.path.join(path, f"{manufacturer}+{modelName}_{int(time.time())}{uuid.uuid4().hex[:8]}.jpg")
                        )
                except Exception as e:
                    # print('e : ', e)
                    continue

        self.driver.close()
    
if __name__ == "__main__":
    ssl._create_default_https_context = ssl._create_unverified_context
    crawling = ImageCrawling()
    crawling.run(keywordDir="keywords.csv", saveDir="database", overWidth=640)
