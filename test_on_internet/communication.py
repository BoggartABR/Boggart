from selenium import webdriver
import time
import sys

abr_alg = sys.argv[1]
SLEEP_TIME = 320

browser = webdriver.Chrome('test_on_internet/chromedriver')
ip = 'localhost'
port = '9998'

url = 'http://' + ip + ':' + port + '/video_serser/' + abr_alg + 'File.html'
browser.get(url)
time.sleep(SLEEP_TIME)
browser.quit()
exit()
