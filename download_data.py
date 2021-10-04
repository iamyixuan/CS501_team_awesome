# download_data.py
"""
Script to download the data from the links
"""
import requests
import os
from bs4 import BeautifulSoup


def listFD(url, ext=''):
    """
    To list all file names under the directory of the website (url)
    :param url: the url for the directory
    :param ext: file extension
    :return: a list of file links
    """
    # Get the html page
    page = requests.get(url).text
    # Parse the result
    soup = BeautifulSoup(page, 'html.parser')
    file_list = []
    for node in soup.find_all('a'):
        if node.get('href').endswith(ext):
            # file_list.append(url + '/' + node.get('href'))
            file_list.append(node.get('href'))
    return file_list
    # return [url + '/' + node.get('href') for node in soup.find_all('a') if node.get('href').endswith(ext)]

def main():
    # Download the data
    # Which year
    years = ('2018', '2019', '2020')
    # Create url
    url = "https://www.ncei.noaa.gov/pub/data/asos-fivemin/6401-" + years[1]
    # file extension
    ext = 'dat'
    # Create local target directory
    cwd = os.getcwd()
    target_path = os.path.join(cwd, 'data', years[1])
    try:
        os.mkdir(target_path)
    except OSError as error:
        print(error)
        pass
    # Create a text of file list
    file_name_list = os.path.join(target_path, "file_list_" + years[1] + ".txt")
    # Write file list and download each of them
    with open(file_name_list, 'w') as f_l:
        for file in listFD(url, ext):
            # Write to file list
            f_l.write(file + '\n')
            # Download each data in the same directory
            resp = requests.get(url + '/' + file)
            with open(os.path.join(target_path, file), 'wb') as f_d:
                f_d.write(resp.content)


if __name__ == '__main__':
    main()
