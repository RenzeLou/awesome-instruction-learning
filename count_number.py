import re
import requests
from bs4 import BeautifulSoup

def query_table_rows(repo_url, readme_path, table_id, head_num=1):
    '''
    Use BeautifulSoup to parse the HTML content of a table in a GitHub README.md file
    return the number of rows in the table
    '''
    # make a GET request to the raw content URL of the README.md file
    raw_url = f'{repo_url}/raw/main/{readme_path}'
    response = requests.get(raw_url)
    # parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')
    # find the table element by its ID
    table = soup.find('table', {'id': table_id})  
    # count the number of rows in the table
    if table:
        rows = table.find_all('tr')
        num_rows = len(rows)
        return num_rows-head_num
    else:
        # throw a warning if the table was not found
        Warning(f'Table not found in {readme_path}')
        num_rows = 0
        return num_rows


with open("README.md", "r", encoding='utf-8') as f_in:
    buf = f_in.readlines()
    count = 0
    sec_count = 0
    count_flag = False
    table_flag = False
    query_flag = False
    sec_title = None
    for line in buf:
        if "Table of Contents" in line:
            count_flag = True  # start counting
            continue
        if count_flag:
            if line.startswith("#"):
                # indicating a new section, then print the previous section's number
                if sec_title is not None:
                    print("==>", sec_title, "\nsec paper number: ", sec_count, "\n")
                    sec_count = 0
                sec_title = line.strip()
                # note the "Corpora" section, if the section name contains "Corpora", then we need to count it seperately
                table_flag = False
                if "Corpora" in line:
                    table_flag = True
            if re.match(r'^\d+\.\s+\*\*', line) and not table_flag:
                count += 1
                sec_count += 1
            elif table_flag and not query_flag:
                # since the "Corpora" section is a table, so we need to use html parser to count the number of rows
                repo_url = 'https://github.com/RenzeLou/awesome-instruction-learning'
                readme_path = 'README.md'
                table_id = "copora-table"
                head_num = 2 # there are 2 rows in the table header, mod it if table head has been changed
                
                row_content = query_table_rows(repo_url, readme_path, table_id, head_num)
                count += row_content
                sec_count += row_content
                query_flag = True
            
print("\nTotal paper number: ", count)